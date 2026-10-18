# Paper artefact: Air-IO-family rows (AirIO no-att / with-att, GINIO-A no-att / with-att) of the ID-SO(3) and
# mount-tilt sweeps of Supplement Table 10 / Figures 6-7; dumps feed compute_idso3_ate.py and cross_model_summary.py.
"""
Phase A+B sweep for AirIO native (non-eq) and VN-AirIO (eq), with-att and no-att.

4 AirIO variants:
  airio_native_noatt  (codenetmotion,           non-eq, no-attitude)
  airio_native_rot    (codewithrot,             non-eq, with-attitude)
  vn_airio_noatt      (vn_codenetmotion_small,  eq SO(3), no-attitude)   -- released GINIO-A model
  vn_airio_rot        (vn_codewithrot_small,    eq SO(3), with-attitude) -- released GINIO-A model

Loads body-frame IMU directly from nanobench_bodyframe_2nd_version (avoids LFA
codepath -- AirIO does NOT use LFA). Runs same ID-SO(3) and
calib-mismatch sweep as run_phase_ab.py for matched comparison.

Output to <out_dir>/aggregate__<model>__<scenario>__<seed>.json.

Release note: the 2026-05-25 sweep loaded the original 0.273M VN-AirIO checkpoints
(experiments/nanobench_{noatt,rot}_vn) with strict=False; those checkpoints predate VNLayerNorm,
do not load against the current code and are not released (MANIFEST note 5). The two VN rows
therefore default to the released `_small` GINIO-A models of Table 2.
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pypose as pp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (AIRIO_ROOT, CKPT_ROOT, DATA_LFA, OUTPUT_DIR, SRC_DIR, TEST_LIST_FILE,  # noqa: E402
                    insert_paths, read_test_list, resolve_ckpt_dir)
insert_paths(AIRIO_ROOT, SRC_DIR)   # same order as the original: airio/, then src/ (first on sys.path)

from model import net_dict  # noqa: E402
from pyhocon import ConfigFactory  # noqa: E402

DATA_NOLFA = DATA_LFA   # body-frame NanoBench root (the original pointed this name at nanobench_bodyframe_2nd_version)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# conf: relative to airio/; ckpt: released folder under --ckpt_root (holds best_model.ckpt)
MODELS = {
    'airio_native_noatt': dict(
        conf='configs/Nanobench/motion_body_noatt.conf',
        ckpt='airio_noatt_nanobench',
        eq=False, withrot=False,
    ),
    'airio_native_rot': dict(
        conf='configs/Nanobench/motion_body_rot.conf',
        ckpt='airio_withatt_nanobench',
        eq=False, withrot=True,
    ),
    'vn_airio_noatt': dict(
        conf='configs/Nanobench/motion_body_noatt_vn_small.conf',
        ckpt='ginio_a_noatt_nanobench',
        eq=True, withrot=False,
    ),
    'vn_airio_rot': dict(
        conf='configs/Nanobench/motion_body_rot_vn_small.conf',
        ckpt='ginio_a_withatt_nanobench',
        eq=True, withrot=True,
    ),
}


def load_model(mname, ckpt_root=CKPT_ROOT, airio_root=AIRIO_ROOT, device=DEVICE):
    info = MODELS[mname]
    conf = ConfigFactory.parse_file(str(Path(airio_root) / info['conf']))
    conf.train.device = device
    net = net_dict[conf.train.network](conf.train).to(device).double()
    ckpt = torch.load(resolve_ckpt_dir(info['ckpt'], ckpt_root) / 'best_model.ckpt', map_location=device, weights_only=True)
    miss, unexp = net.load_state_dict(ckpt['model_state_dict'], strict=False)
    if miss or unexp:
        print(f'  [{mname}] ckpt loose: miss={len(miss)} unexp={len(unexp)}')
    net.eval()
    return net, info, 200  # window_size=200 for all Nanobench AirIO models


def build_windows_airio(seq_dir, win, stride):
    """Read imu0_resampled.npy (body-frame), build windows for AirIO input.
       Returns gyr (W,T,3), acc (W,T,3), rot_logmap (W,T,3), gt_vel_body (W,3),
               quats (W,T,4) for SO(3) rotation if needed.
    """
    a = np.load(seq_dir / 'imu0_resampled.npy')
    N = a.shape[0]
    starts = np.arange(0, N - win + 1, stride)
    W = len(starts)
    gyr = np.empty((W, win, 3))
    acc = np.empty((W, win, 3))
    qts = np.empty((W, win, 4))
    rot_logmap = np.empty((W, win, 3))
    gt_vel = np.empty((W, 3))
    for i, s in enumerate(starts):
        chunk = a[s:s+win]
        gyr[i] = chunk[:, 1:4]
        acc[i] = chunk[:, 4:7]
        qts[i] = chunk[:, 7:11]
        rot_logmap[i] = Rotation.from_quat(chunk[:, 7:11]).as_rotvec()
        gt_vel[i] = chunk[-1, 14:17]
    return gyr, acc, qts, rot_logmap, gt_vel


def apply_perturb_airio(scenario, gyr, acc, qts, rot_logmap, gt_vel, R):
    if scenario == 'clean':
        return gyr, acc, qts, rot_logmap, gt_vel, False
    if scenario == 'id_so3':
        gyr_r = np.einsum('ij,wtj->wti', R, gyr)
        acc_r = np.einsum('ij,wtj->wti', R, acc)
        gt_r  = (R @ gt_vel.T).T
        # rotate body-frame quats: q_new = q_R . q_old (R rotates body frame globally)
        q_R = Rotation.from_matrix(R).as_quat()
        qts_r = np.empty_like(qts)
        for w in range(qts.shape[0]):
            qts_r[w] = (Rotation.from_quat(q_R[None,:]) * Rotation.from_quat(qts[w])).as_quat()
        rot_r = Rotation.from_quat(qts_r.reshape(-1,4)).as_rotvec().reshape(qts.shape[0],qts.shape[1],3)
        return gyr_r, acc_r, qts_r, rot_r, gt_r, True
    if scenario == 'calib_mismatch':
        gyr_r = np.einsum('ij,wtj->wti', R, gyr)
        acc_r = np.einsum('ij,wtj->wti', R, acc)
        # rot input (attitude) is GT, NOT mount-tilted -- that's what makes calib mismatch interesting
        return gyr_r, acc_r, qts, rot_logmap, gt_vel, False
    raise ValueError(scenario)


def run_one(mname, scenario, R, seed_tag, batch=64, *, ckpt_root=CKPT_ROOT, airio_root=AIRIO_ROOT,
            data_root=DATA_NOLFA, test_list=None, out_dir=OUTPUT_DIR, device=DEVICE):
    info = MODELS[mname]
    net, _, win = load_model(mname, ckpt_root, airio_root, device)
    withrot = info['withrot']
    stride = max(1, int(200 / 10))  # 200Hz, sample_freq=10 -> stride 20

    data_root = Path(data_root)
    test_list = read_test_list() if test_list is None else test_list
    out_dir = Path(out_dir)
    (out_dir / 'dumps').mkdir(parents=True, exist_ok=True)

    per_seq = {}
    t0 = time.time()
    for seq in test_list:
        seq_dir = data_root / seq
        if not seq_dir.exists(): continue
        gyr, acc, qts, rot, gt_vel = build_windows_airio(seq_dir, win, stride)
        gyr_p, acc_p, qts_p, rot_p, gt_target, unrot = apply_perturb_airio(scenario, gyr, acc, qts, rot, gt_vel, R)

        W = gyr_p.shape[0]
        mu_all = np.empty((W, 3))
        sig_all = np.empty((W, 3, 3))

        with torch.no_grad():
            for s in range(0, W, batch):
                b_gyr = torch.from_numpy(gyr_p[s:s+batch]).to(device).double()
                b_acc = torch.from_numpy(acc_p[s:s+batch]).to(device).double()
                data = {'gyro': b_gyr, 'acc': b_acc}
                if withrot:
                    b_rot = torch.from_numpy(rot_p[s:s+batch]).to(device).double()
                    out = net(data, b_rot)
                else:
                    out = net(data, None)
                # Take LAST timestep prediction (CNN output sequence)
                v = out['net_vel'][:, -1, :]  # (B,3)
                cov_raw = out['cov']
                if cov_raw is not None:
                    c = cov_raw[:, -1, :]  # (B,3) -- already exp(raw-5)
                    # diag Sigma
                    Sigma_b = torch.diag_embed(c)
                else:
                    Sigma_b = torch.eye(3, device=device).double().expand(v.shape[0], 3, 3) * 1.0
                mu_all[s:s+batch] = v.cpu().numpy()
                sig_all[s:s+batch] = Sigma_b.cpu().numpy()

        if unrot:
            mu_all = (R.T @ mu_all.T).T
            sig_all = np.einsum('ij,wjk,lk->wil', R.T, sig_all, R.T)
            gt_eval = gt_vel
        else:
            gt_eval = gt_target

        err = mu_all - gt_eval
        rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
        inv_S = np.linalg.inv(sig_all + 1e-12 * np.eye(3))
        M = np.einsum('wi,wij,wj->w', err, inv_S, err)
        sign, logabsdet = np.linalg.slogdet(sig_all + 1e-12 * np.eye(3))
        nll = 0.5 * (M + logabsdet + 3*np.log(2*np.pi))
        per_seq[seq] = dict(
            rmse_vel=rmse,
            mean_mahal=float(M.mean()),
            mean_logdet=float(logabsdet.mean()),
            mean_nll=float(nll.mean()),
            coverage_95=float(np.mean(M <= 7.815)),
            coverage_90=float(np.mean(M <= 6.251)),
        )
        np.savez_compressed(out_dir / 'dumps' / f'{mname}__{scenario}__{seed_tag}__{seq}.npz',
                            mu=mu_all, Sigma=sig_all, gt=gt_eval, err=err)

    agg = {k: float(np.mean([per_seq[s][k] for s in per_seq])) for k in
           ['rmse_vel','mean_mahal','mean_logdet','mean_nll','coverage_95','coverage_90']}
    out = dict(model=mname, scenario=scenario, seed_tag=seed_tag, agg=agg, per_seq=per_seq,
               R=R.tolist(), elapsed=time.time()-t0)
    with open(out_dir / f'aggregate__{mname}__{scenario}__{seed_tag}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  [{mname:20s} | {scenario:14s} | {seed_tag:18s}] '
          f'rmse_vel={agg["rmse_vel"]:.4f}  cov95={agg["coverage_95"]:.3f}  ({time.time()-t0:.1f}s)')
    return out


def random_so3(seed, deg=None):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.deg2rad(deg) if deg is not None else rng.uniform(0, np.pi)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', default='all', choices=['A','B','all'])
    parser.add_argument('--model', default=None, choices=list(MODELS))
    parser.add_argument('--ckpt_root', default=str(CKPT_ROOT), help='released checkpoint folders')
    parser.add_argument('--airio_root', default=str(AIRIO_ROOT), help='airio/ (configs are resolved relative to it)')
    parser.add_argument('--data_root', default=str(DATA_NOLFA), help='body-frame NanoBench root')
    parser.add_argument('--test_list', default=str(TEST_LIST_FILE))
    parser.add_argument('--out_dir', default=str(OUTPUT_DIR))
    parser.add_argument('--device', default=DEVICE)
    args = parser.parse_args()
    targets = [args.model] if args.model else list(MODELS.keys())
    kw = dict(ckpt_root=args.ckpt_root, airio_root=args.airio_root, data_root=args.data_root,
              test_list=read_test_list(args.test_list), out_dir=args.out_dir, device=args.device)

    if args.phase in ('A','all'):
        I3 = np.eye(3)
        for m in targets:
            run_one(m, 'clean', I3, 'clean', **kw)

    if args.phase in ('B','all'):
        for m in targets:
            for deg in [30, 60, 90, 120, 180]:
                for ax in range(3):
                    R = random_so3(seed=1000+ax, deg=deg)
                    run_one(m, 'id_so3', R, f'deg{deg}_seed{ax}', **kw)
        for m in targets:
            for deg in [3, 5, 10, 15, 20]:
                for ax in range(3):
                    R = random_so3(seed=2000+ax, deg=deg)
                    run_one(m, 'calib_mismatch', R, f'deg{deg}_seed{ax}', **kw)


if __name__ == '__main__':
    main()
