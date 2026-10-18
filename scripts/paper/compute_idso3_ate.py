# Paper artefact: trajectory-level Umeyama ATE under ID-SO(3) per rotation angle (the ID-SO(3) columns of
# Supplement Table 10 / Figure 6) for the 10 swept models, computed from the per-window dumps of run_phase_ab*.py.
"""Compute trajectory-level ATE_um under ID-SO(3) perturbation for each model.

For each model in {4 LN-ResNet, 4 AirIO, 2 EqNIO}:
  - For each scenario in {clean, id_so3 at deg in {30,60,90,120,180}}:
    - For each seq:
      - Load per-window prediction dump (mu, gt, err)
      - For LN-ResNet / AirIO: mu is BODY-frame velocity. Convert to world via R_last(t).
      - For EqNIO: mu is already WORLD-frame velocity.
      - Integrate predicted velocity -> predicted trajectory.
      - Umeyama-align to GT trajectory; compute RMSE.
    - Aggregate mean ATE over 17 seqs.
    - For ID-SO(3): aggregate over 3 random axes per deg.

Output: <out_dir>/trajectory_ate_idso3.json -- per-model per-deg ATE_um.
"""
import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_LFA, DATA_NOLFA, OUTPUT_DIR, TEST_LIST_FILE, read_test_list  # noqa: E402

WIN, STRIDE = 200, 20
DT_STRIDE = STRIDE / 100.0

# (key, body_or_world, data root: 'lfa' = nanobench_bodyframe_2nd_version, 'nolfa' = the nolfa-named copy)
MODELS = {
    # LN-ResNet -- body-frame mu
    'ln_resnet_lfa_MSE':       dict(frame='body', data='lfa'),
    'ln_resnet_lfa_NLL':       dict(frame='body', data='lfa'),
    'ln_resnet_chol_MSE':      dict(frame='body', data='nolfa'),
    'ln_resnet_chol_NLL':      dict(frame='body', data='nolfa'),
    # AirIO -- body-frame mu (last-step prediction)
    'airio_native_noatt':      dict(frame='body', data='lfa'),
    'airio_native_rot':        dict(frame='body', data='lfa'),
    'vn_airio_noatt':          dict(frame='body', data='lfa'),
    'vn_airio_rot':            dict(frame='body', data='lfa'),
    # EqNIO -- already world-frame mu
    'EqNIO_O2':                dict(frame='world', data='lfa'),
    'VN_EqNIO_SO3':            dict(frame='world', data='lfa'),
}


def umeyama_rmse(P, Q):
    cP = P.mean(0); cQ = Q.mean(0)
    Pc = P - cP; Qc = Q - cQ
    H = Pc.T @ Qc
    U, S, Vt = svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T
    s = (Pc * (Qc @ R)).sum() / (Pc ** 2).sum()
    t = cQ - s * R @ cP
    P_aligned = s * (R @ P.T).T + t
    err = P_aligned - Q
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


def trajectory_ate(mu, frame, seq_npy):
    """mu: (W, 3) per-window predicted velocity.
       frame: 'body' or 'world'.
       seq_npy: (N, 17) raw NanoBench data.
       Returns ATE_um (m).
    """
    W_actual = mu.shape[0]
    N = seq_npy.shape[0]
    # Auto-detect stride: (N - WIN) // (W - 1) gives stride for W windows
    stride = max(1, (N - WIN) // max(1, W_actual - 1))
    dt_stride = stride / 100.0  # NanoBench is 100 Hz
    starts = np.arange(0, N - WIN + 1, stride)[:W_actual]
    if len(starts) < W_actual:
        return None
    gt_pos = seq_npy[starts + WIN - 1, 11:14]  # (W, 3)
    if frame == 'body':
        R_last = Rotation.from_quat(seq_npy[starts + WIN - 1, 7:11]).as_matrix()
        mu_world = np.einsum('wij,wj->wi', R_last, mu)
    else:
        mu_world = mu
    pred_pos = np.zeros_like(gt_pos)
    pred_pos[0] = gt_pos[0]
    for i in range(1, W_actual):
        pred_pos[i] = pred_pos[i - 1] + mu_world[i - 1] * dt_stride
    return umeyama_rmse(pred_pos, gt_pos)


def collect_one(mname, scenario, seed_tag, dumps, test, data_roots):
    info = MODELS[mname]
    ates = []
    for seq in test:
        dp = dumps / f'{mname}__{scenario}__{seed_tag}__{seq}.npz'
        if not dp.exists():
            continue
        mu = np.load(dp)['mu']
        seq_npy = np.load(data_roots[info['data']] / seq / 'imu0_resampled.npy')
        ate = trajectory_ate(mu, info['frame'], seq_npy)
        if ate is not None:
            ates.append(ate)
    if not ates:
        return None
    return float(np.mean(ates))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default=str(OUTPUT_DIR), help='folder with dumps/ from run_phase_ab*.py; output goes here')
    ap.add_argument('--data_lfa', default=str(DATA_LFA))
    ap.add_argument('--data_nolfa', default=str(DATA_NOLFA))
    ap.add_argument('--test_list', default=str(TEST_LIST_FILE))
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    DUMPS = OUT / 'dumps'
    TEST = read_test_list(args.test_list)
    data_roots = {'lfa': Path(args.data_lfa), 'nolfa': Path(args.data_nolfa)}

    results = {}
    ID_DEGS = [30, 60, 90, 120, 180]
    for m in MODELS:
        results[m] = {}
        # Clean
        results[m]['clean'] = collect_one(m, 'clean', 'clean', DUMPS, TEST, data_roots)
        # ID-SO(3)
        for deg in ID_DEGS:
            seeds = [collect_one(m, 'id_so3', f'deg{deg}_seed{s}', DUMPS, TEST, data_roots) for s in range(3)]
            seeds = [v for v in seeds if v is not None]
            results[m][f'id_so3_{deg}'] = float(np.mean(seeds)) if seeds else None
        print(f'{m:25s} | clean={results[m]["clean"]:.4f}'
              f' | so3_30={results[m]["id_so3_30"]:.4f}'
              f' | so3_90={results[m]["id_so3_90"]:.4f}'
              f' | so3_180={results[m]["id_so3_180"]:.4f}'
              f' | delta_180={(results[m]["id_so3_180"]-results[m]["clean"])/results[m]["clean"]*100:+.1f}%')
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'trajectory_ate_idso3.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved', OUT / 'trajectory_ate_idso3.json')


if __name__ == '__main__':
    main()
