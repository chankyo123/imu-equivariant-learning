# Paper artefact: EqNIO O(2) and GINIO-E (TorchScript) rows of the ID-SO(3) / mount-tilt sweeps of Supplement
# Table 10 / Figures 6-7, plus the gravity-tilt probe; dumps feed compute_idso3_ate.py, compute_eqnio_ate.py
# and cross_model_summary.py.
"""
Phase A+B sweep for EqNIO O(2) and VN-EqNIO SO(3) torchscript models on NanoBench.

These models train on TLIO data (express_in_local_gravity_aligned=True), so input
must be gravity-aligned world-frame IMU (gravity along +z, removed from accel).

For NanoBench (body-frame npy), we:
  1. Convert body-frame IMU to world-frame via R_body_to_world(t) (using GT quats)
  2. Subtract gravity from accel
  3. Run network -> output (frame_2x2 or 3x3, mu, log_std)  in world frame
  4. Compare against world-frame GT velocity (= R_body_to_world(-1) @ vel_Body)

Scenarios:
  clean             -- no perturbation
  id_so3            -- random SO(3) rotation R applied to world-frame IMU + gravity
                      (rotates entire world); un-rotate prediction
  mount_tilt        -- R_mount applied to body-frame IMU only (GT quat unchanged)
  gravity_tilt      -- same as id_so3 but with small deg points (1-15 deg), focused on
                      EqNIO O(2)'s gravity-alignment assumption

Released TorchScript models (checkpoints/MANIFEST.md): EqNIO_O2 -> tlio_eqnio_o2_ours/model_torchscript.pt
(formerly overnight/o2_paper), VN_EqNIO_SO3 -> nanobench_ginio_e_so3aug_tlio/model_torchscript.pt (formerly
overnight/vn_so3_so3aug); both mappings are recorded in the released parameters.json `out_dir`.
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CKPT_ROOT, DATA_LFA, OUTPUT_DIR, TEST_LIST_FILE, read_test_list, resolve_ckpt_dir  # noqa: E402

DATA_DIR = DATA_LFA   # body-frame NanoBench root

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRAVITY = 9.80665  # m/s^2 along +z in world frame

# ts: released folder under --ckpt_root (holds model_torchscript.pt)
MODELS = {
    'EqNIO_O2': dict(
        ts='tlio_eqnio_o2_ours',
        arch='o2_frame',  # 2D frame predictor + 3D pred + 3D log_std
        window_size=200,
    ),
    'VN_EqNIO_SO3': dict(
        ts='nanobench_ginio_e_so3aug_tlio',
        arch='so3_frame',  # 3D frame + 3D pred + 3D log_std
        window_size=200,
    ),
}


# ----- preprocess (mirror of meas_source_torchscript.py:preprocess_o2/so3) -----
def preprocess_o2(gyro, accel, device=DEVICE):
    """gyro,accel: numpy (N,3). Build cross-product vector features + invariant scalars.
       Returns (vector, scalar, orig_sca) as torch tensors with batch dim."""
    v1 = np.zeros((*gyro.shape[:-1], 3))
    v2 = np.zeros((*gyro.shape[:-1], 3))
    mask = (np.linalg.norm(gyro[..., :-1], axis=-1) == 0).astype(np.int32)
    Rflip = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    gyro_flip = gyro @ Rflip.T
    gyro_flip[..., -1] = 0
    v1[mask == 0] = np.cross(gyro[mask == 0], gyro_flip[mask == 0])
    v2[mask == 0] = np.cross(gyro[mask == 0], v1[mask == 0])
    x = np.zeros_like(gyro); x[..., -2] = 1
    v1[mask == 1] = np.cross(x[mask == 1], gyro[mask == 1])
    v2[mask == 1] = np.cross(gyro[mask == 1], v1[mask == 1])
    gyro_norm = np.linalg.norm(gyro, axis=-1, keepdims=True)
    v1 = v1 * gyro_norm / np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-7, 1e13)
    v2 = v2 * gyro_norm / np.clip(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-7, 1e13)

    a_t = torch.from_numpy(accel).float().to(device)
    v1_t = torch.from_numpy(v1).float().to(device)
    v2_t = torch.from_numpy(v2).float().to(device)
    a = a_t[..., :2].unsqueeze(-2)
    v1_xy = v1_t[..., :2].unsqueeze(-2)
    v2_xy = v2_t[..., :2].unsqueeze(-2)
    scalar = torch.cat(
        (a_t[..., -1:], v1_t[..., -1:], v2_t[..., -1:],
         torch.norm(a, dim=-1), torch.norm(v1_xy, dim=-1), torch.norm(v2_xy, dim=-1),
         (a @ v1_xy.permute(0, 2, 1)).squeeze(-1),
         (v1_xy @ v2_xy.permute(0, 2, 1)).squeeze(-1),
         (a @ v2_xy.permute(0, 2, 1)).squeeze(-1)),
        dim=-1)
    vector = torch.cat((a, v1_xy, v2_xy), dim=-2).permute(0, 2, 1).unsqueeze(0)
    orig_sca = torch.cat((a_t[..., -1:], v1_t[..., -1:], v2_t[..., -1:]), dim=-1).unsqueeze(0)
    return vector, scalar.unsqueeze(0), orig_sca


def preprocess_so3(gyro, accel, device=DEVICE):
    """gyro,accel: numpy (N,3). Returns (vector, scalar) tensors with batch dim."""
    v1 = np.zeros((*gyro.shape[:-1], 3))
    v2 = np.zeros((*gyro.shape[:-1], 3))
    mask = (np.linalg.norm(gyro[..., :-1], axis=-1) == 0).astype(np.int32)
    Rflip = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    gyro_flip = gyro @ Rflip.T
    gyro_flip[..., -1] = 0
    v1[mask == 0] = np.cross(gyro[mask == 0], gyro_flip[mask == 0])
    v2[mask == 0] = np.cross(gyro[mask == 0], v1[mask == 0])
    x = np.zeros_like(gyro); x[..., -2] = 1
    v1[mask == 1] = np.cross(x[mask == 1], gyro[mask == 1])
    v2[mask == 1] = np.cross(gyro[mask == 1], v1[mask == 1])
    gyro_norm = np.linalg.norm(gyro, axis=-1, keepdims=True)
    v1 = v1 * gyro_norm / np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-7, 1e13)
    v2 = v2 * gyro_norm / np.clip(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-7, 1e13)

    a_t = torch.from_numpy(accel).float().to(device)
    v1_t = torch.from_numpy(v1).float().to(device)
    v2_t = torch.from_numpy(v2).float().to(device)
    na = a_t.norm(dim=-1, keepdim=True)
    nv1 = v1_t.norm(dim=-1, keepdim=True)
    nv2 = v2_t.norm(dim=-1, keepdim=True)
    av1 = (a_t * v1_t).sum(-1, keepdim=True)
    v1v2 = (v1_t * v2_t).sum(-1, keepdim=True)
    av2 = (a_t * v2_t).sum(-1, keepdim=True)
    scalar = torch.cat([na, nv1, nv2, av1, v1v2, av2], dim=-1)
    vector = torch.stack([a_t, v1_t, v2_t], dim=-1)
    return vector.unsqueeze(0), scalar.unsqueeze(0)


def build_windows(seq_dir, win, stride):
    """Read NanoBench body-frame npy; extract per-window gyro_body, acc_body, quats,
       gt_vel_body."""
    a = np.load(seq_dir / 'imu0_resampled.npy')
    N = a.shape[0]
    starts = np.arange(0, N - win + 1, stride)
    W = len(starts)
    gyr_b = np.empty((W, win, 3))
    acc_b = np.empty((W, win, 3))
    qts = np.empty((W, win, 4))
    gt_vel_body = np.empty((W, 3))
    R_last_body_to_world = np.empty((W, 3, 3))
    for i, s in enumerate(starts):
        chunk = a[s:s+win]
        gyr_b[i] = chunk[:, 1:4]
        acc_b[i] = chunk[:, 4:7]
        qts[i] = chunk[:, 7:11]
        gt_vel_body[i] = chunk[-1, 14:17]
        # last sample's body->world rotation (used to map GT vel_Body to world)
        R_last_body_to_world[i] = Rotation.from_quat(chunk[-1, 7:11]).as_matrix()
    return gyr_b, acc_b, qts, gt_vel_body, R_last_body_to_world


def body_to_world_imu(gyr_b, acc_b, qts):
    """Convert per-sample body-frame IMU to world-frame, subtract gravity from accel.
       Returns gyr_w (W,T,3), acc_w_no_g (W,T,3)."""
    W, T = gyr_b.shape[:2]
    gyr_w = np.empty_like(gyr_b)
    acc_w = np.empty_like(acc_b)
    for w in range(W):
        Rmats = Rotation.from_quat(qts[w]).as_matrix()  # (T,3,3) body->world
        gyr_w[w] = np.einsum('tij,tj->ti', Rmats, gyr_b[w])
        acc_w[w] = np.einsum('tij,tj->ti', Rmats, acc_b[w])
    acc_w_no_g = acc_w.copy()
    acc_w_no_g[..., 2] -= GRAVITY
    return gyr_w, acc_w_no_g


def apply_perturb_world(scenario, gyr_w, acc_w, R_last_body_to_world, gt_vel_body, R):
    """Apply test-time perturbation to world-frame IMU.
       For id_so3, gravity_tilt: rotate world frame globally by R.
       For mount_tilt: rotated already in body frame (use mount_tilt_world wrapper instead)."""
    if scenario == 'clean':
        return gyr_w, acc_w, gt_vel_body, R_last_body_to_world, False
    if scenario in ('id_so3', 'gravity_tilt'):
        # Rotate the entire world frame by R. World-frame IMU becomes R @ world_IMU.
        # Gravity vector becomes R @ [0,0,g] (no longer along z if R not yaw).
        # GT vel_body rotates only if we rotate the body too -- here we keep body fixed,
        # so just rotate prediction back after.
        gyr_r = np.einsum('ij,wtj->wti', R, gyr_w)
        # acc_w is gravity-removed. Under world rotation by R: acc_new = R @ acc_old
        # But the gravity removal was acc_w = R_body_to_world @ acc_body - [0,0,g].
        # Under world rotation R, the model expects input with gravity along its own z.
        # We feed R @ acc_w, which means the model sees IMU as if gravity is along R @ z.
        # The model's gravity-alignment assumption is violated for non-yaw R -- that IS
        # the test.
        acc_r = np.einsum('ij,wtj->wti', R, acc_w)
        # GT body-frame vel unchanged; world-frame GT changes by R if we use world GT.
        # For comparison: un-rotate prediction by R^T -> compare against original world GT.
        return gyr_r, acc_r, gt_vel_body, R_last_body_to_world, True
    raise ValueError(scenario)


def apply_mount_tilt_body(gyr_b, acc_b, R_mount):
    """Rotate BODY-frame IMU only (gyr,acc) -- leave quat untouched.
       Returns rotated gyr_b, acc_b."""
    gyr_new = np.einsum('ij,wtj->wti', R_mount, gyr_b)
    acc_new = np.einsum('ij,wtj->wti', R_mount, acc_b)
    return gyr_new, acc_new


def run_one(mname, scenario, R, seed_tag, batch=64, *, ckpt_root=CKPT_ROOT, data_dir=DATA_DIR,
            test_list=None, out_dir=OUTPUT_DIR, device=DEVICE):
    info = MODELS[mname]
    ts_path = resolve_ckpt_dir(info['ts'], ckpt_root) / 'model_torchscript.pt'
    net = torch.jit.load(str(ts_path), map_location=device).to(device).eval()
    win = info['window_size']
    stride = 20  # match LN-ResNet sweep
    arch = info['arch']

    data_dir = Path(data_dir)
    test_list = read_test_list() if test_list is None else test_list
    out_dir = Path(out_dir)
    (out_dir / 'dumps').mkdir(parents=True, exist_ok=True)

    per_seq = {}
    t0 = time.time()
    for seq in test_list:
        seq_dir = data_dir / seq
        if not seq_dir.exists(): continue
        gyr_b, acc_b, qts, gt_body, R_lb2w = build_windows(seq_dir, win, stride)

        # Apply mount-tilt BEFORE world conversion
        unrot = False
        if scenario == 'mount_tilt':
            gyr_b, acc_b = apply_mount_tilt_body(gyr_b, acc_b, R)
        gyr_w, acc_w = body_to_world_imu(gyr_b, acc_b, qts)

        # For id_so3 / gravity_tilt, rotate world-frame data
        if scenario in ('id_so3', 'gravity_tilt'):
            gyr_w, acc_w, gt_body, R_lb2w, unrot = apply_perturb_world(
                scenario, gyr_w, acc_w, R_lb2w, gt_body, R)

        # GT world-frame velocity (target for EqNIO)
        gt_world = np.einsum('wij,wj->wi', R_lb2w, gt_body)
        # If we rotated the world by R, GT world also rotates: gt_world_new = R @ gt_world_old
        # But we keep gt_world_old as the original; we'll un-rotate prediction to match.

        W = gyr_w.shape[0]
        mu_all = np.empty((W, 3))
        sig_all = np.empty((W, 3, 3))

        with torch.no_grad():
            for s in range(0, W, batch):
                gW = gyr_w[s:s+batch]  # (B, T, 3)
                aW = acc_w[s:s+batch]
                if arch == 'o2_frame':
                    # preprocess takes per-window (T,3), we batch-loop
                    mu_b = []; cov_b = []
                    for b in range(gW.shape[0]):
                        vec, sca, orig = preprocess_o2(gW[b], aW[b], device)
                        frame, pred, pred_cov = net(vec, sca, orig)
                        # frame: (1, 2, 2). pred: (1, 3). pred_cov: (1, 3) log-std (xy) + scalar z
                        # Per meas_source: tlio_frame_fullCov branch uses _fullCov head which
                        # outputs (mean, cov_logstd_3). Sigma assembly:
                        #   Sigma_xy_world = frame^T @ diag(exp(2*log_std_xy)) @ frame
                        #   Sigma_zz = exp(2*log_std_z)
                        pred = pred[0].cpu().numpy()
                        pc = pred_cov[0].cpu().numpy()
                        frame_np = frame[0].cpu().numpy()  # (2,2)
                        Sigma_xy = frame_np.T @ np.diag(np.exp(2*pc[:2])) @ frame_np
                        Sigma = np.zeros((3, 3))
                        Sigma[:2, :2] = Sigma_xy
                        Sigma[2, 2] = float(np.exp(2 * pc[2]))
                        # pred is world-frame xy + z
                        mu_b.append(pred); cov_b.append(Sigma)
                elif arch == 'so3_frame':
                    mu_b = []; cov_b = []
                    for b in range(gW.shape[0]):
                        vec, sca = preprocess_so3(gW[b], aW[b], device)
                        frame, pred, pred_cov = net(vec, sca)
                        # frame: (1,3,3), pred: (1,3) canonical, pred_cov: (1,3) log-std diag in canonical
                        frame_np = frame[0].cpu().numpy()
                        pred_np = pred[0].cpu().numpy()  # canonical
                        pc = pred_cov[0].cpu().numpy()
                        # Disp_world = frame^T @ disp_canonical
                        mu_world = frame_np.T @ pred_np
                        # Sigma_world = frame^T @ diag(exp(2*log_std)) @ frame
                        Sigma = frame_np.T @ np.diag(np.exp(2*pc)) @ frame_np
                        mu_b.append(mu_world); cov_b.append(Sigma)
                else:
                    raise ValueError(arch)
                for j, (m, c) in enumerate(zip(mu_b, cov_b)):
                    mu_all[s+j] = m
                    sig_all[s+j] = c

        if unrot:
            # Un-rotate world prediction back to original world frame for comparison
            mu_all = (R.T @ mu_all.T).T
            sig_all = np.einsum('ij,wjk,lk->wil', R.T, sig_all, R.T)

        err = mu_all - gt_world
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
                            mu=mu_all, Sigma=sig_all, gt=gt_world, err=err)

    agg = {k: float(np.mean([per_seq[s][k] for s in per_seq])) for k in
           ['rmse_vel','mean_mahal','mean_logdet','mean_nll','coverage_95','coverage_90']}
    out = dict(model=mname, scenario=scenario, seed_tag=seed_tag, agg=agg, per_seq=per_seq,
               R=R.tolist(), elapsed=time.time()-t0)
    with open(out_dir / f'aggregate__{mname}__{scenario}__{seed_tag}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  [{mname:14s} | {scenario:14s} | {seed_tag:18s}] '
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
    parser.add_argument('--phase', default='all', choices=['A','B','gravity','all'])
    parser.add_argument('--model', default=None, choices=list(MODELS))
    parser.add_argument('--ckpt_root', default=str(CKPT_ROOT), help='released checkpoint folders (model_torchscript.pt)')
    parser.add_argument('--data_dir', default=str(DATA_DIR), help='body-frame NanoBench root')
    parser.add_argument('--test_list', default=str(TEST_LIST_FILE))
    parser.add_argument('--out_dir', default=str(OUTPUT_DIR))
    parser.add_argument('--device', default=DEVICE)
    args = parser.parse_args()
    targets = [args.model] if args.model else list(MODELS.keys())
    kw = dict(ckpt_root=args.ckpt_root, data_dir=args.data_dir, test_list=read_test_list(args.test_list),
              out_dir=args.out_dir, device=args.device)

    I3 = np.eye(3)
    if args.phase in ('A','all'):
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
                    run_one(m, 'mount_tilt', R, f'deg{deg}_seed{ax}', **kw)

    if args.phase in ('gravity','all'):
        # Gravity-tilt test: small out-of-plane rotation (1-15 deg) -- focus on EqNIO O(2)
        # since the gravity-alignment assumption is what's broken.
        for m in targets:
            for deg in [1, 3, 5, 10, 15]:
                for ax in range(3):
                    R = random_so3(seed=3000+ax, deg=deg)
                    run_one(m, 'gravity_tilt', R, f'deg{deg}_seed{ax}', **kw)


if __name__ == '__main__':
    main()
