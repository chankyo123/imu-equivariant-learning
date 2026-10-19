# Paper artefact: Air-IO-family NanoBench rows of Table 2 (AirIO no-att 5.579, AirIO with-att 1.205) and the
# original 0.273M GINIO-A models of Supplement Table 5; writes round1_nanobench_metrics.json, an input of
# build_v2_paper_quality.py.
"""
AWS Round 1 Nanobench evaluation.

Takes the 4 freshly trained Air-IO models (Nanobench: VN no-att, VN with-att, AirIO no-att,
AirIO with-att) and computes ATE/RMSE/RPE on the test set, then compares to existing
Trial 55 and Trial 96 results from unified_eval.

Workflow per model:
  1. Run `python inference_motion.py --config <conf> --device cuda:X` -> produces
     experiments/<exp_dir>/<conf_name>/net_output.pickle
  2. Load that pickle, get body-frame predicted velocities + timestamps per sequence.
  3. Use GT orientation to rotate body->world, integrate to world trajectory.
  4. Compute ATE_aligned (Umeyama), RMSE_no_align, RPE@2s.

The trajectory math here is reused from unified_eval.py.

Release note: the two "VN" entries are the original 0.273M VN-AirIO models
(experiments/nanobench_{noatt,rot}_vn), which are not released and do not load against the
current airio/model/vn_codenet.py (MANIFEST note 5); they are skipped unless you trained them.
The released GINIO-A rows of Table 2 are evaluated by round1_eval_small.py. For the two AirIO
baselines the released checkpoints are symlinked into airio/experiments/... automatically.
"""
import argparse
import json
import os
import pickle
import sys
import subprocess
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (AIRIO_ROOT, CKPT_ROOT, DATA_WORLD, OUTPUT_DIR, TEST_LIST_FILE,  # noqa: E402
                    ensure_airio_experiment, read_test_list)

NANO_ROOT = str(DATA_WORLD)
AIRIO_ROOT = str(AIRIO_ROOT)
OUT_DIR = str(OUTPUT_DIR)


# -- Model registry: (label, config_path, exp_dir, conf_name) ------------------
MODELS = [
    {
        "label": "VN no-att",
        "config": "configs/Nanobench/motion_body_noatt_vn.conf",
        "exp_dir": "experiments/nanobench_noatt_vn/motion_body_noatt_vn",
        "device": "cuda:0",
    },
    {
        "label": "VN with-att",
        "config": "configs/Nanobench/motion_body_rot_vn.conf",
        "exp_dir": "experiments/nanobench_rot_vn/motion_body_rot_vn",
        "device": "cuda:0",
    },
    {
        "label": "AirIO no-att (baseline)",
        "config": "configs/Nanobench/motion_body_noatt.conf",
        "exp_dir": "experiments/nanobench_noatt/motion_body_noatt",
        "device": "cuda:0",
    },
    {
        "label": "AirIO with-att (baseline)",
        "config": "configs/Nanobench/motion_body_rot.conf",
        "exp_dir": "experiments/nanobench/motion_body_rot",
        "device": "cuda:0",
    },
]


def load_gt(seq, nano_root=None):
    """Load nanobench GT trajectory."""
    nano_root = NANO_ROOT if nano_root is None else nano_root
    arr = np.load(f"{nano_root}/{seq}/imu0_resampled.npy")
    ts = arr[:, 0] / 1e6
    quat = arr[:, 7:11]
    pos = arr[:, 11:14]
    return ts, quat, pos


def umeyama_align(src, dst):
    """Best-fit R + t (no scale) mapping src to dst. Returns aligned src."""
    src = np.asarray(src); dst = np.asarray(dst)
    sm = src.mean(0); dm = dst.mean(0)
    s = src - sm; d = dst - dm
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dm - R @ sm
    return (R @ src.T).T + t


def rpe_at_dt(pp_pos, gg_pos, ts, dt_s):
    """Relative pose error at delta-t."""
    N = len(ts)
    errs = []
    for i in range(N):
        target = ts[i] + dt_s
        j = np.searchsorted(ts, target)
        if j >= N: break
        if j == i: continue
        d_pp = pp_pos[j] - pp_pos[i]
        d_gg = gg_pos[j] - gg_pos[i]
        errs.append(np.linalg.norm(d_pp - d_gg))
    if not errs: return float("nan")
    return float(np.sqrt(np.mean(np.array(errs) ** 2)))


def metrics_from_traj(pp_pos, gg_pos, ts):
    """ATE_aligned, RMSE_no_align, RPE@2s, drift_endpoint."""
    pp_aligned = umeyama_align(pp_pos, gg_pos)
    ate = float(np.sqrt(np.mean(np.sum((pp_aligned - gg_pos) ** 2, axis=1))))
    rmse = float(np.sqrt(np.mean(np.sum((pp_pos - gg_pos) ** 2, axis=1))))
    rpe2s = rpe_at_dt(pp_pos, gg_pos, ts, 2.0)
    traj_len = float(np.sum(np.linalg.norm(np.diff(gg_pos, axis=0), axis=1)))
    drift = float(np.linalg.norm(pp_pos[-1] - gg_pos[-1])) / max(traj_len, 1e-6)
    return {"ATE_aligned": ate, "RMSE_no_align": rmse, "RPE@2s": rpe2s, "drift_endpoint": drift}


def compute_world_traj(net_vel_body, ts, quat_xyzw_at_ts):
    """Integrate body-frame velocity to world-frame trajectory using GT orientation."""
    # net_vel_body: (T, 3); ts: (T,); quat_xyzw_at_ts: (T, 4)
    R = Rotation.from_quat(quat_xyzw_at_ts).as_matrix()  # (T, 3, 3)
    vel_W = np.einsum("nij,nj->ni", R, net_vel_body)
    # integrate via trapezoidal rule
    dt = np.diff(ts)
    vel_mid = 0.5 * (vel_W[:-1] + vel_W[1:])
    dp = vel_mid * dt[:, None]
    pos_W = np.zeros_like(vel_W)
    pos_W[1:] = np.cumsum(dp, axis=0)
    return pos_W


def evaluate_model(model_info, test_seqs, nano_root=None, airio_root=None):
    """Returns {seq: metrics dict}."""
    nano_root = NANO_ROOT if nano_root is None else nano_root
    airio_root = AIRIO_ROOT if airio_root is None else airio_root
    pickle_path = os.path.join(airio_root, model_info["exp_dir"], "net_output.pickle")
    if not os.path.exists(pickle_path):
        print(f"  [{model_info['label']}] no pickle at {pickle_path} -- skipping")
        return None

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    seq_metrics = {}
    for seq in test_seqs:
        if seq not in data:
            continue
        d = data[seq]
        ts_pred = d["ts"].cpu().numpy() if hasattr(d["ts"], "cpu") else np.asarray(d["ts"])
        ts_pred = ts_pred.squeeze()   # (T, 1) -> (T,)
        net_vel = d["net_vel"].cpu().numpy() if hasattr(d["net_vel"], "cpu") else np.asarray(d["net_vel"])
        if net_vel.ndim == 3:
            net_vel = net_vel[0]      # in case of leading singleton batch dim

        # Load GT for this sequence
        ts_gt, quat_gt, pos_gt = load_gt(seq, nano_root)

        # Interpolate GT to ts_pred (nearest neighbor)
        idxs = np.searchsorted(ts_gt, ts_pred).clip(0, len(ts_gt) - 1)
        quat_at_ts = quat_gt[idxs]
        pos_at_ts = pos_gt[idxs]

        # Body->world integration
        pred_pos = compute_world_traj(net_vel, ts_pred, quat_at_ts)
        # Anchor to GT start (predicted pos starts at 0; shift to match)
        pred_pos += (pos_at_ts[0] - pred_pos[0])

        m = metrics_from_traj(pred_pos, pos_at_ts, ts_pred)
        seq_metrics[seq] = m

    return seq_metrics


def run_inference(model_info, airio_root=None, ckpt_root=CKPT_ROOT, link_ckpts=True):
    """Run inference_motion.py for a model. Skips if pickle already exists."""
    airio_root = AIRIO_ROOT if airio_root is None else airio_root
    pickle_path = os.path.join(airio_root, model_info["exp_dir"], "net_output.pickle")
    if os.path.exists(pickle_path):
        print(f"  [{model_info['label']}] pickle exists, skipping inference")
        return
    if link_ckpts and ensure_airio_experiment(model_info["exp_dir"], ckpt_root, airio_root) is None:
        print(f"  [{model_info['label']}] no checkpoint under {airio_root}/{model_info['exp_dir']}/ckpt "
              f"and no released checkpoint for it -- skipping inference")
        return False
    cmd = [
        sys.executable, "inference_motion.py",
        "--config", model_info["config"],
        "--device", model_info["device"],
    ]
    print(f"  [{model_info['label']}] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=airio_root, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{model_info['label']}] inference FAILED")
        print(result.stderr[-2000:])
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-inference", action="store_true", help="Use existing pickle files only")
    ap.add_argument("--device", default="cuda:0", help="Override device for inference")
    ap.add_argument("--data_root", default=NANO_ROOT, help="world-frame NanoBench root (nanobench_processed_2nd_version)")
    ap.add_argument("--airio_root", default=AIRIO_ROOT, help="airio/ (inference_motion.py runs from here)")
    ap.add_argument("--ckpt_root", default=str(CKPT_ROOT), help="released checkpoint folders")
    ap.add_argument("--no_link_ckpts", action="store_true",
                    help="do not symlink released checkpoints into airio/experiments/<exp>/ckpt")
    ap.add_argument("--test_list", default=str(TEST_LIST_FILE))
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Override devices
    for m in MODELS:
        m["device"] = args.device

    # Test sequences (from nanobench config)
    test_seqs = read_test_list(args.test_list)
    print(f"Test sequences: {len(test_seqs)}")

    # 1. Inference for each model
    if not args.skip_inference:
        print("\n=== INFERENCE ===")
        for m in MODELS:
            run_inference(m, args.airio_root, args.ckpt_root, not args.no_link_ckpts)

    # 2. Compute metrics
    print("\n=== METRICS ===")
    all_metrics = {}
    for m in MODELS:
        print(f"  {m['label']}:")
        seq_m = evaluate_model(m, test_seqs, args.data_root, args.airio_root)
        if seq_m is None:
            continue
        all_metrics[m["label"]] = seq_m

    # 3. Summary table
    print("\n=== SUMMARY ===")
    keys = ["ATE_aligned", "RMSE_no_align", "RPE@2s", "drift_endpoint"]
    common = set(test_seqs)
    for d in all_metrics.values():
        common &= set(d.keys())
    print(f"Common sequences: {len(common)}")

    lines = [
        "AWS Round 1 -- Nanobench results (n=17 test seqs, body frame, 80 epochs)",
        f"Common across all models: {len(common)} seqs",
        "",
        f"{'Model':30s} | " + " | ".join(f"{k:>13s}" for k in keys),
        "-" * 95,
    ]
    rows = []
    for mname, d in all_metrics.items():
        means = {k: float(np.mean([d[s][k] for s in common if not np.isnan(d[s][k])])) for k in keys}
        rows.append((mname, means))
        lines.append(f"{mname:30s} | " + " | ".join(f"{means[k]:13.4f}" for k in keys))

    print("\n".join(lines))

    # Save metrics
    out_json = os.path.join(args.out_dir, "round1_nanobench_metrics.json")
    out_txt = os.path.join(args.out_dir, "round1_nanobench_summary.txt")
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {out_json}\n        {out_txt}")


if __name__ == "__main__":
    main()
