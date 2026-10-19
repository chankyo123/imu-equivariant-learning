# Paper artefact: GINIO-A rows of Table 2 (no-att 1.430 / with-att 1.045 Umeyama ATE, the released 0.045M "small"
# models); writes round1_nanobench_metrics_small.json.
"""
Round-1 eval extended to the /3-channel-reduced VN-AirIO ('small') variants.
Reuses the same trajectory integration + Umeyama-aligned ATE pipeline as
round1_eval.py for direct comparability with VN no-att (1.502) / VN with-att (1.062).
"""
import argparse, json, os, pickle, sys, subprocess
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# Import helpers from sibling round1_eval to avoid duplication
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from round1_eval import (  # noqa: E402
    load_gt, umeyama_align, rpe_at_dt, metrics_from_traj,
    compute_world_traj, NANO_ROOT, AIRIO_ROOT, OUT_DIR,
)
from _paths import CKPT_ROOT, TEST_LIST_FILE, ensure_airio_experiment, read_test_list  # noqa: E402

MODELS = [
    {
        "label": "VN no-att (small, /3-reduced)",
        "config": "configs/Nanobench/motion_body_noatt_vn_small.conf",
        "exp_dir": "experiments/nanobench_noatt_vn_small/motion_body_noatt_vn_small",
        "device": "cuda:0",
    },
    {
        "label": "VN with-att (small, /3-reduced)",
        "config": "configs/Nanobench/motion_body_rot_vn_small.conf",
        "exp_dir": "experiments/nanobench_rot_vn_small/motion_body_rot_vn_small",
        "device": "cuda:1",
    },
]


def evaluate_model(model_info, test_seqs, nano_root=NANO_ROOT, airio_root=AIRIO_ROOT):
    pickle_path = os.path.join(airio_root, model_info["exp_dir"], "net_output.pickle")
    if not os.path.exists(pickle_path):
        print(f"  [{model_info['label']}] no pickle -- skipping")
        return None
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    seq_metrics = {}
    for seq in test_seqs:
        if seq not in data:
            continue
        d = data[seq]
        ts_pred = d["ts"].cpu().numpy() if hasattr(d["ts"], "cpu") else np.asarray(d["ts"])
        ts_pred = ts_pred.squeeze()
        net_vel = d["net_vel"].cpu().numpy() if hasattr(d["net_vel"], "cpu") else np.asarray(d["net_vel"])
        if net_vel.ndim == 3:
            net_vel = net_vel[0]
        ts_gt, quat_gt, pos_gt = load_gt(seq, nano_root)
        idxs = np.searchsorted(ts_gt, ts_pred).clip(0, len(ts_gt) - 1)
        quat_at_ts = quat_gt[idxs]
        pos_at_ts = pos_gt[idxs]
        pred_pos = compute_world_traj(net_vel, ts_pred, quat_at_ts)
        pred_pos += (pos_at_ts[0] - pred_pos[0])
        seq_metrics[seq] = metrics_from_traj(pred_pos, pos_at_ts, ts_pred)
    return seq_metrics


def run_inference(model_info, airio_root=AIRIO_ROOT, ckpt_root=CKPT_ROOT, link_ckpts=True):
    pickle_path = os.path.join(airio_root, model_info["exp_dir"], "net_output.pickle")
    if os.path.exists(pickle_path):
        print(f"  [{model_info['label']}] pickle exists, skipping")
        return True
    if link_ckpts and ensure_airio_experiment(model_info["exp_dir"], ckpt_root, airio_root) is None:
        print(f"  [{model_info['label']}] no checkpoint under {airio_root}/{model_info['exp_dir']}/ckpt "
              f"and no released checkpoint for it -- skipping inference")
        return False
    cmd = [sys.executable, "inference_motion.py", "--config", model_info["config"], "--device", model_info["device"]]
    print(f"  [{model_info['label']}] running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=airio_root, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="override the per-model device (default: cuda:0 / cuda:1 as in the original)")
    ap.add_argument("--data_root", default=NANO_ROOT, help="world-frame NanoBench root")
    ap.add_argument("--airio_root", default=AIRIO_ROOT)
    ap.add_argument("--ckpt_root", default=str(CKPT_ROOT))
    ap.add_argument("--no_link_ckpts", action="store_true",
                    help="do not symlink released checkpoints into airio/experiments/<exp>/ckpt")
    ap.add_argument("--test_list", default=str(TEST_LIST_FILE))
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device:
        for m in MODELS:
            m["device"] = args.device

    test_seqs = read_test_list(args.test_list)

    all_results = {}
    for m in MODELS:
        print(f'\n=== {m["label"]} ===')
        run_inference(m, args.airio_root, args.ckpt_root, not args.no_link_ckpts)
        per_seq = evaluate_model(m, test_seqs, args.data_root, args.airio_root)
        if per_seq is None:
            continue
        all_results[m["label"]] = per_seq
        ate = [v["ATE_aligned"] for v in per_seq.values()]
        rmse = [v["RMSE_no_align"] for v in per_seq.values()]
        rpe = [v["RPE@2s"] for v in per_seq.values()]
        drift = [v["drift_endpoint"] for v in per_seq.values()]
        print(f'  ATE mean={np.mean(ate):.4f}  median={np.median(ate):.4f}  RMSE={np.mean(rmse):.3f}  '
              f'RPE@2s={np.mean(rpe):.3f}  drift={np.mean(drift):.3f}  (n={len(ate)})')

    out_path = os.path.join(args.out_dir, "round1_nanobench_metrics_small.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
