# Paper artefact: the Table-2 metric definition (Umeyama-aligned ATE, unaligned RMSE, RPE@2s, end-point drift)
# and the cross-model NanoBench comparison (per-model/per-sequence metrics table + trajectory overlays) that
# preceded canonical_v2.json; round1_eval.py and build_v2_paper_quality.py reuse this trajectory math.
"""
Unified evaluation script: same ATE/RMSE/RPE definitions for all models on nanobench test split.

Workflow:
  1. For each model, load its predicted body-frame velocity per sequence.
  2. Rotate predicted body vel to world via GT orientation.
  3. Integrate to predicted world trajectory.
  4. Compute consistent metrics:
       - ATE_aligned: align traj (umeyama: rotation+translation, no scale) then RMSE
       - RMSE: no alignment, RMSE of pos error
       - RPE@2s: RMSE of (dp - dp_gt) over 2-sec windows in integrated trajectory
       - drift: |pos_end_err| / traj_length

Loaders per model:
  - imu-eq (ResNet, ln_resnet; src/main_net.py --mode test): per-seq trajectory.txt has
    [ts, pred_xyz, gt_xyz, pred_vel_xyz]
  - AirIO (airio/inference_motion.py): net_output.pickle dict per seq -> 'net_vel' is body-frame predicted vel
  - EqNIO (eqnio/TLIO-master/src/main_net.py --mode test): same imu-eq trajectory.txt format

Model inputs are given on the command line (the original script hard-coded them):
  --imueq_out DIR            root holding one sub-directory per Lie-Neuron test run
  --imueq_models name=subdir one entry per model (default: the released NanoBench checkpoint names)
  --airio_pickle [name=]path one entry per Air-IO net_output.pickle (bare path -> name "AirIO_body")
  --eqnio_dir [name=]dir     one entry per EqNIO/GINIO-E test-output directory
Models whose outputs are absent are skipped.

Outputs (under --out_dir):
  - all_models_metrics.json (per-model, per-seq)
  - summary.txt (table), per_sequence.csv
  - plots/{seq}.png (multi-model overlay 2D xy plot, one per sequence)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO, AIRIO_ROOT, DATA_WORLD, OUTPUT_DIR, TEST_LIST_FILE, read_test_list  # noqa: E402

# Defaults (repo-relative). Former hard-coded roots: NANO_ROOT (nanobench_processed_2nd_version), IMUEQ_OUT
# (imu-equivariant-learning/batch_test_outputs_nano_body), AIRIO_NET (Air-IO/experiments/nanobench/motion_body_rot/
# net_output.pickle), RESULTS_ROOT/unified_metrics + trajectory_plots/4model_overlay.
NANO_ROOT = str(DATA_WORLD)
IMUEQ_OUT = str(REPO / "test_outputs")
DEFAULT_IMUEQ_MODELS = [
    # name=subdir under --imueq_out; subdir = --out_dir you gave `src/main_net.py --mode test`
    "nanobench_ginio_rd_t55=nanobench_ginio_rd_t55",
    "nanobench_ginio_rs_chol_mse=nanobench_ginio_rs_chol_mse",
    "nanobench_ginio_rs_chol_nll=nanobench_ginio_rs_chol_nll",
    "nanobench_ginio_rd_lfa_mse=nanobench_ginio_rd_lfa_mse",
    "nanobench_ginio_rd_lfa_nll=nanobench_ginio_rd_lfa_nll",
]
DEFAULT_AIRIO_PICKLES = [
    f"nanobench_airio_noatt={AIRIO_ROOT}/experiments/nanobench_noatt/motion_body_noatt/net_output.pickle",
    f"nanobench_airio_withatt={AIRIO_ROOT}/experiments/nanobench/motion_body_rot/net_output.pickle",
    f"nanobench_ginio_a_noatt={AIRIO_ROOT}/experiments/nanobench_noatt_vn_small/motion_body_noatt_vn_small/net_output.pickle",
    f"nanobench_ginio_a_withatt={AIRIO_ROOT}/experiments/nanobench_rot_vn_small/motion_body_rot_vn_small/net_output.pickle",
]
DEFAULT_EQNIO_DIRS = [
    f"nanobench_ginio_e_so3aug_tlio={REPO}/test_outputs/nanobench_ginio_e_so3aug_tlio",
]


def load_gt(seq, nano_root=NANO_ROOT):
    """Load GT trajectory for a sequence from the source npy."""
    arr = np.load(f"{nano_root}/{seq}/imu0_resampled.npy")
    ts = arr[:, 0] / 1e6
    quat = arr[:, 7:11]
    pos = arr[:, 11:14]
    vel_W = arr[:, 14:17]
    return ts, quat, pos, vel_W


def integrate_vel_world(ts, vel_W, pos_init):
    """Integrate world-frame vel to world-frame pos using trapezoid rule."""
    pos = np.zeros_like(vel_W)
    pos[0] = pos_init
    dt = np.diff(ts)
    # midpoint
    v_mid = 0.5 * (vel_W[:-1] + vel_W[1:])
    pos[1:] = pos_init + np.cumsum(v_mid * dt[:, None], axis=0)
    return pos


def umeyama_alignment(src, dst, with_scale=False):
    """Align src to dst (both N,3) via rotation + translation (no scale by default).
    Returns aligned src."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return (R @ src.T).T + t


def metrics_from_traj(pred_pos, gt_pos, ts, rpe_window_s=2.0):
    """Compute ATE_aligned, RMSE_no_align, RPE@2s, drift."""
    # truncate to min length
    n = min(len(pred_pos), len(gt_pos), len(ts))
    pred_pos = pred_pos[:n]
    gt_pos = gt_pos[:n]
    ts = ts[:n]

    # RMSE no align
    err = pred_pos - gt_pos
    rmse_no_align = np.sqrt((err ** 2).sum(axis=1).mean())

    # ATE aligned
    pred_aligned = umeyama_alignment(pred_pos, gt_pos)
    err_a = pred_aligned - gt_pos
    ate_aligned = np.sqrt((err_a ** 2).sum(axis=1).mean())

    # RPE@2s
    dt = ts[1] - ts[0]
    step = max(1, int(round(rpe_window_s / dt)))
    if n > step:
        dp = pred_pos[step:] - pred_pos[:-step]
        dp_gt = gt_pos[step:] - gt_pos[:-step]
        rpe = np.linalg.norm(dp - dp_gt, axis=1)
        rpe_rmse = np.sqrt((rpe ** 2).mean())
    else:
        rpe_rmse = np.nan

    # drift: end-point error / trajectory length
    traj_len = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum()
    end_err = np.linalg.norm(pred_pos[-1] - gt_pos[-1])
    drift = end_err / max(traj_len, 1e-6)

    return {
        "ATE_aligned": float(ate_aligned),
        "RMSE_no_align": float(rmse_no_align),
        "RPE@2s": float(rpe_rmse),
        "drift_endpoint": float(drift),
        "traj_len": float(traj_len),
        "n_samples": int(n),
    }


def load_imueq_pred(model_dir, seq):
    """imu-eq trajectory.txt format: [ts, pred_xyz, gt_xyz, pred_vel_xyz] (10 cols).
    model_dir: directory holding one sub-directory per sequence (also used for EqNIO outputs)."""
    p = f"{model_dir}/{seq}/trajectory.txt"
    if not os.path.exists(p):
        return None
    traj = np.loadtxt(p, delimiter=",")
    ts = traj[:, 0]
    pred_pos = traj[:, 1:4]
    gt_pos = traj[:, 4:7]
    return ts, pred_pos, gt_pos


def load_airio_pred(airio_net, seq):
    """AirIO net_output.pickle: dict[seq] with body-frame predicted vel ('net_vel') + timestamps ('ts')."""
    with open(airio_net, "rb") as f:
        d = pickle.load(f)
    # the pickle is structured per-sequence -- find the right key
    if seq not in d:
        # AirIO key might be by data_drive entry index, not name
        return None
    s = d[seq]
    vel_body = np.asarray(s["net_vel"])
    ts_us = np.asarray(s["ts"]).flatten()
    return ts_us, vel_body


def compute_airio_world_traj(airio_net, seq, nano_root=NANO_ROOT):
    """For AirIO: load body vel, rotate to world via GT orientation interp, integrate."""
    pickled = load_airio_pred(airio_net, seq)
    if pickled is None:
        return None
    ts_pred_raw, vel_body = pickled
    if vel_body.ndim == 3:
        vel_body = vel_body[0]  # (N, 3)
    ts_pred = ts_pred_raw.astype(float)
    # AirIO ts is already in seconds (unix epoch, ~1.77e9). GT ts in source npy is in microseconds (~1.77e15).
    # Only divide if it looks like microseconds.
    if ts_pred.max() > 1e12:
        ts_pred = ts_pred / 1e6

    gt_ts, gt_quat, gt_pos, _ = load_gt(seq, nano_root)
    # interpolate GT orientation at predicted timestamps
    from scipy.interpolate import interp1d
    n = min(len(ts_pred), len(vel_body))
    ts_pred = ts_pred[:n]
    vel_body = vel_body[:n]

    # match each pred_ts to closest GT idx
    idx = np.clip(np.searchsorted(gt_ts, ts_pred), 0, len(gt_ts) - 1)
    R_arr = Rotation.from_quat(gt_quat[idx]).as_matrix()  # (N, 3, 3) world<-body
    vel_W = np.einsum("nij,nj->ni", R_arr, vel_body)

    pos_init = gt_pos[idx[0]]
    pred_pos_W = integrate_vel_world(ts_pred, vel_W, pos_init)
    gt_pos_at_pred = gt_pos[idx]
    return ts_pred, pred_pos_W, gt_pos_at_pred


def parse_pairs(items, default_label=None, root=None):
    """'name=path' -> (name, path); bare 'path' -> (default_label or basename, path). Relative paths are
    resolved against `root` when given."""
    out = []
    for it in items or []:
        if "=" in it:
            name, path = it.split("=", 1)
        else:
            path = it
            name = default_label or os.path.basename(os.path.normpath(it))
        if root is not None and not os.path.isabs(path):
            path = os.path.join(root, path)
        out.append((name, path))
    return out


def load_traj(kind, path, seq, nano_root):
    """Uniform loader -> (ts, pred_pos, gt_pos) or None."""
    if kind == "airio":
        return compute_airio_world_traj(path, seq, nano_root)
    return load_imueq_pred(path, seq)


def run(args):
    nano_root = args.data_root
    out_dir = args.out_dir
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    metrics_out = os.path.join(out_dir, "all_models_metrics.json")
    summary_out = os.path.join(out_dir, "summary.txt")
    per_seq_csv = os.path.join(out_dir, "per_sequence.csv")

    test_seqs = read_test_list(args.test_list)
    print(f"Test sequences: {len(test_seqs)}")

    # Model sources: name -> (kind, path). kind "trajdir" = imu-eq / EqNIO trajectory.txt tree, "airio" = pickle.
    sources = {}
    for name, sub in parse_pairs(args.imueq_models, root=args.imueq_out):
        sources[name] = ("trajdir", sub)
    for name, p in parse_pairs(args.eqnio_dir):
        sources[name] = ("trajdir", p)
    for name, p in parse_pairs(args.airio_pickle, default_label="AirIO_body"):
        sources[name] = ("airio", p)

    all_metrics = {}  # {model: {seq: metrics}}

    for model_name, (kind, path) in sources.items():
        if not os.path.exists(path):
            print(f"  {model_name}: {path} not found -- skipping model")
            continue
        all_metrics[model_name] = {}
        for seq in test_seqs:
            try:
                r = load_traj(kind, path, seq, nano_root)
            except Exception as e:
                print(f"  {model_name} {seq}: {e}")
                continue
            if r is None:
                if kind == "airio":
                    print(f"  {model_name}: skip {seq} (not in net_output.pickle)")
                continue
            ts, pp_pos, gg_pos = r
            m = metrics_from_traj(pp_pos, gg_pos, ts)
            all_metrics[model_name][seq] = m
        if not all_metrics[model_name]:
            print(f"  {model_name}: no sequences evaluated -- dropping from the table")
            del all_metrics[model_name]

    # save raw
    with open(metrics_out, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved per-seq metrics to {metrics_out}")

    # summary table
    summary_lines = ["========== UNIFIED METRICS (mean over common test seqs) =========="]
    common = set(test_seqs)
    for mname, d in all_metrics.items():
        common &= set(d.keys())
    summary_lines.append(f"Common sequences across all models: {len(common)}")
    if not all_metrics:
        summary_lines.append("No model outputs found -- abort summary")
        print("\n".join(summary_lines))
        with open(summary_out, "w") as f: f.write("\n".join(summary_lines))
        return
    if not common:
        summary_lines.append("No common sequences -- abort summary")
        print("\n".join(summary_lines))
        with open(summary_out, "w") as f: f.write("\n".join(summary_lines))
        return

    keys = ["ATE_aligned", "RMSE_no_align", "RPE@2s", "drift_endpoint"]
    summary_lines.append("")
    summary_lines.append(f"{'Model':22s} | " + " | ".join(f"{k:>13s}" for k in keys))
    summary_lines.append("-" * 80)
    for mname in all_metrics:
        d = all_metrics[mname]
        means = {k: np.mean([d[s][k] for s in common if not np.isnan(d[s][k])]) for k in keys}
        summary_lines.append(f"{mname:22s} | " + " | ".join(f"{means[k]:13.4f}" for k in keys))

    # per-seq table
    summary_lines.append("")
    summary_lines.append("--- Per-sequence ATE_aligned ---")
    summary_lines.append(f"{'sequence':40s} | " + " | ".join(f"{m:>16s}" for m in all_metrics))
    for s in sorted(common):
        vals = [all_metrics[m].get(s, {}).get("ATE_aligned", np.nan) for m in all_metrics]
        summary_lines.append(f"{s:40s} | " + " | ".join(f"{v:16.3f}" for v in vals))

    text = "\n".join(summary_lines)
    print(text)
    with open(summary_out, "w") as f: f.write(text)
    print(f"\nSaved summary to {summary_out}")

    # CSV per-seq
    model_names = list(all_metrics.keys())
    with open(per_seq_csv, "w") as f:
        f.write("sequence,metric," + ",".join(model_names) + "\n")
        for k in keys:
            for s in sorted(common):
                vals = [f"{all_metrics[m].get(s, {}).get(k, float('nan')):.4f}" for m in model_names]
                f.write(f"{s},{k}," + ",".join(vals) + "\n")
    print(f"Saved per-seq CSV to {per_seq_csv}")

    # multi-model overlay plots per sequence
    print(f"\nGenerating overlay plots in {plot_dir}/")
    colors = {
        "ResNet_body_noLFA": "tab:blue",
        "ln_resnet_body_LFA": "tab:orange",
        "AirIO_body": "tab:green",
        "EqNIO_O2": "tab:red",
    }
    for seq in sorted(common):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # gt from any model (same)
        gt_pos = None
        for mname in all_metrics:
            if seq in all_metrics[mname]:
                # reload traj for plotting
                kind, path = sources[mname]
                r = load_traj(kind, path, seq, nano_root)
                if r is None: continue
                ts, pp_pos, gg_pos = r
                if gt_pos is None:
                    ax.plot(gg_pos[:, 0], gg_pos[:, 1], "k-", linewidth=2.5, label="GT", zorder=10)
                    gt_pos = gg_pos
                ax.plot(pp_pos[:, 0], pp_pos[:, 1], "--",
                        color=colors.get(mname, "gray"), linewidth=1.5,
                        label=f"{mname} (ATE={all_metrics[mname][seq]['ATE_aligned']:.2f})")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{seq} -- 2D xy trajectory (top-down)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{seq}.png", dpi=100)
        plt.close()
    print(f"  -> {len(common)} plots saved to {plot_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=NANO_ROOT, help="world-frame NanoBench root (GT trajectories)")
    ap.add_argument("--test_list", default=str(TEST_LIST_FILE))
    ap.add_argument("--imueq_out", default=IMUEQ_OUT,
                    help="root holding Lie-Neuron test outputs (src/main_net.py --mode test --out_dir <imueq_out>/<subdir>)")
    ap.add_argument("--imueq_models", nargs="*", default=DEFAULT_IMUEQ_MODELS, metavar="NAME=SUBDIR",
                    help="Lie-Neuron models as name=subdir (subdir relative to --imueq_out)")
    ap.add_argument("--airio_pickle", nargs="*", default=DEFAULT_AIRIO_PICKLES, metavar="[NAME=]PATH",
                    help="Air-IO net_output.pickle files (airio/inference_motion.py); bare path -> name AirIO_body")
    ap.add_argument("--eqnio_dir", nargs="*", default=DEFAULT_EQNIO_DIRS, metavar="[NAME=]DIR",
                    help="EqNIO / GINIO-E test-output dirs with per-sequence trajectory.txt")
    ap.add_argument("--out_dir", default=str(OUTPUT_DIR / "unified"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
