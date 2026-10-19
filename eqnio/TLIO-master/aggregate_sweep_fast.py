"""Fast aggregator -- reads only the needed columns from the huge EKF state txt files.

Usage: python aggregate_sweep_fast.py [--sweep_dir /tmp/eqnio_sweep] [--gt_path <repo>/local_data/tlio_golden/]
(ported from Server B; the GT path default is now relative to the repository root)
"""
import argparse
import csv
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.environ.get("EQNIO_SWEEP_DIR", "/tmp/eqnio_sweep")
GT_PATH = os.environ.get("TLIO_GOLDEN", os.path.join(REPO_ROOT, "local_data", "tlio_golden")) + "/"

# Columns of interest in not_vio_state.txt: pos x/y/z = 12/13/14, ts = 27
USECOLS = [12, 13, 14, 27]


def parse_slug(name):
    m = re.match(r"(.+?)__([a-z][a-z0-9]*?)_([\d.]+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), float(m.group(3))


def filter_ate_for_seq(txt_path, gt_path, start_idx=2000):
    """Compute filter ATE for ONE sequence. Uses pandas to read only 4 cols (~3 MB instead of 400 MB)."""
    df = pd.read_csv(txt_path, header=None, usecols=USECOLS, dtype=np.float64)
    kf = df.values  # shape (N, 4): [px, py, pz, ts]
    if kf.shape[0] < start_idx + 100:
        return None
    kf_p = kf[start_idx:, :3]
    kf_ts = kf[start_idx:, 3]
    if not os.path.exists(gt_path):
        return None
    gt = np.load(gt_path)
    gt_ts = gt[:, 0] * 1e-6
    gt_p = gt[:, -6:-3]
    # Pad GT bounds if needed
    if gt_ts[0] > kf_ts[0]:
        gt_ts = np.insert(gt_ts, 0, kf_ts[0])
        gt_p = np.concatenate([gt_p[0:1], gt_p], axis=0)
    if gt_ts[-1] < kf_ts[-1]:
        gt_ts = np.insert(gt_ts, -1, kf_ts[-1])
        gt_p = np.concatenate([gt_p, gt_p[-1:]], axis=0)
    inter_gt_p = interp1d(gt_ts, gt_p, axis=0)(kf_ts)
    return float(np.sqrt(np.mean(np.linalg.norm(kf_p - inter_gt_p, axis=1) ** 2)))


def aggregate():
    rows = []
    cfg_dirs = sorted([d for d in os.listdir(SWEEP)
                       if os.path.isdir(os.path.join(SWEEP, d)) and not d.startswith("net_")])
    for cfg in cfg_dirs:
        cfg_dir = os.path.join(SWEEP, cfg)
        parsed = parse_slug(cfg)
        if not parsed:
            continue
        model, ptype, deg = parsed
        ates = []
        for folder in sorted(os.listdir(cfg_dir)):
            seq_dir = os.path.join(cfg_dir, folder)
            txt = os.path.join(seq_dir, "not_vio_state.txt")
            if not (os.path.isdir(seq_dir) and os.path.exists(txt)):
                continue
            gt_path = os.path.join(GT_PATH, folder, "imu0_resampled.npy")
            try:
                ate = filter_ate_for_seq(txt, gt_path)
            except Exception as e:
                print(f"  ERR {cfg}/{folder}: {e}")
                continue
            if ate is not None:
                ates.append(ate)
        mean_ate = float(np.mean(ates)) if ates else None
        rows.append({"model": model, "ptype": ptype, "deg": deg,
                     "n_seq": len(ates), "filter_ate_m": mean_ate})
        print(f"  {model:20s} {ptype:8s} deg={deg:5.1f}  n={len(ates):2d}  "
              f"ate={mean_ate:.4f}" if mean_ate else f"  {model:20s} {ptype:8s} deg={deg:5.1f}  N/A", flush=True)

    rows.sort(key=lambda r: (r["model"], r["ptype"], r["deg"]))
    out_csv = os.path.join(SWEEP, "results_filter.csv")
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=["model", "ptype", "deg", "n_seq", "filter_ate_m"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv} with {len(rows)} rows")
    return rows


def network_aggregate():
    """Read network metrics.json from net_* dirs."""
    import json
    rows = []
    for entry in sorted(os.listdir(SWEEP)):
        if not entry.startswith("net_"):
            continue
        full = os.path.join(SWEEP, entry)
        parsed = parse_slug(entry[len("net_"):])
        if not parsed:
            continue
        model, ptype, deg = parsed
        ates = []
        # Network metrics.json is at out_dir level (test.py writes once at end)
        mj = os.path.join(full, "metrics.json")
        if os.path.exists(mj):
            try:
                d = json.load(open(mj))
                for v in d.values():
                    if isinstance(v, dict):
                        a = v.get("ronin", {}).get("ate")
                        if a is not None:
                            ates.append(a)
            except Exception as e:
                print(f"  ERR {entry}: {e}")
        mean_ate = float(np.mean(ates)) if ates else None
        rows.append({"model": model, "ptype": ptype, "deg": deg,
                     "n_seq": len(ates), "net_ate_m": mean_ate})

    rows.sort(key=lambda r: (r["model"], r["ptype"], r["deg"]))
    out_csv = os.path.join(SWEEP, "results_network.csv")
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=["model", "ptype", "deg", "n_seq", "net_ate_m"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv} with {len(rows)} rows")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default=SWEEP)
    ap.add_argument("--gt_path", default=GT_PATH, help="TLIO golden root (contains <seq>/imu0_resampled.npy)")
    a = ap.parse_args()
    SWEEP, GT_PATH = a.sweep_dir, a.gt_path.rstrip("/") + "/"
    print("=== FILTER ATE ===")
    aggregate()
    print("\n=== NETWORK ATE ===")
    network_aggregate()
