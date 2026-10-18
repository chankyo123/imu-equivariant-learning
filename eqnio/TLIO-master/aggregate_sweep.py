"""
Aggregate filter ATE + network ATE from sweep_eqnio_perturb.py outputs.

Reads:
  <sweep_dir>/<model>__<ptype>_<deg>/<seq>/not_vio_state.txt
  <sweep_dir>/net_<model>__<ptype>_<deg>/<seq>/metrics.json (network output via test.py)

Writes:
  <sweep_dir>/results_filter.csv
  <sweep_dir>/results_network.csv

Usage: python aggregate_sweep.py [--sweep_dir /tmp/eqnio_sweep] [--gt_path <repo>/local_data/tlio_golden/]
(ported from Server B; the GT path default is now relative to the repository root)
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP = os.environ.get("EQNIO_SWEEP_DIR", "/tmp/eqnio_sweep")
GT_PATH = os.environ.get("TLIO_GOLDEN", os.path.join(REPO_ROOT, "local_data", "tlio_golden")) + "/"


def parse_slug(name):
    # tlio_o2__gravity_10 / tlio_o2_so3aug__lfa_5
    m = re.match(r"(.+?)__([a-z]+)_([\d.]+)$", name)
    if not m:
        return None
    model, ptype, deg = m.group(1), m.group(2), float(m.group(3))
    return model, ptype, deg


def filter_ate_for_dir(exp_dir):
    """Compute mean filter ATE across all seq subfolders in exp_dir."""
    ates = []
    for folder in sorted(os.listdir(exp_dir)):
        seq_path = os.path.join(exp_dir, folder)
        if not os.path.isdir(seq_path):
            continue
        txt = os.path.join(seq_path, "not_vio_state.txt")
        npy = os.path.join(seq_path, "not_vio_state.txt.npy")
        if os.path.exists(txt):
            kf = np.loadtxt(txt, delimiter=",")
        elif os.path.exists(npy):
            kf = np.load(npy)
        else:
            continue
        if kf.ndim != 2 or kf.shape[0] < 4000:
            continue
        gt_path = os.path.join(GT_PATH, folder, "imu0_resampled.npy")
        if not os.path.exists(gt_path):
            continue
        gt = np.load(gt_path)
        start_idx = 2000
        kf_ts = kf[start_idx:, 27]
        kf_p = kf[start_idx:, 12:15]
        gt_ts = gt[:, 0] * 1e-6
        gt_p = gt[:, -6:-3]
        if gt_ts[0] > kf_ts[0]:
            gt_ts = np.insert(gt_ts, 0, kf_ts[0])
            gt_p = np.concatenate([gt_p[0].reshape(1, 3), gt_p], axis=0)
        if gt_ts[-1] < kf_ts[-1]:
            gt_ts = np.insert(gt_ts, -1, kf_ts[-1])
            gt_p = np.concatenate([gt_p, gt_p[-1].reshape(1, 3)], axis=0)
        inter_gt_p = interp1d(gt_ts, gt_p, axis=0)(kf_ts)
        ate = float(np.sqrt(np.mean(np.linalg.norm(kf_p - inter_gt_p, axis=1) ** 2)))
        ates.append((folder, ate))
    if not ates:
        return None, 0
    mean = float(np.mean([a for _, a in ates]))
    return mean, len(ates)


def network_ate_for_dir(net_dir):
    """Read network metrics.json (one per seq subfolder) — actually test.py writes ONE metrics.json
    at out_dir level. We read it."""
    p = os.path.join(net_dir, "metrics.json")
    if not os.path.exists(p):
        # Try the same dir for any file...
        for name in os.listdir(net_dir):
            cand = os.path.join(net_dir, name, "metrics.json")
            if os.path.exists(cand):
                p = cand
                break
        else:
            return None, 0
    try:
        d = json.load(open(p))
    except Exception:
        return None, 0
    ates = [v.get("ronin", {}).get("ate") for v in d.values() if isinstance(v, dict)]
    ates = [a for a in ates if a is not None]
    if not ates:
        return None, 0
    return float(np.mean(ates)), len(ates)


def main():
    global SWEEP, GT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default=SWEEP)
    ap.add_argument("--gt_path", default=GT_PATH, help="TLIO golden root (contains <seq>/imu0_resampled.npy)")
    a = ap.parse_args()
    SWEEP, GT_PATH = a.sweep_dir, a.gt_path.rstrip("/") + "/"
    rows_filter = []
    rows_net = []
    for entry in sorted(os.listdir(SWEEP)):
        full = os.path.join(SWEEP, entry)
        if not os.path.isdir(full):
            continue
        if entry.startswith("net_"):
            parsed = parse_slug(entry[len("net_"):])
            if not parsed:
                continue
            model, ptype, deg = parsed
            ate, n = network_ate_for_dir(full)
            rows_net.append({"model": model, "ptype": ptype, "deg": deg, "n_seq": n, "net_ate_m": ate})
        else:
            parsed = parse_slug(entry)
            if not parsed:
                continue
            model, ptype, deg = parsed
            ate, n = filter_ate_for_dir(full)
            rows_filter.append({"model": model, "ptype": ptype, "deg": deg, "n_seq": n, "filter_ate_m": ate})

    rows_filter.sort(key=lambda r: (r["model"], r["ptype"], r["deg"]))
    rows_net.sort(key=lambda r: (r["model"], r["ptype"], r["deg"]))

    with open(os.path.join(SWEEP, "results_filter.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["model", "ptype", "deg", "n_seq", "filter_ate_m"])
        w.writeheader()
        w.writerows(rows_filter)
    with open(os.path.join(SWEEP, "results_network.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["model", "ptype", "deg", "n_seq", "net_ate_m"])
        w.writeheader()
        w.writerows(rows_net)

    print("\n=== FILTER ATE ===")
    for r in rows_filter:
        v = r["filter_ate_m"]
        print(f"  {r['model']:18s} {r['ptype']:8s} deg={r['deg']:5.1f}  n={r['n_seq']:2d}  ate={'NaN' if v is None else f'{v:.3f}m'}")
    print("\n=== NETWORK ATE ===")
    for r in rows_net:
        v = r["net_ate_m"]
        print(f"  {r['model']:18s} {r['ptype']:8s} deg={r['deg']:5.1f}  n={r['n_seq']:2d}  ate={'NaN' if v is None else f'{v:.3f}m'}")


if __name__ == "__main__":
    main()
