# Paper artefact: clean trajectory-level Umeyama ATE of EqNIO O(2) / GINIO-E from the Phase-A dumps, used as the
# cross-check of the EqNIO-family entries of canonical_v2.json (Table 2).
"""Quick: compute trajectory-level ATE_um for EqNIO O(2) + VN-EqNIO SO(3) clean baseline
   from window-level prediction dumps, for direct comparison with canonical_v2 numbers.

   Procedure per seq:
     1. Load dumps (W,3) predicted world-frame velocity at stride 20 (= 0.2s @ 100Hz).
     2. Read raw NanoBench npy for GT world-frame position trajectory.
     3. Sub-sample GT position at same stride.
     4. Integrate predicted vel -> predicted trajectory.
     5. Umeyama-align predicted to GT, compute RMSE.

   Note: EqNIO outputs *displacement per window* (1s window_time, so disp ~ velocity).
         For trajectory: position_pred[t+1] = position_pred[t] + mu[t] * stride_dt.
"""
import json
import sys
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA_LFA, OUTPUT_DIR, TEST_LIST_FILE, read_test_list  # noqa: E402

WIN, STRIDE_SAMPLES = 200, 20  # window=200 samples@100Hz=2s; stride=20 samples=0.2s
DT_STRIDE = STRIDE_SAMPLES / 100.0  # 0.2s


def umeyama_rmse(P, Q):
    """P, Q: (N, 3). Estimate similarity (sR + t) that maps P->Q, return RMSE."""
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


def trajectory_ate_for_model(model_name, dumps, data, test):
    per_seq_ate = {}
    for seq in test:
        dump_p = dumps / f'{model_name}__clean__clean__{seq}.npz'
        if not dump_p.exists():
            continue
        d = np.load(dump_p)
        mu = d['mu']  # (W, 3) predicted world-frame velocity (or disp/dt)
        W = mu.shape[0]

        # Read GT trajectory at matching window-end timestamps
        a = np.load(data / seq / 'imu0_resampled.npy')  # (N, 17)
        starts = np.arange(0, a.shape[0] - WIN + 1, STRIDE_SAMPLES)
        # Position at the LAST sample of each window
        gt_pos = np.array([a[s + WIN - 1, 11:14] for s in starts])  # (W, 3)

        # Integrate predicted velocity -> predicted trajectory (anchored at GT start)
        # Cumulative integration: position[i] = position[0] + sum_{k<i} mu[k] * dt
        pred_pos = np.zeros_like(gt_pos)
        pred_pos[0] = gt_pos[0]
        for i in range(1, W):
            pred_pos[i] = pred_pos[i - 1] + mu[i - 1] * DT_STRIDE
        per_seq_ate[seq] = umeyama_rmse(pred_pos, gt_pos)

    if not per_seq_ate:
        return None
    vals = list(per_seq_ate.values())
    return dict(per_seq=per_seq_ate, mean=float(np.mean(vals)), median=float(np.median(vals)), n=len(vals))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out_dir', default=str(OUTPUT_DIR), help='folder with dumps/ from run_phase_ab_eqnio.py; output goes here')
    ap.add_argument('--data_dir', default=str(DATA_LFA), help='body-frame NanoBench root')
    ap.add_argument('--test_list', default=str(TEST_LIST_FILE))
    args = ap.parse_args()
    OUT = Path(args.out_dir)
    DUMPS = OUT / 'dumps'
    TEST = read_test_list(args.test_list)

    out = {}
    for m in ['EqNIO_O2', 'VN_EqNIO_SO3']:
        r = trajectory_ate_for_model(m, DUMPS, Path(args.data_dir), TEST)
        if r:
            out[m] = r
            print(f'{m:15s} | mean ATE_um = {r["mean"]:.4f}  median = {r["median"]:.4f}  n = {r["n"]}')
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'eqnio_trajectory_ate.json', 'w') as f:
        json.dump(out, f, indent=2)
