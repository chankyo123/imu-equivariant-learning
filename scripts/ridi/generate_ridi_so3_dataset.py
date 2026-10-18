#!/usr/bin/env python3
"""
Generate an SO(3)-rotated copy of body-frame RIDI for the filter equivariance test (RIDI ID/SO(3)).

For each sequence in <id_root>/test_list.txt:
  1. Sample a uniform random rotation R_so3 in SO(3) (seeded by the sequence name for reproducibility)
  2. Rotate every CSV row's [gyr_3, acc_3] of imu_samples_calibrated.csv: gyr' = R_so3 @ gyr, acc' = R_so3 @ acc
  3. Save R_so3 (3x3) into the sequence dir as R_so3_init.npy

The filter, when given --so3_rot_path "<so3_root>/{seq}/R_so3_init.npy" (eqnio/TLIO-master/src/
main_filter.py or main_filter_riekf.py), right-multiplies its VIO-derived init R by R_so3^T so that
the integrated world trajectory is identical to the identity case for an exactly equivariant model.

Other files (npy, calibration.json, description.json, raw csv) are SYMLINKED back to the
identity dir to save space -- the filter does not read them for the rotated quantities.
Defaults are relative to the repository root; override with --id_root/--so3_root.
"""
import argparse
import csv
import hashlib
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id_root", default=os.path.join(REPO_ROOT, "local_data_bodyframe", "ridi_tlio"),
                    help="identity (unrotated) body-frame RIDI root")
    ap.add_argument("--so3_root", default=os.path.join(REPO_ROOT, "local_data_bodyframe", "ridi_tlio_so3"),
                    help="output root for the SO(3)-rotated copy")
    args = ap.parse_args()
    ID_ROOT, SO3_ROOT = args.id_root, args.so3_root

    os.makedirs(SO3_ROOT, exist_ok=True)
    with open(os.path.join(ID_ROOT, "test_list.txt")) as f:
        test_seqs = [s.strip() for s in f if s.strip()]

    # Copy test_list verbatim so the filter finds the same sequences
    shutil.copy(os.path.join(ID_ROOT, "test_list.txt"), os.path.join(SO3_ROOT, "test_list.txt"))

    for seq in test_seqs:
        sp_in = os.path.join(ID_ROOT, seq)
        sp_out = os.path.join(SO3_ROOT, seq)
        os.makedirs(sp_out, exist_ok=True)

        # Deterministic per-sequence RNG so re-runs use the same R_so3
        seed = int(hashlib.md5(seq.encode()).hexdigest()[:8], 16)
        R_so3 = Rotation.random(random_state=np.random.RandomState(seed)).as_matrix()
        np.save(os.path.join(sp_out, "R_so3_init.npy"), R_so3)

        # Rotate the calibrated CSV
        cin = os.path.join(sp_in, "imu_samples_calibrated.csv")
        cout = os.path.join(sp_out, "imu_samples_calibrated.csv")
        with open(cin) as f:
            r = csv.reader(f)
            header = next(r)
            rows = np.array([[float(x) for x in row] for row in r if len(row) >= 8])
        ts = rows[:, 0:1]
        tmp = rows[:, 1:2]
        g = rows[:, 2:5]  # N x 3
        a = rows[:, 5:8]  # N x 3
        g_r = (R_so3 @ g.T).T
        a_r = (R_so3 @ a.T).T
        out = np.hstack([ts, tmp, g_r, a_r])
        with open(cout, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in out:
                w.writerow([int(row[0])] + [f"{x:.10g}" for x in row[1:]])

        # Symlink the other files -- the filter reads them as-is
        for fn in ("imu0_resampled.npy", "imu0_resampled_description.json",
                   "calibration.json", "imu_samples_0.csv"):
            src = os.path.join(sp_in, fn)
            dst = os.path.join(sp_out, fn)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

    print(f"SO(3) dataset: {SO3_ROOT}")
    print(f"seqs: {len(test_seqs)}")


if __name__ == "__main__":
    main()
