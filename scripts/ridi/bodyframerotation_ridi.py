#!/usr/bin/env python3
"""
Body-frame rotation for RIDI sequences (TLIO-format directories).
Math verified bit-exact against the TLIO body-frame output on Server B.

For each sequence directory in --src:
  - load imu0_resampled.npy (cols: ts gyr(3) acc(3) quat(4) pos(3) vel(3) -- 17)
  - quat is qxyzw_World_Device, so R = R_W_D; R^T rotates world vectors into device (body)
  - gyr_in_World, acc_in_World, vel_World  ->  multiply by R^T  ->  body frame
  - pos kept in world (used as GT for ATE comparison)
  - quat kept as-is (the filter needs it for initialisation)
  - all other files copied as-is (calibration.json, csv); imu0_resampled_description.json
    column names are renamed *_in_World -> *_in_Body, vel_World -> vel_Body

Defaults are relative to the repository root (local_data/...); override with --src/--dst.
"""
import argparse
import json
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(REPO_ROOT, "local_data", "ridi_tlio"),
                    help="world-frame RIDI (TLIO format) root")
    ap.add_argument("--dst", default=os.path.join(REPO_ROOT, "local_data_bodyframe", "ridi_tlio"),
                    help="output body-frame root")
    args = ap.parse_args()
    SRC, DST = args.src, args.dst

    os.makedirs(DST, exist_ok=True)
    processed, skipped = 0, 0
    for subdir in sorted(os.listdir(SRC)):
        sp = os.path.join(SRC, subdir)
        if not os.path.isdir(sp):
            continue
        npy_path = os.path.join(sp, "imu0_resampled.npy")
        if not os.path.exists(npy_path):
            skipped += 1
            continue
        data = np.load(npy_path)
        assert data.shape[1] == 17, f"unexpected cols in {subdir}: {data.shape}"
        quat = data[:, 7:11]
        R = Rotation.from_quat(quat).as_matrix()           # R_W_D
        Rinv = np.transpose(R, (0, 2, 1))                   # R_D_W
        gyr_b = np.einsum("ijk,ik->ij", Rinv, data[:, 1:4])
        acc_b = np.einsum("ijk,ik->ij", Rinv, data[:, 4:7])
        vel_b = np.einsum("ijk,ik->ij", Rinv, data[:, 14:17])
        out = np.concatenate([data[:, :1], gyr_b, acc_b, quat, data[:, 11:14], vel_b], axis=1)

        dst_dir = os.path.join(DST, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        np.save(os.path.join(dst_dir, "imu0_resampled.npy"), out)

        for fn in os.listdir(sp):
            if fn.endswith(".npy"):
                continue
            src_f = os.path.join(sp, fn)
            dst_f = os.path.join(dst_dir, fn)
            if fn == "imu0_resampled_description.json":
                with open(src_f) as f:
                    j = json.load(f)
                cols = j.get("columns_name(width)", [])
                for i, c in enumerate(cols):
                    if "in_World" in c:
                        cols[i] = c.replace("in_World", "in_Body")
                    if c == "vel_World(3)":
                        cols[i] = "vel_Body(3)"
                j["columns_name(width)"] = cols
                with open(dst_f, "w") as f:
                    json.dump(j, f, indent=4)
            else:
                shutil.copy(src_f, dst_f)
        processed += 1

    # Copy non-sequence files (test_list.txt, etc.)
    for fn in os.listdir(SRC):
        fp = os.path.join(SRC, fn)
        if os.path.isfile(fp):
            shutil.copy(fp, DST)

    print(f"processed={processed} skipped={skipped}")
    print(f"dst: {DST}")


if __name__ == "__main__":
    main()
