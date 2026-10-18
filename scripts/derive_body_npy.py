"""
Derive body-frame nanobench from existing world-frame nanobench_processed_2nd_version.

Source npy columns (17):
  ts_us(1), gyr_in_World(3), acc_in_World(3), qxyzw_W_D(4), pos_W_D(3), vel_World(3)

Output npy columns (17, same schema but body-frame IMU + body-frame velocity):
  ts_us(1), gyr_in_Body(3), acc_in_Body(3), qxyzw_W_D(4), pos_W_D(3), vel_Body(3)

Transformations per timestep t:
  R(t) = Rotation.from_quat(qxyzw_W_D[t])      # body -> world
  gyr_B[t] = R(t).T @ gyr_W[t]
  acc_B[t] = R(t).T @ acc_W[t]
  vel_B[t] = R(t).T @ vel_W[t]
  pos_W_D[t], qxyzw_W_D[t]: unchanged (carrying orientation/world pos to allow later LFA)

Run:
  python derive_body_npy.py
"""

import json
import os
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation

SRC = "local_data/nanobench_processed_2nd_version"
DST = "local_data/nanobench_bodyframe_2nd_version"

GRAVITY_W = np.array([0.0, 0.0, -9.80665])


def convert_seq(src_dir, dst_dir, sanity_record):
    os.makedirs(dst_dir, exist_ok=True)
    npy_path = os.path.join(src_dir, "imu0_resampled.npy")
    desc_path = os.path.join(src_dir, "imu0_resampled_description.json")
    cal_path = os.path.join(src_dir, "calibration.json")

    data = np.load(npy_path)  # (N, 17)
    assert data.shape[1] == 17, f"unexpected width {data.shape[1]} in {npy_path}"

    ts = data[:, 0:1]
    gyr_W = data[:, 1:4]
    acc_W = data[:, 4:7]
    quat_W_D = data[:, 7:11]   # (qx, qy, qz, qw) world<-device
    pos_W_D = data[:, 11:14]
    vel_W = data[:, 14:17]

    R = Rotation.from_quat(quat_W_D).as_matrix()    # (N, 3, 3) world<-body
    RT = np.transpose(R, (0, 2, 1))                  # body<-world

    gyr_B = np.einsum("nij,nj->ni", RT, gyr_W)
    acc_B = np.einsum("nij,nj->ni", RT, acc_W)
    vel_B = np.einsum("nij,nj->ni", RT, vel_W)

    out = np.concatenate(
        [ts, gyr_B, acc_B, quat_W_D, pos_W_D, vel_B], axis=1
    ).astype(np.float64)
    np.save(os.path.join(dst_dir, "imu0_resampled.npy"), out)

    with open(desc_path) as f:
        desc = json.load(f)
    desc["columns_name(width)"] = [
        "ts_us(1)",
        "gyr_compensated_in_Body(3)",
        "acc_compensated_in_Body(3)",
        "qxyzw_World_Device(4)",
        "pos_World_Device(3)",
        "vel_Body(3)",
    ]
    desc["frame_note"] = (
        "Derived from world-frame npy by R(t)^T. acc includes gravity in body frame."
    )
    with open(os.path.join(dst_dir, "imu0_resampled_description.json"), "w") as f:
        json.dump(desc, f, indent=2)

    if os.path.exists(cal_path):
        shutil.copy(cal_path, os.path.join(dst_dir, "calibration.json"))

    speed = np.linalg.norm(vel_W, axis=1)
    stationary = speed < 0.05  # < 5 cm/s
    sanity_record["seq"] = os.path.basename(src_dir)
    sanity_record["num_rows"] = int(data.shape[0])
    sanity_record["frac_stationary"] = float(stationary.mean())
    if stationary.sum() > 10:
        acc_B_stat = acc_B[stationary].mean(axis=0)
        sanity_record["acc_B_stationary_mean"] = acc_B_stat.tolist()
        sanity_record["acc_B_stationary_norm"] = float(np.linalg.norm(acc_B_stat))


def main(repo_root):
    src_root = os.path.join(repo_root, SRC)
    dst_root = os.path.join(repo_root, DST)
    os.makedirs(dst_root, exist_ok=True)

    seqs = sorted(
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    )
    print(f"converting {len(seqs)} sequences from {src_root} -> {dst_root}")

    sanity_records = []
    for s in seqs:
        rec = {}
        try:
            convert_seq(
                os.path.join(src_root, s), os.path.join(dst_root, s), rec
            )
            sanity_records.append(rec)
        except Exception as e:
            print(f"  FAIL {s}: {e}", file=sys.stderr)

    # copy split files
    for name in ("train_list.txt", "val_list.txt", "test_list.txt"):
        sp = os.path.join(src_root, name)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(dst_root, name))

    # sanity check summary
    stationary_records = [
        r for r in sanity_records if "acc_B_stationary_mean" in r
    ]
    print(f"\nstationary sanity check on {len(stationary_records)} sequences:")
    if stationary_records:
        norms = np.array(
            [r["acc_B_stationary_norm"] for r in stationary_records]
        )
        z_mean = np.array(
            [r["acc_B_stationary_mean"][2] for r in stationary_records]
        )
        print(f"  acc_B norm @ stationary: mean={norms.mean():.3f} std={norms.std():.3f}")
        print(f"    -> should be ~9.81 if quat convention correct")
        print(f"  acc_B z-component @ stationary: mean={z_mean.mean():.3f} std={z_mean.std():.3f}")
        print(f"    -> should be ~-9.81 (gravity in body z when upright)")
        print(f"\n  per-seq breakdown (first 5):")
        for r in stationary_records[:5]:
            print(
                f"    {r['seq']:40s} "
                f"frac_stat={r['frac_stationary']:.2f} "
                f"acc_B_mean={r['acc_B_stationary_mean']} "
                f"|acc_B|={r['acc_B_stationary_norm']:.3f}"
            )

    print(f"\ndone. body-frame data in {dst_root}")


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    main(repo_root)
