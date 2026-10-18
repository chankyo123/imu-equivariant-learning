#!/usr/bin/env python3
"""
RIDI raw CSV has slight timestamp jitter (~4944 us between samples, not exactly 5000 us).
The filter's IMU buffer expects perfectly periodic timestamps for its
`np.where(net_t_us == t_us_end)[0][0]` exact-match.

Linearly interpolate the IMU readings to a perfectly periodic 5000 us (= 200 Hz) grid.
This is the value the sensor would have produced at those exact times under a smoothness
assumption -- fine since human-motion IMU content is band-limited far below the 200 Hz Nyquist.

Overwrites imu_samples_calibrated.csv IN PLACE with the resampled version (keeps
imu_samples_0.csv as the raw record). Default --root is relative to the repository root.
"""
import argparse
import csv
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "local_data_bodyframe", "ridi_tlio"),
                    help="RIDI (TLIO format) root whose <seq>/imu_samples_calibrated.csv are resampled in place")
    ap.add_argument("--dt_ns", type=int, default=5_000_000, help="target period in ns (default 5e6 = 200 Hz)")
    args = ap.parse_args()
    ROOT, DT_NS = args.root, args.dt_ns

    n = 0
    for seq in sorted(os.listdir(ROOT)):
        sp = os.path.join(ROOT, seq)
        if not os.path.isdir(sp):
            continue
        cf = os.path.join(sp, "imu_samples_calibrated.csv")
        if not os.path.exists(cf):
            continue

        with open(cf) as f:
            rdr = csv.reader(f)
            header = next(rdr)
            rows = np.array([[float(x) for x in r] for r in rdr if len(r) >= 8])

        ts_ns = rows[:, 0].astype(np.float64)
        feats = rows[:, 1:]  # temperature + gyro + acc = 7 cols
        # Snap to a periodic grid that starts at the first sample's timestamp.
        t0 = int(np.round(ts_ns[0]))
        t1 = int(ts_ns[-1])
        new_ts = np.arange(t0, t1 + 1, DT_NS, dtype=np.int64)
        # Per-column linear interpolation
        new_feats = np.stack(
            [np.interp(new_ts.astype(np.float64), ts_ns, feats[:, c]) for c in range(feats.shape[1])],
            axis=1,
        )
        out = np.hstack([new_ts.astype(np.float64).reshape(-1, 1), new_feats])
        with open(cf, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            # Write timestamps as integers (the filter parses them to int) and floats for the rest
            for row in out:
                w.writerow([int(row[0])] + [f"{x:.10g}" for x in row[1:]])
        n += 1

    print(f"RIDI calibrated CSVs resampled to exact {1e9 / DT_NS:.0f} Hz ({n} sequences) under {ROOT}")


if __name__ == "__main__":
    main()
