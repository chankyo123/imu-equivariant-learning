# Data

All three code families read the **TLIO layout**: one folder per sequence plus split lists at the
dataset root. Datasets live under `local_data/` (git-ignored); the split lists we used are
committed under [`splits/`](../splits/).

```text
local_data/<dataset>/
├── <sequence>/
│   ├── imu0_resampled.npy               # (N, 17): ts_us | gyr(3) | acc(3) | q_xyzw_World_Device(4) | pos_World(3) | vel(3)
│   ├── imu0_resampled_description.json  # column names, units, sampling rate
│   ├── calibration.json                 # offline IMU calibration (biases, scale/misalignment)
│   └── imu_samples_0.csv                # raw IMU stream; only needed to run the filters
├── train_list.txt
├── val_list.txt
└── test_list.txt
```

Frames: `gyr`/`acc` are world-frame in the `*_processed_*` and `tlio_golden` sets and body-frame
in the `*_bodyframe_*` sets; `vel` follows the same convention. The Lie-Neuron loader picks the
branch from the **path**: a path containing `bodyframe` selects the sensor-frame branch, and a
path containing `nolfa` turns Last-Frame Alignment off (the two NanoBench body-frame folders are
the same bytes; the `nolfa` one is a symlink).

## TLIO golden (Table 1, Supp. Tables 5–7, Fig. 5/11/12)

Public release of the TLIO authors (CC BY-NC). 284 / 36 / 36 sequences, 200 Hz, world frame.

```bash
pip install gdown
gdown 14YKW7PsozjHo_EdxivKvumsQB7JMw1eg          # golden-new-format-cc-by-nc-with-imus-v1.5.zip
mkdir -p local_data && unzip golden-new-format-cc-by-nc-with-imus-v1.5.zip -d local_data/
```

The filter runs also need the rotated test copy `so3_60` (EqNIO's ID/SO(3) protocol); generate it
with `eqnio/TLIO-master/src/dataprep_test_augmentation/dataprep_ekf.py` (edit the `path` constant).

## RIDI (Table 1, Supp. Table 9)

Public `data_publish_v2` release of the RIDI authors. Convert to the TLIO layout with the scripts
under `scripts/ridi/` (`resample_ridi_to_periodic.py` → `bodyframerotation_ridi.py` for the
sensor-frame copy, `generate_ridi_so3_dataset.py` for the rotated test set). Split lists:
`splits/ridi_tlio/`.

## NanoBench (Table 2, Supp. Tables 8–9, Fig. 8–11)

Crazyflie 2.1 flights with Vicon ground truth ([NanoBench](https://arxiv.org/abs/2603.09908)).
We use a 100 Hz world-frame conversion with gravity **removed** from the accelerometer
(`nanobench_processed_2nd_version`; the description JSON that ships with it incorrectly says
gravity is included — irrelevant for the NN-only Table 2, fatal for a filter run). Splits:
74 / 16 / 17 sequences, `splits/nanobench_processed_2nd_version/`. The three most aggressive test
sequences (`B11_random_waypoints_rep1`, `B9_trefoil_fast_rep8`, `B3_figure8_fast_rep2`) are also
reported separately in the paper.

```bash
# sensor-frame copy for GINIO-RD/RS, and the no-LFA alias
python scripts/derive_body_npy.py                                 # SRC/DST at the top of the script
ln -s nanobench_bodyframe_2nd_version local_data/nanobench_bodyframe_nolfa_2nd_version
```

Our processed split is being published as a data release (see the README *News*); until then,
build it from the NanoBench raw data with `scripts/derive_body_npy.py` and the lists in `splits/`.

## Fetch remounting (Table 3 / camera-ready Tables 10–12)

Collected by us: an Adafruit BNO055 inside a Fetch mobile base, 23 nominal-mount sequences and
23 + 23 sequences with the IMU physically remounted about the x and y axes (TC1 / TC2). Shipped as
`fetch_dataset.zip` with the paper supplement (TLIO layout, 69 sequences); it will be mirrored on
the release page. The models, configs and loader used for the Fetch rows were not recovered from the
original machine (see `docs/KNOWN_ISSUES.md`).

## AquaticVision (Appendix K)

Auxiliary evaluation; data, code and models were not recovered (see `docs/KNOWN_ISSUES.md`).
