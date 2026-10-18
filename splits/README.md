# Dataset splits used in the paper

These are the exact `train_list.txt` / `val_list.txt` / `test_list.txt` files that the released
checkpoints were trained and evaluated with. Place each dataset under `local_data/<dataset>/` in
the TLIO layout and copy (or symlink) the three lists to the dataset root; every loader in `src/`,
`eqnio/` and `airio/` reads them from there. Table numbers below follow the camera-ready PDF.

| folder | dataset copy the lists belong to | train / val / test | used by | notes |
|---|---|---:|---|---|
| `tlio_golden/` | TLIO golden release (`golden-new-format-cc-by-nc-with-imus-v1.5`), 200 Hz, world-frame IMU | 283 / 35 / 36 | Table 1 and Table 5 TLIO rows, Tables 6-7, Table 9 TLIO rows, Fig. 5, Fig. 11 (left), Fig. 12; the TLIO-trained GINIO-E of Table 2 | the handoff `train_list.txt` and `val_list.txt` each contain one blank line (284 / 36 lines); the sequence sets are 283 / 35 / 36 and the three sets are disjoint. The sensor-frame (LFA) copy `local_data_bodyframe/tlio_golden` uses the same lists |
| `tlio_golden/test_list1.txt` | 5 of the 36 test sequences | 5 | Fig. 5(b) epoch selection (`tlio_fig5b_*`) | the frozen-mean covariance models were selected on these 5 *test* sequences, not on the validation split |
| `ridi/` | RIDI `data_publish_v2` converted to the TLIO layout (`ridi_tlio/`, world frame, 200 Hz) | 49 / 23 / 23 | Table 1 and Table 5 RIDI rows, Table 9 RIDI rows, main-text NN-only comparison (GINIO 0.927 vs IMUNet) | **`val_list.txt` is identical to `test_list.txt`**, so `checkpoint_best.pt` of every RIDI model was selected on the test sequences. `train_list.txt` equals RIDI's official `list_train_publish_v2.txt` (49 sequences); `test_list.txt` is RIDI's official `list_test_publish_v2.txt` (25 sequences) minus `hang_leg_new3` and `huayi_handheld_test1`. The official RIDI lists (`list_*_publish_v2.txt`, with placement labels, and `list_crosssubject.txt`, `list_submission.txt`) are included for reference only; the paper used the 23-sequence `test_list.txt`. Server B's filter runs for the RIDI EqNIO rows covered 22 of the 23 sequences (`yajie_handheld2` missing) |
| `nanobench/` | our processed NanoBench (`nanobench_processed_2nd_version`, 100 Hz, Vicon ground truth) and the two sensor-frame derivatives `nanobench_bodyframe_2nd_version` (LFA on) and `nanobench_bodyframe_nolfa_2nd_version` (LFA off) | 74 / 16 / 17 | Table 2, Table 4, Table 8, Table 9 NanoBench rows, Figs. 8-11 (right) | the three NanoBench copies share byte-identical lists (verified on both machines), so use this one folder for all of them. The Air-IO configs (`airio/configs/datasets/Nanobench/nanobench_body.conf`) list the same 74 / 16 / 17 sequences inline; their `test` block is the 17-sequence reported split and is also what `train_motion.py` uses to pick `best_model.ckpt` (upstream AirIO behaviour, applies to AirIO and GINIO-A alike); the 16-sequence `eval` block is only logged |

## Cross-check between the two machines

Server A (AWS box) held `tlio_golden` and the two NanoBench copies; Server B (Alienware) held all
four datasets. The sequence **sets** are identical on both machines for every list that exists on
both (`sha256` of the sorted content). The only byte difference is the line *order* of
`tlio_golden/test_list.txt` (Server B's order is shipped here; the set is the same 36 sequences).
RIDI exists only on Server B.

## NanoBench caveat: gravity has been removed from the processed accelerometer

In `nanobench_processed_2nd_version` and both `bodyframe_*` derivatives the accelerometer channel
has gravity removed, although the accompanying `imu0_resampled_description.json` says the signal
includes gravity (Server B `results/nanobench_bias_2026/SUMMARY.md`). This does not affect the
NN-only Table 2 / Table 8 / Table 9 numbers (the networks were trained and tested on the same
processed data), but it makes any filter run on these files invalid: the TLIO/EqNIO trackers expect
gravity in the accelerometer and diverge (Server B's RIEKF converged on 1 of 17 sequences). Rebuild
the processed set with gravity if you want to connect a filter on NanoBench.

## Test-sequence lists at a glance

* TLIO test (36): the standard TLIO golden test split (sequence ids such as `145820422949970`).
* RIDI test (23): `dan_bag2 dan_body3 dan_leg2 hang_bag_speed2 hang_body_backward2 hang_body_test1
  hang_handheld_side_test2 hang_handheld_test1 huayi_bag_test1 huayi_body_test1 huayi_leg_front3
  ma_handheld2 tang_bag2 tang_body2 tang_handheld2 xiaojing_body2 xiaojing_handheld2 yajie_body2
  yajie_handheld2 zhicheng_bag2 zhicheng_body2 zhicheng_handheld2 zhicheng_leg2`.
* NanoBench test (17): `B2_circle_slow_rep1 B2_circle_slow_rep2 B2_circle_medium_rep2
  B3_figure8_fast_rep2 B5_helix_rep1 B5_helix_medium_rep1 B6_linear_ramp_rep2 B8_star_slow_rep2
  B9_trefoil_slow_rep3 B9_trefoil_slow_rep8 B9_trefoil_slow_rep16 B9_trefoil_medium_rep4
  B9_trefoil_medium_rep9 B9_trefoil_medium_rep13 B9_trefoil_fast_rep8 B10_lissajous_slow_rep1
  B11_random_waypoints_rep1`.

The ID/SO(3) TLIO evaluation of Table 1 additionally needs the rotated copy `so3_60` of the test
split (per-sequence random rotations, rotated ground truth); it is not a list but a data copy and is
not included here (see `checkpoints/MANIFEST.md`, trap 6).
