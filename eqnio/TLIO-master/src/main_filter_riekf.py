"""
EqNIO main_filter with RIEKF + body-frame flags grafted in (ported from Server B).

Adds these flags to the EqNIO filter framework:
  --use_riekf {True,False}
  --body_frame {True,False}
  --input_3 {True,False}

The tracker module is sourced from the GINIO tree (this repository's top-level
src/, formerly imu-equivariant-learning), which has RIEKF + body_frame implemented
and validated. Run from anywhere, e.g.
  python eqnio/TLIO-master/src/main_filter_riekf.py --root_dir <data>/ --model_path <ts.pt> \
      --model_param_path <parameters.json> --out_dir <out> --use_riekf True --body_frame True

This is a minimal patch:
  - Argparse + main loop stays as in EqNIO's main_filter.py
  - sys.path is pre-pended with imu-eq/src so `from tracker.xxx import yyy`
    resolves to the imu-eq tracker module (RIEKF + body_frame capable)
  - We expose the same eval-string flags imu-eq's runner expects
"""

import argparse
import datetime
import json
import os
import sys
import warnings
from pprint import pprint

import numpy as np

# Resolve the GINIO tracker source (RIEKF + body_frame capable). In this repository the
# former imu-equivariant-learning tree is the top-level src/ (eqnio/TLIO-master/src -> ../../../src).
# Override with the GINIO_SRC environment variable if the layout differs.
IMU_EQ_SRC = os.environ.get("GINIO_SRC") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../src",
))
assert os.path.isdir(os.path.join(IMU_EQ_SRC, "tracker")), f"GINIO src/tracker not found at {IMU_EQ_SRC}"
sys.path.insert(0, IMU_EQ_SRC)

from numba.core.errors import NumbaPerformanceWarning
from tracker.imu_tracker_runner import ImuTrackerRunner
from utils.argparse_utils import add_bool_arg
from utils.logging import logging

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    io_groups = parser.add_argument_group("io")
    io_groups.add_argument("--root_dir", type=str, default=None, required=True)
    io_groups.add_argument("--dataset_number", type=int, default=None)
    io_groups.add_argument("--model_path", type=str, default=None)
    io_groups.add_argument("--model_param_path", type=str, default=None, required=True)
    io_groups.add_argument("--out_dir", type=str, default=None, required=True)
    io_groups.add_argument("--out_filename", type=str, default="not_vio_state.txt")
    io_groups.add_argument("--save_as_npy", action="store_true")
    io_groups.add_argument("--sim_data_path", type=str, default="imu-sim.txt")
    io_groups.add_argument("--start_from_ts", type=int, default=None)
    add_bool_arg(io_groups, "visualize", default=False)
    add_bool_arg(io_groups, "erase_old_log", default=False)

    net_groups = parser.add_argument_group("network")
    net_groups.add_argument("--cpu", action="store_true")

    # RIEKF / body-frame / input dim flags (eval-string, imu-eq style)
    filter_group = parser.add_argument_group("filter tuning:")
    filter_group.add_argument("--body_frame", type=str, default="False")
    filter_group.add_argument("--use_riekf", type=str, default="False")
    filter_group.add_argument("--input_3", type=str, default="False")
    filter_group.add_argument("--run_data", type=str, default=None)
    filter_group.add_argument("--use_const_cov", type=str, default="False")

    filter_group.add_argument("--update_freq", type=float, default=20.0)
    filter_group.add_argument("--sigma_na", type=float, default=np.sqrt(1e-3))
    filter_group.add_argument("--sigma_ng", type=float, default=np.sqrt(1e-4))
    filter_group.add_argument("--ita_ba", type=float, default=1e-4)
    filter_group.add_argument("--ita_bg", type=float, default=1e-6)
    filter_group.add_argument("--init_attitude_sigma", type=float, default=1.0 / 180.0 * np.pi)
    filter_group.add_argument("--init_yaw_sigma", type=float, default=0.1 / 180.0 * np.pi)
    filter_group.add_argument("--init_vel_sigma", type=float, default=1.0)
    filter_group.add_argument("--init_pos_sigma", type=float, default=0.001)
    filter_group.add_argument("--init_bg_sigma", type=float, default=0.0001)
    filter_group.add_argument("--init_ba_sigma", type=float, default=0.02)
    filter_group.add_argument("--g_norm", type=float, default=9.81)
    filter_group.add_argument("--meascov_scale", type=float, default=10.0)
    filter_group.add_argument("--initialize_with_vio", type=str, default="True")
    filter_group.add_argument("--initialize_with_offline_calib", type=str, default="False")
    filter_group.add_argument("--mahalanobis_fail_scale", type=float, default=0)
    filter_group.add_argument(
        "--so3_rot_path",
        type=str,
        default=None,
        help="Path pattern for per-seq R_so3 .npy. Use {seq} placeholder. "
             "When set, IMU CSV is assumed already rotated by R_so3; "
             "filter init R is right-multiplied by R_so3^T for equivariance test.",
    )

    debug_groups = parser.add_argument_group("debug")
    debug_groups.add_argument("--const_cov_val_x", type=float, default=np.power(0.1, 2.0))
    debug_groups.add_argument("--const_cov_val_y", type=float, default=np.power(0.1, 2.0))
    debug_groups.add_argument("--const_cov_val_z", type=float, default=np.power(0.1, 2.0))
    add_bool_arg(debug_groups, "use_vio_meas", default=False)
    add_bool_arg(debug_groups, "debug_using_vio_ba", default=False)
    add_bool_arg(debug_groups, "add_sim_meas_noise", default=False)
    debug_groups.add_argument("--sim_meas_cov_val", type=float, default=np.power(0.01, 2.0))
    debug_groups.add_argument("--sim_meas_cov_val_z", type=float, default=np.power(0.01, 2.0))

    args = parser.parse_args()
    np.set_printoptions(linewidth=2000)
    logging.info("Program options (EqNIO + RIEKF):")
    logging.info(pprint(vars(args)))

    # build test list
    test_list_name = "test_list.txt"
    if args.run_data and args.run_data != "0":
        test_list_name = f"test_list{args.run_data}.txt"
    data_list = os.path.join(args.root_dir, test_list_name)
    with open(data_list) as f:
        data_names = [
            s.strip().split(",")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]

    os.makedirs(args.out_dir, exist_ok=True)
    param_dict = vars(args)
    param_dict["date"] = str(datetime.datetime.now())
    param_dict["tracker_source"] = IMU_EQ_SRC
    with open(args.out_dir + "/parameters.json", "w") as fp:
        fp.write(json.dumps(param_dict, indent=4, sort_keys=True))

    def _maybe_patch_so3(runner, seq_name):
        """If --so3_rot_path supplied, wrap reset_filter_state_from_vio so that
        after VIO-derived Rs are loaded, every R is right-multiplied by R_so3^T.
        Math: with IMU body-frame rotated by R_so3 (gyr'=R_so3 g, acc'=R_so3 a),
        R_W_B'_init = R_W_B_init @ R_so3^T preserves the world trajectory.
        """
        if not args.so3_rot_path:
            return
        path = args.so3_rot_path.replace("{seq}", seq_name)
        R_so3 = np.load(path)
        assert R_so3.shape == (3, 3), f"R_so3 must be 3x3, got {R_so3.shape}"
        Rt = R_so3.T
        orig = runner.reset_filter_state_from_vio
        def patched(this, input_3=False, use_riekf=False):
            orig(this, input_3, use_riekf)
            st = this.filter.state
            for i in range(len(st.si_Rs)):
                st.si_Rs[i] = st.si_Rs[i] @ Rt
            st.s_R = st.s_R @ Rt
        runner.reset_filter_state_from_vio = patched
        logging.info(f"SO(3) test: applied R_so3^T from {path} to init R; det={np.linalg.det(R_so3):.6f}")

    if args.dataset_number is not None:
        seq_name = data_names[args.dataset_number]
        logging.info(f"One-shot: dataset={seq_name}")
        runner = ImuTrackerRunner(args, seq_name)
        _maybe_patch_so3(runner, seq_name)
        runner.run_tracker(args, args.dataset_number)
    else:
        n_data = len(data_names)
        for i, name in enumerate(data_names):
            logging.info(f"Processing {i+1}/{n_data} dataset {name}")
            try:
                runner = ImuTrackerRunner(args, name)
                runner.run_tracker(args, i)
            except (FileExistsError, OSError) as e:
                print(e)
                continue
