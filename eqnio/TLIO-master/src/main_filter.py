"""
TLIO Stochastic Cloning Extended Kalman Filter
Input: IMU data
Measurement: window displacement estimates from networks
Filter states: position, velocity, rotation, IMU biases
"""

import argparse
import datetime
import json
import os

# silence NumbaPerformanceWarning
import warnings
from pprint import pprint

import numpy as np
from numba.core.errors import NumbaPerformanceWarning
from tracker.imu_tracker_runner import ImuTrackerRunner
from utils.argparse_utils import add_bool_arg
from utils.logging import logging


warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ----------------------- io params -----------------------
    io_groups = parser.add_argument_group("io")

    io_groups.add_argument(
        "--root_dir", type=str, 
        default=None, help="Path to data directory"
    )
    io_groups.add_argument("--dataset_number", type=int, default=None)
    io_groups.add_argument("--model_path", type=str, default=None)
    io_groups.add_argument("--model_param_path", type=str, default=None, required=True)
    io_groups.add_argument("--out_dir", type=str, default=None)
    io_groups.add_argument("--out_filename", type=str, default="not_vio_state.txt")
    io_groups.add_argument("--save_as_npy", action="store_true")
    io_groups.add_argument("--sim_data_path", type=str, default="imu-sim.txt")
    io_groups.add_argument(
        "--start_from_ts", type=int, default=None
    )  # dataloader loading data from timestamp (us)

    add_bool_arg(io_groups, "visualize", default=False, 
            help="Opens up a visualization window")
    add_bool_arg(io_groups, "erase_old_log", default=False)

    # ----------------------- network params -----------------------
    net_groups = parser.add_argument_group("network")
    net_groups.add_argument("--cpu", action="store_true")
    # Whether the filter rotates IMU samples to local-yaw-aligned frame before feeding the network.
    # True (default, EqNIO original): for tlio_o2 which was trained with yaw normalization.
    # False: for chol-style models which were trained on raw world-frame IMU.
    add_bool_arg(net_groups, "yaw_aligned", default=True)

    # ----------------------- hybrid evaluation -----------------------
    hybrid_groups = parser.add_argument_group("hybrid")
    hybrid_groups.add_argument("--secondary_model_path", type=str, default=None,
                               help="If set, also load this TorchScript model and combine outputs in world frame")
    hybrid_groups.add_argument("--secondary_arch", type=str, default=None,
                               help="arch_type for secondary model (e.g. eq_o2_frame_fullCov_2vec_2deep or ln_resnet_chol)")
    hybrid_groups.add_argument("--hybrid_mode", type=str, default=None,
                               choices=[None, "mean_from_secondary", "cov_from_secondary"],
                               help="mean_from_secondary: B (mean=secondary, cov=primary). cov_from_secondary: A (mean=primary, cov=secondary).")

    # ----------------------- filter params -----------------------
    filter_group = parser.add_argument_group("filter tuning:")

    filter_group.add_argument("--update_freq", type=float, default=20.0)  # (Hz)

    filter_group.add_argument(
        "--sigma_na", type=float, default=np.sqrt(1e-3)
    )  # accel noise  m/s^2
    filter_group.add_argument(
        "--sigma_ng", type=float, default=np.sqrt(1e-4)
    )  # gyro noise  rad/s
    filter_group.add_argument(
        "--ita_ba", type=float, default=1e-4
    )  # accel bias noise  m/s^2/sqrt(s)
    filter_group.add_argument(
        "--ita_bg", type=float, default=1e-6
    )  # gyro bias noise  rad/s/sqrt(s)

    filter_group.add_argument(
        "--init_attitude_sigma", type=float, default=1.0 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument(
        "--init_yaw_sigma", type=float, default=0.1 / 180.0 * np.pi
    )  # rad
    filter_group.add_argument("--init_vel_sigma", type=float, default=1.0)  # m/s
    filter_group.add_argument("--init_pos_sigma", type=float, default=0.001)  # m
    filter_group.add_argument(
        "--init_bg_sigma", type=float, default=0.0001
    )  # rad/s  0.001
    filter_group.add_argument("--init_ba_sigma", type=float, default=0.02)  # m/s^2  0.02
    filter_group.add_argument("--g_norm", type=float, default=9.81)

    filter_group.add_argument("--meascov_scale", type=float, default=10.0) #10.0

    add_bool_arg(
        filter_group, "initialize_with_vio", default=True
    )  # initialize state with gt state
    add_bool_arg(
        filter_group, "initialize_with_offline_calib", default=True
    )  # initialize bias state with offline calib or 0

    filter_group.add_argument(
        "--mahalanobis_fail_scale", type=float, default=0
    )  # if nonzero then mahalanobis gating test would scale the covariance by this scale if failed

    # ----------------------- test-time perturbation -----------------------
    perturb_groups = parser.add_argument_group("test perturbation")
    perturb_groups.add_argument(
        "--test_perturb_type", type=str, default="none",
        choices=["none", "gravity", "lfa", "so3"],
        help="Inject perturbation into raw IMU stream before EKF/network see it. "
             "'gravity': fixed horizontal-axis rotation per sequence (sensor mis-mount). "
             "'lfa': smooth time-varying random rotation with peak |rvec|=test_perturb_deg. "
             "'so3': fixed random SO(3) rotation of magnitude test_perturb_deg per sequence."
    )
    perturb_groups.add_argument(
        "--test_perturb_deg", type=float, default=0.0,
        help="Perturbation magnitude in degrees (gravity: max uniform tilt; lfa: per-sequence peak)."
    )
    perturb_groups.add_argument(
        "--test_perturb_seed", type=int, default=0,
        help="RNG seed offset for reproducible perturbations across runs."
    )
    perturb_groups.add_argument(
        "--test_list", type=str, default="test_list.txt",
        help="Test list filename relative to root_dir."
    )
    perturb_groups.add_argument(
        "--so3_rot_path", type=str, default=None,
        help="Path pattern for per-seq R_so3 .npy. Use {seq} placeholder. "
             "When set, IMU CSV is assumed already rotated by R_so3; "
             "filter init R is right-multiplied by R_so3^T for equivariance test."
    )

    # ----------------------- debug params -----------------------
    debug_groups = parser.add_argument_group("debug")
    # covariance alternatives (note: if use_vio_meas is true, meas constant with default value 1e-4)
    add_bool_arg(debug_groups, "use_const_cov", default=False)
    debug_groups.add_argument(
        "--const_cov_val_x", type=float, default=np.power(0.1, 2.0)#0.1, 2.0
    )
    debug_groups.add_argument(
        "--const_cov_val_y", type=float, default=np.power(0.1, 2.0)#0.1, 2.0
    )
    debug_groups.add_argument(
        "--const_cov_val_z", type=float, default=np.power(0.1, 2.0)#0.1, 2.0
    )

    # measurement alternatives (note: if use_vio_meas is false, add_sim_meas_noise must be false)
    add_bool_arg(
        debug_groups,
        "use_vio_meas",
        default=False,
        help='If using "vio" measurement for filter update instead of ouptut network',
    )
    add_bool_arg(debug_groups, "debug_using_vio_ba", default=False)
    add_bool_arg(
        debug_groups, "add_sim_meas_noise", default=False
    )  # adding noise on displacement measurement when using vio measurement
    debug_groups.add_argument(
        "--sim_meas_cov_val", type=float, default=np.power(0.01, 2.0)
    )
    debug_groups.add_argument(
        "--sim_meas_cov_val_z", type=float, default=np.power(0.01, 2.0)
    )

    args = parser.parse_args()

    np.set_printoptions(linewidth=2000)

    logging.info("Program options:")
    logging.info(pprint(vars(args)))
    # run filter
    data_list = os.path.join(args.root_dir, args.test_list)
    with open(data_list) as f:
        data_names = [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    param_dict = vars(args)
    param_dict["date"] = str(datetime.datetime.now())
    with open(args.out_dir + "/parameters.json", "w") as parameters_file:
        parameters_file.write(json.dumps(param_dict, indent=4, sort_keys=True))

    def _maybe_patch_so3(runner, seq_name):
        """Equivariance protocol: when --so3_rot_path is set, right-multiply every
        VIO-derived rotation by R_so3^T at filter init. With raw IMU also rotated
        by R_so3 (gyr'=R g, acc'=R a), the world trajectory is invariant
        (R_W_B'(t)=R_W_B(t) @ R^T cancels with R applied to IMU)."""
        if not getattr(args, "so3_rot_path", None):
            return
        path = args.so3_rot_path.replace("{seq}", seq_name)
        R_so3 = np.load(path)
        assert R_so3.shape == (3, 3), f"R_so3 must be 3x3, got {R_so3.shape}"
        Rt = R_so3.T
        orig = runner.reset_filter_state_from_vio
        def patched(this):
            orig(this)
            st = this.filter.state
            for i in range(len(st.si_Rs)):
                st.si_Rs[i] = st.si_Rs[i] @ Rt
            st.s_R = st.s_R @ Rt
        runner.reset_filter_state_from_vio = patched
        logging.info(f"SO(3) equivariance test: applied R_so3^T from {path}; det={np.linalg.det(R_so3):.6f}")

    # load offline calibration for IMU
    if args.dataset_number is not None:
        logging.info("Running in one-shot mode")
        logging.info("Using dataset {}".format(data_names[args.dataset_number]))
        seq_name = data_names[args.dataset_number]
        trackerRunner = ImuTrackerRunner(args, seq_name)
        _maybe_patch_so3(trackerRunner, seq_name)
        trackerRunner.run_tracker(args, args.dataset_number)
    else:
        logging.info("Running in batch mode")
        # add metadata for logging
        n_data = len(data_names)
        for i, name in enumerate(data_names):
            logging.info(f"Processing {i} / {n_data} dataset {name}")
            try:
                trackerRunner = ImuTrackerRunner(args, name)
                trackerRunner.run_tracker(args, i)
            except FileExistsError as e:
                print(e)
                continue
            except OSError as e:
                print(e)
                continue
