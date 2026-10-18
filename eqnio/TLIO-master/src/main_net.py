"""
IMU network training/testing/evaluation for displacement and covariance
Input: Nx6 IMU data
Output: 3x1 displacement, 3x1 covariance parameters
"""

import network
from utils.argparse_utils import add_bool_arg
import os
# os.add_dll_directory(os.getcwd())
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # ------------------ directories -----------------
    # NOTE now they are assumed to be under root_dir with new format
    #parser.add_argument("--train_list", type=str, default=None)
    #parser.add_argument("--val_list", type=str, default=None)
    #parser.add_argument("--test_list", type=str, default=None) 
    parser.add_argument(
        "--root_dir", type=str, 
        default="../TLIO-master/local_data/tlio_golden", help="Path to data directory"
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument(
        "--test_list", type=str, 
        default="test_list.txt", help="Path to test list"
    )

    # ------------------ architecture and training -----------------
    parser.add_argument("--lr", type=float, default=1e-04) 
    parser.add_argument("--batch_size", type=int, default=1024) 
    parser.add_argument("--epochs", type=int, default=50, help="max num epochs")

    parser.add_argument("--arch", type=str, default="eq_o2_frame_fullCov_2vec_2deep") #, resnet
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument("-j", "--workers", type=int, default=10)
    parser.add_argument("--dataset_style", type=str, default="mmap", 
            help="'ram', 'mmap', or 'iter'. See dataloader/tlio_data.py for more details")
    add_bool_arg(parser, "persistent_workers", default=True)

    # ------------------ commons -----------------
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument(
        "--imu_freq", type=float, default=200.0, help="imu_base_freq is a multiple"
    )
    parser.add_argument("--imu_base_freq", type=float, default=1000.0)

    # ----- perturbation -----
    add_bool_arg(parser, "do_bias_shift", default=True)
    parser.add_argument("--accel_bias_range", type=float, default=0.2)  # 5e-2 0.2
    parser.add_argument("--gyro_bias_range", type=float, default=0.05)  # 1e-3 0.05

    add_bool_arg(parser, "perturb_gravity", default=True)
    parser.add_argument(
        "--perturb_gravity_theta_range", type=float, default=5.0
    )  # degrees 5.0
    add_bool_arg(parser, "yaw_augmentation", default=False)
    add_bool_arg(parser,"dataloader_bias_gravity_aug", default=False) ## to test sensitivity
    add_bool_arg(parser, "express_in_local_gravity_aligned", default=True) ## set False when testing imu-eq-learning models (e.g. ln_resnet_chol)
    # SO(3) data augmentation: random rotation R in SO(3) applied per training window
    add_bool_arg(parser, "so3_augment", default=False)
    parser.add_argument("--so3_augment_max_deg", type=float, default=180.0,
                        help="max axis-angle magnitude in degrees for SO(3) data augmentation (180 ~= uniform on SO(3))")
    # Test-time LFA-style perturbation (time-varying relative rotation, anchored at last frame)
    parser.add_argument("--lfa_perturb_max_deg", type=float, default=0.0,
                        help="if > 0, apply per-sample LFA relative-rotation noise with this peak magnitude (deg) at TEST time")
    parser.add_argument("--lfa_perturb_mode", type=str, default="smooth", choices=["smooth", "ramp"],
                        help="LFA noise temporal pattern: 'smooth' random walk or 'ramp' linear decay")
    parser.add_argument("--so3_test_deg", type=float, default=0.0,
                        help="If > 0, apply test-time fixed-magnitude SO(3) rotation (deg) per window (IMU only; targets not rotated)")

    # ----- window size and inference freq -----
    parser.add_argument("--past_time", type=float, default=0.0)  # s
    parser.add_argument("--window_time", type=float, default=1)  # s
    parser.add_argument("--future_time", type=float, default=0.0)  # s

    # ----- for sampling in training / stepping in testing -----
    parser.add_argument("--sample_freq", type=float, default=20.0)  # hz default=20.0 -they claim to use this in final system

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "save_plot", default=True)
    parser.add_argument("--rpe_window", type=float, default="1.0")  # s default="2.0" ## this is only for plotting

    args = parser.parse_args()

    ###########################################################
    # Main
    ###########################################################
    if args.mode == "train":
        network.net_train(args)
    elif args.mode == "test":
        network.net_test(args)
    elif args.mode == "eval":
        network.net_eval(args)
    else:
        raise ValueError("Undefined mode")
