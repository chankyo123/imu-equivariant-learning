"""
IMU network training/testing/evaluation for displacement and covariance
Input: Nx6 IMU data
Output: 3x1 displacement, 3x1 covariance parameters
"""

import network
from utils.argparse_utils import add_bool_arg
import numpy as np

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
        default="local_data/tlio_golden", help="Path to data directory"
    )
    parser.add_argument("--out_dir", type=str, default="outputs/resnet_seq")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)

    # ------------------ architecture and training -----------------
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10000, help="max num epochs")
    # parser.add_argument("--arch", type=str, default="resnet")
    parser.add_argument("--arch", type=str, default="vn_resnet")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--body_frame", type=str, default="True")
    # add_bool_arg(parser, "body_frame", default=True)
    parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument("-j", "--workers", type=int, default=4)
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
    parser.add_argument("--accel_bias_range", type=float, default=0.2)  # 5e-2
    parser.add_argument("--gyro_bias_range", type=float, default=0.05)  # 1e-3

    add_bool_arg(parser, "perturb_gravity", default=True)
    parser.add_argument(
        "--perturb_gravity_theta_range", type=float, default=5.0
    )  # degrees

    # ----- window size and inference freq -----
    parser.add_argument("--past_time", type=float, default=0.0)  # s
    parser.add_argument("--window_time", type=float, default=1.0)  # s
    parser.add_argument("--future_time", type=float, default=0.0)  # s

    # ----- for sampling in training / stepping in testing -----
    parser.add_argument("--sample_freq", type=float, default=20.0)  # hz

    # ===================== loss / covariance options =====================
    parser.add_argument(
        "--cov_type", type=str, default="auto",
        choices=["auto", "diag", "eq_spec", "eq_chol", "eq_chol_svd"],
        help="Covariance head type. 'auto' picks from arch/out_dir name. "
             "'eq_chol_svd' = SVD/polar-decomposed Sigma with non-degeneracy "
             "regulariser (preferred over plain 'eq_chol')."
    )
    parser.add_argument("--mse_warmup_epochs", type=int, default=10,
                        help="Epochs of pure MSE before NLL turns on.")
    parser.add_argument("--mse_anchor_weight", type=float, default=0.1,
                        help="MSE anchor weight added to NLL after warmup (P4 hybrid).")
    parser.add_argument("--nll_weight", type=float, default=1.0,
                        help="Weight on NLL term after warmup.")
    parser.add_argument("--eq_chol_eps", type=float, default=1e-3,
                        help="Sigma = A A^T + eps*I; small floor for SPD-ness.")

    # ===================== reproducibility =====================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=0.1)

    # ===================== data augmentation (지금은 dead link인 걸 살림) =====================
    parser.add_argument("--accel_noise_std", type=float, default=0.0,
                        help="Gaussian noise std added to accel during training.")
    parser.add_argument("--gyro_noise_std", type=float, default=0.0,
                        help="Gaussian noise std added to gyro during training.")
    parser.add_argument("--bias_scale", type=float, default=1.0,
                        help="Multiplier on (accel/gyro)_bias_range. <1.0 = weaker bias aug.")
    parser.add_argument("--bias_curriculum_epochs", type=int, default=0,
                        help="If >0, linearly ramp bias_scale from 0 to 1 over this many epochs.")
    parser.add_argument("--init_log_a_scale", type=float, default=np.log(0.15),
                    help="Initial value of log_a_scale parameter in Cholesky head.")

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "save_plot", default=True)
    parser.add_argument("--rpe_window", type=float, default="2.0")  # s

    args = parser.parse_args()

    import numpy as np, torch, random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
