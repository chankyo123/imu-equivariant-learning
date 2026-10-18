"""
Patched torchscript converter that handles EqNIO O(2) and VN-EqNIO SO(3) models.

The original convert_model_to_torchscript.py assumes a single-input model (TLIO/RONIN
style). EqNIO O(2) takes (vector, scalar, orig_sca) and SO(3) takes (vector, scalar).
This script picks the right input shapes per arch and traces correctly.
"""
import argparse
import json
import os
from pprint import pprint

import numpy as np
import torch
from network.model_factory import get_model
from utils.logging import logging


def make_dummy_inputs(arch, N):
    """Return tuple of dummy tensors matching the forward signature."""
    if "so3_frame" in arch:
        # Eq_Motion_Model_SO3_fullCov.forward(vector, scalar)
        vector = torch.zeros(1, N, 3, 3)   # (B, N, 3 vectors, 3 dims)
        scalar = torch.zeros(1, N, 6)       # (B, N, 6 invariants)
        return (vector, scalar)
    elif "o2_frame" in arch:
        # Eq_Motion_Model_fullCov.forward(vector, scalar, original_scalars)
        # From preprocess_o2: vector (1, N, 2, 3) via permute, scalar (1, N, 9), orig (1, N, 3)
        vector = torch.zeros(1, N, 2, 3)
        scalar = torch.zeros(1, N, 9)
        orig   = torch.zeros(1, N, 3)
        return (vector, scalar, orig)
    elif "resnet_2scalars_tlio_frame" in arch or "resnet_fullCov_tlio_frame" in arch:
        # Single tensor input (B, 6, N)
        return (torch.zeros(1, 6, N),)
    else:
        # Default: single tensor (B, 6, N)
        return (torch.zeros(1, 6, N),)


def load_and_convert(args):
    past_data_size = int(args.past_time * args.imu_freq_net)
    disp_window_size = int(args.window_time * args.imu_freq_net)
    N = past_data_size + disp_window_size

    net_config = {"in_dim": N // 32 + 1}
    net = get_model(args.arch, net_config, 6, 3)

    # load trained weights
    device = torch.device("cpu") if (not torch.cuda.is_available() or args.cpu) else torch.device("cuda:0")
    if device.type == "cpu":
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path)

    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval().to(device)

    dummies = make_dummy_inputs(args.arch, N)
    dummies = tuple(t.to(device) for t in dummies)

    logging.info(f"Tracing arch={args.arch} with input shapes: {[tuple(t.shape) for t in dummies]}")
    traced = torch.jit.trace(net, dummies, strict=False)
    out_path = os.path.join(args.out_dir, "model_torchscript.pt")
    traced.save(out_path)
    logging.info(f"Saved torchscript to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--model_param_path", required=True, type=str)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.model_param_path) as f:
        data_json = json.load(f)
        args.imu_freq_net = data_json["imu_freq"]
        args.past_time = data_json["past_time"]
        args.window_time = data_json["window_time"]
        args.arch = data_json["arch"]

    np.set_printoptions(linewidth=2000)
    logging.info("Program options:")
    logging.info(pprint(vars(args)))
    os.makedirs(args.out_dir, exist_ok=True)
    load_and_convert(args)
