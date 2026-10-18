"""
Nanobench dataset for AirIO.

Reads existing world-frame nanobench imu0_resampled.npy and derives
body-frame raw IMU for AirIO consumption.

Source npy columns (17):
  ts_us(1), gyr_in_World(3), acc_in_World(3), qxyzw_W_D(4), pos_W_D(3), vel_World(3)

Note: source acc has gravity already removed (verified empirically).
"""
import os
import numpy as np
import pypose as pp
import torch
import copy
import json
from scipy.spatial.transform import Rotation
from .dataset import Sequence


class Nanobench(Sequence):
    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.80665,
        remove_g=False,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.data = dict()
        self.g_vector = torch.tensor([0, 0, gravity], dtype=torch.double)

        # Load source npy + description
        seq_dir = os.path.join(data_root, data_name)
        arr = np.load(os.path.join(seq_dir, "imu0_resampled.npy"))
        assert arr.shape[1] == 17, f"unexpected shape {arr.shape}"

        ts_us = arr[:, 0]
        gyr_W = arr[:, 1:4]
        acc_W = arr[:, 4:7]
        quat_xyzw = arr[:, 7:11]  # (qx, qy, qz, qw) world<-body
        pos_W = arr[:, 11:14]
        vel_W = arr[:, 14:17]   # NOTE: this is world-frame in the original npy

        # Derive body-frame IMU via R^T (world<-body, so R^T maps world to body)
        R = Rotation.from_quat(quat_xyzw).as_matrix()  # (N, 3, 3) world<-body
        RT = np.transpose(R, (0, 2, 1))
        gyr_B = np.einsum("nij,nj->ni", RT, gyr_W)
        acc_B = np.einsum("nij,nj->ni", RT, acc_W)

        # Convert ts_us to seconds for AirIO convention
        ts_s = ts_us / 1e6

        # Populate self.data
        self.data["gyro"] = torch.tensor(gyr_B, dtype=torch.float64)
        self.data["acc"] = torch.tensor(acc_B, dtype=torch.float64)
        self.data["gt_translation"] = torch.tensor(pos_W, dtype=torch.float64)
        self.data["gt_orientation"] = pp.SO3(torch.tensor(quat_xyzw, dtype=torch.float64))
        self.data["velocity"] = torch.tensor(vel_W, dtype=torch.float64)  # WORLD frame; update_coordinate will rotate to body if needed
        self.data["time"] = torch.tensor(ts_s, dtype=torch.float64)
        self.data["gt_time"] = torch.tensor(ts_s, dtype=torch.float64)
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

        # AirIO pipeline
        self.set_orientation(rot_path, data_name, rot_type)
        self.update_coordinate(coordinate, mode)
        self.remove_gravity(remove_g)

    def get_length(self):
        return self.data["time"].shape[0]

    def update_coordinate(self, coordinate, mode):
        if coordinate is None:
            return
        if coordinate == "glob_coord":
            self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
            self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            # velocity already in world (no change)
        elif coordinate == "body_coord":
            self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
            if mode != "infevaluate" and mode != "inference":
                self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate}")

    def set_orientation(self, exp_path, data_name, rotation_type):
        if rotation_type is None or rotation_type == "None" or str(rotation_type).lower() == "gtrot":
            return
        # AirIMU / integration variants not supported for nanobench yet
        raise NotImplementedError(
            f"Nanobench currently only supports GT orientation; got rotation_type={rotation_type}"
        )

    def remove_gravity(self, remove_g):
        # Source npy already has gravity removed from world acc → body acc derived also gravity-free.
        # If remove_g=True, do nothing (already done). If False, do nothing too (body acc has no gravity).
        # This keeps behavior consistent with the data we have.
        if remove_g:
            # caller wanted gravity removed: already done at preprocessing, no-op
            pass
