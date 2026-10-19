"""
TLIO golden dataset for AirIO.

Reads TLIO-golden world-frame imu0_resampled.npy (same 17-column layout as Nanobench)
and derives body-frame raw IMU for AirIO consumption.

Source npy columns (17):
  ts_us(1), gyr_in_World(3), acc_in_World(3), qxyzw_W_D(4), pos_W_D(3), vel_World(3)

KEY DIFFERENCE from Nanobench: TLIO golden has acc WITH gravity (z mean ~9.8 m/s²),
while Nanobench has acc with gravity already removed (z mean ~0). So this class
implements `remove_gravity` properly (Nanobench's is a no-op).
"""
import os
import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation
from .dataset import Sequence


class TLIO(Sequence):
    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.80665,
        remove_g=True,    # default True for TLIO (data has gravity, model wants gravity-free body acc)
        **kwargs,
    ):
        super().__init__()
        data_name = str(data_name)   # TLIO sequence IDs are numeric strings; HOCON parses as int
        self.data_root = data_root
        self.data_name = data_name
        self.data = dict()
        self.g_vector = torch.tensor([0, 0, gravity], dtype=torch.double)

        seq_dir = os.path.join(data_root, data_name)
        arr = np.load(os.path.join(seq_dir, "imu0_resampled.npy"))
        assert arr.shape[1] == 17, f"unexpected shape {arr.shape}"

        ts_us = arr[:, 0]
        gyr_W = arr[:, 1:4]
        acc_W = arr[:, 4:7]              # includes gravity (z ~ 9.8)
        quat_xyzw = arr[:, 7:11]
        pos_W = arr[:, 11:14]
        vel_W = arr[:, 14:17]

        # World→body rotation: R is world<-body, so R^T maps world to body
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        RT = np.transpose(R, (0, 2, 1))
        gyr_B = np.einsum("nij,nj->ni", RT, gyr_W)
        acc_B = np.einsum("nij,nj->ni", RT, acc_W)   # body acc, still with gravity-in-body

        ts_s = ts_us / 1e6

        self.data["gyro"] = torch.tensor(gyr_B, dtype=torch.float64)
        self.data["acc"] = torch.tensor(acc_B, dtype=torch.float64)
        self.data["gt_translation"] = torch.tensor(pos_W, dtype=torch.float64)
        self.data["gt_orientation"] = pp.SO3(torch.tensor(quat_xyzw, dtype=torch.float64))
        self.data["velocity"] = torch.tensor(vel_W, dtype=torch.float64)
        self.data["time"] = torch.tensor(ts_s, dtype=torch.float64)
        self.data["gt_time"] = torch.tensor(ts_s, dtype=torch.float64)
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

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
        elif coordinate == "body_coord":
            # Rotate world gravity to body frame, per timestep. self.g_vector becomes (N, 3).
            self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
            if mode != "infevaluate" and mode != "inference":
                self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate}")

    def set_orientation(self, exp_path, data_name, rotation_type):
        if rotation_type is None or rotation_type == "None" or str(rotation_type).lower() == "gtrot":
            return
        raise NotImplementedError(
            f"TLIO currently only supports GT orientation; got rotation_type={rotation_type}"
        )

    def remove_gravity(self, remove_g):
        """Subtract gravity from body acc. g_vector is (N, 3) in body frame after update_coordinate."""
        if remove_g:
            self.data["acc"] = self.data["acc"] - self.g_vector
