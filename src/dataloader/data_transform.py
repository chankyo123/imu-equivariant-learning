"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import math
import torch
#from pytorch3d.transforms import so3_exponential_map
from utils.torch_math_utils import so3_exp_map

# TODO augs for mag and barom
class TransformAddNoiseBias:
    def __init__(self, input_sensors, 
            accel_noise_std, gyro_noise_std, accel_bias_range, gyro_bias_range, 
            mag_bias_range, barom_press_bias_range, barom_temp_bias_range):
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_range = gyro_bias_range
        self.accel_bias_range = accel_bias_range
        self.mag_bias_range = mag_bias_range
        self.barom_press_bias_range = barom_press_bias_range
        self.barom_temp_bias_range = barom_temp_bias_range
        self.input_sensors = input_sensors
        
    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]
            
        for sensor, feat in feats_new.items():
            N, _InDim, T = feat.shape
            feat_aug = feat.clone()
            if "imu" in sensor:
                # assert feat.shape[1] == 6
                if feat.shape[1] == 6:
                    None
                    # shift in the accel and gyro bias terms
                    feat_aug[:, :3, :] += (
                        (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                        * self.gyro_bias_range / 0.5
                        # * self.gyro_bias_range / 0.5 * 0.5
                        # * self.gyro_bias_range / 0.5 * 0.1
                        # * 0
                        # * self.gyro_bias_range * 4
                    )
                    feat_aug[:, 3:6, :] += (
                        (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                        * self.accel_bias_range / 0.5
                        # * self.accel_bias_range / 0.5 * 0.5
                        # * self.accel_bias_range / 0.5 * 0.1
                        # * 0
                        # * self.accel_bias_range * 4
                    )

                    # add gaussian noise
                    feat_aug[:, :3, :] += (
                        torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std * 0
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std * 2
                    )
                    feat_aug[:, 3:6, :] += (
                        torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 0
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 2
                    )
                elif feat.shape[1] == 9 or feat.shape[1] == 18:
                    # shift in the accel and gyro bias terms, also add bias in vel_body
                    # print('here')
                    
                    feat_aug[:, :3, :] += (
                        (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                        * self.gyro_bias_range / 0.5
                        # * self.gyro_bias_range * 4
                        # * self.gyro_bias_range * 8
                    )
                    feat_aug[:, 3:6, :] += (
                        (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                        * self.accel_bias_range / 0.5
                        # * self.accel_bias_range * 4
                        # * self.accel_bias_range * 8
                        
                        # * self.accel_bias_range / 0.1
                        # * 0.5 * self.accel_bias_range / 0.1
                        
                    )
                    feat_aug[:, 6:9, :] += (
                        (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                        # * 0.02 / 0.5
                        # * 0.02 * 4
                        # * 0.02 * 6
                        * 0.02 * 5
                        
                        # * 2 * self.accel_bias_range / 0.1
                        # * 4 * self.accel_bias_range / 0.1
                        
                        # * self.accel_bias_range / 0.1
                        # * 2 * self.accel_bias_range / 0.1
                        # * 0.5 * self.accel_bias_range / 0.1
                        # * 3 * self.accel_bias_range / 0.1
                    )
                    # if feat.shape[1] == 18:
                    #     feat_aug[:, 9:12, :] += (
                    #         (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    #         * self.gyro_bias_range / 0.5
                    #         # * 0.5 * self.gyro_bias_range / 0.5
                    #     )
                    #     feat_aug[:, 12:15, :] += (
                    #         (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    #         * self.gyro_bias_range / 0.5
                    #         # * 0.5 * self.gyro_bias_range / 0.5
                    #     )
                    #     feat_aug[:, 15:18, :] += (
                    #         (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    #         * self.gyro_bias_range / 0.5
                    #         # * 0.5 * self.gyro_bias_range / 0.5
                    #     )
                    # add gaussian noise
                    feat_aug[:, :3, :] += (
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std
                        torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std * 4
                    )
                    feat_aug[:, 3:6, :] += (
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std
                        torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 4
                    )
                    feat_aug[:, 6:9, :] += (
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 2
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 4
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 6
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 10
                        
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * 0.005 * 10
                        torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * 0
                        
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 2
                        # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 0.5
                    )
                    # if feat.shape[1] == 18:
                    #     feat_aug[:, 9:12, :] += (
                    #         torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std
                    #         # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.gyro_noise_std * 0.5
                    #     )
                    #     feat_aug[:, 12:15, :] += (
                    #         torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std
                    #         # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 0.5
                    #     )
                    #     feat_aug[:, 15:18, :] += (
                    #         torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std
                    #         # torch.randn(N, 3, T, device=feat.device, dtype=feat.dtype) * self.accel_noise_std * 0.5
                    #     )
                    
                else:
                    assert feat.shape[1] == 6 or feat.shape[1] == 9

            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] += (
                    (torch.rand(N, 3, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.mag_bias_range / 0.5
                )
                
                # TODO gaussian noise? 

            elif "barom" in sensor:
                assert feat.shape[1] == 2
                feat_aug[:, 0:1, :] += (
                    (torch.rand(N, 1, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.barom_press_bias_range / 0.5
                )
                feat_aug[:, 1:2, :] += (
                    (torch.rand(N, 1, 1, device=feat.device, dtype=feat.dtype) - 0.5)
                    * self.barom_temp_bias_range / 0.5
                )

            else:
                assert False
            
            feats_new[sensor] = feat_aug
        
        sample_new = {k: v for k,v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        return sample_new

class TransformPerturbGravity:
    def __init__(self,  input_sensors, theta_range_deg):
        self.theta_range_deg = theta_range_deg
        self.input_sensors = input_sensors

    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]

        # get rotation vector of random horizontal direction
        angle_rand_rad = (
            torch.rand(sample["ts_us"].shape[0], device=sample["ts_us"].device, dtype=torch.float32) * math.pi * 2
        )
        theta_rand_rad = (
            torch.rand(sample["ts_us"].shape[0], device=sample["ts_us"].device, dtype=torch.float32)
            * math.pi
            * self.theta_range_deg
            / 180.0
        )
        c = torch.cos(angle_rand_rad)
        s = torch.sin(angle_rand_rad)
        zeros = torch.zeros_like(angle_rand_rad)
        vec_rand = torch.stack([c, s, zeros], dim=1)
        rvec = theta_rand_rad[:, None] * vec_rand  # N x 3
        R_mat = so3_exp_map(rvec)  # N x 3 x 3

        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                # assert feat.shape[1] == 6
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
            elif "barom" in sensor:
                # Nothing to do here
                pass
            else:
                assert False
            feats_new[sensor] = feat_aug
        
        sample_new = {k: v for k,v in sample.items() if k != "feats"}

        sample_new["feats"] = feats_new
        return sample_new


class TransformInYawPlane:
    """this transform object:
        - rotate imu data in horizontal plane with a random planar rotation
        - rotate the target the same way
    this brings invariance in the data to planar rotation
    this can also prevent the network to learn cues specific to the IMU placement
    """

    def __init__(self, input_sensors, angle_half_range_rad=math.pi):
        """
        Random yaw angles will be in [-angle_half_range, angle_half_range]
        """
        self.input_sensors = input_sensors
        self.angle_half_range_rad = angle_half_range_rad

    def __call__(self, sample):
        # Cloning of the tensors is done in the loop
        feats_new = {k: v for k,v in sample["feats"].items()} # Dict of sensor_name: [ts_normalized, sensor_data]
        # rotate in the yaw plane
        N = sample["ts_us"].shape[0]
        rand_unif = 2*torch.rand((N), device=sample["ts_us"].device, dtype=torch.float32) - 1 # in [-1,1]
        angle_rad = rand_unif * self.angle_half_range_rad
        c = torch.cos(angle_rad)  # N
        s = torch.sin(angle_rad)  # N
        ones = torch.ones_like(c)  # N
        zeros = torch.zeros_like(s)  # N
        R_newWorld_from_oldWorld_flat = torch.stack(
            (c, -s, zeros, s, c, zeros, zeros, zeros, ones), dim=1
        )  # N x 9
        R_newWorld_from_oldWorld = R_newWorld_from_oldWorld_flat.reshape((N, 3, 3))
        
        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                assert feat.shape[1] == 6
                feat_aug[:, :3, :] = torch.einsum(
                    "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                )
                feat_aug[:, 3:6, :] = torch.einsum(
                    "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, 3:6, :]
                )
            elif "mag" in sensor:
                assert feat.shape[1] == 3
                feat_aug[:, :3, :] = torch.einsum(
                    "nik,nkt->nit", R_newWorld_from_oldWorld, feat[:, :3, :]
                )
            elif "barom" in sensor:
                # Nothing to do here
                pass
            else:
                assert False

            feats_new[sensor] = feat_aug

        sample_new = {
            k: v for k,v in sample.items() 
            if k.split("second_")[-1].split("_same")[0] not in ["feats","targ_dt_World","vel_World"]
        }
        sample_new["feats"] = feats_new
        # Handle the target data. Only displacement and vel need rotating, not relative rotation (already relative).
        # for k in "targ_dt_World", "vel_World":
        for k in "targ_dt_World", "vel_World", "targ_dt_Body":
            sample_new[k] = torch.einsum("nik,ntk->nti", R_newWorld_from_oldWorld, sample[k])

        return sample_new

class TransformLFARelativeRotationNoise:
    """
    Test-time perturbation for LFA-aligned inputs.

    Assumes sample["feats"][sensor] is already expressed in the last-frame-aligned
    frame S_n. We inject a time-varying relative rotation error:
        z_k^lfa -> DeltaR_k z_k^lfa
    with DeltaR_T = I, so the last timestep remains the anchor frame.

    This models imperfect relative rotations used inside LFA, not physical sensor
    misalignment. Therefore targets are NOT rotated.
    """
    def __init__(self, input_sensors, max_deg=3.0, mode="smooth"):
        self.input_sensors = input_sensors
        print("max deg :", max_deg)
        self.max_rad = max_deg * math.pi / 180.0
        self.mode = mode

    def _sample_rvec_smooth(self, N, T, device, dtype):
        """
        Generate smooth time-varying rotation vectors with rvec[:, -1] = 0.
        Each window is rescaled so its maximum perturbation magnitude equals max_deg.
        """
        if T == 1:
            return torch.zeros(N, 1, 3, device=device, dtype=dtype)

        inc = torch.randn(N, T - 1, 3, device=device, dtype=dtype)
        rvec = torch.zeros(N, T, 3, device=device, dtype=dtype)

        for t in range(T - 2, -1, -1):
            rvec[:, t, :] = rvec[:, t + 1, :] + inc[:, t, :]

        max_norm = rvec.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-8)
        rvec = rvec * (self.max_rad / max_norm).unsqueeze(-1)
        rvec[:, -1, :] = 0.0
        return rvec

    def _sample_rvec_ramp(self, N, T, device, dtype):
        """
        One random axis per window, linearly decaying to zero at the last frame.
        """
        axis = torch.randn(N, 3, device=device, dtype=dtype)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        alpha = torch.linspace(1.0, 0.0, T, device=device, dtype=dtype).view(1, T, 1)
        rvec = axis[:, None, :] * self.max_rad * alpha
        rvec[:, -1, :] = 0.0
        return rvec

    def __call__(self, sample):
        feats_new = {k: v.clone() for k, v in sample["feats"].items()}

        for sensor, feat in feats_new.items():
            if sensor not in self.input_sensors or "imu" not in sensor:
                continue

            # feat: [N, C, T]
            N, C, T = feat.shape
            device, dtype = feat.device, feat.dtype

            if self.mode == "ramp":
                rvec = self._sample_rvec_ramp(N, T, device, dtype)
            else:
                rvec = self._sample_rvec_smooth(N, T, device, dtype)

            R_mat = so3_exp_map(rvec.reshape(-1, 3)).reshape(N, T, 3, 3)

            feat_t = feat.permute(0, 2, 1).contiguous()   # [N, T, C]
            feat_aug = feat_t.clone()

            assert C % 3 == 0, f"Expected vector-valued features, got C={C}"
            for s in range(0, C, 3):
                feat_aug[..., s:s+3] = torch.matmul(
                    R_mat, feat_t[..., s:s+3].unsqueeze(-1)
                ).squeeze(-1)

            feats_new[sensor] = feat_aug.permute(0, 2, 1).contiguous()

        sample_new = {k: v for k, v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        return sample_new


class TransformSensorMisalignment:
    """
    Reviewer 4/5 대응: 센서가 물리적으로 삐뚤어져서 장착된(Calibration Error) 상황 시뮬레이션.
    하나의 윈도우(시퀀스) 전체에 대해 동일한 랜덤 회전 편향을 가합니다.
    """
    def __init__(self, misalignment_deg=3.0):
        self.misalignment_rad = misalignment_deg * math.pi / 180.0

    def __call__(self, sample):
        feats_new = {k: v.clone() for k, v in sample["feats"].items()}
        N = sample["ts_us"].shape[0]
        
        # 1. 윈도우 당 고정된 회전 축(Random Axis)과 지정된 각도(Magnitude) 생성
        axis = torch.randn(N, 3, device=sample["ts_us"].device, dtype=torch.float32)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        rvec = axis * self.misalignment_rad
        R_mat = so3_exp_map(rvec) # Shape: (N, 3, 3)

        for sensor, feat in feats_new.items():
            if "imu" in sensor:
                feat_aug = feat.clone()
                # 윈도우 전체(T개)에 대해 동일한 R_mat 적용
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
                feats_new[sensor] = feat_aug
                
        sample_new = {k: v for k, v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        
        # 중요: 물리적 센서 장착이 돌아갔으므로, 모델은 '돌아간 센서 프레임' 기준의 속도를 예측하게 됩니다.
        # 따라서 정답(Target) 속도/변위도 함께 돌려주어야 공정한 ATE 평가가 가능합니다.
        for k in ["targ_dt_World", "vel_World", "targ_dt_Body", "vel_Body"]:
            if k in sample:
                sample_new[k] = torch.einsum("nik,ntk->nti", R_mat, sample[k])
                
        return sample_new
    
"""
if __name__ == "__main__":
    # test
    def get_sample():
        acc = torch.tensor([[0], [1.0], [0]]).repeat(repeats=(1, 1, 200))
        gyr = torch.tensor([[0], [1.0], [0]]).repeat((1, 1, 200))
        Rarg = torch.tensor(torch.eye(3)).repeat(1, 1, 1)
        targ = torch.tensor([0, 1.0, 0]).repeat(1, 1)
        print(acc.shape)
        print(gyr.shape)
        print(Rarg)
        print(targ.shape)
        samples = (torch.cat((acc, gyr), dim=1), Rarg, targ)
        return samples

    transform = TransformInYawPlane("imu0")
    feat_aug, Rarg_aug, targ_aug = transform(
        get_sample(), angle_rad=torch.tensor(math.pi / 2)
    )

    print(feat_aug)
    print(Rarg_aug)
    print(targ_aug)
"""


class TransformSO3TrainAug:
    """
    Training-time SO(3) augmentation: random SO(3) rotation per window with UNIFORM magnitude in
    [0, max_deg]. Applied to both gyro and accel.
    Targets (vel_World, targ_dt_World, etc.) are ALSO rotated to maintain ground-truth consistency.
    """
    def __init__(self, input_sensors, max_deg=180.0):
        self.input_sensors = input_sensors
        self.max_rad = float(max_deg) * math.pi / 180.0

    def __call__(self, sample):
        feats_new = {k: v for k, v in sample["feats"].items()}
        N = sample["ts_us"].shape[0]
        device = sample["ts_us"].device
        # Random axis (uniform on S^2) × magnitude in [0, max_rad]
        axis = torch.randn(N, 3, device=device, dtype=torch.float32)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
        mag = torch.rand(N, 1, device=device, dtype=torch.float32) * self.max_rad
        rvec = axis * mag
        R_mat = so3_exp_map(rvec)
        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
            elif "mag" in sensor:
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
            feats_new[sensor] = feat_aug
        sample_new = {k: v for k, v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        # Rotate targets (must be equivariant for training to work)
        for k in ("targ_dt_World", "vel_World", "targ_dt_Body", "vel_Body"):
            if k in sample_new and sample_new[k] is not None:
                sample_new[k] = torch.einsum("nik,ntk->nti", R_mat, sample_new[k])
        return sample_new


class TransformTestSO3Rotation:
    """
    Test-time SO(3) perturbation: fixed random SO(3) rotation per window of magnitude EXACTLY
    deg (axis uniform on S^2, angle = deg degrees). Applied to gyro + accel AND to the vector
    targets, so predictions are scored in the rotated frame. This equals the paper's ID/SO(3)
    protocol (rotate inputs, rotate vector targets, un-rotate predictions): an exactly
    equivariant network incurs zero error. An earlier version of this transform left the
    targets unrotated, which penalises equivariant models by construction; do not use it.
    """
    def __init__(self, input_sensors, theta_deg):
        self.input_sensors = input_sensors
        self.theta_rad = float(theta_deg) * math.pi / 180.0

    def __call__(self, sample):
        feats_new = {k: v for k, v in sample["feats"].items()}
        N = sample["ts_us"].shape[0]
        device = sample["ts_us"].device
        # Uniform random axis on S^2 per window
        axis = torch.randn(N, 3, device=device, dtype=torch.float32)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rvec = axis * self.theta_rad
        R_mat = so3_exp_map(rvec)  # (N, 3, 3)
        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
            elif "mag" in sensor:
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
            feats_new[sensor] = feat_aug
        sample_new = {k: v for k, v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        for k in ("targ_dt_World", "vel_World", "targ_dt_Body", "vel_Body"):
            if k in sample_new and sample_new[k] is not None:
                sample_new[k] = torch.einsum("nik,ntk->nti", R_mat, sample_new[k])
        return sample_new


class TransformTestGravityTilt:
    """
    Test-time gravity tilt perturbation: fixed horizontal-axis rotation of EXACT magnitude `deg`
    per window. Axis ∈ xy-plane (uniform yaw). Models a sensor-mounting tilt error.
    """
    def __init__(self, input_sensors, theta_deg):
        self.input_sensors = input_sensors
        self.theta_rad = float(theta_deg) * math.pi / 180.0

    def __call__(self, sample):
        feats_new = {k: v for k, v in sample["feats"].items()}
        N = sample["ts_us"].shape[0]
        device = sample["ts_us"].device
        # Horizontal axis (uniform yaw)
        phi = torch.rand(N, device=device, dtype=torch.float32) * 2 * math.pi
        axis = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], dim=1)
        rvec = axis * self.theta_rad
        R_mat = so3_exp_map(rvec)
        for sensor, feat in feats_new.items():
            feat_aug = feat.clone()
            if "imu" in sensor:
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
            elif "mag" in sensor:
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
            feats_new[sensor] = feat_aug
        sample_new = {k: v for k, v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        return sample_new


class TransformLFARelativeRotationNoise:
    """
    Test-time perturbation for LFA-aligned inputs.

    Assumes sample["feats"][sensor] is already expressed in the last-frame-aligned
    frame S_n. We inject a time-varying relative rotation error:
        z_k^lfa -> DeltaR_k z_k^lfa
    with DeltaR_T = I, so the last timestep remains the anchor frame.

    This models imperfect relative rotations used inside LFA, not physical sensor
    misalignment. Therefore targets are NOT rotated.
    """
    def __init__(self, input_sensors, max_deg=3.0, mode="smooth"):
        self.input_sensors = input_sensors
        print("max deg :", max_deg)
        self.max_rad = max_deg * math.pi / 180.0
        self.mode = mode

    def _sample_rvec_smooth(self, N, T, device, dtype):
        """
        Generate smooth time-varying rotation vectors with rvec[:, -1] = 0.
        Each window is rescaled so its maximum perturbation magnitude equals max_deg.
        """
        if T == 1:
            return torch.zeros(N, 1, 3, device=device, dtype=dtype)

        inc = torch.randn(N, T - 1, 3, device=device, dtype=dtype)
        rvec = torch.zeros(N, T, 3, device=device, dtype=dtype)

        for t in range(T - 2, -1, -1):
            rvec[:, t, :] = rvec[:, t + 1, :] + inc[:, t, :]

        max_norm = rvec.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-8)
        rvec = rvec * (self.max_rad / max_norm).unsqueeze(-1)
        rvec[:, -1, :] = 0.0
        return rvec

    def _sample_rvec_ramp(self, N, T, device, dtype):
        """
        One random axis per window, linearly decaying to zero at the last frame.
        """
        axis = torch.randn(N, 3, device=device, dtype=dtype)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        alpha = torch.linspace(1.0, 0.0, T, device=device, dtype=dtype).view(1, T, 1)
        rvec = axis[:, None, :] * self.max_rad * alpha
        rvec[:, -1, :] = 0.0
        return rvec

    def __call__(self, sample):
        feats_new = {k: v.clone() for k, v in sample["feats"].items()}

        for sensor, feat in feats_new.items():
            if sensor not in self.input_sensors or "imu" not in sensor:
                continue

            # feat: [N, C, T]
            N, C, T = feat.shape
            device, dtype = feat.device, feat.dtype

            if self.mode == "ramp":
                rvec = self._sample_rvec_ramp(N, T, device, dtype)
            else:
                rvec = self._sample_rvec_smooth(N, T, device, dtype)

            R_mat = so3_exp_map(rvec.reshape(-1, 3)).reshape(N, T, 3, 3)

            feat_t = feat.permute(0, 2, 1).contiguous()   # [N, T, C]
            feat_aug = feat_t.clone()

            assert C % 3 == 0, f"Expected vector-valued features, got C={C}"
            for s in range(0, C, 3):
                feat_aug[..., s:s+3] = torch.matmul(
                    R_mat, feat_t[..., s:s+3].unsqueeze(-1)
                ).squeeze(-1)

            feats_new[sensor] = feat_aug.permute(0, 2, 1).contiguous()

        sample_new = {k: v for k, v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        return sample_new


class TransformSensorMisalignment:
    """
    Reviewer 4/5 대응: 센서가 물리적으로 삐뚤어져서 장착된(Calibration Error) 상황 시뮬레이션.
    하나의 윈도우(시퀀스) 전체에 대해 동일한 랜덤 회전 편향을 가합니다.
    """
    def __init__(self, misalignment_deg=3.0):
        self.misalignment_rad = misalignment_deg * math.pi / 180.0

    def __call__(self, sample):
        feats_new = {k: v.clone() for k, v in sample["feats"].items()}
        N = sample["ts_us"].shape[0]
        
        # 1. 윈도우 당 고정된 회전 축(Random Axis)과 지정된 각도(Magnitude) 생성
        axis = torch.randn(N, 3, device=sample["ts_us"].device, dtype=torch.float32)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        rvec = axis * self.misalignment_rad
        R_mat = so3_exp_map(rvec) # Shape: (N, 3, 3)

        for sensor, feat in feats_new.items():
            if "imu" in sensor:
                feat_aug = feat.clone()
                # 윈도우 전체(T개)에 대해 동일한 R_mat 적용
                feat_aug[:, :3, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, :3, :])
                feat_aug[:, 3:6, :] = torch.einsum("nik,nkt->nit", R_mat, feat[:, 3:6, :])
                feats_new[sensor] = feat_aug
                
        sample_new = {k: v for k, v in sample.items() if k != "feats"}
        sample_new["feats"] = feats_new
        
        # 중요: 물리적 센서 장착이 돌아갔으므로, 모델은 '돌아간 센서 프레임' 기준의 속도를 예측하게 됩니다.
        # 따라서 정답(Target) 속도/변위도 함께 돌려주어야 공정한 ATE 평가가 가능합니다.
        for k in ["targ_dt_World", "vel_World", "targ_dt_Body", "vel_Body"]:
            if k in sample:
                sample_new[k] = torch.einsum("nik,ntk->nti", R_mat, sample[k])
                
        return sample_new
    
"""
if __name__ == "__main__":
    # test
    def get_sample():
        acc = torch.tensor([[0], [1.0], [0]]).repeat(repeats=(1, 1, 200))
        gyr = torch.tensor([[0], [1.0], [0]]).repeat((1, 1, 200))
        Rarg = torch.tensor(torch.eye(3)).repeat(1, 1, 1)
        targ = torch.tensor([0, 1.0, 0]).repeat(1, 1)
        print(acc.shape)
        print(gyr.shape)
        print(Rarg)
        print(targ.shape)
        samples = (torch.cat((acc, gyr), dim=1), Rarg, targ)
        return samples

    transform = TransformInYawPlane("imu0")
    feat_aug, Rarg_aug, targ_aug = transform(
        get_sample(), angle_rad=torch.tensor(math.pi / 2)
    )

    print(feat_aug)
    print(Rarg_aug)
    print(targ_aug)
"""
