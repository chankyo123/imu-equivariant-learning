"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import time
import json
import torch
import shutil
import ctypes
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation
import copy

from utils.from_scipy import compute_euler_from_matrix
from utils.logging import get_logger
from .constants import *
import math
from utils.torch_math_utils import so3_exp_map

log = get_logger(__name__)

class SequencesDataset:
    """
    A template class for sequences dataset in TLIO training.
    Each subclass is expected to load data in a different way, but from the same data format.
    """

    def __init__(
        self,
        data_path,
        split,
        genparams,
        only_n_sequence=-1,
        sequence_subset=None,
        normalize_sensor_data=True,
        verbose=False,
        test_file_path = None,
        start_index = 0,
        theta_range_deg = 0,
        dataloader_bias_gravity_aug = False,
        accel_bias_range = 0,
        gyro_bias_range = 0,
        so3_augment = False,           # NEW: enable full random SO(3) rotation per window
        so3_augment_max_deg = 180.0,   # NEW: max rotation magnitude in degrees (180 = uniform SO(3))

    ):
        self.data_path = data_path
        self.split = split
        self.genparams = genparams

        self.only_n_sequence = only_n_sequence
        self.sequence_subset = sequence_subset
        self.normalize_sensor_data = normalize_sensor_data
        self.verbose = verbose
        self.test_file_path = test_file_path
        self.start_index = start_index
        self.theta_range_deg = theta_range_deg
        self.dataloader_bias_gravity_aug = dataloader_bias_gravity_aug
        self.accel_bias_range = accel_bias_range
        self.gyro_bias_range = gyro_bias_range
        self.so3_augment = so3_augment
        self.so3_augment_max_deg = so3_augment_max_deg
        
        # The list of relevant sensor file names based on data_style
        self.sensor_file_basenames = self.get_sensor_file_basenames()
            
        # Index the mem-mapped files and open them (data is not read from disk here)
        self.load_list()
        if self.verbose:
            self.log_dataset_info()
        
    def get_base_sensor_name(self):
        return self.sensor_file_basenames[0]
    
    def load_list(self):
        assert torch.utils.data.get_worker_info() is None, "load_list() can only be called in main proc!"

        #list_info = np.loadtxt(
        #    os.path.join(self.data_path, f"{self.split}_list.txt"), 
        #    dtype=np.dtype(str),
        #)
        if self.test_file_path is not None:
            with open(os.path.join(self.data_path, self.test_file_path)) as f:
                list_info = np.array([s.strip() for s in f.readlines() if len(s.strip()) > 0])
        else:
            with open(os.path.join(self.data_path, f"{self.split}_list.txt")) as f:
                list_info = np.array([s.strip() for s in f.readlines() if len(s.strip()) > 0])

        # For picking exactly some particular sequences
        if self.sequence_subset is not None:
            to_keep = np.array([s in self.sequence_subset for s in list_info])
            assert np.count_nonzero(to_keep) == len(self.sequence_subset), \
                    f"Could not find some sequences from sequence_subset in data list"
            list_info = list_info[to_keep]

        if self.split == "train" and self.only_n_sequence > 0:
            list_info = list_info[:self.only_n_sequence]
            
        # Handle empty lists (i.e., if you don't want to do test or val or something)
        self.data_list = []
        if len(list_info) > 0:
            self.data_list = list_info
        
        # Load the descriptions of all the data (column info and num rows)
        self.data_descriptions = []
        seqs_to_remove = [] # The seqs, not the index
        for seq_id in self.data_list:
            seq_desc = {}
            valid = True
            for i, sensor_basename in enumerate(self.sensor_file_basenames):
                with open(os.path.join(self.data_path, seq_id, 
                        sensor_basename+"_description.json"), 'r') as f: 
                    d = json.load(f)
                    if i == 0 and d["num_rows"] < self.genparams.window_size:
                        valid = False
                        log.warning(f"Sequence {seq_id} being ignored since it is too short ({d['num_rows']} rows)")
                        break
                    seq_desc[sensor_basename] = d
            
            if valid:
                self.data_descriptions.append(seq_desc)
            else:
                seqs_to_remove.append(seq_id)

        # Remove too short sequences from list
        if len(seqs_to_remove) > 0:
            self.data_list = np.array([seq for seq in self.data_list if seq not in seqs_to_remove])

    def get_sensor_file_basenames(self):
        if self.genparams.data_style == "aligned":
            return [COMBINED_SENSOR_NAME]
        elif self.genparams.data_style == "resampled":
            return [s + "_resampled" for s in ALL_SENSORS_LIST if s in self.genparams.input_sensors]
        elif self.genparams.data_style == "raw":
            return [s for s in ALL_SENSORS_LIST if s in self.genparams.input_sensors]
        else:
            raise ValueError(f"Invalid data_style {self.genparams.data_style}")
    
    def log_dataset_info(self):
        cumulated_duration_hrs = 0
        self.max_num_rows = None
        self.min_num_rows = None
        for i, seq_id in enumerate(self.data_list):
            seq_fps = {}
            desc = self.data_descriptions[i]
            for j, sensor_basename in enumerate(self.sensor_file_basenames):
                sensor_desc = desc[sensor_basename]
                num_cols = sum([
                    int(c.split("(")[1].split(")")[0]) for c in sensor_desc["columns_name(width)"]
                ])
                cumulated_duration_hrs += 1e-6 * (sensor_desc["t_end_us"] - sensor_desc["t_start_us"]) / 60 / 60
                self.max_num_rows = (
                    sensor_desc["num_rows"] if self.max_num_rows is None
                    else max(sensor_desc["num_rows"], self.max_num_rows)
                )
                self.min_num_rows = (
                    sensor_desc["num_rows"] if self.min_num_rows is None
                    else min(sensor_desc["num_rows"], self.min_num_rows)
                )
    
        # log some statitstics
        #log.info(f"Using these sequences: {list(self.data_list)}")
        log.info(
            f"Cumulated {self.split} dataset duration is {cumulated_duration_hrs:.3f} hours"
        )
        log.info(
            f"Number of {self.split} sequences is {len(self.data_descriptions)}"
        )
        #log.info(
        #    f"Number of {self.split} samples is {self.length} "
        #    f"(decimated by {self.genparams.decimator}x)"
        #)
        log.info(f"Min/max sequences length={self.min_num_rows}, {self.max_num_rows}") 
    
    def poses_to_target(self, rot, pos):
        # Calculate relative info on the fly
        # targ is what we want to regress from these features
        R_W_0 = Rotation.from_quat(rot[self.start_index:self.start_index+1]).as_matrix()
        R_W_i = Rotation.from_quat(rot).as_matrix()

        # NOTE R_W_i @ R_W_0.transpose() looks strange, but it is the delta rotation between the two times
        # aligned with the world frame instead of body frame.
        targ_dR_World = R_W_i @ R_W_0.transpose([0,2,1])
        targ_dt_World = pos - pos[self.start_index:self.start_index+1] # Displacement in global frame
        return targ_dR_World, targ_dt_World
    
    # def positional_encoding(self, x, num_frequencies, include_input= True):
    #     pos_out = []
    #     if include_input:
    #         pos_out.append(x)
    #     for i in range(num_frequencies):
    #         pos_out.append(np.sin(x/np.power(1e4, (2*i)/x.shape[0])))#pos_out.append(np.sin((2 ** i) * np.pi * x))
    #         pos_out.append(np.cos(x/np.power(1e4, (2*i)/x.shape[0])))#np.cos((2 ** i) * np.pi * x)

    #     pos_out = np.concatenate(pos_out, axis=-1,)
    #     return pos_out
    
    def positional_encoding(self, x, num_frequencies, incl_input=True):
        prepend = 1 if incl_input else 0
        enc_sz = x.shape[1] * (prepend + 2 * num_frequencies)
        res = np.zeros((x.shape[0], enc_sz))

        if incl_input:
            res[:, :x.shape[1]] = x
            #np.sin(x/np.power(1e4, (2*i)/x.shape[0]))
        powers = np.power(1e4, (2*np.arange(num_frequencies))/x.shape[0]) # (L,)
        sin_phases =  x[:, None, :]/powers[None, :, None] # (N, L, D)
        cos_phases = np.pi / 2 - sin_phases
        phases = np.stack([sin_phases, cos_phases], axis=-2) # (N, L, 2, D)
        # print(phases.shape)
        flat = phases.reshape((phases.shape[0], -1))
        res[:, prepend*x.shape[1]:] = np.sin(flat)

        return res

    def preprocess_o2(self,feat):

        feat = feat.transpose(1,0) #(n,d) --> (200,6)
        accel = feat[...,-3:]
        gyro = feat[...,:3]
        # mask = (np.linalg.norm(np.cross(gyro, accel),axis=-1)==0).astype(np.int32)
        v1 = np.zeros((*feat.shape[:-1],3))
        v2 = np.zeros((*feat.shape[:-1],3))
        
        # mask += (gyro[...,-1]==0).astype(np.int32)*1
        # mask += (np.linalg.norm(gyro[...,:-1], axis=-1) == 0).astype(np.int32)*2
        mask = (np.linalg.norm(gyro[...,:-1], axis=-1) == 0).astype(np.int32)

        # v1[mask==0] = np.cross(gyro[mask==0],accel[mask==0])
        # v2[mask==0] = np.cross(gyro[mask==0],v1[mask==0])
        
        R = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        gyro_flip = copy.deepcopy(gyro@R.T)
        gyro_flip[...,-1] = 0

        # v1[mask==1] = np.cross(gyro[mask==1], gyro_flip[mask==1])
        # v2[mask==1] = np.cross(gyro[mask==1],v1[mask==1])

        v1[mask==0] = np.cross(gyro[mask==0], gyro_flip[mask==0])
        v2[mask==0] = np.cross(gyro[mask==0],v1[mask==0])

        # v1[mask==2,-1] = 1
        # v2[mask==2] = np.cross(v1[mask==2],gyro[mask==2])

        x = np.zeros_like(gyro)
        x[...,-2] = 1
        v1[mask==1] = np.cross(x[mask==1], gyro[mask==1])
        v2[mask==1] = np.cross(gyro[mask==1], v1[mask==1])

        gyro_norm = np.linalg.norm(gyro,axis=-1,keepdims=True)
        v1 = v1 * gyro_norm /np.clip(np.linalg.norm(v1, axis=-1,keepdims=True),1e-7,1e13)
        v2 = v2 * gyro_norm/np.clip(np.linalg.norm(v2,axis=-1,keepdims=True),1e-7,1e13)

        return np.concatenate([accel,v1,v2],axis=-1).transpose(1,0)


    def unpack_data_window(self, seq_data, seq_desc, row):
        feats = {}
        ts_us_base_sensor = None
        if self.genparams.data_style == "aligned":
            data_chunk = seq_data[COMBINED_SENSOR_NAME][row:row+self.genparams.window_size]
            # Make sure idx was valid with sufficient padding for window
            assert data_chunk.shape[0] == self.genparams.window_size
            
            #ts_us, gyr, acc, rot, pos, vel = np.hsplit(data_chunk, [1, 4, 7, 11, 14])
            ts_us = data_chunk[:,0:1]
            ts_us_base_sensor = np.copy(ts_us)

            ## adding positional encoded timestamp as a feature (needed for eq_transformer arch)
            ts_ind = np.arange(1, ts_us.shape[0]+1)
            feats["pe_ts"] = self.positional_encoding(ts_ind.reshape((-1,1)), num_frequencies=4, incl_input=True)

            # Check which sensor data we need to load
            # TODO make this less hard-coded for the columns somehow
            # Maybe the description json could be better for getting the columns
            #if "imu0" in self.genparams.input_sensors:
            if "imu0" in self.genparams.input_sensors:
                feats["imu0"] = data_chunk[:,1:7] # gyro0, accelerometer0
            if "imu1" in self.genparams.input_sensors:
                feats["imu1"] = data_chunk[:,7:13] # gyro1, accelerometer1
            if "mag0" in self.genparams.input_sensors:
                feats["mag0"] = data_chunk[:,13:16]
            if "barom0" in self.genparams.input_sensors:
                feats["barom0"] = data_chunk[:,16:18]
            
            # All sensors have the same timestamps in this data_style, just concat here
            #ts_normalized = 2 * (ts_us - ts_us[0]) / (ts_us[-1] - ts_us[0]) - 1
            for k, v in feats.items():
            #    feats[k] = np.concatenate([ts_normalized, v], axis=1).astype(np.float32).T
                feats[k] = v.astype(np.float32).T
            
            rot, pos, vel = data_chunk[:,-10:-6], data_chunk[:,-6:-3], data_chunk[:,-3:]

        elif self.genparams.data_style == "resampled":
            
            # With resampled data, the "approximate_frequency" in the json file is exact,
            # so we can quickly index the timestamps of sensors in different memmap files.
            base_sensor_freq = seq_desc[self.get_base_sensor_name()]["approximate_frequency_hz"]
            base_sensor_window_start_time = None
            base_sensor_window_end_time = None
            for i, sensor_name in enumerate(self.sensor_file_basenames):
                if i == 0:
                    sensor_row = row
                    window_size = self.genparams.window_size
                else:
                    # Index the row based on sensor frequency.
                    sensor_freq = seq_desc[sensor_name]["approximate_frequency_hz"]
                    sensor_seq_start_time = seq_desc[sensor_name]["t_start_us"]
                    # TODO off by one possible here from rounding/flooring
                    sensor_row = int(1e-6*(base_sensor_window_start_time - sensor_seq_start_time) * sensor_freq)
                    # TODO should calculate all the window sizes at startup so that we don't
                    # accidentally get an off-by-one window size error from float errors
                    window_size = int(self.genparams.window_size * sensor_freq / base_sensor_freq)
            
                data_chunk = seq_data[sensor_name][sensor_row:sensor_row+window_size]
                # Make sure idx was valid with sufficient padding for window
                assert data_chunk.shape[0] == window_size
                ts_us = data_chunk[:,0:1]
                ts_ind = np.arange(1, ts_us.shape[0]+1)
                feats["pe_ts"] = self.positional_encoding(ts_ind.reshape((-1,1)), num_frequencies=4, incl_input=True)
                if i == 0:
                    ts_us_base_sensor = np.copy(ts_us)
                    base_sensor_window_start_time = ts_us[0]
                    base_sensor_window_end_time = ts_us[-1]
                    # GT data comes from base sensor
                    rot, pos, vel = data_chunk[:,-10:-6], data_chunk[:,-6:-3], data_chunk[:,-3:]
                
                if "imu" in sensor_name:
                    feat = data_chunk[:,1:7] # gyro, accelerometer
                if "mag" in sensor_name:
                    feat = data_chunk[:,1:4]
                if "barom" in sensor_name:
                    feat = data_chunk[:,1:3]
                
                # All timestamps relative to base sensor
                #ts_normalized = 2 * (
                #    ts_us - ts_us_base_sensor[0]
                #) / (
                #    ts_us_base_sensor[-1] - ts_us_base_sensor[0]
                #) - 1
                
                #feats[sensor_name.split("_")[0]] = np.concatenate([ts_normalized, feat], axis=1).astype(np.float32).T
                feats[sensor_name.split("_")[0]] = feat.astype(np.float32).T

        else:
            raise NotImplementedError()
        
        gt_data = ts_us_base_sensor, rot.astype(np.float32), pos.astype(np.float32), vel.astype(np.float32)
        
        return feats, gt_data

    def data_chunk_from_seq_data(self, seq_data, seq_desc, row):
        feats, gt_data = self.unpack_data_window(seq_data, seq_desc, row)

        # Normalize the raw sensor data into something better for learning (sensor-dependent)
        if self.normalize_sensor_data:
            feats = self.normalize_feats(feats)

        ## flag to experiment with and without gravity compensating accel data -- we can directly subtract as the data is in world frame
        if self.genparams.g_compensate:
            # print('entering gravity compensation')
            for k, v in feats.items():
                if "imu0" in k:
                    # print('IMU data is being gravity compensated!')
                    assert feats[k].shape[0] == 6
                    feats[k] -= np.array([0, 0, 0, 0, 0, 9.81]).reshape((-1,1)) # feat = gyr, accel and shape is (6, samples)
            
        

        ts_us, rot, pos, vel = gt_data
        targ_dR_World, targ_dt_World = self.poses_to_target(rot, pos)

        R_world_gla = np.eye(3)
        if self.genparams.express_in_local_gravity_aligned:
            #assert False
            # print('Output data is also being aligned to local gravity aligned!')
            R_W_0 = Rotation.from_quat(rot[0:1]).as_matrix()
            angles_t0 = compute_euler_from_matrix(
                R_W_0, "xyz", extrinsic=True
            )
            ri_z = angles_t0[0,2]
            c = np.cos(ri_z)
            s = np.sin(ri_z)
            R_world_gla = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            targ_dt_World = np.einsum("ji,tj->ti", R_world_gla, targ_dt_World)

            # Only IMU and mag data need to be rotated (not barometer) - since the input is compensated rotated in world from npy file
            for k, v in feats.items():
                if "imu0" in k:
                    # print('IMU data is also being aligned to local gravity aligned!')
                    assert feats[k].shape[0] == 6
                    feats[k][:3] = np.einsum("ji,jt->it", R_world_gla, feats[k][:3])
                    feats[k][3:] = np.einsum("ji,jt->it", R_world_gla, feats[k][3:])
                elif "mag" in k:
                    assert feats[k].shape[0] == 3
                    feats[k] = np.einsum("ji,jt->it", R_world_gla, feats[k])
            
        elif self.genparams.express_in_local_frame: 
            #assert False
            # print('Entered local frame data prep step!')
            R_world_gla = Rotation.from_quat(rot[0]).as_matrix()
            #print(Rotation.from_quat(rot[0]).as_matrix().shape)
            targ_dt_World = np.einsum("ji,tj->ti", R_world_gla, targ_dt_World)
            # Only IMU and mag data need to be rotated (not barometer) - since the input is compensated rotated in world from npy file
            for k, v in feats.items():
                if "imu0" in k:
                    # print('IMU data is also being aligned to local frame!')
                    assert feats[k].shape[0] == 6
                    feats[k][:3] = np.einsum("ji,jt->it", R_world_gla, feats[k][:3])
                    feats[k][3:] = np.einsum("ji,jt->it", R_world_gla, feats[k][3:])
                elif "mag" in k:
                    assert feats[k].shape[0] == 3
                    feats[k] = np.einsum("ji,jt->it", R_world_gla, feats[k])
           
            

        else: ## global frame

            ## since both input and output are already in global frame, no rotations needed
            # print('global frame! No rotations for both imu and output data!')
            targ_dt_World = targ_dt_World
            feats = feats
        
        ### do gravity perturbation here
        if self.dataloader_bias_gravity_aug:
            feats['imu0'] = torch.Tensor(feats['imu0'])
            if self.theta_range_deg!=0:
                angle_rand_rad = (
                    torch.rand(1, device=feats['imu0'].device, dtype=torch.float32) * math.pi * 2
                )
                theta_rand_rad = (
                    torch.rand(1, device=feats['imu0'].device, dtype=torch.float32)
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

                feats['imu0'][:3, :] = torch.einsum("ik,kt->it", R_mat.squeeze(0), feats['imu0'][:3, :])
                feats['imu0'][3:6, :] = torch.einsum("ik,kt->it", R_mat.squeeze(0), feats['imu0'][3:6, :])
            
            ### do accel and gyro bias perturbation here
            if self.gyro_bias_range!=0:
                feats['imu0'][:3, :] += (
                    (torch.rand(3, 1, device=feats['imu0'].device, dtype=feats['imu0'].dtype) - 0.5)
                    * self.gyro_bias_range / 0.5
                )
            if self.accel_bias_range!=0:
                feats['imu0'][3:6, :] += (
                    (torch.rand(3, 1, device=feats['imu0'].device, dtype=feats['imu0'].dtype) - 0.5)
                    * self.accel_bias_range / 0.5
                )
            feats['imu0'] = feats['imu0'].numpy().astype(np.float32)


        # ---- SO(3) data augmentation ----
        # Apply a random SO(3) rotation R to IMU (gyro, accel) and to the target displacement / velocity.
        # This exposes the model to arbitrary orientations of the IMU sensor frame.
        # Magnitude controlled by so3_augment_max_deg (default 180 ~ uniform on SO(3)).
        if getattr(self, "so3_augment", False):
            # Sample a random axis-angle within the requested magnitude
            theta = np.random.uniform(0.0, np.deg2rad(self.so3_augment_max_deg))
            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            from scipy.spatial.transform import Rotation as _Rot
            R_so3 = _Rot.from_rotvec(axis * theta).as_matrix()
            # feats['imu0'] is shape (6, T): rows 0..2 = gyro (3D), rows 3..5 = accel (3D)
            feats['imu0'][:3, :] = np.einsum("ij,jt->it", R_so3, feats['imu0'][:3, :])
            feats['imu0'][3:6, :] = np.einsum("ij,jt->it", R_so3, feats['imu0'][3:6, :])
            # targets in world frame: shape (T, 3)
            targ_dt_World = np.einsum("ij,tj->ti", R_so3, targ_dt_World)
            vel = np.einsum("ij,tj->ti", R_so3, vel)
            # Update R_world_gla bookkeeping so that test code that uses it remains consistent.
            # We left-multiply by R_so3 (since the world frame as observed by the IMU has been rotated).
            R_world_gla = R_so3 @ R_world_gla

        feats['feat_o2'] = self.preprocess_o2(feats['imu0'])
        # We may return multiple windows, so place them all in here for convenience.
        windows = {
            "main": {
                "ts_us": ts_us,
                "feats": feats,
                "targ_dR_World": targ_dR_World.astype(np.float32),
                "targ_dt_World": targ_dt_World.astype(np.float32),
                "vel_World": vel.astype(np.float32),
                "R_world_gla": R_world_gla,
            }
        }

        return seq_desc[self.get_base_sensor_name()], windows
    
    def normalize_feats(self, feats):
        """
        Normalize the sensor data from its raw form to some normalized form, typically in [-1,1] or [0,1].
        """
        
        new_feats = {}
        for sensor_name, feat in feats.items():
            # Note that all feat are [1+C,T] where C is channels in sensor data and T is tme dimension.
            # The 1+ is because the sensor data is concatenated with normalized time stamp.
            new_feat = np.copy(feat)
            # Check for nan/inf here (sometimes can pop up in the data)
            new_feat[~np.isfinite(new_feat)] = 0.0
            """  NOTE makes values too small, and disrupts bias perturbation logic
            if "imu" in sensor_name:
                assert new_feat.shape[0] == 6
                # See T74692750 for more info.
                # Out of the two IMUs, the one with the max range is at +/-8G and +/-1000 deg/sec.
                # Normalize by this one so that both IMU values have the same meaning, and are normalized in [-1,1]
                minmax_acc_range_g = 8 # In unit of Gs
                minmax_ang_vel_range_deg_per_sec = 1000
                # IMU values should be in [-1,1] after this
                new_feat[:3] = new_feat[:3] / (minmax_ang_vel_range_deg_per_sec / 180 * np.pi) # gyro
                new_feat[3:6] = new_feat[3:6] / (minmax_acc_range_g * 9.81) # accelerometer
            """
            if "mag" in sensor_name:
                assert new_feat.shape[0] == 3
                # Convert to Gauss, which is closer to 1 in magnitude (Earth's field is around .25-.65 Gauss, and 
                # can be negative here since the magnetomete returns a magnetic field vector instead of magnitude)
                GAUSS_IN_TESLA = 10_000
                new_feat[:3] = new_feat[:3] * GAUSS_IN_TESLA
            if "barom" in sensor_name:
                assert new_feat.shape[0] == 2
                # Pressure converted to bar and normalized heuristically to fit into [-1,1] better.
                # Setting -1,1 to be the min/max pressure/temp ever recorded leads to very small differences
                # in the values for normal situations, so just picked min/max based on some normal daily values on Earth.
                PA_IN_BAR = 100_000
                """
                avg_bar = 1.01325 # Average barometric pressure on earth
                max_bar_deviation = 0.01 # plus/minus avg is what we are considering
                min_bar, max_bar = avg_bar - max_bar_deviation, avg_bar + max_bar_deviation
                new_feat[0] = 2 * (new_feat[0] / PA_IN_BAR - min_bar) / (max_bar - min_bar) - 1
                min_temp = -100
                max_temp = 100
                new_feat[1] = 2 * (new_feat[1] - min_temp) / (max_temp - min_temp) - 1
                """
                new_feat[0] /= PA_IN_BAR # convert pa to bar

            new_feats[sensor_name] = new_feat
        
        return new_feats
    
    def load_data_chunk(self, seq_idx, row):
        raise NotImplementedError("Did not override load_data_chunk!!!")

    def load_and_preprocess_data_chunk(self, seq_idx, row_in_seq, num_rows_in_seq):
        # If training, randomize the row a bit so that we can get better coverage of the data
        # while still respecting the decimator and indexing.
        if self.split == "train":
            row_in_seq = min(num_rows_in_seq-1, row_in_seq + np.random.randint(self.genparams.decimator))
        meta_dict, windows = self.load_data_chunk(seq_idx, row_in_seq)

        ret = {
            "seq_id": self.data_list[seq_idx],
        }
        ret.update(windows["main"]) # Main target and GT data corresponding to seq_idx and row_in_seq
        return ret

    ##########################################################################
    # Functions needed by IntegrateRoninCallback
    ##########################################################################

    def get_ts_last_imu_us(self, seq_idx=0):
        raise NotImplementedError("Did not override get_ts_last_imu_us!!!")

    def get_gt_traj_center_window_times(self, seq_idx=0):
        raise NotImplementedError("Did not override get_gt_traj_center_window_times!!!")
