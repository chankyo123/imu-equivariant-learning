#!/usr/bin/env python3

import json
from typing import Optional

import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from tracker.imu_buffer import ImuBuffer
from tracker.imu_calib import ImuCalib
from tracker.meas_source_torchscript import MeasSourceTorchScript
from tracker.scekf import ImuMSCKF
from utils.dotdict import dotdict
from utils.from_scipy import compute_euler_from_matrix
from utils.logging import logging
from utils.math_utils import mat_exp
from scipy.spatial.transform import Rotation
import re

class ImuTracker:
    """
    ImuTracker is responsible for feeding the EKF with the correct data
    It receives the imu measurement, fills the buffer, runs the network with imu data in buffer
    and drives the filter.
    """

    def __init__(
        self,
        model_path,
        model_param_path,
        update_freq,
        filter_tuning_cfg,
        imu_calib: Optional[ImuCalib] = None,
        force_cpu=False,
        vio_path = None,
        json_path = None,
        body_frame = False,
        use_riekf = False,
        input_3 = False,
        out_dir = None,
    ):

        config_from_network = dotdict({})
        with open(model_param_path) as json_file:
            data_json = json.load(json_file)
            config_from_network["imu_freq_net"] = data_json["imu_freq"]
            config_from_network["past_time"] = data_json["past_time"]
            config_from_network["window_time"] = data_json["window_time"]
            config_from_network["arch"] = data_json["arch"]

        # frequencies and sizes conversion
        if not (
            config_from_network.past_time * config_from_network.imu_freq_net
        ).is_integer():
            raise ValueError(
                "past_time cannot be represented by integer number of IMU data."
            )
        if not (
            config_from_network.window_time * config_from_network.imu_freq_net
        ).is_integer():
            raise ValueError(
                "window_time cannot be represented by integer number of IMU data."
            )
        self.imu_freq_net = (
            config_from_network.imu_freq_net
        )  # imu frequency as input to the network
        self.past_data_size = int(
            config_from_network.past_time * config_from_network.imu_freq_net
        )
        self.disp_window_size = int(
            config_from_network.window_time * config_from_network.imu_freq_net
        )
        self.net_input_size = self.disp_window_size + self.past_data_size

        # EXAMPLE :
        # if using 200 samples with step size 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        # if using 400 samples with 200 past data and clone_every_n_netimu_sample 10, inference at 20 hz
        # we do update between clone separated by 19=update_distance_num_clone-1 other clone
        # if not (config_from_network.imu_freq_net / update_freq).is_integer():
        #     raise ValueError("update_freq must be divisible by imu_freq_net.")
        if not (config_from_network.window_time * update_freq).is_integer():
            raise ValueError(
                "window_time cannot be represented by integer number of updates."
            )
        self.update_freq = update_freq
        self.clone_every_n_netimu_sample = int(
            config_from_network.imu_freq_net / update_freq
        )  # network inference/filter update interval
        # assert (
        #     config_from_network.imu_freq_net % update_freq == 0
        # )  # imu frequency must be a multiple of update frequency
        self.update_distance_num_clone = int(
            config_from_network.window_time * update_freq
        )

        # time
        self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
        self.dt_update_us = int(
            1.0 / self.update_freq * 1e6
        )  # multiple of interpolation interval

        # logging
        logging.info(
            f"Network Input Time: {config_from_network.past_time + config_from_network.window_time} = {config_from_network.past_time} + {config_from_network.window_time} (s)"
        )
        logging.info(
            f"Network Input size: {self.net_input_size} = {self.past_data_size} + {self.disp_window_size} (samples)"
        )
        logging.info("IMU interpolation frequency: %s (Hz)" % self.imu_freq_net)
        logging.info("Measurement update frequency: %s (Hz)" % self.update_freq)
        logging.info(
            "Filter update stride state number: %i" % self.update_distance_num_clone
        )
        logging.info(
            f"Interpolating IMU measurement every {self.dt_interp_us}us for the network input"
        )

        # IMU initial calibration
        self.icalib = imu_calib
        self.filter_tuning_cfg = filter_tuning_cfg # Config
        # MSCKF
        self.filter = ImuMSCKF(filter_tuning_cfg)

        net_config = {"in_dim": (self.past_data_size + self.disp_window_size) // 32 + 1}
        self.meas_source = MeasSourceTorchScript(model_path, force_cpu)

        self.imu_buffer = ImuBuffer()

        #  This callback is called at first update if set
        self.callback_first_update = None
        # This callback can be use to bypass network use for measurement
        self.debug_callback_get_meas = None

        # keep track of past timestamp and measurement
        self.last_t_us = -1

        # keep track of the last measurement received before next interpolation time
        self.t_us_before_next_interpolation = -1
        self.last_acc_before_next_interp_time = None
        self.last_gyr_before_next_interp_time = None

        self.next_interp_t_us = None
        self.next_aug_t_us = None
        self.has_done_first_update = False
        
        self.body_frame = body_frame
        self.vio_path = vio_path
        self.json_path = json_path
        self.use_riekf = use_riekf
        self.input_3 = input_3
        self.model_path = model_path
        print("self.update_freq : ", self.update_freq)
        self.out_dir = out_dir
        self.vio_data = np.load(self.vio_path)
        
        
    @jit(forceobj=True, parallel=False, cache=False)
    def _get_imu_samples_for_network(self, t_begin_us, t_oldest_state_us, t_end_us):
        # extract corresponding network input data
        net_tus_begin = t_begin_us
        net_tus_end = t_end_us - self.dt_interp_us

        net_acc, net_gyr, net_tus = self.imu_buffer.get_data_from_to(
            net_tus_begin, net_tus_end, self.update_freq
        )
        # print(net_tus[-10:])
        # print(net_gyr.shape, self.net_input_size, net_tus.shape)
        assert net_gyr.shape[0] == self.net_input_size
        assert net_acc.shape[0] == self.net_input_size
        
        net_gyr_w = net_gyr
        net_acc_w = net_acc
        # print(net_tus_begin, net_tus[:3]) #og TLIO에서 net_tus_end - net_tus[-1] = 1000 딱 떨어짐, 그리고 5000씩 딱딱 맞춰 증가, 200hz 이아헹선 아예 같음
        # print("is running here?2")
        if self.body_frame:
            if "last_align" in self.model_path:
                Rs_bofbi = np.zeros((net_tus.shape[0], 3, 3))  # N x 3 x 3
                Rs_bofbi[-1, :, :] = np.eye(3)
                bg = self.filter.state.s_bg
                # print(bg)
                # assert False
                for j in range(1, net_tus.shape[0]):
                    dt_us = net_tus[-j] - net_tus[-j - 1]
                    dR = mat_exp((net_gyr[-j, :].reshape((3, 1)) - bg) * dt_us * 1e-6)
                    Rs_bofbi[-j - 1, :, :] = dR.dot(Rs_bofbi[-j, :, :])
                
                Rs_bofbi = Rs_bofbi.transpose(0,2,1)
                # print(Rs_bofbi[-1])
                # print(Rs_bofbi[-200])
                # assert False
                
                net_acc_w = np.einsum("tij,tj->ti", Rs_bofbi, net_acc)  # N x 3
                
            
        else:
            # get data from filter
            R_oldest_state_wfb, _ = self.filter.get_past_state(t_oldest_state_us)  #   3 x 3  #using ekf
            
            # # R_oldest_state_wfb, _ = self.filter.get_past_state(t_oldest_state_us+50000)  # 3 x 3    #using riekf
            # vio_data = np.load(s/elf.vio_path)
            # time = vio_data[:, 0]
            # qxyzw_World_Device = vio_data[:, -10:-6]
            # interp_funcs_quaternion = [interp1d(time, qxyzw_World_Device[:, i], fill_value="extrapolate") for i in range(qxyzw_World_Device.shape[1])]
            # quaternion_gt = np.array([interp_func(t_oldest_state_us) for interp_func in interp_funcs_quaternion])
            # quaternion_gt = quaternion_gt / np.linalg.norm(quaternion_gt)
            # R_oldest_state_wfb = Rotation.from_quat(quaternion_gt).as_matrix()  # bd -> world # 3 x 3    #using riekf
            
            # net_acc_w = np.zeros_like(net_acc)
            # net_gyr_w = np.zeros_like(net_gyr)
            # q_interpolated = np.zeros((len(net_tus), 4))

            # for i, t in enumerate(net_tus):
            #     closest_idx = np.argmin(np.abs(time - t))
            #     q_interpolated[i] = qxyzw_World_Device[closest_idx]

            # # Apply the rotation to each IMU data point
            # for i in range(len(net_acc)):
            #     q = q_interpolated[i]
            #     rot_matrix = Rotation.from_quat(q).as_matrix()
                
            #     # Rotate the acceleration and gyroscope data
            #     net_acc_w[i] = rot_matrix @ net_acc[i]
            #     net_gyr_w[i] = rot_matrix @ net_gyr[i]
            
            ### change the input of the network to be in local frame
            ri_z = compute_euler_from_matrix(R_oldest_state_wfb, "xyz", extrinsic=True)[
                0, 2
            ]
            Ri_z = np.array(
                [
                    [np.cos(ri_z), -(np.sin(ri_z)), 0],
                    [np.sin(ri_z), np.cos(ri_z), 0],
                    [0, 0, 1],
                ]
            )
            R_oldest_state_wfb = Ri_z.T @ R_oldest_state_wfb

            bg = self.filter.state.s_bg
            # dynamic rotation integration using filter states
            # Rs_net will contains delta rotation since t_begin_us
            Rs_bofbi = np.zeros((net_tus.shape[0], 3, 3))  # N x 3 x 3
            Rs_bofbi[0, :, :] = np.eye(3)
            for j in range(1, net_tus.shape[0]):
                dt_us = net_tus[j] - net_tus[j - 1]
                dR = mat_exp((net_gyr[j, :].reshape((3, 1)) - bg) * dt_us * 1e-6)
                Rs_bofbi[j, :, :] = Rs_bofbi[j - 1, :, :].dot(dR)

            # find delta rotation index at time ts_oldest_state
            oldest_state_idx_in_net = np.where(net_tus == t_oldest_state_us)[0][0]

            # rotate all Rs_net so that (R_oldest_state_wfb @ (Rs_bofbi[idx].inv() @ Rs_bofbi[i])
            # so that Rs_net[idx] = R_oldest_state_wfb
            R_bofboldstate = (
                R_oldest_state_wfb @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
            )  # [3 x 3]
            Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)
            net_acc_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_acc)  # N x 3
            net_gyr_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)  # N x 3
        # print("is running here?3")

        return net_gyr_w, net_acc_w

    def _compensate_measurement_with_initial_calibration(self, gyr_raw, acc_raw):
        if self.icalib:
            # #logging.info("Using bias from initial calibration")
            # init_ba = self.icalib.accelBias
            # init_bg = self.icalib.gyroBias
            # # calibrate raw imu data
            
            # acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
            #     acc_raw, gyr_raw
            # )  # removed offline bias and scaled
            # # acc_biascpst, gyr_biascpst = acc_raw, gyr_raw
            
            init_ba = np.zeros((3, 1))
            init_bg = np.zeros((3, 1))
            acc_biascpst, gyr_biascpst = acc_raw, gyr_raw
        else:
            #logging.info("Using zero bias")
            init_ba = np.zeros((3, 1))
            init_bg = np.zeros((3, 1))
            acc_biascpst, gyr_biascpst = acc_raw, gyr_raw
        return gyr_biascpst, acc_biascpst, init_bg, init_ba

    def _after_filter_init_member_setup(self, t_us, gyr_biascpst, acc_biascpst):
        self.next_interp_t_us = t_us
        self.next_aug_t_us = t_us
        self._add_interpolated_imu_to_buffer(acc_biascpst, gyr_biascpst, t_us)
        self.next_aug_t_us = t_us + self.dt_update_us

        self.last_t_us = t_us

        self.t_us_before_next_interpolation = t_us
        self.last_acc_before_next_interp_time = acc_biascpst
        self.last_gyr_before_next_interp_time = gyr_biascpst

    def init_with_state_at_time(self, t_us, R, v, p, gyr_raw, acc_raw):
        assert R.shape == (3, 3)
        assert v.shape == (3, 1)
        assert p.shape == (3, 1)

        logging.info(f"Initializing filter at time {t_us*1e-6}")
        (
            gyr_biascpst,
            acc_biascpst,
            init_bg,
            init_ba,
        ) = self._compensate_measurement_with_initial_calibration(gyr_raw, acc_raw)
        self.filter.initialize_with_state(t_us, R, v, p, init_ba, init_bg, self.use_riekf)
        self._after_filter_init_member_setup(t_us, gyr_biascpst, acc_biascpst)
        # print(gyr_biascpst - gyr_raw)
        # print(acc_biascpst - acc_raw)
        # assert False
        return False

    def _init_without_state_at_time(self, t_us, gyr_raw, acc_raw):
        assert isinstance(t_us, int)
        logging.info(f"Initializing filter at time {t_us*1e-6}")
        (
            gyr_biascpst,
            acc_biascpst,
            init_bg,
            init_ba,
        ) = self._compensate_measurement_with_initial_calibration(gyr_raw, acc_raw)
        self.filter.initialize(t_us, acc_biascpst, init_ba, init_bg, self.use_riekf)
        self._after_filter_init_member_setup(t_us, gyr_biascpst, acc_biascpst)

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw, i, progress):
        assert isinstance(t_us, int)
        # if t_us - self.last_t_us > 3e3:
        #     logging.warning(f"Big IMU gap : {t_us - self.last_t_us}us")

        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw, i, progress)
        else:
            self._init_without_state_at_time(t_us, gyr_raw, acc_raw)
            return False

    def _on_imu_measurement_after_init(self, t_us, gyr_raw, acc_raw, i, progress):
        """
        For new IMU measurement, after the filter has been initialized
        """
        assert isinstance(t_us, int)

        # Eventually calibrate
        if self.icalib:
            # if ("zero_bias" not in self.out_dir and "/local_data/" in self.out_dir) or ("/local_data/" not in self.out_dir and "zero_bias" in self.out_dir):
            # if ("no_bias" in self.out_dir) != ("/local_data/" in self.out_dir):
            if True:
                acc_biascpst = acc_raw
                gyr_biascpst = gyr_raw
            else:
                # calibrate raw imu data with offline calibation
                # this is used for network feeding
                acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                    acc_raw, gyr_raw
                )  # removed offline bias and scaled

                # # calibrate raw imu data with offline calibation scale
                # # this is used for the filter. By not applying offline bias
                # # we expect the filter to estimate bias similar to the offline
                
                # calibrated one
                acc_raw, gyr_raw = self.icalib.scale_raw(
                    acc_raw, gyr_raw
                )  # only offline scaled - into the filter
            
        else:
            acc_biascpst = acc_raw
            gyr_biascpst = gyr_raw

        # decide if we need to interpolate imu data or do update
        # print(t_us, self.next_interp_t_us, self.dt_update_us)
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        # assert False
        # if not self.use_riekf : 
        do_augmentation_and_update = t_us >= self.next_aug_t_us
        # print(t_us, self.next_interp_t_us, self.next_aug_t_us)

        # if augmenting the state, check that we compute interpolated measurement also
        # assert (
        #     do_augmentation_and_update and do_interpolation_of_imu
        # ) or not do_augmentation_and_update, (
        #     "Augmentation and interpolation does not match!"
        # )

        # augmentation propagation / propagation
        # propagate at IMU input rate, augmentation propagation depends on t_augmentation_us
        t_augmentation_us = self.next_aug_t_us if do_augmentation_and_update else None

        # IMU interpolation and data saving for network (using compensated IMU)
        
        ######### if using gravity compensated body acceleration ###
        if "local_bodyframe_uf20" in self.out_dir:
            # print("compensated!")
            vio_data = self.vio_data
            # print(t_us, vio_data[-1,0])
            if t_us < vio_data[0,0] or t_us > vio_data[-1,0]:
                return
            g_world = np.array([0, 0, 9.81]).reshape(-1,1)
            
            # # print(self.out_dir)
            # time_gt = vio_data[:, 0]
            # qxyzw_World_Device = vio_data[:, -10:-6]
            # # interp_funcs_quaternion = [interp1d(time_gt, qxyzw_World_Device[:, i], fill_value="extrapolate") for i in range(qxyzw_World_Device.shape[1])]
            # # interp_funcs_quaternion = [interp1d(time_gt, qxyzw_World_Device[:, i], fill_value=(qxyzw_World_Device[0, i], qxyzw_World_Device[-1, i]), bounds_error=False) for i in range(qxyzw_World_Device.shape[1])]
            # interp_funcs_quaternion = [interp1d(time_gt, qxyzw_World_Device[:, i]) for i in range(qxyzw_World_Device.shape[1])]

            # quaternion_gt = np.array([interp_func(t_us) for interp_func in interp_funcs_quaternion])
            # quaternion_gt = quaternion_gt / np.linalg.norm(quaternion_gt)
            # R_gt = Rotation.from_quat(quaternion_gt).as_matrix()  # world -> bd
            # R_gt = R_gt.T
            
            # R_gt = R_gt.T
            # print(R_gt.T)
            # print(R_filter)
            # acc_body = R_gt.T @ g_world
            R_filter, _, _, _, _ = self.filter.get_evolving_state()
            acc_body = R_filter.T @ g_world
            
            # ## apply calibration in gravity ##
            # gyro_body = np.zeros((3,1))
            # # print("acc_body : ", acc_body)
            # acc_body, _ = self.imu_calib.calibrate_raw(acc_body, gyro_body)
            # # print("acc_body_cal : ", acc_body)
            # print("acc_biascpst : ", acc_biascpst)
            acc_biascpst = acc_biascpst - acc_body
            # ## apply calibration in gravity ##
        ######### if using gravity compensated body acceleration ###
        
        if do_interpolation_of_imu:
            self._add_interpolated_imu_to_buffer(acc_biascpst, gyr_biascpst, t_us)
        
        if not self.use_riekf : 
            self.filter.propagate(
                acc_raw, gyr_raw, t_us, t_augmentation_us=t_augmentation_us
            )
            # print("propagate running!")
            
        else:
            if "so3" in self.vio_path:
                rotation_info = np.load("./../so3_local_data_bodyframe2/rpy_values.npy")
                rotation_rpy = rotation_info[i,:]
                m_b2bprime = Rotation.from_euler('xyz', rotation_rpy, degrees=False).as_matrix()
            else:
                m_b2bprime = np.eye(3)
            
            # assert t_augmentation_us is not None
            self.filter.propagate_riekf(
                # acc_raw, gyr_raw, t_us, m_b2bprime
                acc_raw, gyr_raw, t_augmentation_us, m_b2bprime
            )
            
        # filter update
        did_update = False
        if not self.use_riekf :
            if do_augmentation_and_update:
                did_update = self._process_update(t_us)
                # print("update running!")
                # print(t_us, self.next_aug_t_us)
                # plan next update/augmentation of state
                self.next_aug_t_us += self.dt_update_us
        else:
            if do_augmentation_and_update:
                # print("--- t_us when update : ", t_us)
                did_update = self._process_update_riekf(t_us, progress)   
                # print("update working!")
                # print(t_us, self.next_aug_t_us)
                
                self.next_aug_t_us += self.dt_update_us
                # self.next_aug_t_us += self.dt_interp_us  #이걸로 항상 고정

        # set last value memory to the current one
        self.last_t_us = t_us

        if t_us < self.t_us_before_next_interpolation:
            self.t_us_before_next_interpolation = t_us
            self.last_acc_before_next_interp_time = acc_biascpst
            self.last_gyr_before_next_interp_time = gyr_biascpst

        return did_update

    def _process_update_riekf(self, t_us, progress):
        logging.debug(f"Upd. @ {t_us * 1e-6} | Ns: {self.filter.state.N} ")
        # get update interval t_begin_us and t_end_us
        # if self.filter.state.N <= self.update_distance_num_clone:
        #     return False
        
        # t_end_us = t_us
        t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        # print("--- t_us from propagate : ", t_end_us)
        
        window_time = int(self.net_input_size/200*1e6)
        t_begin_us = t_end_us - window_time
        # t_begin_us = t_end_us - 1500000   #300hz
        # t_begin_us = t_end_us - 1000000   #200hz
        # t_begin_us = t_end_us - 100000  #20hz
        t_oldest_state_us = t_begin_us
        
        # t_oldest_state_us = self.filter.state.si_timestamps_us[
        #     self.filter.state.N - self.update_distance_num_clone - 1
        # ]
        # t_begin_us = t_oldest_state_us - self.dt_interp_us * self.past_data_size
        # t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        
        # If we do not have enough IMU data yet, just wait for next time
        # print(self.imu_buffer.net_t_us[0], self.imu_buffer.net_t_us[-1], self.imu_buffer.net_t_us.shape)
        # print(t_begin_us, t_end_us)
        # print(t_end_us < self.imu_buffer.net_t_us[0])
        
        # if t_begin_us < self.imu_buffer.net_t_us[0] or self.imu_buffer.net_t_us[-1] - self.imu_buffer.net_t_us[0] < 1100000 or self.imu_buffer.net_t_us[-1] < t_end_us:
        if t_begin_us < self.imu_buffer.net_t_us[0]:
        # if t_end_us > self.imu_buffer.net_t_us[-1]:
            return False
        # initialize with vio at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)
        assert t_begin_us <= t_oldest_state_us
        if self.debug_callback_get_meas:
            meas, meas_cov = self.debug_callback_get_meas(t_oldest_state_us, t_end_us)
        else:  # using network for measurements
            with open(self.json_path, 'r') as f:
                data_json = json.load(f)
            t_start_us1 = data_json["t_start_us"]
            t_end_us1 = data_json["t_end_us"]
            net_gyr_w, net_acc_w = self._get_imu_samples_for_network(
                t_begin_us, t_oldest_state_us, t_end_us
            )
            
            vio_data = np.load(self.vio_path)
            
            # if self.input_3:
            #     # input_4 = True
            #     input_4 = False
                
            #     num_data = net_gyr_w.shape[0]
            #     net_vel_body = np.empty((0, 3))
                
            #     net_ori_b2w = np.empty((0, 9))
                
            #     vio_data = np.load(self.vio_path)
            #     ts_data = np.array(self.filter.state.si_timestamps_us)
            #     v_data = np.squeeze(np.array(self.filter.state.si_vs))
            #     R_data = np.array(self.filter.state.si_Rs)
            #     # print(R_data[-1], self.filter.state.s_R)
            #     # assert False
                
            #     for i in range(num_data): 
            #         timestamp = t_begin_us + 5000*i
            #         if ts_data.size < 3 or v_data.shape[0] < 200:
            #             print("using gt", ts_data.size, v_data.shape[0])
            #             index = round(vio_data[:, 0].shape[0] * ((timestamp - t_start_us1) / (t_end_us1 - t_start_us1)))
                        
            #             time = vio_data[:, 0]
            #             velocity = vio_data[:, -3:]
            #             interp_funcs = [interp1d(time, velocity[:, i], fill_value="extrapolate") for i in range(velocity.shape[1])]
            #             v_past2 = np.array([interp_func(timestamp) for interp_func in interp_funcs]).reshape(3,-1)
                        
            #             # if index >= len(vio_data):
            #             #     v_past2 = vio_data[-1,-3:]  # using imu0_resampled.npy
            #             #     if input_4:
            #             #         R_past = vio_data[-1,-10:-6]
            #             #         R_past_matrix = Rotation.from_quat(R_past).as_matrix()
            #             #         ori_column1 = R_past_matrix[:,0].reshape(3)
            #             #         ori_column2 = R_past_matrix[:,1].reshape(3)
            #             #         ori_column3 = R_past_matrix[:,2].reshape(3)
            #             #         stack_ori = np.hstack((ori_column1, ori_column2, ori_column3))

            #             # else:
            #             #     v_past2 = vio_data[index,-3:]  # using imu0_resampled.npy
            #             #     if input_4:
            #             #         R_past = vio_data[index,-10:-6]
            #             #         R_past_matrix = Rotation.from_quat(R_past).as_matrix()
            #             #         ori_column1 = R_past_matrix[:,0].reshape(3)
            #             #         ori_column2 = R_past_matrix[:,1].reshape(3)
            #             #         ori_column3 = R_past_matrix[:,2].reshape(3)
            #             #         stack_ori = np.hstack((ori_column1, ori_column2, ori_column3))
            #             v_at_timestamp_bd = v_past2
            #         else:
            #             # print("i :  ", i)
            #             v_at_timestamp = v_data[5*(-200+i), :]
            #             v_at_timestamp_bd = v_at_timestamp
                        
            #             index = round(vio_data[:, 0].shape[0] * ((timestamp - t_start_us1) / (t_end_us1 - t_start_us1)))
            #             if index >= len(vio_data):
            #                 v_past2 = vio_data[-1,-3:]  # using imu0_resampled.npy
            #             else:
            #                 v_past2 = vio_data[index,-3:]
            #             v_at_timestamp_bd_gt = v_past2
            #             # v_at_timestamp_bd= v_at_timestamp_bd_gt
            #             # print("v_at_timestamp_bd_gt : ", v_at_timestamp_bd_gt)
            #             # print("v_at_timestamp_bd : ", v_at_timestamp_bd)
                        
            #             if input_4:
            #                 R_at_timestamp = R_data[5*(-200+i), :, :]
            #                 ori_column1 = R_at_timestamp[:,0].reshape(3)
            #                 ori_column2 = R_at_timestamp[:,1].reshape(3)
            #                 ori_column3 = R_at_timestamp[:,2].reshape(3)
            #                 stack_ori = np.hstack((ori_column1, ori_column2, ori_column3))

            #         net_vel_body = np.vstack((net_vel_body, v_at_timestamp_bd.reshape(3)))
            #         if input_4:
            #             net_ori_b2w = np.vstack((net_ori_b2w, stack_ori.reshape(9)))
                        
            # else:
            #     net_vel_body = None                        
               
            # meas, meas_cov = self.meas_source.get_displacement_measurement(
            #     net_gyr_w, net_acc_w, net_vel_body, net_ori_b2w, self.input_3, input_4
            # )
            
            # #2. using R - gt
            # # print(t_end_us, t_start_us1 , t_end_us1)
            # # index = round(vio_data[:, 0].shape[0] * ((t_end_us - t_start_us1) / (t_end_us1 - t_start_us1)))
            # # # print((t_end_us - t_start_us1) / (t_end_us1 - t_start_us1)*100)
            # # if index >= len(vio_data):
            # #     meas = vio_data[-1,-3:].reshape(3,-1)
            # #     meas_cov = meas
            # # else:
            # #     meas = vio_data[index,-3:].reshape(3,-1)
            # #     meas_cov = meas
            
            # time = vio_data[:, 0]
            # velocity = vio_data[:, -3:]
            # interp_funcs = [interp1d(time, velocity[:, i], fill_value="extrapolate") for i in range(velocity.shape[1])]
            # meas_gt = np.array([interp_func(t_end_us) for interp_func in interp_funcs]).reshape(3,-1)
            # meas_cov = meas_gt
            # print(meas.reshape(-1)-net_vel_body[-1].reshape(-1))
            
            ts_data = np.array(self.filter.state.si_timestamps_us)
            v_data = np.squeeze(np.array(self.filter.state.si_vs))
            
            # print(ts_data.size , v_data.shape)
            # print(ts_data[5*(-199)], ts_data[-1], t_end_us)
            
            time = vio_data[:, 0]
            velocity = vio_data[:, -3:]
            qxyzw_World_Device = vio_data[:, -10:-6]
            interp_funcs = [interp1d(time, velocity[:, i], fill_value="extrapolate") for i in range(velocity.shape[1])]
            
            interp_funcs_quaternion = [interp1d(time, qxyzw_World_Device[:, i], fill_value="extrapolate") for i in range(qxyzw_World_Device.shape[1])]
            quaternion_gt = np.array([interp_func(t_end_us) for interp_func in interp_funcs_quaternion])
            quaternion_gt = quaternion_gt / np.linalg.norm(quaternion_gt)
            R_gt = Rotation.from_quat(quaternion_gt).as_matrix()  # bd -> world
            meas_gt = np.array([interp_func(t_end_us) for interp_func in interp_funcs]).reshape(3,-1)
            
            meas = meas_gt
            meas_cov = np.eye(3)

            # if ts_data.size < 3 or v_data.shape[0] < self.update_freq:
            # if ts_data.size < 3 or v_data.shape[0] < self.net_input_size:
            if ts_data.size < 3 or v_data.shape[0] < self.net_input_size and self.input_3:
                # print(ts_data.size, v_data.shape[0])
                # print("using gt")
                meas = np.array([interp_func(t_end_us) for interp_func in interp_funcs]).reshape(3,-1)
                meas_cov = np.eye(3)
            else:
                net_vel_body = np.empty((0, 3))
                num_data = net_gyr_w.shape[0]
                # print(ts_data[5*(-199)], ts_data[-1], t_end_us)
                # for i in range(num_data): 
                #     timestamp = t_begin_us + 5000*i
                #     print(ts_data[5*(-199+i)-1], t_end_us)
                #     v_at_timestamp = v_data[5*(-199+i)-1, :]
                #     v_at_timestamp_bd = v_at_timestamp
                #     net_vel_body = np.vstack((net_vel_body, v_at_timestamp_bd.reshape(3)))
                if self.update_freq == 1000:
                    step = 5
                    indices = np.arange(-199, 1) * step + (v_data.shape[0] - 1)
                    # print(v_data.shape[0])
                    # print(indices)
                    # assert False
                    timestamps = t_begin_us + np.arange(num_data) * 5000
                    assert np.all(ts_data[indices] <= t_end_us), "Timestamp data out of range"
                    net_vel_body = v_data[indices, :]
                    
                    # print(net_vel_body.shape)
                    t_data = ts_data[indices]
                    # print(t_data[-10:])
                    # assert False
                elif self.update_freq == 200:
                    step = 1
                    indices = np.arange(-19, 1) * step + (v_data.shape[0] - 1)
                    t_data = ts_data[indices]
                    # print(v_data.shape[0])
                    # print(indices)
                    assert np.all(ts_data[indices] <= t_end_us), "Timestamp data out of range"
                    ##1. use filter velocity
                    net_vel_body = v_data[indices, :]
                    # ##2. use gt velocity
                    # net_vel_body = np.array([interp_func(t_data) for interp_func in interp_funcs]).T
                    
                    # print(net_vel_body.shape)
                    # print(t_data[-10:], t_end_us)
                    # assert False
                elif self.update_freq == 20:
                    if self.input_3:
                        step = 1
                        indices = np.arange(-(self.net_input_size-1), 1) * step + (v_data.shape[0] - 1)
                        t_data = ts_data[indices]
                        # print(v_data.shape[0])
                        # print(indices)
                        assert np.all(ts_data[indices] <= t_end_us), "Timestamp data out of range"
                        
                        ##1. use filter velocity
                        net_vel_body = v_data[indices, :]
                        
                        # # ##2. use gt velocity
                        # net_vel_body = np.array([interp_func(t_data) for interp_func in interp_funcs]).T
                        
                        # ##3. combination of using measurement after gt
                        # if progress < 0.5 : 
                        #     net_vel_body = np.array([interp_func(t_data) for interp_func in interp_funcs]).T
                        # else:
                        #     net_vel_body = v_data[indices, :]
                        # # print("progress : ", progress)
                            
                        
                        # #4. using filter velocity-world
                        # vio_data = np.load(self.vio_path)
                        # net_vel_body = v_data[indices, :]
                        
                        # time = vio_data[:, 0]
                        # qxyzw_World_Device = vio_data[:, -10:-6]
                        # interp_funcs_quaternion = [interp1d(time, qxyzw_World_Device[:, i], fill_value="extrapolate") for i in range(qxyzw_World_Device.shape[1])]
                        # t_data = ts_data[indices]
                        
                        # quaternions_gt = np.array([[interp_func(t) for interp_func in interp_funcs_quaternion] for t in t_data])
                        # quaternions_gt = np.array([q / np.linalg.norm(q) for q in quaternions_gt])
                        # Rwfb = np.array([Rotation.from_quat(quaternion).as_matrix() for quaternion in quaternions_gt])
                        # net_vel_body = np.einsum("tij,tj->ti", Rwfb, net_vel_body)  # N x 3
                        
                        
                        # print(net_vel_body.shape)
                        # print(self.imu_buffer.net_t_us[-10:], t_end_us) #diff : 5000 
                        # print(t_data[-10:], t_end_us)   #diff : 5000 
                
                ## comment out to put meas = meas_gt ###
                if "gtv" not in self.out_dir:
                    net_ori_b2w = None
                    input_4 = False
                    meas, meas_cov = self.meas_source.get_displacement_measurement(
                        net_gyr_w, net_acc_w, net_vel_body, net_ori_b2w, self.input_3, input_4
                    )
                    # print(meas.reshape(-1)-meas_gt.reshape(-1), meas_gt.reshape(-1))
                    
                ## comment out to put meas = meas_gt ###
                    
                    # ##put gt_v_z value
                    # # print(meas.reshape(-1))
                    # # print(meas.reshape(-1)-meas_gt.reshape(-1))
                    # meas[-1,0] = meas_gt[-1,0]
                    # # assert False
                    # ##put gt_v_z value
                # print(meas)
                
                # meas += ((np.random.rand(3, 1) - 0.5) * 0.02 * 9)
                
                # print(meas.reshape(-1)-meas_gt.reshape(-1))
            # print(t_end_us, ts_data)
            # print(meas.reshape(-1)-meas_gt.reshape(-1), meas_gt.reshape(-1))
            # print(interpolated_velocity.reshape(3,-1) - meas)
        
        if not self.body_frame:
            #when using world velocity measurement(estimation)
            meas = R_gt.T @ meas #world to body
        
        #for equiv-cov
        # meas_cov = R_data[-1] @ meas_cov @ R_data[-1].T
        # filter update
        self.filter.update_riekf(meas, meas_cov, t_oldest_state_us, t_end_us)
        self.has_done_first_update = True
        # marginalization of all past state with timestamp before or equal ts_oldest_state

        
        # Get the index of the closest timestamp
        # print(" self.filter.state.si_timestamps_us.index : ", self.filter.state.si_timestamps_us)
        # oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        # oldest_idx = self.filter.state.si_timestamps_us.index(1659265477)
        # timestamps_array = np.array(self.filter.state.si_timestamps_us)
        # differences = np.abs(timestamps_array - t_oldest_state_us)
        # oldest_idx = np.argmin(differences)
        
        # cut_idx = oldest_idx
        # logging.debug(f"marginalize {cut_idx}")
        # self.filter.marginalize(cut_idx)
        
        self.imu_buffer.throw_data_before(t_begin_us)
        return True
    
    def _process_update(self, t_us):
        logging.debug(f"Upd. @ {t_us * 1e-6} | Ns: {self.filter.state.N} ")
        # get update interval t_begin_us and t_end_us
        # print("code running?", self.filter.state.N, self.update_distance_num_clone)
        if self.filter.state.N <= self.update_distance_num_clone:
            return False
        t_oldest_state_us = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1
        ]
        t_begin_us = t_oldest_state_us - self.dt_interp_us * self.past_data_size
        t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        
        # print("t_begin_us : ", t_begin_us, t_end_us - t_begin_us)
        
        # print(self.imu_buffer.net_t_us[0], self.imu_buffer.net_t_us[-1], self.imu_buffer.net_t_us.shape)
        # print(t_begin_us, t_end_us)
        # 3436084207 3437184207 (221,)
        # 3436184207 3437184207
        # assert False
        
        # If we do not have enough IMU data yet, just wait for next time
        
        if t_begin_us < self.imu_buffer.net_t_us[0]:
            return False
        # initialize with vio at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)
        assert t_begin_us <= t_oldest_state_us
        if self.debug_callback_get_meas:
            meas, meas_cov = self.debug_callback_get_meas(t_oldest_state_us, t_end_us)
        else:  # using network for measurements
            net_gyr_w, net_acc_w = self._get_imu_samples_for_network(
                t_begin_us, t_oldest_state_us, t_end_us
            )
            # print(self.imu_buffer.net_t_us[-10:]) #diff : 10000 
            meas, meas_cov = self.meas_source.get_displacement_measurement(
                net_gyr_w, net_acc_w
            )
            
        # filter update
        self.filter.update(meas, meas_cov, t_oldest_state_us, t_end_us)
        self.has_done_first_update = True
        # marginalization of all past state with timestamp before or equal ts_oldest_state
        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        cut_idx = oldest_idx
        logging.debug(f"marginalize {cut_idx}")
        self.filter.marginalize(cut_idx)
        self.imu_buffer.throw_data_before(t_begin_us)
        return True

    def _add_interpolated_imu_to_buffer(self, acc_biascpst, gyr_biascpst, t_us):
        self.imu_buffer.add_data_interpolated(
            self.t_us_before_next_interpolation,
            t_us,
            self.last_gyr_before_next_interp_time,
            gyr_biascpst,
            self.last_acc_before_next_interp_time,
            acc_biascpst,
            self.next_interp_t_us,
        )
        
        # self.next_interp_t_us += int(self.dt_interp_us / 5)
        if self.update_freq == 20:
            # self.next_interp_t_us += int(self.dt_interp_us/5)
            self.next_interp_t_us += self.dt_interp_us
        # elif self.update_freq == 1000:
        #     self.next_interp_t_us += int(self.dt_interp_us/5)
        else:
            self.next_interp_t_us += self.dt_update_us
            
            
