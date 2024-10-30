#!/usr/bin/env python3

import os

import numpy as np
import progressbar
from dataloader.data_io import DataIO
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from tracker.imu_calib import ImuCalib
from tracker.imu_tracker import ImuTracker
from utils.o3d_visualizer import O3dVisualizer
from utils.dotdict import dotdict
from utils.logging import logging

import re
import time

class ImuTrackerRunner:
    """
    This class is responsible for going through a dataset, feed imu tracker and log its result
    """

    def __init__(self, args, dataset):
        # initialize data IO
        self.input = DataIO()
        self.input.load_all(dataset, args)
        self.input.load_vio(dataset, args)
        self.visualizer = None
        if args.visualize:
            vio_ghost = np.concatenate([
                self.input.vio_ts_us[:,None], self.input.vio_rq, self.input.vio_p
            ], axis=1)
            self.visualizer = O3dVisualizer(vio_ghost)

        # log file initialization
        outdir = os.path.join(args.out_dir, dataset)
        if os.path.exists(outdir) is False:
            os.mkdir(outdir)
        outfile = os.path.join(outdir, args.out_filename)
        if os.path.exists(outfile):
            if not args.erase_old_log:
                logging.warning(f"{outfile} already exists, skipping")
                raise FileExistsError
            else:
                os.remove(outfile)
                logging.warning("previous log file erased")

        self.outfile = os.path.join(outdir, args.out_filename)
        self.f_state = open(outfile, "w")
        self.f_debug = open(os.path.join(outdir, "debug.txt"), "w")
        logging.info(f"writing to {outfile}")

        if "sim" not in args.root_dir:
            imu_calib = ImuCalib.from_offline_calib(dataset, args)
            self.imu_calib = imu_calib
        else:
            imu_calib = None

        filter_tuning = dotdict(
            {
                "g_norm": args.g_norm,
                "sigma_na": args.sigma_na,
                "sigma_ng": args.sigma_ng,
                "ita_ba": args.ita_ba,
                "ita_bg": args.ita_bg,
                "init_attitude_sigma": args.init_attitude_sigma,  # rad
                "init_yaw_sigma": args.init_yaw_sigma,  # rad
                "init_vel_sigma": args.init_vel_sigma,  # m/s
                "init_pos_sigma": args.init_pos_sigma,  # m
                "init_bg_sigma": args.init_bg_sigma,  # rad/s
                "init_ba_sigma": args.init_ba_sigma,  # m/s^2
                "meascov_scale": args.meascov_scale,
                "const_cov_val_x": args.const_cov_val_x,  # sigma^2
                "const_cov_val_y": args.const_cov_val_y,  # sigma^2
                "const_cov_val_z": args.const_cov_val_z,  # sigma^2
                "add_sim_meas_noise": args.add_sim_meas_noise,
                "sim_meas_cov_val": args.sim_meas_cov_val,
                "sim_meas_cov_val_z": args.sim_meas_cov_val_z,
                "mahalanobis_fail_scale": args.mahalanobis_fail_scale,
                "use_const_cov": eval(args.use_const_cov),
            }
        )
        if "sim" not in args.root_dir:
            match = re.search(r'/(\d{15,16})/', self.outfile)
            replace_string = "./../so3_local_data_bodyframe/146734859523827/imu0_resampled.npy"
            
        else:
            match = re.search(r'/(\d{1,3})/', self.outfile)
            replace_string = "./../june_sim_imu_longerseq/1/imu0_resampled.npy"
            
        if match:
            extracted_string = match.group(1)
            # print("Extracted string:", extracted_string)
        else:
            extracted_string = None
            print("No match found")

        if extracted_string:
            if "sim" not in args.root_dir:
                # vio_path = re.sub(r'/\d{15,16}/', f'/{extracted_string}/', replace_string)
                vio_path = args.root_dir + extracted_string + "/imu0_resampled.npy" 
            else:
                # vio_path = re.sub(r'/(?:[1-9]|[1-9][0-9]|[1-3][0-9][0-9]|400)/', f'/{extracted_string}/', replace_string)
                # vio_path = re.sub(r'/\d{1,3}/', f'/{extracted_string}/', replace_string)
                vio_path = args.root_dir + extracted_string + "/imu0_resampled.npy" 
                
            # print("Generated vio_path:", vio_path)
        else:
            print("Cannot generate vio_path as no extracted string was found")
            
        print("vio_path : ", vio_path)
        print("extracted_substring : ", extracted_string)
        json_path = os.path.join(args.root_dir, extracted_string + "/imu0_resampled_description.json")
        print("json_path : ", json_path, " / body_frame : ", args.body_frame, "use_const_cov of TLIO : ", args.use_const_cov)
        self.vio_path = vio_path
        
        # ImuTracker object
        self.tracker = ImuTracker(
            model_path=args.model_path,
            model_param_path=args.model_param_path,
            update_freq=args.update_freq,
            filter_tuning_cfg=filter_tuning,
            imu_calib=imu_calib,
            #force_cpu=True,
            vio_path = vio_path,
            json_path = json_path,
            body_frame = eval(args.body_frame),
            use_riekf = eval(args.use_riekf),
            input_3 = eval(args.input_3),
            out_dir = args.out_dir,
        )

        # output
        self.log_output_buffer = None

    def __del__(self):
        try:
            self.f_state.close()
            self.f_debug.close()
        except Exception as e:
            logging.exception(e)

    def add_data_to_be_logged(self, ts, acc, gyr, with_update):
        # filter data logger
        R, v, p, ba, bg = self.tracker.filter.get_evolving_state()
        ba = ba
        bg = bg
        Sigma, Sigma15 = self.tracker.filter.get_covariance()
        sigmas = np.diag(Sigma15).reshape(15, 1)
        sigmasyawp = self.tracker.filter.get_covariance_yawp().reshape(16, 1)
        inno, meas, pred, meas_sigma, inno_sigma = self.tracker.filter.get_debug()

        if not with_update:
            inno *= np.nan
            meas *= np.nan
            pred *= np.nan
            meas_sigma *= np.nan
            inno_sigma *= np.nan

        ts_temp = ts.reshape(1, 1)
        temp = np.concatenate(
            [
                v,
                p,
                ba,
                bg,
                acc,
                gyr,
                ts_temp,
                sigmas,
                inno,
                meas,
                pred,
                meas_sigma,
                inno_sigma,
                sigmasyawp,
            ],
            axis=0,
        )
        vec_flat = np.append(R.ravel(), temp.ravel(), axis=0)
        if self.log_output_buffer is None:
            self.log_output_buffer = vec_flat
        else:
            self.log_output_buffer = np.vstack((self.log_output_buffer, vec_flat))

        if self.log_output_buffer.shape[0] > 100:
            np.savetxt(self.f_state, self.log_output_buffer, delimiter=",")
            self.log_output_buffer = None

    def run_tracker(self, args, iter_num):
        # initialize debug callbacks
        def initialize_with_vio_at_first_update(this):
            logging.info(
                f"Initialize filter from vio state just after time {this.last_t_us*1e-6}"
            )
            self.reset_filter_state_from_vio(this, args.input_3, args.use_riekf)

        def initialize_at_first_update(this):
            logging.info(f"Re-initialize filter at first update")
            self.reset_filter_state_pv(args.input_3, args.use_riekf)

        if eval(args.initialize_with_vio):
            self.tracker.callback_first_update = initialize_with_vio_at_first_update
        else:
            self.tracker.callback_first_update = initialize_at_first_update

        if args.use_vio_meas:
            self.tracker.debug_callback_get_meas = lambda t0, t1: self.input.get_meas_from_vio(
                t0, t1
            )

        # Loop through the entire dataset and feed the data to the imu tracker
        n_data = self.input.dataset_size
        start_index = int(0.1 * n_data)
        end_index = int(0.95 * n_data)
        vio_data = np.load(self.vio_path)
        
        if "short_seq" in self.outfile :
            imu_range = range(start_index, end_index)
        else:
            imu_range = range(n_data)
            
        for i in progressbar.progressbar(imu_range, redirect_stdout=True):

            start_time = time.time()
            debug_time = 0
            process_time = 0
            vis_time = 0
            init_time = 0
            
            # print("i/n : ", i/n_data * 100)
            # obtain next raw IMU measurement from data loader
            ts, acc_raw, gyr_raw = self.input.get_datai(i)
            # print(ts.shape, acc_raw.shape)        (1), (3,1)
            # assert False
            t_us = int(ts * 1e6)
            
            get_data_time = time.time() - start_time
            
            # if "comp" in self.outfile :
            #     use_gravity_comp_csv = True
            # else:
            #     use_gravity_comp_csv = False
            
            # if use_gravity_comp_csv:
            #     if ts < self.input.vio_ts[0]:
            #             continue

            #     time_gt = vio_data[:, 0]
            #     qxyzw_World_Device = vio_data[:, -10:-6]
            #     # interp_funcs_quaternion = [interp1d(time_gt, qxyzw_World_Device[:, i], fill_value="extrapolate") for i in range(qxyzw_World_Device.shape[1])]
            #     # interp_funcs_quaternion = [interp1d(time_gt, qxyzw_World_Device[:, i], fill_value=(qxyzw_World_Device[0, i], qxyzw_World_Device[-1, i]), bounds_error=False) for i in range(qxyzw_World_Device.shape[1])]
            #     interp_funcs_quaternion = [interp1d(time_gt, qxyzw_World_Device[:, i]) for i in range(qxyzw_World_Device.shape[1])]

            #     quaternion_gt = np.array([interp_func(t_us) for interp_func in interp_funcs_quaternion])
            #     quaternion_gt = quaternion_gt / np.linalg.norm(quaternion_gt)
            #     R_gt = Rotation.from_quat(quaternion_gt).as_matrix()  # bd -> world
            #     g_world = np.array([0, 0, 9.81]).reshape(-1,1)
            #     R_gt = R_gt.T
            #     acc_body = R_gt.T @ g_world
                
            #     # ## apply calibration in gravity ##
            #     # gyro_body = np.zeros((3,1))
            #     # # print("acc_body : ", acc_body)
            #     # acc_body, _ = self.imu_calib.calibrate_raw(acc_body, gyro_body)
            #     # # print("acc_body_cal : ", acc_body)
            #     # ## apply calibration in gravity ##
                
            #     acc_raw = acc_raw - acc_body
                
            # cheat a bit with filter state in case for debugging
            if args.debug_using_vio_ba:
                
                debug_start_time = time.time()
                
                vio_ba = interp1d(self.input.vio_calib_ts, self.input.vio_ba, axis=0)(
                    ts
                )
                vio_bg = interp1d(self.input.vio_calib_ts, self.input.vio_bg, axis=0)(
                    ts
                )
                self.tracker.filter.state.s_ba = np.atleast_2d(vio_ba).T
                self.tracker.filter.state.s_bg = np.atleast_2d(vio_bg).T
                
                debug_time = time.time() - debug_start_time

            if self.tracker.filter.initialized:
                
                process_start_time = time.time()
                
                progress = i / n_data
                did_update = self.tracker.on_imu_measurement(t_us, gyr_raw, acc_raw, iter_num, progress)
                process_time = time.time() - process_start_time
                self.add_data_to_be_logged(
                    ts,
                    self.tracker.last_acc_before_next_interp_time,  # beware when imu drops, it might not be what you want here
                    self.tracker.last_gyr_before_next_interp_time,  # beware when imu drops, it might not be what you want here
                    with_update=did_update,
                )
                
                
                if i % 100 == 0 and self.visualizer is not None:
                    T_World_Imu = np.eye(4)
                    T_World_Imu[:3,:3] = self.tracker.filter.state.s_R
                    T_World_Imu[:3,3:4] = self.tracker.filter.state.s_p
                    self.visualizer.update(
                        t_us, 
                        {"tlio": T_World_Imu},    
                        {"tlio": [T_World_Imu[:3,3]]},    
                    )
                    
                    vis_time = time.time() - vis_start_time
                    
            else:
                
                init_start_time = time.time()
                
                # initialize to gt state R,v,p and offline calib
                if not eval(args.initialize_with_vio):
                    self.tracker.on_imu_measurement(t_us, gyr_raw, acc_raw)
                else:
                    if eval(args.initialize_with_offline_calib):
                        # init_ba = self.tracker.icalib.accelBias
                        # init_bg = self.tracker.icalib.gyroBias
                        init_ba = np.zeros((3, 1))
                        init_bg = np.zeros((3, 1))
                    else:
                        init_ba = np.zeros((3, 1))
                        init_bg = np.zeros((3, 1))

                    if ts < self.input.vio_ts[0]:
                        continue
                    print(ts)
                    vio_p = interp1d(self.input.vio_ts, self.input.vio_p, axis=0)(ts)
                    vio_v = interp1d(self.input.vio_ts, self.input.vio_v, axis=0)(ts)
                    vio_eul = interp1d(self.input.vio_ts, self.input.vio_eul, axis=0)(
                        ts
                    )
                    
                    # print(ts, self.input.vio_ts[0],self.input.vio_ts[-1])
                    # print(init_ba, init_bg)
                    # print(vio_eul, self.input.vio_eul[0])
                    # assert False
                    vio_R = Rotation.from_euler(
                        "xyz", vio_eul, degrees=True
                    ).as_matrix()
                    vio_R_0 = Rotation.from_euler(
                        "xyz", self.input.vio_eul[0], degrees=True
                    ).as_matrix()
                    self.tracker.init_with_state_at_time(
                        t_us,
                        vio_R,
                        # vio_R_0,
                        np.atleast_2d(vio_v).T,
                        np.atleast_2d(vio_p).T,
                        init_ba,
                        init_bg,
                    )
                    
                init_time = time.time() - init_start_time
            # print(f"Iteration {i}: get_data_time={get_data_time:.4f}s, "
            #         f"vis_time={vis_time:.4f}s, init_time={init_time:.4f}s, "
            #         f"debug_time={debug_time:.4f}s, process_time={process_time:.4f}s")
            
        self.f_state.close()
        self.f_debug.close()
        if args.save_as_npy:
            # actually convert the .txt to npy to be more storage friendly
            states = np.loadtxt(self.outfile, delimiter=",")
            np.save(self.outfile + ".npy", states)
            os.remove(self.outfile)

    def scale_raw_dynamic(self, t, acc, gyr):
        """ This scale with gt data, for debug purpose only"""
        idx = np.searchsorted(self.input.vio_calib_ts, t)
        acc_cal = np.dot(self.input.vio_accelScaleInv[idx, :, :], acc)
        gyr_cal = np.dot(self.input.vio_gyroScaleInv[idx, :, :], gyr) - np.dot(
            self.input.vio_gyroGSense[idx, :, :], acc
        )
        return acc_cal, gyr_cal

    def reset_filter_state_from_vio(self, this: ImuTracker, input_3 = False, use_riekf = False):
        """ This reset the filter state from vio state as found in input """
        # compute from vio
        inp = self.input  # for convenience
        state = this.filter.state  # for convenience
        vio_ps = []
        vio_Rs = []
        vio_vs = []
        
        for i, t_init_us in enumerate(state.si_timestamps_us):
            t_init_s = t_init_us * 1e-6
            ps = np.atleast_2d(interp1d(inp.vio_ts, inp.vio_p, axis=0)(t_init_s)).T
            vio_ps.append(ps)
            vio_eul = interp1d(inp.vio_ts, inp.vio_eul, axis=0)(t_init_s)
            vio_Rs.append(Rotation.from_euler("xyz", vio_eul, degrees=True).as_matrix())
            if input_3 or use_riekf:
                vio_v = np.atleast_2d(interp1d(inp.vio_ts, inp.vio_v, axis=0)(t_init_s)).T
                vio_vs.append(vio_v)  # Append velocity to the list

        ts = state.s_timestamp_us * 1e-6
        vio_p = np.atleast_2d(interp1d(inp.vio_ts, inp.vio_p, axis=0)(ts)).T
        vio_v = np.atleast_2d(interp1d(inp.vio_ts, inp.vio_v, axis=0)(ts)).T
        vio_eul = interp1d(inp.vio_ts, inp.vio_eul, axis=0)(ts)
        vio_R = Rotation.from_euler("xyz", vio_eul, degrees=True).as_matrix()
        
        this.filter.reset_state_and_covariance(
            vio_Rs, vio_ps, vio_vs, vio_R, vio_v, vio_p, state.s_ba, state.s_bg, use_riekf
            # vio_Rs, vio_ps, vio_R, vio_v, vio_p, state.s_ba, state.s_bg
        )

    def reset_filter_state_pv(self, input_3 = False, use_riekf = False):
        """ Reset filter states p and v with zeros """
        state = self.tracker.filter.state
        ps = []
        vs = []
        for i in state.si_timestamps_us:
            ps.append(np.zeros((3, 1)))
            if input_3 or use_riekf:
                vs.append(np.zeros((3, 1)))
        p = np.zeros((3, 1))
        v = np.zeros((3, 1))
        self.tracker.filter.reset_state_and_covariance(
            state.si_Rs, ps, vs, state.s_R, v, p, state.s_ba, state.s_bg, use_riekf
        )