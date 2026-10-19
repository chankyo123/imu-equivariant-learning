#!/usr/bin/env python3

import os

import numpy as np
import progressbar
from tqdm import tqdm
from dataloader.data_io import DataIO
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from tracker.imu_calib import ImuCalib
from tracker.imu_tracker import ImuTracker
# from utils.o3d_visualizer import O3dVisualizer
from utils.dotdict import dotdict
from utils.logging import logging

import re


class TestPerturbHook:
    """Inject a per-sample rotation into the raw IMU stream before the EKF sees it.

    type='gravity': fixed horizontal-axis rotation per sequence (uniform magnitude
        in [0, deg]) -- models a constant sensor-mount tilt that breaks any
        gravity-equivariance assumption.
    type='lfa': smooth random-walk rotation rvec(t) scaled so per-sequence peak
        magnitude equals `deg`. Models time-varying calibration noise.
    type='so3': fixed random SO(3) rotation of magnitude exactly `deg` per sequence.
    type='none': pass-through (default; normal runs untouched).
    """
    def __init__(self, perturb_type, perturb_deg, seed, dataset_name):
        self.type = perturb_type
        self.deg = float(perturb_deg)
        # Deterministic seed per (type, deg, dataset) so reruns reproduce.
        h = (hash((perturb_type, round(self.deg, 4), str(dataset_name))) & 0xFFFFFFFF)
        self.rng = np.random.default_rng((seed + h) & 0xFFFFFFFF)
        self.R_fixed = None
        self.rvec_cache = None
        if self.type == "gravity" and self.deg > 0:
            theta = self.rng.uniform(0.0, np.deg2rad(self.deg))
            phi = self.rng.uniform(0.0, 2 * np.pi)
            axis = np.array([np.cos(phi), np.sin(phi), 0.0]) * theta
            self.R_fixed = Rotation.from_rotvec(axis).as_matrix()
        elif self.type == "so3" and self.deg > 0:
            z = self.rng.standard_normal(3)
            z /= np.linalg.norm(z) + 1e-12
            theta = np.deg2rad(self.deg)
            self.R_fixed = Rotation.from_rotvec(z * theta).as_matrix()

    def precompute_lfa(self, N):
        if self.type != "lfa" or self.deg <= 0 or N <= 1:
            return
        max_rad = np.deg2rad(self.deg)
        inc = self.rng.standard_normal((N - 1, 3))
        rvec = np.zeros((N, 3))
        for t in range(N - 2, -1, -1):
            rvec[t] = rvec[t + 1] + inc[t]
        max_norm = float(np.linalg.norm(rvec, axis=1).max())
        if max_norm > 1e-8:
            rvec = rvec * (max_rad / max_norm)
        rvec[-1] = 0.0
        self.rvec_cache = rvec

    def apply(self, i, acc, gyr):
        if self.type == "none" or self.deg <= 0:
            return acc, gyr
        if self.type in ("gravity", "so3"):
            R = self.R_fixed
        elif self.type == "lfa":
            if self.rvec_cache is None:
                return acc, gyr
            R = Rotation.from_rotvec(self.rvec_cache[i]).as_matrix()
        else:
            return acc, gyr
        return R @ acc, R @ gyr


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
            print('Open3d error not visualizing!')
            vio_ghost = np.concatenate([
                self.input.vio_ts_us[:,None], self.input.vio_rq, self.input.vio_p
            ], axis=1)
            # self.visualizer = O3dVisualizer(vio_ghost)

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

        imu_calib = ImuCalib.from_offline_calib(dataset, args)

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
                "use_const_cov": args.use_const_cov,
                "const_cov_val_x": args.const_cov_val_x,  # sigma^2
                "const_cov_val_y": args.const_cov_val_y,  # sigma^2
                "const_cov_val_z": args.const_cov_val_z,  # sigma^2
                "add_sim_meas_noise": args.add_sim_meas_noise,
                "sim_meas_cov_val": args.sim_meas_cov_val,
                "sim_meas_cov_val_z": args.sim_meas_cov_val_z,
                "mahalanobis_fail_scale": args.mahalanobis_fail_scale,
            }
        )
        # Try multiple regex patterns to extract the sequence name from the outfile path:
        #   TLIO: 15-16 digit numeric IDs
        #   sim:  1-3 digit IDs
        #   RIDI: alphanumeric like dan_bag1, hang_handheld_test1
        vio_path = None
        for pattern in [r'/(\d{15,16})/', r'/(\d{1,3})/', r'/([a-zA-Z_]+\d*[a-zA-Z_]*\d*)/']:
            match = re.search(pattern, self.outfile)
            if match:
                extracted_string = match.group(1)
                candidate = os.path.join(args.root_dir.rstrip("/"), extracted_string, "imu0_resampled.npy")
                if os.path.exists(candidate):
                    vio_path = candidate
                    break
        if vio_path is None:
            # Last-resort fallback: `dataset` is the sequence name passed to this runner
            vio_path = os.path.join(args.root_dir.rstrip("/"), dataset, "imu0_resampled.npy")
            print(f"vio_path fallback to: {vio_path}")
        # ImuTracker object
        self.tracker = ImuTracker(
            model_path=args.model_path,
            model_param_path=args.model_param_path,
            update_freq=args.update_freq,
            filter_tuning_cfg=filter_tuning,
            imu_calib=imu_calib,
            force_cpu=getattr(args, "cpu", False),
            vio_path = vio_path,
            secondary_model_path=getattr(args, "secondary_model_path", None),
            secondary_arch=getattr(args, "secondary_arch", None),
            hybrid_mode=getattr(args, "hybrid_mode", None),
            yaw_aligned=getattr(args, "yaw_aligned", True),
        )

        # Test-time perturbation hook (applies to raw IMU before the tracker sees it; 'none' by default)
        self.perturb_hook = TestPerturbHook(
            perturb_type=getattr(args, "test_perturb_type", "none"),
            perturb_deg=getattr(args, "test_perturb_deg", 0.0),
            seed=getattr(args, "test_perturb_seed", 0),
            dataset_name=dataset,
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
            self.reset_filter_state_from_vio(this)

        def initialize_at_first_update(this):
            logging.info(f"Re-initialize filter at first update")
            self.reset_filter_state_pv()

        if args.initialize_with_vio:
            self.tracker.callback_first_update = initialize_with_vio_at_first_update
        else:
            self.tracker.callback_first_update = initialize_at_first_update

        if args.use_vio_meas:
            self.tracker.debug_callback_get_meas = lambda t0, t1: self.input.get_meas_from_vio(
                t0, t1
            )

        # Loop through the entire dataset and feed the data to the imu tracker
        n_data = self.input.dataset_size
        # Pre-compute the LFA rotation schedule now that the stream length is known
        self.perturb_hook.precompute_lfa(n_data)
        for i in tqdm(range(n_data)):
            # obtain next raw IMU measurement from data loader
            ts, acc_raw, gyr_raw = self.input.get_datai(i)
            # Inject test-time perturbation (gravity tilt / LFA-style noise / fixed SO(3)); no-op by default
            acc_raw, gyr_raw = self.perturb_hook.apply(i, acc_raw, gyr_raw)
            t_us = int(ts * 1e6)
            # cheat a bit with filter state in case for debugging
            if args.debug_using_vio_ba:
                vio_ba = interp1d(self.input.vio_calib_ts, self.input.vio_ba, axis=0)(
                    ts
                )
                vio_bg = interp1d(self.input.vio_calib_ts, self.input.vio_bg, axis=0)(
                    ts
                )
                self.tracker.filter.state.s_ba = np.atleast_2d(vio_ba).T
                self.tracker.filter.state.s_bg = np.atleast_2d(vio_bg).T

            if self.tracker.filter.initialized:
                # did_update = self.tracker.on_imu_measurement(t_us, gyr_raw, acc_raw)
                did_update = self.tracker.on_imu_measurement(t_us, gyr_raw, acc_raw, iter_num)
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
            else:
                # initialize to gt state R,v,p and offline calib
                if not args.initialize_with_vio:
                    self.tracker.on_imu_measurement(t_us, gyr_raw, acc_raw)
                else:
                    if args.initialize_with_offline_calib:
                        init_ba = self.tracker.icalib.accelBias
                        init_bg = self.tracker.icalib.gyroBias
                    else:
                        init_ba = np.zeros((3, 1))
                        init_bg = np.zeros((3, 1))

                    if ts < self.input.vio_ts[0]:
                        continue
                    vio_p = interp1d(self.input.vio_ts, self.input.vio_p, axis=0)(ts)
                    vio_v = interp1d(self.input.vio_ts, self.input.vio_v, axis=0)(ts)
                    vio_eul = interp1d(self.input.vio_ts, self.input.vio_eul, axis=0)(
                        ts
                    )
                    vio_R = Rotation.from_euler(
                        "xyz", vio_eul, degrees=True
                    ).as_matrix()
                    
                    # ##### initialize with no initial information ####
                    # SO(3) robustness augmentation -- only applied when the rpy_values.npy file is present.
                    # Skipped silently for standard filter evaluation.
                    rpy_path = "./../so3_60/rpy_values.npy"
                    if os.path.exists(rpy_path):
                        rotation_info = np.load(rpy_path)
                        rotation_rpy = rotation_info[iter_num, :]
                        m_b2bprime = Rotation.from_euler('xyz', rotation_rpy, degrees=False).as_matrix()
                        vio_R = m_b2bprime.T @ vio_R  # (left)
                        init_ba = m_b2bprime.T @ init_ba
                        init_bg = m_b2bprime.T @ init_bg
                    ##### initialize with no initial information ####
                    
                    
                 
                    self.tracker.init_with_state_at_time(
                        t_us,
                        vio_R,
                        np.atleast_2d(vio_v).T,
                        np.atleast_2d(vio_p).T,
                        init_ba,
                        init_bg,
                    )

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

    def reset_filter_state_from_vio(self, this: ImuTracker):
        """ This reset the filter state from vio state as found in input """
        # compute from vio
        inp = self.input  # for convenience
        state = this.filter.state  # for convenience
        vio_ps = []
        vio_Rs = []
        for i, t_init_us in enumerate(state.si_timestamps_us):
            t_init_s = t_init_us * 1e-6
            ps = np.atleast_2d(interp1d(inp.vio_ts, inp.vio_p, axis=0)(t_init_s)).T
            vio_ps.append(ps)
            vio_eul = interp1d(inp.vio_ts, inp.vio_eul, axis=0)(t_init_s)
            vio_Rs.append(Rotation.from_euler("xyz", vio_eul, degrees=True).as_matrix())

        ts = state.s_timestamp_us * 1e-6
        vio_p = np.atleast_2d(interp1d(inp.vio_ts, inp.vio_p, axis=0)(ts)).T
        vio_v = np.atleast_2d(interp1d(inp.vio_ts, inp.vio_v, axis=0)(ts)).T
        vio_eul = interp1d(inp.vio_ts, inp.vio_eul, axis=0)(ts)
        vio_R = Rotation.from_euler("xyz", vio_eul, degrees=True).as_matrix()

        this.filter.reset_state_and_covariance(
            vio_Rs, vio_ps, vio_R, vio_v, vio_p, state.s_ba, state.s_bg
        )

    def reset_filter_state_pv(self):
        """ Reset filter states p and v with zeros """
        state = self.tracker.filter.state
        ps = []
        for i in state.si_timestamps_us:
            ps.append(np.zeros((3, 1)))
        p = np.zeros((3, 1))
        v = np.zeros((3, 1))
        self.tracker.filter.reset_state_and_covariance(
            state.si_Rs, ps, state.s_R, v, p, state.s_ba, state.s_bg
        )
