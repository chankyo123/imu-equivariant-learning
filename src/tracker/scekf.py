import numpy as np
from numba import jit
from utils.from_scipy import compute_euler_from_matrix
from utils.logging import logging
from utils.math_utils import Jr_exp, hat, mat_exp, mat_exp_vec, mat_log, rot_2vec
from tracker.LieGroup import *


class State(object):
    """docstring for State"""

    def __init__(self):
        super(State, self).__init__()
        self.s_R = None
        self.s_v = None
        self.s_p = None
        self.s_ba = None
        self.s_bg = None
        self.s_timestamp_us = -1  # current state time
        self.N = 0  # number of past states
        self.si_Rs = []  # past states
        self.si_ps = []  # past states
        self.si_vs = []  # past states (velocity)
        self.si_Rs_fej = []
        self.si_ps_fej = []
        self.si_vs_fej = [] 
        self.si_timestamps_us = []
        self.unobs_shift = None
        
        self.InEKF_StateType = "WorldCentric"
        self.InEKF_ErrorType = "RightInvariant"
        self.dimX = 5   # se2(3)
        self.dimTheta = 6   # b_g, b_a
        self.dimP = 15  # R,v,k,bg,ba

    def initialize_state(self, t_us, R, v, p, ba_init, bg_init):
        # take the first accel data to get gravity direction
        assert isinstance(t_us, int)
        self.s_R = R
        self.s_v = v  # m/s
        self.s_p = p  # m
        self.s_bg = bg_init  # rad/s
        self.s_ba = ba_init  # m/s^2
        self.s_timestamp_us = t_us
        self.si_Rs = []
        self.si_ps = []
        self.si_vs = []
        self.si_Rs_fej = []
        self.si_ps_fej = []
        self.si_vs_fej = []
            

        self.si_timestamps_us = []
        self.unobs_shift = self.generate_unobservable_shift()

    def reset_state(self, Rs, ps, vs, R, v, p, ba_init, bg_init):
        # take the first accel data to get gravity direction
        self.s_R = R
        self.s_v = v  # m/s
        self.s_p = p  # m
        self.s_bg = bg_init  # rad/s
        self.s_ba = ba_init  # m/s^2
        self.si_Rs = Rs
        self.si_ps = ps
        self.si_vs = vs
        self.si_Rs_fej = Rs
        self.si_ps_fej = ps
        self.si_vs_fej = vs
        # self.unobs_shift = self.generateUnobservableShift() # TODO(dcaruso)

    def __repr__(self):
        return f"R:\n{self.s_R}\nv:\n{self.s_v}\np:\n{self.s_p}\nbg:\n{self.s_bg}\nba:\n{self.s_ba}"

    def apply_correction(self, dX):
        dX_past = dX[:-15]
        dX_evol = dX[-15:]
        assert dX_past.flatten().shape[0] == (
            self.N * 6
        ), f"number of past error states {dX_past.flatten().shape[0]} does not match the number of states in the filter! {self.N * 6}"

        if self.N > 0:
            temp = dX_past.reshape((self.N, 6))
            dps = np.expand_dims(temp[:, 3:6], axis=2)  # Nx3x1
            dthetas = temp[:, 0:3]
            dRs = mat_exp_vec(dthetas)  # Nx3x3
            Rs_past = np.stack(self.si_Rs, axis=0)  # Nx3x3
            ps_past = np.stack(self.si_ps, axis=0)  # Nx3x3

            Rs_past_new = np.matmul(dRs, Rs_past)
            ps_past_new = ps_past + dps

            N = Rs_past.shape[0]
            self.si_Rs = np.split(Rs_past_new.reshape(N * 3, 3), N, 0)
            self.si_ps = np.split(ps_past_new.reshape(N * 3, 1), N, 0)

        # update current state
        dtheta = dX_evol[:3]
        dv = dX_evol[3:6]
        dp = dX_evol[6:9]
        dbg = dX_evol[9:12]
        dba = dX_evol[12:15]

        dR = mat_exp(dtheta)

        self.s_R = dR.dot(self.s_R)
        self.s_v = self.s_v + dv
        self.s_p = self.s_p + dp
        self.s_bg = self.s_bg + dbg
        self.s_ba = self.s_ba + dba

    def compute_correction(self, target_state):
        assert target_state.N == self.N
        dX = np.zeros((self.N * 6 + 15, 1))

        for i in range(len(self.si_Rs)):
            dX[6 * i : 6 * i + 3] = mat_log(
                target_state.si_Rs[i] @ self.si_Rs[i].inverse()
            )
            dX[6 * i + 3 : 6 * i + 6] = target_state.si_ps[i] - self.si_ps[i]

        dX[6 * self.N :][0:3, 0] = mat_log(target_state.s_R @ self.s_R.T)
        dX[6 * self.N :][3:6, :] = target_state.s_v - self.s_v
        dX[6 * self.N :][6:9, :] = target_state.s_p - self.s_p
        dX[6 * self.N :][9:12, :] = target_state.s_bg - self.s_bg
        dX[6 * self.N :][12:15, :] = target_state.s_ba - self.s_ba

        return dX

    def generate_unobservable_shift(self):
        """ returns a dX element along unobservable directions """
        assert self.N == 0
        g = np.array([[0], [0], [1]])
        dX = np.zeros((15, 4))
        dX[0:3, [0]] = g
        dX[3:6, [0]] = -hat(self.s_v) @ g
        dX[6:9, [0]] = -hat(self.s_p) @ g
        dX[6:9, 1:4] = np.eye(3)
        return dX


@jit(nopython=True, parallel=False, cache=True)
def propagate_rvt_and_jac(R_k, v_k, p_k, b_gk, b_ak, gyr, acc, g, dt):
    def hat(v):
        v = v.flatten()
        R = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return R

    dtheta = (gyr - b_gk) * dt
    dRd = mat_exp(dtheta)
    Rd = R_k @ dRd
    dv_w = R_k @ (acc - b_ak) * dt
    dp_w = 0.5 * dv_w * dt
    gdt = g * dt
    gdt22 = 0.5 * gdt * dt
    vd = v_k + dv_w + gdt
    pd = p_k + v_k * dt + dp_w + gdt22

    A = np.eye(15)
    A[3:6, 0:3] = -hat(dv_w)
    A[6:9, 0:3] = -hat(dp_w)
    A[6:9, 3:6] = np.eye(3) * dt
    A[0:3, 9:12] = -Rd @ Jr_exp(dtheta) * dt
    A[3:6, 12:15] = -R_k * dt
    A[6:9, 12:15] = -0.5 * R_k * dt * dt

    return Rd, vd, pd, A


def get_rotation_from_gravity(acc):
    # take the first accel data to get gravity direction
    ig_w = np.array([0, 0, 1.0]).reshape((3, 1))
    return rot_2vec(acc, ig_w)


class ImuMSCKF:
    def __init__(self, config=None):

        # sanity check
        expected_attribute = [
            "sigma_na",
            "sigma_ng",
            "ita_ba",
            "ita_bg",
            "init_attitude_sigma",
            "init_yaw_sigma",
            "init_vel_sigma",
            "init_pos_sigma",
            "init_bg_sigma",
            "init_ba_sigma",
            "mahalanobis_fail_scale",
        ]
        if not all(hasattr(config, attr) for attr in expected_attribute):
            logging.warning(
                "At least one filter parameter tuning will be left at its default value."
            )

        # constants
        g_norm = getattr(config, "g_norm", 9.81)
        self.g = np.array([0, 0, -g_norm]).reshape((3, 1))

        # parameters
        self.sigma_na = getattr(config, "sigma_na", np.sqrt(1e-3))  # accel noise m/s^2
        self.sigma_ng = getattr(config, "sigma_ng", np.sqrt(1e-4))  # gyro noise rad/s
        self.ita_ba = getattr(config, "ita_ba", 1e-4)  # accel bias noise m/s^2/sqrt(s)
        self.ita_bg = getattr(config, "ita_bg", 1e-6)  # gyro bias noise rad/s/sqrt(s)

        # initial uncertainty
        self.init_attitude_sigma = getattr(
            config, "init_attitude_sigma", 10.0 / 180.0 * np.pi
        )  # rad
        self.init_yaw_sigma = getattr(
            config, "init_yaw_sigma", 0.1 / 180.0 * np.pi
        )  # rad
        self.init_vel_sigma = getattr(config, "init_vel_sigma", 1.0)  # m/s
        self.init_pos_sigma = getattr(config, "init_pos_sigma", 0.001)  # m
        self.init_bg_sigma = getattr(config, "init_bg_sigma", 0.0001)  # rad/s
        self.init_ba_sigma = getattr(config, "init_ba_sigma", 0.2)  # m/s^2

        # other tuning parameters
        self.meascov_scale = getattr(config, "meascov_scale", 1.0)
        self.mahalanobis_factor = 1

        # debug config
        self.use_const_cov = getattr(config, "use_const_cov", False)
        if self.use_const_cov:
            self.const_cov_val_x = config.const_cov_val_x
            self.const_cov_val_y = config.const_cov_val_y
            self.const_cov_val_z = config.const_cov_val_z

        self.add_sim_meas_noise = getattr(config, "add_sim_meas_noise", False)
        if self.add_sim_meas_noise:
            self.sim_meas_cov_val = config.sim_meas_cov_val
            self.sim_meas_cov_val_z = config.sim_meas_cov_val_z

        self.mahalanobis_fail_scale = getattr(config, "mahalanobis_fail_scale", 0)
        self.last_success_mahalanobis = None
        self.force_mahalanobis_until = None

        # state variables
        self.W = None  # IMU measurement noise
        self.Q = None  # stochastic noise (random walk)
        self.R = None  # measurement noise
        self.Sigma = None  # full state covariance
        self.Sigma15 = None  # evolving state covariance
        self.state = State()
        self.last_timestamp_reset_us = None

        # IMU interpolated data online saving
        self.imu_data_int = np.array([])

        # flags
        self.initialized = False
        self.converged = False
        self.first_update = True

        # debug log
        self.innovation = np.zeros((3, 1))
        self.meas = np.zeros((3, 1))
        self.pred = np.zeros((3, 1))
        self.meas_sigma = np.zeros((3, 1))
        self.inno_sigma = np.zeros((3, 1))

    def reset_covariance(self, use_riekf = False):
        var_atti = np.power(self.init_attitude_sigma, 2.0)
        var_yaw = np.power(self.init_yaw_sigma, 2.0)
        var_vel = np.power(self.init_vel_sigma, 2.0)
        var_pos = np.power(self.init_pos_sigma, 2.0)
        var_bg_init = np.power(self.init_bg_sigma, 2.0)
        var_ba_init = np.power(self.init_ba_sigma, 2.0)

        Cov = np.zeros((15 + 6 * self.state.N, 15 + 6 * self.state.N))
        if not use_riekf:
            for i, _ in enumerate(self.state.si_timestamps_us):
                Cov[6 * i :, 6 * i :][0:3, 0:3] = np.diag([var_atti, var_atti, var_yaw])
                Cov[6 * i :, 6 * i :][3:6, 3:6] = np.diag([var_pos, var_pos, var_pos])

        Cov15 = Cov[-15:, -15:]  # no copy, ref
        Cov15[:3, :3] = np.diag(np.array([var_atti, var_atti, var_yaw]))
        Cov15[3:6, 3:6] = np.diag(np.array([var_vel, var_vel, var_vel]))
        Cov15[6:9, 6:9] = np.diag(np.array([var_pos, var_pos, var_pos]))
        Cov15[9:12, 9:12] = np.diag(np.array([var_bg_init, var_bg_init, var_bg_init]))
        Cov15[12:15, 12:15] = np.diag(np.array([var_ba_init, var_ba_init, var_ba_init]))
        self.Sigma = Cov
        if use_riekf:
            self.Sigma15 = Cov15
        else:
            self.Sigma15 = Cov[-15:, -15:]
        # self.Sigma15 = Cov[-15:, -15:]

    def prepare_filter(self, use_riekf = False):
        # define noise covariances
        var_a = np.power(self.sigma_na, 2.0)
        var_g = np.power(self.sigma_ng, 2.0)
        var_ba = np.power(self.ita_ba, 2.0)
        var_bg = np.power(self.ita_bg, 2.0)
        self.W = np.diag(np.array([var_g, var_g, var_g, var_a, var_a, var_a]))
        self.Q = np.diag(np.array([var_bg, var_bg, var_bg, var_ba, var_ba, var_ba]))
        # initialize state covariance
        self.reset_covariance(use_riekf)

    def initialize_state(self, t_us, R, v, p, ba_init, bg_init):
        self.state.initialize_state(t_us, R, v, p, ba_init, bg_init)
        self.last_timestamp_reset_us = t_us

    def reset_state_and_covariance(self, Rs, ps, vs, R, v, p, ba_init, bg_init, use_riekf = False):
        if not use_riekf:
            assert len(Rs) == self.state.N
            assert len(ps) == self.state.N
        self.state.reset_state(Rs, ps, vs, R, v, p, ba_init, bg_init)
        self.reset_covariance(use_riekf)
        self.last_timestamp_reset_us = self.state.s_timestamp_us

        if not use_riekf:
            assert len(Rs) == self.state.N
            assert len(ps) == self.state.N
            assert self.Sigma.shape[0] == self.state.N * 6 + 15
            assert self.Sigma.shape[1] == self.state.N * 6 + 15

    def initialize_with_state(self, t_us, R, v, p, ba_init, bg_init, use_riekf = False):
        assert isinstance(t_us, int)
        self.prepare_filter(use_riekf)
        self.initialized = True
        self.initialize_state(t_us, R, v, p, ba_init, bg_init)
        logging.info("filter initialized with full state!")

    def initialize(self, t_us, acc, ba_init, bg_init, use_riekf = False):
        assert isinstance(t_us, int)
        self.prepare_filter(use_riekf)
        self.initialized = True
        self.initialize_state(
            t_us,
            get_rotation_from_gravity(acc),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            ba_init,
            bg_init,
        )
        logging.info("filter initialized with gravity!")

    def get_past_state(self, t_us):
        assert isinstance(t_us, int)
        state_idx = self.state.si_timestamps_us.index(t_us)
        R = self.state.si_Rs[state_idx]
        p = self.state.si_ps[state_idx]

        return R, p

    def get_evolving_state(self):
        R = self.state.s_R.copy()
        v = self.state.s_v.copy()
        p = self.state.s_p.copy()
        bg = self.state.s_bg.copy()
        ba = self.state.s_ba.copy()
        return R, v, p, ba, bg

    def get_covariance(self):
        Sigma = self.Sigma
        Sigma15 = self.Sigma15
        return Sigma, Sigma15

    def get_covariance_yawp(self):
        return self.Sigma15[[2, 6, 7, 8], :][:, [2, 6, 7, 8]]

    def get_info_along_unobservable_shift(self):
        return np.diag(
            self.state.unobs_shift.T
            @ np.linalg.pinv(self.Sigma)
            @ self.state.unobs_shift
        )

    def get_debug(self):
        return self.innovation, self.meas, self.pred, self.meas_sigma, self.inno_sigma

    def check_filter_convergence(self):
        """ let us assume the filter has converged after 10 second"""
        return self.state.si_timestamps_us[0] - self.last_timestamp_reset_us > int(
            10 * 1e6
        )

    def is_mahalanobis_activated(self):
        if not self.converged:
            return False

        if self.force_mahalanobis_until is not None:
            if self.state.s_timestamp_us < self.force_mahalanobis_until:
                return False

        if self.last_success_mahalanobis is not None:
            if self.state.s_timestamp_us > self.last_success_mahalanobis + int(
                0.5 * 1e6
            ):
                logging.warning(
                    "Deactivating Mahalanobis test, because failed for too long."
                )
                self.last_success_mahalanobis = None
                self.force_mahalanobis_until = self.state.s_timestamp_us + int(1e6)
                return False
        return True

    def propagate_riekf(self, acc, gyr, t_us, m_b2bprime):

        # Bias corrected IMU measurements
        R_k, v_k, p_k, b_ak, b_gk = (
            self.state.s_R,
            self.state.s_v,
            self.state.s_p,
            self.state.s_ba,
            self.state.s_bg,
        )
        w = gyr - b_gk
        a = acc - b_ak

        # Get current state estimate and dimensions
        dimX = self.state.dimX
        X = np.eye(dimX)
        X[0:3, 0:3] = R_k
        X[0:3, 3] = v_k.reshape(3, )
        X[0:3, 4] = p_k.reshape(3, )

        Xinv = np.eye(dimX)
        RT = X[0:3, 0:3].T
        Xinv[0:3, 0:3] = RT
        for i in range(3, dimX):
            Xinv[0:3, i] = -RT @ X[0:3, i]
        
        P = self.Sigma15
        dimP = P.shape[0]
        dimTheta = 6  # b_ak, b_gk
        if t_us:
            dt_us = t_us - self.state.s_timestamp_us
            dt = dt_us * 1e-6
            
            # ------------ Propagate Covariance ------------- #
            Phi = self.StateTransitionMatrix(w,a,dt)
            Qd = self.DiscreteNoiseMatrix(Phi,dt)
            P_pred = Phi @ P @ Phi.T + Qd

            # add noise from bias model
            # P_pred[-6:, -6:] += dt * self.Q
            # P_pred = Phi @ P @ Phi.T
            
            # #if no bias update
            P_pred[:dimP - dimTheta, dimP - dimTheta:] = np.zeros((dimP - dimTheta, dimTheta))
            P_pred[dimP - dimTheta:, :dimP - dimTheta] = np.zeros((dimTheta, dimP - dimTheta))
            P_pred[dimP - dimTheta:, dimP - dimTheta:] = np.eye(dimTheta)
            
            # ------------ Propagate Mean ------------- #
            phi = w * dt
            G0 = Gamma_SO3(phi, 0)
            G1 = Gamma_SO3(phi, 1)
            G2 = Gamma_SO3(phi, 2)

            X_pred = X.copy()
            if self.state.InEKF_StateType == 'WorldCentric':
                # Propagate world-centric state estimate
                X_pred[0:3, 0:3] = R_k @ G0
                X_pred[0:3, 3] = (v_k + (R_k @ G1 @ a + m_b2bprime @ self.g) * dt).reshape(3)
                X_pred[0:3, 4] = (p_k + v_k * dt + (R_k @ G2 @ a + 0.5 * m_b2bprime @ self.g) * dt * dt).reshape(3)
            else:
                # Propagate body-centric state estimate
                G0t = G0.T
                X_pred[0:3, 0:3] = G0t @ R_k
                X_pred[0:3, 3] = (G0t @ (v_k - (G1 @ a + R_k @ self.g) * dt)).reshape(3)
                X_pred[0:3, 4] = (G0t @ (p_k + v_k * dt - (G2 @ a + 0.5 * R_k @ self.g) * dt * dt)).reshape(3)
                for j in range(5, dimX):
                    X_pred[0:3, j] = G0t @ X[0:3, j]

            # ------------ Update State ------------- #
            self.state.s_R = X_pred[0:3, 0:3]
            self.state.s_v = X_pred[0:3, 3].reshape(-1, 1)
            self.state.s_p = X_pred[0:3, 4].reshape(-1, 1)
            self.state.s_ba = b_ak     # No change in bias in propagate step
            self.state.s_bg = b_gk     # No change in bias in propagate step
            self.state.s_timestamp_us = t_us
            self.Sigma15 = P_pred
            
            self.state.si_Rs.append(self.state.s_R)  
            self.state.si_vs.append(self.state.s_R.T @ self.state.s_v)  #save body velocity
            self.state.si_timestamps_us.append(t_us)
        
    
    def update_riekf(self, meas, meas_cov, t_begin_us, t_end_us):
        R, v, p, b_a, b_g = (
            self.state.s_R,
            self.state.s_v,
            self.state.s_p,
            self.state.s_ba,
            self.state.s_bg,
        )
        dimX = self.state.dimX
        dimP = self.state.dimP
        X = np.eye(dimX)
        X[0:3, 0:3] = R
        X[0:3, 3] = v.reshape(3, )
        X[0:3, 4] = p.reshape(3, )

        Theta = np.concatenate((b_g, b_a), axis=0)
        P = self.Sigma15
        # print(dimP)
        # assert False
        dimTheta = self.state.dimTheta

        # # if no bias update
        # Theta = np.zeros((6,1))
        P[dimP-dimTheta:dimP-dimTheta+6,dimP-dimTheta:dimP-dimTheta+6] = 0.0001*np.eye(6)
        P[0:dimP-dimTheta,dimP-dimTheta:dimP] = np.zeros((dimP-dimTheta,dimTheta))
        P[dimP-dimTheta:dimP,0:dimP-dimTheta] = np.zeros((dimTheta,dimP-dimTheta))

        # # Map from left invariant to right invariant error temporarily
        # if self.state.InEKF_ErrorType == 'LeftInvariant':
        #     Adj = np.eye(dimP)
        #     Adj[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(X)
        #     P = (Adj @ P @ Adj.T)
        
        
        # N = 0.1*np.eye(3) #so3 3으로 해보다가 1로 바꿈
        # N = 1*np.eye(3) #used when using gtv
        N = 5*np.eye(3)
        # N = 0.1*np.eye(3)
        # N = 0.01*np.eye(3)
        
        # N = self.meascov_scale * meas_cov
        # print('N!!')
        # N = 0.001*np.eye(3)
        
        if self.use_const_cov:
            val_x = self.const_cov_val_x
            val_y = self.const_cov_val_y
            val_z = self.const_cov_val_z
            N = self.meascov_scale * np.diag(np.array([val_x, val_y, val_z]))
            print('N value : ', np.array([val_x, val_y, val_z]))  #set to be -> N value :  [0.01 0.01 0.01]
        
        # symmetrize N
        N = 0.5 * (N + N.T)
        N[N < 1e-10] = 0
        
        H = np.zeros((3, dimP))
        H[0:3,3:6] = np.eye(3) # I
        
        mahalanobis_factor = 0
        if mahalanobis_factor > 0:
            # Mahalanobis gating test
            PHT = P @ H.T
            S_temp = H @ PHT + N
            Sinv_temp = np.linalg.inv(S_temp)
            b_tmp = np.array([0, 0, 0, -1, 0]).reshape(-1,1)
            Y_tmp = np.vstack((meas, np.array([-1, 0]).reshape(-1, 1)))
            Z_tmp = (X @ Y_tmp - b_tmp)[0:3].reshape(-1,1)
            
            normalized_square_error = np.linalg.multi_dot(
                [Z_tmp.T, Sinv_temp, Z_tmp]
            )
            # maha_constant = 30
            # maha_constant = 20
            maha_constant = 1
            # maha_constant = 300
            # maha_constant = 1.4
            test_failed = normalized_square_error > mahalanobis_factor * 11.345 * maha_constant  #cov2까지는 200을 곱했었음
            # wait for convergence before failing
            if test_failed:
                print("Mahalanobis test failed... xi2 =", normalized_square_error)
                return
            # break
                    
                    
        b = np.array([0, 0, 0, -1, 0]).reshape(-1,1)
        Y = np.vstack((meas, np.array([-1, 0]).reshape(-1, 1)))
        Z = X @ Y - b    # XY-b
        Z = Z[0:3]
        self.innovation = Z[0:3]
                    
        PHT = P @ H.T
        S = H @ PHT + N
        S_inv = np.linalg.inv(S)
        K = PHT @ S_inv
        delta = K @ Z
        
        dX = Exp_SEK3(delta[0:np.shape(delta)[0]-dimTheta])
        dTheta = delta[np.shape(delta)[0]-dimTheta:]
        
        self.meas = X[0:3,0:3] @ meas
        self.pred = X[0:3,3:4]
        self.R = N
        self.meas_sigma = np.sqrt(np.diag(N)).reshape(3, 1)
        self.inno_sigma = np.sqrt(np.diag(S)).reshape(3, 1)
        
        X_new = dX @ X  #right invaraint
        Theta_new = Theta + dTheta

        # Set new state       
        self.state.s_R = X_new[0:3, 0:3]
        self.state.s_v = X_new[0:3, 3].reshape(-1, 1)
        self.state.s_p = X_new[0:3, 4].reshape(-1, 1)
        self.state.s_ba = Theta_new[3:6].reshape(-1, 1)
        self.state.s_bg = Theta_new[0:3].reshape(-1, 1)
        
        # self.state.si_Rs.append(self.state.s_R)  
        # self.state.si_vs.append(self.state.s_R.T @ self.state.s_v)  #save body velocity
        # self.state.si_timestamps_us.append(t_end_us)
        # # self.state.si_vs.append(self.state.s_v)     #save world velocity

        # Update Covariance
        IKH = np.eye(dimP) - K @ H
        P_new = IKH @ P @ IKH.T + K @ N @ K.T
        
        # # Don't update yaw covariance
        # yaw_index = dimP - dimTheta + 2
        # P_new[yaw_index, :] = 0
        # P_new[:, yaw_index] = 0
        # P_new[yaw_index, yaw_index] = 0.00001

        # Map from right invariant back to left invariant error
        if self.state.InEKF_ErrorType == 'LeftInvariant':
            AdjInv = np.eye(dimP)
            AdjInv[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(self.state_.Xinv())
            P_new = (AdjInv @ P_new @ AdjInv.T)

        # Set new covariance
        # print('P_new.shape : ', P_new.shape)
        
        # normalized_square_error = np.linalg.multi_dot(
        #     [Z.T, S_inv, Z]
        # )
        # print("Mahalanobis error =", normalized_square_error)
        self.Sigma15 = P_new
        
        
    def propagate(self, acc, gyr, t_us, t_augmentation_us=None):

        R_k, v_k, p_k, b_ak, b_gk = (
            self.state.s_R,
            self.state.s_v,
            self.state.s_p,
            self.state.s_ba,
            self.state.s_bg,
        )

        N = self.state.N
        # evolving state propagation
        dt_us = t_us - self.state.s_timestamp_us
        R_kp1, v_kp1, p_kp1, Akp1 = propagate_rvt_and_jac(
            R_k, v_k, p_k, b_gk, b_ak, gyr, acc, self.g, dt_us * 1e-6
        )
        b_gkp1 = b_gk
        b_akp1 = b_ak

        B = np.zeros((15, 6))
        B[0:3, 0:3] = -Akp1[0:3, 9:12]
        B[3:6, 3:6] = -Akp1[3:6, 12:15]
        B[6:9, 3:6] = -Akp1[6:9, 12:15]

        # partial integration for state augmentation
        if t_augmentation_us:
            # past state propagation (partial integration)
            dtd_us = t_augmentation_us - self.state.s_timestamp_us
            Rd, vd, pd, Ad = propagate_rvt_and_jac(
                R_k, v_k, p_k, b_gk, b_ak, gyr, acc, self.g, dtd_us * 1e-6
            )

            # propagate covariance
            JA = np.zeros((6, 15))
            JA[0:3, :] = Ad[0:3, :]
            JA[3:6, :] = Ad[6:9, :]

            A_aug = np.zeros(((15 + 6 * (N + 1)), (15 + 6 * N)))
            A_aug[0 : 6 * N, 0 : 6 * N] = np.eye(6 * N)
            A_aug[-15 - 6 : -15, -15:] = JA
            A_aug[-15:, -15:] = Akp1

            BJ = np.zeros((6, 6))
            BJ[0:3, 0:3] = -Ad[0:3, 9:12]
            BJ[3:6, 3:6] = -Ad[6:9, 12:15]

            B_aug = np.zeros(((15 + 6), 6))
            B_aug[-15 - 6 : -15, :] = BJ
            B_aug[-15:, :] = B

            # past state augmentation
            assert Rd.shape == (3, 3), "inserted past rotation state shape incorrect"
            self.state.si_Rs.append(Rd)
            self.state.si_ps.append(pd)
            self.state.si_Rs_fej.append(Rd)
            self.state.si_ps_fej.append(pd)
            self.state.si_timestamps_us.append(t_augmentation_us)

            self.state.N += 1

            # print("state augmented, current number of past states: %s" % self.N)
        else:  # aug==False
            # propagate covariance
            A_aug = np.eye((15 + 6 * N))
            A_aug[-15:, -15:] = Akp1

            B_aug = np.zeros((15, 6))
            B_aug[-15:, :] = B

        Sigma_kp1 = propagate_covariance(
            A_aug, B_aug, dt_us * 1e-6, self.Sigma, self.W, self.Q
        )

        # update states and covariance variables
        self.state.s_R = R_kp1
        self.state.s_v = v_kp1
        self.state.s_p = p_kp1
        self.state.s_ba = b_akp1
        self.state.s_bg = b_gkp1
        self.state.s_timestamp_us = t_us
        self.Sigma = Sigma_kp1
        self.Sigma15 = self.Sigma[-15:, -15:]
        self.state.unobs_shift = A_aug @ self.state.unobs_shift  # propagate unobs shift

    def update(self, meas, meas_cov, t_begin_us, t_end_us):
        """ 
           pred: pj - pi in the world frame
           pred_cov: log(sigma) of the measurements in 3D
           meas_cov [3 x 3] : covariance of measurement matrix 
        """
        if not self.converged and self.check_filter_convergence():
            logging.info("Filter is now assumed to have converged")
            self.converged = True

        try:
            begin_idx = self.state.si_timestamps_us.index(t_begin_us)
            end_idx = self.state.si_timestamps_us.index(t_end_us)
        except Exception:
            logging.error("timestamps not found in past states!")
            import ipdb

            ipdb.set_trace()
            exit(1)

        self.R = self.meascov_scale * meas_cov
        # self.R = 2*np.eye(3)
        # set constant measurement covariance
        if self.use_const_cov:
            val_x = self.const_cov_val_x
            val_y = self.const_cov_val_y
            val_z = self.const_cov_val_z
            self.R = self.meascov_scale * np.diag(np.array([val_x, val_y, val_z]))

        # symmetrize R
        self.R = 0.5 * (self.R + self.R.T)
        self.R[self.R < 1e-10] = 0

        # add measurement noise (if using simulation)
        if self.add_sim_meas_noise:
            wp = np.random.multivariate_normal(
                [0, 0, 0],
                np.diag(
                    [
                        self.sim_meas_cov_val,
                        self.sim_meas_cov_val,
                        self.sim_meas_cov_val_z,
                    ]
                ),
                (1),
            ).T
            meas = meas + wp

        Ri = self.state.si_Rs[begin_idx]
        ri_eul = compute_euler_from_matrix(Ri, "xyz", extrinsic=True)  # in radians
        ri_y = ri_eul[0, 1]
        ri_z = ri_eul[0, 2]
        Ri_z = np.array(
            [
                [np.cos(ri_z), -(np.sin(ri_z)), 0],
                [np.sin(ri_z), np.cos(ri_z), 0],
                [0, 0, 1],
            ]
        )
        pred = Ri_z.T.dot(self.state.si_ps[end_idx] - self.state.si_ps[begin_idx])

        assert begin_idx < end_idx, "begin_idx is larger than end_idx!"
        assert (
            end_idx < self.state.N
        ), "end_idx is larger than the number of past states in the filter!"
        H = np.zeros((3, 15 + 6 * self.state.N))
        H[:, (6 * begin_idx + 3) : (6 * begin_idx + 6)] = -Ri_z.T
        H[:, (6 * end_idx + 3) : (6 * end_idx + 6)] = Ri_z.T
        Hz = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [np.cos(ri_z) * np.tan(ri_y), np.sin(ri_z) * np.tan(ri_y), 1],
            ]
        )
        H[:, (6 * begin_idx) : (6 * begin_idx + 3)] = np.linalg.multi_dot(
            [Ri_z.T, hat(self.state.si_ps[end_idx] - self.state.si_ps[begin_idx]), Hz]
        )

        # check for singularity, if has singularity drop the update
        if abs(np.cos(ri_y)) < 1e-5:
            logging.warning(f"Singularity in H matrix, dropping update")
            return

        assert (
            self.Sigma.shape[0] == H.shape[1]
        ), "state covariance and matrix H does not match shape!"

        if self.mahalanobis_factor > 0:
            # Mahalanobis gating test
            S_temp = np.linalg.multi_dot([H, self.Sigma, H.T]) + self.R
            Sinv_temp = np.linalg.inv(S_temp)
            normalized_square_error = np.linalg.multi_dot(
                [(meas - pred).T, Sinv_temp, meas - pred]
            )
            # threshold from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm for nu=3, p =99
            test_failed = normalized_square_error > self.mahalanobis_factor * 11.345
            # wait for convergence before failing
            if self.is_mahalanobis_activated() and test_failed:
                if self.mahalanobis_fail_scale == 0:
                    # do not do the update
                    # print("Mahalanobis test failed... xi2 =", normalized_square_error)
                    return
                else:
                    self.R = self.mahalanobis_fail_scale * self.R
            else:
                self.last_success_mahalanobis = self.state.s_timestamp_us

        S = np.linalg.multi_dot([H, self.Sigma, H.T]) + self.R
        Sinv = np.linalg.inv(S)
        self.innovation = meas - pred
        self.meas = meas
        self.pred = pred
        self.meas_sigma = np.sqrt(np.diag(self.R)).reshape(3, 1)
        self.inno_sigma = np.sqrt(np.diag(S)).reshape(3, 1)

        # obtain kalman gain
        K = np.linalg.multi_dot([self.Sigma, H.T, Sinv])
        delta_X = K.dot(meas - pred)
        self.state.apply_correction(delta_X)

        # update covariance
        Sigma_up = self.Sigma - np.linalg.multi_dot([K, H, self.Sigma])
        Sigma_up = 0.5 * (Sigma_up + Sigma_up.T)
        self.Sigma = Sigma_up
        self.Sigma15 = self.Sigma[-15:, -15:]

    def marginalize(self, cut_idx):

        # marginalize states prior to cut_idx
        self.state.si_Rs = self.state.si_Rs[cut_idx + 1 :]
        self.state.si_ps = self.state.si_ps[cut_idx + 1 :]
        self.state.si_Rs_fej = self.state.si_Rs_fej[cut_idx + 1 :]
        self.state.si_ps_fej = self.state.si_ps_fej[cut_idx + 1 :]
        self.state.si_timestamps_us = self.state.si_timestamps_us[cut_idx + 1 :]
        self.state.unobs_shift = self.state.unobs_shift[6 * (cut_idx + 1) :, :]
        self.Sigma = self.Sigma[6 * (cut_idx + 1) :, 6 * (cut_idx + 1) :]
        self.state.N = self.state.N - (cut_idx + 1)
        
    def StateTransitionMatrix(self, w, a, dt):
        phi = w*dt
        G0 = Gamma_SO3(phi,0)
        G1 = Gamma_SO3(phi,1)
        G2 = Gamma_SO3(phi,2)
        G0t = np.transpose(G0)
        G1t = np.transpose(G1)
        G2t = np.transpose(G2)
        G3t = Gamma_SO3(-phi,3)
        dimX = self.state.dimX
        dimTheta = self.state.dimTheta
        dimP = self.state.dimP
        Phi = np.eye(dimP)

        ax = skew(a)
        wx = skew(w)
        wx2 = wx@wx

        dt2 = dt*dt
        dt3 = dt2*dt
        theta = np.linalg.norm(w)
        theta2 = theta*theta
        theta3 = theta2*theta
        theta4 = theta3*theta
        theta5 = theta4*theta
        theta6 = theta5*theta
        theta7 = theta6*theta
        thetadt = theta*dt
        thetadt2 = thetadt*thetadt
        thetadt3 = thetadt2*thetadt
        sinthetadt = np.sin(thetadt)
        costhetadt = np.cos(thetadt)
        sin2thetadt = np.sin(2*thetadt)
        cos2thetadt = np.cos(2*thetadt)
        thetadtcosthetadt = thetadt*costhetadt
        thetadtsinthetadt = thetadt*sinthetadt


        Phi25L = G0t@(ax@G2t*dt2\
                + ((sinthetadt-thetadtcosthetadt)/(theta3))*(wx@ax)\
                - ((cos2thetadt-4*costhetadt+3)/(4*theta4))*(wx@ax@wx)\
                + ((4*sinthetadt+sin2thetadt-4*thetadtcosthetadt-2*thetadt)/(4*theta5))*(wx@ax@wx2)\
                + ((thetadt2-2*thetadtsinthetadt-2*costhetadt+2)/(2*theta4))*(wx2@ax)\
                - ((6*thetadt-8*sinthetadt+sin2thetadt)/(4*theta5))*(wx2@ax@wx)\
                + ((2*thetadt2-4*thetadtsinthetadt-cos2thetadt+1)/(4*theta6))*(wx2@ax@wx2))

        Phi35L = G0t@(ax@G3t*dt3\
                - ((thetadtsinthetadt+2*costhetadt-2)/(theta4))*(wx@ax)\
                - ((6*thetadt-8*sinthetadt+sin2thetadt)/(8*theta5))*(wx@ax@wx)
                - ((2*thetadt2+8*thetadtsinthetadt+16*costhetadt+cos2thetadt-17)/(8*theta6))*(wx@ax@wx2)\
                + ((thetadt3+6*thetadt-12*sinthetadt+6*thetadtcosthetadt)/(6*theta5))*(wx2@ax)\
                - ((6*thetadt2+16*costhetadt-cos2thetadt-15)/(8*theta6))*(wx2@ax@wx)\
                + ((4*thetadt3+6*thetadt-24*sinthetadt-3*sin2thetadt+24*thetadtcosthetadt)/(24*theta7))*(wx2@ax@wx2))


        tol = 1e-6
        if theta < tol:
            Phi25L = (1/2)*ax*dt2
            Phi35L = (1/6)*ax*dt3


        if (self.state.InEKF_StateType == 'WorldCentric' and self.state.InEKF_ErrorType == 'LeftInvariant') or \
           (self.state.InEKF_StateType == 'BodyCentric' and self.state.InEKF_ErrorType == 'RightInvariant'):
            
            Phi[0:3,0:3] = G0t.copy()
            Phi[3:6,0:3] = -G0t@skew(G1@a)*dt
            Phi[6:9,0:3] = -G0t@skew(G2@a)*dt2
            Phi[3:6,3:6] = G0t.copy()
            Phi[6:9,3:6] = G0t*dt
            Phi[6:9,6:9] = G0t.copy()

            for i in np.arange(5,dimX):
                Phi[(i-2)*3:(i-2)*3+3,(i-2)*3:(i-2)*3+3] = G0t.copy()

            Phi[0:3,dimP-dimTheta:dimP-dimTheta+3] = -G1t*dt
            Phi[3:6,dimP-dimTheta:dimP-dimTheta+3] = Phi25L
            Phi[6:9,dimP-dimTheta:dimP-dimTheta+3] = Phi35L
            Phi[3:6,dimP-dimTheta+3:dimP-dimTheta+6] = -G1t*dt
            Phi[6:9,dimP-dimTheta+3:dimP-dimTheta+6] = -G0t@G2*dt2


            
        else:
            gx = skew(self.g)
            R = self.state.s_R
            v = self.state.s_v
            p = self.state.s_p
            RG0 = R@G0
            RG1dt = R@G1*dt
            RG2dt2 = R@G2*dt2
            Phi[3:6,0:3] = gx*dt
            Phi[6:9,0:3] = 0.5*gx*dt2
            Phi[6:9,3:6] = np.eye(3)*dt
            
            # #if bias update
            # Phi[0:3,dimP-dimTheta:dimP-dimTheta+3] = -RG1dt
            # Phi[3:6,dimP-dimTheta:dimP-dimTheta+3] = -skew(v+RG1dt@a+self.g*dt)@RG1dt + RG0@Phi25L
            # Phi[6:9,dimP-dimTheta:dimP-dimTheta+3] = -skew(p+v*dt+RG2dt2@a+0.5*self.g*dt2)@RG1dt + RG0@Phi35L
            
            
            # for i in range(5, dimX):
            #     Phi[(i - 2) * 3:(i - 2) * 3 + 3, dimP - dimTheta:dimP] = -skew(self.state.get_vector(i)) @ RG1dt
            # Phi[3:6,dimP-dimTheta+3:dimP-dimTheta+6] = -RG1dt
            # Phi[6:9,dimP-dimTheta+3:dimP-dimTheta+6] = -RG2dt2
            
            #else
            Phi[:, dimP - dimTheta:] = np.zeros((dimP, dimTheta))

        return Phi
    
    # Compute Discrete noise matrix
    def DiscreteNoiseMatrix(self, Phi, dt):
        dimX = self.state.dimX    # SE2(3)
        dimTheta = 6    # b_a, b_g
        dimP = 15   # R, v, p, b_a, b_g
        G = np.eye(dimP)

        # Compute 'Right Invariant' in 'World Centric' Model
        # Compute G using Adjoint of Xk if needed, otherwise identity (Assumes unpropagated state)
        R_k, v_k, p_k = (
            self.state.s_R,
            self.state.s_v,
            self.state.s_p,
        )
        
        WorldX = np.eye(dimX)
        WorldX[0:3, 0:3] = R_k
        WorldX[0:3, 3] = v_k.reshape(3, )
        WorldX[0:3, 4] = p_k.reshape(3, )
        
        G[0:dimP-dimTheta,0:dimP-dimTheta] = Adjoint_SEK3(WorldX)
        
        # Continuous noise covariance 
        Qc = np.zeros((dimP,dimP))  # Landmark noise terms will remain zero
        Qc[0:3,0:3] = self.W[0:3,0:3]
        Qc[3:6,3:6] = self.W[3:6,3:6]

        # #if bias update
        # Qc[dimP-dimTheta:dimP-dimTheta+3,dimP-dimTheta:dimP-dimTheta+3] = self.Q[0:3,0:3]
        # Qc[dimP-dimTheta+3:dimP-dimTheta+6,dimP-dimTheta+3:dimP-dimTheta+6] = self.Q[3:6,3:6]

        # Noise Covariance Discretization
        PhiG = Phi@G
        Qd = PhiG @ Qc @ np.transpose(PhiG) * dt    # Approximated discretized noise matrix
        # Qd[-6:,-6:] += self.Q * dt    # Approximated discretized noise matrix

        return Qd


@jit(nopython=True, parallel=False, cache=True)
def propagate_covariance(A_aug, B_aug, dt, Sigma, W, Q):
    """
    :param A_aug:
    :param B_aug: [15 x 6] or [15+6 x 6]
    :return:
    """
    dim_new_state = A_aug.shape[0] - A_aug.shape[1]  # either 0 or 6
    assert B_aug.shape[0] == 15 + dim_new_state
    A = A_aug[-15 - dim_new_state :, -15:]
    AT = A_aug[-15 - dim_new_state :, -15:].T
    ret = np.zeros((A_aug.shape[0], A_aug.shape[0]))
    ret[: -15 - dim_new_state, : -15 - dim_new_state] = Sigma[
        :-15, :-15
    ]  # copy top-left block
    ret[: -15 - dim_new_state, -15 - dim_new_state :] = (
        Sigma[:-15, -15:] @ AT
    )  # top-right corner
    ret[-15 - dim_new_state :, : -15 - dim_new_state] = (
        A @ Sigma[-15:, :-15]
    )  # bottom-left corner
    ret[-15 - dim_new_state :, -15 - dim_new_state :] = (
        A @ Sigma[-15:, -15:] @ AT
    )  # bottom-right corner
    # add noise from input
    ret[-15 - dim_new_state :, -15 - dim_new_state :] += (
        B_aug @ W @ B_aug.T
    )  # only non zero on last block
    # add noise from bias model
    ret[-6:, -6:] += dt * Q
    return ret
