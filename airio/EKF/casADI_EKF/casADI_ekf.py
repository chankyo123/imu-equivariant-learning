import casadi as ca
import numpy as np
from liecasadi import SO3Tangent

class CasADIEKF():
    def __init__(self):
        self.model = CasADIMU()
        x  = ca.MX.sym('x',15);  P = ca.MX.sym('P',15,15)
        u  = ca.MX.sym('u',12);  obs = ca.MX.sym('z',3)
        Q  = ca.MX.sym('Q',12,12); R = ca.MX.sym('R',3,3)
        dt = ca.MX.sym('dt')
        self.I  = ca.MX.eye(15)

        x_post, P_post = self._filtering(x, u, dt, obs, P, Q, R)
        x_next, P_next = self._predict(x, u, dt, P, Q)
        self.filtering = ca.Function('filtering',
                  [x, u, dt, obs, P, Q, R],
                  [x_post, P_post],
                  {'jit':True})
        self.predict = ca.Function('predict',
                  [x, u, dt, P, Q],
                  [x_next, P_next],
                  {'jit':True})
    
    def _filtering(self, state, intput, dt, obs, P, Q, R):
        x_next = self.model.state_transition(state, intput, dt)
        A = self.model.A(state, intput, dt)
        B = self.model.B(state, intput, dt)
        C = self.model.C(state, intput, dt)
        P = A @ P @ A.T + B @ Q @ B.T

        z = self.model.observation(state, intput, dt)
        S = C @ P @ C.T + R
        K = ca.solve(S, (P @ C.T).T).T # 3. Kalman gain
        e = obs - z    #    predicted observation error
        
        x_next = x_next + K @ e                   # 4. Posteriori state
        P = (self.I - K @ C) @ P @ (self.I - K @ C).T + K @ R @ K.T                   # 5. Posteriori covariance
        return x_next, P
    
    def _predict(self, state, intput, dt, P, Q):
        x_next = self.model.state_transition(state, intput, dt)
        A = self.model.A(state, intput, dt)
        B = self.model.B(state, intput, dt)
        P = A @ P @ A.T + B @ Q @ B.T
        return x_next, P
  

class CasADIMU():
    def __init__(self):
        self.g = ca.DM([0., 0., 9.81])

        # ------ State Transition ------
        x = ca.MX.sym('x', 15)
        u = ca.MX.sym('u', 12)
        dt = ca.MX.sym('dt')
        x_next = self._f_expr(x, u, dt)
        self.f_transition = ca.Function('f_transition', [x, u, dt], [x_next]) 

        # ------    Observation --------

        z = self._h_expr(x, u, dt)
        self.f_observation = ca.Function('f_observation', [x, u, dt], [z])

        # ------ Jacobian ------
        A_sym = ca.jacobian(x_next, x)
        B_sym = ca.jacobian(x_next, u)
        C_sym = ca.jacobian(z, x)

        self.A = ca.Function('A', [x,u,dt], [A_sym])
        self.B = ca.Function('B', [x,u,dt], [B_sym])
        self.C = ca.Function('C', [x,u,dt], [C_sym])
    
    def _f_expr(self, x, u, dt):
        r = x[:3]
        v = x[3:6]
        p = x[6:9]
        w_m = u[:3]
        a_m = u[3:6]
        bg = u[6:9]
        ba = u[9:12]
        init_rot = SO3Tangent(r).exp()
        w = w_m - bg
        a = a_m - ba - init_rot.inverse().act(self.g)

        Dr = SO3Tangent(w * dt).exp()
        Dv = Dr.act(a * dt)
        Dp = Dv * dt + Dr.act(a * 0.5 * dt**2)

        R = (init_rot * Dr).log().vec
        V = v + init_rot.act(Dv)
        P = p + init_rot.act(Dp) + v * dt
        
        return ca.vertcat(R, V, P, bg, ba)
    
    def state_transition(self, state, input, dt):
        return self.f_transition(state, input, dt)
    
    def _h_expr(self, x, u, dt):
        x = self.f_transition(x, u, dt)
        rot = SO3Tangent(x[:3]).exp()
        velo = rot.inverse().act(x[3:6])
        return velo
    
    def observation(self, x, u, dt):
        return self.f_observation(x, u, dt)
    


if __name__ == "__main__":
    import tqdm
    imu = CasADIEKF()
    for i in tqdm.tqdm(range(10000)):

        state_np = np.zeros(15)
        state_np[:3] = [0.0, 0.0, 0.0]
        state_np[3:6] = [1.0, 0.0, 0.0]
        state_np[6:9] = [0.0, 0.0, 0.0]
        state_np[9:12] = [0.01, -0.02, 0.005]
        state_np[12:15] = [0.1, -0.1, 0.05]

        input_np = np.zeros(12)
        input_np[:3] = [0.05, -0.03, 0.02]    
        input_np[3:6] = [0.0, 9.81, 0.0]      
        input_np[6:9] = state_np[9:12]        
        input_np[9:12] = state_np[12:15]      


        dt_np = 0.01  # 10ms
        obs_np = np.zeros(3)
        P_np = np.eye(15)
        Q_np = np.eye(12)
        R_np = np.eye(3)
        xp, Pp = imu.forward(state_np, input_np, dt_np, obs_np, P_np, Q_np, R_np)
        xn, Pn = imu.state_propagate(state_np, input_np, dt_np, P_np, Q_np)