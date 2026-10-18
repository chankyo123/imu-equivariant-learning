import numpy as np
import torch
from network.covariance_parametrization import DiagonalParam, EquivCovariance
from utils.logging import logging


class MeasSourceTorchScript:
    """ Loading a torchscript has the advantage that we do not need to reconstruct the original network class to
        load the weights, the network structure is contained into the torchscript file.
    """
    def __init__(self, model_path, force_cpu=False):
        # load trained network model
        logging.info("Loding {}...".format(model_path))
        if not torch.cuda.is_available() or force_cpu:
            torch.init_num_threads()
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            self.device = torch.device("cpu")
            self.net = torch.jit.load(model_path, map_location="cpu")
        else:
            self.device = torch.device("cuda:0")
            # NOTE TLIO baseline model won't work on GPU unless we ass map_location
            # https://github.com/pytorch/pytorch/issues/78207
            self.net = torch.jit.load(model_path, map_location=self.device)

        self.net.to(self.device)
        self.net.eval()
        logging.info("Model {} loaded to device {}.".format(model_path, self.device))
        self.model_path = model_path

    def get_displacement_measurement(self, net_gyr_w, net_acc_w, net_vel_body = None, net_ori_b2w = None, input_3 = False, input_4 = False, clip_small_disp=False):
        with torch.no_grad():
            if input_3 : 
                if not input_4 : 
                    features = np.concatenate([net_gyr_w, net_acc_w, net_vel_body], axis=1)  # N x 9
                else:
                    features = np.concatenate([net_gyr_w, net_acc_w, net_vel_body, net_ori_b2w], axis=1)  # N x 18
                    
            else:
                features = np.concatenate([net_gyr_w, net_acc_w], axis=1)  # N x 6
            features_t = torch.unsqueeze(
                torch.from_numpy(features.T).float().to(self.device), 0
            )  # 1 x 6 or 9 x N
            netargs = [features_t]
            outputs = self.net(*netargs)

            if type(outputs) == tuple:  # Legacy
                # if input_3: 
                if "3res" in self.model_path :
                    meas, meas_cov = outputs
                # elif "eq_" in self.model_path : 
                #     a, meas_cov, meas = outputs
                else:
                    meas, meas_cov = outputs
                    # meas, meas_V, meas_cov = outputs
                    meas_cov = meas_cov.reshape(1, -1)
            elif type(outputs) == dict:  # New output format
                meas, meas_cov = outputs["pred"], outputs["pred_log_std"]
                # If this is the case, the network predicts over the whole window at high frequency.
                # TODO utilize the whole window measurements. May improve.
                if meas.dim() == 3:
                    meas = meas[:, -1]
                    meas_cov = meas_cov[:, -1]

            assert meas.dim() == 2  # [B,3]
            assert meas_cov.dim() == 2

            meas = meas.cpu().detach().numpy()
            
            meas_cov[meas_cov < -4] = -4  # exp(-3) =~ 0.05
            meas_cov = DiagonalParam.vec2Cov(meas_cov).cpu().detach().numpy()[0, :, :]
            # meas_cov_tensor = EquivCovariance.vec2Cov(meas_V, meas_cov)
            # meas_cov = meas_cov_tensor.squeeze(0).cpu().detach().numpy()
            # v = meas_cov  # shape: [B, 3]
            # alpha, beta = 0.5, 1.0 

            # vvT = torch.einsum('bi,bj->bij', v, v)  # [B, 3, 3]
            # norm_v_sq = torch.sum(v ** 2, dim=1, keepdim=True)  # [B, 1]
            # I = torch.eye(v.size(1), device=v.device).unsqueeze(0).expand(v.size(0), -1, -1)  # [B, 3, 3]

            # meas_cov = alpha * vvT + beta * (norm_v_sq.unsqueeze(-1) * I - vvT)  # [B, 3, 3]
            # meas_cov = meas_cov.cpu().detach().numpy()[0]  # [3, 3] if B == 1
            
            meas = meas.reshape((3, 1))

            # Our equivalent of zero position update (TODO need stronger prior to keep it still)
            if clip_small_disp and np.linalg.norm(meas) < 0.001:
                meas = 0 * meas
                # meas_cov = 1e-6 * np.eye(3)

            return meas, meas_cov
