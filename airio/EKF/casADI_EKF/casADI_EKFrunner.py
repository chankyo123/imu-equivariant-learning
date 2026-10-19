import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import os
import argparse
import numpy as np
import pypose as pp
import tqdm

import torch
import torch.utils.data as Data

from pyhocon import ConfigFactory
from datasets import imu_seq_collate, SeqInfDataset
 
from utils import CPU_Unpickler, interp_xyz

from utils.visualize_state import visualize_velocity, visualize_rotations

from casADI_ekf import CasADIEKF
from matplotlib import pyplot as plt
from EKF.ekfutil import plot_bias_subplots, interp_xyz
import yaml
import numpy as np

class EKF_runner():
    def __init__(self,
                q=torch.ones(12, dtype=torch.float64) * 0.01,
                r=torch.ones(3,  dtype=torch.float64) * 0.01,
                p=torch.zeros(15, dtype=torch.float64)):
        self.state = torch.zeros(15, dtype=torch.float64)      # x₀
        self.P     = torch.eye(15, dtype=torch.float64) * p**2 # P₀
        self.P_list = [self.P]

        self.current_idx = 0
        self.state_lengeth = 9
        self.window = 1

        self.r = r
        self.q = q
        self.p = p

        self.est_history = []
        self.gravity = np.array([0., 0., 9.81007])
        self.data = None

        self.filter = CasADIEKF()
    
    def get_result(self):
        return torch.stack(self.est_history), torch.stack(self.P_list)

    def propogate_update(self, imu_data, observation, Q=None, R=None):
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input = torch.cat([imu_data["gyro"], imu_data["acc"], d_bias_gyro, d_bias_acc], dim=-1).cpu().numpy()
        dt_np = float(imu_data['dt'])
        z_np = observation.cpu().numpy()
        Q_np = Q.cpu().numpy()
        R_np = R.cpu().numpy()
        x_np, P_np = self.filter.filtering(self.state[...,None].cpu().numpy(),input[...,None],dt_np, z_np, 
                                        self.P.cpu().numpy(), Q_np,R_np)
        self.state = torch.from_numpy(np.array(x_np, dtype=np.float64)).squeeze()
        self.P     = torch.from_numpy(np.array(P_np, dtype=np.float64)).squeeze()
    def propogate_state(self, imu_data, Q=None):
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input = torch.cat([imu_data["gyro"], imu_data["acc"], d_bias_gyro, d_bias_acc], dim=-1).cpu().numpy()
        dt_np = float(imu_data['dt'])
        Q_np = Q.cpu().numpy()

        x_np, P_np = self.filter.predict(self.state[...,None].cpu().numpy(), input[...,None], dt_np, 
                                        self.P.cpu().numpy(), Q_np)
        self.state = torch.from_numpy(np.array(x_np, dtype=np.float64)).squeeze()
        self.P     = torch.from_numpy(np.array(P_np, dtype=np.float64)).squeeze()

    def run(self, imu_data, observation = None, Q=None, R=None):
        if observation is not None:
            self.propogate_update(imu_data, observation, Q=Q, R=R)
        else:
            self.propogate_state(imu_data, Q=Q)
            
        self.est_history.append(self.state.clone())
        self.P_list.append(self.P.clone())
        self.current_idx+=1


def load_results(path):
    if path is not None:
        result_path = os.path.join(path, 'net_output.pickle')
        if os.path.isfile(result_path):
            with open(result_path, 'rb') as handle:
                state_load = CPU_Unpickler(handle).load()
            return state_load
        else:
            raise Exception(f"Unable to load the network result: {result_path}")
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="configs/casADI_EKF/Blackbird/blackbird.yaml", help="the configuration of the dataset")

    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))

    with open(args.conf, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_conf = ConfigFactory.parse_file(config["dataset"]).inference
    exp_conf     = config.get("exp", {})
    train_conf   = config.get("train", {})

    airio_path   = exp_conf.get("airio_exp")
    airimu_path  = exp_conf.get("airimu_exp")
    folder      = exp_conf.get("savedir")
    os.makedirs(folder, exist_ok=True)

    inference_state_load = load_results(airio_path)
    airimu_ori_load = load_results(airimu_path)

    # cov_scale
    imu_cov_scale = train_conf.get("imu_cov_scale", 1.)
    obs_cov_scale = train_conf.get("obs_cov_scale", 1.)

    # bias_weight
    bias_weight = train_conf.get("bias_weight", 1e-12)

    # device
    device = torch.device("cpu")
    gravity_vec = torch.tensor([0., 0., 9.81007], dtype=torch.double)

    for data_conf in dataset_conf.data_list:
        print(data_conf)
        for data_name in data_conf.data_drive:
            print(data_conf.data_root, data_name)
            inference_state = airimu_ori_load[data_name] 
            dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device = device, name = data_conf.name,duration=1, step_size=1, drop_last=False, conf = dataset_conf)
            infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1, 
                                            collate_fn=imu_seq_collate, 
                                            shuffle=False, drop_last=False)

            init = dataset_inf.get_init_value()
            gravity = dataset_inf.get_gravity()
            integrator_outstate = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity=gravity,
                reset=False
            ).to(device).double()
            
            # dict_keys(['cov', 'net_vel', 'ts'])
            io_result = inference_state_load[data_name]

            ekf = EKF_runner()
            
            # STEP 1 state initialization
            initial_state = torch.zeros(15, dtype=torch.float64)
            initial_state[:3] = init["rot"].Log()
            initial_state[3:6] = init["vel"]
            initial_state[6:9] = init["pos"]
            ekf.state = initial_state
            
            # STEP 2 covariance initialization
            # the uncertainty of the initial state
            io_index = 0
            air_imu_index = 0 
            gt_state = {"pos": [], "vel": [], "rot": []}
            
            t_range = tqdm.tqdm(dataset_inf)
            prev_gt_vel = None
            prev_gt_rot = None
            for data in t_range:
                # add the measurement of the airimu 
                imu_data = {"gyro": data["gyro"][0], "acc": data["acc"][0], "dt": data["dt"][0]}
                io_stamp = io_result["ts"][io_index]
                
                # STEP 2 add the learned covariance
                r = torch.ones(3, dtype=torch.float64)
                q = torch.ones(12, dtype=torch.float64) * bias_weight # TODO: NEED TO BE TUNED
                q[:3] = data["gyro_cov"][0]
                q[3:6] = data["acc_cov"][0] * imu_cov_scale

                
                if io_stamp - data["timestamp"].abs() < 0.001:
                    observation = io_result["net_vel"][io_index]
                    cov = io_result["cov"][0][io_index]
                    r[:3] = cov * obs_cov_scale
                    io_index+=1
                else:
                    observation = None

                Q = torch.eye(12, dtype=torch.float64) * q 
                R = torch.eye(3, dtype=torch.float64) * r
                
                ekf.run(imu_data, observation=observation, Q=Q, R=R)
                gt_state["pos"].append(data["gt_pos"][0])
                gt_state["vel"].append(data["gt_vel"][0])
                gt_state["rot"].append(data["gt_rot"][0])

            gtpos = torch.stack(gt_state["pos"])
            gtrot = torch.stack(gt_state["rot"])
            gtvel = torch.stack(gt_state["vel"])
            ekf_result, ekf_cov = ekf.get_result()
            ekf_result = ekf_result.numpy()
            
            if data_conf.name == "BlackBird":
                data_name = os.path.dirname(data_name).split('/')[1]
            else:
                data_name = data_name
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            np.save(os.path.join(folder, f"{data_name}_ekf_result.npy"), ekf_result)
            ekf_poses = np.zeros((ekf_result.shape[0], 8))
            ekf_poses[:, 1:4] = ekf_result[:, 6:9]
            ekf_poses[:, 4:] = pp.so3(ekf_result[:, :3]).Exp().numpy()
            np.save(os.path.join(folder, f"{data_name}_ekf_poses.npy"), ekf_poses)
            
            plt.figure()
            plt.plot(ekf_result[:, 6], ekf_result[:, 7], label="EKF")
            plt.plot(gtpos[:, 0], gtpos[:, 1], label="GT")
            
            plt.savefig(os.path.join(folder, f"{data_name}_ekf_result.png"))
            
            # visualize the net velocity
            io_ts = io_result["ts"][:,0]
            net_vel = io_result["net_vel"]
            interp_net_vel = interp_xyz(dataset_inf.data["time"], io_ts, net_vel.numpy())
            interp_net_vel = dataset_inf.data["gt_orientation"] @ interp_net_vel
            interp_net_vel = interp_net_vel[:len(ekf_result)].numpy()

            plot_bias_subplots(ekf_result[:, 9:12], title="EKF Bias", save_path=os.path.join(folder, f"{data_name}_bias.png"))
            visualize_velocity(f"EKF_vel_{data_name}", gtvel, ekf_result[:, 3:6], interp_net_vel, save_folder=folder)