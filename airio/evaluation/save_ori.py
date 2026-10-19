# output the trajctory in the world frame for visualization and evaluation
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import argparse
import pypose as pp
import torch.utils.data as Data

from pyhocon import ConfigFactory
from datasets import SeqInfDataset, SeqDataset, imu_seq_collate

from utils import CPU_Unpickler, integrate
from utils.visualize_state import visualize_rotations
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataconf", type=str, help="the configuration of the dataset")
    parser.add_argument("--exp", type=str, default=None, help="Directory of AirIMU outputs ")
    parser.add_argument("--savedir",type=str,default = "./result/loss_result/orientations",help = "save directory")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu, Default is cuda:0")

    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference

    # Load the network results
    if args.exp is not None:
        net_result_path = os.path.join(args.exp, 'net_output.pickle')
        if os.path.isfile(net_result_path):
            with open(net_result_path, 'rb') as handle:
                inference_state_load = CPU_Unpickler(handle).load()
        else:
            raise Exception(f"Unable to load the network result: {net_result_path}")
    
    # Create the output folder
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)
    save_states = {}

    # Process each dataset
    for data_conf in dataset_conf.data_list:
        for data_name in data_conf.data_drive:
            print(f"dataset: {data_conf.name}, sequence: {data_name}")
            save_cur_state = {}
            # DataLoader for the raw IMU data
            dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=200, step_size=200, drop_last=False, conf = dataset_conf)
            loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False, drop_last=False)
            # DataLoader for the AirIMU corrected data
            inference_state = inference_state_load[data_name] 
            dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device = args.device, name = data_conf.name,duration=200, step_size=200, drop_last=False, conf = dataset_conf)
            infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1, 
                                        collate_fn=imu_seq_collate, 
                                        shuffle=False, drop_last=False)

            # Initialize the IMU preintegrator for the raw and AirIMU corrected data
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()
            integrator_outstate = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity=gravity,
                reset=False
            ).to(args.device).double()
            integrator_infstate = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'], gravity = gravity,
                reset=False
            ).to(args.device).double()
            
            # Integrate the raw data and the AirIMU corrected data
            outstate = integrate(
                integrator_outstate, loader, init, 
                device=args.device, gtinit=False, save_full_traj=True,
                use_gt_rot=False
            )
            infstate = integrate(
                integrator_infstate, infloader, init, 
                device=args.device, gtinit=False, save_full_traj=True,
                use_gt_rot=False
            )
            
            # Save the results
            save_cur_state["airimu_rot"] = infstate['orientations'][0]
            save_cur_state["inte_rot"] = outstate['orientations'][0]
            save_states[data_name] = save_cur_state

            # Visualize the results 
            if data_conf.name == "BlackBird":
                save_prefix = os.path.dirname(data_name).split('/')[1]
            else:
                save_prefix = data_name
            visualize_rotations(save_prefix,outstate['orientations_gt'][0],outstate['orientations'][0],inf_rot=infstate['orientations'][0],save_folder=folder)

        net_result_path = os.path.join(folder, 'orientation_output.pickle')
        print("save orientation, ", net_result_path)
        with open(net_result_path, 'wb') as handle:
            pickle.dump(save_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
