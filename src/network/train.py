"""
This file includes the main libraries in the network training module.
"""

import json
import os
import signal
import sys
import time
from functools import partial
from os import path as osp

import numpy as np
import torch
#from dataloader.dataset_fb import FbSequenceDataset
from dataloader.tlio_data import TlioData
from network.losses import get_loss
from network.model_factory import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.logging import logging
from utils.utils import to_device

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(network, data_loader, device, epoch, body_frame_3regress = False, body_frame = False, transforms=[]):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    targets_vel_all, preds_vel_all = [], []
    network.eval()

    for bid, sample in enumerate(data_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        if body_frame_3regress: 
            pred, pred_cov, pred_vel = network(feat)
            # print(feat.shape, pred.shape, pred_cov.shape)
        else:
            pred, pred_cov = network(feat)
            pred_vel = pred.clone()
        
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
            if body_frame: 
                # print("body frame is running in inference!!")
                targ_vel = sample["vel_Body"][:,-1,:]
                targ = sample["vel_Body"][:,-1,:]
                # print("Jere!!!!")
                # targ_vel = sample["targ_dt_Body"][:,-1,:]
                # targ = sample["targ_dt_Body"][:,-1,:]
            else:
                # targ_vel = sample["vel_World"][:,-1,:]
                targ_vel = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)

        if body_frame_3regress: 
            loss = get_loss(pred_vel, pred_cov, targ_vel, epoch, body_frame_3regress)
        else:
            #1. use dt as target
            # print("second loss is running in inference!")
            loss = get_loss(pred, pred_cov, targ, epoch, False)
            
            #2. use v as target
            # loss = get_loss(pred, pred_cov, targ_vel, epoch, False)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        targets_vel_all.append(torch_to_numpy(targ_vel))
        preds_vel_all.append(torch_to_numpy(pred_vel))
        
        losses_all.append(torch_to_numpy(loss))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    targets_vel_all = np.concatenate(targets_vel_all, axis=0)
    preds_vel_all = np.concatenate(preds_vel_all, axis=0)
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
        "targets_vel": targets_vel_all,
        "preds_vel": preds_vel_all,
    }
    return attr_dict


def do_train(network, train_loader, device, epoch, optimizer, input_dim, transforms=[], body_frame_3regress = False, body_frame = False):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    train_targets_vel, train_preds_vel  = [], []
    network.train()

    #for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        if bid <6 and (epoch< 11 or (epoch <= 100 and epoch>=90) or (epoch <=200 and epoch >= 190)):
            print('bid = ', bid)
            
        # >>> SO(3) Equivariance Check
        rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        accelerometer_data = feat[:, :3,:].to(torch.float32)
        accelerometer_data = accelerometer_data.permute(1,0,2).reshape(3,-1)
        gyroscope_data = feat[:, 3:6, :].to(torch.float32)
        gyroscope_data = gyroscope_data.permute(1,0,2).reshape(3,-1)
        if input_dim in (9, 18):
            vel_data = feat[:, 6:9, :].to(torch.float32)
            vel_data = vel_data.permute(1,0,2).reshape(3,-1)
            if input_dim == 18:
                ori_data_1 = feat[:, 9:12, :].to(torch.float32)
                ori_data_2 = feat[:, 12:15, :].to(torch.float32)
                ori_data_3 = feat[:, 15:18, :].to(torch.float32)
                ori_data_1 = ori_data_1.permute(1,0,2).reshape(3,-1)
                ori_data_2 = ori_data_2.permute(1,0,2).reshape(3,-1)
                ori_data_3 = ori_data_3.permute(1,0,2).reshape(3,-1)
                
        rotated_accelerometer_data = torch.matmul(rotation_matrix, accelerometer_data)
        rotated_accelerometer_data = rotated_accelerometer_data.reshape(rotated_accelerometer_data.size(0), feat.size(0), feat.size(2))
        rotated_accelerometer_data = rotated_accelerometer_data.permute(1,0,2)
        rotated_gyroscope_data = torch.matmul(rotation_matrix, gyroscope_data)
        rotated_gyroscope_data = rotated_gyroscope_data.reshape(rotated_gyroscope_data.size(0), feat.size(0), feat.size(2))
        rotated_gyroscope_data = rotated_gyroscope_data.permute(1,0,2)
        if input_dim in (9, 18):
            rotated_vel_data = torch.matmul(rotation_matrix, vel_data)
            rotated_vel_data = rotated_vel_data.reshape(rotated_vel_data.size(0), feat.size(0), feat.size(2))
            rotated_vel_data = rotated_vel_data.permute(1,0,2)
            if input_dim == 18:
                rotated_ori_data1 = torch.matmul(rotation_matrix, ori_data_1)
                rotated_ori_data2 = torch.matmul(rotation_matrix, ori_data_2)
                rotated_ori_data3 = torch.matmul(rotation_matrix, ori_data_3)
                rotated_ori_data1 = rotated_ori_data1.reshape(rotated_ori_data1.size(0), feat.size(0), feat.size(2))
                rotated_ori_data1 = rotated_ori_data1.permute(1,0,2)
                rotated_ori_data2 = rotated_ori_data2.reshape(rotated_ori_data2.size(0), feat.size(0), feat.size(2))
                rotated_ori_data2 = rotated_ori_data2.permute(1,0,2)
                rotated_ori_data3 = rotated_ori_data3.reshape(rotated_ori_data3.size(0), feat.size(0), feat.size(2))
                rotated_ori_data3 = rotated_ori_data3.permute(1,0,2)
        
        if input_dim in (9, 18):
            feat_rot = torch.cat((rotated_accelerometer_data, rotated_gyroscope_data, rotated_vel_data), dim=1)
            if input_dim == 18:
                feat_rot = torch.cat((rotated_accelerometer_data, rotated_gyroscope_data, rotated_vel_data, rotated_ori_data1, rotated_ori_data2, rotated_ori_data3), dim=1)
        else:
            feat_rot = torch.cat((rotated_accelerometer_data, rotated_gyroscope_data), dim=1)
        # <<< SO(3) Equivariance Check
        
        if body_frame_3regress:
            pred, pred_cov, pred_vel = network(feat)
        else:
            pred, pred_cov= network(feat)
            pred_vel = pred.clone()
            
        if body_frame_3regress: 
            pred_rot, pred_cov_rot, pred_vel_rot = network(feat_rot)
        else:
            pred_rot, pred_cov_rot = network(feat_rot)
        # # print(torch.matmul(pred_vel,rotation_matrix)[:3,:3])
        # # print(pred_vel_rot[:3,:3])
        # # print()

        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
            if body_frame:
                # print("body frame is running in training!!")
                targ_vel = sample["vel_Body"][:,-1,:]
                targ = sample["vel_Body"][:,-1,:]
                # print("Jere!!")
                # targ_vel = sample["targ_dt_Body"][:,-1,:]
                # targ = sample["targ_dt_Body"][:,-1,:]
            else:
                # 1. learn vel
                # targ_vel = sample["vel_World"][:,-1,:]
                # 2. learn disp
                targ_vel = sample["targ_dt_World"][:,-1,:]
                targ = sample["vel_Body"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)

        if body_frame_3regress: 
            loss = get_loss(pred_vel, pred_cov, targ_vel, epoch, body_frame_3regress)
        else:
            #1. use dt as target
            # print("second loss is running in training!")
            loss = get_loss(pred, pred_cov, targ, epoch, False)
            
            # 2. use v as target
            # loss = get_loss(pred, pred_cov, targ_vel, epoch, False)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
        train_targets_vel.append(torch_to_numpy(targ_vel))
        train_preds_vel.append(torch_to_numpy(pred_vel))
            
        #print("Loss full: ", loss)

        loss = loss.mean()
        loss.backward()

        #print("Loss mean: ", loss.item())
        
        # print("Gradients:")
        # for name, param in network.named_parameters():
        #    if param.requires_grad:
        #         print(f'Parameter: {name} - gradient statistics:')
        #         print(name, ": ", param.grad)
                # print(f'  - Max: {torch.max(param.grad).item()}')
                # print(f'  - Min: {torch.min(param.grad).item()}')
                # print(f'  - Mean: {torch.mean(param.grad).item()}')
                # print(f'  - Std: {torch.std(param.grad).item()}')

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)
        # torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=False)
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_targets_vel = np.concatenate(train_targets_vel, axis=0)
    train_preds_vel = np.concatenate(train_preds_vel, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
        "targets_vel": train_targets_vel,
        "preds_vel": train_preds_vel,
    }
    return train_attr_dict


def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """

    mse_loss = np.mean((attr_dict["targets"] - attr_dict["preds"]) ** 2, axis=0)
    ml_loss = np.average(attr_dict["losses"])
    sigmas = np.exp(attr_dict["preds_cov"])
    # If it's sequential, take the last one
    if len(mse_loss.shape) == 2:
        assert mse_loss.shape[0] == 3
        mse_loss = mse_loss[:, -1]
        assert sigmas.shape[1] == 3
        sigmas = sigmas[:,:,-1]
    summary_writer.add_scalar(f"{mode}_loss/loss_x", mse_loss[0], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_y", mse_loss[1], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_z", mse_loss[2], epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", np.mean(mse_loss), epoch)
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_x", sigmas[:, 0], epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_y", sigmas[:, 1], epoch)
    summary_writer.add_histogram(f"{mode}_hist/sigma_z", sigmas[:, 2], epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(
        f"{mode}: average ml loss: {ml_loss}, average mse loss: {mse_loss}/{np.mean(mse_loss)}"
    )


def save_model(args, epoch, network, optimizer, best, interrupt=False):
    if interrupt:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
    if best:
        model_path = osp.join(args.out_dir, "checkpoint_best.pt")        
    else:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dict(
        [
            ("past_data_size", int(args.past_time * args.imu_freq)),
            ("window_size", int(args.window_time * args.imu_freq)),
            ("future_data_size", int(args.future_time * args.imu_freq)),
            ("step_size", int(args.imu_freq / args.sample_freq)),
        ]
    )
    net_config = {
        "in_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 32
        + 1
    }

    return data_window_config, net_config


def net_train(args):
    """
    Main function for network training
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        #if args.train_list is None:
        #    raise ValueError("train_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not osp.isdir(osp.join(args.out_dir, "checkpoints")):
                os.makedirs(osp.join(args.out_dir, "checkpoints"))
            if not osp.isdir(osp.join(args.out_dir, "logs")):
                os.makedirs(osp.join(args.out_dir, "logs"))
            with open(
                os.path.join(args.out_dir, "parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        #if args.val_list is None:
        #    logging.warning("val_list is not specified.")
        if args.continue_from is not None:
            if osp.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Perturb on bias: %s" % args.do_bias_shift)
    logging.info("Perturb on gravity: %s" % args.perturb_gravity)
    logging.info("Sample frequency: %s" % args.sample_freq)

    train_loader, val_loader = None, None
    start_t = time.time()
    
    data = TlioData(
        args.root_dir, 
        batch_size=args.batch_size, 
        dataset_style=args.dataset_style, 
        num_workers=args.workers,
        persistent_workers=args.persistent_workers,
        window_time=args.window_time
    )
    data.prepare_data()
    
    train_list = data.get_datalist("train")

    """
    try:
        train_dataset = FbSequenceDataset(
            args.root_dir, train_list, args, data_window_config, mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
    except OSError as e:
        logging.error(e)
        return
    """
    train_loader = data.train_dataloader()
    train_transforms = data.get_train_transforms()

    #train (world/bodyframe) imu data
    # body_frame = True
    # body_frame_3regress = True
    
    # body_frame = False
    # body_frame_3regress = False
    
    body_frame = eval(args.body_frame)
    if "3res" in args.out_dir:
        print("we regress 2 value!!")
        body_frame_3regress = False
    elif "/res" in args.out_dir or "resnet" in args.out_dir or "/eq_" in args.out_dir or "/vn_" in args.out_dir or "/ln_" in args.out_dir:
        body_frame_3regress = False
    else:
        body_frame_3regress = True
    if not body_frame: 
        train_transforms = data.get_train_transforms()
    else:
        train_transforms = data.get_train_transforms_bodyframe()
        
    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(data.train_dataset)}")

    #if args.val_list is not None:
    if data.val_dataset is not None:
        val_list = data.get_datalist("val")
        """
        try:
            val_dataset = FbSequenceDataset(
                args.root_dir, val_list, args, data_window_config, mode="val"
            )
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
        except OSError as e:
            logging.error(e)
            return
        """
        val_loader = data.val_dataloader()
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(data.val_dataset)}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim)
    network.to(device)
    total_params = network.get_num_params()
    logging.info(f'Network "{args.arch}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")

    # optimizer = torch.optim.Adam(network.parameters(), args.lr)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
        optimizer, factor=0.2, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from is not None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    summary_writer = SummaryWriter(osp.join(args.out_dir, "logs"))
    summary_writer.add_text("info", f"total_param: {total_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")
    #attr_dict = get_inference(network, train_loader, device, start_epoch, train_transforms)
    #write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train")
    #if val_loader is not None:
    #    attr_dict = get_inference(network, val_loader, device, start_epoch)
    #    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "val")

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, best=False, interrupt=True)
        sys.exit()

    best_val_loss = np.inf
    consumed_times = []
    consumed_gpu = []
    
    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        train_attr_dict = do_train(network, train_loader, device, epoch, optimizer, args.input_dim, train_transforms, body_frame_3regress, body_frame)
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
        torch.cuda.reset_peak_memory_stats()
        mem_str = f'GPU Mem: {mem_used_max_GB:.3f}GB'
        logging.info(mem_str)
        consumed_gpu.append(mem_used_max_GB)    
        
        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        time_mem_log = osp.join(args.out_dir, 'time_mem_log.txt')
        with open(time_mem_log, 'w') as log_file:
            log_file.write(f"Epoch {epoch}: {end_t - start_t:.4f} seconds, {mem_used_max_GB:.3f}GB\n")
        consumed_times.append(end_t - start_t)    
            
        if val_loader is not None:
            val_attr_dict = get_inference(network, val_loader, device, epoch, body_frame_3regress, body_frame)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer, best=True)
            scheduler.step(np.mean(val_attr_dict["losses"]))
        else:
            save_model(args, epoch, network, optimizer, best=False)
            
        if epoch in {8,9,19,49, 80, 90, 95, 96, 98, 99, 109, 119, 129, 149, 159, 179, 189, 195, 199, 209,219,229,239,249,259,269,299, 399}:
            save_model(args, epoch, network, optimizer, best=False, interrupt=False)
    
    mean_epoch_time = np.mean(consumed_times)
    mean_epoch_gpu = np.mean(consumed_gpu)
    with open(time_mem_log, 'a') as log_file:
        log_file.write(f"Mean Epoch Time: {mean_epoch_time:.4f} seconds, {mean_epoch_gpu:.3f}GB\n")
    logging.info(f"time usage: {mean_epoch_time:.3f}s")
    mem_str = f'GPU Mem: {mean_epoch_gpu:.3f}GB'
    logging.info(mem_str)
    
    logging.info("Training complete.")

    return
