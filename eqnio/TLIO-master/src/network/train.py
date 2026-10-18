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
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def torch_to_numpy(torch_arr):
    return torch_arr.detach().cpu().numpy()

def preprocess_for_eq_transformer(sample):
    ## read the positional encoded timestamps
    scalar = sample["feats"]["pe_ts"]
    ## read the imu data
    feat = sample["feats"]["imu0"]
    feat = feat.permute(0,2,1)
    ## separate the vectors and scalars from features - gyro, accel
    gyro = torch.cat((feat[..., 0].unsqueeze(-1), feat[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    accel = torch.cat((feat[..., 3].unsqueeze(-1), feat[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    vector = torch.cat((accel, gyro), dim=-2)
    scalar = torch.cat((scalar, feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1), torch.norm(accel, dim=-1), torch.norm(gyro, dim=-1), torch.einsum('... a b, ... c b -> ... c', accel, gyro)), dim=-1)
    return vector.permute(0,1,3,2), scalar  

def postprocess_eq_transformer(out_vec, out_sca):
    return torch.cat((out_vec.squeeze(-1), out_sca[..., 0].unsqueeze(-1)), dim=-1), out_sca[..., 1:]

def postprocess_eq_transformer_3scalars(out_vec, out_sca):
    return torch.cat((out_vec.squeeze(-1), out_sca[..., 0].unsqueeze(-1)), dim=-1), torch.cat((out_sca[..., 1].unsqueeze(-1), out_sca[..., 1].unsqueeze(-1), out_sca[..., 2].unsqueeze(-1)), dim=-1)

def postprocess_eq_transformer_6v_6s(out_vec, out_sca):
    cov = torch.cat((out_vec[..., 1:], out_sca[..., 1:].unsqueeze(-2)), dim=-2)
    mu = cov.mean(-1)
    cov = (torch.matmul((cov - mu.unsqueeze(-1)), torch.transpose(cov - mu.unsqueeze(-1), 1, 2))/(cov.shape[-1]-1))
    return torch.cat((out_vec[..., 0].squeeze(-1), out_sca[..., 0].unsqueeze(-1)), dim=-1), cov

def preprocess_wo_t(sample):
    ## read the imu data
    feat = sample["feats"]["imu0"]
    feat = feat.permute(0,2,1)
    ## separate the vectors and scalars from features - gyro, accel
    gyro = torch.cat((feat[..., 0].unsqueeze(-1), feat[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    accel = torch.cat((feat[..., 3].unsqueeze(-1), feat[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    vector = torch.cat((accel, gyro), dim=-2)
    scalar = torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1), torch.norm(accel, dim=-1), torch.norm(gyro, dim=-1), torch.einsum('... a b, ... c b -> ... c', accel, gyro)), dim=-1)
    return vector.permute(0,1,3,2), scalar 

def preprocess_wo_t_tlio_frame(sample):
    ## read the imu data
    feat = sample["feats"]["imu0"]
    feat = feat.permute(0,2,1)
    ## separate the vectors and scalars from features - gyro, accel
    gyro = torch.cat((feat[..., 0].unsqueeze(-1), feat[..., 1].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    accel = torch.cat((feat[..., 3].unsqueeze(-1), feat[..., 4].unsqueeze(-1)), dim=-1).unsqueeze(-2)
    vector = torch.cat((accel, gyro), dim=-2)
    scalar = torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1), torch.norm(accel, dim=-1), torch.norm(gyro, dim=-1), torch.einsum('... a b, ... c b -> ... c', accel, gyro)), dim=-1)
    return vector.permute(0,1,3,2), scalar, torch.cat((feat[..., 2].unsqueeze(-1), feat[..., -1].unsqueeze(-1)), dim=-1)

def postprocess_tlio_frame_2scalars(pred, cov):
    return pred, torch.cat((cov[..., 0].unsqueeze(-1), cov[..., 0].unsqueeze(-1), cov[..., 1].unsqueeze(-1)), dim=-1)

def preprocess_tlio_w_t(sample): # 9 time positional encoding values
    return torch.cat((sample["feats"]["imu0"].permute(0,2,1),sample["feats"]["pe_ts"]), dim=-1).permute(0,2,1)

def preprocess_o2(sample):
    feat = sample['feats']['feat_o2'].permute(0,2,1).float()
    a = feat[...,:2].unsqueeze(-2)
    v1 = feat[...,3:5].unsqueeze(-2)
    v2 = feat[...,6:-1].unsqueeze(-2)
    scalar = torch.cat((feat[...,2].unsqueeze(-1), feat[...,5].unsqueeze(-1), feat[...,-1].unsqueeze(-1),torch.norm(a,dim=-1),torch.norm(v1,dim=-1),torch.norm(v2,dim=-1),(a@v1.permute(0,1,3,2)).squeeze(-1),(v1@v2.permute(0,1,3,2)).squeeze(-1),(a@v2.permute(0,1,3,2)).squeeze(-1)),dim=-1)
    return torch.cat((a,v1,v2),dim=-2).permute(0,1,3,2), scalar, torch.cat((feat[...,2].unsqueeze(-1), feat[...,5].unsqueeze(-1), feat[...,-1].unsqueeze(-1)), dim=-1)

def preprocess_so3(sample):
    """SO(3) preprocessing for Eq_Motion_Model_SO3_fullCov.
    Reuses the same feat_o2 cache produced by the dataloader (9 channels = 3 vectors x 3D)
    but keeps the full 3D vectors and exposes only SO(3)-invariant scalars (norms + dot products).
    Returns:
        vector: (B, T, vec_dim=3, F=3)  -- 3 input features (a, v1, v2) as full 3D vectors
        scalar: (B, T, 6)               -- 3 norms + 3 dot products (all SO(3)-invariant)
    """
    feat = sample['feats']['feat_o2'].permute(0,2,1).float()  # (B, T, 9)
    a  = feat[..., 0:3]
    v1 = feat[..., 3:6]
    v2 = feat[..., 6:9]
    na  = torch.norm(a, dim=-1, keepdim=True)
    nv1 = torch.norm(v1, dim=-1, keepdim=True)
    nv2 = torch.norm(v2, dim=-1, keepdim=True)
    av1  = (a  * v1).sum(-1, keepdim=True)
    v1v2 = (v1 * v2).sum(-1, keepdim=True)
    av2  = (a  * v2).sum(-1, keepdim=True)
    scalar = torch.cat([na, nv1, nv2, av1, v1v2, av2], dim=-1)        # (B, T, 6)
    vector = torch.stack([a, v1, v2], dim=-1)                          # (B, T, 3, 3)
    return vector, scalar


def get_inference(network, data_loader, device, epoch, arch_type, transforms=[]):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all, frames_all = [], [], [], [], []

    # if arch_type == 'eq_transformer': ## just for einsum rn
    #     network = network.double()
    network.eval()

    for bid, sample in enumerate(data_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)

        frame = None

        if 'resnet_2scalars_tlio_frame' in arch_type:
            frame,out_vec, out_sca = network(sample["feats"]["imu0"])
            pred, pred_cov = postprocess_tlio_frame_2scalars(out_vec, out_sca)
        elif 'resnet_fullCov_tlio_frame' in arch_type:
            frame,pred, pred_cov = network(sample["feats"]["imu0"])
        elif '_3scalars' in arch_type and 'wo_t' not in arch_type:
            vector, scalar = preprocess_for_eq_transformer(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_3scalars(out_vec, out_sca)
        elif '_6v_6s' in arch_type and 'wo_t' not in arch_type:
            vector, scalar = preprocess_for_eq_transformer(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_6v_6s(out_vec, out_sca)
        elif '_wo_t_6v_6s' in arch_type:
            vector, scalar = preprocess_wo_t(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_6v_6s(out_vec, out_sca)
        elif '_wo_t_3scalars' in arch_type:
            vector, scalar = preprocess_wo_t(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_3scalars(out_vec, out_sca)
        elif '_wo_t_tlio_frame_2scalars' in arch_type:
            vector, scalar, orig_sca = preprocess_wo_t_tlio_frame(sample)
            frame,out_vec, out_sca = network(vector.float(), scalar.float(), orig_sca.float())
            pred, pred_cov = postprocess_tlio_frame_2scalars(out_vec, out_sca)
        elif 'o2_frame_2scalars' in arch_type:
            frame,out_vec, out_sca = network(*preprocess_o2(sample))
            pred, pred_cov = postprocess_tlio_frame_2scalars(out_vec, out_sca)
        elif 'so3_frame' in arch_type:
            # Our SO(3)-equivariant extension. preprocess_so3 returns (vector_3D, scalar_6D).
            frame, pred, pred_cov = network(*preprocess_so3(sample))
        elif 'o2_frame' in arch_type:
            frame,pred,pred_cov = network(*preprocess_o2(sample))
        elif 'tlio_frame_fullCov' in arch_type or '_PearsonCov' in arch_type:
            vector, scalar, orig_sca = preprocess_wo_t_tlio_frame(sample)
            frame,pred, pred_cov = network(vector.float(), scalar.float(), orig_sca.float())
        
        elif arch_type == 'resnet_w_t':
            feat = preprocess_tlio_w_t(sample)
            pred, pred_cov = network(feat.float())
        else:
            feat = sample["feats"]["imu0"]
            pred, pred_cov = network(feat)
        
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)

        if ('_fullCov' in arch_type or '_PearsonCov' in arch_type) and '_frame_avg' not in arch_type:
            if 'so3_frame' in arch_type:
                # SO(3) full 3D rotation: targ_canonical = frame @ targ_world (frame is R^T)
                targ = torch.einsum('bij,bj->bi', frame, targ)
            else:
                targ = torch.concat([torch.matmul(frame, targ[:,:2].unsqueeze(-1)).squeeze(-1),targ[:,-1].unsqueeze(-1)], dim=-1)
            
        if '_frame' in arch_type:
            frames_all.append(torch_to_numpy(frame))
        loss = get_loss(pred, pred_cov, targ, epoch, arch_type)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        losses_all.append(torch_to_numpy(loss))
        # torch.cuda.empty_cache()

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)

    if '_frame' in arch_type:
        frames_all = np.concatenate(frames_all, axis=0)
        attr_dict = {
            "targets": targets_all,
            "preds": preds_all,
            "preds_cov": preds_cov_all,
            "losses": losses_all,
            "frames":frames_all,
            }
    else:
        attr_dict = {
            "targets": targets_all,
            "preds": preds_all,
            "preds_cov": preds_cov_all,
            "losses": losses_all,
        }
    return attr_dict


def do_train(network, train_loader, device, epoch, optimizer, arch_type, transforms=[]):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses, frames_all = [], [], [], [], []
    
    # if arch_type == 'eq_transformer': ## just for einsum rn
    #     network = network.double()
    network.train()

    #for bid, (feat, targ, _, _) in enumerate(train_loader):
    for bid, sample in enumerate(tqdm(train_loader)):
        # print('entered the loader embedding')
        sample = to_device(sample, device)
        type(sample)
        for transform in transforms:
            sample = transform(sample)
        frame = None

        if 'resnet_2scalars_tlio_frame' in arch_type:
            frame,out_vec, out_sca = network(sample["feats"]["imu0"])
            pred, pred_cov = postprocess_tlio_frame_2scalars(out_vec, out_sca)
        elif 'resnet_fullCov_tlio_frame' in arch_type:
            frame,pred, pred_cov = network(sample["feats"]["imu0"])
        elif  '_3scalars' in arch_type and 'wo_t' not in arch_type:
            vector, scalar = preprocess_for_eq_transformer(sample)
            optimizer.zero_grad()
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_3scalars(out_vec, out_sca)
        elif '_6v_6s' in arch_type and 'wo_t' not in arch_type:
            vector, scalar = preprocess_for_eq_transformer(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_6v_6s(out_vec, out_sca)
        elif '_wo_t_6v_6s' in arch_type:
            vector, scalar = preprocess_wo_t(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_6v_6s(out_vec, out_sca)
        elif '_wo_t_3scalars' in arch_type:
            vector, scalar = preprocess_wo_t(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_3scalars(out_vec, out_sca)
        elif '_wo_t_tlio_frame_2scalars' in arch_type:
            vector, scalar, orig_sca = preprocess_wo_t_tlio_frame(sample)
            frame,out_vec, out_sca = network(vector.float(), scalar.float(), orig_sca.float())
            pred, pred_cov = postprocess_tlio_frame_2scalars(out_vec, out_sca)
        elif 'o2_frame_2scalars' in arch_type:
            frame,out_vec, out_sca = network(*preprocess_o2(sample))
            pred, pred_cov = postprocess_tlio_frame_2scalars(out_vec, out_sca)
        elif 'so3_frame' in arch_type:
            # Our SO(3)-equivariant extension. preprocess_so3 returns (vector_3D, scalar_6D).
            frame, pred, pred_cov = network(*preprocess_so3(sample))
        elif 'o2_frame' in arch_type:
            frame,pred,pred_cov = network(*preprocess_o2(sample))
        elif 'tlio_frame_fullCov' in arch_type or '_PearsonCov' in arch_type:
            vector, scalar, orig_sca = preprocess_wo_t_tlio_frame(sample)
            frame,pred, pred_cov = network(vector.float(), scalar.float(), orig_sca.float())
        elif arch_type == 'resnet_w_t':
            feat = preprocess_tlio_w_t(sample)
            pred, pred_cov = network(feat.float())
        else:
            feat = sample["feats"]["imu0"]
            optimizer.zero_grad()
            pred, pred_cov = network(feat)

        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:,-1,:]
        else:
            # Leave off zeroth element since it's 0's. Ex: Net predicts 199 if there's 200 GT
            targ = sample["targ_dt_World"][:,1:,:].permute(0,2,1)
        
        if ('_fullCov' in arch_type or '_PearsonCov' in arch_type) and '_frame_avg' not in arch_type:
            if 'so3_frame' in arch_type:
                # SO(3) full 3D rotation: targ_canonical = frame @ targ_world (frame is R^T)
                targ = torch.einsum('bij,bj->bi', frame, targ)
            else:
                targ = torch.concat([torch.matmul(frame, targ[:,:2].unsqueeze(-1)).squeeze(-1),targ[:,-1].unsqueeze(-1)], dim=-1)
            
        if '_frame' in arch_type:
            frames_all.append(torch_to_numpy(frame))
        loss = get_loss(pred, pred_cov, targ, epoch, arch_type)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov))
        train_losses.append(torch_to_numpy(loss))
        
            
        #print("Loss full: ", loss)

        loss = loss.mean()
        loss.backward()

        # pclsss mean: ", loss.item())
        
        #print("Gradients:")
        # for name, param in network.named_parameters():
        #    if param.requires_grad:
        #        print(name, ": ", param.grad)

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=False)#0.1
        optimizer.step()
        # del pred_cov
        # del pred
        # torch.cuda.empty_cache()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)

    if '_frame' in arch_type:
        frames_all = np.concatenate(frames_all, axis=0)
        train_attr_dict = {
            "targets": train_targets,
            "preds": train_preds,
            "preds_cov": train_preds_cov,
            "losses": train_losses,
            "frames":frames_all,
            }
    else:
        train_attr_dict = {
            "targets": train_targets,
            "preds": train_preds,
            "preds_cov": train_preds_cov,
            "losses": train_losses,
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
        # if args.train_list is None:
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
        # if args.val_list is None:
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

    theta_range_deg = 0
    dataloader_bias_gravity_aug = False
    accel_bias_range = 0
    gyro_bias_range = 0
    if 'o2' in args.arch:
        theta_range_deg = 5
        dataloader_bias_gravity_aug = True
        accel_bias_range = 0.2
        gyro_bias_range = 0.05
    
    data = TlioData(
        args.root_dir,
        window_size = data_window_config['window_size']+data_window_config["past_data_size"],
        batch_size=args.batch_size,
        dataset_style=args.dataset_style,
        num_workers=args.workers,
        persistent_workers=args.persistent_workers,
        start_index = data_window_config["past_data_size"],
        theta_range_deg = theta_range_deg,
        dataloader_bias_gravity_aug = dataloader_bias_gravity_aug,
        accel_bias_range = accel_bias_range,
        gyro_bias_range = gyro_bias_range,
        so3_augment = getattr(args, "so3_augment", False),
        so3_augment_max_deg = getattr(args, "so3_augment_max_deg", 180.0),
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

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, eps=1e-12
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
    all_mse_loss = []
    all_mse_loss_x = []
    all_mse_loss_y = []
    all_mse_loss_z = []
    all_ml_loss = []
    sigmas_after_training = np.empty(0)
    all_mse_loss_val = []
    all_mse_loss_x_val = []
    all_mse_loss_y_val = []
    all_mse_loss_z_val = []
    all_ml_loss_val = []
    sigmas_after_training_val = np.empty(0)
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
        train_attr_dict = do_train(network, train_loader, device, epoch, optimizer, args.arch, train_transforms)

        mse_loss = np.mean((train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2, axis=0)
        ml_loss = np.average(train_attr_dict["losses"])
        
        all_mse_loss_x.append(mse_loss[0])
        all_mse_loss_y.append(mse_loss[1])
        all_mse_loss_z.append(mse_loss[2])
        all_mse_loss.append(np.mean(mse_loss))
        all_ml_loss.append(ml_loss)


        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")
        save_model(args, epoch, network, optimizer, best=False)

        if val_loader is not None:
            val_attr_dict = get_inference(network, val_loader, device, epoch, args.arch)

            mse_loss_val = np.mean((val_attr_dict["targets"] - val_attr_dict["preds"]) ** 2, axis=0)
            ml_loss_val = np.average(val_attr_dict["losses"])
            all_mse_loss_x_val.append(mse_loss_val[0])
            all_mse_loss_y_val.append(mse_loss_val[1])
            all_mse_loss_z_val.append(mse_loss_val[2])
            all_mse_loss_val.append(np.mean(mse_loss_val))
            all_ml_loss_val.append(ml_loss_val)

            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer, best=True)
                sigmas_after_training = np.exp(train_attr_dict["preds_cov"])
                sigmas_after_training_val = np.exp(val_attr_dict["preds_cov"])
        else:
            save_model(args, epoch, network, optimizer, best=False)
            sigmas_after_training = np.exp(train_attr_dict["preds_cov"])
            sigmas_after_training_val = np.exp(val_attr_dict["preds_cov"])

        ## saving the losses to plot graphs on training
        res_np_array = np.array([all_mse_loss_x, all_mse_loss_y, all_mse_loss_z, all_mse_loss, all_ml_loss])
        np.save(osp.join(args.out_dir, "training_losses.npy"),res_np_array.T)

        val_res_np_array = np.array([all_mse_loss_x_val, all_mse_loss_y_val, all_mse_loss_z_val, all_mse_loss_val, all_ml_loss_val])
        np.save(osp.join(args.out_dir, "val_losses_during_training.npy"),val_res_np_array.T)


        ## saving the sigmas for the training and val dataset corresponding to last checkpoint
        np.save(osp.join(args.out_dir, "sigmas_after_training.npy"),sigmas_after_training)
        np.save(osp.join(args.out_dir, "val_sigmas_after_training.npy"),sigmas_after_training_val)


    logging.info("Training complete.")

    return
