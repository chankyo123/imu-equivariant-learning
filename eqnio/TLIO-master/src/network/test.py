"""
This file includes the main libraries in the network testing module
"""

import json
import os
from os import path as osp

import matplotlib.pyplot as plt
import torch
#from dataloader.dataset_fb import FbSequenceDataset
from dataloader.tlio_data import TlioData
from dataloader.memmapped_sequences_dataset import MemMappedSequencesDataset
from dataloader.data_transform import TransformAddNoiseBias, TransformPerturbGravity, TransformInYawPlane
from network.losses import get_loss
from network.model_factory import get_model
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from utils.dotdict import dotdict
from utils.utils import to_device
from utils.logging import logging
from utils.math_utils import *


def compute_rpe(rpe_ns, ps, ps_gt, yaw, yaw_gt):
    import numpy as _np
    ns = ps_gt.shape[0]
    if ns - rpe_ns <= 100:
        # sequence too short for RPE — return NaN markers, skip
        return float('nan'), float('nan'), float('nan'), _np.zeros((0, 3))
    assert ps.shape == ps_gt.shape
    assert yaw.shape == yaw_gt.shape

    rpes = []
    relative_yaw_errors = []
    for i in range(0, ns - rpe_ns, 100):
        chunk = ps[i : i + rpe_ns, :]
        chunk_gt = ps_gt[i : i + rpe_ns, :]
        chunk_yaw = yaw[i : i + rpe_ns, :]
        chunk_yaw_gt = yaw_gt[i : i + rpe_ns, :]
        initial_error_yaw = wrap_rpy(chunk_yaw[0, :] - chunk_yaw_gt[0, :])
        # Patch: scipy version differences sometimes return 2D as_matrix for 1-elem yaw → handle both
        R_init = Rotation.from_euler("z", float(np.asarray(initial_error_yaw).flatten()[0]), degrees=True).as_matrix()  # (3,3)
        rel_p = (chunk[-1, :] - chunk[0, :]).reshape(3, 1)                 # (3, 1)
        rotated = (R_init @ rel_p).T                                       # (1, 3)
        final_error_p_relative = rotated - (chunk_gt[[-1], :] - chunk_gt[[0], :])  # (1, 3)
        final_error_yaw = wrap_rpy(chunk_yaw[[-1], :] - chunk_yaw_gt[[-1], :])
        rpes.append(final_error_p_relative)
        relative_yaw_errors.append(wrap_rpy(final_error_yaw - initial_error_yaw))
    rpes = np.concatenate(rpes, axis=0)
    relative_yaw_errors = np.concatenate(relative_yaw_errors, axis=0)

    plt.figure("relative yaw error")
    plt.plot(relative_yaw_errors)
    plt.figure("rpes list")
    plt.plot(rpes)
    # compute statistics over z separately
    rpe_rmse = np.sqrt(np.mean(np.sum(rpes ** 2, axis=1)))
    rpe_rmse_z = np.sqrt(np.mean(rpes[:, 2] ** 2))
    relative_yaw_rmse = np.sqrt(np.mean(relative_yaw_errors ** 2))
    return rpe_rmse, rpe_rmse_z, relative_yaw_rmse, rpes


def pose_integrate(args, dataset, preds, frame = None):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    

    if frame is not None:
        if frame.shape[-1] == 3:
            # SO(3) full 3D frame: apply frame.T (i.e. R) to full 3D prediction
            preds = np.einsum('tji,tj->ti', frame, preds)
        else:
            # O(2) 2x2 frame: rotate xy only, keep z untouched
            w = np.einsum('tji,tj->ti', frame, preds[:,:2])
            preds = np.concatenate([w, preds[:,-1].reshape((-1,1))],axis=-1)

    dp_t = args.window_time
    pred_vels = preds / dp_t

    #ind = np.array([i[1] for i in dataset.index_map], dtype=np.int)
    #delta_int = int(
    #    args.window_time * args.imu_freq / 2.0
    #)  # velocity as the middle of the segment
    if not (args.window_time * args.imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    #ind_intg = ind + delta_int  # the indices of doing integral

    ts = dataset.get_ts_last_imu_us() * 1e-6 ## original tlio it is 1e-6 tried 1e-9
    r_gt, pos_gt = dataset.get_gt_traj_center_window_times()
    eul_gt = r_gt.as_euler("xyz", degrees=True)

    #dts = np.mean(ts[ind_intg[1:]] - ts[ind_intg[:-1]])
    dts = np.mean(ts[1:] - ts[:-1]) #tried 1 here to check if it scales correctly
    #pos_intg = np.zeros([pred_vels.shape[0] + 1, args.output_dim])
    pos_intg = np.zeros([pred_vels.shape[0], args.output_dim])
    #pos_intg[0] = pos_gt[0]
    pos_intg = np.cumsum(pred_vels[:, :] * dts, axis=0) + pos_gt[0]
    #ts_intg = np.append(ts[ind_intg], ts[ind_intg[-1]] + dts)
    ts_intg = np.append(ts[0], ts[-1] + dts)

    #ts_in_range = ts[ind_intg[0] : ind_intg[-1]]  # s
    pos_pred = pos_intg #interp1d(ts_intg, pos_intg, axis=0)(ts_in_range)
    #ori_pred = dataset.orientations[0][ind_intg[0] : ind_intg[-1], :]
    eul_pred = eul_gt #Rotation.from_quat(ori_pred).as_euler("xyz", degrees=True)
    
    #print("SHAPES", ts.shape, pos_pred.shape, pos_gt.shape, eul_pred.shape, eul_gt.shape)

    traj_attr_dict = {
        "ts": ts, #ts_in_range,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "eul_pred": eul_pred,
        "eul_gt": eul_gt,
    }

    return traj_attr_dict


def compute_metrics_and_plotting(args, net_attr_dict, traj_attr_dict, arch_type):
    """
    Obtain trajectory and compute metrics.
    """

    """ ------------ Trajectory metrics ----------- """
    ts = traj_attr_dict["ts"]
    pos_pred = traj_attr_dict["pos_pred"]
    pos_gt = traj_attr_dict["pos_gt"]
    eul_pred = traj_attr_dict["eul_pred"]
    eul_gt = traj_attr_dict["eul_gt"]

    # get RMSE
    rmse = np.sqrt(np.mean(np.linalg.norm(pos_pred - pos_gt, axis=1) ** 2))
    # get ATE
    diff_pos = pos_pred - pos_gt
    ate = np.mean(np.linalg.norm(diff_pos, axis=1))
    # get RMHE (yaw)
    diff_eul = wrap_rpy(eul_pred - eul_gt)
    rmhe = np.sqrt(np.mean(diff_eul[:, 2] ** 2))
    # get position drift
    traj_lens = np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1))
    drift_pos = np.linalg.norm(pos_pred[-1, :] - pos_gt[-1, :])
    drift_ratio = drift_pos / traj_lens
    # get yaw drift
    duration = ts[-1] - ts[0]
    drift_ang = np.linalg.norm(
        diff_eul[-1, 2] - diff_eul[0, 2]
    )  # beginning not aligned
    drift_ang_ratio = drift_ang / duration
    # get RPE on position and yaw
    ns_rpe = int(args.rpe_window * args.imu_freq)
    rpe_rmse, rpe_rmse_z, relative_yaw_rmse, rpes = compute_rpe(
        ns_rpe, pos_pred, pos_gt, eul_pred[:, [2]], eul_gt[:, [2]]
    )

    metrics = {
        "ronin": {
            "rmse": rmse,
            "ate": ate,
            "rmhe": rmhe,
            "drift_pos (m/m)": drift_ratio,
            "drift_yaw (deg/s)": drift_ang_ratio,
            "rpe": rpe_rmse,
            "rpe_z": rpe_rmse_z,
            "rpe_yaw": relative_yaw_rmse,
        }
    }

    """ ------------ Network loss metrics ----------- """
    mse_loss = np.mean(
        (net_attr_dict["targets"] - net_attr_dict["preds"]) ** 2, axis=0
    )  # 3x1
    likelihood_loss = np.mean(net_attr_dict["losses"], axis=0)  # 3x1
    avg_mse_loss = np.mean(mse_loss)
    avg_likelihood_loss = np.mean(likelihood_loss)
    metrics["ronin"]["mse_loss_x"] = float(mse_loss[0])
    metrics["ronin"]["mse_loss_y"] = float(mse_loss[1])
    metrics["ronin"]["mse_loss_z"] = float(mse_loss[2])
    metrics["ronin"]["mse_loss_avg"] = float(avg_mse_loss)
    metrics["ronin"]["likelihood_loss_x"] = float(likelihood_loss[0])
    metrics["ronin"]["likelihood_loss_y"] = float(likelihood_loss[1])
    metrics["ronin"]["likelihood_loss_z"] = float(likelihood_loss[2])
    metrics["ronin"]["likelihood_loss_avg"] = float(avg_likelihood_loss)

    """ ------------ Data for plotting ----------- """
    total_pred = net_attr_dict["preds"].shape[0]
    pred_ts = (1.0 / args.sample_freq) * np.arange(total_pred)
    if '_6v_6s' in arch_type or '_frame_fullCov' in arch_type:
        pred_sigmas = net_attr_dict["preds_cov"]
    else:
        pred_sigmas = np.exp(net_attr_dict["preds_cov"])
    if '_frame' in arch_type:
        plot_dict = {
            "ts": ts,
            "pos_pred": pos_pred,
            "pos_gt": pos_gt,
            "pred_ts": pred_ts,
            "preds": net_attr_dict["preds"],
            "targets": net_attr_dict["targets"],
            "pred_sigmas": pred_sigmas,
            "rmse": rmse,
            "rpe_rmse": rpe_rmse,
            "rpes": rpes,
            "frames" : net_attr_dict["frames"]
        }
    else:
        plot_dict = {
            "ts": ts,
            "pos_pred": pos_pred,
            "pos_gt": pos_gt,
            "pred_ts": pred_ts,
            "preds": net_attr_dict["preds"],
            "targets": net_attr_dict["targets"],
            "pred_sigmas": pred_sigmas,
            "rmse": rmse,
            "rpe_rmse": rpe_rmse,
            "rpes": rpes,
        }

    return metrics, plot_dict


def plot_3d_2var(x, y1, y2, xlb, ylbs, lgs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1[:, i], label=lgs[0])
        plt.plot(x, y2[:, i], label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def plot_3d_1var(x, y, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        if x is not None:
            plt.plot(x, y[:, i])
        else:
            plt.plot(y[:, i])
        plt.ylabel(ylbs[i])
        plt.grid(True)
    if xlb is not None:
        plt.xlabel(xlb)
    return fig


def plot_3d_2var_with_sigma(
    x, y1, y2, sig, xlb, ylbs, lgs, num=None, dpi=None, figsize=None
):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    y1_plus_sig = y1 + 3 * sig
    y1_minus_sig = y1 - 3 * sig
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1_plus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(x, y1_minus_sig[:, i], "-g", linewidth=0.2)
        plt.fill_between(
            x, y1_plus_sig[:, i], y1_minus_sig[:, i], facecolor="green", alpha=0.5
        )
        plt.plot(x, y1[:, i], "-b", linewidth=0.5, label=lgs[0])
        plt.plot(x, y2[:, i], "-r", linewidth=0.5, label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def plot_3d_1var_with_sigma(x, y, sig, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    plus_sig = 3 * sig
    minus_sig = -3 * sig
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, plus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(x, minus_sig[:, i], "-g", linewidth=0.2)
        plt.fill_between(
            x, plus_sig[:, i], minus_sig[:, i], facecolor="green", alpha=0.5
        )
        plt.plot(x, y[:, i], "-b", linewidth=0.5)
        plt.ylabel(ylbs[i])
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def make_plots(args, plot_dict, outdir):
    ts = plot_dict["ts"]
    pos_pred = plot_dict["pos_pred"]
    pos_gt = plot_dict["pos_gt"]
    pred_ts = plot_dict["pred_ts"]
    preds = plot_dict["preds"]
    targets = plot_dict["targets"]
    pred_sigmas = plot_dict["pred_sigmas"]
    rmse = plot_dict["rmse"]
    rpe_rmse = plot_dict["rpe_rmse"]
    rpes = plot_dict["rpes"]

    dpi = 90
    figsize = (16, 9)

    fig1 = plt.figure(num="prediction vs gt", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1])
    plt.plot(pos_gt[:, 0], pos_gt[:, 1])
    plt.axis("equal")
    plt.legend(["Predicted", "Ground truth"])
    plt.title("2D trajectory and ATE error against time")
    plt.subplot2grid((3, 2), (2, 0))
    plt.plot(np.linalg.norm(pos_pred - pos_gt, axis=1))
    plt.legend(["RMSE:{:.3f}, RPE:{:.3f}".format(rmse, rpe_rmse)])
    for i in range(3):
        plt.subplot2grid((3, 2), (i, 1))
        plt.plot(preds[:, i])
        plt.plot(targets[:, i])
        plt.legend(["Predicted", "Ground truth"])
        plt.title("{}".format(targ_names[i]))
    plt.tight_layout()
    plt.grid(True)

    fig2 = plot_3d_2var(
        ts,
        pos_pred,
        pos_gt,
        xlb="t(s)",
        ylbs=["x(m)", "y(m)", "z(m)"],
        lgs=["RONIN", "Ground Truth"],
        num="Position",
        dpi=dpi,
        figsize=figsize,
    )
    fig3 = plot_3d_2var_with_sigma(
        pred_ts,
        preds,
        targets,
        pred_sigmas,
        xlb="t(s)",
        ylbs=["x(m)", "y(m)", "z(m)"],
        lgs=["imu", "vio"],
        num="Displacement",
        dpi=dpi,
        figsize=figsize,
    )
    fig4 = plot_3d_1var_with_sigma(
        pred_ts,
        preds - targets,
        pred_sigmas,
        xlb="t(s)",
        ylbs=["x(m)", "y(m)", "z(m)"],
        num="Displacement errors",
        dpi=dpi,
        figsize=figsize,
    )
    fig5 = plot_3d_1var(
        None,
        rpes,
        xlb=None,
        ylbs=["x(m)", "y(m)", "z(m)"],
        num=f"RTE error over {args.rpe_window}s",
        dpi=dpi,
        figsize=figsize,
    )

    pred_norm = np.linalg.norm(preds[:, 0:2], axis=1)
    targ_norm = np.linalg.norm(targets[:, 0:2], axis=1)
    pred_ang = np.arctan2(preds[:, 0], preds[:, 1])
    targ_ang = np.arctan2(targets[:, 0], targets[:, 1])
    ang_diff = targ_ang - pred_ang
    ang_diff = ang_diff + 2 * np.pi * (ang_diff <= -np.pi)
    ang_diff = ang_diff - 2 * np.pi * (ang_diff > np.pi)

    fig6 = plt.figure(num="2D Displacement norm and heading", dpi=dpi, figsize=(16, 9))
    plt.title("2D Displacement norm and heading")
    plt.subplot(411)
    plt.plot(pred_ts, pred_norm, "-b", linewidth=0.5, label="imu")
    plt.plot(pred_ts, targ_norm, "-r", linewidth=0.5, label="vio")
    plt.ylabel("distance (m)")
    plt.legend()
    plt.grid(True)
    plt.subplot(412)
    plt.plot(pred_ts, pred_norm - targ_norm, "-b", linewidth=0.5)
    plt.ylabel("distance (m)")
    plt.grid(True)
    plt.subplot(413)
    plt.plot(pred_ts, pred_ang, "-b", linewidth=0.5)
    plt.plot(pred_ts, targ_ang, "-r", linewidth=0.5)
    plt.ylabel("angle (rad)")
    plt.grid(True)
    plt.subplot(414)
    plt.plot(pred_ts, ang_diff, "-b", linewidth=0.5)
    plt.ylabel("angle (rad)")
    plt.xlabel("t")
    plt.grid(True)

    fig1.savefig(osp.join(outdir, "view.png"))
    fig2.savefig(osp.join(outdir, "pos.png"))
    fig3.savefig(osp.join(outdir, "pred.svg"))
    fig4.savefig(osp.join(outdir, "pred-err.svg"))
    fig5.savefig(osp.join(outdir, "rpe.svg"))
    fig6.savefig(osp.join(outdir, "norm_angle.svg"))

    plt.close("all")

    return


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

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
    print('cov:', cov[0])
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
    """SO(3) preprocessing for Eq_Motion_Model_SO3_fullCov. Mirrors train.preprocess_so3."""
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
    scalar = torch.cat([na, nv1, nv2, av1, v1v2, av2], dim=-1)
    vector = torch.stack([a, v1, v2], dim=-1)
    return vector, scalar


def get_inference(network, data_loader, device, epoch, arch_type):
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
        frame = None

        if arch_type == 'eq_transformer':
            vector, scalar = preprocess_for_eq_transformer(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer(out_vec, out_sca)
        if 'resnet_2scalars_tlio_frame' in arch_type:
            frame,out_vec, out_sca = network(sample["feats"]["imu0"])
            pred, pred_cov = postprocess_tlio_frame_2scalars(out_vec, out_sca)
        elif 'resnet_fullCov_tlio_frame' in arch_type:
            frame,pred, pred_cov = network(sample["feats"]["imu0"])
        elif  '_3scalars' in arch_type and 'wo_t' not in arch_type:
            vector, scalar = preprocess_for_eq_transformer(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_3scalars(out_vec, out_sca)
        elif '_6v_6s' in arch_type and 'wo_t' not in arch_type:
            vector, scalar = preprocess_for_eq_transformer(sample)
            out_vec, out_sca = network(vector.float(), scalar.float())
            pred, pred_cov = postprocess_eq_transformer_6v_6s(out_vec, out_sca)
        elif arch_type == 'rnin_vio_model_lstm':
            feat = sample["feats"]["imu0"]
            pred, pred_cov = network(feat.unsqueeze(dim=1))
            pred = pred.squeeze(dim=1)
            pred_cov = pred_cov.squeeze(dim=1)
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
            # Our SO(3)-equivariant extension
            frame, pred, pred_cov = network(*preprocess_so3(sample))
        elif 'o2_frame' in arch_type:
            # # 1. 원래 *preprocess_o2()로 한 번에 넘기던 것을 3개의 변수로 분리해서 받습니다.
            # vec, sca, orig_sca = preprocess_o2(sample)
            
            # # ==========================================================
            # # [수정] 네트워크 Forward 직전에 첫 번째 배치(bid==0)에서 1번 측정
            # # ==========================================================
            # if bid == 0:
            #     import thop
            #     # 공정성을 위해 batch_size를 1로 자릅니다 (0:1 형태 유지)
            #     dummy_vec = vec[0:1, ...].clone()
            #     dummy_sca = sca[0:1, ...].clone()
            #     dummy_orig_sca = orig_sca[0:1, ...].clone()
                
            #     macs, params = thop.profile(
            #         network, 
            #         inputs=(dummy_vec, dummy_sca, dummy_orig_sca), 
            #         verbose=False
            #     )
            #     print(f"\n{'='*50}")
            #     print(f"[EqNIO ({arch_type})] FLOPs(MACs): {macs / 1e6:.2f} M, Params: {params / 1e6:.2f} M")
            #     print(f"{'='*50}\n")
            # # ==========================================================
            
            # # 2. 실제 네트워크 포워드 (기존과 완벽히 동일한 연산)
            # frame, pred, pred_cov = network(vec, sca, orig_sca)
            
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

        targ = sample["targ_dt_World"][:,-1,:]

        if ('_fullCov' in arch_type or '_PearsonCov' in arch_type) and '_frame_avg' not in arch_type:
            if 'so3_frame' in arch_type:
                targ = torch.einsum('bij,bj->bi', frame, targ)
            else:
                targ = torch.concat([torch.matmul(frame, targ[:,:2].unsqueeze(-1)).squeeze(-1),targ[:,-1].unsqueeze(-1)], dim=-1)
            
        if '_frame' in arch_type:
            frames_all.append(torch_to_numpy(frame))
        # Only grab the last prediction in this case
        if len(pred.shape) == 3:
            pred = pred[:,:,-1]
            pred_cov = pred_cov[:,:,-1]
        
        assert len(pred.shape) == 2
        loss = get_loss(pred, pred_cov, targ, 0, arch_type) #epoch 0 so i get mse loss for test

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


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


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

    data_window_config = dotdict()
    data_window_config.past_data_size = int(args.past_time * args.imu_freq)
    data_window_config.window_size = int(args.window_time * args.imu_freq)
    data_window_config.future_data_size = int(args.future_time * args.imu_freq)
    data_window_config.step_size = int(args.imu_freq / args.sample_freq)
    data_window_config.data_style = "resampled"
    data_window_config.input_sensors = ["imu0"]
    data_window_config.decimator = 10
    data_window_config.express_in_local_gravity_aligned = getattr(args, "express_in_local_gravity_aligned", True)
    data_window_config.express_in_local_frame = False
    data_window_config.g_compensate = False

    net_config = {
        "in_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 32
        + 1
    }

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
    data_window_config.window_size = data_window_config.window_size + data_window_config.past_data_size
    return data_window_config, net_config


def net_test(args):
    """
    Main function for network testing
    Generate trajectories, plots, and metrics.json file
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            logging.info(f"Testing output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return
    # print(args.test_list,' is visible')
    test_list_path = osp.join(args.root_dir, args.test_list)#"test_list.txt"
    test_list = get_datalist(test_list_path)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {args.model_path} loaded to device {device}.")
    num_params = sum(p.numel() for p in network.parameters())
    logging.info(f"Number of parameters in the network: {num_params}")
    # initialize containers
    all_metrics = {}

    for data in test_list:
        logging.info(f"Processing {data}...")
        try:
            #seq_dataset = FbSequenceDataset(
            #    args.root_dir, [data], args, data_window_config, mode="test"
            #)

            seq_dataset = MemMappedSequencesDataset(
                args.root_dir,
                'test',
                data_window_config,
                sequence_subset=[data],
                store_in_ram=True,
                test_file_path = args.test_list,
                start_index = data_window_config["past_data_size"],
                theta_range_deg = args.perturb_gravity_theta_range,
                dataloader_bias_gravity_aug = args.dataloader_bias_gravity_aug,
                accel_bias_range = args.accel_bias_range,
                gyro_bias_range = args.gyro_bias_range
            )

            seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=False)

        except OSError as e:
            print(e)
            continue
        outdir = osp.join(args.out_dir, data)
        if osp.exists(outdir) is False:
            os.mkdir(outdir)
        else:
            print('Already Processed!')
            continue

        # Obtain trajectory
        net_attr_dict = get_inference(network, seq_loader, device, epoch=50, arch_type = args.arch)
        if '_frame' in args.arch and '_frame_avg' not in args.arch:
            traj_attr_dict = pose_integrate(args, seq_dataset, net_attr_dict["preds"], net_attr_dict['frames'])
        else:
            traj_attr_dict = pose_integrate(args, seq_dataset, net_attr_dict["preds"])
        outdir = osp.join(args.out_dir, data)
        if osp.exists(outdir) is False:
            os.mkdir(outdir)
        outfile = osp.join(outdir, "trajectory.txt")
        trajectory_data = np.concatenate(
            [
                traj_attr_dict["ts"].reshape(-1, 1),
                traj_attr_dict["pos_pred"],
                traj_attr_dict["pos_gt"],
            ],
            axis=1,
        )
        np.savetxt(outfile, trajectory_data, delimiter=",")

        # obtain metrics
        metrics, plot_dict = compute_metrics_and_plotting(
            args, net_attr_dict, traj_attr_dict, arch_type = args.arch
        )
        logging.info(metrics)
        all_metrics[data] = metrics

        outfile_net = osp.join(outdir, "net_outputs.txt")
        if '_frame' in args.arch and '_frame_avg' not in args.arch:
            net_outputs_data = np.concatenate(
                [
                    plot_dict["pred_ts"].reshape(-1, 1),
                    plot_dict["preds"],
                    plot_dict["targets"],
                    plot_dict["pred_sigmas"].reshape((plot_dict["preds"].shape[0], -1)),
                    plot_dict["frames"].reshape((plot_dict["preds"].shape[0], -1)),
                ],
                axis=1,
            )
            np.savetxt(outfile_net, net_outputs_data, delimiter=",")
        else:
            net_outputs_data = np.concatenate(
                [
                    plot_dict["pred_ts"].reshape(-1, 1),
                    plot_dict["preds"],
                    plot_dict["targets"],
                    plot_dict["pred_sigmas"].reshape((plot_dict["preds"].shape[0], -1)),
                ],
                axis=1,
            )
            np.savetxt(outfile_net, net_outputs_data, delimiter=",")

        if args.save_plot:
            make_plots(args, plot_dict, outdir)

        try:
            with open(args.out_dir + "/metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=1)
        except ValueError as e:
            raise e
        except OSError as e:
            print(e)
            continue
        except Exception as e:
            raise e

    return
