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


def _forward_with_cov_train(network, feat, cov_type):
    """train/val용 forward dispatch."""
    if cov_type == 'eq_spec':
        pred, pred_V, pred_inv = network(feat)
        return pred, (pred_V, pred_inv), pred_inv
    elif cov_type == 'eq_chol' or cov_type == 'eq_chol_svd':
        # Same network output (A in R^{B x 3 x 3}); only the loss interpretation differs.
        pred, pred_A = network(feat)
        sig2 = (pred_A ** 2).sum(dim=-1)
        pred_log_std = 0.5 * torch.log(sig2.clamp(min=1e-8))
        return pred, pred_A, pred_log_std
    else:
        pred, pred_cov = network(feat)
        return pred, pred_cov, pred_cov


def get_inference(network, data_loader, device, epoch,
                  body_frame_3regress=False, body_frame=False, transforms=[],
                  cov_type='diag', mse_warmup_epochs=10):
    """
    Validation inference. Mirrors do_train's forward/loss dispatch.
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
            pred_cov_params = pred_cov
            pred_cov_for_log = pred_cov
        else:
            pred, pred_cov_params, pred_cov_for_log = _forward_with_cov_train(
                network, feat, cov_type)
            pred_vel = pred.clone()

        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:, -1, :]
            if body_frame:
                targ_vel = sample["vel_Body"][:, -1, :]
                targ = sample["vel_Body"][:, -1, :]
            else:
                targ_vel = sample["targ_dt_World"][:, -1, :]
        else:
            targ = sample["targ_dt_World"][:, 1:, :].permute(0, 2, 1)

        if body_frame_3regress:
            loss = get_loss(pred_vel, pred_cov_params, targ_vel, epoch,
                            body_frame_3regress=True, cov_type='diag',
                            mse_warmup_epochs=mse_warmup_epochs)
        else:
            loss = get_loss(pred, pred_cov_params, targ, epoch,
                            body_frame_3regress=False,
                            cov_type=cov_type,
                            mse_warmup_epochs=mse_warmup_epochs,
                            nll_weight=1.0,
                            mse_anchor_weight=0.0)   # val에서는 anchor 빼기

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov_for_log))
        targets_vel_all.append(torch_to_numpy(targ_vel))
        preds_vel_all.append(torch_to_numpy(pred_vel))
        losses_all.append(torch_to_numpy(loss))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    targets_vel_all = np.concatenate(targets_vel_all, axis=0)
    preds_vel_all = np.concatenate(preds_vel_all, axis=0)
    return {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
        "targets_vel": targets_vel_all,
        "preds_vel": preds_vel_all,
    }


def do_train(network, train_loader, device, epoch, optimizer, input_dim,
             transforms=[], body_frame_3regress=False, body_frame=False,
             cov_type='diag', mse_warmup_epochs=10, mse_anchor_weight=0.1,
             nll_weight=1.0, grad_clip=0.1, freeze_cov_keys=None):
    """
    Train network for one epoch using a specified data loader.
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    train_targets_vel, train_preds_vel = [], []
    network.train()
    if freeze_cov_keys is not None:
        # Keep frozen mean/backbone in EVAL (running BN stats, no dropout) so the mean is
        # byte-identical across the diagonal and spectral arms; only the cov head trains.
        network.eval()
        for _nm, _m in network.named_modules():
            if _nm.startswith(freeze_cov_keys):
                _m.train()

    for bid, sample in enumerate(train_loader):
        sample = to_device(sample, device)
        for transform in transforms:
            sample = transform(sample)
        feat = sample["feats"]["imu0"]
        optimizer.zero_grad()
        if bid < 6 and (epoch < 11 or (90 <= epoch <= 100) or (190 <= epoch <= 200)):
            print('bid = ', bid)

        # ---- Forward (cov_type 분기) ----
        if body_frame_3regress:
            pred, pred_cov, pred_vel = network(feat)
            pred_cov_params = pred_cov
            pred_cov_for_log = pred_cov
        else:
            if cov_type == 'eq_spec':
                pred, pred_V, pred_inv = network(feat)
                pred_cov_params = (pred_V, pred_inv)
                pred_cov_for_log = pred_inv                     # (B,3) log-std
                pred_vel = pred.clone()
            elif cov_type == 'eq_chol' or cov_type == 'eq_chol_svd':
                pred, pred_A = network(feat)                    # pred_A: (B,3,3)
                pred_cov_params = pred_A
                with torch.no_grad():
                    sig2 = (pred_A ** 2).sum(dim=-1)            # diag(A A^T)
                    pred_cov_for_log = 0.5 * torch.log(sig2.clamp(min=1e-8))
                pred_vel = pred.clone()
            else:
                pred, pred_cov = network(feat)
                pred_cov_params = pred_cov
                pred_cov_for_log = pred_cov
                pred_vel = pred.clone()

        # ---- Target 결정 ----
        if len(pred.shape) == 2:
            targ = sample["targ_dt_World"][:, -1, :]
            if body_frame:
                targ_vel = sample["vel_Body"][:, -1, :]
                targ = sample["vel_Body"][:, -1, :]
            else:
                targ_vel = sample["targ_dt_World"][:, -1, :]
        else:
            targ = sample["targ_dt_World"][:, 1:, :].permute(0, 2, 1)

        # ---- Loss ----
        if body_frame_3regress:
            loss = get_loss(pred_vel, pred_cov_params, targ_vel, epoch,
                            body_frame_3regress=True, cov_type='diag',
                            mse_warmup_epochs=mse_warmup_epochs)
        else:
            loss = get_loss(pred, pred_cov_params, targ, epoch,
                    body_frame_3regress=False,
                    cov_type=cov_type,
                    mse_warmup_epochs=mse_warmup_epochs,
                    nll_weight=nll_weight,
                    mse_anchor_weight=mse_anchor_weight)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_preds_cov.append(torch_to_numpy(pred_cov_for_log))
        train_losses.append(torch_to_numpy(loss))
        train_targets_vel.append(torch_to_numpy(targ_vel))
        train_preds_vel.append(torch_to_numpy(pred_vel))

        loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip, error_if_nonfinite=True)
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
        if sigmas.ndim == 3 and sigmas.shape[1] == 3:
            sigmas = sigmas[:, :, -1]
    summary_writer.add_scalar(f"{mode}_loss/loss_x", mse_loss[0], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_y", mse_loss[1], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_z", mse_loss[2], epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", np.mean(mse_loss), epoch)
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    # sigma 차원 안전 처리: 1D면 평균, 2D면 x/y/z 따로
    if sigmas.ndim == 2 and sigmas.shape[1] >= 3:
        summary_writer.add_histogram(f"{mode}_hist/sigma_x", sigmas[:, 0], epoch)
        summary_writer.add_histogram(f"{mode}_hist/sigma_y", sigmas[:, 1], epoch)
        summary_writer.add_histogram(f"{mode}_hist/sigma_z", sigmas[:, 2], epoch)
    else:
        summary_writer.add_histogram(f"{mode}_hist/sigma_all", sigmas.reshape(-1), epoch)
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
    
    # data = TlioData(
    #     args.root_dir, 
    #     batch_size=args.batch_size, 
    #     dataset_style=args.dataset_style, 
    #     num_workers=args.workers,
    #     persistent_workers=args.persistent_workers,
    #     window_time=args.window_time
    # )
    aug_opts = {
        "do_bias_shift": bool(args.do_bias_shift),
        "bias_shift_options": {
            "accel_bias_range": args.accel_bias_range * args.bias_scale,
            "gyro_bias_range":  args.gyro_bias_range  * args.bias_scale,
            "accel_noise_std":  args.accel_noise_std,
            "gyro_noise_std":   args.gyro_noise_std,
            "mag_bias_range": 0.05,
            "barom_press_bias_range": 0.01,
            "barom_temp_bias_range": 1.0,
        },
        "perturb_gravity": bool(args.perturb_gravity),
        "perturb_gravity_theta_range": float(args.perturb_gravity_theta_range),
        "so3_train_aug": bool(getattr(args, "so3_train_aug", False)),
        "so3_train_aug_max_deg": float(getattr(args, "so3_train_aug_max_deg", 180.0)),
    }

    data = TlioData(
        args.root_dir,
        batch_size=args.batch_size,
        dataset_style=args.dataset_style,
        num_workers=args.workers,
        persistent_workers=args.persistent_workers,
        window_time=args.window_time,
        augmentation_options=aug_opts,
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
        body_frame_3regress = False
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
    net_config["init_log_a_scale"] = getattr(args, "init_log_a_scale", None)
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim)
    network.to(device)
    total_params = network.get_num_params()
    logging.info(f'Network "{args.arch}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")
    # ---- resolve cov_type once (needed for freeze_mean head selection) ----
    if getattr(args, "cov_type", "auto") != "auto":
        _cov_type0 = args.cov_type
    else:
        _al = (args.arch or "").lower()
        _cov_type0 = 'eq_chol' if _al == 'ln_resnet_chol' else ('eq_spec' if _al == 'ln_resnet_cov' else 'diag')

    # ---- freeze_mean: load a base checkpoint, freeze backbone+mean, train ONLY the cov head ----
    freeze_cov_keys = None
    if getattr(args, "freeze_mean", False):
        if args.model_path is not None and osp.exists(args.model_path):
            base = torch.load(args.model_path, map_location=device)
            sd = base.get("model_state_dict", base)
            missing, unexpected = network.load_state_dict(sd, strict=False)
            logging.info(f"[freeze_mean] loaded base mean from {args.model_path}: "
                         f"{len(missing)} missing / {len(unexpected)} unexpected keys")
        else:
            logging.warning("[freeze_mean] no valid --model_path base; freezing a randomly-init mean!")
        COVKEYS = {
            'eq_spec': ('output_block2', 'output_block3', 'output_block4',
                        'output_block5', 'output_block6', 'output_block7', 'inv_head_shared'),
            'diag':    ('output_block_diag',),
            'eq_chol': ('output_block_a1', 'output_block_a2', 'output_block_a3', 'log_a_scale'),
        }[_cov_type0]
        freeze_cov_keys = COVKEYS
        n_tr = n_fr = 0
        for name, p in network.named_parameters():
            keep = name.startswith(COVKEYS)
            p.requires_grad = keep
            n_tr += p.numel() if keep else 0
            n_fr += 0 if keep else p.numel()
        logging.info(f"[freeze_mean] cov_type={_cov_type0}: trainable(cov head)={n_tr}  "
                     f"frozen(mean+backbone)={n_fr}")


    # optimizer = torch.optim.Adam(network.parameters(), args.lr)
    _trainable = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(_trainable, lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=10, eps=1e-12
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
        # train_attr_dict = do_train(network, train_loader, device, epoch, optimizer, args.input_dim, train_transforms, body_frame_3regress, body_frame)
        if getattr(args, "cov_type", "auto") != "auto":
            cov_type = args.cov_type
        else:
            # arch만 보고 결정 (out_dir에 chol/cov가 우연히 들어있는 경우 방지)
            arch_lower = (args.arch or "").lower()
            if arch_lower == "ln_resnet_chol":
                cov_type = 'eq_chol'
            elif arch_lower == "ln_resnet_cov":
                cov_type = 'eq_spec'
            else:
                cov_type = 'diag'
        if epoch == 1:  # 한 번만 로깅
            logging.info(f"Using cov_type = {cov_type}")

        train_attr_dict = do_train(network, train_loader, device, epoch, optimizer,
                            args.input_dim, train_transforms,
                            body_frame_3regress, body_frame,
                            cov_type=cov_type,
                            mse_warmup_epochs=args.mse_warmup_epochs,
                            mse_anchor_weight=args.mse_anchor_weight,
                            nll_weight=args.nll_weight,
                            grad_clip=args.grad_clip,
                            freeze_cov_keys=freeze_cov_keys)
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
            # val_attr_dict = get_inference(network, val_loader, device, epoch, body_frame_3regress, body_frame)
            val_attr_dict = get_inference(network, val_loader, device, epoch,
                  body_frame_3regress, body_frame,
                  cov_type=cov_type,
                  mse_warmup_epochs=args.mse_warmup_epochs)
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
