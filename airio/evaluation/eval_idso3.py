"""ID/SO(3) evaluation under BOTH readings of the test-time rotation protocol.

    python -m evaluation.eval_idso3 --config configs/Nanobench/motion_body_a2_spec_small.conf \
                                    --seed 0 --n_rot 5

Supplement G.1 says the ID/SO(3) test "rotates all vector-valued inputs by a sampled R
and unrotates the predictions".  For a body-frame network that reading is only the same
as a physical sensor remount when *every* input channel is a genuine body-frame vector.
The submitted with-attitude model feeds `Log(R_WS)`, which is not, so the two readings
come apart -- and the reported ID/SO(3) invariance is a property of the protocol rather
than of the model.  This script measures both:

    (I) identity          no rotation -- the plain test-set number
    (P) physical remount  imu -> Q imu,  R_WS -> R_WS Q^T          <- what a remount does
    (G) G.1 as written    imu -> Q imu,  attitude channel -> Q f   <- what the text says

For an attitude encoding that transforms as a vector (A1/A2/A3), (P) and (G) are the same
computation, so the two columns must agree exactly and both must equal (I).

Reported per protocol:
    ATE_um     Umeyama-aligned absolute trajectory error, metres (integrated body velocity)
    equiv      max || unrot(pred under rotation) - pred at identity ||, relative to signal
    cov-law    || Sigma_rot - Q Sigma Q^T ||_F / || Sigma ||_F, the congruence the filter needs
"""
import argparse
import os
import sys

import numpy as np
import pypose as pp
import torch
import torch.utils.data as Data
import tqdm
from pyhocon import ConfigFactory
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import SeqeuncesMotionDataset, collate_fcs      # noqa: E402
from model import net_dict                                    # noqa: E402
from model.attitude import attitude_feature, rep_of           # noqa: E402
from utils import move_to                                     # noqa: E402


def rotate_batch(data, label, Q, rep, protocol):
    """Apply the test-time rotation. Returns (data, attitude_feature)."""
    d = dict(data)
    if Q is not None:
        d["gyro"] = torch.einsum("ij,...j->...i", Q, data["gyro"])
        d["acc"] = torch.einsum("ij,...j->...i", Q, data["acc"])
    gt_rot = label["gt_rot"][:, :-1, :]
    if Q is None or rep is None:
        return d, attitude_feature(gt_rot, rep)
    if protocol == "physical":
        Qinv = pp.SO3(torch.tensor(Rot.from_matrix(Q.cpu().numpy()).inv().as_quat(),
                                   dtype=gt_rot.dtype, device=gt_rot.device))
        return d, attitude_feature(gt_rot @ Qinv, rep)
    if protocol == "g1":
        f = attitude_feature(gt_rot, rep)
        return d, torch.einsum("ij,...j->...i", Q, f)
    raise ValueError(protocol)


@torch.no_grad()
def run_sequence(network, loader, Q, rep, protocol, device):
    """Inference over one sequence; predictions are un-rotated back to the body frame."""
    vel, cov, cov_R, ts = [], [], [], []
    for data, _, label in loader:
        data, label = move_to([data, label], device)
        d, rot = rotate_batch(data, label, Q, rep, protocol)
        out = network(d, rot)
        v = out["net_vel"]
        R = out.get("cov_R", None)
        if R is None:
            # A diagonal head reports Sigma = diag(c) in whatever frame it was evaluated in,
            # i.e. its eigenframe is the identity *of the rotated frame*.  Materialise that
            # frame BEFORE un-rotating, so the un-rotation below applies to it too.  Leaving
            # it as a bare identity would compare diag(c_rot) against diag(c_id) -- a test of
            # INVARIANCE, which is the wrong law and precisely the mistake this audit flags in
            # the repo's own test_vn_codenet_equivariance.py.
            R = (torch.eye(3, dtype=v.dtype, device=device)
                 .reshape(9).expand(*v.shape[:-1], 9).contiguous())
        if Q is not None:                                   # unrotate before metrics
            v = torch.einsum("ji,...j->...i", Q, v)         # Q^T v
            Vm = R.reshape(*R.shape[:-1], 3, 3)
            R = torch.einsum("ji,...jk->...ik", Q, Vm).reshape(*R.shape)   # Q^T V
        vel.append(v)
        cov.append(out["cov"] if out["cov"] is not None else torch.zeros_like(v))
        cov_R.append(R)
        ts.append(network.get_label(data["ts"][..., None])[0])
    cat = lambda x: torch.cat(x, dim=-2)
    return (cat(vel)[0].cpu().numpy(), cat(cov)[0].cpu().numpy(),
            cat(cov_R)[0].cpu().numpy(), cat(ts).cpu().numpy().flatten())


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - mu_s).T @ (dst - mu_d))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return (R @ src.T).T + (mu_d - R @ mu_s)


def ate_for(vel_body, ts, data_root, seq):
    """Umeyama-aligned ATE from integrated body velocity, matching unified_eval.py."""
    arr = np.load(os.path.join(data_root, seq, "imu0_resampled.npy"))
    gt_ts, gt_quat, gt_pos = arr[:, 0] / 1e6, arr[:, 7:11], arr[:, 11:14]
    ts = ts / 1e6 if ts.max() > 1e12 else ts
    n = min(len(ts), len(vel_body))
    ts, vel_body = ts[:n], vel_body[:n]
    idx = np.clip(np.searchsorted(gt_ts, ts), 0, len(gt_ts) - 1)
    vel_W = np.einsum("nij,nj->ni", Rot.from_quat(gt_quat[idx]).as_matrix(), vel_body)
    pos = np.zeros_like(vel_W)
    pos[0] = gt_pos[idx[0]]
    pos[1:] = gt_pos[idx[0]] + np.cumsum(0.5 * (vel_W[:-1] + vel_W[1:])
                                         * np.diff(ts)[:, None], axis=0)
    gtp = gt_pos[idx]
    return float(np.sqrt(((umeyama(pos, gtp) - gtp) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_rot", type=int, default=5)
    ap.add_argument("--seqlen", type=int, default=1000)
    args = ap.parse_args()

    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = args.device
    name = os.path.split(args.config)[-1].split(".")[0]
    if args.seed is not None:
        name = f"{name}_s{args.seed}"
    exp_dir = os.path.join(conf.general.exp_dir, name)

    network = net_dict[conf.train.network](conf.train).to(args.device).double().eval()
    ck = torch.load(os.path.join(exp_dir, "ckpt/best_model.ckpt"),
                    map_location=args.device, weights_only=True)
    missing, unexpected = network.load_state_dict(ck["model_state_dict"], strict=True)
    rep = rep_of(network)
    print(f"{exp_dir}\n  net={conf.train.network}  att_rep={rep}  epoch={ck['epoch']}  "
          f"params={sum(p.numel() for p in network.parameters()):,}")

    dc = conf.dataset.inference
    dc["mode"] = "inference"
    dc["exp_dir"] = exp_dir
    dc.data_list[0]["window_size"] = args.seqlen
    dc.data_list[0]["step_size"] = args.seqlen
    collate = collate_fcs[conf.dataset.collate.type]
    data_root = dc.data_list[0]["data_root"]
    seqs = list(dc.data_list[0].data_drive)

    rng = np.random.default_rng(0)
    Qs = []
    for _ in range(args.n_rot):
        ax = rng.standard_normal(3)
        Qs.append(torch.tensor(Rot.from_rotvec(ax / np.linalg.norm(ax)
                                               * rng.uniform(0.2, np.pi)).as_matrix(),
                               dtype=torch.float64, device=args.device))

    protocols = [("(I) identity", None, None)]
    if rep is not None:
        protocols += [("(P) physical remount", "physical", Qs),
                      ("(G) G.1 rotate-all-inputs", "g1", Qs)]
    else:
        protocols += [("(P) physical remount", "physical", Qs)]

    acc = {p[0]: {"ate": [], "eq": [], "cov": []} for p in protocols}
    for seq in tqdm.tqdm(seqs, desc="sequences"):
        ds = SeqeuncesMotionDataset(data_set_config=dc, data_path=seq, data_root=data_root)
        ld = Data.DataLoader(dataset=ds, batch_size=1, shuffle=False,
                             collate_fn=collate, drop_last=False)
        base = None
        for tag, proto, qs in protocols:
            if proto is None:
                v, c, R, ts = run_sequence(network, ld, None, rep, None, args.device)
                base = (v, c, R)
                acc[tag]["ate"].append(ate_for(v, ts, data_root, seq))
                continue
            ates, eqs, covs = [], [], []
            for Q in qs:
                v, c, R, ts = run_sequence(network, ld, Q, rep, proto, args.device)
                ates.append(ate_for(v, ts, data_root, seq))
                scale = np.abs(base[0]).max()
                eqs.append(np.abs(v - base[0]).max() / scale)
                V0, V1 = base[2].reshape(-1, 3, 3), R.reshape(-1, 3, 3)
                S0 = V0 @ (base[1][..., None] * V0.transpose(0, 2, 1))
                S1 = V1 @ (c[..., None] * V1.transpose(0, 2, 1))
                covs.append(float(np.mean(np.linalg.norm(S1 - S0, axis=(-2, -1))
                                          / (np.linalg.norm(S0, axis=(-2, -1)) + 1e-30))))
            acc[tag]["ate"].append(float(np.mean(ates)))
            acc[tag]["eq"].append(float(np.mean(eqs)))
            acc[tag]["cov"].append(float(np.mean(covs)))

    print(f"\n{'protocol':28s} {'ATE_um (m)':>11s} {'equiv err':>11s} {'cov-law err':>12s}")
    print("-" * 66)
    for tag, _, _ in protocols:
        a = acc[tag]
        eq = f"{np.mean(a['eq']):11.3e}" if a["eq"] else f"{'-':>11s}"
        cv = f"{np.mean(a['cov']):12.3e}" if a["cov"] else f"{'-':>12s}"
        print(f"{tag:28s} {np.mean(a['ate']):11.4f} {eq} {cv}")
    print("\n  equiv err   = || unrot(pred) - pred_identity || / max|pred|   (0 = equivariant)")
    print("  cov-law err = || Sigma_rot^unrot - Sigma_identity ||_F / ||Sigma||_F")
    print(f"  n_seq = {len(seqs)}   n_rot = {args.n_rot}")


if __name__ == "__main__":
    main()
