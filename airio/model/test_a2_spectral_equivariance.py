"""Pre-training verification for the A2 + spectral-covariance redesign.

Run:  python -m model.test_a2_spectral_equivariance      (from Air-IO/, CPU only)

The symmetry under test is a *physical sensor remount* by Q in SO(3):

    imu -> Q imu ,   R_WS -> R_WS Q^T ,   and required:   v -> Q v ,  Sigma -> Q Sigma Q^T .

V0  regression   the patched feature path reproduces the submitted `.Log().tensor()` bitwise
V1  feature law  || f(R Q^T) - Q f(R) ||                                per encoding
V2  losslessness R_WS is recoverable from the A2 channels
V3  mean equiv.  || net(remount) - Q net ||                             end-to-end, float64
V4  cov law      || Sigma(remount) - Q Sigma Q^T ||_F / || Sigma ||_F   end-to-end, float64
V5  protocol     for A2, the G.1 "rotate all inputs" reading == the physical remount
V6  NLL invariance   the training objective is unchanged by a remount
V7  grad flow    every attitude channel and every cov parameter receives gradient
V8  SPD + params Sigma is positive definite; parameter counts vs the submitted model
"""
import sys
import os

import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation as Rot
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import net_dict                                          # noqa: E402
from model.attitude import attitude_feature, rep_of                 # noqa: E402
from model.loss_func import full_ln_cov_loss, diag_ln_cov_loss      # noqa: E402

DT = torch.float64
import os
SEQ = os.path.join(os.environ.get("GINIO_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "local_data")),
                   "nanobench_processed_2nd_version/B11_random_waypoints_rep1/imu0_resampled.npy")
TOL_EXACT, TOL_E2E = 1e-12, 1e-9
CONF = SimpleNamespace(propcov=True)

ARMS = [("A0  Log(R_WS)      [submitted]", "vn_a0_spec_small"),
        ("A1  R_WS^T e_z", "vn_a1_spec_small"),
        ("A2  rows of R_WS", "vn_a2_spec_small"),
        ("A3  Log(R_last^T R_t)", "vn_a3_spec_small"),
        ("--  no attitude    [control]", "vn_noatt_spec_small")]

# ---------------------------------------------------------------- real NanoBench data
a = np.load(SEQ)
T, B = 200, 4
starts = np.linspace(0, len(a) - T - 1, B).astype(int)
gyro = torch.tensor(np.stack([a[s:s + T, 1:4] for s in starts]), dtype=DT)
acc = torch.tensor(np.stack([a[s:s + T, 4:7] for s in starts]), dtype=DT)
quat = torch.tensor(np.stack([a[s:s + T, 7:11] for s in starts]), dtype=DT)
# stored quaternions are unit only to ~5e-9; renormalise so the tolerances below
# measure the map, not the dataset's float32-era rounding
quat = quat / quat.norm(dim=-1, keepdim=True)
R_WS = pp.SO3(quat)
DATA = {"gyro": gyro, "acc": acc}
results = []


def remount(R, Q):
    """R_WS -> R_WS Q^T, the right action of a sensor remount."""
    return R @ pp.SO3(torch.tensor(Q.inv().as_quat(), dtype=DT))


def rand_Q(seed):
    rng = np.random.default_rng(seed)
    ax = rng.standard_normal(3)
    return Rot.from_rotvec(ax / np.linalg.norm(ax) * rng.uniform(0.2, np.pi))


def rec(name, val, tol, fmt="{:.3e}", expect_fail=False):
    ok = val < tol
    if expect_fail:
        print(f"   {name:44s} {fmt.format(val):>12s}   "
              f"{'BROKEN (expected)' if not ok else 'unexpectedly clean'}")
        results.append(not ok)          # the defect must still be present
        return ok
    results.append(ok)
    print(f"   {name:44s} {fmt.format(val):>12s}   {'PASS' if ok else 'FAIL'}")
    return ok


hdr = lambda s: print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)

# ---------------------------------------------------------------------------- V0
hdr("V0  regression — patched feature path == submitted `.Log().tensor()`")
# float64 precision, not bitwise: attitude_feature renormalises the quaternion first
# (see model/attitude.py), which is an identity on an already-unit quaternion up to one ULP
rec("logmap encoding reproduces submitted feature",
    (attitude_feature(R_WS, "logmap") - R_WS.Log().tensor()).abs().max().item(), 1e-15)

# ---------------------------------------------------------------------------- V1
hdr("V1  feature law   max || f(R Q^T) - Q f(R) ||   over 8 remounts (float64)")
for label, key in ARMS:
    rep = getattr(net_dict[key], "att_rep")
    if rep is None:
        continue
    worst = 0.0
    for s in range(8):
        Q = rand_Q(s)
        Qm = torch.tensor(Q.as_matrix(), dtype=DT)
        f0 = attitude_feature(R_WS, rep)
        fQ = attitude_feature(remount(R_WS, Q), rep)
        if f0.dim() == 3:
            f0, fQ = f0.unsqueeze(-2), fQ.unsqueeze(-2)
        worst = max(worst, (fQ - torch.einsum("ij,btcj->btci", Qm, f0)).abs().max().item())
    rec(label, worst, TOL_EXACT, expect_fail=key == "vn_a0_spec_small")

# ---------------------------------------------------------------------------- V2
hdr("V2  losslessness — is R_WS recoverable from the A2 channels?")
rows = attitude_feature(R_WS, "rows")
rec("|| stack(rows) - R_WS ||  (rows ARE the matrix)",
    (rows - R_WS.matrix()).abs().max().item(), 1e-15)
rec("|| Log(recovered) - Log(R_WS) ||   (bijective with A0)",
    (pp.mat2SO3(rows).Log().tensor() - R_WS.Log().tensor()).abs().max().item(), 1e-12)
rec("A2 has no branch cut: max channel norm deviation from 1",
    (rows.norm(dim=-1) - 1).abs().max().item(), 1e-12)

# ---------------------------------------------------------------------------- V3-V6
hdr("V3/V4/V6  end-to-end through the network (random init, float64, 5 remounts)")
print(f"   {'arm':30s} {'V3 mean-eq':>12s} {'V4 cov-law':>12s} {'V6 dNLL':>12s}")
print("   " + "-" * 72)
e2e = {}
for label, key in ARMS:
    torch.manual_seed(11)
    net = net_dict[key](CONF).double().eval()
    rep = rep_of(net)
    rot0 = attitude_feature(R_WS, rep)
    with torch.no_grad():
        o0 = net(DATA, rot0)
    targ = torch.randn_like(o0["net_vel"])
    nll0 = full_ln_cov_loss(o0["net_vel"] - targ, o0["cov_R"], o0["cov"]).item()

    m_err = c_err = n_err = 0.0
    for s in range(5):
        Q = rand_Q(100 + s)
        Qm = torch.tensor(Q.as_matrix(), dtype=DT)
        dQ = {"gyro": torch.einsum("ij,btj->bti", Qm, gyro),
              "acc": torch.einsum("ij,btj->bti", Qm, acc)}
        rotQ = attitude_feature(remount(R_WS, Q), rep)
        with torch.no_grad():
            oQ = net(dQ, rotQ)
        need_v = torch.einsum("ij,btj->bti", Qm, o0["net_vel"])
        m_err = max(m_err, (oQ["net_vel"] - need_v).abs().max().item())

        S0 = (o0["cov_R"].reshape(B, -1, 3, 3) @ torch.diag_embed(o0["cov"])
              @ o0["cov_R"].reshape(B, -1, 3, 3).transpose(-1, -2))
        SQ = (oQ["cov_R"].reshape(B, -1, 3, 3) @ torch.diag_embed(oQ["cov"])
              @ oQ["cov_R"].reshape(B, -1, 3, 3).transpose(-1, -2))
        need_S = torch.einsum("ij,btjk,lk->btil", Qm, S0, Qm)
        c_err = max(c_err, ((SQ - need_S).norm(dim=(-2, -1))
                            / need_S.norm(dim=(-2, -1))).max().item())
        # the objective must not care where the sensor is bolted
        nllQ = full_ln_cov_loss(oQ["net_vel"] - torch.einsum("ij,btj->bti", Qm, targ),
                                oQ["cov_R"], oQ["cov"]).item()
        n_err = max(n_err, abs(nllQ - nll0) / (abs(nll0) + 1e-12))
    e2e[key] = (m_err, c_err, n_err)
    ok = max(m_err, c_err, n_err) < TOL_E2E
    if key == "vn_a0_spec_small":                      # the defect under test
        results.append(not ok)
        tag = "BROKEN (expected)" if not ok else "unexpectedly clean"
    else:
        results.append(ok)
        tag = "PASS" if ok else "FAIL"
    print(f"   {label:30s} {m_err:12.3e} {c_err:12.3e} {n_err:12.3e}   {tag}")

print("\n   for reference, the SUBMITTED heads under the same physical remount:")
torch.manual_seed(11)
sub = net_dict["vn_codewithrot_small"](CONF).double().eval()
Q = rand_Q(100)
Qm = torch.tensor(Q.as_matrix(), dtype=DT)
with torch.no_grad():
    s0 = sub(DATA, attitude_feature(R_WS, "logmap"))
    sQ = sub({"gyro": torch.einsum("ij,btj->bti", Qm, gyro),
              "acc": torch.einsum("ij,btj->bti", Qm, acc)},
             attitude_feature(remount(R_WS, Q), "logmap"))
sub_m = (sQ["net_vel"] - torch.einsum("ij,btj->bti", Qm, s0["net_vel"])).abs().max().item()
sub_c = ((torch.diag_embed(sQ["cov"]) - torch.einsum(
    "ij,btjk,lk->btil", Qm, torch.diag_embed(s0["cov"]), Qm)).norm(dim=(-2, -1))
    / torch.diag_embed(s0["cov"]).norm(dim=(-2, -1))).max().item()
print(f"   {'vn_codewithrot_small (A0+diag)':30s} {sub_m:12.3e} {sub_c:12.3e}"
      f"          <- the defect being fixed")

# ---------------------------------------------------------------------------- V5
hdr("V5  protocol agreement — for A2, does supplement G.1 mean the physical remount?")
torch.manual_seed(11)
net = net_dict["vn_a2_spec_small"](CONF).double().eval()
Q = rand_Q(7)
Qm = torch.tensor(Q.as_matrix(), dtype=DT)
f_phys = attitude_feature(remount(R_WS, Q), "rows")
f_g1 = torch.einsum("ij,btcj->btci", Qm, attitude_feature(R_WS, "rows"))
rec("A2  || f_G.1 - f_physical ||", (f_g1 - f_phys).abs().max().item(), TOL_EXACT)
f0 = attitude_feature(R_WS, "logmap")
rec("A0  || f_G.1 - f_physical ||   [submitted]",
    (torch.einsum("ij,btj->bti", Qm, f0)
     - attitude_feature(remount(R_WS, Q), "logmap")).abs().max().item(),
    TOL_EXACT, expect_fail=True)

# ---------------------------------------------------------------------------- V7
hdr("V7  gradient flow")
for label, key in ARMS:
    torch.manual_seed(11)
    net = net_dict[key](CONF).double()
    rep = rep_of(net)
    r = None if rep is None else attitude_feature(R_WS, rep).clone().requires_grad_(True)
    out = net(DATA, r)
    loss = full_ln_cov_loss(out["net_vel"] - torch.randn_like(out["net_vel"]),
                            out["cov_R"], out["cov"])
    loss.backward()
    dead = [n for n, p in net.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    if r is None:
        ch = "n/a"
        ok = not dead
    else:
        g = r.grad.abs().mean(dim=(0, 1, -1)) if r.dim() == 4 else r.grad.abs().mean()
        ch = ", ".join(f"{v:.1e}" for v in np.atleast_1d(g.detach().numpy()))
        ok = (not dead) and float(np.atleast_1d(g.detach().numpy()).min()) > 0
    results.append(ok)
    print(f"   {label:30s} att-ch |grad| = {ch:26s} dead params: {len(dead):2d}   "
          f"{'PASS' if ok else 'FAIL'}")

# ---------------------------------------------------------------------------- V8
hdr("V8  covariance is SPD, and parameter cost vs the submitted model")
torch.manual_seed(11)
net = net_dict["vn_a2_spec_small"](CONF).double().eval()
with torch.no_grad():
    o = net(DATA, attitude_feature(R_WS, "rows"))
V = o["cov_R"].reshape(B, -1, 3, 3)
S = V @ torch.diag_embed(o["cov"]) @ V.transpose(-1, -2)
ev = torch.linalg.eigvalsh(S)
rec("orthonormality  || V^T V - I ||", (V.transpose(-1, -2) @ V
                                        - torch.eye(3, dtype=DT)).abs().max().item(), 1e-12)
rec("right-handed    max | det V - 1 |", (torch.linalg.det(V) - 1).abs().max().item(), 1e-12)
rec("SPD             -min eigenvalue", -ev.min().item(), 0.0, "{:.3e}")
rec("symmetry        || S - S^T ||", (S - S.transpose(-1, -2)).abs().max().item(), 1e-14)
print(f"      initial variance: mean {o['cov'].mean():.3e}  "
      f"(submitted head starts at exp(-5) = {np.exp(-5.0):.3e})")

print()
n_sub = sum(p.numel() for p in net_dict["vn_codewithrot_small"](CONF).parameters())
print(f"   {'model':34s} {'params':>10s} {'vs submitted':>14s}")
print("   " + "-" * 62)
for label, key in [("vn_codewithrot_small [submitted]", "vn_codewithrot_small")] + \
                  [(l.split("  ")[0] + "  " + k, k) for l, k in ARMS]:
    n = sum(p.numel() for p in net_dict[key](CONF).parameters())
    print(f"   {label:34s} {n:10,d} {n - n_sub:+14,d}")

print("\n" + "=" * 78)
print(f"{sum(results)}/{len(results)} checks passed"
      f"{'  — ALL GREEN' if all(results) else '  — FAILURES PRESENT'}")
print("=" * 78)
sys.exit(0 if all(results) else 1)
