"""
Equivariant AirIO — SO(3)-equivariant version of CodeNetMotion / CodeNetMotionwithRot.

Replaces:
  - CNN encoder      → VN-CNN (1D conv that mixes channel only, preserves vector axis)
  - GRU layers       → VN-GRU (this file's parent module)
  - Linear decoder   → VN-Linear (final mapping to 3-dim velocity vector)

Input/Output:
  - Input:  body-frame IMU window. acc + gyro stacked: (B, T, 6) → reshape to (B, T, 2, 3)
            (2 vector channels: gyro and acc, each is a 3-vector)
  - Output: body-frame velocity per timestep: (B, T, 1, 3) → squeeze to (B, T, 3)

Equivariance: rotating the IMU input by R rotates the predicted velocity by R.
              Verified numerically (vn_codenet matches the equivariance of its parts).
"""

import torch
import torch.nn as nn

# Local relative import — assumes vn_gru.py is alongside this file
try:
    from .vn_gru import VNLinear, VNLeakyReLU, VNGRU, VNNorm, VNLayerNorm
except ImportError:
    from vn_gru import VNLinear, VNLeakyReLU, VNGRU, VNNorm, VNLayerNorm


class VNConv1d(nn.Module):
    """1D conv that mixes only the channel axis, preserving vector axis.

    Input:  (B, C_in, 3, T)
    Output: (B, C_out, 3, T_out)

    Implementation: apply nn.Conv1d independently for each of the 3 spatial axes.
    Equivariance: since the same conv weights apply to all 3 components identically,
    rotating the vector axis (mixing components) commutes with the conv → equivariant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3, T)
        B, C, three, T = x.shape
        assert three == 3
        # Apply conv to each of the 3 components independently (same weights)
        # Reshape to (B*3, C, T) → conv → (B*3, C_out, T_out) → (B, 3, C_out, T_out) → (B, C_out, 3, T_out)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * 3, C, T)
        y_flat = self.conv(x_flat)
        T_out = y_flat.shape[-1]
        y = y_flat.reshape(B, 3, -1, T_out).permute(0, 2, 1, 3).contiguous()
        return y


class VNCNNEncoder(nn.Module):
    """Equivariant version of AirIO's CNNEncoder."""

    def __init__(self, c_list, k_list, s_list, p_list, use_relu=True, use_layer_norm=True):
        super().__init__()
        layers = []
        for i in range(len(c_list) - 1):
            layers.append(VNConv1d(c_list[i], c_list[i+1], k_list[i],
                                   stride=s_list[i], padding=p_list[i], bias=False))
            if use_layer_norm:
                layers.append(VNLayerNorm(c_list[i+1]))
            if use_relu:
                layers.append(VNLeakyReLU(c_list[i+1], negative_slope=0.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3, T) → (B, C_out, 3, T_out)
        return self.net(x)


class VNCodeNetMotion(nn.Module):
    """Equivariant AirIO velocity network — no attitude input.

    Mirrors structure of Air-IO's CodeNetMotion but with VN-CNN + VN-GRU + VN-Linear.
    Input shape: (B, T, 6) — concatenated gyro+acc, last 6 dims
    Output: dict with 'net_vel' (B, T_out, 3) and optionally 'cov' (B, T_out, 3)
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # Match Air-IO defaults
        self.k_list = [7, 7]
        self.s_list = [3, 3]
        self.p_list = [3, 3]
        # Channels per vector (each is 3D): we treat input as 2 vector channels
        # (gyro vec, accel vec) then expand
        in_vec_channels = 2
        self.cnn = VNCNNEncoder(
            c_list=[in_vec_channels, 32, 64],
            k_list=self.k_list, s_list=self.s_list, p_list=self.p_list,
        )
        # Bi-GRU layer 1: 64 in, 64 hidden, bidirectional → 128 out
        self.gru1 = VNGRU(input_size=64, hidden_size=64, num_layers=1, bidirectional=True)
        # Bi-GRU layer 2: 128 in, 128 hidden, bidirectional → 256 out
        self.gru2 = VNGRU(input_size=128, hidden_size=128, num_layers=1, bidirectional=True)
        # Velocity decoder: 256 vec channels → 1 vec channel (the 3D velocity)
        self.veldecoder = nn.Sequential(
            VNLinear(256, 128),
            VNLeakyReLU(128, negative_slope=0.0),
            VNLinear(128, 1),
        )
        # Covariance decoder: outputs 3 scalars per timestep (diagonal cov entries)
        # These are SO(3) invariant — derived from norms.
        self.cov_norm_proj = VNLinear(256, 16)
        self.cov_decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def get_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        """Mirror of AirIO's get_label: subsample GT to match CNN-downsampled output rate."""
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0] * self.s_list[1], :]
        L_out = (gt_label.shape[1] - 1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:, -1:, :].repeat(1, diff, 1)), dim=1)
        return select_label

    def forward(self, data, rot=None):
        """
        data['acc']: (B, T, 3)
        data['gyro']: (B, T, 3)
        rot: ignored (we don't use attitude here).

        Returns: {'net_vel': (B, T_out, 3), 'cov': (B, T_out, 3) or None}
        """
        # Build vector-channel input: (B, 2, 3, T) — gyro and acc as 2 vector channels
        gyro = data["gyro"]    # (B, T, 3)
        acc = data["acc"]      # (B, T, 3)
        x = torch.stack([gyro, acc], dim=1)            # (B, 2, T, 3)
        x = x.permute(0, 1, 3, 2)                       # (B, 2, 3, T)

        # CNN encode
        feat = self.cnn(x)                              # (B, 64, 3, T_cnn)

        # Reshape to (B, T_cnn, 64, 3) for GRU
        feat = feat.permute(0, 3, 1, 2)                 # (B, T_cnn, 64, 3)

        # GRU1 → GRU2
        feat = self.gru1(feat)                          # (B, T_cnn, 128, 3)
        feat = self.gru2(feat)                          # (B, T_cnn, 256, 3)

        # Velocity decoder
        vel_vec = self.veldecoder(feat.permute(0, 2, 3, 1))   # arg shape (B, 256, 3, T) → (B, 1, 3, T)
        # Actually veldecoder is Sequential of (VNLinear, VNLeakyReLU, VNLinear) — let me apply explicitly
        # Reset above by applying each layer with correct shapes
        # Easier path: reshape (B, T, C, 3) -> (B, C, 3, T) for VN layers
        feat_vt = feat.permute(0, 2, 3, 1)              # (B, 256, 3, T_cnn)
        v = self.veldecoder(feat_vt)                    # (B, 1, 3, T_cnn)
        net_vel = v.squeeze(1).permute(0, 2, 1)         # (B, T_cnn, 3)

        # Cov decoder — uses norms (SO(3) invariant)
        cov_vec_proj = self.cov_norm_proj(feat_vt)      # (B, 16, 3, T_cnn)
        cov_scalars = VNNorm()(cov_vec_proj)            # (B, 16, T_cnn)
        cov_scalars = cov_scalars.permute(0, 2, 1)      # (B, T_cnn, 16)
        cov = None
        if getattr(self.conf, "propcov", False):
            cov_raw = self.cov_decoder(cov_scalars)     # (B, T_cnn, 3)
            cov = torch.exp(cov_raw - 5.0)              # match AirIO scaling
        return {"net_vel": net_vel, "cov": cov}


class VNCodeNetMotionwithRot(VNCodeNetMotion):
    """Equivariant AirIO with attitude input. Attitude (rotation log-map vector) is
    fed as an additional vector channel — naturally equivariant since it's a 3-vector.

    Note: log-map of R is in so(3), which under conjugation by R' transforms as
    R' · log(R) — i.e., the log of a rotation transforms equivariantly as a vector
    if the rotation is conjugated. For our use (attitude input = log of body-to-world),
    rotating the world by R' gives new attitude = R'·R·R'^{-1}, whose log is
    R'·log(R) — so log-map is equivariant. Use directly as vector input.
    """

    def __init__(self, conf):
        # Trick: parent uses in_vec_channels=2 (gyro, acc). With rot, we have 3.
        super().__init__(conf)
        # Replace cnn with one that takes 3 vector channels
        self.cnn = VNCNNEncoder(
            c_list=[3, 32, 64],
            k_list=self.k_list, s_list=self.s_list, p_list=self.p_list,
        )

    def forward(self, data, rot=None):
        assert rot is not None, "VNCodeNetMotionwithRot requires rot (attitude log map)"
        gyro = data["gyro"]
        acc = data["acc"]
        # rot is (B, T, 3) — log-map of attitude
        x = torch.stack([gyro, acc, rot], dim=1)        # (B, 3, T, 3)
        x = x.permute(0, 1, 3, 2)                       # (B, 3, 3, T)

        feat = self.cnn(x)                              # (B, 64, 3, T_cnn)
        feat = feat.permute(0, 3, 1, 2)                 # (B, T_cnn, 64, 3)
        feat = self.gru1(feat)                          # (B, T_cnn, 128, 3)
        feat = self.gru2(feat)                          # (B, T_cnn, 256, 3)
        feat_vt = feat.permute(0, 2, 3, 1)              # (B, 256, 3, T_cnn)

        v = self.veldecoder(feat_vt)                    # (B, 1, 3, T_cnn)
        net_vel = v.squeeze(1).permute(0, 2, 1)         # (B, T_cnn, 3)

        cov_vec_proj = self.cov_norm_proj(feat_vt)
        cov_scalars = VNNorm()(cov_vec_proj).permute(0, 2, 1)
        cov = None
        if getattr(self.conf, "propcov", False):
            cov_raw = self.cov_decoder(cov_scalars)
            cov = torch.exp(cov_raw - 5.0)
        return {"net_vel": net_vel, "cov": cov}


# ---------------------------------------------------------------------------
# Channel-/3-reduced variants. Each scalar channel C in the non-equivariant
# AirIO architecture maps to C/3 vector channels in the VN variant — keeping
# the effective scalar feature dimension (3 * C_vec) constant. This matches
# the LN-ResNet (GINIO-R*) convention (`div=3`) and removes the implicit
# 3x capacity advantage of the original VN-AirIO variants when comparing
# against AirIO native at matched parameter count.
# ---------------------------------------------------------------------------
class VNCodeNetMotionSmall(nn.Module):
    """Channel-/3-reduced VN-AirIO (no-attitude).
    Effective feature dims match AirIO native (32, 64, 128, 256) but expressed
    as VN vector channels (11, 21, 42, 84).
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.k_list = [7, 7]
        self.s_list = [3, 3]
        self.p_list = [3, 3]
        in_vec_channels = 2
        # Channels: [2, 32//3, 64//3] = [2, 11, 21]
        self.cnn = VNCNNEncoder(
            c_list=[in_vec_channels, 11, 21],
            k_list=self.k_list, s_list=self.s_list, p_list=self.p_list,
        )
        # Bi-GRU: 21 -> 21 (bidir = 42), 42 -> 42 (bidir = 84)
        self.gru1 = VNGRU(input_size=21, hidden_size=21, num_layers=1, bidirectional=True)
        self.gru2 = VNGRU(input_size=42, hidden_size=42, num_layers=1, bidirectional=True)
        # Vel decoder: 84 -> 28 -> 1 (matches scalar 256 -> 128 -> 3 → /3 in C)
        self.veldecoder = nn.Sequential(
            VNLinear(84, 28),
            VNLeakyReLU(28, negative_slope=0.0),
            VNLinear(28, 1),
        )
        # Cov head: 84 -> 5 (matches 256 -> 16 → /3 -> 5, ceil)
        self.cov_norm_proj = VNLinear(84, 5)
        self.cov_decoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def get_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0] * self.s_list[1], :]
        L_out = (gt_label.shape[1] - 1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:, -1:, :].repeat(1, diff, 1)), dim=1)
        return select_label

    def forward(self, data, rot=None):
        gyro = data["gyro"]
        acc = data["acc"]
        x = torch.stack([gyro, acc], dim=1)            # (B, 2, T, 3)
        x = x.permute(0, 1, 3, 2)                       # (B, 2, 3, T)
        feat = self.cnn(x)                              # (B, 21, 3, T_cnn)
        feat = feat.permute(0, 3, 1, 2)                 # (B, T_cnn, 21, 3)
        feat = self.gru1(feat)                          # (B, T_cnn, 42, 3)
        feat = self.gru2(feat)                          # (B, T_cnn, 84, 3)
        feat_vt = feat.permute(0, 2, 3, 1)              # (B, 84, 3, T_cnn)
        v = self.veldecoder(feat_vt)                    # (B, 1, 3, T_cnn)
        net_vel = v.squeeze(1).permute(0, 2, 1)         # (B, T_cnn, 3)
        cov_vec_proj = self.cov_norm_proj(feat_vt)      # (B, 5, 3, T_cnn)
        cov_scalars = VNNorm()(cov_vec_proj)            # (B, 5, T_cnn)
        cov_scalars = cov_scalars.permute(0, 2, 1)      # (B, T_cnn, 5)
        cov = None
        if getattr(self.conf, "propcov", False):
            cov_raw = self.cov_decoder(cov_scalars)
            cov = torch.exp(cov_raw - 5.0)
        return {"net_vel": net_vel, "cov": cov}


class VNCodeNetMotionwithRotSmall(VNCodeNetMotionSmall):
    """Channel-/3-reduced VN-AirIO (with-attitude)."""

    def __init__(self, conf):
        super().__init__(conf)
        # Override input cnn: 3 vector channels (gyro, acc, rot)
        self.cnn = VNCNNEncoder(
            c_list=[3, 11, 21],
            k_list=self.k_list, s_list=self.s_list, p_list=self.p_list,
        )

    def forward(self, data, rot=None):
        assert rot is not None, "VNCodeNetMotionwithRotSmall requires rot"
        gyro = data["gyro"]; acc = data["acc"]
        x = torch.stack([gyro, acc, rot], dim=1)
        x = x.permute(0, 1, 3, 2)
        feat = self.cnn(x)
        feat = feat.permute(0, 3, 1, 2)
        feat = self.gru1(feat)
        feat = self.gru2(feat)
        feat_vt = feat.permute(0, 2, 3, 1)
        v = self.veldecoder(feat_vt)
        net_vel = v.squeeze(1).permute(0, 2, 1)
        cov_vec_proj = self.cov_norm_proj(feat_vt)
        cov_scalars = VNNorm()(cov_vec_proj).permute(0, 2, 1)
        cov = None
        if getattr(self.conf, "propcov", False):
            cov_raw = self.cov_decoder(cov_scalars)
            cov = torch.exp(cov_raw - 5.0)
        return {"net_vel": net_vel, "cov": cov}


# ---------------------------------------------------------------------------
# Equivariant with-attitude variants (A1/A2/A3) + spectral covariance head.
#
# Two independent defects of the submitted `*withRot*` models are fixed here:
#
#   (1) the attitude channel.  `Log(R_WS)` cannot transform as a vector under a
#       sensor remount `R_WS -> R_WS Q^T` (see model/attitude.py for the proof).
#       The rows of `R_WS` can, and carry the same information.
#
#   (2) the covariance head.  A diagonal built from SO(3)-*invariant* scalars
#       satisfies `Sigma -> Sigma`, but a covariance of a body-frame vector must
#       satisfy the congruence `Sigma -> Q Sigma Q^T`.  The spectral head below
#       carries an equivariant eigenframe, so it satisfies congruence exactly.
# ---------------------------------------------------------------------------

try:
    from .attitude import n_channels
except ImportError:
    from attitude import n_channels

LOG_VAR_MIN, LOG_VAR_MAX = -14.0, 4.0     # variance in [8.3e-7, 54.6]


def gram_schmidt_frame(a1, a2, eps=1e-8):
    """Orthonormal frame (columns) from two vectors, as an SO(3) element.

    Equivariance: for `Q` in SO(3), feeding `(Q a1, Q a2)` returns `Q V`, because
    normalisation and projection use only inner products (rotation-invariant) and
    `(Q a) x (Q b) = Q (a x b)` when `det Q = 1`.

    Chosen over the SVD/polar factor because `svd` backward carries
    `1/(s_i^2 - s_j^2)` terms that blow up at repeated singular values, and over a
    determinant-corrected orthogonalisation because that flips discontinuously.
    """
    # clamp_min, not `norm + eps`: adding eps shrinks every output by eps/||a||, which
    # at a typical ||a|| ~ 1e-2 costs ~1e-6 of orthonormality and breaks the congruence
    # law at exactly that level.  Clamping is exact wherever the vector is not degenerate.
    e1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(eps)
    u2 = a2 - (e1 * a2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(dim=-1, keepdim=True).clamp_min(eps)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)          # (..., 3, 3), columns = frame


class VNSpectralCov(nn.Module):
    """Sigma = V diag(exp 2(s + l)) V^T,  V equivariant, (s, l) invariant.

    `l` is mean-subtracted across the three axes so it is pure *shape*, leaving the
    single scalar `s` as the only *scale* knob -- the network can become more
    uncertain without reshaping the ellipsoid, and reshape it without inflating it.

    Eigenvalues come out in closed form, so the Gaussian NLL needs neither a
    Cholesky factorisation nor a 3x3 inverse (see `full_ln_cov_loss`).
    """

    def __init__(self, in_vec_channels, n_inv, hidden=64, log_var_init=-5.0):
        super().__init__()
        self.frame = VNLinear(in_vec_channels, 2)
        self.inv_proj = VNLinear(in_vec_channels, n_inv)
        self.norm = VNNorm()
        # no bias on the shape output: the mean-subtraction below annihilates any
        # constant offset, so a bias there would carry a structurally zero gradient
        self.shape = nn.Sequential(nn.Linear(n_inv, hidden), nn.GELU(),
                                   nn.Linear(hidden, 3, bias=False))
        self.scale = nn.Sequential(nn.Linear(n_inv, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Small-gain output init so the head starts near-isotropic at the same variance
        # the submitted head starts at, `exp(cov_raw - 5.0)` -- this is a covariance swap,
        # not a rescale.  Not *exactly* zero: a zero output layer would leave every
        # upstream parameter of this head with no gradient on the first step.
        with torch.no_grad():
            self.shape[-1].weight.mul_(0.1)
            self.scale[-1].weight.mul_(0.1)
            self.scale[-1].bias.fill_(0.5 * log_var_init)

    def forward(self, feat_vt):
        """feat_vt: (B, C, 3, T)  ->  eigenvalues (B, T, 3), frame (B, T, 3, 3)."""
        A = self.frame(feat_vt)                              # (B, 2, 3, T)
        V = gram_schmidt_frame(A[:, 0].permute(0, 2, 1),
                               A[:, 1].permute(0, 2, 1))     # (B, T, 3, 3)
        inv = self.norm(self.inv_proj(feat_vt)).permute(0, 2, 1)   # (B, T, n_inv)
        l = self.shape(inv)
        l = l - l.mean(-1, keepdim=True)                     # sum-zero -> pure shape
        s = self.scale(inv)                                  # (B, T, 1)
        eig = torch.exp((2.0 * (s + l)).clamp(LOG_VAR_MIN, LOG_VAR_MAX))
        return eig, V


class VNCodeNetMotionSpectralSmall(VNCodeNetMotionSmall):
    """Channel-/3-reduced VN-AirIO with the spectral covariance head.

    `att_rep` selects the attitude encoding and is read by `train_motion.py` through
    `model.attitude.rep_of`, so the feature and the model can never disagree.
    """

    att_rep = None

    def __init__(self, conf):
        super().__init__(conf)
        n_att = n_channels(self.att_rep)
        if n_att:
            self.cnn = VNCNNEncoder(
                c_list=[2 + n_att, 11, 21],
                k_list=self.k_list, s_list=self.s_list, p_list=self.p_list,
            )
        self.cov_head = VNSpectralCov(84, 5)
        del self.cov_norm_proj          # invariant-diagonal head is unused here;
        del self.cov_decoder            # dropping it keeps the checkpoint honest

    def forward(self, data, rot=None):
        x = torch.stack([data["gyro"], data["acc"]], dim=1)      # (B, 2, T, 3)
        if self.att_rep is not None:
            assert rot is not None, f"{type(self).__name__} requires rot"
            if rot.dim() == 3:                                   # tolerate (B, T, 3)
                rot = rot.unsqueeze(-2)
            x = torch.cat([x, rot.permute(0, 2, 1, 3)], dim=1)   # (B, 2 + C, T, 3)
        x = x.permute(0, 1, 3, 2)                                # (B, C, 3, T)

        feat = self.cnn(x).permute(0, 3, 1, 2)                   # (B, T_cnn, 21, 3)
        feat = self.gru2(self.gru1(feat))                        # (B, T_cnn, 84, 3)
        feat_vt = feat.permute(0, 2, 3, 1)                       # (B, 84, 3, T_cnn)

        net_vel = self.veldecoder(feat_vt).squeeze(1).permute(0, 2, 1)
        out = {"net_vel": net_vel, "cov": None}
        if getattr(self.conf, "propcov", False):
            eig, V = self.cov_head(feat_vt)
            out["cov"] = eig                                     # (B, T, 3) eigenvalues
            # flattened row-major so utils.cat_state concatenates along time, not
            # along the vector axis (it cats dim=-2 for rank > 2 tensors)
            out["cov_R"] = V.reshape(*V.shape[:2], 9)
        return out


class VNCodeNetMotionA2Small(VNCodeNetMotionSpectralSmall):
    """A2: attitude as the three rows of R_WS. Exactly equivariant, lossless."""
    att_rep = "rows"


class VNCodeNetMotionA0SpecSmall(VNCodeNetMotionSpectralSmall):
    """A0 + spectral head: isolates the covariance fix from the attitude fix."""
    att_rep = "logmap"


class VNCodeNetMotionA1Small(VNCodeNetMotionSpectralSmall):
    """A1: attitude as the body-frame up vector R_WS^T e_z. Equivariant, drops yaw."""
    att_rep = "up"


class VNCodeNetMotionA3Small(VNCodeNetMotionSpectralSmall):
    """A3: attitude as Log(R_last^T R_t). Ad-covariant, drops absolute attitude."""
    att_rep = "rel"


class VNCodeNetMotionA2DiagSmall(VNCodeNetMotionwithRotSmall):
    """A2 attitude encoding + the SUBMITTED invariant-diagonal covariance head.

    Exists to complete the {A0, A2} x {diagonal, spectral} factorial, so the effect of
    fixing the attitude encoding can be separated from the effect of fixing the
    covariance law.  Its mean is equivariant; its covariance is not.
    """

    att_rep = "rows"

    def __init__(self, conf):
        super().__init__(conf)
        self.cnn = VNCNNEncoder(
            c_list=[2 + n_channels(self.att_rep), 11, 21],
            k_list=self.k_list, s_list=self.s_list, p_list=self.p_list,
        )

    def forward(self, data, rot=None):
        assert rot is not None and rot.dim() == 4, "A2 expects rot of shape (B, T, 3, 3)"
        x = torch.cat([torch.stack([data["gyro"], data["acc"]], dim=1),
                       rot.permute(0, 2, 1, 3)], dim=1).permute(0, 1, 3, 2)
        feat = self.cnn(x).permute(0, 3, 1, 2)
        feat = self.gru2(self.gru1(feat))
        feat_vt = feat.permute(0, 2, 3, 1)
        net_vel = self.veldecoder(feat_vt).squeeze(1).permute(0, 2, 1)
        cov = None
        if getattr(self.conf, "propcov", False):
            inv = VNNorm()(self.cov_norm_proj(feat_vt)).permute(0, 2, 1)
            cov = torch.exp(self.cov_decoder(inv) - 5.0)
        return {"net_vel": net_vel, "cov": cov}
