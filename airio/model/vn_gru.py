"""
SO(3)-equivariant GRU layers using Vector Neuron (VN) ops.

Hidden state and input are vector fields h ∈ R^(B, C, 3, T) where C is channels
and the trailing 3 is the vector dimension that transforms under SO(3).

Design (following Vector Neuron LSTM, Deng et al. NeurIPS 2021):
- Gates (reset, update) are *scalar* values derived from SO(3)-invariant norms of
  input/hidden vectors. Scalar gates multiply vector channels element-wise → preserves
  equivariance (Rv * s = s * Rv).
- Candidate hidden state h_tilde uses VN-Linear (equivariant) on input and hidden,
  followed by a VN nonlinearity (VNLeakyReLU acts on direction without breaking SO(3)).
- The VN-Linear layer is a regular nn.Linear applied to the channel dim — leaves the
  vector dim (3) untouched, so each component (x, y, z) is mixed identically.

This file does NOT depend on the imu-equivariant-learning codebase; the VN ops are
re-implemented here for AirIO to keep the project self-contained.

Equivariance contract:
    For any R ∈ SO(3):  vn_gru(R @ x, R @ h0) == R @ vn_gru(x, h0)
    (element-wise multiply with rotation matrix on the vector axis).

Numerical verification: see test_vn_equivariance.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks (re-implementation of VN ops, self-contained for AirIO)
# ──────────────────────────────────────────────────────────────────────────────


class VNLinear(nn.Module):
    """Equivariant linear over channel dim. Input/output shape: (..., C_in, 3) or (..., C_in, 3, T)."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C_in, 3, [T])
        # Move channel to last for nn.Linear, then move back.
        x_last = x.transpose(-2 if x.dim() == 3 else -3, -1) if False else None  # noop placeholder
        # Generic: swap channel (-3 if 4D w/ T, else -2) with last
        if x.dim() == 3:           # (B, C, 3)
            x_t = x.transpose(1, 2)            # (B, 3, C)
            y = self.lin(x_t)                  # (B, 3, C_out)
            return y.transpose(1, 2)           # (B, C_out, 3)
        elif x.dim() == 4:         # (B, C, 3, T)
            x_t = x.permute(0, 2, 3, 1)        # (B, 3, T, C)
            y = self.lin(x_t)                  # (B, 3, T, C_out)
            return y.permute(0, 3, 1, 2)       # (B, C_out, 3, T)
        else:
            raise ValueError(f"Unsupported shape {x.shape}")


class VNLeakyReLU(nn.Module):
    """Equivariant leaky-ReLU on vector field. Acts based on dot-product with a learned
    direction (also derived from input → equivariant)."""

    def __init__(self, in_channels: int, negative_slope: float = 0.2, share_nonlinearity: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        # Learned direction per channel (or shared)
        out_dir = 1 if share_nonlinearity else in_channels
        self.map_to_dir = VNLinear(in_channels, out_dir, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3) or (B, C, 3, T)
        d = self.map_to_dir(x)                # same shape as x
        dotprod = (x * d).sum(dim=-2 if x.dim() == 3 else -2, keepdim=True)
        # The "vector axis" is -2 in (B, C, 3) and (B, C, 3, T). Be explicit:
        if x.dim() == 3:
            dotprod = (x * d).sum(dim=2, keepdim=True)        # (B, C, 1)
            d_norm_sq = (d * d).sum(dim=2, keepdim=True)
        else:
            dotprod = (x * d).sum(dim=2, keepdim=True)        # (B, C, 1, T)
            d_norm_sq = (d * d).sum(dim=2, keepdim=True)
        mask = (dotprod >= 0).float()
        out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return out


class VNNorm(nn.Module):
    """Compute per-channel L2 norm — produces SO(3)-invariant scalars."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3) or (B, C, 3, T) → (B, C) or (B, C, T)
        if x.dim() == 3:
            return torch.linalg.norm(x, dim=2)  # (B, C)
        elif x.dim() == 4:
            return torch.linalg.norm(x, dim=2)  # (B, C, T)
        else:
            raise ValueError(f"Unsupported {x.shape}")


class VNLayerNorm(nn.Module):
    """Equivariant LayerNorm for VN vector channels.

    Standard nn.LayerNorm would mix vector components → break SO(3) equivariance.
    Pattern (from EqNIO/Deng et al.):
      1. Compute per-channel norm (rotation-invariant scalar).
      2. Unit-normalize each vector channel (preserve direction).
      3. Apply nn.LayerNorm on the scalar norms.
      4. Rescale unit vectors by the normalized norms.

    Why: directions stay rotation-equivariant; magnitudes get standardized across
    the channel dimension. Helps gradient flow on long sequences (TLIO pedestrian)
    where VN-GRU magnitudes can drift.

    Input shape: (B, C, 3) or (B, C, 3, T)  (vector axis at -2 when 3D, or -2 when 4D)
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # (B, C, 3): norm over vector axis 2 → (B, C)
            norms = torch.linalg.norm(x, dim=2)
            unit = x / norms.unsqueeze(-1).clamp(min=self.eps)
            # nn.LayerNorm over last dim (C)
            scaled = self.ln(norms)                              # (B, C)
            return unit * scaled.unsqueeze(-1)                   # (B, C, 3)
        elif x.dim() == 4:
            # (B, C, 3, T): norm over vector axis 2 → (B, C, T)
            norms = torch.linalg.norm(x, dim=2)                  # (B, C, T)
            unit = x / norms.unsqueeze(2).clamp(min=self.eps)    # (B, C, 3, T)
            # LayerNorm over channel axis: need to move C to last
            n_t = norms.transpose(1, 2)                          # (B, T, C)
            scaled = self.ln(n_t).transpose(1, 2)                # (B, C, T)
            return unit * scaled.unsqueeze(2)                    # (B, C, 3, T)
        else:
            raise ValueError(f"Unsupported {x.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# VN-GRU cell + sequential layer
# ──────────────────────────────────────────────────────────────────────────────


class VNGRUCell(nn.Module):
    """Vector-Neuron GRU cell.

    Hidden state h_t and input x_t are vector fields of shape (B, C, 3).

    Equations (vector-equivariant adaptation of standard GRU):
        n_x   = ||W_n x||                       # scalar features from x
        n_h   = ||W_n h_prev||                  # scalar features from h
        r     = σ(MLP_r(concat(n_x, n_h)))      # scalar gate, (B, C)
        z     = σ(MLP_z(concat(n_x, n_h)))      # scalar gate, (B, C)
        c     = vn_relu(W_xc x  + r ⊙_v (W_hc h_prev))   # candidate vector, (B, C_h, 3)
        h_new = (1 − z) ⊙_v h_prev + z ⊙_v c    # vector update

    where "⊙_v" denotes broadcasting a scalar gate (B, C, 1) across the vector axis.

    Equivariance: every vector op is equivariant; scalars (gates) are invariant.
    Composition is equivariant.
    """

    def __init__(self, in_channels: int, hidden_channels: int, gate_norm_channels: int = 32,
                 use_layer_norm: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_layer_norm = use_layer_norm

        # Vector → invariant scalars (via norms of projected vectors)
        self.norm_proj_x = VNLinear(in_channels, gate_norm_channels)
        self.norm_proj_h = VNLinear(hidden_channels, gate_norm_channels)

        # Scalar MLP for gates (operates on invariant features → invariant)
        gate_in = gate_norm_channels * 2
        self.gate_mlp_r = nn.Sequential(
            nn.Linear(gate_in, hidden_channels),
            nn.Sigmoid(),
        )
        self.gate_mlp_z = nn.Sequential(
            nn.Linear(gate_in, hidden_channels),
            nn.Sigmoid(),
        )

        # Vector linears for candidate
        self.vn_xc = VNLinear(in_channels, hidden_channels)
        self.vn_hc = VNLinear(hidden_channels, hidden_channels)
        self.vn_relu = VNLeakyReLU(hidden_channels, negative_slope=0.0)
        # Equivariant LayerNorm on the candidate (stabilizes long-sequence updates on
        # pedestrian motion, where input magnitudes drift over time)
        self.vn_ln = VNLayerNorm(hidden_channels) if use_layer_norm else nn.Identity()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x_t:    (B, C_in, 3)
        h_prev: (B, C_h, 3)
        Returns h_new: (B, C_h, 3)
        """
        # Invariant scalars
        n_x = VNNorm()(self.norm_proj_x(x_t))     # (B, gate_norm_channels)
        n_h = VNNorm()(self.norm_proj_h(h_prev))  # (B, gate_norm_channels)
        scalar_feats = torch.cat([n_x, n_h], dim=-1)  # (B, 2*gate_norm_channels)

        r = self.gate_mlp_r(scalar_feats)  # (B, C_h) ∈ (0,1), invariant
        z = self.gate_mlp_z(scalar_feats)  # (B, C_h) ∈ (0,1), invariant

        # Vector candidate
        xc = self.vn_xc(x_t)                  # (B, C_h, 3)
        hc = self.vn_hc(h_prev)               # (B, C_h, 3)
        # Apply reset gate r to h-contribution: broadcast (B, C_h) → (B, C_h, 1)
        c_pre = xc + r.unsqueeze(-1) * hc
        c = self.vn_relu(c_pre)                # (B, C_h, 3)
        c = self.vn_ln(c)                      # equivariant LN on candidate

        # Vector update: (1-z) * h_prev + z * c
        h_new = (1.0 - z).unsqueeze(-1) * h_prev + z.unsqueeze(-1) * c
        return h_new


class VNGRU(nn.Module):
    """Sequential VN-GRU layer.

    Input:  (B, T, C_in, 3)
    Output: (B, T, C_hidden, 3)
    Bidirectional supported (concatenates forward + backward along channel axis).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        gate_norm_channels: int = 32,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Stack of cells: layer 0 takes input_size, layer 1+ takes prev layer output size
        self.forward_cells = nn.ModuleList()
        for layer in range(num_layers):
            in_c = input_size if layer == 0 else (hidden_size * (2 if bidirectional else 1))
            self.forward_cells.append(VNGRUCell(in_c, hidden_size, gate_norm_channels))

        if bidirectional:
            self.backward_cells = nn.ModuleList()
            for layer in range(num_layers):
                in_c = input_size if layer == 0 else (hidden_size * 2)
                self.backward_cells.append(VNGRUCell(in_c, hidden_size, gate_norm_channels))

    def _run_one_layer(self, x: torch.Tensor, cell: VNGRUCell, reverse: bool = False) -> torch.Tensor:
        """Roll through time for a single cell. x: (B, T, C_in, 3) → (B, T, C_h, 3)."""
        B, T = x.shape[0], x.shape[1]
        seq = list(range(T - 1, -1, -1)) if reverse else list(range(T))
        h = torch.zeros(B, cell.hidden_channels, 3, device=x.device, dtype=x.dtype)
        outs_t = []
        for t in seq:
            x_t = x[:, t]
            h = cell(x_t, h)
            outs_t.append(h)
        if reverse:
            outs_t = outs_t[::-1]
        return torch.stack(outs_t, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C_in, 3)
        Returns: (B, T, C_out, 3) where C_out = hidden_size * (2 if bidirectional else 1)

        For multi-layer bidirectional: at each layer, run fwd + bwd in parallel on
        the previous layer's CONCATENATED output, then concat the two directions
        again. This matches torch.nn.GRU(bidirectional=True) behavior.
        """
        out = x
        for layer_idx in range(self.num_layers):
            fwd = self._run_one_layer(out, self.forward_cells[layer_idx], reverse=False)
            if self.bidirectional:
                bwd = self._run_one_layer(out, self.backward_cells[layer_idx], reverse=True)
                out = torch.cat([fwd, bwd], dim=2)   # concat along channel axis
            else:
                out = fwd
        return out
