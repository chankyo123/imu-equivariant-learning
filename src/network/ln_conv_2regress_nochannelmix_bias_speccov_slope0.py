"""
ln189-ANCHORED spectral/diagonal covariance head — CORRECTED slope-0.0 backbone (rebuttal path A).

FIX (2026-08-08): the DEPLOYED ln189 headline model uses VNLeakyReLU negative_slope=0.0
(arch 'ln_resnet' = ln_conv_2regress_nochannelmix_bias), NOT slope_1_bias (0.1). The two backbones
share identical weight NAMES (slope is a constant, not a parameter), so an ln189 checkpoint loads
byte-identical into either, but the FORWARD activations differ -> different mean velocity.
Proven with proper net_test: native slope-0.0 + ln189 checkpoint_best reproduces batch_test
per-seq ATE EXACTLY (median 0.906), while slope-0.1 gives 1.879. So this class subclasses the
slope-0.0 backbone; with --freeze_mean the frozen mean is byte-identical AND forward-identical to
the deployed ln189 -> the diagonal-vs-spectral filter comparison is anchored to the real 1.05 model.
"""
import numpy as np
import torch
import torch.nn as nn

# slope-0.0 backbone (the DEPLOYED ln189 variant).
from network.ln_conv_2regress_nochannelmix_bias import (
    SO3EquivariantReluBracketLayers,
    FcBlock,
    get_vector_feature,
)


def orthonormalize_via_svd(A, eps=1e-6):
    """Project A (B,3,3) onto SO(3) via SVD (out-of-place last-row flip; autograd-safe)."""
    A = A + eps * torch.eye(3, device=A.device, dtype=A.dtype).unsqueeze(0)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    R = U @ Vh
    detR = torch.det(R)
    sign = torch.sign(detR)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Vh_corrected = torch.cat([Vh[:, :-1, :], Vh[:, -1:, :] * sign.view(-1, 1, 1)], dim=1)
    R = U @ Vh_corrected
    return R


class SO3EquivariantReluBracketLayers_speccov_s0(SO3EquivariantReluBracketLayers):
    """ln189 slope-0.0 backbone+mean (loaded/frozen) + spectral/diagonal covariance head (trained)."""

    def __init__(self, block_type, in_dim, out_dim, group_sizes, inter_dim,
                 zero_init_residual=True, cov_head='spectral'):
        super().__init__(block_type, in_dim, out_dim, group_sizes, inter_dim,
                         zero_init_residual=zero_init_residual)
        self.cov_head = cov_head
        self.log_sigma_min = float(np.log(1e-3))
        self.log_sigma_max = float(np.log(10.0))

        # --- spectral head (fresh; distinct names so ln189's output_block1/2 load cleanly) ---
        self.spec_V1 = FcBlock(512 // 3 * 3, out_dim, inter_dim)
        self.spec_V2 = FcBlock(512 // 3 * 3, out_dim, inter_dim)
        self.spec_V3 = FcBlock(512 // 3 * 3, out_dim, inter_dim)
        self.spec_e1 = FcBlock(512 // 3 * 3, out_dim, inter_dim)
        self.spec_e2 = FcBlock(512 // 3 * 3, out_dim, inter_dim)
        self.spec_e3 = FcBlock(512 // 3 * 3, out_dim, inter_dim)
        self.inv_head_shared = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
            nn.Linear(32, 1, bias=True),
        )
        with torch.no_grad():
            nn.init.normal_(self.inv_head_shared[-1].weight, 0, 0.01)
            nn.init.constant_(self.inv_head_shared[-1].bias, -1.5)

        # coordinate-frame diagonal head.
        self.output_block_diag = FcBlock(512 // 3 * 3, out_dim, inter_dim)

    def _features(self, x):
        """ln189 slope-0.0 backbone forward, verbatim, up to the shared feature map."""
        x = x.unsqueeze(1)
        x = get_vector_feature(x)
        x = torch.permute(x, (0, 3, 2, 1))
        x = torch.reshape(x, (-1, x.size(2), x.size(3)))
        x = self.ln_conv(x)
        x = torch.reshape(x, (-1, 3, x.size(1), x.size(2)))
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.input_block_bn(x)
        b, c, h, w = x.shape
        x = self.liebracket(x)
        x = self.ln_pool(x)
        x = torch.reshape(x, (b, c, h, -1))
        x = self.residual_groups1(x)
        x = self.ln_pool(x)
        x = self.residual_groups4(x)
        x = self.ln_pool(x)
        return x

    def forward(self, x):
        feat = self._features(x)
        mean = self.output_block1(feat)   # ln189 mean head (loaded, frozen) -> byte-identical velocity

        if self.cov_head == 'diag':
            diag_logstd = self.output_block_diag(feat)
            diag_logstd = torch.clamp(diag_logstd, min=self.log_sigma_min, max=self.log_sigma_max)
            return mean, diag_logstd

        eq1 = self.spec_V1(feat); eq2 = self.spec_V2(feat); eq3 = self.spec_V3(feat)
        A = torch.stack([eq1, eq2, eq3], dim=-1)
        V = orthonormalize_via_svd(A)
        n1 = self.spec_e1(feat).norm(dim=-1).clamp(min=1e-6)
        n2 = self.spec_e2(feat).norm(dim=-1).clamp(min=1e-6)
        n3 = self.spec_e3(feat).norm(dim=-1).clamp(min=1e-6)
        inv_in = torch.stack([n1, n2, n3], dim=1).unsqueeze(-1)
        B_ = inv_in.shape[0]
        inv_out = self.inv_head_shared(inv_in.reshape(B_ * 3, 1)).reshape(B_, 3)
        inv_output = torch.clamp(inv_out, min=self.log_sigma_min, max=self.log_sigma_max)
        return mean, V, inv_output
