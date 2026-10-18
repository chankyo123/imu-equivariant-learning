import torch
import torch.nn as nn

from models.vn_layers import VNBatchNorm, VNLeakyReLU
from models.lie_neurons_layers import LNLinear, LNLieBracket
from models.utils.vn_dgcnn_util import get_vector_feature


def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def apply_conv1d_vector(x, conv):
    """
    x: [B, C, 3, T]
    conv: nn.Conv1d operating on [B*3, C, T]
    """
    b, c, k, t = x.shape
    assert k == 3
    y = x.permute(0, 2, 1, 3).reshape(b * 3, c, t)
    y = conv(y)
    t2 = y.size(-1)
    c2 = y.size(1)
    y = y.reshape(b, 3, c2, t2).permute(0, 2, 1, 3).contiguous()
    return y


class LocalMeanPool1d(nn.Module):
    """
    Device-safe local mean pooling over time for vector features.
    x: [B, C, 3, T] -> [B, C, 3, ceil(T/kernel)]
    """
    def __init__(self, kernel=2):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        chunks = []
        for i in range(0, x.size(-1), self.kernel):
            chunks.append(x[..., i:i + self.kernel].mean(dim=-1, keepdim=True))
        return torch.cat(chunks, dim=-1)


class EqFeatureDropout(nn.Module):
    """
    Equivariant-safe dropout:
    one scalar mask per feature channel, shared across the 3-vector components
    (and shared across time if x is 4D).
    """
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p <= 0.0 or (not self.training):
            return x
        keep = 1.0 - self.p
        shape = [x.size(0), x.size(1)] + [1] * (x.ndim - 2)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class DropPath(nn.Module):
    """
    Residual-branch stochastic depth.
    One scalar mask per sample, shared across all channels/vector dims/time.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob <= 0.0 or (not self.training):
            return x
        keep = 1.0 - self.drop_prob
        shape = [x.size(0)] + [1] * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep)
        return x * random_tensor / keep


class EqChannelGate(nn.Module):
    """
    Channel gate from invariant norms.
    Gate is scalar per channel, so SO(3) equivariance is preserved.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, 3, T]
        s = torch.norm(x, dim=2).mean(dim=-1)   # [B, C], invariant
        g = self.mlp(s).unsqueeze(2).unsqueeze(3)
        return x * g


class EqTemporalAttnPool(nn.Module):
    """
    Invariant attention scores over time, equivariant weighted sum of vectors.
    """
    def __init__(self, channels, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [B, C, 3, T]
        phi = torch.norm(x, dim=2).transpose(1, 2)   # [B, T, C], invariant
        a = self.score(phi).squeeze(-1)              # [B, T]
        a = torch.softmax(a, dim=-1)
        z = (x * a[:, None, None, :]).sum(dim=-1)    # [B, C, 3], equivariant
        return z


class LN_PreActBracketBlock1D(nn.Module):
    """
    Pre-activation equivariant residual block.
    Uses:
      BN -> VNReLU -> Conv
      BN -> VNReLU -> LieBracket -> EqFeatureDropout -> Conv
      + identity skip
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 feature_drop=0.0, drop_path=0.0):
        super().__init__()
        self.bn1 = VNBatchNorm(in_planes, dim=3)
        self.act1 = VNLeakyReLU(in_planes, negative_slope=0.0)
        self.conv1 = conv3x1(in_planes, planes, stride=stride)

        self.bn2 = VNBatchNorm(planes, dim=3)
        self.act2 = VNLeakyReLU(planes, negative_slope=0.0)
        self.bracket = LNLieBracket(planes, algebra_type='so3', share_nonlinearity=False)
        self.feature_drop = EqFeatureDropout(feature_drop)
        self.conv2 = conv3x1(planes, planes, stride=1)

        self.downsample = downsample
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.act1(out)

        if self.downsample is not None:
            identity = apply_conv1d_vector(out, self.downsample[0])
            identity = self.downsample[1](identity)

        out = apply_conv1d_vector(out, self.conv1)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.bracket(out)
        out = self.feature_drop(out)
        out = apply_conv1d_vector(out, self.conv2)
        out = self.drop_path(out)

        return out + identity


class EqVectorHead(nn.Module):
    """
    Equivariant vector head.
    Used for BOTH mean and covariance, exactly as requested.
    Output: [B, 3]
    """
    def __init__(self, in_channels, hidden_channels=384,
                 attn_hidden=128, feature_drop=0.10):
        super().__init__()
        self.pool = EqTemporalAttnPool(in_channels, hidden=attn_hidden)

        self.lin1 = LNLinear(in_channels, hidden_channels)
        self.bn1 = VNBatchNorm(hidden_channels, dim=3)
        self.act1 = VNLeakyReLU(hidden_channels, negative_slope=0.0)

        self.lin2 = LNLinear(hidden_channels, hidden_channels)
        self.bn2 = VNBatchNorm(hidden_channels, dim=3)
        self.bracket = LNLieBracket(hidden_channels, algebra_type='so3', share_nonlinearity=False)
        self.act2 = VNLeakyReLU(hidden_channels, negative_slope=0.0)

        self.feature_drop = EqFeatureDropout(feature_drop)
        self.lin3 = LNLinear(hidden_channels, 1)

    def forward(self, x):
        # x: [B, C, 3, T]
        z = self.pool(x)        # [B, C, 3]
        z = self.lin1(z)
        z = self.bn1(z)
        z = self.act1(z)

        z = self.lin2(z)
        z = self.bn2(z)
        z = self.bracket(z)
        z = self.act2(z)
        z = self.feature_drop(z)

        z = self.lin3(z)        # [B, 1, 3]
        return z.squeeze(1)     # [B, 3]


class SO3EquivariantReluBracketLayers_peract(nn.Module):
    """
    Wide pre-activation SO(3)-equivariant IMU model.

    Input:
        x: [B, 200, 6] or [B, 6, 200]

    Output:
        mean: [B, 3]
        covariance: [B, 3]   # same equivariant head structure as requested
    """
    def __init__(
        self,
        block_type,                 # kept for API compatibility
        in_dim,
        out_dim,
        group_sizes,
        inter_dim=None,             # kept for API compatibility
        zero_init_residual=True,
        stem_width=24,
        widths=(48, 96, 192, 384),
        head_hidden=384,
        max_drop_path=0.10,
        feature_drop=0.10,
        gate_reduction=4,
    ):
        super().__init__()

        if in_dim != 6:
            raise ValueError(
                f"This version is written for 6D IMU windows (acc+gyro). Got in_dim={in_dim}."
            )
        if out_dim != 3:
            raise ValueError(
                f"This version assumes velocity/log-cov outputs of size 3. Got out_dim={out_dim}."
            )

        # Force the new block internally
        self.block = LN_PreActBracketBlock1D
        self.group_sizes = tuple(group_sizes)
        self.widths = tuple(widths)
        self.inplanes = widths[0]

        # ---- split stem: acc and gyro processed separately, then fused ----
        self.acc_stem = nn.Conv1d(1, stem_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.gyr_stem = nn.Conv1d(1, stem_width, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem_bn = VNBatchNorm(2 * stem_width, dim=4)
        self.stem_act = VNLeakyReLU(2 * stem_width, negative_slope=0.0)
        self.stem_fuse = nn.Conv1d(2 * stem_width, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_fuse_bn = VNBatchNorm(widths[0], dim=4)
        self.stem_bracket = LNLieBracket(widths[0], algebra_type='so3', share_nonlinearity=False)
        self.stem_gate = EqChannelGate(widths[0], reduction=gate_reduction)
        self.local_pool = LocalMeanPool1d(kernel=2)

        # stochastic-depth schedule across all residual blocks
        total_blocks = sum(self.group_sizes)
        dp_rates = torch.linspace(0.0, max_drop_path, total_blocks).tolist()
        ptr = 0

        self.stage1 = self._make_residual_group1d(
            self.block, widths[0], self.group_sizes[0], stride=1,
            dp_rates=dp_rates[ptr:ptr + self.group_sizes[0]], feature_drop=feature_drop
        )
        ptr += self.group_sizes[0]
        self.gate1 = EqChannelGate(widths[0], reduction=gate_reduction)

        self.stage2 = self._make_residual_group1d(
            self.block, widths[1], self.group_sizes[1], stride=2,
            dp_rates=dp_rates[ptr:ptr + self.group_sizes[1]], feature_drop=feature_drop
        )
        ptr += self.group_sizes[1]
        self.gate2 = EqChannelGate(widths[1], reduction=gate_reduction)

        self.stage3 = self._make_residual_group1d(
            self.block, widths[2], self.group_sizes[2], stride=2,
            dp_rates=dp_rates[ptr:ptr + self.group_sizes[2]], feature_drop=feature_drop
        )
        ptr += self.group_sizes[2]
        self.gate3 = EqChannelGate(widths[2], reduction=gate_reduction)

        self.stage4 = self._make_residual_group1d(
            self.block, widths[3], self.group_sizes[3], stride=2,
            dp_rates=dp_rates[ptr:ptr + self.group_sizes[3]], feature_drop=feature_drop
        )
        self.gate4 = EqChannelGate(widths[3], reduction=gate_reduction)

        self.final_bn = VNBatchNorm(widths[3], dim=4)
        self.final_act = VNLeakyReLU(widths[3], negative_slope=0.0)

        self.mean_head = EqVectorHead(
            in_channels=widths[3],
            hidden_channels=head_hidden,
            attn_hidden=max(128, widths[3] // 2),
            feature_drop=feature_drop,
        )
        self.cov_head = EqVectorHead(
            in_channels=widths[3],
            hidden_channels=head_hidden,
            attn_hidden=max(128, widths[3] // 2),
            feature_drop=feature_drop,
        )

        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block, planes, group_size, stride, dp_rates, feature_drop):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                VNBatchNorm(planes * block.expansion, dim=4),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                feature_drop=feature_drop,
                drop_path=dp_rates[0],
            )
        )
        self.inplanes = planes * block.expansion

        for i in range(1, group_size):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    feature_drop=feature_drop,
                    drop_path=dp_rates[i],
                )
            )
        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual=True):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, VNBatchNorm):
                nn.init.constant_(m.bn.weight, 1.0)
                nn.init.constant_(m.bn.bias, 0.0)
            elif isinstance(m, nn.Linear):
                if m.bias is None:
                    nn.init.normal_(m.weight, 0.0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    nn.init.constant_(m.bias, 0.0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LN_PreActBracketBlock1D):
                    nn.init.constant_(m.bn2.bn.weight, 0.0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _prepare_input(self, x):
        # Accept [B, 200, 6] or [B, 6, 200]
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [B, 200, 6] or [B, 6, 200], got {tuple(x.shape)}"
            )

        if x.size(1) == 200 and x.size(2) == 6:
            x = x.transpose(1, 2).contiguous()   # -> [B, 6, 200]
        elif x.size(1) == 6:
            x = x.contiguous()
        else:
            raise ValueError(f"Unrecognized input shape {tuple(x.shape)}")

        # get_vector_feature expects [B, 1, 6, T]
        xv = get_vector_feature(x.unsqueeze(1))          # [B, T, 2, 3]
        xv = xv.permute(0, 2, 3, 1).contiguous()        # [B, 2, 3, T]
        return xv

    def forward(self, x):
        # x -> [B, 2, 3, 200]
        x = self._prepare_input(x)

        # split stem
        x_acc = x[:, 0:1, :, :]                         # [B, 1, 3, T]
        x_gyr = x[:, 1:2, :, :]                         # [B, 1, 3, T]

        x_acc = apply_conv1d_vector(x_acc, self.acc_stem)   # [B, stem, 3, 100]
        x_gyr = apply_conv1d_vector(x_gyr, self.gyr_stem)   # [B, stem, 3, 100]

        x = torch.cat([x_acc, x_gyr], dim=1)            # [B, 2*stem, 3, 100]
        x = self.stem_bn(x)
        x = self.stem_act(x)
        x = apply_conv1d_vector(x, self.stem_fuse)      # [B, C1, 3, 100]
        x = self.stem_fuse_bn(x)
        x = self.stem_bracket(x)
        x = self.stem_gate(x)
        x = self.local_pool(x)                          # [B, C1, 3, 50]

        x = self.stage1(x)
        x = self.gate1(x)

        x = self.stage2(x)                              # 50 -> 25
        x = self.gate2(x)

        x = self.stage3(x)                              # 25 -> 13
        x = self.gate3(x)

        x = self.stage4(x)                              # 13 -> 7
        x = self.gate4(x)

        x = self.final_bn(x)
        x = self.final_act(x)

        mean = self.mean_head(x)                        # [B, 3]
        covariance = self.cov_head(x)                  # [B, 3]

        return mean, covariance