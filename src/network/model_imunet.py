"""IMUNet — faithful reimplementation of the efficient depthwise-separable 1D-CNN inertial
regression backbone (Zeinali, Ehsani, Ghassemian, "IMUNet: Efficient Regression Architecture
for Inertial IMU Navigation and Positioning", 2023). MobileNet-style depthwise-separable
convolutions for edge-efficient IMU regression.

No official implementation exists in this workspace, so this is a from-scratch reimplementation.
Adapted to the TLIO-style interface: input [B, in_dim, N] -> (mean[B,3], logstd[B,3]), so it is a
drop-in NN-only baseline that trains/evaluates through the same pipeline as `resnet` (cov_type='diag').
"""
import torch
import torch.nn as nn


class SepConv1d(nn.Module):
    """Depthwise-separable 1D convolution: depthwise (per-channel) 3-kernel conv + pointwise 1x1."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x


class SepBlock(nn.Module):
    """Residual depthwise-separable block (two SepConv1d); identity/projection shortcut."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = SepConv1d(in_ch, out_ch, stride=stride)
        self.conv2 = SepConv1d(out_ch, out_ch, stride=1)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.conv2(self.conv1(x))
        return out + idt


class IMUNet(nn.Module):
    """IMUNet backbone + TLIO-style dual regression head (mean, logstd)."""
    def __init__(self, in_dim=6, out_dim=3, base=64, widths=(64, 128, 256, 512), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, base, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages = []
        c = base
        for w, n in zip(widths, blocks):
            stages.append(SepBlock(c, w, stride=1 if w == base else 2))
            for _ in range(1, n):
                stages.append(SepBlock(w, w, stride=1))
            c = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)

        def head():
            return nn.Sequential(
                nn.Linear(c, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(512, out_dim),
            )
        self.mean_head = head()
        self.logstd_head = head()
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.mean_head(x), self.logstd_head(x)  # (mean, logstd), like ResNet1D
