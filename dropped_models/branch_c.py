"""DROPPED 30 Aug 2026 -- not part of the CrossGuard submission.

Branch B and Branch C were cut for time. With the freeze at 31 Aug midday and
Branch A (DINOv2 ViT-L/14) already at 0.9909 worst-cell robust AUROC on val, the
remaining hours bought either two more half-trained branches feeding an
unvalidated blend, or one model calibrated and evaluated properly. We took the
second.

The code here is complete and its tests pass; what it lacks is trained weights.
Neither branch finished training, so neither produced the validation bundle the
fusion gate needed, and the gate was never run on real data. We are not claiming
fusion would not have helped -- only that we did not measure it.

Branch C: fixed SRM residual filters followed by a compact residual CNN.

The fixed high-pass front end suppresses semantic colour content and exposes
the local noise/resampling traces that complement the global ViT branches.
The trainable network is roughly 20M parameters with the default stage depths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


RAW_MEAN = (0.0, 0.0, 0.0)
RAW_STD = (1.0, 1.0, 1.0)


def _srm_kernels() -> torch.Tensor:
    # Three standard residual patterns, zero-padded to one 5x5 bank. Each is
    # applied independently to R/G/B via grouped convolution (9 outputs).
    k1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0
    k2 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0
    k3 = torch.tensor([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=torch.float32) / 12.0
    return torch.stack((k1, k2, k3))


class SRMHighPass(nn.Module):
    def __init__(self, clip: float = 3.0):
        super().__init__()
        # grouped conv expects [out_channels, in_channels/groups, H, W].
        bank = _srm_kernels().unsqueeze(1).repeat(3, 1, 1, 1)
        self.register_buffer("weight", bank, persistent=True)
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.conv2d(x, self.weight, padding=2, groups=3)
        return residual.clamp(-self.clip, self.clip)


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = _norm(out_channels)
        self.skip = (nn.Identity() if stride == 1 and in_channels == out_channels else
                     nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride,
                                             bias=False), _norm(out_channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class BranchC(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), depths=(2, 2, 3, 3),
                 dropout: float = 0.1):
        super().__init__()
        if len(channels) != len(depths) or not channels:
            raise ValueError("channels and depths must have equal non-zero length")
        self.channels = tuple(int(value) for value in channels)
        self.depths = tuple(int(value) for value in depths)
        self.dropout = float(dropout)

        self.srm = SRMHighPass()
        self.stem = nn.Sequential(
            nn.Conv2d(9, self.channels[0], 5, 2, 2, bias=False),
            _norm(self.channels[0]), nn.GELU(),
        )
        stages = []
        current = self.channels[0]
        for stage_index, (width, depth) in enumerate(zip(self.channels, self.depths)):
            blocks = []
            for block_index in range(depth):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(ResidualBlock(current, width, stride))
                current = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(current, 1))
        self.embed_dim = current

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.srm(x)
        x = self.stem(x)
        x = self.stages(x)
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        logit = self.head(feat).squeeze(-1)
        return (logit, feat) if return_features else logit

    def param_report(self) -> dict:
        rows = {
            "srm_fixed": self.srm.weight.numel(),
            "cnn": sum(param.numel() for name, param in self.named_parameters()
                       if not name.startswith("head.")),
            "head": sum(param.numel() for param in self.head.parameters()),
        }
        rows["total"] = sum(param.numel() for param in self.parameters())
        rows["trainable"] = sum(param.numel() for param in self.parameters()
                                 if param.requires_grad)
        return rows

    def config(self) -> dict:
        return {"branch": "c", "channels": self.channels, "depths": self.depths,
                "dropout": self.dropout, "embed_dim": self.embed_dim,
                "img_size": 448, "mean": RAW_MEAN, "std": RAW_STD}
