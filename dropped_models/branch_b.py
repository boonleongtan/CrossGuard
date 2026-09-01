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

Branch B: frozen OpenAI CLIP ViT-L/14 plus a linear detector head.

The encoder is deliberately frozen and kept in eval mode. Training still sees
the live distorted views produced by :mod:`aigid.data`; only the detector head
updates. This is both the mandatory frozen-CLIP baseline and the optional
second logit in the CrossGuard stacker.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BranchB(nn.Module):
    """Frozen CLIP image encoder with one trainable binary head.

    ``encoder`` and ``embed_dim`` are injectable so unit tests can exercise
    the contract without downloading the ~300M-parameter ViT-L checkpoint.
    Production construction uses ``open_clip_torch``.
    """

    def __init__(self, model_name: str = "ViT-L-14-quickgelu",
                 pretrained: str = "openai",
                 normalize: bool = True, encoder: nn.Module | None = None,
                 embed_dim: int | None = None):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = bool(normalize)

        if encoder is None:
            import open_clip

            clip = open_clip.create_model(model_name, pretrained=pretrained)
            encoder = clip.visual
            embed_dim = embed_dim or _visual_output_dim(encoder)
        elif embed_dim is None:
            embed_dim = _visual_output_dim(encoder)

        self.encoder = encoder
        self.embed_dim = int(embed_dim)
        self.head = nn.Linear(self.embed_dim, 1)
        self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
        return self

    def train(self, mode: bool = True):
        # ``super().train`` would enable stochastic layers in the frozen CLIP
        # tower. Put just the detector head in train mode and keep CLIP stable.
        super().train(mode)
        self.encoder.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # No autograd graph through the frozen 304M encoder; this is the VRAM
        # property that makes Branch B practical on a 16-24 GB cloud GPU.
        with torch.no_grad():
            feat = self.encoder(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim == 3:                  # token sequence [B, N, C]
            feat = feat.mean(dim=1)
        elif feat.ndim == 4:                # feature map [B, C, H, W]
            feat = feat.mean(dim=(-2, -1))
        if self.normalize:
            feat = F.normalize(feat, dim=-1)
        return feat

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        logit = self.head(feat).squeeze(-1)
        return (logit, feat) if return_features else logit

    def trainable_state_dict(self) -> dict:
        """Small checkpoint payload; the public OpenAI encoder is reloaded."""
        return {f"head.{key}": value for key, value in self.head.state_dict().items()}

    def load_trainable_state_dict(self, state: dict) -> None:
        head = {key.removeprefix("head."): value for key, value in state.items()
                if key.startswith("head.")}
        if set(head) != {"weight", "bias"}:
            raise ValueError("Branch B checkpoint must contain head.weight and head.bias")
        self.head.load_state_dict(head)

    def param_report(self) -> dict:
        encoder = sum(param.numel() for param in self.encoder.parameters())
        head = sum(param.numel() for param in self.head.parameters())
        return {"encoder": encoder, "head": head, "total": encoder + head,
                "trainable": sum(param.numel() for param in self.parameters()
                                 if param.requires_grad)}

    def config(self) -> dict:
        return {"branch": "b", "model_name": self.model_name,
                "pretrained": self.pretrained, "embed_dim": self.embed_dim,
                "normalize": self.normalize, "img_size": 224,
                "mean": CLIP_MEAN, "std": CLIP_STD}


def _visual_output_dim(encoder: nn.Module) -> int:
    for attr in ("output_dim", "embed_dim", "num_features"):
        value = getattr(encoder, attr, None)
        if isinstance(value, int):
            return value
    proj = getattr(encoder, "proj", None)
    if isinstance(proj, torch.Tensor) and proj.ndim == 2:
        return int(proj.shape[-1])
    raise ValueError("cannot infer CLIP visual output dimension; pass embed_dim")
