"""Branch A — the fine-tuned backbone used by CrossGuard.

DINOv2 ViT-L/14 at 448 px is the shipped configuration. EVA02-L/14 is the
supported alternative. Both use the same head and fine-tuning path. GAP over
final patch tokens produces a single logit.

LP-FT staging (§5):
  * ``freeze_for_lp()``   — train only the linear head on frozen features
  * ``enable_ft(...)``    — add LoRA r=32 (attention + MLP, all blocks) and
                            unfreeze the last 4 transformer blocks at low LR
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Verify the exact hub strings on the training box:  timm.list_models('*dinov2*')
BACKBONES = {
    # Apache-2.0. timm interpolates the pretrained position embeddings to the
    # requested input size.
    "dinov2-l14-448": "vit_large_patch14_dinov2.lvd142m",
    "eva02-l14-448":  "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",  # MIT, native 448^2
}
SUPPORTED_BACKBONE_NAMES = frozenset(BACKBONES.values())


class CorrectionFFN(nn.Module):
    """TeleAI-TeleGuard's detail (§5): the distorted branch's features pass
    through a small residual FFN before the feature-MSE term, not matched raw."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        return x + self.net(x)


class BranchA(nn.Module):
    def __init__(self, backbone: str = "dinov2-l14-448", img_size: int = 448,
                 pool: str = "gap", pretrained: bool = True,
                 weights: str | None = None):
        super().__init__()
        import timm

        if backbone not in BACKBONES and backbone not in SUPPORTED_BACKBONE_NAMES:
            supported = ", ".join(sorted(BACKBONES))
            raise ValueError(
                f"unsupported backbone {backbone!r}; choose one of: {supported}")
        name = BACKBONES.get(backbone, backbone)
        self.backbone_name = name
        # `weights` optionally supplies a local pretrained checkpoint for a
        # supported backbone. The shipped path uses timm's standard pretrained
        # configuration instead.
        overlay = dict(file=weights) if weights else None
        self.backbone = timm.create_model(
            name, pretrained=pretrained or weights is not None, num_classes=0,
            global_pool="", img_size=img_size, pretrained_cfg_overlay=overlay)
        self.embed_dim = self.backbone.num_features
        self.num_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        self.pool = pool

        head_dim = self.embed_dim * (2 if pool == "gap+cls" else 1)
        self.head = nn.Linear(head_dim, 1)
        self.head_dim = head_dim
        self.correction = CorrectionFFN(head_dim)
        self._lora = False

    def set_grad_checkpointing(self, enable: bool = True):
        """Trade compute for VRAM — needed for full-FT on a 24 GB card."""
        m = getattr(self.backbone, "base_model", self.backbone)
        m = getattr(m, "model", m)
        if hasattr(m, "set_grad_checkpointing"):
            m.set_grad_checkpointing(enable)
        return self

    # ── features / forward ─────────────────────────────────────────────────
    def features(self, x):
        f = self.backbone.forward_features(x)
        if f.ndim == 3:                       # [B, N, C] tokens
            patch = f[:, self.num_prefix:].mean(dim=1)
            if self.pool == "gap+cls":
                return torch.cat([patch, f[:, 0]], dim=-1)
            return patch
        return f                             # already pooled

    def forward(self, x, return_features: bool = False):
        feat = self.features(x)
        logit = self.head(feat).squeeze(-1)
        return (logit, feat) if return_features else logit

    # ── LP-FT staging (§5) ─────────────────────────────────────────────────
    def freeze_for_lp(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(True)
        return self

    # Every attention + MLP projection across both supported backbones:
    #   DINOv2 ViT-L    -> fused qkv, proj, mlp fc1/fc2
    #   EVA02-L/14    → split q_proj/k_proj/v_proj, proj, SwiGLU fc1_g/fc1_x/fc2
    # peft matches by suffix and skips names a given backbone doesn't have.
    LORA_TARGETS = ("qkv", "q_proj", "k_proj", "v_proj", "proj",
                    "fc1", "fc2", "fc1_g", "fc1_x")

    def enable_ft(self, lora_r: int = 32, lora_alpha: int = 64,
                  unfreeze_last_n: int = 4, lora_targets=None):
        lora_targets = lora_targets or self.LORA_TARGETS
        from peft import LoraConfig, get_peft_model

        # §4.2 scopes LoRA to "attention + MLP, all blocks". A bare "proj"
        # suffix also matches `patch_embed.proj` — the patch-embedding conv,
        # which is neither — on both backbones, so target by full name inside
        # the transformer blocks only.
        names = [n for n, m in self.backbone.named_modules()
                 if isinstance(m, nn.Linear) and n.startswith("blocks.")
                 and n.rsplit(".", 1)[-1] in set(lora_targets)]
        if not names:
            raise RuntimeError(
                f"no LoRA targets matched in {self.backbone_name}; "
                f"check the module names against LORA_TARGETS")
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                         bias="none", target_modules=names)
        self.backbone = get_peft_model(self.backbone, cfg)
        self._lora = True
        for blk in self._blocks()[-unfreeze_last_n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        return self

    def _blocks(self):
        m = self.backbone
        m = getattr(m, "base_model", m)      # peft LoraModel
        m = getattr(m, "model", m)           # → timm model
        return m.blocks

    def set_lora_scale(self, scale: float):
        """Inference-time LoRA scaling sweep (§5 step 5): {0.5, 0.75, 1.0}."""
        if not self._lora:
            return
        for m in self.backbone.modules():
            if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict):
                if not hasattr(m, "_base_scaling"):
                    m._base_scaling = dict(m.scaling)
                for k, base in m._base_scaling.items():
                    m.scaling[k] = base * scale

    # ── optimiser groups with the §5 LRs ──────────────────────────────────
    def param_groups(self, head_lr: float = 5e-4, lora_lr: float = 1e-4,
                     block_lr: float = 2e-5):
        head, lora, blocks = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith(("head.", "correction.")):
                head.append(p)
            elif "lora_" in n:
                lora.append(p)
            else:
                blocks.append(p)
        groups = [{"params": head, "lr": head_lr, "name": "head"}]
        if lora:
            groups.append({"params": lora, "lr": lora_lr, "name": "lora"})
        if blocks:
            groups.append({"params": blocks, "lr": block_lr, "name": "blocks"})
        return groups

    # ── parameter report — feeds `predict --report-params` (§3.3 / A2) ─────
    def param_report(self) -> dict:
        rows = {name: sum(p.numel() for p in mod.parameters())
                for name, mod in (("backbone", self.backbone),
                                  ("head", self.head),
                                  ("correction", self.correction))}
        rows["total"] = sum(rows.values())
        rows["trainable"] = sum(p.numel() for p in self.parameters()
                                if p.requires_grad)
        return rows

    def config(self) -> dict:
        return {"backbone": self.backbone_name, "embed_dim": self.embed_dim,
                "pool": self.pool, "lora": self._lora}
