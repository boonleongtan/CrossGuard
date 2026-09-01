"""Shared constants for CrossGuard inference and training."""
from __future__ import annotations

# ImageNet stats for the shipped DINOv2 timm checkpoint.
# EVA02-448 would need CLIP-style stats; the submitted path is DINOv2.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
