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

Score one branch checkpoint into a logit bundle.

This is the ONLY step here that loads a model, and the only one that may read
test -- and test only for the single final evaluation pass,
never for a decision. Every ship/no-ship choice is made by `dropped_models/stacker.py`,
which reads exported bundles, imports no torch, and refuses any bundle not
stamped `split=val`.

    score --checkpoint runs/m3/b/best.pt --split val --output bundles/b_val.npz

Branch A must be exported in the same format by the M2/M4 evaluation path; the
format itself lives in `dropped_models/bundles.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dropped_models.bundles import CELLS, save_bundle
from dropped_models.train_m3 import (branch_defaults, build_model, load_model_checkpoint,
                            make_dataset, make_loader, resolve_device, score_loader)


def score_checkpoint(checkpoint_path: str, split: str, manifest: str | None,
                     image_root: str | None, output: str, batch_size: int,
                     workers: int, device_name: str, cap: int, seed: int) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    branch = checkpoint.get("branch")
    if branch not in {"b", "c"}:
        raise ValueError(f"unsupported M3 checkpoint branch {branch!r}")
    model = build_model(branch, checkpoint.get("config", {}))
    load_model_checkpoint(model, branch, checkpoint["model"])
    device = resolve_device(device_name)
    model.to(device).eval()
    img_size = int(checkpoint.get("config", {}).get(
        "img_size", branch_defaults(branch)["img_size"]))

    logits, labels, ids = [], None, None
    for cell in CELLS:
        dataset = make_dataset(
            branch, split, manifest, image_root, False, img_size,
            grid_cell=None if cell == "clean" else cell, cap=cap, seed=seed)
        loader = make_loader(dataset, batch_size, workers, False, device)
        cell_logits, cell_labels = score_loader(model, loader, device)
        cell_ids = dataset.rows["path"].astype(str).to_numpy()
        if ids is None:
            ids, labels = cell_ids, cell_labels
        elif not np.array_equal(ids, cell_ids) or not np.array_equal(labels, cell_labels):
            raise RuntimeError(f"row order changed while scoring {cell}")
        logits.append(cell_logits)

    matrix = np.stack(logits, axis=1)
    destination = Path(output)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(destination, ids, labels, CELLS, matrix, branch, split)
    return {"output": str(destination), "branch": branch, "split": split,
            "rows": len(ids), "cells": len(CELLS)}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="export one checkpoint's logits")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--split", choices=("val", "test"), required=True,
                       help="test is permitted here -- this exports logits and "
                            "decides nothing. Fusion (dropped_models/stacker.py) "
                            "accepts val bundles only")
    score.add_argument("--manifest", default=None)
    score.add_argument("--image-root", default=None)
    score.add_argument("--output", required=True)
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--workers", type=int, default=4)
    score.add_argument("--device", default="auto")
    score.add_argument("--cap", type=int, default=0)
    score.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    result = score_checkpoint(args.checkpoint, args.split, args.manifest,
                              args.image_root, args.output, args.batch_size,
                              args.workers, args.device, args.cap, args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
