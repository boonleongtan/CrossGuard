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

The exported-logit bundle format: one file, read by scoring and by fusion.

A bundle is a compressed ``.npz`` holding one branch's logits for every image
in one split, across clean + all 14 transform cells:

    ids     (N,)              image path, the join key across branches
    labels  (N,)  int8        0 real, 1 fake
    cells   (C,)              cell names, "clean" first
    logits  (N, C)  float     uncalibrated logit per (image, cell)
    branch  scalar            which branch produced this
    split   scalar            which split it was scored on

`split` is not decoration. The stacker gate is a ship/no-ship DECISION and must
never read test, so `load_bundle(..., require_split="val")` lets that be
enforced by inspecting the FILE rather than by trusting a caller to pass the
right path. An unstamped bundle is refused rather than assumed safe.

This module deliberately imports no torch and no sklearn: it is the neutral
contract between `eval_m3.score` (which needs models and may read test) and
`stacker` (which needs neither and must not).
"""
from __future__ import annotations

import numpy as np

from aigid.distort import GRID_CELL_NAMES

CELLS = ("clean", *GRID_CELL_NAMES)


def load_bundle(path: str, require_split: str | None = None) -> dict:
    """Read one bundle, optionally refusing any split but ``require_split``."""
    with np.load(path, allow_pickle=False) as data:
        required = {"ids", "labels", "cells", "logits"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing bundle keys {sorted(missing)}")
        bundle = {key: data[key].copy() for key in required}
        bundle["split"] = str(data["split"]) if "split" in data.files else None

    if bundle["logits"].shape != (len(bundle["ids"]), len(bundle["cells"])):
        raise ValueError(f"{path}: logits shape does not match ids/cells")
    if len(set(bundle["ids"].tolist())) != len(bundle["ids"]):
        raise ValueError(f"{path}: duplicate image ids")

    if require_split is not None:
        if bundle["split"] is None:
            raise ValueError(
                f"{path}: bundle carries no split stamp, so it cannot be shown "
                f"to be {require_split!r}. Re-export it with `eval_m3 score "
                f"--split {require_split}`; an unstamped bundle is refused "
                f"rather than assumed safe.")
        if bundle["split"] != require_split:
            raise ValueError(
                f"{path}: bundle is split {bundle['split']!r}, but this step "
                f"accepts only {require_split!r}. The stacker gate is a "
                f"ship/no-ship DECISION and must not read test: doing so is "
                f"model selection on the final split. Export val bundles with "
                f"`eval_m3 score --split val`.")
    return bundle


def save_bundle(path, ids, labels, cells, logits, branch: str, split: str):
    """Write one bundle. Always stamps `branch` and `split`."""
    np.savez_compressed(
        path, ids=np.asarray(ids), labels=np.asarray(labels).astype(np.int8),
        cells=np.asarray(cells), logits=np.asarray(logits),
        branch=np.asarray(branch), split=np.asarray(split))


def align_bundle(reference: dict, candidate: dict, name: str) -> dict:
    """Reorder `candidate` onto `reference`'s image order.

    Branches are scored in separate runs, so their row orders need not match.
    Aligning by id -- and then asserting the labels agree -- turns a mismatched
    or partial export into a loud failure instead of a silent misalignment that
    would quietly train the stacker on shuffled targets.
    """
    if not np.array_equal(reference["cells"], candidate["cells"]):
        raise ValueError(f"{name}: transform cells differ from baseline")
    positions = {value: index for index, value in enumerate(candidate["ids"].tolist())}
    try:
        order = np.asarray([positions[value] for value in reference["ids"].tolist()])
    except KeyError as error:
        raise ValueError(f"{name}: missing image id {error.args[0]!r}") from error
    if len(positions) != len(reference["ids"]):
        raise ValueError(f"{name}: image-id set differs from baseline")
    aligned = {"ids": candidate["ids"][order],
               "labels": candidate["labels"][order],
               "cells": candidate["cells"],
               "logits": candidate["logits"][order],
               "split": candidate.get("split")}
    if not np.array_equal(reference["labels"], aligned["labels"]):
        raise ValueError(f"{name}: labels differ from baseline")
    return aligned
