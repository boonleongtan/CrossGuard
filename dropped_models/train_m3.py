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

Train CrossGuard M3 branches B (frozen CLIP) and C (SRM CNN).

Examples:

    python -m dropped_models.train_m3 --branch b --manifest data/manifest.parquet \
        --image-root data/full --out runs/branch_b --epochs 2

    python -m dropped_models.train_m3 --branch c --manifest data/manifest.parquet \
        --image-root data/full --out runs/branch_c --epochs 2

Both branches consume the existing shared manifest and live distortion
pipeline. Model selection uses worst-cell AUROC across the rules' 14 cells.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dropped_models.branch_b import BranchB, CLIP_MEAN, CLIP_STD
from dropped_models.branch_c import BranchC, RAW_MEAN, RAW_STD
from aigid.data import BranchADataset
from aigid.distort import GRID_CELL_NAMES


class BudgetStop(RuntimeError):
    """Raised when the wall-clock cost guard has expired."""


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def branch_defaults(branch: str) -> dict:
    if branch == "b":
        return {"img_size": 224, "mean": CLIP_MEAN, "std": CLIP_STD,
                "lr": 1e-3}
    if branch == "c":
        return {"img_size": 448, "mean": RAW_MEAN, "std": RAW_STD,
                "lr": 1e-4}
    raise ValueError(f"unknown branch {branch!r}")


def build_model(branch: str, config: dict | None = None):
    config = config or {}
    if branch == "b":
        return BranchB(model_name=config.get("model_name", "ViT-L-14-quickgelu"),
                       pretrained=config.get("pretrained", "openai"),
                       normalize=config.get("normalize", True))
    if branch == "c":
        return BranchC(channels=tuple(config.get("channels", (64, 128, 256, 512))),
                       depths=tuple(config.get("depths", (2, 2, 3, 3))),
                       dropout=float(config.get("dropout", 0.1)))
    raise ValueError(f"unknown branch {branch!r}")


def _cap_rows(dataset: BranchADataset, cap: int, seed: int) -> None:
    """Apply a deterministic, class-balanced smoke/evaluation cap."""
    if cap <= 0 or len(dataset.rows) <= cap:
        return
    per_class = max(1, cap // 2)
    parts = []
    for label, frame in dataset.rows.groupby("label", sort=True):
        parts.append(frame.sample(n=min(per_class, len(frame)), random_state=seed))
    rows = (dataset.rows.iloc[0:0] if not parts else
            pd.concat(parts, ignore_index=True))
    dataset.rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_dataset(branch: str, split: str, manifest: str | None,
                 image_root: str | None, train: bool, img_size: int,
                 grid_cell: str | None = None, cap: int = 0, seed: int = 0):
    defaults = branch_defaults(branch)
    dataset = BranchADataset(
        split, size=img_size, train=train, distort=train and grid_cell is None,
        grid_cell=grid_cell, manifest_path=manifest, image_root=image_root,
        mean=defaults["mean"], std=defaults["std"], seed=seed,
    )
    _cap_rows(dataset, cap, seed)
    return dataset


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool,
                device: torch.device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      persistent_workers=workers > 0)


@torch.no_grad()
def score_loader(model, loader, device: torch.device,
                 deadline: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        if deadline is not None and time.time() >= deadline:
            raise BudgetStop("M3 cost guard expired while scoring")
        x = batch["clean"].to(device, non_blocking=True)
        logit = model(x)
        scores.append(logit.float().cpu().numpy())
        labels.append(batch["label"].numpy())
    if not scores:
        return np.empty(0), np.empty(0)
    return np.concatenate(scores), np.concatenate(labels)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return (float("nan") if len(np.unique(labels)) < 2 else
            float(roc_auc_score(labels, scores)))


def evaluate_grid(model, branch: str, split: str, manifest: str | None,
                  image_root: str | None, img_size: int, batch_size: int,
                  workers: int, device: torch.device, cap: int, seed: int,
                  deadline: float | None = None) -> dict:
    metrics = {}
    if deadline is not None and time.time() >= deadline:
        raise BudgetStop("M3 cost guard expired before validation")
    clean = make_dataset(branch, split, manifest, image_root, False, img_size,
                         cap=cap, seed=seed)
    scores, labels = score_loader(
        model, make_loader(clean, batch_size, workers, False, device), device,
        deadline)
    metrics["clean"] = safe_auroc(labels, scores)

    for cell in GRID_CELL_NAMES:
        if deadline is not None and time.time() >= deadline:
            raise BudgetStop("M3 cost guard expired during validation grid")
        dataset = make_dataset(branch, split, manifest, image_root, False, img_size,
                               grid_cell=cell, cap=cap, seed=seed)
        scores, labels = score_loader(
            model, make_loader(dataset, batch_size, workers, False, device),
            device, deadline)
        metrics[cell] = safe_auroc(labels, scores)
    finite = [metrics[cell] for cell in GRID_CELL_NAMES
              if np.isfinite(metrics[cell])]
    metrics["worst_cell"] = min(finite) if finite else float("nan")
    metrics["macro_robust"] = float(np.mean(finite)) if finite else float("nan")
    return metrics


def model_checkpoint_state(model, branch: str) -> dict:
    return (model.trainable_state_dict() if branch == "b" else
            model.state_dict())


def load_model_checkpoint(model, branch: str, state: dict) -> None:
    if branch == "b":
        model.load_trainable_state_dict(state)
    else:
        model.load_state_dict(state)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best: float,
                    args, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "branch": args.branch,
        "model": model_checkpoint_state(model, args.branch),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_worst_cell": float(best),
        "metrics": metrics,
        "config": model.config(),
        "train_args": {key: value for key, value in vars(args).items()
                       if isinstance(value, (str, int, float, bool, type(None)))},
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train(args) -> dict:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    defaults = branch_defaults(args.branch)
    img_size = args.img_size or defaults["img_size"]
    lr = args.lr or defaults["lr"]
    max_train_seconds = float(getattr(args, "max_train_seconds", 0) or 0)
    deadline = time.time() + max_train_seconds if max_train_seconds > 0 else None

    model = build_model(args.branch).to(device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)
    start_epoch, best = 0, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("branch") != args.branch:
            raise ValueError(f"resume checkpoint is branch {checkpoint.get('branch')!r}")
        load_model_checkpoint(model, args.branch, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_worst_cell", best))

    train_data = make_dataset(args.branch, "train", args.manifest, args.image_root,
                              True, img_size, cap=args.train_cap, seed=args.seed)
    loader = make_loader(train_data, args.batch_size, args.workers, True, device)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    out = Path(args.out)
    history = []

    print(json.dumps({"device": str(device), "branch": args.branch,
                      "parameters": model.param_report(), "train_rows": len(train_data),
                      "img_size": img_size, "lr": lr}, indent=2))

    stopped_by_budget = False
    for epoch in range(start_epoch, args.epochs):
        if deadline is not None and time.time() >= deadline:
            stopped_by_budget = True
            break
        model.train()
        train_data.set_epoch(epoch)
        running_loss, seen, started = 0.0, 0, time.time()
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            clean = batch["clean"].to(device, non_blocking=True)
            distorted = batch["distorted"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            views = torch.cat((clean, distorted), dim=0)
            targets = torch.cat((labels, labels), dim=0)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(views)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            scaler.step(optimizer)
            scaler.update()

            batch_n = len(labels)
            running_loss += float(loss) * batch_n
            seen += batch_n
            if (step + 1) % args.log_every == 0:
                print(f"epoch={epoch} step={step + 1} loss={running_loss / seen:.4f} "
                      f"images/s={seen / max(time.time() - started, 1e-6):.1f}")
            if args.max_steps and step + 1 >= args.max_steps:
                break
            if deadline is not None and time.time() >= deadline:
                metrics = {"budget_stop": True, "epoch": epoch,
                           "step_in_epoch": step + 1,
                           "max_train_seconds": max_train_seconds}
                save_checkpoint(out / "budget_stop.pt", model, optimizer, epoch,
                                best, args, metrics)
                stopped_by_budget = True
                break

        if stopped_by_budget:
            metrics = {"budget_stop": True,
                       "max_train_seconds": max_train_seconds}
        elif args.no_eval_cells:
            metrics = {}
        else:
            try:
                metrics = evaluate_grid(
                    model, args.branch, "val", args.manifest,
                    args.image_root, img_size,
                    args.eval_batch_size or args.batch_size, args.workers,
                    device, args.cell_cap, args.seed, deadline)
            except BudgetStop as exc:
                metrics = {"budget_stop": True, "eval_stop": True,
                           "reason": str(exc),
                           "max_train_seconds": max_train_seconds}
                save_checkpoint(out / "budget_stop.pt", model, optimizer,
                                epoch, best, args, metrics)
                stopped_by_budget = True
        metric = metrics.get("worst_cell", -running_loss / max(seen, 1))
        record = {"epoch": epoch, "loss": running_loss / max(seen, 1),
                  "metrics": metrics}
        history.append(record)
        print(json.dumps(record, indent=2))

        if stopped_by_budget:
            (out / "history.json").write_text(json.dumps(history, indent=2,
                                                           allow_nan=True))
            break

        if np.isfinite(metric) and metric > best:
            best = metric
            save_checkpoint(out / "best.pt", model, optimizer, epoch, best,
                            args, metrics)
        save_checkpoint(out / "last.pt", model, optimizer, epoch, best,
                        args, metrics)
        (out / "history.json").write_text(json.dumps(history, indent=2,
                                                       allow_nan=True))

    return {"branch": args.branch, "best_worst_cell": best,
            "out": str(out), "epochs_completed": len(history),
            "stopped_by_budget": stopped_by_budget}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("b", "c"), required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--out", default="runs/m3")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=0,
                        help="0 selects 224 for B and 448 for C")
    parser.add_argument("--lr", type=float, default=0.0,
                        help="0 selects 1e-3 for B and 1e-4 for C")
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-cap", type=int, default=0)
    parser.add_argument("--cell-cap", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-train-seconds", type=int, default=0,
                        help="hard wall-clock stop for cost-capped runs")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-eval-cells", action="store_true")
    return parser


def main() -> int:
    result = train(make_parser().parse_args())
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
