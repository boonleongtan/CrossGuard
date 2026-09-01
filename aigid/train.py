"""LP-FT training for Branch A.

This module is the single training implementation. It runs on any
judge-controlled GPU server once the public-data manifest and image root are
available locally.

The default path is full fine-tuning, not LoRA:

    # linear-probe stage: head only, frozen features.
    # 2 epochs on the real build, not 5: LP fits a 2.1M-param head on frozen
    # features, so most of the gain lands in the first pass, and `warmup =
    # steps_per_epoch` below makes 2 the floor that still leaves one epoch of
    # cosine decay.
    python -m aigid.train --stage lp --epochs 2 --out runs/dev_lp

    # fine-tune stage: all blocks at 1e-5, head warm, SWA over the final
    # epochs. This is the DEFAULT path -- --grad-checkpoint is NOT needed
    # on an H100, where full-FT peaks at 34.6 GB of 85 at batch 16 / 448.
    python -m aigid.train --stage ft --resume runs/dev_lp/last.pt \
        --epochs 2 --out runs/dev_full

    # LoRA path: retained for the ablation table, not the ship path
    python -m aigid.train --stage ft --ft-path lora --resume runs/dev_lp/last.pt \
        --epochs 2 --out runs/dev_ft

Small local subsets are useful for validating the *pipeline* -- loss goes down,
checkpoints resume, metrics compute, and the objective wires up. They will
overfit; model quality means nothing at that scale. Point --manifest at the
full public build for a real reproduction run.

Objective in the fine-tuning stage:
    L_cls(clean) + L_cls(dist) + 0.5 * KL(p_dist || p_clean)
    + 0.25 * MSE(g(f_dist), f_clean)
LP stage is BCE + label smoothing 0.05; FT stage is Focal(gamma=2.0, alpha=0.5).

Selection is worst-cell robust AUROC over the transform grid, not clean AUROC:
at NTIRE 2026 clean AUROC mis-ranked the entry built like Branch A by six places.
`--no-eval-cells` falls back to clean val AUROC for quick smoke runs only.

Passing `--resume` a checkpoint from the SAME stage restores the optimiser,
scheduler, EMA and epoch counter. A checkpoint from a different stage, such as
the LP -> FT handoff, contributes weights only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aigid.branch_a import BACKBONES, BranchA
from aigid.data import BranchADataset
from aigid.distort import GRID_CELL_NAMES

CLASS_CONVENTION = "pred = P(AI-generated); label 1 = fake, label 0 = real"


# ─────────────────────────────────────────────────────────────── losses ──────
def bce_smoothed(logits, targets, smoothing=0.05):
    t = targets * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, t)


def focal(logits, targets, gamma=2.0, alpha=0.5):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t).pow(gamma) * ce).mean()


def binary_kl(logit_p, logit_q):
    """KL(p ‖ q) for Bernoulli, p from the distorted branch, q (target) detached."""
    log_p = F.logsigmoid(logit_p)
    log_1p = F.logsigmoid(-logit_p)
    log_q = F.logsigmoid(logit_q).detach()
    log_1q = F.logsigmoid(-logit_q).detach()
    p = log_p.exp()
    return (p * (log_p - log_q) + (1 - p) * (log_1p - log_1q)).mean()


# ─────────────────────────────────────────────────────────────────── EMA ─────
class EMA:
    """Parameter EMA. The shadow copy lives on `device` — keep it on 'cpu' to
    save ~1.2 GB of VRAM on a 24 GB card (the GPU↔CPU copy per step is cheap
    next to a ViT-L backward)."""

    def __init__(self, model, decay=0.999, device="cpu"):
        self.decay = decay
        self.device = device
        self.shadow = copy.deepcopy(model).to(device).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(m.to(self.device), alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m.to(self.device))

    def to(self, device):
        self.shadow.to(device)
        return self.shadow


# ──────────────────────────────────────────────────────────────── eval ───────
@torch.no_grad()
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    s, y = [], []
    for b in loader:
        logit = model(b["clean"].to(device))
        s.append(torch.sigmoid(logit).float().cpu().numpy())
        y.append(b["label"].numpy())
    return auroc(np.concatenate(s), np.concatenate(y))


@torch.no_grad()
def per_content(model, loader, device):
    """AUROC split by the manifest's `content` bucket, alongside the pooled number.

    The corpus is not content-balanced: `web` runs ~20 fakes per real, because
    WildFake's LAION-prompted generators (SDXL, Midjourney, DALLE2, SD-*) have
    only one matched real source. Content correlates with the label there, so a
    model can rank on "looks like a web scrape" instead of on generation
    artifacts, and pooled AUROC would reward it either way.

    Reporting per bucket is what separates the two: a headline that holds up
    WITHIN each content bucket is not being carried by the imbalance, and one
    that collapses in `face` or `scene` while `web` stays high is. That is the
    §5.5 limitation stated as a measurement rather than a caveat.

    Buckets with one class present score nan (AUROC is undefined), which is why
    the caller reports them rather than folding them into a min.
    """
    model.eval()
    s, y = [], []
    for b in loader:
        s.append(torch.sigmoid(model(b["clean"].to(device))).float().cpu().numpy())
        y.append(b["label"].numpy())
    scores = np.concatenate(s)
    labels = np.concatenate(y)

    # The loader must not shuffle: rows line up with the dataset's manifest by
    # position, and a shuffled pass would pair scores with the wrong content.
    # The dev manifest carries only the columns the loader needs, so `content`
    # is absent there; a full-build manifest always has it. Missing column and
    # length mismatch both mean "cannot attribute a score to a bucket", so both
    # fall back to the pooled number rather than failing the epoch.
    rows = loader.dataset.rows
    if "content" not in rows.columns or len(rows) != len(scores):
        return auroc(scores, labels), {}
    out = {}
    for bucket, idx in rows.groupby("content").groups.items():
        i = np.asarray(idx, dtype=int)
        out[str(bucket)] = auroc(scores[i], labels[i])
    return auroc(scores, labels), out


@torch.no_grad()
def worst_cell(model, device, split, size, cap, manifest, workers, image_root=None):
    """Min per-cell AUROC over the §6.1 grid — the number §6.2 selects on.
    Capped at `cap` images per cell for speed on the dev run."""
    model.eval()
    out = {}
    for cell in GRID_CELL_NAMES:
        ds = BranchADataset(split, size=size, train=False, grid_cell=cell,
                            manifest_path=manifest, image_root=image_root)
        if cap and len(ds) > cap:
            # stratified: the manifest is label-sorted, so a head slice would be
            # single-class and AUROC undefined
            per = max(1, cap // 2)
            parts = [g.sample(min(len(g), per), random_state=0)
                     for _, g in ds.rows.groupby("label")]
            ds.rows = pd.concat(parts).reset_index(drop=True)
        dl = DataLoader(ds, batch_size=32, num_workers=workers)
        s, y = [], []
        for b in dl:
            s.append(torch.sigmoid(model(b["clean"].to(device))).float().cpu().numpy())
            y.append(b["label"].numpy())
        out[cell] = auroc(np.concatenate(s), np.concatenate(y))
    worst = min((v for v in out.values() if v == v), default=float("nan"))
    return worst, out


# ─────────────────────────────────────────────────────── checkpoint I/O ──────
def save_ckpt(path, model, ema, opt, sched, epoch, step, args, swa=None,
              best=None):
    """Every field a resume needs, plus the §1 checkpoint contract (config,
    calibration temperature, threshold, class convention, checksum).

    On the checksum: `state_shape_sha256` digests the state_dict's {name: shape}
    map, NOT its values -- two epochs of the same run share it, and so do two
    different runs of the same config. It is a load-compatibility gate for
    `predict.py` (fail with "wrong architecture" rather than a wall of key
    mismatches), not an integrity check. Two limits worth knowing: it cannot
    detect corrupted weights, and it does not move with `img_size` -- the backbone is
    RoPE-based and has no pos_embed, so 384 and 448 hash identically. The input
    size lives in `args`. File integrity for the shipped checkpoint comes from
    the HF Hub LFS digest on download (A1), not from inside the file, where
    anything that corrupts the weights could corrupt the hash too.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    blob = json.dumps({k: list(v.shape) for k, v in state.items()}, sort_keys=True)
    ckpt = {
        "model": state,
        "ema": ema.shadow.state_dict() if ema is not None else None,
        "swa": swa.state_dict() if swa is not None else None,
        "opt": opt.state_dict() if opt else None,
        "sched": sched.state_dict() if sched else None,
        "epoch": epoch, "step": step,
        # The best selection score so far. Without it a resume restarts at -inf
        # and the first epoch after a preemption overwrites best.pt whatever it
        # scores -- which is precisely the run §5's checkpoint discipline exists
        # to protect.
        "best": best,
        "config": model.config(),
        "param_report": model.param_report(),
        # Both are placeholders until scripts/calibrate.py fits them on val
        # (§5) and writes them back in. predict.py ships sigmoid(logit/T), so an
        # uncalibrated checkpoint ships raw sigmoids against a 0.5 cut.
        # `threshold` is the balanced-accuracy operating point; the 1%/5% FPR
        # points §7's deployment page uses live in the calibration report
        # alongside the checkpoint, not in here.
        "calibration_temperature": 1.0,
        "threshold": 0.5,
        "class_convention": CLASS_CONVENTION,
        "state_shape_sha256": hashlib.sha256(blob.encode()).hexdigest(),
        "args": vars(args),
    }
    torch.save(ckpt, path)


# ───────────────────────────────────────────────────────────────── train ─────
def run(args):
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device == "cuda":
        # Fixed 448² input every step + Ampere TF32 for the fp32 ops (norms, head).
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = BranchA(backbone=args.backbone, img_size=args.img_size,
                    pool=args.pool, pretrained=not args.no_pretrained,
                    weights=args.weights).to(device)

    # weights_only defaults to True from torch 2.6; this checkpoint carries
    # `args`, `config` and optimiser/scheduler state, none of which survive it.
    # predict.py:88 passes the same flag for the same reason.
    prev = (torch.load(args.resume, map_location="cpu", weights_only=False)
            if args.resume else None)
    # Same stage AND same FT path -> a real mid-run resume (optimiser, scheduler,
    # EMA, epoch counter all restored). Anything else -- the LP -> FT handoff -- is
    # a weights-only warm start, and the parameter set legitimately differs.
    pargs = (prev or {}).get("args", {}) or {}
    # The fallback matches --ft-path's default: a checkpoint saved before the
    # flag existed carries no ft_path, and reading it as anything other than the
    # default would resolve a legitimate same-stage resume as a weights-only
    # warm start -- silently dropping optimiser, scheduler and EMA state.
    same_run = bool(prev) and pargs.get("stage") == args.stage \
        and pargs.get("ft_path", "full") == args.ft_path

    if prev is not None and not same_run:
        # Load before staging: an LP checkpoint predates the LoRA wrapper, so
        # its key names only match the unwrapped module.
        miss = model.load_state_dict(prev["model"], strict=False)
        print(f"warm start from {args.resume} (stage {pargs.get('stage')} -> "
              f"{args.stage}; {len(miss.missing_keys)} missing, "
              f"{len(miss.unexpected_keys)} unexpected keys)")

    if args.stage == "lp":
        model.freeze_for_lp()
    elif args.ft_path == "full":
        # §5 step 2, full-FT path: every block trainable at 1e-5, head warm from
        # the LP stage. Both NTIRE top-2 teams full-fine-tuned; §8's benchmark
        # decides between this and the LoRA path, and nothing else does.
        model.freeze_for_lp()
        for prm in model.backbone.parameters():
            prm.requires_grad_(True)
    else:
        model.freeze_for_lp()          # heads stay trainable
        model.enable_ft(lora_r=args.lora_r, unfreeze_last_n=args.unfreeze_last_n)
    if args.grad_checkpoint:
        model.set_grad_checkpointing(True)
    model.to(device)

    if prev is not None and same_run:
        model.load_state_dict(prev["model"], strict=True)
        print(f"resuming {args.stage}/{args.ft_path} run from {args.resume} "
              f"(epoch {prev['epoch']}, step {prev['step']})")

    print("param report:", json.dumps(model.param_report(), indent=2))

    train_ds = BranchADataset("train", size=args.img_size, train=True,
                              distort=args.stage == "ft" or args.distort_lp,
                              manifest_path=args.manifest, image_root=args.image_root, seed=args.seed)
    val_ds = BranchADataset("val", size=args.img_size, train=False,
                            distort=False, manifest_path=args.manifest, image_root=args.image_root)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=device == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=device == "cuda")

    # §5's "all blocks at 1e-5" applies to the full-FT path; the LoRA path keeps
    # the unfrozen last-4 blocks at 2e-5.
    block_lr = 1e-5 if (args.stage == "ft" and args.ft_path == "full") else 2e-5
    opt = torch.optim.AdamW(model.param_groups(block_lr=block_lr),
                            weight_decay=args.weight_decay)

    # The scheduler is stepped once per OPTIMISER step, so its horizon has to be
    # counted in optimiser steps too. Counting micro-batches instead stretches
    # the one-epoch warmup to `grad_accum` epochs and leaves cosine unfinished.
    accum = max(1, args.grad_accum)
    steps_per_epoch = max(1, len(train_dl) // accum)
    warmup = steps_per_epoch                      # §5: one-epoch warmup
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, total_steps - warmup))],
        milestones=[warmup])
    ema = None if args.no_ema else EMA(model, decay=args.ema_decay,
                                       device=args.ema_device)
    amp = device == "cuda"
    # Autocast runs in bf16, which keeps fp32's exponent range, so there is
    # nothing to rescale -- GradScaler is fp16 machinery. Disabled rather than
    # deleted: its four call sites below are pass-throughs when it is off, so
    # switching to fp16 is this one flag. Left enabled it would also SKIP an
    # optimiser step on any inf/nan without logging, which is a silent failure
    # mode we get nothing back for.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # §5 step 3: SWA over the final epochs, full-FT path only. ViT-L carries no
    # BatchNorm, so there is no `update_bn` pass to pay for — "no extra compute"
    # holds literally.
    swa = None
    swa_from = (args.epochs - args.swa_epochs
                if args.stage == "ft" and args.ft_path == "full"
                and args.swa_epochs > 0 else None)

    step = 0
    micro = 0
    best = -1.0
    start_epoch = 0
    if prev is not None and same_run:
        if prev.get("opt"):
            opt.load_state_dict(prev["opt"])
        if prev.get("sched"):
            sched.load_state_dict(prev["sched"])
        if ema is not None and prev.get("ema"):
            ema.shadow.load_state_dict(prev["ema"])
        if prev.get("swa"):
            swa = torch.optim.swa_utils.AveragedModel(model, device=args.ema_device)
            swa.load_state_dict(prev["swa"])
        start_epoch, step = int(prev["epoch"]) + 1, int(prev["step"])
        # `best` only carries over if the checkpoint was selected on the SAME
        # metric. Resuming a worst-cell run with --no-eval-cells (or the
        # reverse) would compare different scales: clean AUROC sits higher, so
        # best.pt would update on noise one way round and never update at all
        # the other. Mismatched -> start fresh and say so.
        same_metric = pargs.get("no_eval_cells") == args.no_eval_cells
        if prev.get("best") is not None and same_metric:
            best = float(prev["best"])
        note = (f"best so far {best:.4f}" if best > -1.0
                else "best starts fresh"
                       + ("" if same_metric else " (selection metric changed)"))
        print(f"  optimiser/scheduler/EMA restored; continuing at epoch "
              f"{start_epoch}, {note}")

    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        for b in train_dl:
            clean = b["clean"].to(device, non_blocking=True)
            dist = b["distorted"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                if args.stage == "lp":
                    loss = bce_smoothed(model(clean), y)
                    parts = {"cls": loss.item()}
                else:
                    lc, fc = model(clean, return_features=True)
                    ld, fd = model(dist, return_features=True)
                    l_cls = focal(lc, y) + focal(ld, y)
                    l_kl = 0.5 * binary_kl(ld, lc)
                    l_feat = 0.25 * F.mse_loss(model.correction(fd), fc.detach())
                    loss = l_cls + l_kl + l_feat
                    parts = {"cls": l_cls.item(), "kl": l_kl.item(),
                             "feat": l_feat.item()}

            scaler.scale(loss / accum).backward()
            micro += 1
            if micro % accum != 0:
                continue

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            if ema is not None:
                ema.update(model)
            step += 1

            if step % args.log_every == 0:
                ips = args.batch_size * accum * args.log_every / (time.time() - t0)
                print(f"e{epoch} s{step} loss {loss.item():.4f} "
                      f"{ {k: round(v, 4) for k, v in parts.items()} } "
                      f"{ips:.1f} img/s")
                t0 = time.time()
            if args.max_steps and step >= args.max_steps:
                break

        if device == "cuda":
            torch.cuda.empty_cache()

        # SWA absorbs this epoch's weights before evaluation, so the number that
        # selects the checkpoint is the number the averaged model actually scores.
        if swa_from is not None and epoch >= swa_from:
            if swa is None:
                swa = torch.optim.swa_utils.AveragedModel(model, device=args.ema_device)
            else:
                swa.update_parameters(model)

        if swa is not None:
            eval_model, tag = swa.module.to(device).eval(), "SWA"
        elif ema is not None:
            eval_model, tag = ema.to(device), "EMA"
        else:
            eval_model, tag = model, "live"
        val_auroc, by_content = per_content(eval_model, val_dl, device)
        msg = f"[epoch {epoch}] val AUROC ({tag}) {val_auroc:.4f}"
        eval_cells = not args.no_eval_cells
        if eval_cells:
            w, cells = worst_cell(eval_model, device, "val", args.img_size,
                                  args.cell_cap, args.manifest, args.workers,
                                  args.image_root)
            msg += f" | worst-cell {w:.4f}"
            print(msg)
            print("  cells:", {k: round(v, 3) for k, v in cells.items()})
        else:
            print(msg)
        if by_content:
            print("  by content:", {k: round(v, 3) for k, v in by_content.items()})
        if ema is not None and args.ema_device != device:
            ema.shadow.to(args.ema_device)
        if swa is not None and args.ema_device != device:
            swa.module.to(args.ema_device)

        # §6.2: worst-cell robust AUROC is the selection metric. Clean AUROC is
        # the fallback only when the grid sweep is explicitly switched off.
        # Decide BEFORE writing last.pt, so the running best travels with the
        # checkpoint a resume reads back rather than lagging it by an epoch.
        select = w if eval_cells else val_auroc
        improved = select == select and select > best
        if improved:
            best = select
        save_ckpt(Path(args.out) / "last.pt", model, ema, opt, sched, epoch, step,
                  args, swa, best)
        if improved:
            save_ckpt(Path(args.out) / "best.pt", model, ema, opt, sched, epoch,
                      step, args, swa, best)
            print(f"  new best ({best:.4f}) -> best.pt")
        if args.max_steps and step >= args.max_steps:
            break

    metric = "clean val AUROC" if args.no_eval_cells else "worst-cell robust AUROC"
    print(f"done. best {metric} {best:.4f}. checkpoints in {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["lp", "ft"], required=True)
    p.add_argument("--ft-path", choices=["full", "lora"], default="full",
                   help="'full' (default) fine-tunes all blocks at "
                        "1e-5 + SWA; 'lora' trains LoRA r=32 + the last four "
                        "blocks and is retained for ablation.")
    p.add_argument("--swa-epochs", type=int, default=1,
                   help="full-FT path only: average the final N epochs")
    p.add_argument("--backbone", default="dinov2-l14-448",
                   choices=sorted(BACKBONES),
                   help="dinov2-l14-448 (ship path; Apache-2.0) | "
                        "eva02-l14-448 (MIT, native 448)")
    p.add_argument("--weights", default=None,
                   help="optional local pretrained checkpoint for a supported "
                        "backbone (never committed)")
    p.add_argument("--manifest", default=None,
                   help="parquet path; default = data/manifest.parquet")
    p.add_argument("--image-root", default=None,
                   help="image dir; default = data/full")
    p.add_argument("--out", default="runs/branch_a")
    p.add_argument("--resume", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--pool", default="gap", choices=["gap", "gap+cls"])
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--unfreeze-last-n", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-device", default="cpu", help="'cpu' saves ~1.2 GB VRAM")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0, help="smoke cap (0 = full)")
    p.add_argument("--distort-lp", action="store_true",
                   help="also apply distortion in the LP stage (default: clean)")
    p.add_argument("--no-eval-cells", action="store_true",
                   help="select on clean val AUROC instead of the worst-cell "
                        "metric. Smoke runs only -- clean AUROC "
                        "mis-ranks robustness.")
    p.add_argument("--cell-cap", type=int, default=200)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="gradient checkpointing -- trades throughput for VRAM. "
                        "NOT needed on an H100: full-FT peaks at 34.6 GB of 85 "
                        "at batch 16 / 448. For smaller cards only.")
    p.add_argument("--grad-accum", type=int, default=1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
