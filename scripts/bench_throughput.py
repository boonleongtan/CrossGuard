"""Throughput benchmark for the Branch A training loop.

200 training steps of Branch A at 448 px, timing the full robustness objective
(both views, focal + KL + feature-MSE) so the number is the step that actually
runs. Reports images/sec and the projected wall-clock for a training run, which
is what sizes the schedule.

    python scripts/bench_throughput.py --device cuda --batch-size 16
    python scripts/bench_throughput.py --paths full-ft --train-images 320000

A smaller local card gives a conservative lower bound -- useful as a smoke test
and a VRAM sanity check, not as the number to plan against.

`--train-images` sets the extrapolation basis. It defaults to the last measured
build's train split; pass the real count once a new build lands, because every
hour figure below scales linearly with it.

This is only a local sizing tool: it measures the same training step used by
`aigid.train` and extrapolates wall-clock from the requested train-image count.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigid.branch_a import BranchA
from aigid.data import BranchADataset
from aigid.train import binary_kl, focal

# Extrapolation basis: the train split of the final public-data build. Pass
# --train-images with the real count from a rebuilt manifest if it differs.
FULL_BUILD_IMAGES = 257_433
EPOCHS = 3


def build_model(path: str, backbone: str, img_size: int, device: str,
                grad_checkpoint: bool = False, weights: str | None = None):
    # `weights` optionally supplies a local pretrained checkpoint for a
    # supported backbone. The normal path uses timm's pretrained configuration.
    m = BranchA(backbone=backbone, img_size=img_size, pretrained=True,
                weights=weights)
    m.freeze_for_lp()
    if path == "lora":
        m.enable_ft(lora_r=32, unfreeze_last_n=4)
    elif path == "full-ft":
        for p in m.backbone.parameters():
            p.requires_grad_(True)
    if grad_checkpoint:
        m.set_grad_checkpointing(True)
    return m.to(device)


def bench_one(path, args, device):
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    model = build_model(path, args.backbone, args.img_size, device,
                        args.grad_checkpoint, getattr(args, "weights", None))
    model.train()

    # §8 budget note: full-FT is run at 1 epoch if that is what fits; LR here only
    # affects numerics, not throughput. blocks at 1e-5 / 2e-5 per §5.
    opt = torch.optim.AdamW(
        model.param_groups(block_lr=1e-5 if path == "full-ft" else 2e-5),
        weight_decay=0.03)

    ds = BranchADataset("train", size=args.img_size, train=True,
                        distort=True, manifest_path=args.manifest,
                        image_root=args.image_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    pin_memory=device == "cuda", persistent_workers=args.workers > 0)

    it = iter(dl)
    amp = device == "cuda"
    # Off for the same reason as aigid/train.py: bf16 needs no loss scaling.
    # The benchmark must time the step that will actually run.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    def one_step():
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        clean = b["clean"].to(device, non_blocking=True)
        dist = b["distorted"].to(device, non_blocking=True)
        y = b["label"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lc, fc = model(clean, return_features=True)
            ld, fd = model(dist, return_features=True)
            # The full §5 objective, so the number that gates the compute plan
            # times the step that will actually run. Focal vs BCE is numerics
            # only, but the KL term is a real (if small) part of the graph.
            loss = (focal(lc, y) + focal(ld, y)
                    + 0.5 * binary_kl(ld, lc)
                    + 0.25 * F.mse_loss(model.correction(fd), fc.detach()))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(args.warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.steps):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    ips = args.steps * args.batch_size / dt
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model, opt, dl
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"path": path, "img_per_s": ips, "peak_gb": peak_gb,
            "trainable_M": trainable / 1e6, "sec_per_200": dt / args.steps * 200}


def extrapolate(ips, epochs=EPOCHS, images=FULL_BUILD_IMAGES):
    return images * epochs / ips / 3600.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backbone", default="dinov2-l14-448")
    p.add_argument("--manifest", default=None)
    p.add_argument("--weights", default=None,
                   help="optional local pretrained checkpoint for a supported "
                        "backbone")
    p.add_argument("--image-root", default=None,
                   help="image dir; use a WSL-native copy, not /mnt/*")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--paths", nargs="+", default=["lora", "full-ft"])
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="needed to fit full-ft on a smaller card; it lowers the "
                        "throughput number, so leave it off when the VRAM fits")
    p.add_argument("--train-images", type=int, default=FULL_BUILD_IMAGES,
                   help=f"train-split size to extrapolate against "
                        f"(default {FULL_BUILD_IMAGES:,}, the last measured "
                        f"build; pass the real count for a new build)")
    args = p.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; this benchmark should be run on a GPU.")
        dev = "cpu"

    print(f"device={dev} backbone={args.backbone} bs={args.batch_size} "
          f"img={args.img_size} steps={args.steps}\n")

    results = []
    for path in args.paths:
        try:
            r = bench_one(path, args, dev)
            results.append(r)
            hrs3 = extrapolate(r["img_per_s"], 3, args.train_images)
            hrs2 = extrapolate(r["img_per_s"], 2, args.train_images)
            print(f"-- {path}")
            print(f"   {r['img_per_s']:8.1f} img/s   peak {r['peak_gb']:.1f} GB   "
                  f"trainable {r['trainable_M']:.0f}M")
            print(f"   train split ({args.train_images:,}): 3 epochs ~= {hrs3:.1f} h   "
                  f"2 epochs ~= {hrs2:.1f} h\n")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"-- {path}: OOM at bs={args.batch_size} img={args.img_size}"
                  f"{' (+ckpt)' if args.grad_checkpoint else ''}; retry with "
                  f"--grad-checkpoint, lower --batch-size, or --img-size 384\n")
            torch.cuda.empty_cache()

    print("=" * 60)
    print(f"projected wall-clock against {args.train_images:,} train images:")
    for r in results:
        h3 = extrapolate(r["img_per_s"], 3, args.train_images)
        h2 = extrapolate(r["img_per_s"], 2, args.train_images)
        h1 = extrapolate(r["img_per_s"], 1, args.train_images)
        print(f"  {r['path']:8s}  1 ep ~= {h1:5.1f} h   2 ep ~= {h2:5.1f} h   "
              f"3 ep ~= {h3:5.1f} h")
    # Cut order under time pressure (§8), unchanged by the budget change:
    # 3 epochs -> 2 epochs -> 384^2 -> image count. NEVER cut the paired-view
    # consistency loss -- it is the robustness mechanism, not overhead.
    print("\n  cut order if the schedule is tight: 3ep -> 2ep -> 384px -> image count")
    print("  (never cut the paired-view consistency loss)")
    by = {r["path"]: r["img_per_s"] for r in results}
    if "lora" in by and "full-ft" in by:
        print(f"\n  full-ft / lora throughput ratio: "
              f"{by['full-ft'] / by['lora']:.2f}x  -- full-FT is the shipped path")


if __name__ == "__main__":
    main()
