"""The graded deliverable: a directory of images in, one JSON file of scores out.

    python -m aigid.predict --input <image-dir> --output predictions.json
    python -m aigid.predict --input <image-dir> --output predictions.json --no-tta
    python -m aigid.predict --report-params            # no images needed
    python -m aigid.predict --input <dir> --output p.json --stub   # no torch

Output is a sorted JSON array of ``{"image_path": ..., "pred": 0.8734}`` and
NOTHING else -- the rules fix that shape, so no `errors` key goes in the graded
file. `pred` is the calibrated probability that the image is AI-generated.

Robustness is a hard requirement, not a nicety: the rules want a score for every
image, so an undecodable file gets ``pred: 0.5`` (abstain) and the run continues.
One corrupt file must never cost the other 9,999. The error list goes to stderr
and to a sidecar ``<output>.errors.json``, which is deliberately NOT the graded
file.

`--stub` runs the whole contract -- discovery, ordering, error handling, JSON
shape -- without importing torch, so the interface can be smoke-tested before a
checkpoint exists.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Rules §5.2 works over consumer image formats; the submitted interface accepts
# these five common formats.
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CHECKPOINT = "runs/ft_full_calibrated/best.pt"

# The score for an image we could not read. 0.5 is the only honest answer: it is
# the decision boundary, so it neither accuses nor clears a file we never saw.
ABSTAIN = 0.5


# ────────────────────────────────────────────────────────── discovery ────────
def find_images(root: Path) -> list[Path]:
    """Every supported image under `root`, recursively, in a stable order.

    Sorted here rather than at output time so the batch order, the error log and
    the JSON all agree -- and so two runs on the same directory are identical.
    """
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUFFIXES)


def image_key(path: Path, root: Path, style: str) -> str:
    """How `image_path` is spelled in the output.

    The rules say "image_path" without fixing the spelling, so this is a guess we
    make explicit rather than bury: default to the path relative to --input,
    which equals the bare filename for a flat directory (the expected case) and
    still disambiguates nested ones.
    """
    if style == "absolute":
        return str(path.resolve())
    if style == "basename":
        return path.name
    return path.relative_to(root).as_posix()


def write_outputs(rows: list[dict], errors: list[dict], out: Path) -> None:
    """The graded file, then the sidecar. Order matters: if the sidecar write
    fails we still want the scores on disk."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["image_path"])
    out.write_text(json.dumps(rows, indent=2), encoding="utf8")
    side = out.with_suffix(out.suffix + ".errors.json") if out.suffix \
        else out.with_name(out.name + ".errors.json")
    side.write_text(json.dumps(errors, indent=2), encoding="utf8")
    print(f"wrote {len(rows)} predictions -> {out}", file=sys.stderr)
    if errors:
        print(f"  {len(errors)} unreadable file(s) scored {ABSTAIN} -> {side}",
              file=sys.stderr)
        for e in errors[:10]:
            print(f"    {e['image_path']}: {e['error']}", file=sys.stderr)
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more", file=sys.stderr)
    else:
        print(f"  no read errors -> {side} (empty)", file=sys.stderr)


# ─────────────────────────────────────────────────────── checkpoint ──────────
def load_checkpoint(path: str):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise SystemExit(f"{path} is not an aigid checkpoint (no 'model' key)")
    return ckpt


def pick_weights(ckpt: dict, prefer: str | None):
    """SWA, then EMA, then the live weights.

    That order is not arbitrary: model selection (§6.2) scored whichever of these
    the trainer was evaluating, so shipping the raw `model` tensors would ship a
    set of weights whose robust AUROC was never measured.
    """
    order = [prefer] if prefer else ["swa", "ema", "model"]
    for key in order:
        sd = ckpt.get(key)
        if not sd:
            continue
        if key == "swa":
            # AveragedModel wraps the net in `.module` and adds n_averaged.
            sd = {k[len("module."):]: v for k, v in sd.items()
                  if k.startswith("module.")}
        return key, sd
    raise SystemExit(f"checkpoint has none of {order}")


def build_model(ckpt: dict, device: str, prefer: str | None):
    """Rebuild the trained net from what the checkpoint records about itself.

    `pretrained=False` matters for more than speed: judges must be able to run
    the shipped checkpoint without reaching for upstream weights or network
    access. Every tensor comes from the checkpoint.
    """
    import torch
    from aigid.branch_a import BranchA

    cfg = ckpt.get("config", {})
    targs = ckpt.get("args", {}) or {}
    if "backbone" not in cfg:
        raise SystemExit(
            "checkpoint has no config['backbone']. Refusing to guess: building "
            "an EVA02 skeleton and loading DINOv2 tensors into it silently "
            "produces a model that scores, but not the one that was trained. "
            "Re-save the checkpoint from aigid.train, which records it.")

    model = BranchA(backbone=cfg["backbone"],
                    img_size=int(targs.get("img_size", 448)),
                    pool=cfg.get("pool", "gap"), pretrained=False)
    if cfg.get("lora"):
        # The saved tensors carry peft's names, so the wrapper has to exist
        # before the load or every LoRA key comes back "unexpected".
        model.freeze_for_lp()
        model.enable_ft(lora_r=int(targs.get("lora_r", 32)),
                        unfreeze_last_n=int(targs.get("unfreeze_last_n", 4)))

    key, sd = pick_weights(ckpt, prefer)

    # Compatibility gate. state_shape_sha256 digests {name: shape}, so a mismatch
    # means this checkpoint was trained against a different architecture -- say
    # so plainly instead of emitting a wall of key errors.
    import hashlib
    want = ckpt.get("state_shape_sha256")
    blob = json.dumps({k: list(v.shape) for k, v in model.state_dict().items()},
                      sort_keys=True)
    got = hashlib.sha256(blob.encode()).hexdigest()
    if want and want != got:
        raise SystemExit(
            "checkpoint/architecture mismatch: this checkpoint was trained "
            f"against a different module layout (expected {want[:16]}, rebuilt "
            f"{got[:16]}). Check --checkpoint, and note the digest does not "
            "cover img_size.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint weights '{key}' do not exactly match the rebuilt "
            f"model: {len(missing)} missing / {len(unexpected)} unexpected "
            "keys. Refusing partial loading because it would calibrate or "
            "score a different model.")
    return model.to(device).eval(), key


def calibrated_score_contract(ckpt: dict, requested_weights: str | None,
                              tta: bool, require: bool = False) -> str | None:
    """Validate and return the exact score path used during calibration."""
    calibration = ckpt.get("calibration")
    if not isinstance(calibration, dict):
        if require:
            raise SystemExit(
                "checkpoint has no embedded accepted calibration record")
        return requested_weights
    if calibration.get("schema_version") != 1:
        raise SystemExit("unsupported or missing calibration schema version")
    if calibration.get("status") != "accepted" or not calibration.get("artifact_id"):
        raise SystemExit("checkpoint calibration record is not accepted/bound")
    if calibration.get("split") != "val":
        raise SystemExit("checkpoint calibration was not fitted on validation")
    expected_weights = calibration.get("weights_key")
    expected_tta = bool(calibration.get("tta"))
    temperature = float(calibration.get("temperature", float("nan")))
    threshold = float(calibration.get("threshold", float("nan")))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise SystemExit("calibration temperature must be finite and positive")
    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise SystemExit("calibration threshold must be within [0, 1]")
    if float(ckpt.get("calibration_temperature", float("nan"))) != temperature:
        raise SystemExit("top-level temperature does not match calibration record")
    if float(ckpt.get("threshold", float("nan"))) != threshold:
        raise SystemExit("top-level threshold does not match calibration record")
    if requested_weights is not None and requested_weights != expected_weights:
        raise SystemExit(
            f"--weights-key {requested_weights} conflicts with calibrated "
            f"weights {expected_weights}")
    if bool(tta) != expected_tta:
        raise SystemExit(
            f"TTA setting conflicts with calibration: calibrated "
            f"tta={'on' if expected_tta else 'off'}")
    return expected_weights


# ──────────────────────────────────────────────────────── inference ──────────
def preprocess(img, size: int):
    """§4.2's inference spatial policy: squish-resize, no crop, so the model sees
    the whole frame. Train and inference differ here deliberately."""
    import numpy as np
    import torch
    from PIL import Image
    from aigid.constants import IMAGENET_MEAN, IMAGENET_STD

    img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def score_batch(model, batch, device, tta: bool, temperature: float):
    """Logits -> calibrated probabilities. hflip TTA averages the two views'
    LOGITS, not their probabilities: averaging after the sigmoid pulls confident
    disagreements toward 0.5 and is not what §5 specifies."""
    import torch
    x = torch.stack(batch).to(device)
    amp = device == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=amp):
        logit = model(x).float()
        if tta:
            logit = (logit + model(torch.flip(x, dims=[3])).float()) / 2.0
    return torch.sigmoid(logit / max(temperature, 1e-6)).cpu().tolist()


def run_predict(args) -> int:
    import torch
    from PIL import Image

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_checkpoint(args.checkpoint)
    weights_key = calibrated_score_contract(
        ckpt, args.weights_key, args.tta, require=True)
    model, key = build_model(ckpt, device, weights_key)
    temperature = float(ckpt.get("calibration_temperature", 1.0))
    size = int((ckpt.get("args") or {}).get("img_size", 448))
    print(f"device={device} weights={key} T={temperature:g} size={size} "
          f"tta={'on' if args.tta else 'off'}", file=sys.stderr)

    root = Path(args.input)
    paths = find_images(root)
    if not paths:
        print(f"no images found under {root}", file=sys.stderr)
    rows, errors, batch, keys = [], [], [], []

    def flush():
        if not batch:
            return
        for k, p in zip(keys, score_batch(model, batch, device, args.tta,
                                          temperature)):
            rows.append({"image_path": k, "pred": round(float(p), 6)})
        batch.clear()
        keys.clear()

    for path in paths:
        name = image_key(path, root, args.path_style)
        try:
            with Image.open(path) as im:
                im.load()
                batch.append(preprocess(im, size))
            keys.append(name)
        except Exception as exc:                     # noqa: BLE001 -- see below
            # Deliberately broad: a truncated JPEG, an unsupported subformat and
            # a permissions error must all degrade to an abstention rather than
            # end the run.
            rows.append({"image_path": name, "pred": ABSTAIN})
            errors.append({"image_path": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if len(batch) >= args.batch_size:
            flush()
    flush()

    write_outputs(rows, errors, Path(args.output))
    return 0


def run_stub(args) -> int:
    """The contract without the model: same discovery, ordering, error handling
    and JSON shape, no torch import. Lets the interface be tested before a
    checkpoint exists (A0)."""
    root = Path(args.input)
    rows, errors = [], []
    for path in find_images(root):
        name = image_key(path, root, args.path_style)
        try:
            with open(path, "rb") as fh:
                if not fh.read(1):
                    raise ValueError("empty file")
            rows.append({"image_path": name, "pred": ABSTAIN})
        except Exception as exc:                     # noqa: BLE001
            rows.append({"image_path": name, "pred": ABSTAIN})
            errors.append({"image_path": name, "error": f"{type(exc).__name__}: {exc}"})
    print("STUB MODE -- every score is a placeholder, not a prediction",
          file=sys.stderr)
    write_outputs(rows, errors, Path(args.output))
    return 0


def run_report_params(args) -> int:
    """§3.3 / A2: the README's parameter count is pasted from this output at
    freeze, so it can never drift from what actually ships."""
    ckpt = load_checkpoint(args.checkpoint)
    model, key = build_model(ckpt, "cpu", args.weights_key)
    rows = model.param_report()
    print(f"checkpoint : {args.checkpoint}")
    print(f"weights    : {key}")
    print(f"config     : {json.dumps(ckpt.get('config', {}), sort_keys=True)}")
    print(f"input size : {(ckpt.get('args') or {}).get('img_size', 448)}")
    print(f"calibration: T={ckpt.get('calibration_temperature', 1.0)} "
          f"threshold={ckpt.get('threshold')}")
    print(f"convention : {ckpt.get('class_convention')}")
    print("\nparameters loaded by this checkpoint:")
    for name, n in rows.items():
        if name in ("total", "trainable"):
            continue
        print(f"  {name:12s} {n:>13,}  ({n/1e6:.1f}M)")
    print(f"  {'TOTAL':12s} {rows['total']:>13,}  ({rows['total']/1e6:.1f}M)")
    print(f"\n2B cap (rules): {'PASS' if rows['total'] < 2e9 else 'FAIL'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="directory of images (recursive)")
    p.add_argument("--output", default="predictions.json")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="local checkpoint path (default: calibrated shipped "
                        "artifact)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-tta", dest="tta", action="store_false",
                   help="skip hflip TTA -- halves inference cost")
    p.add_argument("--weights-key", choices=["swa", "ema", "model"], default=None,
                   help="override which tensors to load (default: swa, else ema, "
                        "else model -- matching what model selection scored)")
    p.add_argument("--path-style", choices=["relative", "basename", "absolute"],
                   default="relative", help="how image_path is spelled")
    p.add_argument("--stub", action="store_true",
                   help="exercise the output contract without torch (A0)")
    p.add_argument("--report-params", action="store_true",
                   help="print the parameter count of every loaded module")
    args = p.parse_args()

    if args.report_params:
        return run_report_params(args)
    if not args.input:
        p.error("--input is required (or use --report-params)")
    if args.stub:
        return run_stub(args)
    return run_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
