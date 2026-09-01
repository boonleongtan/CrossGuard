"""Calibration: fit the temperature and freeze the operating threshold.

    python scripts/calibrate.py --checkpoint runs/ft/best.pt --dry-run
    python scripts/calibrate.py --checkpoint runs/ft_full/best.pt \
        --out runs/ft_full_calibrated/best.pt --cap 8000

`--out` is REQUIRED and must differ from `--checkpoint`. Calibration reads the
trained checkpoint and writes a new one; it never rewrites its input, so an
interrupt mid-`torch.save` costs you nothing but a re-run. Pointing both at the
same path is refused rather than obeyed -- the trained weights take GPU-hours to
reproduce and this step takes minutes.

The diagnostics JSON defaults beside the output checkpoint. The accepted
record is also embedded in that checkpoint, so consumers cannot silently pair
one model with another model's operating points.

`predict.py` ships `sigmoid(logit / T)` and reads both `calibration_temperature`
and `threshold` out of the checkpoint. Nothing else in the pipeline writes them,
so without this step every shipped `pred` is an uncalibrated sigmoid and the
threshold is a placeholder 0.5.

The graded submission carries `pred` alone, so the threshold never enters a
scored number. It drives the demo decision and deployment triage bands. Two
operating points are reported: the balanced-accuracy threshold that ships in
the checkpoint, and the 1%/5% FPR points used for the deployment framing. On a
well-separated validation split, balanced accuracy is flat across a wide band
of thresholds; the fit ships the centre of that band rather than the scan's
argmax, and writes the full threshold curve into the report so the choice is
auditable.

Both quantities come off the validation split, never test. The evaluation set
is mixed clean + distorted: one clean pass plus one pass per robustness-grid
cell, matching the distribution `predict.py` meets in the wild.

Weights follow `predict.py`'s own selection (SWA -> EMA -> raw) via its
`build_model`, because a temperature fitted to weights other than the ones that
ship calibrates the wrong model.

Two things this deliberately does NOT do:

* It does not touch the model. Temperature scaling is a single scalar fitted
  post-hoc on frozen logits; refitting weights here would invalidate the model
  selection that already happened.
* It does not re-evaluate on test, or report a robustness number. That is M4's
  sweep. This writes two scalars and the diagnostics that justify them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigid.data import BranchADataset          # noqa: E402
from aigid.distort import GRID_CELL_NAMES      # noqa: E402
from aigid.predict import build_model, load_checkpoint  # noqa: E402


CALIBRATION_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temp_sibling(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def atomic_write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_sibling(path)
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch_save(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_sibling(path)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ────────────────────────────────────────────────────────── logit capture ────
@torch.no_grad()
def collect_logits(model, device, split, size, cap, manifest, workers,
                   image_root=None, batch_size=32, tta=True):
    """Uncalibrated logits over clean + every §6.1 grid cell.

    TTA is on by default because `predict.py` defaults to it: the temperature
    has to be fitted to the same logit the deployed path produces, and the
    hflip average is not a monotone function of the single-view logit, so a T
    fitted without TTA is not the T that calibrates the shipped scores.
    """
    model.eval()
    views, out_logits, out_labels = ["clean"] + list(GRID_CELL_NAMES), [], []
    per_view = {}

    for view in views:
        ds = BranchADataset(split, size=size, train=False,
                            distort=False,
                            grid_cell=None if view == "clean" else view,
                            manifest_path=manifest, image_root=image_root)
        if cap and len(ds) > cap:
            # Stratified: the manifest is label-sorted, so a head slice would be
            # single-class and both the fit and the AUROC would be degenerate.
            import pandas as pd
            per = max(1, cap // 2)
            parts = [g.sample(min(len(g), per), random_state=0)
                     for _, g in ds.rows.groupby("label")]
            ds.rows = pd.concat(parts).reset_index(drop=True)
        if len(ds) == 0:
            print(f"  {view}: no rows, skipped", file=sys.stderr)
            continue

        dl = DataLoader(ds, batch_size=batch_size, num_workers=workers)
        lg, yy = [], []
        amp = device == "cuda"
        for b in dl:
            x = b["clean"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logit = model(x).float()
                if tta:
                    logit = (logit + model(torch.flip(x, dims=[3])).float()) / 2.0
            lg.append(logit.cpu())
            yy.append(b["label"].float())
        lg = torch.cat(lg)
        yy = torch.cat(yy)
        per_view[view] = (lg, yy)
        out_logits.append(lg)
        out_labels.append(yy)
        print(f"  {view}: {len(lg)} images", file=sys.stderr)

    if not out_logits:
        raise SystemExit(f"no rows in split {split!r} -- nothing to calibrate on")
    return torch.cat(out_logits), torch.cat(out_labels), per_view


# ───────────────────────────────────────────────────────────── temperature ───
def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """The T minimising NLL of sigmoid(logit / T).

    Optimised over log T with LBFGS, so T stays positive by construction rather
    than by clamping a raw parameter that the optimiser can walk negative.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100,
                            line_search_fn="strong_wolfe")
    lg, y = logits.detach(), labels.detach()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def nll(logits, labels, t=1.0):
    return float(torch.nn.functional.binary_cross_entropy_with_logits(
        logits / t, labels).item())


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, equal-width bins. Reported, not optimised."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared probability error. Reported, not optimised."""
    return float(np.mean(np.square(probs - labels)))


# ───────────────────────────────────────────────────────────── threshold ─────
def threshold_at_fpr(probs: np.ndarray, labels: np.ndarray, target_fpr: float):
    """Lowest threshold whose FPR on the reals is <= `target_fpr`.

    §7's deployment framing commits the operating point to 1% FPR, because the
    asymmetry it names is real: a false positive wrongly flags a genuine
    creator. That is a different quantity from the balanced-accuracy argmax and
    has to be computed, not inferred from it.

    Lowest-such-threshold rather than any of them: FPR is monotone
    non-increasing in the threshold, so every value above the returned one also
    clears the target, and the lowest is the one that keeps the most TPR.
    Returns (threshold, achieved_fpr, tpr, reached) -- the achieved FPR is
    reported rather than assumed, since a discrete score set cannot hit an
    arbitrary target exactly, and `reached` says whether the target was met at
    all. A target BELOW the floor set by reals scoring at the very top is
    unreachable: with n reals the finest achievable non-zero FPR is 1/n, and
    ties at 1.0 raise that floor further. Returning the best available point
    while silently reporting an FPR above target would put a number in the
    deployment page that the model does not actually achieve, so the caller is
    told.
    """
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        raise SystemExit("split has only one class -- cannot fit an FPR threshold")

    # Candidates as in best_threshold: balanced accuracy, FPR and TPR are all
    # piecewise-constant in the threshold, so midpoints between adjacent
    # distinct scores reach every attainable value.
    uniq = np.unique(probs)
    cands = ((uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1
             else np.array([float(uniq[0])]))
    cands = np.concatenate([[0.0], cands, [1.0]])

    for t in cands:
        pred = probs >= t
        fpr = float(pred[neg].mean())
        if fpr <= target_fpr:
            return float(t), fpr, float(pred[pos].mean()), True

    # Unreachable. Fall back to the strictest threshold there is and say so.
    pred = probs >= 1.0
    return 1.0, float(pred[neg].mean()), float(pred[pos].mean()), False


def best_threshold(probs: np.ndarray, labels: np.ndarray):
    """Threshold maximising balanced accuracy (§5), plus the plateau it sits on.

    Balanced accuracy, not raw accuracy: the val split is not guaranteed
    class-balanced, and a raw-accuracy argmax on a skewed split drifts toward
    the majority class. Candidates are the midpoints between adjacent distinct
    scores -- balanced accuracy is piecewise-constant in the threshold, so every
    distinct value it can take is reachable from one of these.

    Returns (threshold, balanced_accuracy, plateau), where `plateau` is the
    interval of thresholds achieving the maximum, measured between the observed
    scores that bound it. On a well-separated val split that interval is WIDE
    -- in the limit it is the whole gap between the classes -- and the argmax
    is then an arbitrary point inside it, hugging whichever score the scan
    reached first. The centre scores identically on val and leaves margin on
    both sides, so that is what ships; `plateau` carries the evidence into the
    report rather than leaving it to be re-derived.
    """
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        raise SystemExit("val split has only one class -- cannot fit a threshold")

    uniq = np.unique(probs)
    cands = ((uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1
             else np.array([float(uniq[0])]))
    cands = np.concatenate([[0.0], cands, [1.0]])

    bas = np.empty(len(cands), dtype=float)
    for i, t in enumerate(cands):
        pred = probs >= t
        bas[i] = 0.5 * (pred[pos].mean() + (~pred[neg]).mean())

    best_ba = float(bas.max())
    at_max = np.flatnonzero(bas >= best_ba - 1e-12)

    # The plateau's extent comes from the SCORES, not from the candidate grid.
    # Candidates are midpoints between adjacent observed scores, so a clean
    # separation -- the case this exists for -- puts exactly ONE candidate in
    # the gap between the classes and would report width 0 for the widest
    # plateau there is. Instead: balanced accuracy is constant on any interval
    # containing no observed score, so the plateau runs from the highest score
    # strictly below the first maximising candidate to the lowest score at or
    # above the last one.
    first, last = float(cands[at_max[0]]), float(cands[at_max[-1]])
    below = probs[probs < first]
    above = probs[probs >= last]
    lo = float(below.max()) if below.size else 0.0
    hi = float(above.min()) if above.size else 1.0
    centre = 0.5 * (lo + hi)

    plateau = {
        "argmax_threshold": float(cands[int(bas.argmax())]),
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
        "n_candidates_at_max": int(at_max.size),
    }
    return centre, best_ba, plateau


def threshold_curve(probs: np.ndarray, labels: np.ndarray, points: int = 101):
    """Balanced accuracy / TPR / FPR sampled on a fixed threshold grid.

    Written into the report so the operating point can be justified -- and
    re-chosen -- without re-running inference. A fixed grid rather than the
    candidate midpoints keeps it small and comparable across checkpoints.
    """
    pos, neg = labels == 1, labels == 0
    grid = np.linspace(0.0, 1.0, points)
    rows = []
    for t in grid:
        pred = probs >= t
        tpr = float(pred[pos].mean()) if pos.any() else float("nan")
        fpr = float(pred[neg].mean()) if neg.any() else float("nan")
        rows.append({"threshold": float(t), "tpr": tpr, "fpr": fpr,
                     "balanced_accuracy": 0.5 * (tpr + (1.0 - fpr))})
    return rows


# ──────────────────────────────────────────────────────────────── main ───────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val",
                   help="held-out split to fit on; test is refused")
    p.add_argument("--manifest", default=None)
    p.add_argument("--image-root", default=None)
    p.add_argument("--cap", type=int, default=2000,
                   help="max images per view, stratified by label")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--weights-key", choices=["swa", "ema", "model"], default=None,
                   help="override predict.py's SWA -> EMA -> raw preference")
    p.add_argument("--no-tta", dest="tta", action="store_false",
                   help="fit without hflip TTA; use only if predict runs --no-tta")
    p.add_argument("--ece-tolerance", type=float, default=0.01,
                   help="how much ECE may worsen before the fit is rejected "
                        "(default 0.01); raise only deliberately")
    p.add_argument("--dry-run", action="store_true",
                   help="report the fit without writing to the checkpoint")
    p.add_argument("--out", required=True,
                   help="write the calibrated checkpoint here. Must differ from "
                        "--checkpoint: calibration never rewrites the trained "
                        "checkpoint, so an interrupt here cannot cost you weights.")
    p.add_argument("--report", default=None,
                   help="write the diagnostics JSON here "
                        "(default: calibration.json beside --out)")
    args = p.parse_args()

    if args.split != "val":
        raise SystemExit(
            f"refusing to calibrate on split {args.split!r}: calibration is "
            "validation-only. Use --split val.")

    if args.cap < 0:
        raise SystemExit("--cap must be >= 0 (0 means all validation rows)")

    # The trained checkpoint is not reproducible without re-running the stage on
    # a GPU, so this script never writes over its own input -- not even when
    # asked directly. Resolved, so a/./b and a/b are caught as the same file.
    if Path(args.out).resolve() == Path(args.checkpoint).resolve():
        raise SystemExit(
            f"refusing to overwrite the trained checkpoint at {args.checkpoint}. "
            "--out must be a different path; calibration writes a new file so an "
            "interrupt here cannot cost you weights.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    source_checkpoint_sha256 = sha256_file(args.checkpoint)
    ckpt = load_checkpoint(args.checkpoint)
    model, key = build_model(ckpt, device, args.weights_key)
    size = int((ckpt.get("args") or {}).get("img_size", 448))
    print(f"checkpoint={args.checkpoint} weights={key} size={size} "
          f"split={args.split} tta={'on' if args.tta else 'off'}", file=sys.stderr)

    logits, labels, per_view = collect_logits(
        model, device, args.split, size, args.cap, args.manifest,
        args.workers, args.image_root, args.batch_size, args.tta)
    if not torch.isfinite(logits).all():
        raise SystemExit("validation produced non-finite logits; refusing fit")
    if not torch.isfinite(labels).all():
        raise SystemExit("validation produced non-finite labels; refusing fit")

    t = fit_temperature(logits, labels)
    if not np.isfinite(t) or t <= 0.0:
        raise SystemExit(f"temperature fit returned invalid T={t!r}")
    probs_before = torch.sigmoid(logits).numpy()
    probs_after = torch.sigmoid(logits / t).numpy()
    y = labels.numpy()

    thr, ba, plateau = best_threshold(probs_after, y)

    # §7's deployment page commits the operating point to 1% FPR, which is a
    # different number from the balanced-accuracy threshold above and has to be
    # measured. Both ship: `threshold` stays the balanced-accuracy operating
    # point predict.py reads, and the FPR-targeted points are reported for the
    # deployment page and §6.4's hard-reals row.
    fpr_points = {}
    for target in (0.01, 0.05):
        t_fpr, achieved, tpr_at, reached = threshold_at_fpr(
            probs_after, y, target)
        fpr_points[f"fpr_{target:g}"] = {
            "threshold": t_fpr, "achieved_fpr": achieved, "tpr": tpr_at,
            "target_reached": reached,
        }
        if not reached:
            print(f"WARNING: {target:g} FPR is not achievable on this split -- "
                  f"the floor is {achieved:.4f} (reals tied at the top of the "
                  f"score range). Do NOT quote {target:g}% FPR in the "
                  f"deployment page; quote {achieved:.4f}.", file=sys.stderr)

    report = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "candidate",
        "artifact_id": None,
        "checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "weights_key": key,
        "split": args.split,
        "tta": args.tta,
        "images": int(len(y)),
        "views": list(per_view.keys()),
        "temperature": t,
        "threshold": thr,
        "balanced_accuracy_at_threshold": ba,
        "threshold_plateau": plateau,
        "operating_points": fpr_points,
        "threshold_curve": threshold_curve(probs_after, y),
        "nll_before": nll(logits, labels, 1.0),
        "nll_after": nll(logits, labels, t),
        "ece_before": ece(probs_before, y),
        "ece_after": ece(probs_after, y),
        "brier_before": brier(probs_before, y),
        "brier_after": brier(probs_after, y),
    }

    summary = {k: v for k, v in report.items() if k != "threshold_curve"}
    print(json.dumps(summary, indent=2))

    # A wide plateau means the argmax carried no information -- on a val split
    # this model separates cleanly, balanced accuracy is flat across a broad
    # band and any point in it scores identically here. Surfaced because it
    # changes how much the shipped threshold should be trusted on test.
    if plateau["width"] > 0.05:
        print(f"NOTE: balanced accuracy is flat across "
              f"[{plateau['lo']:.4f}, {plateau['hi']:.4f}] "
              f"(width {plateau['width']:.4f}). Shipping the plateau centre "
              f"{thr:.4f}; the argmax {plateau['argmax_threshold']:.4f} is an "
              f"arbitrary point in that band. See threshold_curve in the "
              f"report, and prefer the 1% FPR operating point for deployment.",
              file=sys.stderr)

    # Keep diagnostics beside the output checkpoint so a global stale report
    # cannot silently be paired with a different model.
    report_path = (Path(args.report) if args.report
                   else Path(args.out).with_name("calibration.json"))

    def write_nonshipping_report(status: str) -> None:
        report["status"] = status
        suffix = report_path.suffix or ".json"
        path = report_path.with_name(
            f"{report_path.stem}.{status}{suffix}")
        atomic_write_json(path, report)
        print(f"wrote {path}", file=sys.stderr)

    # Two acceptance checks, because NLL alone does not mean "better calibrated".
    #
    # 1. NLL must not rise. T=1 is in the feasible set, so at the optimum it
    #    cannot -- a rise means the fit did not converge.
    if report["nll_after"] > report["nll_before"]:
        print("WARNING: NLL rose after scaling -- the fit did not converge. "
              "Not writing.", file=sys.stderr)
        write_nonshipping_report("rejected")
        return 1

    # 2. ECE must not get materially worse. NLL is minimised by *ranking*, and on
    #    a model whose logits are tiny but correctly ordered the minimiser is a
    #    near-zero T that multiplies them enormously: every probability saturates
    #    to 0 or 1, NLL falls, and reliability collapses. Observed on a
    #    6-step checkpoint: T=0.014 took NLL 0.677 -> 0.285 while ECE went
    #    0.0008 -> 0.118. `pred` ships as a probability (§1), so a T that wins on
    #    NLL by destroying reliability is the wrong answer.
    if report["ece_after"] > report["ece_before"] + args.ece_tolerance:
        print(f"WARNING: ECE worsened {report['ece_before']:.4f} -> "
              f"{report['ece_after']:.4f} (tolerance {args.ece_tolerance}). "
              f"T={t:.4f} improves NLL by sharpening, not by calibrating. "
              f"Not writing.\n"
              f"  If the model is undertrained its logits carry little "
              f"magnitude information and there is nothing to calibrate yet.\n"
              f"  Override with --ece-tolerance if this is understood.",
              file=sys.stderr)
        write_nonshipping_report("rejected")
        return 1

    # A T far from 1 is legitimate on a genuinely mis-scaled model, but it is
    # also the signature of the failure above, so it is always surfaced.
    if not (0.2 <= t <= 5.0):
        print(f"NOTE: T={t:.4f} is far from 1.0. Check the reliability numbers "
              f"above before shipping this checkpoint.", file=sys.stderr)

    if args.dry_run:
        write_nonshipping_report("dry_run")
        print("dry run -- checkpoint unchanged", file=sys.stderr)
        return 0

    report["status"] = "accepted"
    artifact_basis = {
        "schema_version": report["schema_version"],
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "weights_key": key,
        "split": args.split,
        "tta": args.tta,
        "temperature": t,
        "threshold": thr,
        "operating_points": fpr_points,
    }
    report["artifact_id"] = hashlib.sha256(
        json.dumps(artifact_basis, sort_keys=True).encode("utf8")).hexdigest()

    ckpt["calibration_temperature"] = t
    ckpt["threshold"] = thr
    ckpt["calibration"] = report
    out_path = Path(args.out)
    atomic_torch_save(out_path, ckpt)
    report["output_checkpoint_sha256"] = sha256_file(out_path)
    atomic_write_json(report_path, report)
    op1 = fpr_points["fpr_0.01"]
    print(f"wrote T={t:.4f} threshold={thr:.4f} into {out_path} "
          f"(1% FPR operating point: {op1['threshold']:.4f} -> "
          f"FPR {op1['achieved_fpr']:.4f} / TPR {op1['tpr']:.4f})",
          file=sys.stderr)
    print(f"wrote bound report {report_path} "
          f"(artifact {report['artifact_id'][:16]})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
