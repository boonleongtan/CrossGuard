#!/usr/bin/env python
"""Single final pass over the test split.

    python scripts/score_test.py --checkpoint runs/ft_full/best.pt
    python scripts/score_test.py --checkpoint runs/ft_full/best.pt --dry-run

This is the ONE step that opens test. Everything before it -- model selection
(§6.2), calibration (§5) -- runs on val, and this script deliberately cannot
retune anything: it reads the temperature and the two FPR thresholds out of the
checkpoint's embedded calibration record, applies them unchanged, and reports what
they achieve. Re-deriving a threshold here would be retuning on the final split.

The gap between the val-fitted FPR and the FPR that threshold actually achieves
on test is itself a reportable number (§6.4), and an honest one -- it is the
only direct evidence of how well val transferred.

What it emits, per §6.4:

  * clean AUROC, AP, balanced accuracy, FPR/TPR
  * macro robust AUROC (equal weight per cell) and worst-cell AUROC
  * clean -> robust degradation
  * held-out-generator AUROC, and per-axis (unseen_gan / unseen_diffusion /
    unseen_other) -- five generators the model never trained on
  * TPR at the val-fitted 1% and 5% FPR points, with the FPR each ACHIEVES here
  * Brier + ECE
  * stratified bootstrap 95% CIs

Named rows §6.4 also asks for -- the tampered slice (SID_Set label 2), the
thumbnail regime (CIFAKE) and hard reals -- are separate passes over eval-only
data; `--slice` runs one without re-running the main sweep.

Logit capture, temperature fitting and the threshold helpers are imported from
`scripts/calibrate.py` rather than reimplemented: the number this reports has to
come from the same code path that produced the calibration, or the two reports
describe different models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigid.distort import GRID_CELL_NAMES              # noqa: E402
from aigid.predict import (                             # noqa: E402
    build_model, calibrated_score_contract, load_checkpoint,
)
from scripts.calibrate import collect_logits            # noqa: E402

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:                                     # pragma: no cover
    raise SystemExit("scikit-learn is required: pip install -r requirements.txt")


# ─────────────────────────────────────────────────────────────── metrics ─────
def auroc(y, p):
    """None rather than a number when a slice is single-class.

    A single-class slice has no AUROC -- sklearn raises, and substituting 0.5
    would put a fabricated number in the results table. The caller reports the
    gap instead.
    """
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, p))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, bins: int = 15):
    """Expected calibration error, equal-width bins on the probability axis.

    Equal-width rather than equal-count: the deployment question is "when the
    model says 0.9, is it right 90% of the time", which is a question about
    score VALUES, and equal-count bins move the bin edges with the score
    distribution so the answer is not comparable across checkpoints.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def bootstrap_ci(y, p, stat, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Stratified bootstrap CI: resample within each class, not across.

    Unstratified resampling lets the class balance drift between replicates,
    which widens the interval with variance that is an artifact of the
    resampling rather than of the model. Returns None when the statistic is
    undefined (single-class slice).
    """
    y, p = np.asarray(y), np.asarray(p)
    if y.min() == y.max():
        return None
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        v = stat(y[idx], p[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2))]


# ──────────────────────────────────────────────────────── slice reporting ────
def slice_rows(rows, mask, name, probs, labels):
    """One named row of the results table, with its CI."""
    y, p = labels[mask], probs[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    a = auroc(y, p)
    return {
        "slice": name,
        "n": int(mask.sum()),
        "n_fake": n_pos,
        "n_real": n_neg,
        "auroc": a,
        "auroc_ci95": bootstrap_ci(y, p, auroc) if a is not None else None,
        # Stated, not silently dropped: a single-class slice is a real property
        # of the split (held-out generators contribute no reals), and the reader
        # needs to know the AUROC is absent for that reason and not by accident.
        "note": None if a is not None else
                "single-class slice; AUROC undefined (no reals in this slice)",
    }


def held_out_axes():
    """The §6.2 fixed axes: {axis: {generator, ...}} for every held-out set."""
    from aigid import canon
    axes = {}
    for ds in canon.DATASETS.values():
        gens = getattr(ds, "held_out_generators", None)
        axis = getattr(ds, "axis", None)
        if gens and axis:
            axes.setdefault(axis, set()).update(gens)
    return axes


# ──────────────────────────────────────────────────────────────── main ───────
def main() -> int:
    p = argparse.ArgumentParser(
        description="Single final pass over test.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test",
                   help="normally test; val is allowed for a dry check")
    p.add_argument("--manifest", default=None)
    p.add_argument("--image-root", default=None)
    p.add_argument("--cap", type=int, default=0,
                   help="per-view cap; 0 = the whole split (the shipped run)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--weights-key", choices=["swa", "ema", "model"], default=None)
    p.add_argument("--no-tta", dest="tta", action="store_false",
                   help="match a predict.py run that also passes --no-tta")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--report", default=None,
                   help="write the JSON report here (default: results/test_<split>.json)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would run, touch no images")
    args = p.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    calibration = ckpt.get("calibration")
    weights_preference = calibrated_score_contract(
        ckpt, args.weights_key, args.tta, require=True)
    temperature = float(ckpt.get("calibration_temperature", 1.0))
    threshold = float(ckpt.get("threshold", 0.5))
    if temperature != float(calibration.get("temperature")):
        raise SystemExit("top-level temperature does not match calibration record")
    if threshold != float(calibration.get("threshold")):
        raise SystemExit("top-level threshold does not match calibration record")
    val_points = calibration.get("operating_points") or {}
    for key in ("fpr_0.01", "fpr_0.05"):
        if (not isinstance(val_points.get(key), dict)
                or val_points[key].get("threshold") is None):
            raise SystemExit(
                f"embedded calibration lacks required operating point {key}")

    # The whole point of this script is that it applies val's numbers unchanged.
    # An uncalibrated checkpoint means calibrate.py never ran, and every
    # probability, ECE and operating point below would describe a model that is
    # not the one being shipped -- so stop rather than emit a plausible table.
    if temperature == 1.0 and threshold == 0.5:
        raise SystemExit(
            "checkpoint is uncalibrated (T=1.0, threshold=0.5 are the "
            "placeholders aigid/train.py writes).\n"
            "Run scripts/calibrate.py on --split val first: §6.4's numbers are "
            "the val-fitted operating points applied to test, and there is "
            "nothing to apply yet.")

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    size = int((ckpt.get("args") or {}).get("img_size", 448))
    views = ["clean"] + list(GRID_CELL_NAMES)

    if args.dry_run:
        print(json.dumps({
            "checkpoint": args.checkpoint, "split": args.split,
            "temperature": temperature, "shipped_threshold": threshold,
            "views": views, "n_views": len(views), "img_size": size,
            "device": device, "tta": args.tta,
        }, indent=2))
        return 0

    model, weights_key = build_model(ckpt, device, weights_preference)
    print(f"scoring {args.split} · {len(views)} views · weights={weights_key} · "
          f"T={temperature:.4f} · thr={threshold:.4f}", file=sys.stderr)

    logits, labels, per_view = collect_logits(
        model, device, args.split, size, args.cap, args.manifest,
        args.workers, image_root=args.image_root,
        batch_size=args.batch_size, tta=args.tta)

    # Calibrated probabilities, exactly as predict.py ships them.
    def probs_of(lg):
        return torch.sigmoid(lg / temperature).numpy()

    clean_lg, clean_y = per_view["clean"]
    clean_p, clean_y = probs_of(clean_lg), clean_y.numpy()

    # ── clean ────────────────────────────────────────────────────────────────
    clean_auroc = auroc(clean_y, clean_p)
    pred = clean_p >= threshold
    pos, neg = clean_y == 1, clean_y == 0
    clean = {
        "auroc": clean_auroc,
        "auroc_ci95": bootstrap_ci(clean_y, clean_p, auroc, args.bootstrap),
        "average_precision": float(average_precision_score(clean_y, clean_p)),
        "balanced_accuracy": 0.5 * (float(pred[pos].mean())
                                    + float((~pred[neg]).mean())),
        "tpr_at_shipped_threshold": float(pred[pos].mean()),
        "fpr_at_shipped_threshold": float(pred[neg].mean()),
        "brier": brier(clean_y, clean_p),
        "ece": ece(clean_y, clean_p),
        "n": int(len(clean_y)),
    }

    # ── robustness grid ──────────────────────────────────────────────────────
    cells = {}
    cell_metrics = {}
    for view in GRID_CELL_NAMES:
        if view not in per_view:
            continue
        lg, y = per_view[view]
        yy = y.numpy()
        pp = probs_of(lg)
        cell_auroc = auroc(yy, pp)
        cells[view] = cell_auroc
        v_pos, v_neg = yy == 1, yy == 0
        shipped_pred = pp >= threshold
        cell_metrics[view] = {
            "auroc": cell_auroc,
            "auroc_drop_vs_clean": (
                clean_auroc - cell_auroc
                if cell_auroc is not None and clean_auroc is not None else None),
            "mean_prob_fake": float(pp[v_pos].mean()),
            "mean_prob_real": float(pp[v_neg].mean()),
            "tpr_at_shipped_threshold": float(shipped_pred[v_pos].mean()),
            "fpr_at_shipped_threshold": float(shipped_pred[v_neg].mean()),
            "balanced_accuracy_at_shipped_threshold":
                0.5 * (float(shipped_pred[v_pos].mean())
                       + float((~shipped_pred[v_neg]).mean())),
            "operating_points": {},
        }
        for op_key, vp in val_points.items():
            val_thr = vp.get("threshold") if isinstance(vp, dict) else None
            if val_thr is None:
                continue
            op_pred = pp >= float(val_thr)
            target = float(vp.get("target_fpr", op_key.removeprefix("fpr_")))
            cell_metrics[view]["operating_points"][op_key] = {
                "val_threshold": float(val_thr),
                "tpr": float(op_pred[v_pos].mean()),
                "fpr": float(op_pred[v_neg].mean()),
                "fpr_gap_vs_target": float(op_pred[v_neg].mean()) - target,
            }
    vals = [v for v in cells.values() if v is not None]
    worst_cell = min(cells, key=lambda k: cells[k]) if vals else None
    robust = {
        "cells": cells,
        # Macro over cells ONLY -- clean is reported separately and averaging it
        # in would dilute the number §6.2 selects on.
        "macro_robust_auroc": float(np.mean(vals)) if vals else None,
        "worst_cell": worst_cell,
        "worst_cell_auroc": cells[worst_cell] if worst_cell else None,
        "clean_to_robust_degradation":
            (clean_auroc - min(vals)) if (vals and clean_auroc) else None,
        "cell_metrics": cell_metrics,
    }

    # ── operating points: val-fitted, applied unchanged ──────────────────────
    # Thresholds come only from the checkpoint's embedded validation record.
    # This scorer never fits or derives a threshold from its input split.
    report_path = (Path(args.report) if args.report
                   else Path("results") / f"test_{args.split}.json")

    operating = {}
    for target in (0.01, 0.05):
        key = f"fpr_{target:g}"
        vp = val_points.get(key) or {}
        val_thr = vp.get("threshold")
        row = {"target_fpr": target, "val_threshold": val_thr}
        pr = clean_p >= float(val_thr)
        row["tpr_on_test"] = float(pr[pos].mean())
        row["fpr_achieved_on_test"] = float(pr[neg].mean())
        # Evaluation only: neither result is fed back into a threshold.
        row["fpr_gap_vs_target"] = row["fpr_achieved_on_test"] - target
        operating[key] = row

    # ── held-out generators, per §6.2's fixed axes ───────────────────────────
    # Slices are taken on the CLEAN view: the axes answer "does it transfer to
    # an unseen generator", which the grid would confound with "does it survive
    # a transform". Both questions are answered, separately.
    from aigid.data import BranchADataset
    ds = BranchADataset(args.split, size=size, train=False, distort=False,
                        manifest_path=args.manifest,
                        image_root=args.image_root)
    axes_report, axes_note = {}, None
    rows = ds.rows
    if len(rows) != len(clean_y):
        # collect_logits caps and reshuffles; without a row-for-row correspondence
        # a slice would be labelled with the wrong images. Say so instead.
        axes_note = (f"per-axis slices skipped: manifest has {len(rows)} rows but "
                     f"{len(clean_y)} were scored (--cap reshuffles). Re-run with "
                     "--cap 0 for the per-axis table.")
    else:
        gen = rows["generator"].to_numpy()
        for axis, gens in sorted(held_out_axes().items()):
            mask = np.isin(gen, list(gens))
            if not mask.any():
                axes_report[axis] = {"slice": axis, "n": 0,
                                     "note": "no rows for this axis in the split"}
                continue
            # A held-out axis has no reals of its own, so AUROC needs the split's
            # reals as the negative class -- otherwise every axis is single-class.
            sel = mask | (clean_y == 0)
            r = slice_rows(rows, sel, axis, clean_p, clean_y)
            r["generators"] = sorted(gens)
            r["n_fake_in_axis"] = int(mask.sum())
            r["note"] = ("AUROC computed against the split's reals; the axis "
                         "itself contributes fakes only")
            axes_report[axis] = r

        all_held = set().union(*held_out_axes().values())
        m = np.isin(gen, list(all_held))
        if m.any():
            r = slice_rows(rows, m | (clean_y == 0), "held_out_all",
                           clean_p, clean_y)
            r["n_fake_in_axis"] = int(m.sum())
            axes_report["held_out_all"] = r

    # ── named eval-only rows (§6.4) ──────────────────────────────────────────
    # Only rows whose data is actually in the build. Verified against the
    # published manifest on 31 Aug:
    #   * CIFAKE (thumbnail regime)  -- PRESENT, ~23.7k fake / ~23.9k real
    #   * SID_Set label 2 (tampered) -- ABSENT. Labels in the build are {0,1};
    #     canon.py:228 records label 2 as never ingested. No row is emitted
    #     rather than an empty one, and the report says why.
    #   * hard reals (screenshots/CGI/filtered) -- ABSENT. No such source was
    #     ever ingested; §3.1's rule was "if nothing clears licence review by
    #     29 Aug, the slice is dropped, not improvised". It was dropped.
    named_rows, named_absent = {}, {}
    if axes_note is None:
        src = rows["source"].to_numpy()
        cif = np.char.startswith(src.astype(str), "CIFAKE/")
        if cif.any():
            r = slice_rows(rows, cif, "thumbnail_regime_cifake", clean_p, clean_y)
            r["note"] = ("32x32 CIFAR-10 reals vs SD-1.4 synthetics upscaled to "
                         f"{size}. §6.4 expects a large drop: this is the extreme "
                         "end of the grid's 0.25x downscale cell, and the model "
                         "trains at full resolution. Reported as a bounded "
                         "limitation, not a headline number.")
            named_rows["thumbnail_regime_cifake"] = r
    named_absent["tampered_slice_sid_label2"] = (
        "no data: SID_Set label 2 was never ingested (canon.py:228); the build "
        "carries labels {0,1} only. The 'editing is out of scope' boundary is "
        "stated in §6.6 as an untested limitation, not measured.")
    named_absent["hard_reals"] = (
        "no data: no screenshot/CGI/render source cleared §3.1 licence review "
        "by the 29 Aug cutoff, so the slice was dropped per the pre-registered "
        "rule rather than improvised. §7's deployment page leads with the 1% "
        "FPR operating point instead.")

    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "weights_key": weights_key,
        "temperature": temperature,
        "shipped_threshold": threshold,
        "tta": args.tta,
        "cap": args.cap,
        "n_views": len(views),
        "clean": clean,
        "robustness": robust,
        "operating_points": operating,
        "held_out_axes": axes_report,
        "held_out_axes_note": axes_note,
        "named_rows": named_rows,
        "named_rows_absent": named_absent,
        "bootstrap_replicates": args.bootstrap,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nclean AUROC        {clean['auroc']:.4f}")
    if robust["worst_cell_auroc"] is not None:
        print(f"worst-cell AUROC   {robust['worst_cell_auroc']:.4f} "
              f"({robust['worst_cell']})")
        print(f"macro robust AUROC {robust['macro_robust_auroc']:.4f}")
    for k, v in operating.items():
        if "tpr_on_test" in v:
            print(f"{k:9s} TPR {v['tpr_on_test']:.4f} at FPR "
                  f"{v['fpr_achieved_on_test']:.4f} "
                  f"(target {v['target_fpr']:g})")
    print(f"\nreport -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
