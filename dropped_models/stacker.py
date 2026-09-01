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

Branch fusion: which combination of A, B and C ships, decided inside val.

WHAT THIS IS FOR
----------------
Three detectors -- A (DINOv2, the workhorse), B (frozen CLIP) and C (SRM
residual CNN) -- each emit one logit per image. The open question is which
combination to ship: A alone, A+B, A+C, or all three. `run_gate` answers it on
evidence by fitting a logistic blend

    fused = w_A*logit_A + w_B*logit_B + w_C*logit_C + bias

for every subset and comparing their worst-cell robust AUROC against A alone.
A combination ships only if it clears ``GATE_DELTA``; otherwise it is reported
as a negative result in the ablation table, which is a
finding, not a wasted branch.

THIS MODULE NEVER READS TEST, BY CONSTRUCTION
---------------------------------------------
It imports no torch and cannot score a model: it reads only exported bundles,
and every read passes ``require_split="val"``, so a test bundle is refused by
`dropped_models.bundles.load_bundle` on inspection of the file itself. Scoring -- the
only step that may touch test, and only for the single final M4 pass -- lives
in `dropped_models/eval_m3.py`. The separation is the point: a gate is a DECISION, and a
decision taken by reading test is model selection on the final split. Nor would
it be one look -- gating A+B, then A+C, then A+B+C queries test three times.

Val is therefore split by image-id hash into a **fit half** (the blend is
fitted here) and a **measure half** (every config is scored here, on images no
stacker saw). One shared measure half is what makes configs comparable to each
other.

WHAT THAT COSTS, STATED PLAINLY
-------------------------------
The original fusion idea asked for the delta on a held-out-generator validation
set. No such set exists in this build: all five held-out generators (GigaGAN,
starGAN, Imagen, VQDM, MAGE) are test-only under `assign_split`, and val holds
only trained-generator fakes.

So this gate measures **robustness transfer** (does the branch help on
distortions the blend was not fitted on?) and NOT **generator transfer** (does
it help on generators nobody trained on?). That is a genuinely weaker question,
and the ablation table must say so. It is the honest trade: the stronger
question can only be asked once, on test, and spending it on a ship/no-ship
decision would leave nothing to report.

STATUS: THE SHIPPED STACKER IS NOT YET IMPLEMENTED
--------------------------------------------------
`run_gate` chooses a config; `fit_final_stacker` (below) is the stub that must
persist one for inference. See its docstring for the remaining contract.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dropped_models.bundles import align_bundle, load_bundle

GATE_DELTA = 0.005


def split_val_halves(ids: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Partition val image ids into a fit half and a measure half.

    By hash of the image id, not by position: the manifest is label-sorted, so
    a positional split would put nearly all reals in one half and nearly all
    fakes in the other, and neither half could compute an AUROC at all.

    Hashing the id also makes the partition deterministic and bundle-order
    independent -- branch A, B and C bundles are aligned to a common id order
    before this is called, but even if they were not, every branch would place
    a given image in the same half. That is what makes "the stacker never saw
    these images" a property of the image rather than of the run.

    Returns (fit_idx, measure_idx) as positional indices into `ids`.
    """
    digests = np.asarray([
        int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:8], 16) % 2
        for value in ids.tolist()
    ])
    fit_idx = np.flatnonzero(digests == 0)
    measure_idx = np.flatnonzero(digests == 1)
    if fit_idx.size == 0 or measure_idx.size == 0:
        raise ValueError("val half-split produced an empty half")
    return fit_idx, measure_idx


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        raise ValueError("AUROC requires both real and fake labels")
    return float(roc_auc_score(labels, scores))


def _fit_stacker(features: np.ndarray, labels: np.ndarray, idx: np.ndarray):
    """Fit the logistic blend on the fit half only.

    `features` is (images, cells, branches). Rows are (image, cell) pairs, so
    one image contributes 15 rows -- clean plus the 14 transform cells -- and
    the fitted blend is shared across all of them. `np.repeat` matches that
    C-order flattening: labels must repeat per-cell within an image, not tile
    across images.
    """
    sub = features[idx]
    flat = sub.reshape(-1, sub.shape[-1])
    y = np.repeat(labels[idx], sub.shape[1])
    stacker = LogisticRegression(max_iter=2000, class_weight="balanced")
    stacker.fit(flat, y)
    return stacker


def _worst_cell(stacker, features: np.ndarray, labels: np.ndarray,
                idx: np.ndarray, cells: list) -> tuple[float, dict]:
    """Per-cell AUROC of the fused score on the measure half; worst over the
    13 distorted cells. `clean` is reported but excluded from the worst-cell
    figure, which is a robustness metric."""
    per_cell = {}
    for index, cell in enumerate(cells):
        fused = stacker.predict_proba(features[idx, index, :])[:, 1]
        per_cell[cell] = _auc(labels[idx], fused)
    robust = [per_cell[c] for c in cells if c != "clean"]
    return (min(robust) if robust else float("nan")), per_cell


def _baseline_worst(features: np.ndarray, labels: np.ndarray,
                    idx: np.ndarray, cells: list) -> tuple[float, dict]:
    """Branch A alone on the measure half -- the reference every config is
    scored against. Column 0 is A by construction (it is stacked first)."""
    per_cell = {}
    for index, cell in enumerate(cells):
        per_cell[cell] = _auc(labels[idx], features[idx, index, 0])
    robust = [per_cell[c] for c in cells if c != "clean"]
    return (min(robust) if robust else float("nan")), per_cell


def _bootstrap_delta(stacker, features, labels, idx, cells, base_worst,
                     rounds: int, seed: int) -> dict:
    """Percentile CI for (config worst-cell - A worst-cell), resampling IMAGES.

    Resampling images rather than (image, cell) rows is the point: the 15 rows
    an image contributes are not independent observations, and resampling them
    separately would understate the interval badly.
    """
    if rounds <= 0:
        return {}
    rng = np.random.default_rng(seed)
    robust_idx = [i for i, c in enumerate(cells) if c != "clean"]
    deltas = []
    for _ in range(rounds):
        pick = idx[rng.integers(0, len(idx), len(idx))]
        y = labels[pick]
        if len(np.unique(y)) < 2:
            continue
        worst_cfg, worst_base = np.inf, np.inf
        for index in robust_idx:
            fused = stacker.predict_proba(features[pick, index, :])[:, 1]
            worst_cfg = min(worst_cfg, roc_auc_score(y, fused))
            worst_base = min(worst_base, roc_auc_score(y, features[pick, index, 0]))
        deltas.append(worst_cfg - worst_base)
    if not deltas:
        return {}
    arr = np.asarray(deltas)
    return {"ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "rounds": int(arr.size)}


def run_gate(baseline_val_path: str, candidate_val_paths: list[str],
             output: str, required_delta: float = GATE_DELTA,
             seed: int = 0, bootstrap: int = 0) -> dict:
    """Compare every branch combination, entirely inside val.

    Enumerates each subset of the supplied candidates -- with two candidates
    that is A, A+B, A+C, A+B+C -- fits one stacker per subset on the val FIT
    half, and scores all of them on the one shared val MEASURE half. A shared
    measure half is what makes the configs comparable to each other; scoring
    each on its own sample would not be.

    Test is never read. `load_bundle(..., require_split="val")` refuses any
    bundle not stamped val, so that is enforced by the files themselves.

    Selection rule, fixed HERE rather than after seeing the numbers, because a
    tie-break invented once the table is on screen is a rationalisation:

      1. a config must clear `required_delta` over A alone to ship at all;
      2. among those that clear it, prefer the config whose delta CI does not
         overlap the next-best's -- and when they DO overlap, take the one with
         FEWER branches. Overlapping intervals mean the data cannot separate
         them, and the simpler model is cheaper to run and to explain.

    The winner is reported as `recommended`, with the reason stated.
    """
    # No candidates is legitimate and is the A-only path: there is nothing to
    # gate, but A's own per-cell numbers are still wanted for the report, and
    # `_baseline_worst` computes them without fitting anything. The loop below
    # degenerates to the single "A" entry.
    candidate_val_paths = list(candidate_val_paths)

    base = load_bundle(baseline_val_path, require_split="val")
    candidates = [align_bundle(base, load_bundle(path, require_split="val"), path)
                  for path in candidate_val_paths]

    names = ["A"] + [Path(p).stem for p in candidate_val_paths]
    features = np.stack([base["logits"], *(c["logits"] for c in candidates)],
                        axis=-1)
    labels = base["labels"]
    cells = base["cells"].tolist()

    fit_idx, measure_idx = split_val_halves(base["ids"], seed=seed)
    for half, idx in (("fit", fit_idx), ("measure", measure_idx)):
        if len(np.unique(labels[idx])) < 2:
            raise ValueError(f"val {half} half is single-class; cannot proceed")

    base_worst, base_cells = _baseline_worst(features, labels, measure_idx, cells)

    configs = []
    n_cand = len(candidates)
    for size in range(0, n_cand + 1):
        for combo in itertools.combinations(range(1, n_cand + 1), size):
            cols = [0, *combo]
            label = "+".join(names[c] for c in cols)
            sub = features[:, :, cols]
            if combo:
                stacker = _fit_stacker(sub, labels, fit_idx)
                worst, per_cell = _worst_cell(stacker, sub, labels,
                                              measure_idx, cells)
            else:
                # A alone: no blend to fit. A one-feature logistic regression
                # is a monotone rescale of the logit and cannot change AUROC,
                # so fitting one would only risk reporting a number that is
                # not what predict.py ships. Use the raw logit.
                stacker, (worst, per_cell) = None, (
                    base_worst, base_cells)
            delta = worst - base_worst
            entry = {
                "config": label,
                "branches": [names[c] for c in cols],
                "n_branches": len(cols),
                "worst_cell": worst,
                "delta_vs_A": delta,
                "passes": bool(delta >= required_delta) if combo else None,
                "per_cell": per_cell,
                "weights": (dict(zip((names[c] for c in cols),
                                     stacker.coef_[0].tolist()))
                            if stacker is not None else None),
                "intercept": (float(stacker.intercept_[0])
                              if stacker is not None else None),
            }
            if combo:
                entry.update(_bootstrap_delta(stacker, sub, labels, measure_idx,
                                              cells, base_worst, bootstrap, seed))
            configs.append(entry)

    ranked = sorted([c for c in configs if c["n_branches"] > 1],
                    key=lambda c: (-c["delta_vs_A"], c["n_branches"]))
    passing = [c for c in ranked if c["passes"]]
    if not candidates:
        recommended, reason = "A", (
            "no candidate branches supplied: A alone, with its per-cell "
            "numbers reported for the reference table")
    elif not passing:
        recommended, reason = "A", (
            f"no combination cleared +{required_delta} worst-cell over A alone "
            f"on the val measure half; ship A and report the rest as negative "
            f"results in the ablation table")
    else:
        best = passing[0]
        simpler = [c for c in passing if c["n_branches"] < best["n_branches"]]
        tie = None
        for c in simpler:
            if ("ci_low" in best and "ci_high" in c
                    and c["ci_high"] >= best["ci_low"]):
                tie = c
                break
        if tie is not None:
            recommended, reason = tie["config"], (
                f"{best['config']} scored higher ({best['delta_vs_A']:+.4f} vs "
                f"{tie['delta_vs_A']:+.4f}) but their bootstrap intervals "
                f"overlap, so the data cannot separate them; taking the config "
                f"with fewer branches")
        else:
            recommended, reason = best["config"], (
                f"highest worst-cell delta ({best['delta_vs_A']:+.4f}) among "
                f"configs clearing +{required_delta}")

    result = {
        "gate": "worst-cell robust AUROC delta, fitted and measured inside val",
        "split_discipline": (
            "stacker fitted on the val FIT half, all configs scored on the "
            "shared val MEASURE half; test is never read here and every "
            "ship/no-ship decision is frozen before it is opened"),
        "measures": ("robustness transfer (unseen distortions), NOT generator "
                     "transfer -- the held-out generators are test-only, so "
                     "generator transfer cannot be gated without spending test"),
        "required_delta": float(required_delta),
        "seed": int(seed),
        "val_fit_images": int(fit_idx.size),
        "val_measure_images": int(measure_idx.size),
        "baseline_worst_cell": base_worst,
        "baseline_per_cell": base_cells,
        "recommended": recommended,
        "recommendation_reason": reason,
        "configs": configs,
        "inputs": {"baseline_val": baseline_val_path,
                   "candidate_val": candidate_val_paths},
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result



# ─────────────────────────────────────────────── the shipped stacker ─────
def fit_final_stacker(baseline_val_path: str, candidate_val_paths: list[str],
                      output: str) -> dict:
    """Fit the blend that actually ships, on ALL of val, and persist it.

    Distinct from `run_gate` on purpose. The gate holds out a measure half
    because it needs an honest estimate of a config it has not seen; once the
    config is CHOSEN, that reason is gone and holding data back only makes the
    shipped coefficients worse. So the gate decides on half, and this refits
    the winner on all of val.

    Still val-only: fitting on test would be training on the final split.

    Writes {"branches", "coef", "intercept", "cells", "n_images"} as JSON --
    four floats for a three-branch blend, so there is nothing to checkpoint.

    REMAINING WORK, and it is on the critical path if any multi-branch config
    ships: `aigid/predict.py` loads ONE checkpoint and applies
    `sigmoid(logit / T)`. To ship a blend it must additionally (1) load every
    branch checkpoint named in `branches`, (2) score each image through all of
    them, (3) apply these coefficients, and (4) calibrate the FUSED score --
    the temperature in a single branch's checkpoint does not calibrate a blend,
    so `scripts/calibrate.py` must be re-run against the fused output. Until
    that exists, only the A-alone path is shippable end to end.
    """
    base = load_bundle(baseline_val_path, require_split="val")
    candidate_val_paths = list(candidate_val_paths)
    if not candidate_val_paths:
        # A alone: predict.py already ships sigmoid(logit_A / T) and there is
        # no blend to persist. Emit the identity so the shipped configuration
        # is recorded explicitly rather than by the absence of a file.
        payload = {
            "branches": ["A"], "coef": [1.0], "intercept": 0.0,
            "cells": base["cells"].tolist(),
            "n_images": int(len(base["labels"])),
            "fitted_on": "n/a -- single branch, no fusion",
            "note": ("A alone: predict.py applies sigmoid(logit / T) with the "
                     "temperature in the checkpoint. No stacker is used and "
                     "none needs loading at inference."),
        }
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2))
        return payload

    candidates = [align_bundle(base, load_bundle(p, require_split="val"), p)
                  for p in candidate_val_paths]
    names = ["A"] + [Path(p).stem for p in candidate_val_paths]

    features = np.stack([base["logits"], *(c["logits"] for c in candidates)],
                        axis=-1)
    labels = base["labels"]
    flat = features.reshape(-1, features.shape[-1])
    y = np.repeat(labels, features.shape[1])

    stacker = LogisticRegression(max_iter=2000, class_weight="balanced")
    stacker.fit(flat, y)

    payload = {
        "branches": names,
        "coef": stacker.coef_[0].tolist(),
        "intercept": float(stacker.intercept_[0]),
        "cells": base["cells"].tolist(),
        "n_images": int(len(labels)),
        "fitted_on": "val (all rows)",
        "note": ("apply as sigmoid(sum(coef * logits) + intercept), then "
                 "recalibrate the fused score -- a branch checkpoint's "
                 "temperature does not calibrate a blend"),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    return payload


# ─────────────────────────────────────────────────────────────── CLI ─────
def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gate = sub.add_parser(
        "gate", help="compare A / A+B / A+C / A+B+C inside val; test unread")
    gate.add_argument("--baseline-val", required=True,
                      help="branch A's val bundle (must be split=val)")
    gate.add_argument("--candidate-val", nargs="*", default=[],
                      help="candidate val bundles (B, C, or both); every "
                           "subset is compared. Omit for the A-only path: no "
                           "gate to run, but A's per-cell table is still "
                           "reported")
    gate.add_argument("--required-delta", type=float, default=GATE_DELTA,
                      help=f"worst-cell gain over A required to ship "
                           f"(default {GATE_DELTA})")
    gate.add_argument("--seed", type=int, default=0,
                      help="seed for the val fit/measure half-split")
    gate.add_argument("--bootstrap", type=int, default=1000,
                      help="bootstrap rounds for the delta CI; 0 disables. "
                           "The CI drives the simpler-config tie-break, so "
                           "disabling it makes near-ties resolve on raw rank")
    gate.add_argument("--output", required=True)

    final = sub.add_parser(
        "fit-final",
        help="refit the CHOSEN config on all of val, for inference")
    final.add_argument("--baseline-val", required=True)
    final.add_argument("--candidate-val", nargs="*", default=[],
                       help="only the branches the gate selected; omit for "
                            "the A-only path")
    final.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.command == "gate":
        result = run_gate(args.baseline_val, args.candidate_val, args.output,
                          args.required_delta, args.seed, args.bootstrap)
        # per_cell is 15 rows per config; the summary stays readable without it.
        summary = {k: v for k, v in result.items()
                   if k not in ("configs", "baseline_per_cell")}
        summary["configs"] = [{k: v for k, v in c.items() if k != "per_cell"}
                              for c in result["configs"]]
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(fit_final_stacker(
            args.baseline_val, args.candidate_val, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
