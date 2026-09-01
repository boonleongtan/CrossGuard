#!/usr/bin/env python
"""Turn graded CrossGuard predictions into a deployment triage sidecar.

The graded submission output is deliberately minimal:

    [{"image_path": "...", "pred": 0.8734}, ...]

This script leaves that file untouched and writes a separate JSON artifact for
the demo/deployment story. The policy thresholds come from the validation-fitted
calibration report, not from the final test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CALIBRATION = Path("runs/ft_full_calibrated/calibration.json")


def default_output_path(predictions: Path) -> Path:
    if predictions.suffix:
        return predictions.with_suffix(predictions.suffix + ".triage.json")
    return predictions.with_name(predictions.name + ".triage.json")


def load_policy(calibration_path: Path) -> dict:
    payload = json.loads(calibration_path.read_text(encoding="utf8"))
    points = payload.get("operating_points") or {}
    strict = points.get("fpr_0.01") or {}
    broad = points.get("fpr_0.05") or {}
    action = strict.get("threshold")
    review = broad.get("threshold")
    if action is None or review is None:
        raise SystemExit(
            f"{calibration_path} lacks fpr_0.01/fpr_0.05 operating thresholds")

    action = float(action)
    review = float(review)
    if not (0.0 <= review <= action <= 1.0):
        raise SystemExit(
            "expected fpr_0.05 threshold <= fpr_0.01 threshold within [0, 1]")

    return {
        "source": str(calibration_path),
        "review_floor_threshold": review,
        "action_threshold": action,
        "review_floor_name": "5% FPR validation operating point",
        "action_name": "1% FPR validation operating point",
    }


def classify_score(score: float, policy: dict) -> dict:
    action = float(policy["action_threshold"])
    review = float(policy["review_floor_threshold"])
    if score >= action:
        return {
            "decision": "likely_aigc",
            "band": "above_1pct_fpr_threshold",
            "reason": "score exceeds the strict 1% FPR operating threshold",
        }
    if score >= review:
        return {
            "decision": "uncertain_review",
            "band": "between_5pct_and_1pct_fpr_thresholds",
            "reason": "score sits in the calibrated abstention band",
        }
    return {
        "decision": "lower_risk",
        "band": "below_5pct_fpr_threshold",
        "reason": "score is below the broader 5% FPR review threshold",
    }


def load_predictions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(rows, list):
        raise SystemExit("predictions file must be a JSON array")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "image_path" not in row or "pred" not in row:
            raise SystemExit(f"row {i} must contain image_path and pred")
        score = float(row["pred"])
        if not 0.0 <= score <= 1.0:
            raise SystemExit(f"row {i} pred is outside [0, 1]: {score}")
    return rows


def build_sidecar(prediction_path: Path, calibration_path: Path) -> dict:
    policy = load_policy(calibration_path)
    rows = []
    summary = {"likely_aigc": 0, "uncertain_review": 0, "lower_risk": 0}
    for row in load_predictions(prediction_path):
        score = float(row["pred"])
        decision = classify_score(score, policy)
        summary[decision["decision"]] += 1
        rows.append({
            "image_path": row["image_path"],
            "pred": score,
            **decision,
        })

    return {
        "source_predictions": str(prediction_path),
        "policy": policy,
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path,
                        help="graded predictions JSON from aigid.predict")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                        help="validation-fitted calibration report")
    parser.add_argument("--output", type=Path,
                        help="triage sidecar path; defaults beside predictions")
    args = parser.parse_args()

    out = args.output or default_output_path(args.predictions)
    sidecar = build_sidecar(args.predictions, args.calibration)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sidecar, indent=2), encoding="utf8")
    print(f"wrote triage sidecar -> {out}")
    print("summary:", json.dumps(sidecar["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
