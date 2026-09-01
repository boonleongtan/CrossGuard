from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.triage_predictions import (build_sidecar, classify_score,
                                        default_output_path, load_policy)


class TriagePredictionsTest(unittest.TestCase):
    def policy(self) -> dict:
        return {"review_floor_threshold": 0.4, "action_threshold": 0.9}

    def test_default_output_path_keeps_graded_file_untouched(self):
        self.assertEqual(default_output_path(Path("predictions.json")),
                         Path("predictions.json.triage.json"))
        self.assertEqual(default_output_path(Path("predictions")),
                         Path("predictions.triage.json"))

    def test_score_bands_match_deployment_policy(self):
        self.assertEqual(classify_score(0.95, self.policy())["decision"],
                         "likely_aigc")
        self.assertEqual(classify_score(0.6, self.policy())["decision"],
                         "uncertain_review")
        self.assertEqual(classify_score(0.2, self.policy())["decision"],
                         "lower_risk")

    def test_load_policy_reads_validation_operating_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({
                "operating_points": {
                    "fpr_0.01": {"threshold": 0.9},
                    "fpr_0.05": {"threshold": 0.4},
                }
            }), encoding="utf8")
            policy = load_policy(path)
        self.assertEqual(policy["action_threshold"], 0.9)
        self.assertEqual(policy["review_floor_threshold"], 0.4)

    def test_build_sidecar_summarizes_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "predictions.json"
            cal = root / "calibration.json"
            pred.write_text(json.dumps([
                {"image_path": "a.jpg", "pred": 0.95},
                {"image_path": "b.jpg", "pred": 0.6},
                {"image_path": "c.jpg", "pred": 0.2},
            ]), encoding="utf8")
            cal.write_text(json.dumps({
                "operating_points": {
                    "fpr_0.01": {"threshold": 0.9},
                    "fpr_0.05": {"threshold": 0.4},
                }
            }), encoding="utf8")
            sidecar = build_sidecar(pred, cal)
        self.assertEqual(sidecar["summary"], {
            "likely_aigc": 1,
            "uncertain_review": 1,
            "lower_risk": 1,
        })
        self.assertEqual(sidecar["rows"][1]["band"],
                         "between_5pct_and_1pct_fpr_thresholds")


if __name__ == "__main__":
    unittest.main()
