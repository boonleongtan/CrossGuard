"""Safety tests for calibration only; no dataset split is opened."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigid.predict import calibrated_score_contract
from scripts.calibrate import atomic_write_json, brier, main as calibrate_main


class CalibrationSafetyTest(unittest.TestCase):
    def _argv(self, *extra: str) -> list[str]:
        return ["calibrate", "--checkpoint", "source.pt",
                "--out", "calibrated.pt", *extra]

    def test_only_validation_split_is_accepted(self):
        for split in ("train", "test", "validation", "VAL"):
            with self.subTest(split=split), mock.patch.object(
                    sys, "argv", self._argv("--split", split)):
                with self.assertRaisesRegex(SystemExit, "validation-only"):
                    calibrate_main()

    def test_negative_cap_is_rejected_before_checkpoint_access(self):
        with mock.patch.object(
                sys, "argv", self._argv("--split", "val", "--cap", "-1")):
            with self.assertRaisesRegex(SystemExit, "--cap must be >= 0"):
                calibrate_main()

    def test_brier_score_known_value(self):
        probs = np.array([0.0, 0.25, 0.75, 1.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(brier(probs, labels), 0.03125)

    def test_json_write_is_atomic_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            atomic_write_json(path, {"status": "accepted"})
            self.assertEqual(json.loads(path.read_text()),
                             {"status": "accepted"})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_score_contract_rejects_weights_and_tta_drift(self):
        calibration = {
            "schema_version": 1, "status": "accepted",
            "artifact_id": "abc", "split": "val",
            "weights_key": "ema", "tta": True,
            "temperature": 1.2, "threshold": 0.4,
        }
        ckpt = {
            "calibration": calibration,
            "calibration_temperature": 1.2,
            "threshold": 0.4,
        }
        self.assertEqual(calibrated_score_contract(ckpt, None, True, True),
                         "ema")
        with self.assertRaisesRegex(SystemExit, "conflicts"):
            calibrated_score_contract(ckpt, "model", True, True)
        with self.assertRaisesRegex(SystemExit, "TTA setting conflicts"):
            calibrated_score_contract(ckpt, None, False, True)

    def test_score_contract_requires_calibration_when_shipping(self):
        self.assertIsNone(calibrated_score_contract({}, None, True, False))
        with self.assertRaisesRegex(SystemExit, "no embedded accepted calibration"):
            calibrated_score_contract({}, None, True, True)

    def test_scorer_source_contains_no_threshold_refit(self):
        source = (Path(__file__).resolve().parents[1]
                  / "scripts" / "score_test.py").read_text(encoding="utf8")
        ast.parse(source)
        self.assertNotIn("threshold_at_fpr", source)
        self.assertNotIn("refit_on_test_would_be", source)


if __name__ == "__main__":
    unittest.main()
