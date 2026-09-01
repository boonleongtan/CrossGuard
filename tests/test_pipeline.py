"""Tests for the shipped single-backbone path.

These cover the claims the write-up actually makes, in the order they are
load-bearing:

  1. The test set is UNSEEN. Every held-out generator must resolve to test and
     to nothing else, for every possible content hash. This is the claim the
     whole evaluation rests on; if it breaks, the headline number is fiction.
  2. The robustness grid is stable. Cell names and count are a published
     contract (14 cells + clean), and scoring code indexes them positionally.
  3. Calibration arithmetic is correct on inputs whose answer is known by
     construction -- including the saturated-val case that this project
     actually has.

Nothing here needs a GPU, a checkpoint, or the dataset.
"""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aigid import canon
from aigid.distort import GRID_CELL_NAMES
from scripts.calibrate import (best_threshold, ece, fit_temperature,
                               threshold_at_fpr, threshold_curve)
import torch


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


# Each WildFake archive is its own dataset key, and each holds out its own
# generators. Pair them so a test asks the registry the question it can answer.
HELD_OUT = [
    ("WildFake_GAN", canon.WILDFAKE_HELD_OUT),
    ("WildFake_Diffusion", canon.WILDFAKE_DIFF_HELD_OUT),
    ("WildFake_Other", canon.WILDFAKE_OTHER_HELD_OUT),
]
TRAINED = [
    ("WildFake_GAN", canon.WILDFAKE_TRAINED),
    ("WildFake_Diffusion", canon.WILDFAKE_DIFF_TRAINED),
]


class UnseenGeneratorTest(unittest.TestCase):
    """The test set's defining property, checked exhaustively over hashes."""

    def test_held_out_generators_never_leave_test(self):
        for dataset, generators in HELD_OUT:
            for generator in sorted(generators):
                for i in range(500):
                    split = canon.assign_split(dataset, "train",
                                               _hash(f"{generator}/{i}"),
                                               label=1, generator=generator)
                    self.assertEqual(split, "test",
                                     f"{generator} leaked into {split!r} at hash {i}")

    def test_trained_generators_never_reach_test(self):
        """The converse: a generator the model trained on must not pad the
        test set, or 'unseen-generator AUROC' would be measuring seen ones."""
        for dataset, generators in TRAINED:
            for generator in sorted(generators):
                for i in range(300):
                    split = canon.assign_split(dataset, "train",
                                               _hash(f"{generator}/{i}"),
                                               label=1, generator=generator)
                    self.assertIn(split, ("train", "val"),
                                  f"trained generator {generator} reached {split!r}")

    def test_held_out_and_trained_sets_are_disjoint(self):
        held = frozenset().union(*(g for _, g in HELD_OUT))
        trained = frozenset().union(*(g for _, g in TRAINED))
        self.assertEqual(held & trained, frozenset())

    def test_assignment_is_deterministic(self):
        """Re-running ingestion must not reshuffle the splits."""
        for generator in ("WildFake/GigaGAN", "WildFake/DF-GAN"):
            h = _hash(f"{generator}/stable")
            first = canon.assign_split("WildFake_GAN", "train", h, 1, generator)
            for _ in range(5):
                self.assertEqual(
                    canon.assign_split("WildFake_GAN", "train", h, 1, generator),
                    first)

    def test_reals_are_split_by_content_hash_not_generator(self):
        """Reals have no generator; they must still spread across all three."""
        seen = {canon.assign_split("WildFake_GAN", "train", _hash(f"real/{i}"),
                                   0, "")
                for i in range(2000)}
        self.assertEqual(seen, {"train", "val", "test"})

    def test_upstream_holdout_splits_into_both_val_and_test(self):
        """A source contributing no test rows would silently change what the
        test set is made of -- the SID_Set/COCO failure the docstring cites."""
        seen = {canon.assign_split("SID_Set", "validation", _hash(f"sid/{i}"))
                for i in range(500)}
        self.assertEqual(seen, {"val", "test"})

    def test_unknown_upstream_split_is_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            canon.assign_split("SID_Set", "nonsense", _hash("x"))


class RobustnessGridTest(unittest.TestCase):
    def test_grid_is_the_published_contract(self):
        self.assertEqual(len(GRID_CELL_NAMES), 14)
        self.assertEqual(len(set(GRID_CELL_NAMES)), len(GRID_CELL_NAMES))

    def test_clean_is_not_a_grid_cell(self):
        """'clean' is reported alongside the grid, never inside it; if it were
        a cell, worst-cell robust AUROC could be satisfied by the easy case."""
        self.assertNotIn("clean", GRID_CELL_NAMES)


class CalibrationMathTest(unittest.TestCase):
    def test_temperature_scales_with_overconfidence(self):
        """T is a monotone response to how peaked the logits are. Sharper
        logits for the same separation must ask for more damping."""
        rng = np.random.default_rng(0)
        n = 4000
        labels = rng.integers(0, 2, n)
        signal = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 1.0, n)
        temps = [fit_temperature(torch.tensor(scale * signal, dtype=torch.float64),
                                 torch.tensor(labels, dtype=torch.float64))
                 for scale in (1.0, 2.0, 4.0)]
        self.assertTrue(all(a < b for a, b in zip(temps, temps[1:])),
                        f"temperature not monotone in logit scale: {temps}")
        self.assertTrue(all(t > 0 for t in temps))

    def test_overconfident_logits_get_temperature_above_one(self):
        rng = np.random.default_rng(1)
        n = 4000
        labels = rng.integers(0, 2, n)
        logits = 4.0 * (np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 1.0, n))
        t = fit_temperature(torch.tensor(logits, dtype=torch.float64),
                            torch.tensor(labels, dtype=torch.float64))
        self.assertGreater(t, 1.5)

    def test_ece_is_zero_for_perfectly_calibrated_probabilities(self):
        rng = np.random.default_rng(2)
        probs = rng.uniform(0, 1, 200_000)
        labels = (rng.uniform(0, 1, probs.size) < probs).astype(int)
        self.assertLess(ece(probs, labels), 0.01)

    def test_threshold_at_fpr_hits_its_target(self):
        rng = np.random.default_rng(3)
        n = 20_000
        labels = np.tile([0, 1], n // 2)
        probs = np.clip(np.where(labels == 1, 0.75, 0.25)
                        + rng.normal(0, 0.15, n), 0, 1)
        thr, achieved, tpr, reached = threshold_at_fpr(probs, labels, 0.05)
        self.assertTrue(reached)
        self.assertLessEqual(achieved, 0.05 + 1e-9)
        self.assertGreater(tpr, 0.5)

    def test_unreachable_fpr_is_flagged_not_silently_reported(self):
        """Reals scoring exactly 1.0 make a low FPR unreachable. Publishing an
        FPR the model does not achieve is the failure this flag exists for."""
        labels = np.array([0] * 100 + [1] * 100)
        probs = np.concatenate([np.full(100, 1.0), np.full(100, 1.0)])
        thr, achieved, tpr, reached = threshold_at_fpr(probs, labels, 0.001)
        self.assertFalse(reached)

    def test_plateau_is_measured_on_observed_scores_not_the_grid(self):
        """The saturated-val case this project has: a wide gap between classes
        must report a WIDE plateau, not 0.0 because only one grid candidate
        falls in the gap."""
        labels = np.array([0] * 500 + [1] * 500)
        probs = np.concatenate([np.full(500, 0.01), np.full(500, 0.99)])
        centre, ba, plateau = best_threshold(probs, labels)
        self.assertAlmostEqual(ba, 1.0, places=6)
        self.assertGreater(plateau["width"], 0.5)
        self.assertGreater(centre, plateau["lo"])
        self.assertLess(centre, plateau["hi"])
        # The centre is the shipped threshold precisely because the argmax is
        # an arbitrary point hugging one edge of a wide plateau.
        self.assertAlmostEqual(centre, 0.5, places=6)

    def test_threshold_curve_is_monotone_in_the_right_directions(self):
        rng = np.random.default_rng(4)
        n = 5000
        labels = np.tile([0, 1], n // 2)
        probs = np.clip(np.where(labels == 1, 0.7, 0.3)
                        + rng.normal(0, 0.2, n), 0, 1)
        curve = threshold_curve(probs, labels, points=51)
        tpr = [row["tpr"] for row in curve]
        fpr = [row["fpr"] for row in curve]
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(tpr, tpr[1:])))
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(fpr, fpr[1:])))


if __name__ == "__main__":
    unittest.main()
