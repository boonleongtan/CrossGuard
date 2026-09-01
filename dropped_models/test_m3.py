from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dropped_models.branch_b import BranchB
from dropped_models.branch_c import BranchC
from dropped_models.bundles import CELLS, load_bundle, save_bundle
from dropped_models.stacker import fit_final_stacker, run_gate, split_val_halves


class TinyEncoder(nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.output_dim)
        self.dropout = nn.Dropout(0.9)

    def forward(self, x):
        return self.proj(x.mean(dim=(-2, -1)))


class BranchContractsTest(unittest.TestCase):
    def test_branch_b_freezes_encoder_and_keeps_it_in_eval(self):
        model = BranchB(encoder=TinyEncoder(), embed_dim=8)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.head.training)
        self.assertFalse(any(param.requires_grad for param in model.encoder.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.head.parameters()))
        logits, features = model(torch.randn(4, 3, 16, 16), return_features=True)
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertEqual(tuple(features.shape), (4, 8))
        self.assertTrue(torch.allclose(features.norm(dim=-1), torch.ones(4), atol=1e-5))

    def test_branch_b_small_checkpoint_round_trip(self):
        first = BranchB(encoder=TinyEncoder(), embed_dim=8)
        second = BranchB(encoder=TinyEncoder(), embed_dim=8)
        second.load_trainable_state_dict(first.trainable_state_dict())
        self.assertTrue(torch.equal(first.head.weight, second.head.weight))
        self.assertTrue(torch.equal(first.head.bias, second.head.bias))

    def test_branch_c_shape_and_fixed_srm_bank(self):
        model = BranchC(channels=(8, 16), depths=(1, 1), dropout=0.0)
        logits, features = model(torch.randn(2, 3, 64, 64), return_features=True)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(features.shape), (2, 16))
        self.assertEqual(len(list(model.srm.parameters())), 0)
        self.assertIn("weight", dict(model.srm.named_buffers()))


def _write_bundle(path: Path, ids, labels, logits, split="val"):
    save_bundle(path, np.asarray(ids), np.asarray(labels), np.asarray(CELLS),
                np.asarray(logits), "a", split)


class BundleGuardTest(unittest.TestCase):
    """The val-only contract is enforced by the FILE, not by convention."""

    def _bundle(self, root, name, split):
        n = 40
        labels = np.tile([0, 1], n // 2)
        _write_bundle(root / name, [f"i{i}" for i in range(n)], labels,
                      np.zeros((n, len(CELLS))), split=split)
        return root / name

    def test_test_split_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._bundle(Path(d), "t.npz", "test")
            load_bundle(str(path))                       # no requirement: fine
            with self.assertRaises(ValueError) as caught:
                load_bundle(str(path), require_split="val")
            self.assertIn("must not read test", str(caught.exception))

    def test_unstamped_bundle_is_refused_not_assumed_safe(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "u.npz"
            n = 40
            np.savez_compressed(path, ids=np.asarray([f"i{i}" for i in range(n)]),
                                labels=np.tile([0, 1], n // 2),
                                cells=np.asarray(CELLS),
                                logits=np.zeros((n, len(CELLS))))
            with self.assertRaises(ValueError) as caught:
                load_bundle(str(path), require_split="val")
            self.assertIn("no split stamp", str(caught.exception))


class HalfSplitTest(unittest.TestCase):
    def test_halves_are_disjoint_deterministic_and_order_independent(self):
        ids = np.asarray([f"img{i}" for i in range(500)])
        fit, measure = split_val_halves(ids)
        self.assertEqual(set(fit.tolist()) & set(measure.tolist()), set())
        self.assertEqual(len(fit) + len(measure), len(ids))
        self.assertTrue(np.array_equal(fit, split_val_halves(ids)[0]))
        # An image lands in the same half regardless of bundle row order.
        perm = np.random.default_rng(3).permutation(len(ids))
        fit_p, _ = split_val_halves(ids[perm])
        self.assertEqual(set(ids[fit].tolist()), set(ids[perm][fit_p].tolist()))


class GateTest(unittest.TestCase):
    def _make(self, root, n=400):
        rng = np.random.default_rng(7)
        cells = len(CELLS)
        labels = np.tile([0, 1], n // 2)
        signal = labels * 2.0 - 1.0
        base = signal[:, None] + rng.normal(0, 0.9, (n, cells))
        base[:, -1] = rng.normal(0, 1.0, n)      # A's failure cell
        strong = signal[:, None] + rng.normal(0, 0.45, (n, cells))
        weak = rng.normal(0, 1.0, (n, cells))    # pure noise: must not pass
        ids = [f"val-{i}" for i in range(n)]
        _write_bundle(root / "a.npz", ids, labels, base)
        rev = np.arange(n - 1, -1, -1)           # exercise id alignment
        _write_bundle(root / "strong.npz", np.asarray(ids)[rev], labels[rev],
                      strong[rev])
        _write_bundle(root / "weak.npz", ids, labels, weak)
        return root

    def test_gate_enumerates_every_subset_and_never_reads_test(self):
        with tempfile.TemporaryDirectory() as d:
            root = self._make(Path(d))
            result = run_gate(str(root / "a.npz"),
                              [str(root / "strong.npz"), str(root / "weak.npz")],
                              str(root / "gate.json"), bootstrap=50)
            self.assertEqual(len(result["configs"]), 4)      # A, A+s, A+w, A+s+w
            self.assertNotIn("test", json.dumps(result["inputs"]).lower())
            self.assertGreater(result["val_fit_images"], 0)
            self.assertGreater(result["val_measure_images"], 0)
            self.assertTrue((root / "gate.json").exists())

    def test_a_useful_branch_passes_and_a_noise_branch_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            root = self._make(Path(d))
            result = run_gate(str(root / "a.npz"),
                              [str(root / "strong.npz"), str(root / "weak.npz")],
                              str(root / "gate.json"), bootstrap=0)
            by_name = {c["config"]: c for c in result["configs"]}
            self.assertTrue(by_name["A+strong"]["passes"])
            self.assertFalse(by_name["A+weak"]["passes"])
            self.assertIn("strong", result["recommended"])

    def test_gate_refuses_a_test_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root = self._make(Path(d))
            n = 400
            _write_bundle(root / "bad.npz", [f"val-{i}" for i in range(n)],
                          np.tile([0, 1], n // 2),
                          np.zeros((n, len(CELLS))), split="test")
            with self.assertRaises(ValueError):
                run_gate(str(root / "a.npz"), [str(root / "bad.npz")],
                         str(root / "x.json"), bootstrap=0)


class FinalStackerTest(unittest.TestCase):
    def test_final_fit_uses_all_of_val_and_records_the_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            n = 200
            rng = np.random.default_rng(1)
            labels = np.tile([0, 1], n // 2)
            sig = labels[:, None] * 2.0 - 1.0
            ids = [f"v{i}" for i in range(n)]
            _write_bundle(root / "a.npz", ids, labels,
                          sig + rng.normal(0, 1.0, (n, len(CELLS))))
            _write_bundle(root / "c.npz", ids, labels,
                          sig + rng.normal(0, 0.5, (n, len(CELLS))))
            payload = fit_final_stacker(str(root / "a.npz"),
                                        [str(root / "c.npz")],
                                        str(root / "final.json"))
            self.assertEqual(payload["branches"], ["A", "c"])
            self.assertEqual(len(payload["coef"]), 2)
            self.assertEqual(payload["n_images"], n)     # ALL of val, not half
            self.assertIn("recalibrate", payload["note"])


if __name__ == "__main__":
    unittest.main()
