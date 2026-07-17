"""Regression tests for the reviewer-driven Access protocol changes.

Uses only the standard-library test runner:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from src.baselines import SpeakerTransitionMatrix, collect_shift_arrays
from src.dataset import (Dialogue, IGNORE_INDEX, apply_self_shift_target,
                         target_index_for_dialogue)
from src.evaluate import (hierarchical_bootstrap_auc,
                          paired_cluster_permutation_auc)
from src.holm_correction import holm_bonferroni
from src.models import CausalStateFactorizedForecaster, FutureWindowTransformer


def dialogue(labels, speakers, did="d"):
    return Dialogue(
        did=did,
        features={"text": np.arange(len(labels) * 3, dtype=np.float32).reshape(len(labels), 3)},
        labels=np.asarray(labels, dtype=np.int64),
        speakers=np.asarray(speakers, dtype=object),
    )


class TargetSemanticsTests(unittest.TestCase):
    def test_immediate_target_is_unchanged(self):
        d = dialogue([0, 0, 2, 1], ["a", "b", "a", "b"])
        np.testing.assert_array_equal(collect_shift_arrays([d])[2], [0, 1, 1])

    def test_next_own_target_and_index(self):
        d = dialogue([0, 1, 2, 1, 2], ["a", "b", "a", "b", "a"])
        apply_self_shift_target([d])
        np.testing.assert_array_equal(
            d.custom_shift_targets, [1, 0, 0, IGNORE_INDEX, IGNORE_INDEX])
        self.assertEqual(target_index_for_dialogue(d, 0), 2)
        self.assertEqual(target_index_for_dialogue(d, 1), 3)
        self.assertEqual(target_index_for_dialogue(d, 2), 4)

    def test_transition_fit_uses_custom_future_index(self):
        d = dialogue([0, 1, 0, 1], ["a", "b", "a", "b"])
        apply_self_shift_target([d])
        model = SpeakerTransitionMatrix(num_emotions=2, alpha=1.0, min_count=0).fit([d])
        # a: 0 -> 0 and b: 1 -> 1; adjacent targets would instead record flips.
        self.assertEqual(model.global_counts[0, 0], 1)
        self.assertEqual(model.global_counts[1, 1], 1)
        self.assertEqual(model.global_counts[0, 1], 0)


class InferenceTests(unittest.TestCase):
    def test_identical_scores_give_zero_delta_and_unit_p(self):
        y = np.asarray([0, 1, 0, 1, 0, 1])
        scores = np.asarray([[.1, .9, .2, .8, .3, .7],
                             [.2, .8, .3, .7, .4, .6]])
        dids = np.asarray(["a", "a", "b", "b", "c", "c"])
        ci = hierarchical_bootstrap_auc(y, scores, scores, dids, n_boot=49, seed=2)
        perm = paired_cluster_permutation_auc(y, scores, scores, dids, n_perm=49, seed=2)
        self.assertEqual(ci["delta_auc"], 0.0)
        self.assertEqual(ci["ci_low"], 0.0)
        self.assertEqual(ci["ci_high"], 0.0)
        self.assertEqual(perm["p_value"], 1.0)

    def test_holm_adjusted_values_are_monotone(self):
        out = holm_bonferroni({"a": .001, "b": .01, "c": .04})
        ordered = sorted(out.values(), key=lambda row: row["rank"])
        adjusted = [row["p_holm"] for row in ordered]
        self.assertEqual(adjusted, sorted(adjusted))
        self.assertAlmostEqual(out["a"]["p_holm"], .003)


class LeakageMaskTests(unittest.TestCase):
    def test_k0_is_invariant_to_future_perturbation(self):
        torch.manual_seed(4)
        model = FutureWindowTransformer(5, hidden=16, heads=4, layers=2,
                                        dropout=0.0, k_future=0).eval()
        x = torch.randn(1, 6, 5)
        altered = x.clone()
        altered[:, 4:, :] = torch.randn_like(altered[:, 4:, :]) * 20
        with torch.no_grad():
            before = model(x)[0, :4]
            after = model(altered)[0, :4]
        torch.testing.assert_close(before, after, rtol=0.0, atol=1e-6)

    def test_state_factorized_heads_are_causal_and_probability_is_bounded(self):
        torch.manual_seed(9)
        model = CausalStateFactorizedForecaster(
            5, num_emotions=4, hidden=12, dropout=0.0).eval()
        x = torch.randn(1, 7, 5)
        altered = x.clone(); altered[:, 5:] += 50
        with torch.no_grad():
            s1, c1, f1 = model.forward_all(x)
            s2, c2, f2 = model.forward_all(altered)
            factor = 1 - (torch.softmax(c1, -1) * torch.softmax(f1, -1)).sum(-1)
        torch.testing.assert_close(s1[:, :5], s2[:, :5], rtol=0.0, atol=1e-6)
        torch.testing.assert_close(c1[:, :5], c2[:, :5], rtol=0.0, atol=1e-6)
        torch.testing.assert_close(f1[:, :5], f2[:, :5], rtol=0.0, atol=1e-6)
        self.assertTrue(bool(torch.all((factor >= 0) & (factor <= 1))))


if __name__ == "__main__":
    unittest.main()
