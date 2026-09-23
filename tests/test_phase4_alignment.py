from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.alignment import (  # noqa: E402
    AlignmentObservation,
    AlignmentValidationError,
    alignment_gap,
    compare_alignment,
    cross_perturbation_accuracy,
    holm_step_down,
    orient_harm,
    paired_cluster_bootstrap,
    paired_contribution_sign_flip,
)
from rpe.evaluation import PreferredDirection  # noqa: E402


def _two_cluster_table(*, perfect: bool) -> tuple[AlignmentObservation, ...]:
    rows = []
    conditions = (
        ("p01", 0.1, 0.0, 0.0),
        ("p01", 0.2, 1.0, 1.0),
        ("p02", 0.1, 0.0 if not perfect else 2.0, 2.0),
        ("p02", 0.2, 1.0 if not perfect else 3.0, 3.0),
    )
    for cluster_id in ("c1", "c2"):
        for perturbation_id, alpha, metric_harm, downstream_harm in conditions:
            rows.append(
                AlignmentObservation(
                    cluster_id=cluster_id,
                    perturbation_id=perturbation_id,
                    alpha=alpha,
                    metric_harm=metric_harm,
                    downstream_harm=downstream_harm,
                )
            )
    return tuple(rows)


def _pair_category_table() -> tuple[AlignmentObservation, ...]:
    rows = []
    conditions = (
        ("p01", 0.1, 0.0, 0.0),
        ("p01", 0.2, 1.0, 1.0),
        ("p01", 0.3, 2.0, 2.0),
        ("p02", 0.1, 0.0, 0.0),
        ("p02", 0.2, 1.0, 2.0),
        ("p02", 0.3, 3.0, 1.0),
    )
    for cluster_id in ("c1", "c2"):
        rows.extend(
            AlignmentObservation(cluster_id, perturbation, alpha, metric, downstream)
            for perturbation, alpha, metric, downstream in conditions
        )
    return tuple(rows)


class SignedHarmTest(unittest.TestCase):
    def test_orients_lower_higher_and_target_outputs_exactly(self) -> None:
        self.assertEqual(
            orient_harm(2.0, 5.0, PreferredDirection.LOWER_IS_BETTER),
            3.0,
        )
        self.assertEqual(
            orient_harm(2.0, 5.0, PreferredDirection.HIGHER_IS_BETTER),
            -3.0,
        )
        self.assertEqual(
            orient_harm(
                3.0,
                8.0,
                PreferredDirection.TARGET_VALUE,
                target_value=5.0,
            ),
            1.0,
        )

    def test_rejects_nonmonotonic_nonfinite_or_missing_target(self) -> None:
        with self.assertRaisesRegex(AlignmentValidationError, "non_monotonic"):
            orient_harm(1.0, 2.0, PreferredDirection.NON_MONOTONIC)
        with self.assertRaisesRegex(AlignmentValidationError, "target_value"):
            orient_harm(1.0, 2.0, PreferredDirection.TARGET_VALUE)
        with self.assertRaisesRegex(AlignmentValidationError, "finite"):
            orient_harm(1.0, math.inf, PreferredDirection.LOWER_IS_BETTER)


class AlignmentGapTest(unittest.TestCase):
    def test_common_sst_gap_matches_hand_oracle(self) -> None:
        result = alignment_gap(_two_cluster_table(perfect=False))
        self.assertEqual(result.observation_count, 8)
        self.assertEqual(result.cluster_ids, ("c1", "c2"))
        self.assertEqual(result.perturbation_ids, ("p01", "p02"))
        self.assertEqual(result.alpha_grid, (0.1, 0.2))
        self.assertAlmostEqual(result.sst, 10.0, places=15)
        self.assertAlmostEqual(result.sse_pooled, 8.0, places=15)
        self.assertAlmostEqual(result.sse_separate, 0.0, places=15)
        self.assertAlmostEqual(result.r2_pooled, 0.2, places=15)
        self.assertAlmostEqual(result.r2_separate, 1.0, places=15)
        self.assertAlmostEqual(result.raw_alignment_gap, 0.8, places=15)
        self.assertAlmostEqual(result.alignment_gap, 0.8, places=15)
        self.assertEqual(
            result.pooled_predictions,
            (1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0),
        )
        self.assertEqual(
            result.separate_predictions,
            (0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0),
        )
        self.assertEqual(
            tuple(item.cluster_id for item in result.cluster_contributions),
            ("c1", "c2"),
        )
        for item in result.cluster_contributions:
            self.assertAlmostEqual(item.value, 0.4, places=15)
        self.assertAlmostEqual(
            sum(item.value for item in result.cluster_contributions),
            result.alignment_gap,
            places=15,
        )

    def test_perfect_candidate_and_paired_contrasts_match_hand_oracle(self) -> None:
        reference = _two_cluster_table(perfect=False)
        candidate = _two_cluster_table(perfect=True)
        comparison = compare_alignment(reference, candidate)
        self.assertAlmostEqual(comparison.reference_gap.alignment_gap, 0.8, places=15)
        self.assertAlmostEqual(comparison.candidate_gap.alignment_gap, 0.0, places=15)
        self.assertAlmostEqual(comparison.reference_accuracy.accuracy, 0.5, places=15)
        self.assertAlmostEqual(comparison.candidate_accuracy.accuracy, 1.0, places=15)
        self.assertAlmostEqual(comparison.d_ag, 0.8, places=15)
        self.assertAlmostEqual(comparison.d_acc, 0.5, places=15)
        self.assertEqual(
            tuple(item.value for item in comparison.ag_contribution_differences),
            (0.4, 0.4),
        )
        self.assertEqual(
            tuple(item.value for item in comparison.acc_contribution_differences),
            (0.5, 0.5),
        )

    def test_rejects_duplicate_incomplete_unequal_or_constant_tables(self) -> None:
        valid = _two_cluster_table(perfect=False)
        with self.assertRaisesRegex(AlignmentValidationError, "duplicate condition"):
            alignment_gap((*valid, valid[0]))
        with self.assertRaisesRegex(AlignmentValidationError, "complete Cartesian"):
            alignment_gap(valid[:-1])
        unequal = tuple(
            row
            for row in valid
            if not (row.perturbation_id == "p02" and row.alpha == 0.2)
        )
        with self.assertRaisesRegex(AlignmentValidationError, "alpha support"):
            alignment_gap(unequal)
        constant = tuple(replace(row, downstream_harm=1.0) for row in valid)
        with self.assertRaisesRegex(AlignmentValidationError, "constant downstream"):
            alignment_gap(constant)
        with self.assertRaisesRegex(AlignmentValidationError, "finite"):
            AlignmentObservation("c", "p", 0.1, math.nan, 0.0)


class CrossPerturbationAccuracyTest(unittest.TestCase):
    def test_pair_categories_and_equal_cluster_weight_match_hand_oracle(self) -> None:
        result = cross_perturbation_accuracy(_pair_category_table())
        self.assertEqual(result.pair_count, 18)
        self.assertEqual(result.strict_agreement_count, 8)
        self.assertEqual(result.strict_disagreement_count, 2)
        self.assertEqual(result.metric_tie_count, 2)
        self.assertEqual(result.downstream_tie_count, 4)
        self.assertEqual(result.double_tie_count, 2)
        self.assertAlmostEqual(result.accuracy, 2.0 / 3.0, places=15)
        for item in result.cluster_contributions:
            self.assertEqual(item.pair_count, 9)
            self.assertAlmostEqual(item.accuracy, 2.0 / 3.0, places=15)

    def test_mse_like_and_perfect_tables_have_half_and_one_accuracy(self) -> None:
        self.assertEqual(
            cross_perturbation_accuracy(_two_cluster_table(perfect=False)).accuracy,
            0.5,
        )
        self.assertEqual(
            cross_perturbation_accuracy(_two_cluster_table(perfect=True)).accuracy,
            1.0,
        )


class AlignmentInferenceTest(unittest.TestCase):
    def test_paired_cluster_bootstrap_recomputes_point_oracle_deterministically(self) -> None:
        reference = _two_cluster_table(perfect=False)
        candidate = _two_cluster_table(perfect=True)
        first = paired_cluster_bootstrap(reference, candidate)
        second = paired_cluster_bootstrap(reference, candidate)
        self.assertEqual(first, second)
        self.assertEqual(first.resamples, 2000)
        self.assertEqual(first.random_seed, 20260817)
        self.assertEqual(first.reference_ag_interval, (0.8, 0.8))
        self.assertEqual(first.candidate_ag_interval, (0.0, 0.0))
        self.assertEqual(first.d_ag_interval, (0.8, 0.8))
        self.assertEqual(first.reference_acc_interval, (0.5, 0.5))
        self.assertEqual(first.candidate_acc_interval, (1.0, 1.0))
        self.assertEqual(first.d_acc_interval, (0.5, 0.5))
        self.assertEqual(first.sample_size, 2)
        self.assertTrue(first.duplicate_cluster_resamples > 0)

    def test_contribution_sign_flip_matches_frozen_monte_carlo_oracle(self) -> None:
        for aggregation, contributions, observed in (
            ("sum", (0.4, 0.4), 0.8),
            ("mean", (0.5, 0.5), 0.5),
        ):
            with self.subTest(aggregation=aggregation):
                result = paired_contribution_sign_flip(
                    contributions,
                    aggregation=aggregation,
                )
                self.assertEqual(result.observed, observed)
                self.assertEqual(result.extreme_resamples, 49974)
                self.assertAlmostEqual(result.p_value, 0.4997450025499745, places=16)
                self.assertEqual(result.resamples, 100000)
                self.assertEqual(result.random_seed, 20260817)
                self.assertEqual(result.p_value_correction, "plus_one")

        zero = paired_contribution_sign_flip((0.0, 0.0), aggregation="mean")
        self.assertEqual(zero.p_value, 1.0)
        self.assertEqual(zero.extreme_resamples, 100000)
        for values, aggregation, expected in (
            ((math.nan,), "sum", "finite"),
            ((1.0,), "bad", "aggregation"),
        ):
            with self.assertRaisesRegex(AlignmentValidationError, expected):
                paired_contribution_sign_flip(values, aggregation=aggregation)

    def test_holm_step_down_matches_hand_values_and_strict_threshold(self) -> None:
        results = holm_step_down(
            {"h1": 0.01, "h2": 0.03, "h3": 0.04},
            alpha=0.05,
        )
        by_id = {result.hypothesis_id: result for result in results}
        self.assertEqual(by_id["h1"].adjusted_p_value, 0.03)
        self.assertEqual(by_id["h2"].adjusted_p_value, 0.06)
        self.assertEqual(by_id["h3"].adjusted_p_value, 0.06)
        self.assertTrue(by_id["h1"].rejected)
        self.assertFalse(by_id["h2"].rejected)
        self.assertFalse(by_id["h3"].rejected)
        exact = holm_step_down({"h": 0.05}, alpha=0.05)[0]
        self.assertEqual(exact.adjusted_p_value, 0.05)
        self.assertFalse(exact.rejected)
        for p_values, alpha, expected in (
            ({"h": math.nan}, 0.05, "p_values"),
            ({"h": 1.1}, 0.05, "p_values"),
            ({"h": 0.1}, 0.0, "alpha"),
        ):
            with self.assertRaisesRegex(AlignmentValidationError, expected):
                holm_step_down(p_values, alpha=alpha)


if __name__ == "__main__":
    unittest.main()
