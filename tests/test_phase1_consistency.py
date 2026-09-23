from __future__ import annotations

import math
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import (  # noqa: E402
    AxisPolicy,
    EvaluationContractError,
    MetricInputKind,
    PreferredDirection,
    ReplicatePairInput,
    Spectrum1D,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    ConsistencyMetricError,
    HalfSplitPearsonConsistencyMetric,
)


def spectrum(
    spectrum_id: str,
    *,
    axis: tuple[float, ...],
    intensity: tuple[float, ...],
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.asarray(axis, dtype="<f8"),
        intensity=np.asarray(intensity, dtype="<f8"),
    )


class HalfSplitPearsonConsistencyMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = HalfSplitPearsonConsistencyMetric()
        self.left = spectrum(
            "left-replicate",
            axis=(100.0, 200.0, 300.0, 400.0),
            intensity=(1.0, 2.0, 3.0, 4.0),
        )
        self.right = spectrum(
            "right-replicate",
            axis=(100.0, 200.0, 300.0, 400.0),
            intensity=(2.0, 0.0, 3.0, 8.0),
        )
        self.request = ReplicatePairInput(self.left, self.right)

    def test_constructor_identity_matches_design(self) -> None:
        self.assertEqual(
            self.metric.metric_id,
            "half_split_pearson_consistency",
        )
        self.assertEqual(
            self.metric.output_id,
            "half_split_pearson_consistency",
        )
        self.assertEqual(self.metric.unit, "correlation")
        self.assertEqual(
            self.metric.input_kind,
            MetricInputKind.REPLICATE_PAIR,
        )
        self.assertEqual(self.metric.axis_policy, AxisPolicy.EXACT_AXIS)
        self.assertEqual(
            self.metric.preferred_direction,
            PreferredDirection.HIGHER_IS_BETTER,
        )

    def test_wrapper_execution_matches_literal_oracle_and_diagnostics(self) -> None:
        result = evaluate_metric(self.metric, self.request)

        self.assertEqual(
            result.metric_id,
            "half_split_pearson_consistency",
        )
        self.assertEqual(result.input_kind, MetricInputKind.REPLICATE_PAIR)
        self.assertEqual(len(result.outputs), 1)

        output = result.outputs[0]
        self.assertEqual(
            output.output_id,
            "half_split_pearson_consistency",
        )
        self.assertEqual(output.unit, "correlation")
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.HIGHER_IS_BETTER,
        )
        self.assertIsNone(output.target_value)
        self.assertAlmostEqual(
            output.value,
            21.0 / math.sqrt(695.0),
            places=15,
        )
        self.assertEqual(
            dict(result.diagnostics),
            {
                "affine_invariant": True,
                "axis_equal": True,
                "consistency_definition": "exact_axis_population_pearson_r",
                "left_spectrum_id": "left-replicate",
                "point_count": 4,
                "right_spectrum_id": "right-replicate",
            },
        )

    def test_positive_affine_transform_returns_one(self) -> None:
        request = ReplicatePairInput(
            spectrum(
                "left",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(1.0, 2.0, 3.0, 4.0),
            ),
            spectrum(
                "right",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(3.0, 5.0, 7.0, 9.0),
            ),
        )
        result = evaluate_metric(self.metric, request)
        self.assertEqual(result.outputs[0].value, 1.0)

    def test_perfect_reversal_returns_minus_one(self) -> None:
        request = ReplicatePairInput(
            spectrum(
                "left",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(1.0, 2.0, 3.0, 4.0),
            ),
            spectrum(
                "right",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(4.0, 3.0, 2.0, 1.0),
            ),
        )
        result = evaluate_metric(self.metric, request)
        self.assertEqual(result.outputs[0].value, -1.0)

    def test_large_common_offset_regression_returns_one(self) -> None:
        request = ReplicatePairInput(
            spectrum(
                "left",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    1.0e16,
                    1.0e16 + 2.0,
                    1.0e16 + 4.0,
                    1.0e16 + 8.0,
                ),
            ),
            spectrum(
                "right",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    1.0e16 + 10.0,
                    1.0e16 + 14.0,
                    1.0e16 + 18.0,
                    1.0e16 + 26.0,
                ),
            ),
        )
        result = evaluate_metric(self.metric, request)
        self.assertAlmostEqual(result.outputs[0].value, 1.0, places=15)

    def test_cross_zero_extreme_affine_regression_returns_one_without_warning(self) -> None:
        request = ReplicatePairInput(
            spectrum(
                "left",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    -1.0e308,
                    -5.0e307,
                    5.0e307,
                    1.0e308,
                ),
            ),
            spectrum(
                "right",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    -2.0e307,
                    -1.0e307,
                    1.0e307,
                    2.0e307,
                ),
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = evaluate_metric(self.metric, request)
        self.assertAlmostEqual(result.outputs[0].value, 1.0, places=15)

    def test_uniform_subnormal_scaling_matches_existing_pearson_behavior(self) -> None:
        request = ReplicatePairInput(
            spectrum(
                "left",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    1.0e-320,
                    2.0e-320,
                    3.0e-320,
                    4.0e-320,
                ),
            ),
            spectrum(
                "right",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    3.0e-320,
                    5.0e-320,
                    7.0e-320,
                    9.0e-320,
                ),
            ),
        )
        result = evaluate_metric(self.metric, request)
        self.assertEqual(result.outputs[0].value, 1.0)

    def test_wrapper_rejects_exact_axis_mismatch(self) -> None:
        request = ReplicatePairInput(
            self.left,
            spectrum(
                "right-shifted",
                axis=(100.5, 200.5, 300.5, 400.5),
                intensity=(1.0, 2.0, 3.0, 4.0),
            ),
        )
        with self.assertRaisesRegex(
            EvaluationContractError,
            "^replicate exact axis: replicates must have exactly equal axes$",
        ):
            evaluate_metric(self.metric, request)

    def test_direct_evaluate_rejects_wrong_request_type(self) -> None:
        with self.assertRaisesRegex(
            ConsistencyMetricError,
            "^request: must be ReplicatePairInput$",
        ):
            self.metric.evaluate(object())

    def test_either_constant_input_rejects(self) -> None:
        varying = spectrum(
            "varying",
            axis=(10.0, 20.0, 30.0, 40.0),
            intensity=(1.0, 2.0, 3.0, 4.0),
        )
        constant = spectrum(
            "constant",
            axis=(10.0, 20.0, 30.0, 40.0),
            intensity=(5.0, 5.0, 5.0, 5.0),
        )
        cases = (
            (
                ReplicatePairInput(constant, varying),
                "left zero variance",
            ),
            (
                ReplicatePairInput(varying, constant),
                "right zero variance",
            ),
        )
        for request, expected_path in cases:
            with self.subTest(path=expected_path):
                with self.assertRaisesRegex(
                    ConsistencyMetricError,
                    f"^{expected_path}: zero variance$",
                ):
                    evaluate_metric(self.metric, request)

    def test_evaluation_does_not_mutate_input_arrays(self) -> None:
        left_axis_before = self.left.axis_cm1.copy()
        left_intensity_before = self.left.intensity.copy()
        right_axis_before = self.right.axis_cm1.copy()
        right_intensity_before = self.right.intensity.copy()

        result = evaluate_metric(self.metric, self.request)

        self.assertAlmostEqual(
            result.outputs[0].value,
            21.0 / math.sqrt(695.0),
            places=15,
        )
        np.testing.assert_array_equal(self.left.axis_cm1, left_axis_before)
        np.testing.assert_array_equal(self.left.intensity, left_intensity_before)
        np.testing.assert_array_equal(self.right.axis_cm1, right_axis_before)
        np.testing.assert_array_equal(self.right.intensity, right_intensity_before)
        self.assertFalse(self.left.axis_cm1.flags.writeable)
        self.assertFalse(self.left.intensity.flags.writeable)
        self.assertFalse(self.right.axis_cm1.flags.writeable)
        self.assertFalse(self.right.intensity.flags.writeable)

    def test_repeated_evaluation_is_deterministic(self) -> None:
        first = evaluate_metric(self.metric, self.request)
        second = evaluate_metric(self.metric, self.request)

        self.assertEqual(first.metric_id, second.metric_id)
        self.assertEqual(first.input_kind, second.input_kind)
        self.assertEqual(first.outputs[0].output_id, second.outputs[0].output_id)
        self.assertEqual(first.outputs[0].unit, second.outputs[0].unit)
        self.assertEqual(
            first.outputs[0].preferred_direction,
            second.outputs[0].preferred_direction,
        )
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertAlmostEqual(
            first.outputs[0].value,
            second.outputs[0].value,
            places=15,
        )


if __name__ == "__main__":
    unittest.main()
