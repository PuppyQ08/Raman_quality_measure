#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import (  # noqa: E402
    AxisPolicy,
    CalibrationInput,
    MetricInputKind,
    PreferredDirection,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    AnalyticalMetricError,
    TechnicalRepeatabilityLodLoqMetric,
)


def calibration_request(
    *,
    target_names: tuple[str, ...] = ("analyte",),
    true_values: tuple[tuple[float, ...], ...],
    predicted_values: tuple[tuple[float, ...], ...],
    blank_values: tuple[tuple[float, ...], ...],
) -> CalibrationInput:
    return CalibrationInput(
        target_names=target_names,
        true_concentrations=np.asarray(true_values, dtype="<f8"),
        predicted_concentrations=np.asarray(predicted_values, dtype="<f8"),
        blank_predictions=np.asarray(blank_values, dtype="<f8"),
    )


class TechnicalRepeatabilityLodLoqMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = TechnicalRepeatabilityLodLoqMetric()
        self.literal_oracle = calibration_request(
            true_values=((0.0,), (1.0,), (2.0,)),
            predicted_values=((1.0,), (3.0,), (5.0,)),
            blank_values=((0.0,), (2.0,)),
        )

    def test_literal_oracle_outputs_identity_and_diagnostics(self) -> None:
        result = evaluate_metric(self.metric, self.literal_oracle)

        self.assertEqual(
            result.metric_id,
            "technical_repeatability_lod_loq",
        )
        self.assertEqual(result.input_kind, MetricInputKind.CALIBRATION)
        self.assertEqual(
            self.metric.input_kind,
            MetricInputKind.CALIBRATION,
        )
        self.assertEqual(
            self.metric.axis_policy,
            AxisPolicy.NOT_APPLICABLE,
        )
        self.assertEqual(len(result.outputs), 3)

        expected_values = {
            "ich_lod": 3.3 * math.sqrt(2.0) / 2.0,
            "ich_loq": 5.0 * math.sqrt(2.0),
            "iupac_lod": 3.0 * math.sqrt(2.0) / 2.0,
        }
        expected_directions = {
            "ich_lod": PreferredDirection.LOWER_IS_BETTER,
            "ich_loq": PreferredDirection.LOWER_IS_BETTER,
            "iupac_lod": PreferredDirection.LOWER_IS_BETTER,
        }
        outputs = {output.output_id: output for output in result.outputs}
        self.assertEqual(set(outputs), set(expected_values))
        for output_id, expected_value in expected_values.items():
            with self.subTest(output_id=output_id):
                output = outputs[output_id]
                self.assertEqual(output.unit, "input_concentration_unit")
                self.assertEqual(
                    output.preferred_direction,
                    expected_directions[output_id],
                )
                self.assertIsNone(output.target_value)
                self.assertAlmostEqual(
                    output.value,
                    expected_value,
                    places=15,
                )

        self.assertEqual(
            dict(result.diagnostics),
            {
                "target_name": "analyte",
                "training_row_count": 3,
                "blank_repeat_count": 2,
                "ols_slope": 2.0,
                "ols_intercept": 1.0,
                "blank_sample_sd": math.sqrt(2.0),
                "sigma_convention": "sample_sd_ddof_1",
                "slope_convention": "ols_predicted_on_true",
                "claim_boundary": (
                    "supplied_prediction_technical_repeatability"
                ),
            },
        )

    def test_metric_is_frozen_and_public_identity_is_not_overridable(self) -> None:
        self.assertEqual(
            self.metric.metric_id,
            "technical_repeatability_lod_loq",
        )
        with self.assertRaises(TypeError):
            TechnicalRepeatabilityLodLoqMetric(metric_id="override")
        with self.assertRaises(FrozenInstanceError):
            self.metric.metric_id = "override"

    def test_direct_metric_call_matches_wrapper_result(self) -> None:
        wrapped = evaluate_metric(self.metric, self.literal_oracle)
        direct = self.metric.evaluate(self.literal_oracle)

        self.assertEqual(direct, wrapped)

    def test_request_must_have_exactly_one_target(self) -> None:
        multi_target = calibration_request(
            target_names=("a", "b"),
            true_values=((0.0, 0.0), (1.0, 1.0)),
            predicted_values=((0.1, 0.2), (0.9, 1.1)),
            blank_values=((0.01, 0.02), (0.03, 0.04)),
        )

        with self.assertRaisesRegex(
            AnalyticalMetricError,
            "^target_names: must contain exactly one target$",
        ):
            self.metric.evaluate(multi_target)

    def test_training_requires_at_least_two_rows(self) -> None:
        too_short = calibration_request(
            true_values=((0.0,),),
            predicted_values=((0.1,),),
            blank_values=((0.01,), (0.02,)),
        )

        with self.assertRaisesRegex(
            AnalyticalMetricError,
            "^training rows: must be at least 2$",
        ):
            self.metric.evaluate(too_short)

    def test_true_values_must_not_be_constant(self) -> None:
        constant_true = calibration_request(
            true_values=((1.0,), (1.0,), (1.0,)),
            predicted_values=((0.5,), (0.7,), (0.9,)),
            blank_values=((0.1,), (0.2,)),
        )

        with self.assertRaisesRegex(
            AnalyticalMetricError,
            "^true concentrations: must contain at least two distinct values$",
        ):
            self.metric.evaluate(constant_true)

    def test_zero_or_negative_slope_is_rejected(self) -> None:
        zero_slope = calibration_request(
            true_values=((0.0,), (1.0,), (2.0,)),
            predicted_values=((1.0,), (1.0,), (1.0,)),
            blank_values=((0.1,), (0.2,)),
        )
        negative_slope = calibration_request(
            true_values=((0.0,), (1.0,), (2.0,)),
            predicted_values=((5.0,), (3.0,), (1.0,)),
            blank_values=((0.1,), (0.2,)),
        )

        with self.assertRaisesRegex(
            AnalyticalMetricError,
            "^ols slope: must be positive and finite$",
        ):
            self.metric.evaluate(zero_slope)
        with self.assertRaisesRegex(
            AnalyticalMetricError,
            "^ols slope: must be positive and finite$",
        ):
            self.metric.evaluate(negative_slope)

    def test_constant_blank_is_rejected(self) -> None:
        constant_blank = calibration_request(
            true_values=((0.0,), (1.0,), (2.0,)),
            predicted_values=((1.0,), (3.0,), (5.0,)),
            blank_values=((7.0,), (7.0,)),
        )

        with self.assertRaisesRegex(
            AnalyticalMetricError,
            "^blank sample sd: must be positive and finite$",
        ):
            self.metric.evaluate(constant_blank)

    def test_blank_sample_sd_uses_ddof_one(self) -> None:
        request = calibration_request(
            true_values=((0.0,), (1.0,)),
            predicted_values=((0.0,), (1.0,)),
            blank_values=((1.0,), (2.0,), (3.0,)),
        )

        result = evaluate_metric(self.metric, request)
        diagnostics = dict(result.diagnostics)
        self.assertAlmostEqual(diagnostics["blank_sample_sd"], 1.0, places=15)
        self.assertEqual(
            diagnostics["sigma_convention"],
            "sample_sd_ddof_1",
        )

    def test_nonfinite_derived_outputs_are_rejected(self) -> None:
        overflowing = calibration_request(
            true_values=((0.0,), (1.0,)),
            predicted_values=((0.0,), (np.nextafter(0.0, 1.0),)),
            blank_values=((0.0,), (np.finfo(np.float64).max,)),
        )

        with self.assertRaisesRegex(
            AnalyticalMetricError,
            "^derived outputs: must be finite$",
        ):
            self.metric.evaluate(overflowing)

    def test_request_arrays_remain_immutable_and_unchanged(self) -> None:
        before_true = self.literal_oracle.true_concentrations.copy()
        before_predicted = self.literal_oracle.predicted_concentrations.copy()
        before_blank = self.literal_oracle.blank_predictions.copy()

        evaluate_metric(self.metric, self.literal_oracle)

        np.testing.assert_array_equal(
            self.literal_oracle.true_concentrations,
            before_true,
        )
        np.testing.assert_array_equal(
            self.literal_oracle.predicted_concentrations,
            before_predicted,
        )
        np.testing.assert_array_equal(
            self.literal_oracle.blank_predictions,
            before_blank,
        )
        self.assertFalse(self.literal_oracle.true_concentrations.flags.writeable)
        self.assertFalse(
            self.literal_oracle.predicted_concentrations.flags.writeable
        )
        self.assertFalse(self.literal_oracle.blank_predictions.flags.writeable)

    def test_deterministic_execution_returns_identical_results(self) -> None:
        first = evaluate_metric(self.metric, self.literal_oracle)
        second = evaluate_metric(self.metric, self.literal_oracle)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
