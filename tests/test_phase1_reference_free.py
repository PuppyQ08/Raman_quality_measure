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
    MetricInputKind,
    PreferredDirection,
    SingleSpectrumInput,
    Spectrum1D,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    ISLikeMetricError,
    ISLikeStructureToNoiseMetric,
)


GAUSSIAN_MAD_CONSTANT = 0.6744897501960817


def spectrum(
    intensity: tuple[float, ...],
    *,
    spectrum_id: str = "spectrum-a",
    axis: tuple[float, ...] | None = None,
) -> Spectrum1D:
    if axis is None:
        axis = tuple(float(index) for index in range(len(intensity)))
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.asarray(axis, dtype="<f8"),
        intensity=np.asarray(intensity, dtype="<f8"),
    )


def request(
    intensity: tuple[float, ...],
    *,
    spectrum_id: str = "spectrum-a",
    axis: tuple[float, ...] | None = None,
) -> SingleSpectrumInput:
    return SingleSpectrumInput(
        spectrum(
            intensity,
            spectrum_id=spectrum_id,
            axis=axis,
        )
    )


class ISLikeStructureToNoiseMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = ISLikeStructureToNoiseMetric()

    def test_literal_oracle_identity_and_diagnostics(self) -> None:
        result = evaluate_metric(self.metric, request((0.0, 2.0, 0.0, 4.0)))

        expected_signal_rms = math.sqrt(11.0) / 2.0
        expected_noise_sigma = (
            2.0 / GAUSSIAN_MAD_CONSTANT / math.sqrt(2.0)
        )
        expected_score = expected_signal_rms / expected_noise_sigma

        self.assertEqual(result.metric_id, "is_like_structure_to_noise")
        self.assertEqual(result.input_kind, MetricInputKind.SINGLE_SPECTRUM)
        self.assertEqual(self.metric.input_kind, MetricInputKind.SINGLE_SPECTRUM)
        self.assertEqual(self.metric.axis_policy, AxisPolicy.SINGLE_AXIS)
        self.assertEqual(len(result.outputs), 1)

        output = result.outputs[0]
        self.assertEqual(output.output_id, "is_like_structure_to_noise")
        self.assertEqual(output.unit, "ratio")
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.HIGHER_IS_BETTER,
        )
        self.assertIsNone(output.target_value)
        self.assertAlmostEqual(output.value, expected_score, places=15)

        self.assertEqual(
            dict(result.diagnostics),
            {
                "definition_boundary": (
                    "classical_is_like_not_learned_score"
                ),
                "gaussian_mad_constant": GAUSSIAN_MAD_CONSTANT,
                "intensity_scale_power2_exponent": 2,
                "noise_estimator": "first_difference_mad_gaussian_sigma",
                "noise_sigma_scaled": expected_noise_sigma / 4.0,
                "point_count": 4,
                "signal_rms_scaled": expected_signal_rms / 4.0,
            },
        )

    def test_offset_and_positive_scale_invariance_hold(self) -> None:
        base = evaluate_metric(self.metric, request((0.0, 2.0, 0.0, 4.0)))
        offset = evaluate_metric(
            self.metric,
            request((100.0, 102.0, 100.0, 104.0)),
        )
        scaled = evaluate_metric(
            self.metric,
            request((0.0, 7.0, 0.0, 14.0)),
        )

        self.assertAlmostEqual(
            offset.outputs[0].value,
            base.outputs[0].value,
            places=15,
        )
        self.assertAlmostEqual(
            scaled.outputs[0].value,
            base.outputs[0].value,
            places=15,
        )

    def test_negative_values_large_offset_and_uniform_subnormal_scale_are_supported(
        self,
    ) -> None:
        base = evaluate_metric(self.metric, request((-3.0, -1.0, -3.0, 1.0)))
        representable_offset = math.ldexp(1.0, 50)
        huge_offset = evaluate_metric(
            self.metric,
            request(
                (
                    representable_offset - 3.0,
                    representable_offset - 1.0,
                    representable_offset - 3.0,
                    representable_offset + 1.0,
                )
            ),
        )
        tiny_scaled = evaluate_metric(
            self.metric,
            request(
                tuple(
                    math.ldexp(value, -1072)
                    for value in (-3.0, -1.0, -3.0, 1.0)
                )
            ),
        )

        self.assertAlmostEqual(
            huge_offset.outputs[0].value,
            base.outputs[0].value,
            places=15,
        )
        self.assertAlmostEqual(
            tiny_scaled.outputs[0].value,
            base.outputs[0].value,
            places=15,
        )

    def test_wrong_request_type_is_rejected(self) -> None:
        wrong = spectrum((0.0, 2.0, 0.0, 4.0))
        with self.assertRaisesRegex(
            ISLikeMetricError,
            "^request: must be SingleSpectrumInput$",
        ):
            self.metric.evaluate(wrong)

    def test_constant_linear_and_too_short_inputs_fail_closed(self) -> None:
        cases = (
            (
                "signal_rms",
                request((5.0, 5.0, 5.0, 5.0)),
            ),
            (
                "noise_sigma",
                request((0.0, 2.0, 4.0, 6.0)),
            ),
            (
                "point_count",
                request((0.0, 1.0)),
            ),
        )

        for expected, metric_request in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ISLikeMetricError, f"^{expected}: "):
                    self.metric.evaluate(metric_request)

    def test_frozen_identity_immutability_and_repeat_determinism(self) -> None:
        first = evaluate_metric(self.metric, request((0.0, 2.0, 0.0, 4.0)))
        second = evaluate_metric(self.metric, request((0.0, 2.0, 0.0, 4.0)))

        self.assertEqual(self.metric.metric_id, "is_like_structure_to_noise")
        with self.assertRaises(TypeError):
            ISLikeStructureToNoiseMetric(metric_id="override")
        with self.assertRaises(FrozenInstanceError):
            self.metric.metric_id = "override"

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
