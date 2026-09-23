from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import (  # noqa: E402
    AxisPolicy,
    MetricInputKind,
    PreferredDirection,
    SingleSpectrumInput,
    SpectralRegion,
    Spectrum1D,
    SpectrumRegionsInput,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    SNRMetricError,
    SignedIntegratedAreaRmsNoiseSnrMetric,
    SignedPeakHeightRmsNoiseSnrMetric,
    SignedReferenceIntervalMeanSnrMetric,
)


def spectrum(
    spectrum_id: str = "spectrum-a",
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


def request(
    *,
    axis: tuple[float, ...],
    intensity: tuple[float, ...],
    regions: tuple[SpectralRegion, ...],
) -> SpectrumRegionsInput:
    return SpectrumRegionsInput(
        spectrum(axis=axis, intensity=intensity),
        regions,
    )


class SignedPeakHeightSnrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.axis = (0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0)
        self.intensity = (
            1.0,
            0.0,
            3.0,
            5.0,
            2.0,
            1.0,
            0.0,
            2.0,
            1.0,
        )
        self.regions = (
            SpectralRegion("signal-band", 2.0, 5.0, "signal"),
            SpectralRegion("noise-band", 8.0, 10.0, "noise"),
            SpectralRegion("ref-band", 8.0, 10.0, "reference"),
        )
        self.metric = SignedPeakHeightRmsNoiseSnrMetric()
        self.request = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=self.regions,
        )

    def test_literal_peak_formula_identity_and_diagnostics(self) -> None:
        result = evaluate_metric(self.metric, self.request)

        self.assertEqual(
            result.metric_id,
            "signed_snr_peak_height_rms_noise",
        )
        self.assertEqual(result.input_kind, MetricInputKind.SPECTRUM_REGIONS)
        self.assertEqual(len(result.outputs), 1)

        output = result.outputs[0]
        self.assertEqual(
            output.output_id,
            "signed_snr_peak_height_rms_noise",
        )
        self.assertEqual(output.unit, "ratio")
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.HIGHER_IS_BETTER,
        )
        self.assertIsNone(output.target_value)
        self.assertAlmostEqual(
            output.value,
            2.0 * math.sqrt(6.0),
            places=15,
        )

        self.assertEqual(
            self.metric.input_kind,
            MetricInputKind.SPECTRUM_REGIONS,
        )
        self.assertEqual(self.metric.axis_policy, AxisPolicy.SINGLE_AXIS)
        self.assertEqual(
            self.metric.__doc__,
            "Signed standardized contrast; not a power SNR or dB SNR.",
        )

        diagnostics = dict(result.diagnostics)
        self.assertEqual(
            diagnostics,
            {
                "denominator_region": {
                    "end_cm1": 10.0,
                    "point_count": 3,
                    "region_id": "noise-band",
                    "role": "noise",
                    "start_cm1": 8.0,
                },
                "intensity_scale_power2_exponent": 2,
                "noise_rms_convention": "centered_population_rms",
                "region_endpoint_policy": "closed_native_samples",
                "region_overlap_policy": "disjoint_closed_physical_intervals",
                "scaled_denominator_anchor": 0.0,
                "scaled_denominator_mean_delta": 0.25,
                "scaled_denominator_rms": math.sqrt(2.0 / 48.0),
                "scaled_signed_numerator": 1.0,
                "signal_region": {
                    "end_cm1": 5.0,
                    "point_count": 3,
                    "region_id": "signal-band",
                    "role": "signal",
                    "start_cm1": 2.0,
                },
                "signed_output": True,
            },
        )
        self.assertNotIn(
            "signed_snr_peak_height_rms_noise",
            repr(diagnostics),
        )

    def test_wrong_request_type_is_rejected(self) -> None:
        wrong = SingleSpectrumInput(
            spectrum(
                axis=self.axis,
                intensity=self.intensity,
            )
        )
        with self.assertRaisesRegex(SNRMetricError, "^request: "):
            self.metric.evaluate(wrong)

    def test_missing_signal_or_noise_is_rejected(self) -> None:
        missing_signal = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(SpectralRegion("noise-band", 8.0, 10.0, "noise"),),
        )
        missing_noise = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(SpectralRegion("signal-band", 2.0, 5.0, "signal"),),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.signal: must contain exactly one consumed region$",
        ):
            self.metric.evaluate(missing_signal)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.noise: must contain exactly one consumed region$",
        ):
            self.metric.evaluate(missing_noise)

    def test_duplicate_signal_or_noise_is_rejected(self) -> None:
        duplicate_signal = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-a", 2.0, 4.0, "signal"),
                SpectralRegion("signal-b", 4.5, 5.0, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
            ),
        )
        duplicate_noise = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-a", 8.0, 8.5, "noise"),
                SpectralRegion("noise-b", 9.0, 10.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.signal: must contain exactly one consumed region$",
        ):
            self.metric.evaluate(duplicate_signal)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.noise: must contain exactly one consumed region$",
        ):
            self.metric.evaluate(duplicate_noise)

    def test_unrelated_roles_are_ignored(self) -> None:
        request_with_other = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
                SpectralRegion("other-band", 0.0, 1.0, "other"),
                SpectralRegion("ref-band", 2.0, 5.0, "reference"),
            ),
        )

        result = evaluate_metric(self.metric, request_with_other)
        self.assertAlmostEqual(
            result.outputs[0].value,
            2.0 * math.sqrt(6.0),
            places=15,
        )

    def test_closed_endpoints_are_included(self) -> None:
        exact = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
            ),
        )
        shifted = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 1.9, 5.1, "signal"),
                SpectralRegion("noise-band", 7.8, 10.0, "noise"),
            ),
        )

        exact_result = evaluate_metric(self.metric, exact)
        shifted_result = evaluate_metric(self.metric, shifted)
        self.assertAlmostEqual(
            exact_result.outputs[0].value,
            shifted_result.outputs[0].value,
            places=15,
        )

    def test_touching_consumed_intervals_are_rejected(self) -> None:
        touching = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.0, 8.0, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions overlap: consumed intervals must be disjoint as closed intervals$",
        ):
            self.metric.evaluate(touching)

    def test_overlap_with_ignored_region_is_accepted(self) -> None:
        ignored_overlap = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
                SpectralRegion("ref-band", 4.0, 9.0, "reference"),
            ),
        )

        result = evaluate_metric(self.metric, ignored_overlap)
        self.assertAlmostEqual(
            result.outputs[0].value,
            2.0 * math.sqrt(6.0),
            places=15,
        )

    def test_empty_signal_or_noise_is_rejected(self) -> None:
        empty_signal = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.1, 3.9, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
            ),
        )
        empty_noise = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 8.1, 8.9, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^signal_region point_count: must select at least 1 native samples$",
        ):
            self.metric.evaluate(empty_signal)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^noise_region point_count: must select at least 2 native samples$",
        ):
            self.metric.evaluate(empty_noise)

    def test_one_point_noise_and_constant_noise_are_rejected(self) -> None:
        one_point_noise = request(
            axis=self.axis,
            intensity=self.intensity,
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 9.0, 9.0 + 1e-12, "noise"),
            ),
        )
        constant_noise = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(5.0, 7.0, 4.0, 4.0),
            regions=(
                SpectralRegion("signal-band", 1.0, 1.0 + 1e-12, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^noise_region point_count: must select at least 2 native samples$",
        ):
            self.metric.evaluate(one_point_noise)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^denominator rms: zero centered population RMS$",
        ):
            self.metric.evaluate(constant_noise)

    def test_rejects_required_noise_l2_term_lost_before_sum(self) -> None:
        tiny = math.ldexp(1.0, -1072)
        underflow_l2 = request(
            axis=(0.0, 1.0, 2.0, 3.0, 4.0),
            intensity=(1.0, 0.0, 1.0, -1.0, tiny),
            regions=(
                SpectralRegion("signal-band", 0.0, 0.0 + 1e-12, "signal"),
                SpectralRegion("noise-band", 1.0, 4.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^denominator residual square: required nonzero L2 term became zero$",
        ):
            evaluate_metric(self.metric, underflow_l2)

    def test_signed_output_orders_positive_zero_and_negative(self) -> None:
        axis = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        intensity = (2.0, 1.0, 0.0, 0.0, 2.0, 1.0)
        positive = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-positive", 0.0, 0.0 + 1e-12, "signal"),
                SpectralRegion("noise-band", 3.0, 5.0, "noise"),
            ),
        )
        zero = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-zero", 1.0, 1.0 + 1e-12, "signal"),
                SpectralRegion("noise-band", 3.0, 5.0, "noise"),
            ),
        )
        negative = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-negative", 2.0, 2.0 + 1e-12, "signal"),
                SpectralRegion("noise-band", 3.0, 5.0, "noise"),
            ),
        )

        positive_value = evaluate_metric(self.metric, positive).outputs[0].value
        zero_value = evaluate_metric(self.metric, zero).outputs[0].value
        negative_value = evaluate_metric(self.metric, negative).outputs[0].value

        self.assertTrue(math.isfinite(positive_value))
        self.assertTrue(math.isfinite(zero_value))
        self.assertTrue(math.isfinite(negative_value))
        self.assertGreater(positive_value, 0.0)
        self.assertEqual(zero_value, 0.0)
        self.assertLess(negative_value, 0.0)
        self.assertGreater(positive_value, zero_value)
        self.assertGreater(zero_value, negative_value)

    def test_positive_power_of_two_scaling_is_invariant(self) -> None:
        base = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )
        scaled = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(24.0, 8.0, 0.0, 16.0),
            regions=base.regions,
        )

        base_value = evaluate_metric(self.metric, base).outputs[0].value
        scaled_value = evaluate_metric(self.metric, scaled).outputs[0].value

        self.assertAlmostEqual(base_value, 2.0, places=15)
        self.assertAlmostEqual(scaled_value, 2.0, places=15)
        self.assertAlmostEqual(base_value, scaled_value, places=15)

    def test_representability_preserving_common_offset_is_invariant(self) -> None:
        base = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )
        offset = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(1027.0, 1025.0, 1024.0, 1026.0),
            regions=base.regions,
        )

        base_value = evaluate_metric(self.metric, base).outputs[0].value
        offset_value = evaluate_metric(self.metric, offset).outputs[0].value

        self.assertAlmostEqual(base_value, 2.0, places=15)
        self.assertAlmostEqual(offset_value, 2.0, places=15)
        self.assertAlmostEqual(base_value, offset_value, places=15)


class SignedIntegratedAreaSnrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = SignedIntegratedAreaRmsNoiseSnrMetric()
        self.literal_request = request(
            axis=(0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0),
            intensity=(
                1.0,
                0.0,
                3.0,
                5.0,
                2.0,
                1.0,
                0.0,
                2.0,
                1.0,
            ),
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
                SpectralRegion("ref-band", 8.0, 10.0, "reference"),
            ),
        )

    def test_literal_integrated_formula_identity_and_diagnostics(self) -> None:
        expected = 17.0 * math.sqrt(21.0) / 14.0

        result = evaluate_metric(self.metric, self.literal_request)

        self.assertEqual(
            result.metric_id,
            "signed_snr_integrated_area_rms_noise",
        )
        self.assertEqual(result.input_kind, MetricInputKind.SPECTRUM_REGIONS)
        self.assertEqual(len(result.outputs), 1)

        output = result.outputs[0]
        self.assertEqual(
            output.output_id,
            "signed_snr_integrated_area_rms_noise",
        )
        self.assertEqual(output.unit, "ratio")
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.HIGHER_IS_BETTER,
        )
        self.assertIsNone(output.target_value)
        self.assertAlmostEqual(output.value, expected, places=15)

        self.assertEqual(
            self.metric.input_kind,
            MetricInputKind.SPECTRUM_REGIONS,
        )
        self.assertEqual(self.metric.axis_policy, AxisPolicy.SINGLE_AXIS)
        self.assertEqual(
            self.metric.__doc__,
            "Signed standardized contrast; not a power SNR or dB SNR.",
        )

        diagnostics = dict(result.diagnostics)
        self.assertEqual(
            diagnostics,
            {
                "denominator_region": {
                    "end_cm1": 10.0,
                    "point_count": 3,
                    "region_id": "noise-band",
                    "role": "noise",
                    "start_cm1": 8.0,
                },
                "intensity_scale_power2_exponent": 2,
                "noise_rms_convention": "centered_population_rms",
                "region_endpoint_policy": "closed_native_samples",
                "region_overlap_policy": "disjoint_closed_physical_intervals",
                "scaled_denominator_anchor": 0.0,
                "scaled_denominator_mean_delta": 0.25,
                "scaled_denominator_rms": math.sqrt(2.0 / 48.0),
                "scaled_signed_numerator": 17.0 / 8.0,
                "signal_region": {
                    "end_cm1": 5.0,
                    "point_count": 3,
                    "region_id": "signal-band",
                    "role": "signal",
                    "start_cm1": 2.0,
                },
                "signed_output": True,
                "axis_scale_power2_exponent": 2,
                "scaled_weight_l2_norm": math.sqrt(3.5),
                "area_quadrature": "native_trapezoid_node_weights",
                "noise_model": (
                    "conditional_iid_signal_samples_with_fixed_estimated_baseline"
                ),
            },
        )
        self.assertNotIn(
            "signed_snr_integrated_area_rms_noise",
            repr(diagnostics),
        )

    def test_one_selected_signal_point_is_rejected(self) -> None:
        one_point_signal = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1e-12, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^signal_region point_count: must select at least 2 native samples$",
        ):
            evaluate_metric(self.metric, one_point_signal)

    def test_two_signal_points_match_hand_oracle(self) -> None:
        two_point = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )

        result = evaluate_metric(self.metric, two_point)
        self.assertAlmostEqual(result.outputs[0].value, math.sqrt(2.0), places=15)

    def test_nonuniform_native_trapezoid_matches_hand_oracle(self) -> None:
        nonuniform = request(
            axis=(0.0, 2.0, 3.0, 8.0, 9.0),
            intensity=(2.0, 3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 3.0, "signal"),
                SpectralRegion("noise-band", 8.0, 9.0, "noise"),
            ),
        )

        result = evaluate_metric(self.metric, nonuniform)
        self.assertAlmostEqual(
            result.outputs[0].value,
            4.0 * math.sqrt(14.0) / 7.0,
            places=15,
        )

    def test_uniform_axis_reduces_to_endpoint_weight_formula(self) -> None:
        uniform = request(
            axis=(0.0, 1.0, 2.0, 3.0, 5.0, 6.0),
            intensity=(3.0, 3.0, 3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 3.0, "signal"),
                SpectralRegion("noise-band", 5.0, 6.0, "noise"),
            ),
        )

        result = evaluate_metric(self.metric, uniform)
        self.assertAlmostEqual(result.outputs[0].value, math.sqrt(10.0), places=15)

    def test_positive_power_of_two_scaling_is_invariant(self) -> None:
        base = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )
        scaled = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(24.0, 8.0, 0.0, 16.0),
            regions=base.regions,
        )

        base_value = evaluate_metric(self.metric, base).outputs[0].value
        scaled_value = evaluate_metric(self.metric, scaled).outputs[0].value
        self.assertAlmostEqual(base_value, scaled_value, places=15)

    def test_representability_preserving_common_offset_is_invariant(self) -> None:
        base = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )
        offset = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(1027.0, 1025.0, 1024.0, 1026.0),
            regions=base.regions,
        )

        base_value = evaluate_metric(self.metric, base).outputs[0].value
        offset_value = evaluate_metric(self.metric, offset).outputs[0].value
        self.assertAlmostEqual(base_value, offset_value, places=15)

    def test_signed_outputs_order_positive_zero_and_negative(self) -> None:
        axis = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        intensity = (3.0, 1.0, 3.0, -1.0, 1.0, -1.0, 0.0, 2.0)

        positive = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-positive", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 6.0, 7.0, "noise"),
            ),
        )
        zero = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-zero", 2.0, 3.0, "signal"),
                SpectralRegion("noise-band", 6.0, 7.0, "noise"),
            ),
        )
        negative = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-negative", 4.0, 5.0, "signal"),
                SpectralRegion("noise-band", 6.0, 7.0, "noise"),
            ),
        )

        positive_value = evaluate_metric(self.metric, positive).outputs[0].value
        zero_value = evaluate_metric(self.metric, zero).outputs[0].value
        negative_value = evaluate_metric(self.metric, negative).outputs[0].value

        self.assertTrue(math.isfinite(positive_value))
        self.assertTrue(math.isfinite(zero_value))
        self.assertTrue(math.isfinite(negative_value))
        self.assertGreater(positive_value, 0.0)
        self.assertEqual(zero_value, 0.0)
        self.assertLess(negative_value, 0.0)
        self.assertGreater(positive_value, zero_value)
        self.assertGreater(zero_value, negative_value)

    def test_exact_weighted_cancellation_returns_finite_zero(self) -> None:
        cancelling = request(
            axis=(0.0, 2.0, 3.0, 8.0, 9.0),
            intensity=(1.0, -1.0, 1.0, -1.0, 1.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 3.0, "signal"),
                SpectralRegion("noise-band", 8.0, 9.0, "noise"),
            ),
        )

        value = evaluate_metric(self.metric, cancelling).outputs[0].value
        self.assertTrue(math.isfinite(value))
        self.assertEqual(value, 0.0)

    def test_raw_negative_signal_intensities_are_accepted(self) -> None:
        negative_signal = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(-1.0, -3.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )

        result = evaluate_metric(self.metric, negative_signal)
        self.assertAlmostEqual(
            result.outputs[0].value,
            -3.0 * math.sqrt(2.0),
            places=14,
        )

    def test_rejects_required_intensity_lost_during_consumed_scaling(self) -> None:
        tiny = float(np.nextafter(0.0, 1.0))
        lost_intensity = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(np.finfo(np.float64).max, tiny, 0.0, 1.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^signal values: required nonzero value became zero after scaling$",
        ):
            evaluate_metric(self.metric, lost_intensity)

    def test_rejects_required_weighted_product_lost_before_sum(self) -> None:
        tiny_ordinate = math.ldexp(1.0, -600)
        tiny_interval = math.ldexp(1.0, -535)
        weighted_product_underflow = request(
            axis=(0.0, tiny_interval, 1.0, 3.0, 4.0),
            intensity=(tiny_ordinate, 0.0, 0.0, -1.0, 1.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 2.0, "signal"),
                SpectralRegion("noise-band", 3.0, 4.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^weighted signal term: required nonzero product became zero$",
        ):
            evaluate_metric(self.metric, weighted_product_underflow)

    def test_rejects_required_weight_l2_term_lost_before_sum(self) -> None:
        import rpe.metrics.snr as snr_module

        tiny = math.ldexp(1.0, -1072)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^weight square: required nonzero L2 term became zero$",
        ):
            snr_module._scaled_relative_trapezoid_weights(
                np.asarray((0.0, tiny, 1.0), dtype=np.float64)
            )

    def test_rejects_interval_lost_under_relative_axis_scaling(self) -> None:
        import rpe.metrics.snr as snr_module

        tiny = float(np.nextafter(0.0, 1.0))
        with self.assertRaisesRegex(
            SNRMetricError,
            "^axis interval: positive original interval became zero after scaling$",
        ):
            snr_module._scaled_relative_trapezoid_weights(
                np.asarray((0.0, tiny, np.finfo(np.float64).max), dtype=np.float64)
            )

    def test_rejects_derived_interval_overflow(self) -> None:
        import rpe.metrics.snr as snr_module

        huge = np.finfo(np.float64).max
        with self.assertRaisesRegex(
            SNRMetricError,
            "^axis interval: produced non-finite value$",
        ):
            snr_module._scaled_relative_trapezoid_weights(
                np.asarray((-huge, huge), dtype=np.float64)
            )

    def test_rejects_finite_mathematical_ratio_overflow(self) -> None:
        tiny_rms = float(np.nextafter(0.0, 1.0))
        overflow_ratio = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(1.0, 1.0, 0.0, tiny_rms),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("noise-band", 2.0, 3.0, "noise"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^ratio: produced non-finite value$",
        ):
            evaluate_metric(self.metric, overflow_ratio)


class SignedReferenceIntervalMeanSnrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = SignedReferenceIntervalMeanSnrMetric()
        self.literal_request = request(
            axis=(0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0),
            intensity=(
                1.0,
                0.0,
                3.0,
                5.0,
                2.0,
                1.0,
                0.0,
                2.0,
                1.0,
            ),
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 8.0, 10.0, "noise"),
                SpectralRegion("ref-band", 8.0, 10.0, "reference"),
            ),
        )

    def test_literal_reference_formula_identity_and_diagnostics(self) -> None:
        expected = 2.8577380332470415

        result = evaluate_metric(self.metric, self.literal_request)

        self.assertEqual(
            result.metric_id,
            "signed_snr_reference_interval_mean",
        )
        self.assertEqual(result.input_kind, MetricInputKind.SPECTRUM_REGIONS)
        self.assertEqual(len(result.outputs), 1)

        output = result.outputs[0]
        self.assertEqual(
            output.output_id,
            "signed_snr_reference_interval_mean",
        )
        self.assertEqual(output.unit, "ratio")
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.HIGHER_IS_BETTER,
        )
        self.assertIsNone(output.target_value)
        self.assertAlmostEqual(output.value, expected, places=15)

        self.assertEqual(
            self.metric.input_kind,
            MetricInputKind.SPECTRUM_REGIONS,
        )
        self.assertEqual(self.metric.axis_policy, AxisPolicy.SINGLE_AXIS)
        self.assertEqual(
            self.metric.__doc__,
            "Signed standardized contrast; not a power SNR or dB SNR.",
        )

        diagnostics = dict(result.diagnostics)
        self.assertEqual(
            diagnostics,
            {
                "denominator_region": {
                    "end_cm1": 10.0,
                    "point_count": 3,
                    "region_id": "ref-band",
                    "role": "reference",
                    "start_cm1": 8.0,
                },
                "intensity_scale_power2_exponent": 2,
                "noise_rms_convention": "centered_population_rms",
                "region_endpoint_policy": "closed_native_samples",
                "region_overlap_policy": "disjoint_closed_physical_intervals",
                "scaled_denominator_anchor": 0.0,
                "scaled_denominator_mean_delta": 0.25,
                "scaled_denominator_rms": math.sqrt(2.0 / 48.0),
                "scaled_signed_numerator": 0.5833333333333334,
                "signal_region": {
                    "end_cm1": 5.0,
                    "point_count": 3,
                    "region_id": "signal-band",
                    "role": "signal",
                    "start_cm1": 2.0,
                },
                "signed_output": True,
            },
        )
        self.assertNotIn(
            "signed_snr_reference_interval_mean",
            repr(diagnostics),
        )

    def test_missing_or_duplicate_signal_or_reference_is_rejected(self) -> None:
        missing_signal = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(SpectralRegion("ref-band", 2.0, 3.0, "reference"),),
        )
        missing_reference = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(SpectralRegion("signal-band", 0.0, 1.0, "signal"),),
        )
        duplicate_signal = request(
            axis=(0.0, 1.0, 2.0, 3.0, 4.0),
            intensity=(3.0, 1.0, 0.0, 2.0, 4.0),
            regions=(
                SpectralRegion("signal-a", 0.0, 0.0 + 1e-12, "signal"),
                SpectralRegion("signal-b", 1.0, 1.0 + 1e-12, "signal"),
                SpectralRegion("ref-band", 3.0, 4.0, "reference"),
            ),
        )
        duplicate_reference = request(
            axis=(0.0, 1.0, 2.0, 3.0, 4.0),
            intensity=(3.0, 1.0, 0.0, 2.0, 4.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-a", 2.0, 3.0, "reference"),
                SpectralRegion("ref-b", 3.5, 4.0, "reference"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.signal: must contain exactly one consumed region$",
        ):
            evaluate_metric(self.metric, missing_signal)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.reference: must contain exactly one consumed region$",
        ):
            evaluate_metric(self.metric, missing_reference)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.signal: must contain exactly one consumed region$",
        ):
            evaluate_metric(self.metric, duplicate_signal)
        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions.reference: must contain exactly one consumed region$",
        ):
            evaluate_metric(self.metric, duplicate_reference)

    def test_unrelated_noise_and_other_roles_are_ignored(self) -> None:
        request_with_ignored = request(
            axis=(0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0),
            intensity=(
                1.0,
                0.0,
                3.0,
                5.0,
                2.0,
                1.0,
                0.0,
                2.0,
                1.0,
            ),
            regions=(
                SpectralRegion("signal-band", 2.0, 5.0, "signal"),
                SpectralRegion("noise-band", 2.0, 5.0, "noise"),
                SpectralRegion("other-band", 0.0, 1.0, "other"),
                SpectralRegion("ref-band", 8.0, 10.0, "reference"),
            ),
        )

        result = evaluate_metric(self.metric, request_with_ignored)
        self.assertEqual(result.outputs[0].value, 2.8577380332470415)

    def test_touching_consumed_intervals_are_rejected(self) -> None:
        touching = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 2.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^regions overlap: consumed intervals must be disjoint as closed intervals$",
        ):
            evaluate_metric(self.metric, touching)

    def test_one_signal_point_is_valid_but_one_reference_point_is_rejected(self) -> None:
        one_signal_point = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 0.0, 2.0, 4.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 0.0 + 1e-12, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )
        one_reference_point = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 0.0, 2.0, 4.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 2.0 + 1e-12, "reference"),
            ),
        )

        self.assertAlmostEqual(
            evaluate_metric(self.metric, one_signal_point).outputs[0].value,
            0.0,
            places=15,
        )
        with self.assertRaisesRegex(
            SNRMetricError,
            "^reference_region point_count: must select at least 2 native samples$",
        ):
            evaluate_metric(self.metric, one_reference_point)

    def test_constant_reference_is_rejected(self) -> None:
        constant_reference = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(5.0, 7.0, 4.0, 4.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^denominator rms: zero centered population RMS$",
        ):
            evaluate_metric(self.metric, constant_reference)

    def test_signed_outputs_order_positive_zero_and_negative(self) -> None:
        axis = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        intensity = (4.0, 1.0, -2.0, 0.0, 2.0, 1.0)
        positive = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-positive", 0.0, 0.0 + 1e-12, "signal"),
                SpectralRegion("ref-band", 3.0, 5.0, "reference"),
            ),
        )
        zero = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-zero", 1.0, 1.0 + 1e-12, "signal"),
                SpectralRegion("ref-band", 3.0, 5.0, "reference"),
            ),
        )
        negative = request(
            axis=axis,
            intensity=intensity,
            regions=(
                SpectralRegion("signal-negative", 2.0, 2.0 + 1e-12, "signal"),
                SpectralRegion("ref-band", 3.0, 5.0, "reference"),
            ),
        )

        positive_value = evaluate_metric(self.metric, positive).outputs[0].value
        zero_value = evaluate_metric(self.metric, zero).outputs[0].value
        negative_value = evaluate_metric(self.metric, negative).outputs[0].value

        self.assertTrue(math.isfinite(positive_value))
        self.assertTrue(math.isfinite(zero_value))
        self.assertTrue(math.isfinite(negative_value))
        self.assertGreater(positive_value, 0.0)
        self.assertEqual(zero_value, 0.0)
        self.assertLess(negative_value, 0.0)
        self.assertGreater(positive_value, zero_value)
        self.assertGreater(zero_value, negative_value)

    def test_raw_negative_signal_and_reference_values_are_accepted(self) -> None:
        negative_values = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(-2.0, -2.0, -5.0, -3.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )

        result = evaluate_metric(self.metric, negative_values)
        self.assertAlmostEqual(
            result.outputs[0].value,
            2.0,
            places=15,
        )

    def test_positive_power_of_two_scaling_is_invariant(self) -> None:
        base = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )
        scaled = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(24.0, 8.0, 0.0, 16.0),
            regions=base.regions,
        )

        base_value = evaluate_metric(self.metric, base).outputs[0].value
        scaled_value = evaluate_metric(self.metric, scaled).outputs[0].value
        self.assertAlmostEqual(base_value, scaled_value, places=15)

    def test_representability_preserving_common_offset_is_invariant(self) -> None:
        base = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(3.0, 1.0, 0.0, 2.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )
        offset = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(1027.0, 1025.0, 1024.0, 1026.0),
            regions=base.regions,
        )

        base_value = evaluate_metric(self.metric, base).outputs[0].value
        offset_value = evaluate_metric(self.metric, offset).outputs[0].value
        self.assertAlmostEqual(base_value, offset_value, places=15)

    def test_large_common_offset_with_preserved_spread_matches_finite_oracle(self) -> None:
        offset = math.ldexp(1.0, 900)
        delta = math.ldexp(1.0, 848)
        large_offset = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(
                offset + 4.0 * delta,
                offset + 4.0 * delta,
                offset,
                offset + 2.0 * delta,
            ),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )

        result = evaluate_metric(self.metric, large_offset)
        self.assertAlmostEqual(
            result.outputs[0].value,
            3.0,
            places=15,
        )

    def test_uniform_subnormal_scale_fixture_matches_finite_oracle(self) -> None:
        unit = float(np.nextafter(0.0, 1.0))
        subnormal = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(4.0 * unit, 2.0 * unit, 0.0, 2.0 * unit),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )

        result = evaluate_metric(self.metric, subnormal)
        self.assertAlmostEqual(
            result.outputs[0].value,
            2.0,
            places=15,
        )

    def test_mixed_max_float_and_min_subnormal_required_terms_fail_closed(self) -> None:
        tiny = float(np.nextafter(0.0, 1.0))
        lost_term = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(np.finfo(np.float64).max, tiny, 0.0, 1.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )

        with self.assertRaisesRegex(
            SNRMetricError,
            "^signal values: required nonzero value became zero after scaling$",
        ):
            evaluate_metric(self.metric, lost_term)

    def test_last_bit_sensitive_reference_mean_uses_sum_then_divide_once(self) -> None:
        first = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(2.0, math.ldexp(1.0, -52), 0.0, 4.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )
        second = request(
            axis=(0.0, 1.0, 2.0, 3.0),
            intensity=(0.0, math.ldexp(1.0, -52), 1.5, 0.0),
            regions=(
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )

        first_result = evaluate_metric(self.metric, first)
        second_result = evaluate_metric(self.metric, second)

        first_expected_scaled_numerator = -0.25
        second_expected_scaled_numerator = -0.75
        first_expected_value = -0.5
        second_expected_value = -1.0

        self.assertEqual(
            first_result.diagnostics["scaled_signed_numerator"],
            first_expected_scaled_numerator,
        )
        self.assertEqual(
            second_result.diagnostics["scaled_signed_numerator"],
            second_expected_scaled_numerator,
        )
        self.assertEqual(first_result.outputs[0].value, first_expected_value)
        self.assertEqual(second_result.outputs[0].value, second_expected_value)

    def test_input_arrays_and_read_only_state_are_unchanged(self) -> None:
        axis = np.asarray((0.0, 1.0, 2.0, 3.0), dtype="<f8")
        intensity = np.asarray((3.0, 1.0, 0.0, 2.0), dtype="<f8")
        axis.setflags(write=False)
        intensity.setflags(write=False)
        frozen_request = SpectrumRegionsInput(
            Spectrum1D(
                spectrum_id="spectrum-a",
                sample_id="sample-a",
                axis_cm1=axis,
                intensity=intensity,
            ),
            (
                SpectralRegion("signal-band", 0.0, 1.0, "signal"),
                SpectralRegion("ref-band", 2.0, 3.0, "reference"),
            ),
        )
        axis_before = axis.copy()
        intensity_before = intensity.copy()

        evaluate_metric(self.metric, frozen_request)

        np.testing.assert_array_equal(axis, axis_before)
        np.testing.assert_array_equal(intensity, intensity_before)
        self.assertFalse(axis.flags.writeable)
        self.assertFalse(intensity.flags.writeable)

    def test_repeat_evaluation_is_deterministic(self) -> None:
        first = evaluate_metric(self.metric, self.literal_request)
        second = evaluate_metric(self.metric, self.literal_request)

        self.assertEqual(first.metric_id, second.metric_id)
        self.assertEqual(first.input_kind, second.input_kind)
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertEqual(first.outputs, second.outputs)


if __name__ == "__main__":
    unittest.main()
