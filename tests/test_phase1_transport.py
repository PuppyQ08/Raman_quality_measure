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
    MetricInputKind,
    PreferredDirection,
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    TransportMetricError,
    Wasserstein1Metric,
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


class Wasserstein1MetricTest(unittest.TestCase):
    def test_identical_physical_measure_returns_zero(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(100.0, 101.0, 103.0, 106.0),
                intensity=(0.0, 3.0, 5.0, 1.0),
            ),
            spectrum(
                "candidate",
                axis=(100.0, 101.0, 103.0, 106.0),
                intensity=(0.0, 3.0, 5.0, 1.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertEqual(result.metric_id, "wasserstein_1_cm1")
        self.assertEqual(len(result.outputs), 1)
        output = result.outputs[0]
        self.assertEqual(output.output_id, "wasserstein_1_cm1")
        self.assertEqual(output.unit, "cm^-1")
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.LOWER_IS_BETTER,
        )
        self.assertIsNone(output.target_value)
        self.assertEqual(output.value, 0.0)
        self.assertEqual(
            dict(result.diagnostics),
            {
                "axis_equal": True,
                "candidate_negative_area_fraction": 0.0,
                "candidate_point_count": 4,
                "mass_construction": "positive_part_trapezoid_node_width",
                "mass_normalized": True,
                "reference_negative_area_fraction": 0.0,
                "reference_point_count": 4,
            },
        )

    def test_point_mass_at_zero_versus_one_returns_one(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0, 2.0),
                intensity=(2.0, 0.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, 1.0, 2.0),
                intensity=(0.0, 2.0, 0.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertAlmostEqual(result.outputs[0].value, 1.0, places=15)

    def test_equivalent_endpoint_measure_on_unequal_grids_returns_zero(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 2.0),
                intensity=(1.0, 1.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, 1.0, 2.0),
                intensity=(2.0, 0.0, 2.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertEqual(result.outputs[0].value, 0.0)
        self.assertEqual(
            dict(result.diagnostics),
            {
                "axis_equal": False,
                "candidate_negative_area_fraction": 0.0,
                "candidate_point_count": 3,
                "mass_construction": "positive_part_trapezoid_node_width",
                "mass_normalized": True,
                "reference_negative_area_fraction": 0.0,
                "reference_point_count": 2,
            },
        )

    def test_pure_translation_by_point_eight_returns_point_eight(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0, 2.0),
                intensity=(0.0, 2.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(0.8, 1.8, 2.8),
                intensity=(0.0, 2.0, 0.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertAlmostEqual(result.outputs[0].value, 0.8, places=15)

    def test_non_overlapping_point_masses_return_ten(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0, 2.0),
                intensity=(2.0, 0.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 11.0, 12.0),
                intensity=(2.0, 0.0, 0.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertAlmostEqual(result.outputs[0].value, 10.0, places=15)

    def test_positive_global_scaling_leaves_w1_unchanged(self):
        base_request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0, 2.0, 4.0),
                intensity=(0.0, 1.0, 4.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, 1.0, 2.0, 4.0),
                intensity=(0.0, 0.0, 5.0, 1.0),
            ),
        )
        scaled_request = SpectrumPairInput(
            spectrum(
                "reference-scaled",
                axis=(0.0, 1.0, 2.0, 4.0),
                intensity=(0.0, 11.0, 44.0, 0.0),
            ),
            spectrum(
                "candidate-scaled",
                axis=(0.0, 1.0, 2.0, 4.0),
                intensity=(0.0, 0.0, 35.0, 7.0),
            ),
        )

        base_result = evaluate_metric(Wasserstein1Metric(), base_request)
        scaled_result = evaluate_metric(Wasserstein1Metric(), scaled_request)

        self.assertAlmostEqual(
            base_result.outputs[0].value,
            scaled_result.outputs[0].value,
            places=15,
        )

    def test_accepts_unequal_length_and_grid_requests(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(100.0, 102.0, 106.0, 109.0),
                intensity=(0.0, 1.0, 3.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(100.5, 103.0, 108.0),
                intensity=(0.0, 2.0, 0.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertTrue(math.isfinite(result.outputs[0].value))
        self.assertGreater(result.outputs[0].value, 0.0)
        self.assertEqual(dict(result.diagnostics)["axis_equal"], False)

    def test_mixed_negative_intensities_use_positive_part_only(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0, 2.0),
                intensity=(-1.0, 2.0, -1.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, 1.0, 2.0),
                intensity=(0.0, 2.0, 0.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertEqual(result.outputs[0].value, 0.0)
        self.assertAlmostEqual(
            dict(result.diagnostics)["reference_negative_area_fraction"],
            1.0 / 3.0,
            places=15,
        )
        self.assertEqual(
            dict(result.diagnostics)["candidate_negative_area_fraction"],
            0.0,
        )

    def test_negative_area_fraction_matches_one_third(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0, 2.0),
                intensity=(-1.0, 2.0, -1.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, 1.0, 2.0),
                intensity=(0.0, 2.0, 0.0),
            ),
        )

        result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertAlmostEqual(
            dict(result.diagnostics)["reference_negative_area_fraction"],
            1.0 / 3.0,
            places=15,
        )

    def test_input_immutability_and_repeat_determinism(self):
        reference_axis = np.asarray((0.0, 1.0, 2.0), dtype="<f8")
        reference_intensity = np.asarray((0.0, 3.0, 0.0), dtype="<f8")
        candidate_axis = np.asarray((0.5, 1.5, 2.5), dtype="<f8")
        candidate_intensity = np.asarray((0.0, 3.0, 0.0), dtype="<f8")
        reference = Spectrum1D(
            "reference",
            "sample-a",
            reference_axis,
            reference_intensity,
        )
        candidate = Spectrum1D(
            "candidate",
            "sample-a",
            candidate_axis,
            candidate_intensity,
        )
        request = SpectrumPairInput(reference, candidate)
        metric = Wasserstein1Metric()

        first = evaluate_metric(metric, request)
        second = evaluate_metric(metric, request)

        self.assertAlmostEqual(
            first.outputs[0].value,
            second.outputs[0].value,
            places=15,
        )
        np.testing.assert_array_equal(
            reference.axis_cm1,
            np.asarray((0.0, 1.0, 2.0), dtype="<f8"),
        )
        np.testing.assert_array_equal(
            reference.intensity,
            np.asarray((0.0, 3.0, 0.0), dtype="<f8"),
        )
        np.testing.assert_array_equal(
            candidate.axis_cm1,
            np.asarray((0.5, 1.5, 2.5), dtype="<f8"),
        )
        np.testing.assert_array_equal(
            candidate.intensity,
            np.asarray((0.0, 3.0, 0.0), dtype="<f8"),
        )
        self.assertFalse(reference.axis_cm1.flags.writeable)
        self.assertFalse(reference.intensity.flags.writeable)
        self.assertFalse(candidate.axis_cm1.flags.writeable)
        self.assertFalse(candidate.intensity.flags.writeable)

    def test_constructor_identity_is_frozen(self):
        metric = Wasserstein1Metric()

        self.assertEqual(metric.metric_id, "wasserstein_1_cm1")
        self.assertEqual(metric.input_kind, MetricInputKind.SPECTRUM_PAIR)
        self.assertEqual(
            metric.axis_policy,
            AxisPolicy.INDEPENDENT_PHYSICAL_AXES,
        )
        self.assertEqual(metric.output_id, "wasserstein_1_cm1")
        self.assertEqual(metric.unit, "cm^-1")
        self.assertEqual(
            metric.preferred_direction,
            PreferredDirection.LOWER_IS_BETTER,
        )
        with self.assertRaises(TypeError):
            Wasserstein1Metric(metric_id="other")

    def test_rejects_one_point_reference(self):
        request = SpectrumPairInput(
            spectrum("reference", axis=(0.0,), intensity=(1.0,)),
            spectrum(
                "candidate",
                axis=(0.0, 1.0),
                intensity=(1.0, 1.0),
            ),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            "reference point count: must be at least 2",
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_one_point_candidate(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0),
                intensity=(1.0, 1.0),
            ),
            spectrum("candidate", axis=(0.0,), intensity=(1.0,)),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            "candidate point count: must be at least 2",
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_all_zero_positive_mass(self):
        cases = (
            (
                SpectrumPairInput(
                    spectrum(
                        "reference",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(0.0, 0.0, 0.0),
                    ),
                    spectrum(
                        "candidate",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(0.0, 2.0, 0.0),
                    ),
                ),
                "reference positive mass",
            ),
            (
                SpectrumPairInput(
                    spectrum(
                        "reference",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(0.0, 2.0, 0.0),
                    ),
                    spectrum(
                        "candidate",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(0.0, 0.0, 0.0),
                    ),
                ),
                "candidate positive mass",
            ),
        )

        for request, path in cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    TransportMetricError,
                    f"{path}: must be positive and finite",
                ):
                    evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_all_negative_positive_mass(self):
        cases = (
            (
                SpectrumPairInput(
                    spectrum(
                        "reference",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(-1.0, -2.0, -3.0),
                    ),
                    spectrum(
                        "candidate",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(0.0, 2.0, 0.0),
                    ),
                ),
                "reference positive mass",
            ),
            (
                SpectrumPairInput(
                    spectrum(
                        "reference",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(0.0, 2.0, 0.0),
                    ),
                    spectrum(
                        "candidate",
                        axis=(0.0, 1.0, 2.0),
                        intensity=(-1.0, -2.0, -3.0),
                    ),
                ),
                "candidate positive mass",
            ),
        )

        for request, path in cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    TransportMetricError,
                    f"{path}: must be positive and finite",
                ):
                    evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_nonfinite_derived_width_from_axis_interval_overflow(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(-1.0e308, 0.0, 1.0e308),
                intensity=(1.0, 1.0, 1.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, 1.0),
                intensity=(1.0, 1.0),
            ),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(
                TransportMetricError,
                "reference widths: contains non-finite derived values",
            ):
                evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_nonfinite_union_interval(self):
        finite_max = np.finfo(np.float64).max
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(-finite_max, -0.75 * finite_max),
                intensity=(1.0, 1.0),
            ),
            spectrum(
                "candidate",
                axis=(0.75 * finite_max, finite_max),
                intensity=(1.0, 1.0),
            ),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(
                TransportMetricError,
                "support intervals: contains non-finite derived values",
            ):
                evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_nonfinite_output(self):
        finite_max = np.finfo(np.float64).max
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(-finite_max, -finite_max / 2.0, 0.0),
                intensity=(2.0, 0.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, finite_max / 2.0, finite_max),
                intensity=(0.0, 0.0, 2.0),
            ),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            "wasserstein_1_cm1: produced non-finite value",
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_positive_mass_underflow_during_normalization(self):
        distance = math.ldexp(1.0, 1000)
        dominant = math.ldexp(1.0, 500)
        tiny = math.ldexp(1.0, -600)
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, distance),
                intensity=(dominant, tiny),
            ),
            spectrum(
                "candidate",
                axis=(0.0, distance),
                intensity=(dominant, 0.0),
            ),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            (
                "reference positive mass: "
                "positive mass underflowed during normalization"
            ),
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_negative_area_underflow_in_diagnostics(self):
        distance = math.ldexp(1.0, 1000)
        dominant = math.ldexp(1.0, 500)
        tiny = math.ldexp(1.0, -600)
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, distance),
                intensity=(dominant, -tiny),
            ),
            spectrum(
                "candidate",
                axis=(0.0, distance),
                intensity=(dominant, 0.0),
            ),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            (
                "reference negative area fraction: "
                "weighted contribution underflowed during scaling"
            ),
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_rejects_positive_probability_lost_during_cdf_accumulation(self):
        distance = math.ldexp(1.0, 1000)
        dominant = math.ldexp(1.0, 500)
        tail = math.ldexp(1.0, 400)
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, distance),
                intensity=(dominant, tail),
            ),
            spectrum(
                "candidate",
                axis=(0.0, distance),
                intensity=(dominant, 0.0),
            ),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            "reference cdf: positive probability lost during accumulation",
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_identical_tiny_terminal_probability_returns_zero_without_warning(self):
        distance = math.ldexp(1.0, 1000)
        dominant = math.ldexp(1.0, 500)
        tail = math.ldexp(1.0, 400)
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, distance),
                intensity=(dominant, tail),
            ),
            spectrum(
                "candidate",
                axis=(0.0, distance),
                intensity=(dominant, tail),
            ),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertEqual(result.outputs[0].value, 0.0)
        self.assertEqual(
            dict(result.diagnostics),
            {
                "axis_equal": True,
                "candidate_negative_area_fraction": 0.0,
                "candidate_point_count": 2,
                "mass_construction": "positive_part_trapezoid_node_width",
                "mass_normalized": True,
                "reference_negative_area_fraction": 0.0,
                "reference_point_count": 2,
            },
        )

    def test_equal_axis_but_different_normalized_measure_still_rejects_cdf_loss(self):
        distance = math.ldexp(1.0, 1000)
        dominant = math.ldexp(1.0, 500)
        tail = math.ldexp(1.0, 400)
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, distance),
                intensity=(dominant, tail),
            ),
            spectrum(
                "candidate",
                axis=(0.0, distance),
                intensity=(dominant, 2.0 * tail),
            ),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            "reference cdf: positive probability lost during accumulation",
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_same_normalized_probability_but_different_raw_intensity_does_not_short_circuit(self):
        distance = math.ldexp(1.0, 1000)
        dominant = math.ldexp(1.0, 500)
        tail = math.ldexp(1.0, 400)
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, distance),
                intensity=(dominant, tail),
            ),
            spectrum(
                "candidate",
                axis=(0.0, distance),
                intensity=(2.0 * dominant, 2.0 * tail),
            ),
        )

        with self.assertRaisesRegex(
            TransportMetricError,
            "reference cdf: positive probability lost during accumulation",
        ):
            evaluate_metric(Wasserstein1Metric(), request)

    def test_accepts_extreme_finite_positive_mass_without_warning_paths(self):
        tiny = float(np.nextafter(0.0, 1.0))
        huge = float(np.finfo(np.float64).max)
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(0.0, 1.0, 3.0),
                intensity=(0.0, tiny, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(0.0, 1.0, 3.0),
                intensity=(0.0, huge, 0.0),
            ),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = evaluate_metric(Wasserstein1Metric(), request)

        self.assertEqual(result.outputs[0].value, 0.0)


if __name__ == "__main__":
    unittest.main()
