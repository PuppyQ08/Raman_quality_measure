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
    EvaluationContractError,
    PreferredDirection,
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    FidelityMetricError,
    MAEMetric,
    MSEMetric,
    NMSEMetric,
    PearsonRMetric,
    RMSEMetric,
    SAMMetric,
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


class FidelityMetricAnalyticTest(unittest.TestCase):
    def setUp(self):
        self.reference = spectrum(
            "reference",
            axis=(100.0, 200.0, 300.0, 400.0),
            intensity=(1.0, 2.0, 3.0, 4.0),
        )
        self.candidate = spectrum(
            "candidate",
            axis=(100.0, 200.0, 300.0, 400.0),
            intensity=(2.0, 0.0, 3.0, 8.0),
        )
        self.shifted_candidate = spectrum(
            "shifted",
            axis=(100.5, 200.5, 300.5, 400.5),
            intensity=(1.0, 2.0, 3.0, 4.0),
        )
        self.request = SpectrumPairInput(self.reference, self.candidate)
        self.identical = SpectrumPairInput(
            self.reference,
            spectrum(
                "copy",
                axis=(100.0, 200.0, 300.0, 400.0),
                intensity=(1.0, 2.0, 3.0, 4.0),
            ),
        )

    def assert_scalar_result(
        self,
        metric,
        *,
        expected_id: str,
        expected_unit: str,
        expected_value: float,
        axis_equal: bool = True,
        point_count: int = 4,
    ):
        result = evaluate_metric(metric, self.request)
        self.assertEqual(result.metric_id, metric.metric_id)
        self.assertEqual(result.input_kind, metric.input_kind)
        self.assertEqual(len(result.outputs), 1)
        output = result.outputs[0]
        self.assertEqual(output.output_id, expected_id)
        self.assertEqual(output.unit, expected_unit)
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.LOWER_IS_BETTER,
        )
        self.assertIsNone(output.target_value)
        self.assertAlmostEqual(output.value, expected_value, places=15)
        self.assertEqual(
            dict(result.diagnostics),
            {"axis_equal": axis_equal, "point_count": point_count},
        )

    def test_mse_matches_hand_derived_oracle(self):
        self.assert_scalar_result(
            MSEMetric(),
            expected_id="mse",
            expected_unit="intensity_squared",
            expected_value=5.25,
        )

    def test_rmse_matches_hand_derived_oracle(self):
        self.assert_scalar_result(
            RMSEMetric(),
            expected_id="rmse",
            expected_unit="intensity",
            expected_value=math.sqrt(5.25),
        )

    def test_mae_matches_hand_derived_oracle(self):
        self.assert_scalar_result(
            MAEMetric(),
            expected_id="mae",
            expected_unit="intensity",
            expected_value=1.75,
        )

    def test_nmse_matches_hand_derived_oracle(self):
        self.assert_scalar_result(
            NMSEMetric(),
            expected_id="nmse",
            expected_unit="ratio",
            expected_value=0.7,
        )

    def test_sam_matches_hand_derived_oracle(self):
        expected = math.acos(
            43.0 / (math.sqrt(30.0) * math.sqrt(77.0))
        )
        self.assert_scalar_result(
            SAMMetric(),
            expected_id="sam",
            expected_unit="radian",
            expected_value=expected,
        )

    def test_pearson_matches_hand_derived_oracle(self):
        expected = 21.0 / math.sqrt(695.0)
        result = evaluate_metric(PearsonRMetric(), self.request)
        self.assertEqual(result.metric_id, "pearson_r")
        self.assertEqual(len(result.outputs), 1)
        output = result.outputs[0]
        self.assertEqual(output.output_id, "pearson_r")
        self.assertEqual(output.unit, "correlation")
        self.assertEqual(
            output.preferred_direction,
            PreferredDirection.HIGHER_IS_BETTER,
        )
        self.assertAlmostEqual(output.value, expected, places=15)
        self.assertEqual(
            dict(result.diagnostics),
            {"axis_equal": True, "point_count": 4},
        )

    def test_pearson_returns_one_for_identical_nonconstant_vectors(self):
        result = evaluate_metric(PearsonRMetric(), self.identical)
        self.assertEqual(result.outputs[0].value, 1.0)
        self.assertEqual(result.outputs[0].unit, "correlation")

    def test_pearson_returns_minus_one_for_perfect_reversal(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(1.0, 2.0, 3.0, 4.0),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(4.0, 3.0, 2.0, 1.0),
            ),
        )
        result = evaluate_metric(PearsonRMetric(), request)
        self.assertEqual(result.outputs[0].value, -1.0)

    def test_pearson_is_invariant_to_positive_affine_transform(self):
        request = SpectrumPairInput(
            self.reference,
            spectrum(
                "candidate",
                axis=(100.0, 200.0, 300.0, 400.0),
                intensity=(7.0, 9.0, 11.0, 13.0),
            ),
        )
        result = evaluate_metric(PearsonRMetric(), request)
        self.assertAlmostEqual(result.outputs[0].value, 1.0, places=15)

    def test_pearson_accepts_shifted_axis_when_point_count_matches(self):
        request = SpectrumPairInput(self.reference, self.shifted_candidate)
        result = evaluate_metric(PearsonRMetric(), request)
        self.assertEqual(result.outputs[0].value, 1.0)
        self.assertEqual(
            dict(result.diagnostics),
            {"axis_equal": False, "point_count": 4},
        )

    def test_pearson_accepts_negative_intensities(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(-3.0, -1.0, 1.0, 3.0),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(-7.0, -3.0, 1.0, 5.0),
            ),
        )
        result = evaluate_metric(PearsonRMetric(), request)
        self.assertEqual(result.outputs[0].value, 1.0)

    def test_pearson_accepts_extreme_finite_same_direction_centered_vectors(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    1.0e300,
                    2.0e300,
                    3.0e300,
                    4.0e300,
                ),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    5.0e299,
                    1.0e300,
                    1.5e300,
                    2.0e300,
                ),
            ),
        )
        try:
            result = evaluate_metric(PearsonRMetric(), request)
        except FidelityMetricError as error:
            self.fail(f"unexpected Pearson rejection for finite vectors: {error}")
        self.assertAlmostEqual(result.outputs[0].value, 1.0, places=15)

    def test_pearson_large_common_offset_regression_returns_one(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    1.0e16,
                    1.0e16 + 2.0,
                    1.0e16 + 4.0,
                    1.0e16 + 8.0,
                ),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    1.0e16 + 10.0,
                    1.0e16 + 14.0,
                    1.0e16 + 18.0,
                    1.0e16 + 26.0,
                ),
            ),
        )
        result = evaluate_metric(PearsonRMetric(), request)
        self.assertAlmostEqual(result.outputs[0].value, 1.0, places=15)

    def test_pearson_cross_zero_extreme_affine_regression_returns_one_without_warning(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0, 30.0, 40.0),
                intensity=(
                    -1.0e308,
                    -5.0e307,
                    5.0e307,
                    1.0e308,
                ),
            ),
            spectrum(
                "candidate",
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
            result = evaluate_metric(PearsonRMetric(), request)
        self.assertAlmostEqual(result.outputs[0].value, 1.0, places=15)

    def test_identical_vectors_return_zero_for_all_metrics(self):
        metrics = (
            MSEMetric(),
            RMSEMetric(),
            MAEMetric(),
            NMSEMetric(),
            SAMMetric(),
        )
        for metric in metrics:
            with self.subTest(metric=metric.metric_id):
                result = evaluate_metric(metric, self.identical)
                self.assertEqual(result.outputs[0].value, 0.0)
                self.assertEqual(
                    dict(result.diagnostics),
                    {"axis_equal": True, "point_count": 4},
                )

    def test_sam_is_pi_over_two_for_orthogonal_vectors(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0),
                intensity=(1.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0),
                intensity=(0.0, 1.0),
            ),
        )
        result = evaluate_metric(SAMMetric(), request)
        self.assertAlmostEqual(result.outputs[0].value, math.pi / 2.0, places=15)

    def test_sam_clips_near_collinear_cosine_above_one(self):
        reference = spectrum(
            "reference",
            axis=(10.0, 20.0),
            intensity=(1.0e154, 1.0e154),
        )
        candidate = spectrum(
            "candidate",
            axis=(10.0, 20.0),
            intensity=(1.0e154, 1.0e154 + 1.0e138),
        )
        result = evaluate_metric(
            SAMMetric(),
            SpectrumPairInput(reference, candidate),
        )
        self.assertEqual(result.outputs[0].value, 0.0)

    def test_sam_accepts_extreme_same_direction_finite_vectors(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0),
                intensity=(1.0e-300, 1.0e-300),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0),
                intensity=(1.0e300, 1.0e300),
            ),
        )
        try:
            result = evaluate_metric(SAMMetric(), request)
        except FidelityMetricError as error:
            self.fail(f"unexpected SAM rejection for finite same-direction vectors: {error}")
        self.assertLess(abs(result.outputs[0].value), 1.0e-7)

    def test_shifted_axis_with_same_intensities_is_accepted_by_every_metric(self):
        request = SpectrumPairInput(self.reference, self.shifted_candidate)
        metrics = (
            MSEMetric(),
            RMSEMetric(),
            MAEMetric(),
            NMSEMetric(),
            SAMMetric(),
        )
        for metric in metrics:
            with self.subTest(metric=metric.metric_id):
                result = evaluate_metric(metric, request)
                self.assertEqual(result.outputs[0].value, 0.0)
                self.assertEqual(
                    dict(result.diagnostics),
                    {"axis_equal": False, "point_count": 4},
                )

    def test_repeated_evaluation_is_scientifically_identical(self):
        metric = SAMMetric()
        first = evaluate_metric(metric, self.request)
        second = evaluate_metric(metric, self.request)
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


class FidelityMetricEdgeCaseTest(unittest.TestCase):
    def test_tiny_residual_keeps_representable_rmse_when_mse_underflows(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0),
                intensity=(0.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0),
                intensity=(1.0e-300, 1.0e-300),
            ),
        )

        mse = evaluate_metric(MSEMetric(), request).outputs[0].value
        rmse = evaluate_metric(RMSEMetric(), request).outputs[0].value
        mae = evaluate_metric(MAEMetric(), request).outputs[0].value

        self.assertEqual(mse, 0.0)
        np.testing.assert_allclose(rmse, 1.0e-300, rtol=1.0e-15, atol=0.0)
        np.testing.assert_allclose(mae, 1.0e-300, rtol=1.0e-15, atol=0.0)

    def test_sam_rejects_zero_norm_reference_or_candidate(self):
        zero = spectrum(
            "zero",
            axis=(10.0, 20.0),
            intensity=(0.0, 0.0),
        )
        nonzero = spectrum(
            "nonzero",
            axis=(10.0, 20.0),
            intensity=(1.0, 2.0),
        )
        cases = (
            SpectrumPairInput(zero, nonzero),
            SpectrumPairInput(nonzero, zero),
        )
        for request in cases:
            with self.subTest(reference=request.reference.spectrum_id):
                with self.assertRaisesRegex(
                    FidelityMetricError,
                    "zero norm",
                ):
                    evaluate_metric(SAMMetric(), request)

    def test_nmse_rejects_zero_energy_reference(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0),
                intensity=(0.0, 0.0),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0),
                intensity=(1.0, 2.0),
            ),
        )
        with self.assertRaisesRegex(
            FidelityMetricError,
            "reference energy",
        ):
            evaluate_metric(NMSEMetric(), request)

    def test_pearson_rejects_zero_variance_reference_or_candidate(self):
        constant = spectrum(
            "constant",
            axis=(10.0, 20.0, 30.0, 40.0),
            intensity=(5.0, 5.0, 5.0, 5.0),
        )
        varying = spectrum(
            "varying",
            axis=(10.0, 20.0, 30.0, 40.0),
            intensity=(1.0, 2.0, 3.0, 4.0),
        )
        cases = (
            (
                SpectrumPairInput(constant, varying),
                "reference zero variance",
            ),
            (
                SpectrumPairInput(varying, constant),
                "candidate zero variance",
            ),
        )
        for request, expected_path in cases:
            with self.subTest(path=expected_path):
                with self.assertRaisesRegex(
                    FidelityMetricError,
                    f"{expected_path}: zero variance",
                ):
                    evaluate_metric(PearsonRMetric(), request)

    def test_metrics_reject_unequal_point_count(self):
        request = SpectrumPairInput(
            spectrum(
                "reference",
                axis=(10.0, 20.0, 30.0),
                intensity=(1.0, 2.0, 3.0),
            ),
            spectrum(
                "candidate",
                axis=(10.0, 20.0),
                intensity=(1.0, 2.0),
            ),
        )
        metrics = (
            MSEMetric(),
            RMSEMetric(),
            MAEMetric(),
            NMSEMetric(),
            SAMMetric(),
        )
        for metric in metrics:
            with self.subTest(metric=metric.metric_id):
                with self.assertRaisesRegex(
                    EvaluationContractError,
                    "point count",
                ):
                    evaluate_metric(metric, request)

    def test_metric_input_arrays_remain_read_only_and_unchanged(self):
        reference = spectrum(
            "reference",
            axis=(100.0, 200.0, 300.0, 400.0),
            intensity=(1.0, 2.0, 3.0, 4.0),
        )
        candidate = spectrum(
            "candidate",
            axis=(100.5, 200.5, 300.5, 400.5),
            intensity=(2.0, 0.0, 3.0, 8.0),
        )
        request = SpectrumPairInput(reference, candidate)
        before_ref_axis = reference.axis_cm1.copy()
        before_ref_intensity = reference.intensity.copy()
        before_cand_axis = candidate.axis_cm1.copy()
        before_cand_intensity = candidate.intensity.copy()

        evaluate_metric(MSEMetric(), request)
        evaluate_metric(SAMMetric(), request)
        evaluate_metric(PearsonRMetric(), request)

        np.testing.assert_array_equal(reference.axis_cm1, before_ref_axis)
        np.testing.assert_array_equal(
            reference.intensity,
            before_ref_intensity,
        )
        np.testing.assert_array_equal(candidate.axis_cm1, before_cand_axis)
        np.testing.assert_array_equal(
            candidate.intensity,
            before_cand_intensity,
        )
        self.assertFalse(reference.axis_cm1.flags.writeable)
        self.assertFalse(reference.intensity.flags.writeable)
        self.assertFalse(candidate.axis_cm1.flags.writeable)
        self.assertFalse(candidate.intensity.flags.writeable)

    def test_public_metric_ids_and_policies_cannot_be_overridden(self):
        for metric_type in (
            MSEMetric,
            RMSEMetric,
            MAEMetric,
            SAMMetric,
            NMSEMetric,
            PearsonRMetric,
        ):
            with self.subTest(metric=metric_type.__name__):
                with self.assertRaises(TypeError):
                    metric_type(metric_id="override")


if __name__ == "__main__":
    unittest.main()
