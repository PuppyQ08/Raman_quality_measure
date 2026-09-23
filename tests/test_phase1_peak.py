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
    Peak1D,
    PeakPairInput,
    PreferredDirection,
    SingleSpectrumInput,
    Spectrum1D,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    MatchedPeakErrorMetric,
    PeakDetectionCurvesMetric,
    PeakMetricError,
)


def peak(
    position_cm1: float,
    *,
    height: float | None = None,
    fwhm_cm1: float | None = None,
    area: float | None = None,
    prominence: float | None = None,
) -> Peak1D:
    return Peak1D(
        position_cm1=position_cm1,
        height=height,
        fwhm_cm1=fwhm_cm1,
        area=area,
        prominence=prominence,
    )


def request(
    *,
    reference_peaks: tuple[Peak1D, ...],
    candidate_peaks: tuple[Peak1D, ...],
    tolerance: float = 2.0,
    thresholds: tuple[float, ...] = (0.0, 4.0),
) -> PeakPairInput:
    return PeakPairInput(
        reference_peaks=reference_peaks,
        candidate_peaks=candidate_peaks,
        position_tolerance_cm1=tolerance,
        prominence_thresholds=thresholds,
    )


def spectrum() -> Spectrum1D:
    return Spectrum1D(
        spectrum_id="spectrum-a",
        sample_id="sample-a",
        axis_cm1=np.asarray((100.0, 101.0), dtype="<f8"),
        intensity=np.asarray((1.0, 2.0), dtype="<f8"),
    )


class PeakDetectionCurvesMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = PeakDetectionCurvesMetric()
        self.request = request(
            reference_peaks=(
                peak(100.0, prominence=5.0),
                peak(200.0, prominence=15.0),
                peak(300.0, prominence=2.0),
            ),
            candidate_peaks=(
                peak(101.0, prominence=6.0),
                peak(198.0, prominence=10.0),
                peak(201.0, prominence=12.0),
                peak(400.0, prominence=8.0),
            ),
        )

    def test_literal_detection_oracle_outputs_curves_and_diagnostics(self) -> None:
        result = evaluate_metric(self.metric, self.request)

        self.assertEqual(result.metric_id, "peak_detection_curves")
        self.assertEqual(result.input_kind, MetricInputKind.PEAK_PAIR)
        self.assertEqual(self.metric.input_kind, MetricInputKind.PEAK_PAIR)
        self.assertEqual(self.metric.axis_policy, AxisPolicy.NOT_APPLICABLE)

        self.assertEqual(
            [output.output_id for output in result.outputs],
            [
                "precision",
                "recall",
                "f1",
                "artifact_peak_ratio",
                "missing_peak_ratio",
                "tolerance_scan",
                "prominence_scan",
            ],
        )

        precision, recall, f1, artifact, missing = result.outputs[:5]
        self.assertAlmostEqual(precision.value, 0.5, places=15)
        self.assertAlmostEqual(recall.value, 2.0 / 3.0, places=15)
        self.assertAlmostEqual(f1.value, 4.0 / 7.0, places=15)
        self.assertAlmostEqual(artifact.value, 0.5, places=15)
        self.assertAlmostEqual(missing.value, 1.0 / 3.0, places=15)

        self.assertEqual(precision.unit, "ratio")
        self.assertEqual(recall.unit, "ratio")
        self.assertEqual(f1.unit, "ratio")
        self.assertEqual(artifact.unit, "ratio")
        self.assertEqual(missing.unit, "ratio")
        self.assertEqual(
            [
                precision.preferred_direction,
                recall.preferred_direction,
                f1.preferred_direction,
                artifact.preferred_direction,
                missing.preferred_direction,
            ],
            [
                PreferredDirection.HIGHER_IS_BETTER,
                PreferredDirection.HIGHER_IS_BETTER,
                PreferredDirection.HIGHER_IS_BETTER,
                PreferredDirection.LOWER_IS_BETTER,
                PreferredDirection.LOWER_IS_BETTER,
            ],
        )

        tolerance_scan = result.outputs[5]
        self.assertEqual(tolerance_scan.output_id, "tolerance_scan")
        self.assertEqual(tolerance_scan.x_name, "position_tolerance")
        self.assertEqual(tolerance_scan.x_values, (1.0, 2.0, 4.0, 8.0))
        self.assertEqual(tolerance_scan.x_unit, "cm^-1")
        self.assertEqual(
            [series.series_id for series in tolerance_scan.series],
            [
                "precision",
                "recall",
                "f1",
                "artifact_peak_ratio",
                "missing_peak_ratio",
            ],
        )
        self.assertEqual(
            [series.unit for series in tolerance_scan.series],
            ["ratio", "ratio", "ratio", "ratio", "ratio"],
        )
        self.assertEqual(
            [series.preferred_direction for series in tolerance_scan.series],
            [
                PreferredDirection.HIGHER_IS_BETTER,
                PreferredDirection.HIGHER_IS_BETTER,
                PreferredDirection.HIGHER_IS_BETTER,
                PreferredDirection.LOWER_IS_BETTER,
                PreferredDirection.LOWER_IS_BETTER,
            ],
        )
        self.assertEqual(
            [series.values for series in tolerance_scan.series],
            [
                (0.5, 0.5, 0.5, 0.5),
                (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
                (4.0 / 7.0, 4.0 / 7.0, 4.0 / 7.0, 4.0 / 7.0),
                (0.5, 0.5, 0.5, 0.5),
                (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            ],
        )

        prominence_scan = result.outputs[6]
        self.assertEqual(prominence_scan.output_id, "prominence_scan")
        self.assertEqual(prominence_scan.x_name, "prominence_threshold")
        self.assertEqual(prominence_scan.x_values, (0.0, 4.0))
        self.assertEqual(prominence_scan.x_unit, "intensity")
        self.assertEqual(
            [series.series_id for series in prominence_scan.series],
            [
                "precision",
                "recall",
                "f1",
                "artifact_peak_ratio",
                "missing_peak_ratio",
            ],
        )
        self.assertEqual(
            [series.values for series in prominence_scan.series],
            [
                (0.5, 0.5),
                (2.0 / 3.0, 1.0),
                (4.0 / 7.0, 2.0 / 3.0),
                (0.5, 0.5),
                (1.0 / 3.0, 0.0),
            ],
        )

        self.assertEqual(
            dict(result.diagnostics),
            {
                "matching_policy": (
                    "distance_sorted_greedy_reference_then_candidate"
                ),
                "tolerance_boundary": "inclusive",
                "position_tolerance_cm1": 2.0,
                "tolerance_scan_grid_cm1": (1, 2, 4, 8),
                "prominence_filter": (
                    "both_sides_inclusive_requires_present"
                ),
                "reference_peak_count": 3,
                "candidate_peak_count": 4,
                "matched_pair_count_at_request_tolerance": 2,
            },
        )

    def test_inclusive_tolerance_boundary_allows_exact_distance_match(self) -> None:
        exact = request(
            reference_peaks=(peak(100.0, prominence=1.0),),
            candidate_peaks=(peak(102.0, prominence=1.0),),
            tolerance=2.0,
            thresholds=(0.0,),
        )

        result = evaluate_metric(self.metric, exact)
        self.assertEqual(
            [output.value for output in result.outputs[:5]],
            [1.0, 1.0, 1.0, 0.0, 0.0],
        )
        self.assertEqual(
            dict(result.diagnostics)["matched_pair_count_at_request_tolerance"],
            1,
        )

    def test_prominence_filter_is_symmetric_and_inclusive(self) -> None:
        filtered = request(
            reference_peaks=(
                peak(100.0, prominence=4.0),
                peak(200.0, prominence=3.0),
            ),
            candidate_peaks=(
                peak(100.0, prominence=4.0),
                peak(201.0, prominence=3.0),
            ),
            tolerance=1.0,
            thresholds=(4.0,),
        )

        result = evaluate_metric(self.metric, filtered)
        prominence_scan = result.outputs[6]
        self.assertEqual(prominence_scan.x_values, (4.0,))
        self.assertEqual(
            [series.values for series in prominence_scan.series],
            [
                (1.0,),
                (1.0,),
                (1.0,),
                (0.0,),
                (0.0,),
            ],
        )

    def test_empty_set_conventions_are_exact_and_finite(self) -> None:
        both_empty = request(
            reference_peaks=(),
            candidate_peaks=(),
            thresholds=(0.0,),
        )
        reference_empty = request(
            reference_peaks=(),
            candidate_peaks=(peak(101.0, prominence=1.0),),
            thresholds=(0.0,),
        )
        candidate_empty = request(
            reference_peaks=(peak(101.0, prominence=1.0),),
            candidate_peaks=(),
            thresholds=(0.0,),
        )

        both_empty_result = evaluate_metric(self.metric, both_empty)
        reference_empty_result = evaluate_metric(self.metric, reference_empty)
        candidate_empty_result = evaluate_metric(self.metric, candidate_empty)

        self.assertEqual(
            [output.value for output in both_empty_result.outputs[:5]],
            [1.0, 1.0, 1.0, 0.0, 0.0],
        )
        self.assertEqual(
            [output.value for output in reference_empty_result.outputs[:5]],
            [0.0, 1.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(
            [output.value for output in candidate_empty_result.outputs[:5]],
            [1.0, 0.0, 0.0, 0.0, 1.0],
        )

    def test_missing_prominence_is_rejected_for_curve_metric(self) -> None:
        missing_reference = request(
            reference_peaks=(peak(100.0, prominence=None),),
            candidate_peaks=(peak(100.0, prominence=1.0),),
            thresholds=(0.0,),
        )
        missing_candidate = request(
            reference_peaks=(peak(100.0, prominence=1.0),),
            candidate_peaks=(peak(100.0, prominence=None),),
            thresholds=(0.0,),
        )

        with self.assertRaisesRegex(
            PeakMetricError,
            "^reference_peaks\\[0\\]\\.prominence: must be present for prominence scan$",
        ):
            self.metric.evaluate(missing_reference)
        with self.assertRaisesRegex(
            PeakMetricError,
            "^candidate_peaks\\[0\\]\\.prominence: must be present for prominence scan$",
        ):
            self.metric.evaluate(missing_candidate)

    def test_wrong_request_type_frozen_identity_immutability_and_repeat_determinism(
        self,
    ) -> None:
        wrong_request = SingleSpectrumInput(spectrum())
        with self.assertRaisesRegex(
            PeakMetricError,
            "^request: must be PeakPairInput$",
        ):
            self.metric.evaluate(wrong_request)

        with self.assertRaises(TypeError):
            PeakDetectionCurvesMetric(metric_id="override")
        with self.assertRaises(FrozenInstanceError):
            self.metric.metric_id = "override"
        with self.assertRaises(FrozenInstanceError):
            self.request.position_tolerance_cm1 = 3.0

        first = evaluate_metric(self.metric, self.request)
        second = evaluate_metric(self.metric, self.request)
        self.assertEqual(first, second)


class MatchedPeakErrorMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = MatchedPeakErrorMetric()
        self.request = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=10.0,
                    fwhm_cm1=4.0,
                    area=20.0,
                    prominence=5.0,
                ),
                peak(
                    200.0,
                    height=20.0,
                    fwhm_cm1=8.0,
                    area=50.0,
                    prominence=15.0,
                ),
                peak(
                    300.0,
                    height=5.0,
                    fwhm_cm1=3.0,
                    area=12.0,
                    prominence=2.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    101.0,
                    height=9.0,
                    fwhm_cm1=5.0,
                    area=18.0,
                    prominence=6.0,
                ),
                peak(
                    198.0,
                    height=30.0,
                    fwhm_cm1=9.0,
                    area=70.0,
                    prominence=10.0,
                ),
                peak(
                    201.0,
                    height=22.0,
                    fwhm_cm1=7.0,
                    area=55.0,
                    prominence=12.0,
                ),
                peak(
                    400.0,
                    height=4.0,
                    fwhm_cm1=2.0,
                    area=10.0,
                    prominence=8.0,
                ),
            ),
        )

    def test_literal_error_oracle_outputs_and_diagnostics(self) -> None:
        result = evaluate_metric(self.metric, self.request)

        self.assertEqual(result.metric_id, "matched_peak_errors")
        self.assertEqual(result.input_kind, MetricInputKind.PEAK_PAIR)
        self.assertEqual(self.metric.input_kind, MetricInputKind.PEAK_PAIR)
        self.assertEqual(self.metric.axis_policy, AxisPolicy.NOT_APPLICABLE)
        self.assertEqual(
            [output.output_id for output in result.outputs],
            [
                "peak_position_mae_cm1",
                "relative_peak_height_mae",
                "peak_fwhm_mae_cm1",
                "peak_area_mae",
            ],
        )
        self.assertEqual(
            [output.unit for output in result.outputs],
            ["cm^-1", "ratio", "cm^-1", "intensity_cm^-1"],
        )
        self.assertEqual(
            [output.preferred_direction for output in result.outputs],
            [
                PreferredDirection.LOWER_IS_BETTER,
                PreferredDirection.LOWER_IS_BETTER,
                PreferredDirection.LOWER_IS_BETTER,
                PreferredDirection.LOWER_IS_BETTER,
            ],
        )
        self.assertEqual(
            [output.value for output in result.outputs],
            [1.0, 0.1, 1.0, 3.5],
        )
        self.assertEqual(
            dict(result.diagnostics),
            {
                "matching_policy": (
                    "distance_sorted_greedy_reference_then_candidate"
                ),
                "tolerance_boundary": "inclusive",
                "position_tolerance_cm1": 2.0,
                "reference_peak_count": 3,
                "candidate_peak_count": 4,
                "matched_pair_count_at_request_tolerance": 2,
                "evaluated_pair_count": 2,
            },
        )

    def test_greedy_tie_resolution_prefers_reference_then_candidate(self) -> None:
        tied = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=10.0,
                    fwhm_cm1=4.0,
                    area=20.0,
                    prominence=1.0,
                ),
                peak(
                    102.0,
                    height=20.0,
                    fwhm_cm1=4.0,
                    area=20.0,
                    prominence=1.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    101.0,
                    height=10.0,
                    fwhm_cm1=4.0,
                    area=20.0,
                    prominence=1.0,
                ),
            ),
            tolerance=1.0,
            thresholds=(0.0,),
        )

        result = evaluate_metric(self.metric, tied)
        self.assertEqual(
            [output.value for output in result.outputs],
            [1.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            dict(result.diagnostics)["matched_pair_count_at_request_tolerance"],
            1,
        )

    def test_missing_matched_properties_zero_reference_height_and_no_matches_reject(
        self,
    ) -> None:
        no_match = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=1.0,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    110.0,
                    height=1.0,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            tolerance=2.0,
            thresholds=(0.0,),
        )
        missing_reference_height = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=None,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    100.0,
                    height=1.0,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            thresholds=(0.0,),
        )
        missing_candidate_fwhm = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=1.0,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    100.0,
                    height=1.0,
                    fwhm_cm1=None,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            thresholds=(0.0,),
        )
        missing_candidate_area = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=1.0,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    100.0,
                    height=1.0,
                    fwhm_cm1=1.0,
                    area=None,
                    prominence=1.0,
                ),
            ),
            thresholds=(0.0,),
        )
        zero_reference_height = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=0.0,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    100.0,
                    height=1.0,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            thresholds=(0.0,),
        )

        with self.assertRaisesRegex(
            PeakMetricError,
            "^matches: must contain at least one matched pair$",
        ):
            self.metric.evaluate(no_match)
        with self.assertRaisesRegex(
            PeakMetricError,
            "^reference_peaks\\[0\\]\\.height: must be present for matched error evaluation$",
        ):
            self.metric.evaluate(missing_reference_height)
        with self.assertRaisesRegex(
            PeakMetricError,
            "^candidate_peaks\\[0\\]\\.fwhm_cm1: must be present for matched error evaluation$",
        ):
            self.metric.evaluate(missing_candidate_fwhm)
        with self.assertRaisesRegex(
            PeakMetricError,
            "^candidate_peaks\\[0\\]\\.area: must be present for matched error evaluation$",
        ):
            self.metric.evaluate(missing_candidate_area)
        with self.assertRaisesRegex(
            PeakMetricError,
            "^reference_peaks\\[0\\]\\.height: must be nonzero for relative height error$",
        ):
            self.metric.evaluate(zero_reference_height)

    def test_derived_nonfinite_error_and_wrong_request_are_rejected(self) -> None:
        overflowing = request(
            reference_peaks=(
                peak(
                    100.0,
                    height=1.0e308,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            candidate_peaks=(
                peak(
                    100.0,
                    height=-1.0e308,
                    fwhm_cm1=1.0,
                    area=1.0,
                    prominence=1.0,
                ),
            ),
            thresholds=(0.0,),
        )
        wrong_request = SingleSpectrumInput(spectrum())

        with self.assertRaisesRegex(
            PeakMetricError,
            "^relative_peak_height_mae: produced non-finite value$",
        ):
            self.metric.evaluate(overflowing)
        with self.assertRaisesRegex(
            PeakMetricError,
            "^request: must be PeakPairInput$",
        ):
            self.metric.evaluate(wrong_request)

    def test_frozen_identity_and_repeat_determinism(self) -> None:
        with self.assertRaises(TypeError):
            MatchedPeakErrorMetric(metric_id="override")
        with self.assertRaises(FrozenInstanceError):
            self.metric.metric_id = "override"

        first = evaluate_metric(self.metric, self.request)
        second = evaluate_metric(self.metric, self.request)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
