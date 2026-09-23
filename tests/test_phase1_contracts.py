from __future__ import annotations

import math
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import (  # noqa: E402
    AxisPolicy,
    CalibrationInput,
    CurveMetricOutput,
    CurveSeries,
    EvaluationContractError,
    Metric,
    MetricInputKind,
    MetricResult,
    Peak1D,
    PeakPairInput,
    PreferredDirection,
    ReplicatePairInput,
    ScalarMetricOutput,
    SingleSpectrumInput,
    SpectralRegion,
    Spectrum1D,
    SpectrumPairInput,
    SpectrumRegionsInput,
    evaluate_metric,
    validate_metric_request,
    validate_metric_result,
)
from rpe.metrics import MSEMetric  # noqa: E402
from rpe.perturb import (  # noqa: E402
    AxisBehavior,
    Perturbation,
    PerturbationContext,
    PerturbationContractError,
    PerturbationResult,
    derive_perturbed_spectrum_id,
    load_perturbation_sweep_config,
    validate_perturbation_result,
)


SWEEP_CONFIG = (
    ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"
)


def spectrum(
    spectrum_id: str = "reference",
    *,
    axis: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0),
    intensity: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.asarray(axis, dtype="<f8"),
        intensity=np.asarray(intensity, dtype="<f8"),
    )


def all_requests() -> tuple[object, ...]:
    reference = spectrum()
    candidate = spectrum("candidate")
    return (
        SpectrumPairInput(reference, candidate),
        SpectrumRegionsInput(
            reference,
            (
                SpectralRegion("signal", 100.0, 250.0, "signal"),
                SpectralRegion("noise", 250.0, 400.0, "noise"),
            ),
        ),
        PeakPairInput(
            (
                Peak1D(150.0, 1.0, 8.0, 6.0, 0.8),
                Peak1D(300.0, -0.5, 10.0, -4.0, 0.3),
            ),
            (Peak1D(151.0, 0.9, 8.5, 5.8, 0.7),),
            2.0,
            (0.1, 0.2, 0.4),
        ),
        SingleSpectrumInput(reference),
        CalibrationInput(
            ("analyte",),
            np.array([[0.0], [1.0]], dtype="<f8"),
            np.array([[0.1], [0.9]], dtype="<f8"),
            np.array([[0.01], [0.02]], dtype="<f8"),
        ),
        ReplicatePairInput(reference, candidate),
    )


class SpectrumContractTest(unittest.TestCase):
    def test_spectrum_copies_arrays_as_read_only_float64(self):
        axis = np.array([100.0, 200.0, 300.0], dtype="<f8")
        intensity = np.array([1.0, 2.0, 3.0], dtype="<f8")

        value = Spectrum1D("spectrum-a", "sample-a", axis, intensity)
        axis[:] = -1.0
        intensity[:] = -1.0

        np.testing.assert_array_equal(
            value.axis_cm1,
            np.array([100.0, 200.0, 300.0], dtype="<f8"),
        )
        np.testing.assert_array_equal(
            value.intensity,
            np.array([1.0, 2.0, 3.0], dtype="<f8"),
        )
        self.assertFalse(value.axis_cm1.flags.writeable)
        self.assertFalse(value.intensity.flags.writeable)
        self.assertFalse(np.shares_memory(value.axis_cm1, axis))
        self.assertFalse(np.shares_memory(value.intensity, intensity))

    def test_spectrum_accepts_none_sample_id(self):
        value = Spectrum1D(
            spectrum_id="spectrum-a",
            sample_id=None,
            axis_cm1=np.array([100.0, 200.0], dtype="<f8"),
            intensity=np.array([1.0, 2.0], dtype="<f8"),
        )
        self.assertIsNone(value.sample_id)

    def test_spectrum_rejects_invalid_identity_dtype_shape_values_or_axis(self):
        base_axis = np.array([100.0, 200.0, 300.0], dtype="<f8")
        base_intensity = np.array([1.0, 2.0, 3.0], dtype="<f8")
        cases = (
            ("spectrum_id", "", "sample", base_axis, base_intensity),
            ("sample_id", "id", "", base_axis, base_intensity),
            (
                "axis dtype",
                "id",
                "sample",
                base_axis.astype("<f4"),
                base_intensity,
            ),
            (
                "intensity dtype",
                "id",
                "sample",
                base_axis,
                base_intensity.astype("<f4"),
            ),
            (
                "axis dimension",
                "id",
                "sample",
                base_axis.reshape(1, -1),
                base_intensity,
            ),
            (
                "array length",
                "id",
                "sample",
                base_axis,
                base_intensity[:2],
            ),
            (
                "axis finite",
                "id",
                "sample",
                np.array([100.0, np.nan, 300.0], dtype="<f8"),
                base_intensity,
            ),
            (
                "intensity finite",
                "id",
                "sample",
                base_axis,
                np.array([1.0, np.inf, 3.0], dtype="<f8"),
            ),
            (
                "axis increasing",
                "id",
                "sample",
                np.array([100.0, 300.0, 200.0], dtype="<f8"),
                base_intensity,
            ),
            (
                "axis increasing",
                "id",
                "sample",
                np.array([100.0, 200.0, 200.0], dtype="<f8"),
                base_intensity,
            ),
            (
                "axis shape",
                "id",
                "sample",
                np.array([], dtype="<f8"),
                np.array([], dtype="<f8"),
            ),
        )
        for expected, spectrum_id, sample_id, axis, intensity in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    EvaluationContractError,
                    expected,
                ):
                    Spectrum1D(
                        spectrum_id,
                        sample_id,
                        axis,
                        intensity,
                    )


class MetricRequestContractTest(unittest.TestCase):
    def test_all_six_request_kinds_are_constructible_and_immutable(self):
        requests = all_requests()

        self.assertEqual(
            [request.input_kind for request in requests],
            list(MetricInputKind),
        )
        for request in requests:
            validate_metric_request(request)
        calibration = requests[4]
        self.assertIsInstance(calibration, CalibrationInput)
        self.assertFalse(calibration.true_concentrations.flags.writeable)
        self.assertFalse(calibration.predicted_concentrations.flags.writeable)
        self.assertFalse(calibration.blank_predictions.flags.writeable)

    def test_region_request_rejects_invalid_or_duplicate_regions(self):
        base = spectrum()
        cases = (
            (
                "region bounds",
                lambda: SpectrumRegionsInput(
                    base,
                    (SpectralRegion("bad", 200.0, 200.0, "signal"),),
                ),
            ),
            (
                "region domain",
                lambda: SpectrumRegionsInput(
                    base,
                    (SpectralRegion("bad", 50.0, 200.0, "signal"),),
                ),
            ),
            (
                "region_ids",
                lambda: SpectrumRegionsInput(
                    base,
                    (
                        SpectralRegion("same", 100.0, 200.0, "signal"),
                        SpectralRegion("same", 200.0, 300.0, "noise"),
                    ),
                ),
            ),
            (
                "regions",
                lambda: SpectrumRegionsInput(base, ()),
            ),
        )
        for expected, build in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    EvaluationContractError,
                    expected,
                ):
                    build()

    def test_peak_request_rejects_invalid_peaks_or_grids(self):
        valid_reference = (
            Peak1D(150.0, 1.0, 8.0, 6.0, 0.8),
            Peak1D(300.0, 0.5, 10.0, 4.0, 0.3),
        )
        valid_candidate = (Peak1D(151.0, 0.9, 8.5, 5.8, 0.7),)
        cases = (
            (
                "position",
                lambda: PeakPairInput(
                    (
                        Peak1D(300.0, 1.0, 8.0, 6.0, 0.8),
                        Peak1D(150.0, 0.5, 10.0, 4.0, 0.3),
                    ),
                    valid_candidate,
                    2.0,
                    (0.1, 0.2),
                ),
            ),
            (
                "fwhm",
                lambda: PeakPairInput(
                    (Peak1D(150.0, 1.0, 0.0, 6.0, 0.8),),
                    valid_candidate,
                    2.0,
                    (0.1, 0.2),
                ),
            ),
            (
                "prominence",
                lambda: PeakPairInput(
                    (Peak1D(150.0, 1.0, 8.0, 6.0, 0.0),),
                    valid_candidate,
                    2.0,
                    (0.1, 0.2),
                ),
            ),
            (
                "position_tolerance",
                lambda: PeakPairInput(
                    valid_reference,
                    valid_candidate,
                    0.0,
                    (0.1, 0.2),
                ),
            ),
            (
                "prominence_thresholds",
                lambda: PeakPairInput(
                    valid_reference,
                    valid_candidate,
                    2.0,
                    (),
                ),
            ),
            (
                "prominence_thresholds",
                lambda: PeakPairInput(
                    valid_reference,
                    valid_candidate,
                    2.0,
                    (0.2, 0.1),
                ),
            ),
            (
                "prominence_thresholds",
                lambda: PeakPairInput(
                    valid_reference,
                    valid_candidate,
                    2.0,
                    (-0.1, 0.2),
                ),
            ),
        )
        for expected, build in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    EvaluationContractError,
                    expected,
                ):
                    build()

    def test_calibration_rejects_bad_names_shapes_blanks_or_values(self):
        true = np.array([[0.0], [1.0]], dtype="<f8")
        predicted = np.array([[0.1], [0.9]], dtype="<f8")
        blank = np.array([[0.01], [0.02]], dtype="<f8")
        cases = (
            (
                "target_names",
                ("same", "same"),
                np.ones((2, 2), dtype="<f8"),
                np.ones((2, 2), dtype="<f8"),
                np.ones((2, 2), dtype="<f8"),
            ),
            (
                "predicted_concentrations shape",
                ("analyte",),
                true,
                predicted[:1],
                blank,
            ),
            (
                "blank_predictions shape",
                ("analyte",),
                true,
                predicted,
                blank[:1],
            ),
            (
                "true_concentrations finite",
                ("analyte",),
                np.array([[0.0], [np.nan]], dtype="<f8"),
                predicted,
                blank,
            ),
            (
                "true_concentrations dtype",
                ("analyte",),
                true.astype("<f4"),
                predicted,
                blank,
            ),
        )
        for expected, names, current_true, current_predicted, current_blank in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    EvaluationContractError,
                    expected,
                ):
                    CalibrationInput(
                        names,
                        current_true,
                        current_predicted,
                        current_blank,
                    )

    def test_validate_request_rejects_non_request(self):
        with self.assertRaisesRegex(
            EvaluationContractError,
            "unsupported request",
        ):
            validate_metric_request(object())


@dataclass(frozen=True)
class StubSpectrumPairMetric:
    metric_id: str = "stub_spectrum_pair"
    input_kind: MetricInputKind = MetricInputKind.SPECTRUM_PAIR
    axis_policy: AxisPolicy = AxisPolicy.INDEX_ALIGNED

    def evaluate(self, request):
        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    "pair_score",
                    0.25,
                    "ratio",
                    PreferredDirection.LOWER_IS_BETTER,
                    None,
                ),
            ),
            diagnostics={
                "axis_equal": bool(
                    np.array_equal(
                        request.reference.axis_cm1,
                        request.candidate.axis_cm1,
                    )
                )
            },
        )


@dataclass(frozen=True)
class StubSpectrumRegionsMetric:
    metric_id: str = "stub_regions"
    input_kind: MetricInputKind = MetricInputKind.SPECTRUM_REGIONS
    axis_policy: AxisPolicy = AxisPolicy.SINGLE_AXIS

    def evaluate(self, request):
        return MetricResult(
            self.metric_id,
            self.input_kind,
            (
                ScalarMetricOutput(
                    "region_score",
                    float(len(request.regions)),
                    "count",
                    PreferredDirection.NON_MONOTONIC,
                    None,
                ),
            ),
            {"region_ids": [region.region_id for region in request.regions]},
        )


@dataclass(frozen=True)
class StubPeakCurveMetric:
    metric_id: str = "stub_peak_curve"
    input_kind: MetricInputKind = MetricInputKind.PEAK_PAIR
    axis_policy: AxisPolicy = AxisPolicy.NOT_APPLICABLE

    def evaluate(self, request):
        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                CurveMetricOutput(
                    output_id="prominence_scan",
                    x_name="prominence_threshold",
                    x_values=request.prominence_thresholds,
                    x_unit="intensity",
                    series=(
                        CurveSeries(
                            "precision",
                            (1.0, 0.9, 0.8),
                            "ratio",
                            PreferredDirection.HIGHER_IS_BETTER,
                            None,
                        ),
                        CurveSeries(
                            "recall",
                            (0.7, 0.6, 0.5),
                            "ratio",
                            PreferredDirection.HIGHER_IS_BETTER,
                            None,
                        ),
                        CurveSeries(
                            "f1",
                            (0.82, 0.72, 0.62),
                            "ratio",
                            PreferredDirection.HIGHER_IS_BETTER,
                            None,
                        ),
                        CurveSeries(
                            "artifact_peak_ratio",
                            (0.0, 0.1, 0.2),
                            "ratio",
                            PreferredDirection.LOWER_IS_BETTER,
                            None,
                        ),
                        CurveSeries(
                            "missing_peak_ratio",
                            (0.3, 0.4, 0.5),
                            "ratio",
                            PreferredDirection.LOWER_IS_BETTER,
                            None,
                        ),
                    ),
                ),
            ),
            diagnostics={"matched_at_default_tolerance": 1},
        )


@dataclass(frozen=True)
class StubSingleSpectrumMetric:
    metric_id: str = "stub_no_reference"
    input_kind: MetricInputKind = MetricInputKind.SINGLE_SPECTRUM
    axis_policy: AxisPolicy = AxisPolicy.SINGLE_AXIS

    def evaluate(self, request):
        return MetricResult(
            self.metric_id,
            self.input_kind,
            (
                ScalarMetricOutput(
                    "quality",
                    0.75,
                    "ratio",
                    PreferredDirection.HIGHER_IS_BETTER,
                    None,
                ),
            ),
            {"point_count": int(request.spectrum.intensity.size)},
        )


@dataclass(frozen=True)
class StubCalibrationMetric:
    metric_id: str = "stub_calibration"
    input_kind: MetricInputKind = MetricInputKind.CALIBRATION
    axis_policy: AxisPolicy = AxisPolicy.NOT_APPLICABLE

    def evaluate(self, request):
        outputs = tuple(
            ScalarMetricOutput(
                output_id,
                value,
                "mol_l",
                PreferredDirection.LOWER_IS_BETTER,
                None,
            )
            for output_id, value in (
                ("ich_lod", 0.033),
                ("ich_loq", 0.1),
                ("iupac_lod", 0.03),
            )
        )
        return MetricResult(
            self.metric_id,
            self.input_kind,
            outputs,
            {"target_count": len(request.target_names)},
        )


@dataclass(frozen=True)
class StubReplicateConsistencyMetric:
    metric_id: str = "stub_replicate"
    input_kind: MetricInputKind = MetricInputKind.REPLICATE_PAIR
    axis_policy: AxisPolicy = AxisPolicy.EXACT_AXIS

    def evaluate(self, request):
        return MetricResult(
            self.metric_id,
            self.input_kind,
            (
                ScalarMetricOutput(
                    "consistency",
                    1.0,
                    "ratio",
                    PreferredDirection.HIGHER_IS_BETTER,
                    None,
                ),
            ),
            {"axis_equal": True},
        )


class MetricStubExecutionTest(unittest.TestCase):
    def test_all_six_metric_families_execute_through_wrapper(self):
        requests = all_requests()
        metrics = (
            StubSpectrumPairMetric(),
            StubSpectrumRegionsMetric(),
            StubPeakCurveMetric(),
            StubSingleSpectrumMetric(),
            StubCalibrationMetric(),
            StubReplicateConsistencyMetric(),
        )

        for metric, request in zip(metrics, requests, strict=True):
            with self.subTest(metric=metric.metric_id):
                self.assertIsInstance(metric, Metric)
                result = evaluate_metric(metric, request)
                validate_metric_result(result)
                self.assertEqual(result.metric_id, metric.metric_id)
                self.assertIs(result.input_kind, request.input_kind)

    def test_peak_stub_returns_multiseries_threshold_curve(self):
        result = evaluate_metric(StubPeakCurveMetric(), all_requests()[2])
        self.assertEqual(len(result.outputs), 1)
        output = result.outputs[0]
        self.assertIsInstance(output, CurveMetricOutput)
        self.assertEqual(output.x_values, (0.1, 0.2, 0.4))
        self.assertEqual(
            [series.series_id for series in output.series],
            [
                "precision",
                "recall",
                "f1",
                "artifact_peak_ratio",
                "missing_peak_ratio",
            ],
        )

    def test_calibration_stub_returns_three_scalar_outputs(self):
        result = evaluate_metric(StubCalibrationMetric(), all_requests()[4])
        self.assertEqual(
            [output.output_id for output in result.outputs],
            ["ich_lod", "ich_loq", "iupac_lod"],
        )
        self.assertTrue(
            all(
                isinstance(output, ScalarMetricOutput)
                for output in result.outputs
            )
        )

    def test_diagnostics_are_recursively_immutable(self):
        source = {
            "nested": {"values": [1, 2]},
        }
        result = MetricResult(
            "metric",
            MetricInputKind.SINGLE_SPECTRUM,
            (
                ScalarMetricOutput(
                    "value",
                    1.0,
                    "ratio",
                    PreferredDirection.HIGHER_IS_BETTER,
                    None,
                ),
            ),
            source,
        )
        source["nested"]["values"][0] = 99
        self.assertEqual(result.diagnostics["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            result.diagnostics["new"] = 1
        with self.assertRaises(TypeError):
            result.diagnostics["nested"]["new"] = 1


class MetricResultMutationTest(unittest.TestCase):
    def scalar(self, **overrides):
        values = {
            "output_id": "value",
            "value": 1.0,
            "unit": "ratio",
            "preferred_direction": PreferredDirection.HIGHER_IS_BETTER,
            "target_value": None,
        }
        values.update(overrides)
        return ScalarMetricOutput(**values)

    def curve(self, **overrides):
        values = {
            "output_id": "curve",
            "x_name": "threshold",
            "x_values": (0.1, 0.2),
            "x_unit": "ratio",
            "series": (
                CurveSeries(
                    "score",
                    (0.5, 0.6),
                    "ratio",
                    PreferredDirection.HIGHER_IS_BETTER,
                    None,
                ),
            ),
        }
        values.update(overrides)
        return CurveMetricOutput(**values)

    def test_scalar_and_curve_contracts_reject_invalid_values(self):
        scalar_cases = (
            ("output_id", {"output_id": ""}),
            ("value finite", {"value": np.nan}),
            (
                "target_value",
                {
                    "preferred_direction": PreferredDirection.TARGET_VALUE,
                    "target_value": None,
                },
            ),
            (
                "target_value",
                {
                    "preferred_direction": (
                        PreferredDirection.HIGHER_IS_BETTER
                    ),
                    "target_value": 1.0,
                },
            ),
        )
        for expected, overrides in scalar_cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    EvaluationContractError,
                    expected,
                ):
                    self.scalar(**overrides)

        curve_cases = (
            ("x_values", {"x_values": ()}),
            ("x_values", {"x_values": (0.2, 0.1)}),
            ("x_values", {"x_values": (0.1, np.inf)}),
            (
                "series values",
                {
                    "series": (
                        CurveSeries(
                            "score",
                            (0.5,),
                            "ratio",
                            PreferredDirection.HIGHER_IS_BETTER,
                            None,
                        ),
                    ),
                },
            ),
            (
                "series_ids",
                {
                    "series": (
                        CurveSeries(
                            "same",
                            (0.5, 0.6),
                            "ratio",
                            PreferredDirection.HIGHER_IS_BETTER,
                            None,
                        ),
                        CurveSeries(
                            "same",
                            (0.7, 0.8),
                            "ratio",
                            PreferredDirection.HIGHER_IS_BETTER,
                            None,
                        ),
                    ),
                },
            ),
        )
        for expected, overrides in curve_cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    EvaluationContractError,
                    expected,
                ):
                    self.curve(**overrides)

    def test_metric_result_rejects_empty_or_duplicate_outputs(self):
        with self.assertRaisesRegex(
            EvaluationContractError,
            "outputs",
        ):
            MetricResult(
                "metric",
                MetricInputKind.SINGLE_SPECTRUM,
                (),
                {},
            )
        duplicate = self.scalar()
        with self.assertRaisesRegex(
            EvaluationContractError,
            "output_ids",
        ):
            MetricResult(
                "metric",
                MetricInputKind.SINGLE_SPECTRUM,
                (duplicate, duplicate),
                {},
            )

    def test_metric_result_rejects_non_json_diagnostics(self):
        with self.assertRaisesRegex(
            EvaluationContractError,
            "canonical-JSON-compatible",
        ):
            MetricResult(
                "metric",
                MetricInputKind.SINGLE_SPECTRUM,
                (self.scalar(),),
                {"bad": {1, 2}},
            )
        with self.assertRaisesRegex(
            EvaluationContractError,
            "finite",
        ):
            MetricResult(
                "metric",
                MetricInputKind.SINGLE_SPECTRUM,
                (self.scalar(),),
                {"bad": math.inf},
            )

    def test_metric_result_rejects_diagnostics_that_duplicate_outputs(self):
        with self.assertRaisesRegex(
            EvaluationContractError,
            "duplicate primary output",
        ):
            MetricResult(
                "metric",
                MetricInputKind.SINGLE_SPECTRUM,
                (self.scalar(output_id="score"),),
                {"nested": {"score": 1.0}},
            )

        with self.assertRaisesRegex(
            EvaluationContractError,
            "duplicate primary output",
        ):
            MetricResult(
                "metric",
                MetricInputKind.PEAK_PAIR,
                (
                    self.curve(
                        output_id="scan",
                        series=(
                            CurveSeries(
                                "artifact_peak_ratio",
                                (0.1, 0.2),
                                "ratio",
                                PreferredDirection.LOWER_IS_BETTER,
                                None,
                            ),
                        ),
                    ),
                ),
                {"artifact_peak_ratio": [0.1, 0.2]},
            )

    def test_wrapper_rejects_wrong_request_or_result_identity(self):
        request = all_requests()[0]
        with self.assertRaisesRegex(
            EvaluationContractError,
            "input_kind",
        ):
            evaluate_metric(StubSingleSpectrumMetric(), request)

        @dataclass(frozen=True)
        class WrongResultId:
            metric_id: str = "expected"
            input_kind: MetricInputKind = MetricInputKind.SPECTRUM_PAIR
            axis_policy: AxisPolicy = AxisPolicy.INDEX_ALIGNED

            def evaluate(self, current):
                return MetricResult(
                    "wrong",
                    current.input_kind,
                    (self_scalar(),),
                    {},
                )

        def self_scalar():
            return self.scalar()

        with self.assertRaisesRegex(
            EvaluationContractError,
            "metric_id",
        ):
            evaluate_metric(WrongResultId(), request)

        @dataclass(frozen=True)
        class NonResult:
            metric_id: str = "non_result"
            input_kind: MetricInputKind = MetricInputKind.SPECTRUM_PAIR
            axis_policy: AxisPolicy = AxisPolicy.INDEX_ALIGNED

            def evaluate(self, current):
                return 1.0

        with self.assertRaisesRegex(
            EvaluationContractError,
            "MetricResult",
        ):
            evaluate_metric(NonResult(), request)

    def test_axis_policies_are_enforced(self):
        exact_metric = replace(
            StubSpectrumPairMetric(),
            axis_policy=AxisPolicy.EXACT_AXIS,
        )
        shifted = SpectrumPairInput(
            spectrum(),
            spectrum(
                "shifted",
                axis=(101.0, 201.0, 301.0, 401.0),
            ),
        )
        with self.assertRaisesRegex(
            EvaluationContractError,
            "exact axis",
        ):
            evaluate_metric(exact_metric, shifted)

        independent_metric = replace(
            StubSpectrumPairMetric(),
            axis_policy=AxisPolicy.INDEPENDENT_PHYSICAL_AXES,
        )
        result = evaluate_metric(independent_metric, shifted)
        self.assertEqual(result.metric_id, independent_metric.metric_id)

        replicate = all_requests()[5]
        wrong_replicate = replace(
            StubReplicateConsistencyMetric(),
            axis_policy=AxisPolicy.INDEX_ALIGNED,
        )
        with self.assertRaisesRegex(
            EvaluationContractError,
            "replicate",
        ):
            evaluate_metric(wrong_replicate, replicate)

        wrong_single = replace(
            StubSingleSpectrumMetric(),
            axis_policy=AxisPolicy.NOT_APPLICABLE,
        )
        with self.assertRaisesRegex(
            EvaluationContractError,
            "axis_policy",
        ):
            evaluate_metric(wrong_single, all_requests()[3])


@dataclass(frozen=True)
class StubState:
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str


@dataclass(frozen=True)
class StubAxisPreservingPerturbation:
    perturbation_id: str = "p10"
    axis_behavior: AxisBehavior = AxisBehavior.PRESERVE

    def prepare(self, spectrum, context):
        return StubState(
            self.perturbation_id,
            spectrum.spectrum_id,
            context.sweep_config_sha256,
            "2" * 64,
        )

    def apply(self, spectrum, alpha, state):
        intensity = (
            spectrum.intensity.copy()
            if alpha == 0.0
            else spectrum.intensity + alpha
        )
        output = Spectrum1D(
            spectrum_id=derive_perturbed_spectrum_id(
                spectrum.spectrum_id,
                self.perturbation_id,
                alpha,
                state.state_digest,
                state.sweep_config_sha256,
            ),
            sample_id=spectrum.sample_id,
            axis_cm1=spectrum.axis_cm1.copy(),
            intensity=np.asarray(intensity, dtype="<f8"),
        )
        return PerturbationResult(
            source_spectrum_id=spectrum.spectrum_id,
            perturbation_id=self.perturbation_id,
            alpha=alpha,
            output=output,
            axis_behavior=self.axis_behavior,
            state_digest=state.state_digest,
            axis_changed=False,
            intensity_changed=alpha != 0.0,
            diagnostics={"delta": alpha},
        )


@dataclass(frozen=True)
class StubP11AxisShiftPerturbation:
    perturbation_id: str = "p11"
    axis_behavior: AxisBehavior = AxisBehavior.TRANSFORM

    def prepare(self, spectrum, context):
        return StubState(
            self.perturbation_id,
            spectrum.spectrum_id,
            context.sweep_config_sha256,
            "1" * 64,
        )

    def apply(self, spectrum, alpha, state):
        axis = (
            spectrum.axis_cm1.copy()
            if alpha == 0.0
            else spectrum.axis_cm1 + alpha * 4.0
        )
        output = Spectrum1D(
            spectrum_id=derive_perturbed_spectrum_id(
                spectrum.spectrum_id,
                self.perturbation_id,
                alpha,
                state.state_digest,
                state.sweep_config_sha256,
            ),
            sample_id=spectrum.sample_id,
            axis_cm1=np.asarray(axis, dtype="<f8"),
            intensity=spectrum.intensity.copy(),
        )
        return PerturbationResult(
            source_spectrum_id=spectrum.spectrum_id,
            perturbation_id=self.perturbation_id,
            alpha=alpha,
            output=output,
            axis_behavior=self.axis_behavior,
            state_digest=state.state_digest,
            axis_changed=alpha != 0.0,
            intensity_changed=False,
            diagnostics={"shift_cm1": alpha * 4.0},
        )


class PerturbationContractTest(unittest.TestCase):
    def sweep_config(self):
        return load_perturbation_sweep_config(SWEEP_CONFIG)

    def context(self):
        config = self.sweep_config()
        return config, PerturbationContext(
            sweep_id=config.sweep_id,
            sweep_config_sha256=config.sha256,
            global_seed=config.global_seed,
        )

    def test_stub_perturbations_conform_and_validate(self):
        source = spectrum()
        config, context = self.context()

        preserve = StubAxisPreservingPerturbation()
        self.assertIsInstance(preserve, Perturbation)
        preserve_state = preserve.prepare(source, context)
        preserve_zero = preserve.apply(source, 0.0, preserve_state)
        validate_perturbation_result(
            source,
            preserve_state,
            preserve_zero,
            config,
        )
        np.testing.assert_array_equal(
            preserve_zero.output.axis_cm1,
            source.axis_cm1,
        )
        np.testing.assert_array_equal(
            preserve_zero.output.intensity,
            source.intensity,
        )
        self.assertFalse(preserve_zero.axis_changed)
        self.assertFalse(preserve_zero.intensity_changed)

        preserve_positive = preserve.apply(source, 0.3, preserve_state)
        validate_perturbation_result(
            source,
            preserve_state,
            preserve_positive,
            config,
        )
        np.testing.assert_array_equal(
            preserve_positive.output.axis_cm1,
            source.axis_cm1,
        )
        np.testing.assert_array_equal(
            preserve_positive.output.intensity,
            source.intensity + 0.3,
        )
        self.assertFalse(preserve_positive.axis_changed)
        self.assertTrue(preserve_positive.intensity_changed)

        p11 = StubP11AxisShiftPerturbation()
        self.assertIsInstance(p11, Perturbation)
        state = p11.prepare(source, context)
        zero = p11.apply(source, 0.0, state)
        validate_perturbation_result(source, state, zero, config)
        np.testing.assert_array_equal(zero.output.axis_cm1, source.axis_cm1)
        np.testing.assert_array_equal(zero.output.intensity, source.intensity)
        self.assertFalse(zero.axis_changed)
        self.assertFalse(zero.intensity_changed)

        shifted = p11.apply(source, 0.2, state)
        validate_perturbation_result(source, state, shifted, config)
        np.testing.assert_array_equal(
            shifted.output.axis_cm1,
            source.axis_cm1 + 0.8,
        )
        np.testing.assert_array_equal(
            shifted.output.intensity,
            source.intensity,
        )
        self.assertTrue(shifted.axis_changed)
        self.assertFalse(shifted.intensity_changed)
        self.assertEqual(shifted.diagnostics["shift_cm1"], 0.8)

    def test_index_aligned_metric_can_consume_axis_shifted_candidate(self):
        source = spectrum()
        config, context = self.context()
        perturbation = StubP11AxisShiftPerturbation()
        state = perturbation.prepare(source, context)
        shifted = perturbation.apply(source, 0.2, state)
        validate_perturbation_result(source, state, shifted, config)

        result = evaluate_metric(
            StubSpectrumPairMetric(),
            SpectrumPairInput(source, shifted.output),
        )
        self.assertEqual(result.outputs[0].output_id, "pair_score")
        self.assertFalse(result.diagnostics["axis_equal"])

    def test_p11_positive_alpha_output_produces_zero_mse_with_axis_equal_false(self):
        source = spectrum()
        config, context = self.context()
        perturbation = StubP11AxisShiftPerturbation()
        state = perturbation.prepare(source, context)
        shifted = perturbation.apply(source, 0.2, state)
        validate_perturbation_result(source, state, shifted, config)

        result = evaluate_metric(
            MSEMetric(),
            SpectrumPairInput(source, shifted.output),
        )
        self.assertEqual(result.outputs[0].value, 0.0)
        self.assertFalse(result.diagnostics["axis_equal"])

    def test_perturbation_diagnostics_are_recursively_immutable(self):
        source = spectrum()
        diagnostics = {"nested": {"values": [1, 2]}}
        output = Spectrum1D(
            spectrum_id=derive_perturbed_spectrum_id(
                source.spectrum_id,
                "p10",
                0.0,
                "2" * 64,
                self.sweep_config().sha256,
            ),
            sample_id=source.sample_id,
            axis_cm1=source.axis_cm1.copy(),
            intensity=source.intensity.copy(),
        )
        result = PerturbationResult(
            source_spectrum_id=source.spectrum_id,
            perturbation_id="p10",
            alpha=0.0,
            output=output,
            axis_behavior=AxisBehavior.PRESERVE,
            state_digest="2" * 64,
            axis_changed=False,
            intensity_changed=False,
            diagnostics=diagnostics,
        )
        diagnostics["nested"]["values"][0] = 99
        self.assertEqual(result.diagnostics["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            result.diagnostics["new"] = 1

    def test_perturbation_mutations_fail_closed(self):
        source = spectrum()
        config, context = self.context()
        preserve = StubAxisPreservingPerturbation()
        preserve_state = preserve.prepare(source, context)
        preserve_positive = preserve.apply(source, 0.3, preserve_state)
        p11 = StubP11AxisShiftPerturbation()
        state = p11.prepare(source, context)
        shifted = p11.apply(source, 0.2, state)

        cases = (
            (
                "unsupported alpha",
                lambda: validate_perturbation_result(
                    source,
                    state,
                    replace(shifted, alpha=0.12),
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "wrong state identity",
                lambda: validate_perturbation_result(
                    source,
                    replace(state, spectrum_id="other-spectrum"),
                    shifted,
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "wrong source identity",
                lambda: validate_perturbation_result(
                    source,
                    state,
                    replace(shifted, source_spectrum_id="other-spectrum"),
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "wrong config identity",
                lambda: validate_perturbation_result(
                    source,
                    state,
                    shifted,
                    replace(config, sha256="f" * 64),
                ),
                PerturbationContractError,
            ),
            (
                "wrong output spectrum id",
                lambda: validate_perturbation_result(
                    source,
                    state,
                    replace(
                        shifted,
                        output=replace(
                            shifted.output,
                            spectrum_id="perturbed-" + "0" * 64,
                        ),
                    ),
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "false axis flag",
                lambda: validate_perturbation_result(
                    source,
                    state,
                    replace(shifted, axis_changed=False),
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "false intensity flag",
                lambda: validate_perturbation_result(
                    source,
                    preserve_state,
                    replace(preserve_positive, intensity_changed=False),
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "preserve with changed axis",
                lambda: validate_perturbation_result(
                    source,
                    preserve_state,
                    replace(
                        preserve_positive,
                        output=Spectrum1D(
                            spectrum_id=derive_perturbed_spectrum_id(
                                source.spectrum_id,
                                "p10",
                                0.3,
                                preserve_state.state_digest,
                                preserve_state.sweep_config_sha256,
                            ),
                            sample_id=source.sample_id,
                            axis_cm1=np.asarray(
                                source.axis_cm1 + 0.3,
                                dtype="<f8",
                            ),
                            intensity=np.asarray(
                                source.intensity + 0.3,
                                dtype="<f8",
                            ),
                        ),
                        axis_changed=True,
                    ),
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "nonzero alpha-zero change",
                lambda: validate_perturbation_result(
                    source,
                    preserve_state,
                    replace(
                        preserve.apply(source, 0.0, preserve_state),
                        output=Spectrum1D(
                            spectrum_id=derive_perturbed_spectrum_id(
                                source.spectrum_id,
                                "p10",
                                0.0,
                                preserve_state.state_digest,
                                preserve_state.sweep_config_sha256,
                            ),
                            sample_id=source.sample_id,
                            axis_cm1=source.axis_cm1.copy(),
                            intensity=np.asarray(
                                source.intensity + 1.0,
                                dtype="<f8",
                            ),
                        ),
                        intensity_changed=True,
                    ),
                    config,
                ),
                PerturbationContractError,
            ),
            (
                "nonfinite diagnostics",
                lambda: PerturbationResult(
                    source_spectrum_id=source.spectrum_id,
                    perturbation_id="p10",
                    alpha=0.0,
                    output=Spectrum1D(
                        spectrum_id=derive_perturbed_spectrum_id(
                            source.spectrum_id,
                            "p10",
                            0.0,
                            "2" * 64,
                            config.sha256,
                        ),
                        sample_id=source.sample_id,
                        axis_cm1=source.axis_cm1.copy(),
                        intensity=source.intensity.copy(),
                    ),
                    axis_behavior=AxisBehavior.PRESERVE,
                    state_digest="2" * 64,
                    axis_changed=False,
                    intensity_changed=False,
                    diagnostics={"bad": math.inf},
                ),
                PerturbationContractError,
            ),
            (
                "transform invalid axis",
                lambda: Spectrum1D(
                    spectrum_id="invalid-axis-spectrum",
                    sample_id=source.sample_id,
                    axis_cm1=np.asarray(
                        [100.0, 100.0, 300.0, 400.0],
                        dtype="<f8",
                    ),
                    intensity=source.intensity.copy(),
                ),
                EvaluationContractError,
            ),
        )
        for label, runner, error_type in cases:
            with self.subTest(label=label):
                with self.assertRaises(error_type):
                    runner()

        with self.assertRaises(PerturbationContractError):
            derive_perturbed_spectrum_id(
                source.spectrum_id,
                "p11",
                0.2,
                "g" * 64,
                config.sha256,
            )
        with self.assertRaises(PerturbationContractError):
            derive_perturbed_spectrum_id(
                source.spectrum_id,
                "p11",
                0.2,
                state.state_digest,
                "g" * 64,
            )

    def test_derive_perturbed_spectrum_id_preserves_signed_zero_as_pure_identity_input(self):
        config = self.sweep_config()
        positive_zero_id = derive_perturbed_spectrum_id(
            "source",
            "p10",
            0.0,
            "2" * 64,
            config.sha256,
        )
        negative_zero_id = derive_perturbed_spectrum_id(
            "source",
            "p10",
            -0.0,
            "2" * 64,
            config.sha256,
        )
        self.assertNotEqual(positive_zero_id, negative_zero_id)

    def test_authoritative_validator_rejects_forged_config_identity_and_signed_zero(self):
        source = spectrum()
        config, context = self.context()
        preserve = StubAxisPreservingPerturbation()
        state = preserve.prepare(source, context)
        zero = preserve.apply(source, 0.0, state)

        forged_configs = (
            ("sha256", replace(config, sha256="0" * 64)),
            ("byte_count", replace(config, byte_count=1)),
        )
        for field_name, forged_config in forged_configs:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    PerturbationContractError,
                    f"config.{field_name}",
                ):
                    validate_perturbation_result(
                        source,
                        state,
                        zero,
                        forged_config,
                    )

        with self.assertRaisesRegex(PerturbationContractError, "alpha"):
            validate_perturbation_result(
                source,
                state,
                replace(zero, alpha=-0.0),
                config,
            )

    def test_validate_perturbation_result_rejects_semantically_mutated_config(self):
        source = spectrum()
        config, context = self.context()
        preserve = StubAxisPreservingPerturbation()
        state = preserve.prepare(source, context)
        zero = preserve.apply(source, 0.0, state)

        with self.assertRaises(PerturbationContractError):
            validate_perturbation_result(
                source,
                state,
                zero,
                replace(config, global_seed=20260818),
            )


if __name__ == "__main__":
    unittest.main()
