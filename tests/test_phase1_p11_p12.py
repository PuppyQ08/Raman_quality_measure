from __future__ import annotations

import hashlib
import math
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D, SpectrumPairInput, evaluate_metric  # noqa: E402
from rpe.metrics import MSEMetric  # noqa: E402
from rpe.perturb import (  # noqa: E402
    P11AxisTransformError,
    P11GlobalWavenumberShift,
    P11AxisTransformState,
    P12AxisTransformError,
    P12QuadraticWavenumberWarp,
    PerturbationContext,
    PerturbationContractError,
    PerturbationSweepConfigError,
    derive_perturbed_spectrum_id,
    load_perturbation_sweep_config,
    validate_perturbation_result,
)


SWEEP_CONFIG = (
    ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"
)
SWEEP_BYTES = 559
SWEEP_SHA256 = (
    "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"
)


def source_spectrum(
    *,
    spectrum_id: str = "fixture-spectrum",
    axis: tuple[float, ...] = (100.0, 200.0, 400.0),
    intensity: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.array(axis, dtype="<f8"),
        intensity=np.array(intensity, dtype="<f8"),
    )


def perturbation_context() -> PerturbationContext:
    return PerturbationContext(
        sweep_id="raman_perturbation_alpha_v1",
        sweep_config_sha256=SWEEP_SHA256,
        global_seed=20260817,
    )


class SharedSweepConfigIdentityTest(unittest.TestCase):
    def test_shared_config_identity_matches_retained_phase1_phase4_file(self):
        raw = SWEEP_CONFIG.read_bytes()
        self.assertEqual(len(raw), SWEEP_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SWEEP_SHA256)
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(config.sha256, SWEEP_SHA256)
        self.assertIn("p11", config.perturbation_ids)
        self.assertIn("p12", config.perturbation_ids)


class P11GlobalWavenumberShiftTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = source_spectrum()
        self.metric = MSEMetric()
        self.perturbation = P11GlobalWavenumberShift(self.config)

    def test_state_dataclass_has_six_public_fields(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(P11AxisTransformState)),
            (
                "perturbation_id",
                "spectrum_id",
                "sweep_config_sha256",
                "state_digest",
                "source_axis_sha256",
                "source_intensity_sha256",
            ),
        )

    def test_prepare_is_deterministic_and_apply_zero_alpha_is_exact_identity(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        repeated_state = self.perturbation.prepare(self.source, self.context)

        self.assertEqual(state.perturbation_id, "p11")
        self.assertEqual(state.spectrum_id, self.source.spectrum_id)
        self.assertEqual(state.sweep_config_sha256, self.config.sha256)
        self.assertEqual(state, repeated_state)

        result = self.perturbation.apply(self.source, 0.0, state)

        self.assertFalse(result.axis_changed)
        self.assertFalse(result.intensity_changed)
        self.assertEqual(result.output.spectrum_id, derive_perturbed_spectrum_id(
            self.source.spectrum_id,
            "p11",
            0.0,
            state.state_digest,
            self.config.sha256,
        ))
        np.testing.assert_array_equal(result.output.axis_cm1, self.source.axis_cm1)
        np.testing.assert_array_equal(result.output.intensity, self.source.intensity)
        self.assertIsNot(result.output.axis_cm1, self.source.axis_cm1)
        self.assertIsNot(result.output.intensity, self.source.intensity)
        self.assertEqual(result.diagnostics["transform"], "global_additive_shift")
        self.assertEqual(result.diagnostics["requested_max_abs_offset_cm1"], 0.0)
        self.assertEqual(result.diagnostics["realized_max_abs_offset_cm1"], 0.0)
        self.assertTrue(result.diagnostics["intensity_preserved"])
        self.assertEqual(result.diagnostics["interpolation"], "none")

    def test_apply_positive_alpha_matches_literal_oracle_and_metric_blindness(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)

        result = self.perturbation.apply(self.source, 0.5, state)

        np.testing.assert_array_equal(
            result.output.axis_cm1,
            np.array([102.0, 202.0, 402.0], dtype="<f8"),
        )
        np.testing.assert_array_equal(result.output.intensity, self.source.intensity)
        self.assertTrue(result.axis_changed)
        self.assertFalse(result.intensity_changed)
        self.assertEqual(result.diagnostics["requested_max_abs_offset_cm1"], 2.0)
        self.assertEqual(result.diagnostics["realized_max_abs_offset_cm1"], 2.0)
        pair = SpectrumPairInput(reference=self.source, candidate=result.output)
        metric_result = evaluate_metric(self.metric, pair)
        self.assertEqual(metric_result.outputs[0].value, 0.0)

    def test_realized_max_offset_is_exactly_monotone_across_frozen_positive_alphas(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)

        maxima = []
        for alpha in self.config.alpha_grid:
            if alpha == 0.0:
                continue
            result = self.perturbation.apply(self.source, alpha, state)
            maxima.append(float(result.diagnostics["realized_max_abs_offset_cm1"]))
            self.assertEqual(result.diagnostics["requested_max_abs_offset_cm1"], 4.0 * alpha)
            np.testing.assert_allclose(
                np.diff(result.output.axis_cm1),
                np.diff(self.source.axis_cm1),
                rtol=1e-15,
                atol=0.0,
            )
        self.assertEqual(maxima, sorted(maxima))

    def test_apply_accepts_content_identical_reconstruction_but_rejects_content_drift(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        rebuilt = source_spectrum()
        result = self.perturbation.apply(rebuilt, 0.05, state)
        validate_perturbation_result(rebuilt, state, result, self.config)

        drifted = source_spectrum(intensity=(1.0, 2.0, 3.5))
        with self.assertRaisesRegex(P11AxisTransformError, "source intensity"):
            self.perturbation.apply(drifted, 0.05, state)

    def test_apply_rejects_wrong_context_state_alpha_and_unrepresentable_axis_change(self) -> None:
        with self.assertRaisesRegex(P11AxisTransformError, "context.sweep_id"):
            self.perturbation.prepare(
                self.source,
                replace(self.context, sweep_id="wrong"),
            )

        state = self.perturbation.prepare(self.source, self.context)
        with self.assertRaisesRegex(P11AxisTransformError, "alpha"):
            self.perturbation.apply(self.source, 0.123, state)
        with self.assertRaisesRegex(P11AxisTransformError, "state.sweep_config_sha256"):
            self.perturbation.apply(
                self.source,
                0.05,
                replace(state, sweep_config_sha256="0" * 64),
            )

        tiny = source_spectrum(axis=(2.0**53, 2.0**53 + 2.0, 2.0**53 + 4.0))
        tiny_state = self.perturbation.prepare(tiny, self.context)
        with self.assertRaisesRegex(P11AxisTransformError, "realized axis transform"):
            self.perturbation.apply(tiny, 0.05, tiny_state)


class P12QuadraticWavenumberWarpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = source_spectrum()
        self.metric = MSEMetric()
        self.perturbation = P12QuadraticWavenumberWarp(self.config)

    def test_apply_positive_alpha_matches_literal_oracle_and_monotone_offsets(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)

        result = self.perturbation.apply(self.source, 0.5, state)

        expected_axis = np.array([100.0, 200.0 + (2.0 / 9.0), 402.0], dtype="<f8")
        np.testing.assert_allclose(result.output.axis_cm1, expected_axis, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(result.output.intensity, self.source.intensity)
        offsets = result.output.axis_cm1 - self.source.axis_cm1
        self.assertTrue(np.all(np.diff(offsets) >= 0.0))
        self.assertTrue(np.all(np.diff(result.output.axis_cm1) > 0.0))
        self.assertEqual(result.diagnostics["transform"], "lower_anchored_quadratic_warp")
        self.assertEqual(result.diagnostics["lower_anchor_cm1"], 100.0)
        self.assertEqual(result.diagnostics["upper_anchor_cm1"], 400.0)
        self.assertEqual(result.diagnostics["normalized_coordinate"], "endpoint_minmax")
        self.assertEqual(result.diagnostics["requested_max_abs_offset_cm1"], 2.0)
        self.assertEqual(result.diagnostics["realized_max_abs_offset_cm1"], 2.0)
        pair = SpectrumPairInput(reference=self.source, candidate=result.output)
        metric_result = evaluate_metric(self.metric, pair)
        self.assertEqual(metric_result.outputs[0].value, 0.0)

    def test_zero_alpha_is_exact_identity_and_positive_alphas_remain_exactly_monotone(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        zero = self.perturbation.apply(self.source, 0.0, state)
        self.assertFalse(zero.axis_changed)
        self.assertFalse(zero.intensity_changed)
        np.testing.assert_array_equal(zero.output.axis_cm1, self.source.axis_cm1)
        np.testing.assert_array_equal(zero.output.intensity, self.source.intensity)

        maxima = []
        for alpha in self.config.alpha_grid:
            if alpha == 0.0:
                continue
            result = self.perturbation.apply(self.source, alpha, state)
            realized = float(result.diagnostics["realized_max_abs_offset_cm1"])
            maxima.append(realized)
            self.assertEqual(realized, 4.0 * alpha)
            self.assertTrue(result.axis_changed)
            self.assertFalse(result.intensity_changed)
        self.assertEqual(maxima, sorted(maxima))

    def test_apply_accepts_content_identical_reconstruction_and_rejects_wrong_source_or_state(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        rebuilt = source_spectrum()
        result = self.perturbation.apply(rebuilt, 0.1, state)
        validate_perturbation_result(rebuilt, state, result, self.config)

        drifted = source_spectrum(axis=(100.0, 210.0, 400.0))
        with self.assertRaisesRegex(P12AxisTransformError, "source axis"):
            self.perturbation.apply(drifted, 0.1, state)
        with self.assertRaisesRegex(P12AxisTransformError, "state.state_digest"):
            self.perturbation.apply(
                rebuilt,
                0.1,
                replace(state, state_digest="0" * 64),
            )

    def test_constructor_rejects_semantically_mutated_config_and_apply_rejects_unrepresentable_warp(self) -> None:
        bad_config = replace(self.config, global_seed=self.config.global_seed + 1)
        with self.assertRaisesRegex(P12AxisTransformError, "config.global_seed"):
            P12QuadraticWavenumberWarp(bad_config)

        tiny = source_spectrum(axis=(2.0**53, 2.0**53 + 2.0, 2.0**53 + 4.0))
        state = self.perturbation.prepare(tiny, self.context)
        with self.assertRaisesRegex(
            P12AxisTransformError,
            "maximum shift|realized axis transform",
        ):
            self.perturbation.apply(tiny, 0.05, state)


class PublicExportsTest(unittest.TestCase):
    def test_public_imports_exist_for_axis_transform_operators(self) -> None:
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(P11GlobalWavenumberShift(config).perturbation_id, "p11")
        self.assertEqual(P12QuadraticWavenumberWarp(config).perturbation_id, "p12")


if __name__ == "__main__":
    unittest.main()
