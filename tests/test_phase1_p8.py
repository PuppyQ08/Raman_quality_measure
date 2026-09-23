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
from rpe.metrics import RMSEMetric  # noqa: E402
from rpe.perturb import (  # noqa: E402
    BaselineDistortionError,
    P8BaselineDistortionState,
    P8LowOrderBaselineDistortion,
    PerturbationContext,
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
    axis: tuple[float, ...] = (113.0, 127.0, 166.0, 245.0, 251.0, 401.0),
    intensity: tuple[float, ...] = (2.0, 3.0, 5.0, 7.0, 11.0, 13.0),
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.array(axis, dtype="<f8"),
        intensity=np.array(intensity, dtype="<f8"),
    )


def zero_spectrum() -> Spectrum1D:
    return Spectrum1D(
        spectrum_id="zero-spectrum",
        sample_id="sample-z",
        axis_cm1=np.array([101.0, 111.0, 151.0, 199.0], dtype="<f8"),
        intensity=np.zeros(4, dtype="<f8"),
    )


def perturbation_context() -> PerturbationContext:
    return PerturbationContext(
        sweep_id="raman_perturbation_alpha_v1",
        sweep_config_sha256=SWEEP_SHA256,
        global_seed=20260817,
    )


class SharedSweepConfigIdentityTest(unittest.TestCase):
    def test_shared_config_identity_matches_retained_phase1_phase4_file(self) -> None:
        raw = SWEEP_CONFIG.read_bytes()
        self.assertEqual(len(raw), SWEEP_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SWEEP_SHA256)
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(config.sha256, SWEEP_SHA256)
        self.assertIn("p08", config.perturbation_ids)


class P8LowOrderBaselineDistortionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = source_spectrum()
        self.metric = RMSEMetric()
        self.perturbation = P8LowOrderBaselineDistortion(self.config)

    def test_state_dataclass_has_nine_public_fields(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(P8BaselineDistortionState)),
            (
                "perturbation_id",
                "spectrum_id",
                "sweep_config_sha256",
                "state_digest",
                "degree",
                "coefficients",
                "standard_baseline_shape",
                "source_axis_sha256",
                "source_intensity_sha256",
            ),
        )

    def test_prepare_is_deterministic_and_shape_is_read_only_centered_unit_rms(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        repeated_state = self.perturbation.prepare(self.source, self.context)

        self.assertEqual(state.perturbation_id, "p08")
        self.assertEqual(state.spectrum_id, self.source.spectrum_id)
        self.assertEqual(state.sweep_config_sha256, self.config.sha256)
        self.assertEqual(state.degree, repeated_state.degree)
        self.assertEqual(state.coefficients, repeated_state.coefficients)
        self.assertEqual(state.state_digest, repeated_state.state_digest)
        np.testing.assert_array_equal(
            state.standard_baseline_shape,
            repeated_state.standard_baseline_shape,
        )
        self.assertIn(state.degree, (2, 3, 4))
        self.assertEqual(len(state.coefficients), state.degree - 1)
        self.assertFalse(state.standard_baseline_shape.flags.writeable)
        self.assertEqual(state.standard_baseline_shape.dtype, np.dtype("<f8"))
        self.assertTrue(np.isfinite(state.standard_baseline_shape).all())
        self.assertAlmostEqual(
            float(np.mean(state.standard_baseline_shape)),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(state.standard_baseline_shape**2))),
            1.0,
            places=12,
        )

    def test_different_spectrum_id_changes_deterministic_shape(self) -> None:
        first = self.perturbation.prepare(self.source, self.context)
        second = self.perturbation.prepare(
            source_spectrum(spectrum_id="fixture-spectrum-b"),
            self.context,
        )

        self.assertNotEqual(first.state_digest, second.state_digest)
        self.assertFalse(
            np.array_equal(
                first.standard_baseline_shape,
                second.standard_baseline_shape,
            )
        )

    def test_zero_alpha_is_exact_identity_and_positive_alpha_preserves_axis(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)

        zero = self.perturbation.apply(self.source, 0.0, state)
        self.assertFalse(zero.axis_changed)
        self.assertFalse(zero.intensity_changed)
        np.testing.assert_array_equal(zero.output.axis_cm1, self.source.axis_cm1)
        np.testing.assert_array_equal(zero.output.intensity, self.source.intensity)
        self.assertIsNot(zero.output.axis_cm1, self.source.axis_cm1)
        self.assertIsNot(zero.output.intensity, self.source.intensity)

        positive = self.perturbation.apply(self.source, 0.2, state)
        self.assertFalse(positive.axis_changed)
        self.assertTrue(positive.intensity_changed)
        np.testing.assert_array_equal(
            positive.output.axis_cm1,
            self.source.axis_cm1,
        )
        self.assertFalse(
            np.array_equal(positive.output.intensity, self.source.intensity)
        )

    def test_positive_alpha_rmse_matches_signal_relative_amplitude_and_is_monotone(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)

        expected_signal_rms = float(
            np.sqrt(np.mean(self.source.intensity.astype(np.float64) ** 2))
        )
        previous_rmse = 0.0
        for alpha in self.config.alpha_grid:
            if alpha == 0.0:
                continue
            result = self.perturbation.apply(self.source, alpha, state)
            self.assertAlmostEqual(
                float(result.diagnostics["signal_rms"]),
                expected_signal_rms,
                places=12,
            )
            self.assertAlmostEqual(
                float(result.diagnostics["amplitude"]),
                alpha * expected_signal_rms,
                places=12,
            )
            self.assertAlmostEqual(
                float(result.diagnostics["realized_added_baseline_rms"]),
                alpha * expected_signal_rms,
                places=12,
            )
            pair = SpectrumPairInput(reference=self.source, candidate=result.output)
            metric_result = evaluate_metric(self.metric, pair)
            realized_rmse = metric_result.outputs[0].value
            self.assertAlmostEqual(
                realized_rmse,
                alpha * expected_signal_rms,
                places=12,
            )
            self.assertGreater(realized_rmse, previous_rmse)
            previous_rmse = realized_rmse

    def test_nonuniform_native_axis_is_accepted_without_interpolation(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        result = self.perturbation.apply(self.source, 0.1, state)

        np.testing.assert_array_equal(result.output.axis_cm1, self.source.axis_cm1)
        self.assertEqual(
            result.diagnostics["baseline_model"],
            "deterministic_legendre_orders_2_to_4",
        )
        self.assertTrue(result.diagnostics["shape_centered"])
        self.assertEqual(float(result.diagnostics["standard_shape_rms"]), 1.0)
        self.assertTrue(result.diagnostics["axis_preserved"])

    def test_apply_accepts_content_identical_reconstruction_but_rejects_content_drift(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        rebuilt = source_spectrum()
        result = self.perturbation.apply(rebuilt, 0.05, state)
        validate_perturbation_result(rebuilt, state, result, self.config)

        drifted_axis = source_spectrum(axis=(113.0, 127.0, 166.0, 245.0, 252.0, 401.0))
        with self.assertRaisesRegex(BaselineDistortionError, "source axis"):
            self.perturbation.apply(drifted_axis, 0.05, state)

        drifted_intensity = source_spectrum(intensity=(2.0, 3.0, 5.0, 7.0, 11.0, 13.5))
        with self.assertRaisesRegex(BaselineDistortionError, "source intensity"):
            self.perturbation.apply(drifted_intensity, 0.05, state)

    def test_apply_rejects_wrong_context_state_alpha_and_positive_round_back_identity(self) -> None:
        with self.assertRaisesRegex(BaselineDistortionError, "context.sweep_id"):
            self.perturbation.prepare(
                self.source,
                replace(self.context, sweep_id="wrong"),
            )

        state = self.perturbation.prepare(self.source, self.context)
        with self.assertRaisesRegex(BaselineDistortionError, "alpha"):
            self.perturbation.apply(self.source, 0.123, state)
        with self.assertRaisesRegex(BaselineDistortionError, "state.sweep_config_sha256"):
            self.perturbation.apply(
                self.source,
                0.05,
                replace(state, sweep_config_sha256="0" * 64),
            )

        tiny = Spectrum1D(
            spectrum_id="tiny-spectrum",
            sample_id="sample-t",
            axis_cm1=np.array([100.0, 120.0, 150.0, 190.0], dtype="<f8"),
            intensity=np.array([5e-324, 5e-324, 1e-323, 5e-324], dtype="<f8"),
        )
        tiny_state = self.perturbation.prepare(tiny, self.context)
        with self.assertRaisesRegex(BaselineDistortionError, "rounds entirely back to identity"):
            self.perturbation.apply(tiny, 0.05, tiny_state)

    def test_zero_signal_branches_to_exact_identity_copy(self) -> None:
        zero = zero_spectrum()
        state = self.perturbation.prepare(zero, self.context)
        result = self.perturbation.apply(zero, 0.8, state)

        self.assertFalse(result.axis_changed)
        self.assertFalse(result.intensity_changed)
        np.testing.assert_array_equal(result.output.axis_cm1, zero.axis_cm1)
        np.testing.assert_array_equal(result.output.intensity, zero.intensity)
        self.assertEqual(float(result.diagnostics["signal_rms"]), 0.0)
        self.assertEqual(float(result.diagnostics["amplitude"]), 0.0)
        self.assertEqual(float(result.diagnostics["realized_added_baseline_rms"]), 0.0)

    def test_apply_rejects_output_overflow(self) -> None:
        huge = Spectrum1D(
            spectrum_id="huge-spectrum",
            sample_id="sample-h",
            axis_cm1=np.array([100.0, 140.0, 200.0, 260.0], dtype="<f8"),
            intensity=np.array([1.6e308, 1.6e308, 1.6e308, 1.6e308], dtype="<f8"),
        )
        state = self.perturbation.prepare(huge, self.context)
        with self.assertRaisesRegex(BaselineDistortionError, "output finite"):
            self.perturbation.apply(huge, 0.8, state)

    def test_constructor_rejects_semantically_mutated_config_and_state_digest_drift(self) -> None:
        bad_config = replace(self.config, global_seed=self.config.global_seed + 1)
        with self.assertRaisesRegex(BaselineDistortionError, "config.global_seed"):
            P8LowOrderBaselineDistortion(bad_config)

        state = self.perturbation.prepare(self.source, self.context)
        with self.assertRaisesRegex(BaselineDistortionError, "state.state_digest"):
            self.perturbation.apply(
                self.source,
                0.1,
                replace(state, state_digest="0" * 64),
            )

    def test_state_and_result_contracts_remain_generic_validator_compatible(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        result = self.perturbation.apply(self.source, 0.3, state)

        self.assertEqual(
            result.output.spectrum_id,
            derive_perturbed_spectrum_id(
                self.source.spectrum_id,
                "p08",
                0.3,
                state.state_digest,
                self.config.sha256,
            ),
        )
        validate_perturbation_result(self.source, state, result, self.config)


class PublicExportsTest(unittest.TestCase):
    def test_public_imports_exist_for_baseline_distortion_operator(self) -> None:
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(P8LowOrderBaselineDistortion(config).perturbation_id, "p08")


if __name__ == "__main__":
    unittest.main()
