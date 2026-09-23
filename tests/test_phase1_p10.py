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
    CorrelatedNoiseError,
    P10CorrelatedNoise,
    P10CorrelatedNoiseState,
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
    axis: tuple[float, ...] = (100.0, 110.0, 130.0, 170.0, 250.0, 410.0),
    intensity: tuple[float, ...] = (1.0, 2.0, 4.0, 7.0, 11.0, 16.0),
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.array(axis, dtype="<f8"),
        intensity=np.array(intensity, dtype="<f8"),
    )


def regular_long_spectrum(
    *,
    spectrum_id: str = "regular-spectrum",
    points: int = 25,
) -> Spectrum1D:
    axis = np.linspace(100.0, 340.0, points, dtype="<f8")
    intensity = np.linspace(1.0, 3.4, points, dtype="<f8")
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-b",
        axis_cm1=axis,
        intensity=intensity,
    )


def zero_spectrum() -> Spectrum1D:
    return Spectrum1D(
        spectrum_id="zero-spectrum",
        sample_id="sample-z",
        axis_cm1=np.array([100.0, 120.0, 150.0, 190.0], dtype="<f8"),
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
        self.assertIn("p10", config.perturbation_ids)


class P10CorrelatedNoiseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = source_spectrum()
        self.metric = RMSEMetric()
        self.perturbation = P10CorrelatedNoise(self.config)

    def test_state_dataclass_has_five_public_fields(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(P10CorrelatedNoiseState)),
            (
                "perturbation_id",
                "spectrum_id",
                "sweep_config_sha256",
                "state_digest",
                "standard_correlated_noise",
            ),
        )

    def test_prepare_is_deterministic_and_noise_is_read_only_centered_unit_rms(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        repeated_state = self.perturbation.prepare(self.source, self.context)

        self.assertEqual(state.perturbation_id, "p10")
        self.assertEqual(state.spectrum_id, self.source.spectrum_id)
        self.assertEqual(state.sweep_config_sha256, self.config.sha256)
        self.assertEqual(state.state_digest, repeated_state.state_digest)
        np.testing.assert_array_equal(
            state.standard_correlated_noise,
            repeated_state.standard_correlated_noise,
        )
        self.assertFalse(state.standard_correlated_noise.flags.writeable)
        self.assertEqual(
            state.standard_correlated_noise.dtype,
            np.dtype("<f8"),
        )
        self.assertTrue(np.isfinite(state.standard_correlated_noise).all())
        self.assertAlmostEqual(
            float(np.mean(state.standard_correlated_noise)),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(state.standard_correlated_noise**2))),
            1.0,
            places=12,
        )

    def test_different_spectrum_id_changes_deterministic_noise(self) -> None:
        first = self.perturbation.prepare(self.source, self.context)
        second = self.perturbation.prepare(
            source_spectrum(spectrum_id="fixture-spectrum-b"),
            self.context,
        )

        self.assertNotEqual(first.state_digest, second.state_digest)
        self.assertFalse(
            np.array_equal(
                first.standard_correlated_noise,
                second.standard_correlated_noise,
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

    def test_positive_alpha_rmse_matches_signal_relative_sigma_and_is_monotone(self) -> None:
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
                float(result.diagnostics["sigma"]),
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

    def test_prepare_noise_shows_stronger_nearby_than_distant_correlation_on_regular_axis(self) -> None:
        spectrum = regular_long_spectrum()
        state = self.perturbation.prepare(spectrum, self.context)

        nearby = np.mean(
            state.standard_correlated_noise[:-1]
            * state.standard_correlated_noise[1:]
        )
        distant = np.mean(
            state.standard_correlated_noise[:-8]
            * state.standard_correlated_noise[8:]
        )
        self.assertGreater(nearby, distant)

    def test_nonuniform_axis_is_accepted_without_resampling(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        result = self.perturbation.apply(self.source, 0.1, state)

        np.testing.assert_array_equal(result.output.axis_cm1, self.source.axis_cm1)
        self.assertEqual(
            result.diagnostics["noise_model"],
            "native_axis_gaussian_correlated",
        )
        self.assertEqual(result.diagnostics["correlation_length_cm1"], 20.0)
        self.assertEqual(result.diagnostics["kernel_normalization"], "row_sum_one")
        self.assertEqual(
            result.diagnostics["boundary_mode"],
            "full_native_gaussian_kernel",
        )

    def test_apply_accepts_content_identical_reconstruction_but_rejects_content_drift(self) -> None:
        state = self.perturbation.prepare(self.source, self.context)
        rebuilt = source_spectrum()
        result = self.perturbation.apply(rebuilt, 0.05, state)
        validate_perturbation_result(rebuilt, state, result, self.config)

        drifted = source_spectrum(intensity=(1.0, 2.0, 4.0, 7.0, 11.0, 16.5))
        with self.assertRaisesRegex(CorrelatedNoiseError, "source intensity"):
            self.perturbation.apply(drifted, 0.05, state)

    def test_apply_rejects_wrong_context_state_alpha_and_positive_round_back_identity(self) -> None:
        with self.assertRaisesRegex(CorrelatedNoiseError, "context.sweep_id"):
            self.perturbation.prepare(
                self.source,
                replace(self.context, sweep_id="wrong"),
            )

        state = self.perturbation.prepare(self.source, self.context)
        with self.assertRaisesRegex(CorrelatedNoiseError, "alpha"):
            self.perturbation.apply(self.source, 0.123, state)
        with self.assertRaisesRegex(CorrelatedNoiseError, "state.sweep_config_sha256"):
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
        with self.assertRaisesRegex(CorrelatedNoiseError, "rounds entirely back to identity"):
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
        self.assertEqual(float(result.diagnostics["sigma"]), 0.0)

    def test_constructor_rejects_mutated_config_and_state_digest_drift(self) -> None:
        bad_config = replace(self.config, global_seed=self.config.global_seed + 1)
        with self.assertRaisesRegex(CorrelatedNoiseError, "config.global_seed"):
            P10CorrelatedNoise(bad_config)

        state = self.perturbation.prepare(self.source, self.context)
        with self.assertRaisesRegex(CorrelatedNoiseError, "state.state_digest"):
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
                "p10",
                0.3,
                state.state_digest,
                self.config.sha256,
            ),
        )
        validate_perturbation_result(self.source, state, result, self.config)
        self.assertAlmostEqual(
            float(result.diagnostics["standard_noise_mean"]),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(result.diagnostics["standard_noise_rms"]),
            1.0,
            places=12,
        )


class PublicExportsTest(unittest.TestCase):
    def test_public_imports_exist_for_correlated_noise_operator(self) -> None:
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(P10CorrelatedNoise(config).perturbation_id, "p10")


if __name__ == "__main__":
    unittest.main()
