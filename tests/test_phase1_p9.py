from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import (  # noqa: E402
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.metrics import MSEMetric, RMSEMetric  # noqa: E402
from rpe.perturb import (  # noqa: E402
    GaussianNoiseError,
    P9GaussianWhiteNoise,
    P9GaussianNoiseState,
    PerturbationContext,
    PerturbationSweepConfigError,
    PerturbationContractError,
    derive_perturbed_spectrum_id,
    derive_state_seed_material,
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
EXPECTED_SEED_MATERIAL = (
    3212070518,
    3624505570,
    1452674395,
    142906084,
)
EXPECTED_STANDARD_NOISE = np.array(
    [
        -0.7570076125090957,
        0.35099831907735685,
        -0.8732786394882762,
        0.06978961181662656,
    ],
    dtype="<f8",
)
EXPECTED_STATE_DIGEST = (
    "a5f68bf7986c81b9aa5c9f2728f227eac402a73f643ae2fe90e8e6c6254a260d"
)
EXPECTED_OUTPUT_IDS = {
    0.0: (
        "perturbed-e3b95985e6b11bfc57105ab11a60f7b3d5f614bf52af6ccee353c353f2e3d968"
    ),
    0.05: (
        "perturbed-a36d109089666b0a811b093a3f06c4540e8ae76679065da81a575b7fa6dd62b5"
    ),
    0.8: (
        "perturbed-d74285630c2cea2d60b4e917151377a55df3a89b477e5380df439e0af238e17a"
    ),
}
EXPECTED_SIGNAL_RMS = 2.7386127875258306
EXPECTED_NOISE_MEAN_SQUARE = 0.3659366293739653
EXPECTED_MSE = {
    0.05: 0.006861311800761851,
    0.8: 1.7564958209950339,
}
EXPECTED_RMSE = {
    0.05: 0.08283303568481509,
    0.8: 1.3253285709570415,
}


def source_spectrum(
    *,
    spectrum_id: str = "fixture-spectrum",
    intensity: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8"),
        intensity=np.array(intensity, dtype="<f8"),
    )


def zero_spectrum() -> Spectrum1D:
    return Spectrum1D(
        spectrum_id="zero-spectrum",
        sample_id="sample-z",
        axis_cm1=np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8"),
        intensity=np.zeros(4, dtype="<f8"),
    )


def perturbation_context():
    return PerturbationContext(
        sweep_id="raman_perturbation_alpha_v1",
        sweep_config_sha256=SWEEP_SHA256,
        global_seed=20260817,
    )


class SharedSweepConfigTest(unittest.TestCase):
    def test_config_has_exact_canonical_identity_and_shared_values(self):
        raw = SWEEP_CONFIG.read_bytes()
        self.assertEqual(len(raw), SWEEP_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SWEEP_SHA256)
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(
            config.alpha_grid,
            (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8),
        )
        self.assertEqual(config.identity_alpha, 0.0)
        self.assertEqual(
            config.perturbation_ids,
            tuple(f"p{n:02d}" for n in range(1, 13)),
        )
        self.assertEqual(config.global_seed, 20260817)
        self.assertEqual(config.bit_generator, "PCG64")
        self.assertEqual(config.sha256, SWEEP_SHA256)

    def test_phase1_and_test_local_phase4_consumer_share_one_identity(self):
        phase1 = load_perturbation_sweep_config(SWEEP_CONFIG)
        phase4_reference = {
            "path": "experiments/shared/raman_perturbation_sweep_v1.json",
            "bytes": SWEEP_BYTES,
            "sha256": SWEEP_SHA256,
        }
        self.assertEqual(
            phase1.path.relative_to(ROOT).as_posix(),
            phase4_reference["path"],
        )
        self.assertEqual(phase1.byte_count, phase4_reference["bytes"])
        self.assertEqual(phase1.sha256, phase4_reference["sha256"])

    def test_config_mutations_fail_closed_before_state_derivation(self):
        canonical = SWEEP_CONFIG.read_text(encoding="utf-8")
        payload = json.loads(canonical)
        mutations = (
            (
                "noncanonical whitespace",
                (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(", ", ": "),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "key reordering",
                (
                    json.dumps(
                        {
                            "bit_generator": payload["bit_generator"],
                            "alpha_grid": payload["alpha_grid"],
                            "global_seed": payload["global_seed"],
                            "identity_alpha": payload["identity_alpha"],
                            "perturbation_ids": payload["perturbation_ids"],
                            "schema_version": payload["schema_version"],
                            "seed_derivation": payload["seed_derivation"],
                            "sweep_id": payload["sweep_id"],
                        },
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "missing final newline",
                canonical.rstrip("\n").encode("utf-8"),
            ),
            (
                "extra key",
                (
                    json.dumps(
                        {
                            **payload,
                            "unexpected": "value",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "duplicate alpha",
                (
                    json.dumps(
                        {
                            **payload,
                            "alpha_grid": [
                                0.0,
                                0.05,
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.65,
                                0.65,
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "reordered alpha",
                (
                    json.dumps(
                        {
                            **payload,
                            "alpha_grid": [
                                0.0,
                                0.1,
                                0.05,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.65,
                                0.8,
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "missing alpha",
                (
                    json.dumps(
                        {
                            **payload,
                            "alpha_grid": [
                                0.0,
                                0.05,
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.65,
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "changed seed",
                (
                    json.dumps(
                        {
                            **payload,
                            "global_seed": 20260818,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "duplicate perturbation id",
                (
                    json.dumps(
                        {
                            **payload,
                            "perturbation_ids": [
                                "p01",
                                "p02",
                                "p03",
                                "p04",
                                "p05",
                                "p06",
                                "p07",
                                "p08",
                                "p09",
                                "p10",
                                "p11",
                                "p11",
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "missing perturbation id",
                (
                    json.dumps(
                        {
                            **payload,
                            "perturbation_ids": payload["perturbation_ids"][:-1],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "nonfinite JSON constant",
                canonical.replace(":20260817,", ":NaN,", 1).encode("utf-8"),
            ),
            (
                "wrong seed-derivation field",
                (
                    json.dumps(
                        {
                            **payload,
                            "seed_derivation": {
                                **payload["seed_derivation"],
                                "fields": ["spectrum_id", "perturbation_id"],
                            },
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "wrong bytes with same shape",
                canonical.replace("PCG64", "PCG63", 1).encode("utf-8"),
            ),
        )
        for label, raw in mutations:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / SWEEP_CONFIG.name
                    path.write_bytes(raw)
                    with self.assertRaises(PerturbationSweepConfigError):
                        load_perturbation_sweep_config(path)

    def test_literal_seed_oracle_and_seed_material_sensitivity(self):
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(
            derive_state_seed_material(
                config,
                perturbation_id="p09",
                spectrum_id="fixture-spectrum",
            ),
            (3212070518, 3624505570, 1452674395, 142906084),
        )
        self.assertNotEqual(
            derive_state_seed_material(
                config,
                perturbation_id="p09",
                spectrum_id="fixture-spectrum",
            ),
            derive_state_seed_material(
                config,
                perturbation_id="p09",
                spectrum_id="fixture-spectrum-2",
            ),
        )
        self.assertNotEqual(
            derive_state_seed_material(
                config,
                perturbation_id="p09",
                spectrum_id="fixture-spectrum",
            ),
            derive_state_seed_material(
                config,
                perturbation_id="p10",
                spectrum_id="fixture-spectrum",
            ),
        )
        self.assertNotEqual(
            derive_state_seed_material(
                config,
                perturbation_id="p09",
                spectrum_id="fixture-spectrum",
            ),
            derive_state_seed_material(
                replace(config, global_seed=20260818),
                perturbation_id="p09",
                spectrum_id="fixture-spectrum",
            ),
        )


class P9GaussianWhiteNoiseTest(unittest.TestCase):
    def setUp(self):
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = source_spectrum()
        self.zero_source = zero_spectrum()
        self.perturbation = P9GaussianWhiteNoise(self.config)

    def test_prepare_has_literal_seed_noise_digest_and_repeatability(self):
        state = self.perturbation.prepare(self.source, self.context)
        repeated = self.perturbation.prepare(self.source, self.context)

        self.assertEqual(
            derive_state_seed_material(
                self.config,
                perturbation_id="p09",
                spectrum_id="fixture-spectrum",
            ),
            EXPECTED_SEED_MATERIAL,
        )
        self.assertEqual(state.perturbation_id, "p09")
        self.assertEqual(state.spectrum_id, self.source.spectrum_id)
        self.assertEqual(state.sweep_config_sha256, self.config.sha256)
        self.assertEqual(state.state_digest, EXPECTED_STATE_DIGEST)
        np.testing.assert_allclose(
            state.standard_noise,
            EXPECTED_STANDARD_NOISE,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(state.standard_noise.dtype, np.dtype("<f8"))
        self.assertTrue(state.standard_noise.flags.c_contiguous)
        self.assertFalse(state.standard_noise.flags.writeable)
        self.assertIsNot(state.standard_noise, repeated.standard_noise)
        self.assertFalse(np.shares_memory(state.standard_noise, repeated.standard_noise))
        np.testing.assert_array_equal(state.standard_noise, repeated.standard_noise)
        self.assertEqual(state.state_digest, repeated.state_digest)

    def test_state_dataclass_has_exactly_five_public_fields(self):
        self.assertEqual(
            tuple(field.name for field in fields(P9GaussianNoiseState)),
            (
                "perturbation_id",
                "spectrum_id",
                "sweep_config_sha256",
                "state_digest",
                "standard_noise",
            ),
        )

    def test_constructor_rejects_semantically_mutated_config_before_prepare(self):
        bad_config = replace(self.config, global_seed=self.config.global_seed + 1)
        with self.assertRaisesRegex(
            GaussianNoiseError,
            "config.global_seed",
        ):
            P9GaussianWhiteNoise(bad_config)

    def test_constructor_rejects_forged_frozen_config_identity_before_prepare(self):
        forged_configs = (
            (
                "sha256",
                replace(self.config, sha256="0" * 64),
            ),
            (
                "byte_count",
                replace(self.config, byte_count=1),
            ),
        )
        for field_name, forged_config in forged_configs:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    GaussianNoiseError,
                    f"config.{field_name}",
                ):
                    P9GaussianWhiteNoise(forged_config)

    def test_unregistered_but_digest_valid_five_field_state_fails_closed_at_apply(self):
        seed_noise = np.array(EXPECTED_STANDARD_NOISE, dtype="<f8")
        forged_state = P9GaussianNoiseState(
            perturbation_id="p09",
            spectrum_id=self.source.spectrum_id,
            sweep_config_sha256=self.config.sha256,
            state_digest=EXPECTED_STATE_DIGEST,
            standard_noise=seed_noise,
        )
        with self.assertRaisesRegex(
            GaussianNoiseError,
            "prepared state",
        ):
            self.perturbation.apply(self.source, 0.05, forged_state)

    def test_content_identical_reconstructed_source_is_accepted(self):
        state = self.perturbation.prepare(self.source, self.context)
        reconstructed_source = Spectrum1D(
            spectrum_id=self.source.spectrum_id,
            sample_id=self.source.sample_id,
            axis_cm1=np.array(self.source.axis_cm1, dtype="<f8", copy=True),
            intensity=np.array(self.source.intensity, dtype="<f8", copy=True),
        )
        baseline = self.perturbation.apply(self.source, 0.05, state)
        reconstructed = self.perturbation.apply(reconstructed_source, 0.05, state)

        self.assertEqual(reconstructed.perturbation_id, baseline.perturbation_id)
        self.assertEqual(reconstructed.source_spectrum_id, baseline.source_spectrum_id)
        self.assertEqual(reconstructed.alpha, baseline.alpha)
        self.assertEqual(reconstructed.state_digest, baseline.state_digest)
        self.assertEqual(reconstructed.output.spectrum_id, baseline.output.spectrum_id)
        self.assertEqual(dict(reconstructed.diagnostics), dict(baseline.diagnostics))
        self.assertIsNot(reconstructed.output, baseline.output)
        self.assertIsNot(reconstructed.output.axis_cm1, baseline.output.axis_cm1)
        self.assertIsNot(reconstructed.output.intensity, baseline.output.intensity)
        np.testing.assert_array_equal(
            reconstructed.output.axis_cm1,
            baseline.output.axis_cm1,
        )
        np.testing.assert_array_equal(
            reconstructed.output.intensity,
            baseline.output.intensity,
        )

    def test_value_equivalent_external_state_requires_matching_registered_key(self):
        registered_state = self.perturbation.prepare(self.source, self.context)
        external_state = P9GaussianNoiseState(
            perturbation_id=registered_state.perturbation_id,
            spectrum_id=registered_state.spectrum_id,
            sweep_config_sha256=registered_state.sweep_config_sha256,
            state_digest=registered_state.state_digest,
            standard_noise=np.array(
                registered_state.standard_noise,
                dtype="<f8",
                copy=True,
            ),
        )
        fresh_instance = P9GaussianWhiteNoise(self.config)
        accepted = self.perturbation.apply(
            self.source,
            0.05,
            external_state,
        )

        self.assertEqual(accepted.source_spectrum_id, self.source.spectrum_id)
        self.assertEqual(accepted.state_digest, external_state.state_digest)

        with self.assertRaisesRegex(
            GaussianNoiseError,
            "prepared state",
        ):
            fresh_instance.apply(self.source, 0.05, external_state)

        fresh_instance.prepare(self.source, self.context)
        result = fresh_instance.apply(self.source, 0.05, external_state)

        self.assertEqual(result.source_spectrum_id, self.source.spectrum_id)
        self.assertEqual(result.state_digest, external_state.state_digest)

    def test_same_id_prepares_register_multiple_content_bindings_without_order_dependence(self):
        source_a = source_spectrum(
            spectrum_id="shared-id",
            intensity=(1.0, 2.0, 3.0, 4.0),
        )
        source_b = source_spectrum(
            spectrum_id="shared-id",
            intensity=(4.0, 3.0, 2.0, 1.0),
        )
        source_c = source_spectrum(
            spectrum_id="shared-id",
            intensity=(1.0, 4.0, 2.0, 3.0),
        )
        state_a = self.perturbation.prepare(source_a, self.context)
        state_b = self.perturbation.prepare(source_b, self.context)

        self.assertEqual(state_a.perturbation_id, state_b.perturbation_id)
        self.assertEqual(state_a.spectrum_id, state_b.spectrum_id)
        self.assertEqual(
            state_a.sweep_config_sha256,
            state_b.sweep_config_sha256,
        )
        self.assertEqual(state_a.state_digest, state_b.state_digest)
        np.testing.assert_array_equal(state_a.standard_noise, state_b.standard_noise)

        for label, source in (("a", source_a), ("b", source_b)):
            for state_label, state in (("a", state_a), ("b", state_b)):
                with self.subTest(source=label, state=state_label):
                    result = self.perturbation.apply(source, 0.05, state)
                    self.assertEqual(result.source_spectrum_id, source.spectrum_id)
                    self.assertFalse(
                        np.array_equal(
                            result.output.intensity,
                            source.intensity,
                        )
                    )

        for state_label, state in (("a", state_a), ("b", state_b)):
            with self.subTest(source="unregistered", state=state_label):
                with self.assertRaisesRegex(
                    GaussianNoiseError,
                    "source axis|source intensity",
                ):
                    self.perturbation.apply(source_c, 0.05, state)

    def test_tiny_finite_signal_uses_scaled_rms_and_representable_noise(self):
        tiny_source = Spectrum1D(
            spectrum_id="tiny-signal",
            sample_id="sample-tiny",
            axis_cm1=np.array([100.0, 200.0], dtype="<f8"),
            intensity=np.array([1.0e-300, 1.0e-300], dtype="<f8"),
        )
        state = self.perturbation.prepare(tiny_source, self.context)
        self.assertTrue(np.any(state.standard_noise != 0.0))

        alpha = 0.8
        result = self.perturbation.apply(tiny_source, alpha, state)
        scale = float(np.max(np.abs(tiny_source.intensity)))
        expected_signal_rms = scale * float(
            np.sqrt(np.mean((tiny_source.intensity / scale) ** 2))
        )
        expected_sigma = alpha * expected_signal_rms
        expected_intensity = (
            tiny_source.intensity + expected_sigma * state.standard_noise
        )

        self.assertTrue(math.isfinite(result.diagnostics["signal_rms"]))
        self.assertTrue(math.isfinite(result.diagnostics["sigma"]))
        np.testing.assert_allclose(
            result.diagnostics["signal_rms"],
            1.0e-300,
            rtol=1.0e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result.diagnostics["sigma"],
            8.0e-301,
            rtol=1.0e-15,
            atol=0.0,
        )
        self.assertGreater(result.diagnostics["sigma"], 0.0)
        self.assertFalse(
            np.array_equal(result.output.intensity, tiny_source.intensity)
        )
        np.testing.assert_allclose(
            result.output.intensity,
            expected_intensity,
            rtol=0.0,
            atol=0.0,
        )

        rmse_values = []
        for selected_alpha in (0.05, 0.2, 0.8):
            selected_result = self.perturbation.apply(
                tiny_source,
                selected_alpha,
                state,
            )
            rmse_values.append(
                evaluate_metric(
                    RMSEMetric(),
                    SpectrumPairInput(tiny_source, selected_result.output),
                ).outputs[0].value
            )
        self.assertTrue(all(value > 0.0 for value in rmse_values))
        self.assertTrue(
            all(
                right > left
                for left, right in zip(rmse_values, rmse_values[1:])
            )
        )

    def test_same_id_source_with_changed_axis_or_intensity_is_rejected(self):
        state = self.perturbation.prepare(self.source, self.context)
        mutated_axis = Spectrum1D(
            spectrum_id=self.source.spectrum_id,
            sample_id=self.source.sample_id,
            axis_cm1=np.array([100.0, 200.0, 300.0, 401.0], dtype="<f8"),
            intensity=np.array(self.source.intensity, dtype="<f8", copy=True),
        )
        mutated_intensity = Spectrum1D(
            spectrum_id=self.source.spectrum_id,
            sample_id=self.source.sample_id,
            axis_cm1=np.array(self.source.axis_cm1, dtype="<f8", copy=True),
            intensity=np.array([1.0, 2.0, 3.0, 4.1], dtype="<f8"),
        )

        with self.assertRaisesRegex(
            GaussianNoiseError,
            "source axis",
        ):
            self.perturbation.apply(mutated_axis, 0.05, state)
        with self.assertRaisesRegex(
            GaussianNoiseError,
            "source intensity",
        ):
            self.perturbation.apply(mutated_intensity, 0.05, state)

    def test_state_constructor_rejects_wrong_dtype_length_and_nonfinite_noise(self):
        invalid_states = (
            (
                "standard_noise dtype",
                np.array([0.0, 1.0, 2.0, 3.0], dtype="<f4"),
            ),
            (
                "standard_noise finite",
                np.array([0.0, 1.0, np.nan, 3.0], dtype="<f8"),
            ),
            (
                "standard_noise shape",
                np.array([], dtype="<f8"),
            ),
        )
        for expected_message, noise in invalid_states:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(GaussianNoiseError, expected_message):
                    P9GaussianNoiseState(
                        perturbation_id="p09",
                        spectrum_id=self.source.spectrum_id,
                        sweep_config_sha256=self.config.sha256,
                        state_digest=EXPECTED_STATE_DIGEST,
                        standard_noise=noise,
                    )

    def test_apply_rejects_wrong_digest_and_source_identity_after_constructor_accepts_state(self):
        state = self.perturbation.prepare(self.source, self.context)
        wrong_digest_state = replace(state, state_digest="f" * 64)
        wrong_source_state = P9GaussianNoiseState(
            perturbation_id="p09",
            spectrum_id="other-spectrum",
            sweep_config_sha256=self.config.sha256,
            state_digest="dbe67a0770a259a17921e6890e172cc1f9cf02b89d25c9df63496f3038af5829",
            standard_noise=np.array(state.standard_noise, dtype="<f8", copy=True),
        )

        with self.assertRaisesRegex(
            GaussianNoiseError,
            "state.state_digest",
        ):
            self.perturbation.apply(self.source, 0.05, wrong_digest_state)
        with self.assertRaisesRegex(
            GaussianNoiseError,
            "state.spectrum_id|prepared state",
        ):
            self.perturbation.apply(self.source, 0.05, wrong_source_state)

    def test_apply_matches_literal_output_ids_formula_diagnostics_and_monotonicity(self):
        state = self.perturbation.prepare(self.source, self.context)
        source_axis_before = self.source.axis_cm1.copy()
        source_intensity_before = self.source.intensity.copy()
        state_noise_before = state.standard_noise.copy()

        mse_values: list[float] = []
        rmse_values: list[float] = []
        positive_alphas = tuple(
            alpha
            for alpha in self.config.alpha_grid
            if alpha != self.config.identity_alpha
        )

        for alpha in self.config.alpha_grid:
            with self.subTest(alpha=alpha):
                result = self.perturbation.apply(self.source, alpha, state)
                validate_perturbation_result(
                    self.source,
                    state,
                    result,
                    self.config,
                )
                self.assertIsNot(result.output, self.source)
                self.assertIsNot(result.output.axis_cm1, self.source.axis_cm1)
                self.assertIsNot(
                    result.output.intensity,
                    self.source.intensity,
                )
                self.assertFalse(np.shares_memory(result.output.axis_cm1, self.source.axis_cm1))
                self.assertFalse(
                    np.shares_memory(result.output.intensity, self.source.intensity)
                )
                self.assertEqual(result.output.spectrum_id, derive_perturbed_spectrum_id(
                    self.source.spectrum_id,
                    "p09",
                    alpha,
                    state.state_digest,
                    self.config.sha256,
                ))
                self.assertEqual(
                    result.output.spectrum_id,
                    derive_perturbed_spectrum_id(
                        self.source.spectrum_id,
                        result.perturbation_id,
                        result.alpha,
                        result.state_digest,
                        self.config.sha256,
                    ),
                )
                self.assertEqual(result.perturbation_id, "p09")
                self.assertEqual(result.source_spectrum_id, self.source.spectrum_id)
                self.assertEqual(result.alpha, alpha)
                self.assertEqual(result.state_digest, EXPECTED_STATE_DIGEST)
                self.assertEqual(
                    dict(result.diagnostics),
                    {
                        "noise_mean_square": EXPECTED_NOISE_MEAN_SQUARE,
                        "signal_rms": EXPECTED_SIGNAL_RMS,
                        "sigma": float(alpha * EXPECTED_SIGNAL_RMS),
                    },
                )
                np.testing.assert_array_equal(result.output.axis_cm1, self.source.axis_cm1)
                if alpha == 0.0:
                    self.assertEqual(
                        result.output.spectrum_id,
                        EXPECTED_OUTPUT_IDS[alpha],
                    )
                    np.testing.assert_array_equal(
                        result.output.intensity,
                        self.source.intensity,
                    )
                    self.assertFalse(result.axis_changed)
                    self.assertFalse(result.intensity_changed)
                else:
                    self.assertFalse(result.axis_changed)
                    self.assertTrue(result.intensity_changed)
                    mse = evaluate_metric(
                        MSEMetric(),
                        SpectrumPairInput(self.source, result.output),
                    ).outputs[0].value
                    rmse = evaluate_metric(
                        RMSEMetric(),
                        SpectrumPairInput(self.source, result.output),
                    ).outputs[0].value
                    mse_values.append(mse)
                    rmse_values.append(rmse)
                    if alpha in EXPECTED_OUTPUT_IDS:
                        self.assertEqual(
                            result.output.spectrum_id,
                            EXPECTED_OUTPUT_IDS[alpha],
                        )
                    if alpha in EXPECTED_MSE:
                        self.assertAlmostEqual(mse, EXPECTED_MSE[alpha], places=15)
                    if alpha in EXPECTED_RMSE:
                        self.assertAlmostEqual(rmse, EXPECTED_RMSE[alpha], places=15)

        self.assertEqual(len(mse_values), len(positive_alphas))
        self.assertEqual(len(rmse_values), len(positive_alphas))
        self.assertTrue(
            all(
                right > left
                for left, right in zip(mse_values, mse_values[1:])
            )
        )
        self.assertTrue(
            all(
                right > left
                for left, right in zip(rmse_values, rmse_values[1:])
            )
        )
        np.testing.assert_array_equal(self.source.axis_cm1, source_axis_before)
        np.testing.assert_array_equal(self.source.intensity, source_intensity_before)
        np.testing.assert_array_equal(state.standard_noise, state_noise_before)

    def test_zero_signal_remains_exact_zero_and_degradation_is_non_decreasing(self):
        state = self.perturbation.prepare(self.zero_source, self.context)
        repeated = self.perturbation.prepare(self.zero_source, self.context)
        np.testing.assert_array_equal(state.standard_noise, repeated.standard_noise)
        self.assertEqual(state.state_digest, repeated.state_digest)
        zero_noise_mean_square = float(np.mean(state.standard_noise**2))

        mse_values = []
        rmse_values = []
        for alpha in self.config.alpha_grid:
            with self.subTest(alpha=alpha):
                result = self.perturbation.apply(self.zero_source, alpha, state)
                validate_perturbation_result(
                    self.zero_source,
                    state,
                    result,
                    self.config,
                )
                np.testing.assert_array_equal(
                    result.output.axis_cm1,
                    self.zero_source.axis_cm1,
                )
                np.testing.assert_array_equal(
                    result.output.intensity,
                    np.zeros(4, dtype="<f8"),
                )
                self.assertTrue(np.isfinite(result.output.intensity).all())
                self.assertEqual(
                    dict(result.diagnostics),
                    {
                        "noise_mean_square": zero_noise_mean_square,
                        "signal_rms": 0.0,
                        "sigma": 0.0,
                    },
                )
                mse_values.append(
                    evaluate_metric(
                        MSEMetric(),
                        SpectrumPairInput(self.zero_source, result.output),
                    ).outputs[0].value
                )
                rmse_values.append(
                    evaluate_metric(
                        RMSEMetric(),
                        SpectrumPairInput(self.zero_source, result.output),
                    ).outputs[0].value
                )

        self.assertEqual(mse_values, [0.0] * len(self.config.alpha_grid))
        self.assertEqual(rmse_values, [0.0] * len(self.config.alpha_grid))
        self.assertTrue(
            all(
                right >= left
                for left, right in zip(mse_values, mse_values[1:])
            )
        )
        self.assertTrue(
            all(
                right >= left
                for left, right in zip(rmse_values, rmse_values[1:])
            )
        )

    def test_invalid_state_context_source_alpha_and_overflow_fail_closed(self):
        state = self.perturbation.prepare(self.source, self.context)

        invalid_state_factories = (
            lambda: replace(state, perturbation_id="p10"),
            lambda: replace(state, spectrum_id="other-spectrum"),
            lambda: replace(state, sweep_config_sha256="0" * 64),
        )
        for factory in invalid_state_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(GaussianNoiseError):
                    invalid_state = factory()
                    self.perturbation.apply(self.source, 0.05, invalid_state)

        invalid_contexts = (
            replace(self.context, sweep_id="wrong"),
            replace(self.context, sweep_config_sha256="0" * 64),
            replace(self.context, global_seed=20260818),
        )
        for invalid_context in invalid_contexts:
            with self.subTest(invalid_context=invalid_context):
                with self.assertRaises(GaussianNoiseError):
                    self.perturbation.prepare(self.source, invalid_context)

        unsupported_alphas = (0.05000000000000001, 0.66, -0.05, -0.0)
        for alpha in unsupported_alphas:
            with self.subTest(alpha=alpha):
                with self.assertRaises(GaussianNoiseError):
                    self.perturbation.apply(self.source, alpha, state)

        with self.assertRaises(GaussianNoiseError):
            self.perturbation.apply(
                source_spectrum(spectrum_id="different-source"),
                0.05,
                state,
            )
        with self.assertRaises(GaussianNoiseError):
            self.perturbation.apply(
                Spectrum1D(
                    spectrum_id=self.source.spectrum_id,
                    sample_id="sample-a",
                    axis_cm1=np.array([100.0, 200.0, 300.0], dtype="<f8"),
                    intensity=np.array([1.0, 2.0, 3.0], dtype="<f8"),
                ),
                0.05,
                state,
            )

        mutated_source = source_spectrum()
        prepared = self.perturbation.prepare(mutated_source, self.context)
        mutated_view = mutated_source.intensity.copy()
        mutated_view[0] = 999.0
        mutated_source = Spectrum1D(
            spectrum_id="fixture-spectrum",
            sample_id="sample-a",
            axis_cm1=np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8"),
            intensity=mutated_view,
        )
        with self.assertRaises(GaussianNoiseError):
            self.perturbation.apply(mutated_source, 0.05, prepared)

        overflow_source = Spectrum1D(
            spectrum_id="overflow-spectrum",
            sample_id="sample-o",
            axis_cm1=np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8"),
            intensity=np.array([1.0e308, 1.0e308, 1.0e308, 1.0e308], dtype="<f8"),
        )
        overflow_state = self.perturbation.prepare(overflow_source, self.context)
        with self.assertRaisesRegex(GaussianNoiseError, "output finite"):
            self.perturbation.apply(overflow_source, 0.8, overflow_state)


if __name__ == "__main__":
    unittest.main()
