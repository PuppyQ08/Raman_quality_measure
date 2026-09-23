from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.methods import TaskLine, load_classical_catalog  # noqa: E402
from rpe.methods.classical import (  # noqa: E402
    DenoisingCanaryError,
    DenoisingFitContext,
    DenoisingFitError,
    DenoisingRunStatus,
    build_denoising_canary_artifact,
    fit_denoising_system,
    load_denoising_canary_config,
    load_denoising_canary_inputs,
    run_stateless_denoising_system,
    transform_fitted_denoiser,
    verify_denoising_canary_artifact,
)
from rpe.methods.classical import denoising as denoising_module  # noqa: E402
from rpe.methods.classical import denoising_canary as canary_module  # noqa: E402


CATALOG_PATH = ROOT / "experiments" / "phase3" / "configs" / "classical_system_catalog_v1.json"
CONFIG_PATH = ROOT / "experiments" / "phase3" / "configs" / "denoising_v1_canary.json"


def system(family: str, **selector: object):
    catalog = load_classical_catalog(CATALOG_PATH)
    matches = [
        value
        for value in catalog.systems
        if value.task_line is TaskLine.DENOISING
        and value.family_id == family
        and all(value.hyperparameters.get(key) == expected for key, expected in selector.items())
    ]
    if len(matches) != 1:
        raise AssertionError((family, selector, len(matches)))
    return matches[0]


def spectrum(spectrum_id: str, intensity: np.ndarray, *, axis: np.ndarray | None = None) -> Spectrum1D:
    values = np.asarray(intensity, dtype="<f8")
    physical_axis = (
        np.arange(values.size, dtype="<f8")
        if axis is None
        else np.asarray(axis, dtype="<f8")
    )
    return Spectrum1D(spectrum_id, None, physical_axis, values)


def context(*record_ids: str, split_id: str = "train") -> DenoisingFitContext:
    return DenoisingFitContext(
        split_id=split_id,
        representation_id="fixture-increasing-f8",
        record_ids=tuple(record_ids),
    )


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class DenoisingWrapperOracleTest(unittest.TestCase):
    def test_sg_preserves_quadratic_and_matches_hand_5_2_center(self) -> None:
        sg = system("savitzky_golay", window_length=5, polyorder=2)
        quadratic = spectrum("quadratic", np.arange(9, dtype="<f8") ** 2)
        result = run_stateless_denoising_system(sg, quadratic)
        self.assertIs(result.status, DenoisingRunStatus.COMPLETE)
        assert result.denoised_intensity is not None
        np.testing.assert_allclose(result.denoised_intensity, quadratic.intensity, rtol=0, atol=1e-12)

        arbitrary = spectrum("arbitrary", np.array([3, 1, 4, 1, 5, 9, 2], dtype="<f8"))
        observed = run_stateless_denoising_system(sg, arbitrary)
        assert observed.denoised_intensity is not None
        expected_center = float(np.dot(np.array([-3, 12, 17, 12, -3]) / 35, arbitrary.intensity[:5]))
        self.assertAlmostEqual(float(observed.denoised_intensity[2]), expected_center, places=14)

    def test_wavelet_threshold_helpers_match_hand_values_and_constant_signal(self) -> None:
        detail = np.array([1.0, 2.0, 100.0], dtype="<f8")
        sigma = denoising_module._wavelet_noise_sigma(detail)
        self.assertEqual(sigma, 1.0 / 0.6744897501960817)
        universal = denoising_module._wavelet_thresholds(
            (detail,), sigma=sigma, strategy="universal_mad", signal_size=16
        )
        self.assertEqual(universal, (sigma * math.sqrt(2.0 * math.log(16.0)),))
        variance = float(np.mean(detail * detail))
        expected_bayes = sigma * sigma / math.sqrt(variance - sigma * sigma)
        bayes = denoising_module._wavelet_thresholds(
            (detail,), sigma=sigma, strategy="bayes_shrink", signal_size=16
        )
        self.assertAlmostEqual(bayes[0], expected_bayes, places=15)
        self.assertEqual(
            denoising_module._wavelet_thresholds(
                (np.zeros(4, dtype="<f8"),),
                sigma=1.0,
                strategy="bayes_shrink",
                signal_size=8,
            ),
            (math.inf,),
        )

        constant = spectrum("constant", np.full(64, 7.0, dtype="<f8"))
        for mode in ("soft", "hard"):
            wavelet = system(
                "wavelet",
                wavelet="db4",
                threshold_mode=mode,
                threshold_strategy="universal_mad",
            )
            result = run_stateless_denoising_system(wavelet, constant)
            self.assertIs(result.status, DenoisingRunStatus.COMPLETE)
            assert result.denoised_intensity is not None
            self.assertEqual(result.denoised_intensity.shape, constant.intensity.shape)
            np.testing.assert_allclose(result.denoised_intensity, 7.0, rtol=0, atol=1e-12)

    def test_whittaker_matches_dense_oracle_and_preserves_linear_nullspace(self) -> None:
        whittaker = system("whittaker_smoothing", **{"lambda": 10.0})
        np.testing.assert_array_equal(
            denoising_module._whittaker_banded_matrix(5, 10.0),
            np.array(
                [
                    [11.0, 51.0, 61.0, 51.0, 11.0],
                    [-20.0, -40.0, -40.0, -20.0, 0.0],
                    [10.0, 10.0, 10.0, 0.0, 0.0],
                ],
                dtype="<f8",
            ),
        )
        source = spectrum("whittaker", np.array([0.0, 1.0, 5.0, 2.0, 4.0], dtype="<f8"))
        result = run_stateless_denoising_system(whittaker, source)
        self.assertIs(result.status, DenoisingRunStatus.COMPLETE)
        assert result.denoised_intensity is not None
        d2 = np.array(
            [[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]],
            dtype="<f8",
        )
        expected = np.linalg.solve(np.eye(5) + 10.0 * d2.T @ d2, source.intensity)
        np.testing.assert_allclose(result.denoised_intensity, expected, rtol=0, atol=1e-13)
        for values in (np.full(8, 4.0), np.linspace(-3.0, 9.0, 8)):
            preserved = run_stateless_denoising_system(whittaker, spectrum("null", values))
            assert preserved.denoised_intensity is not None
            np.testing.assert_allclose(preserved.denoised_intensity, values, rtol=0, atol=2e-13)

    def test_pca_fit_is_centered_train_only_and_rejects_rebound_axis(self) -> None:
        pca = system("pca_reconstruction", n_components=2)
        training = (
            spectrum("p0", np.array([1, 2, 3, 4], dtype="<f8")),
            spectrum("p1", np.array([2, 4, 6, 8], dtype="<f8")),
            spectrum("p2", np.array([3, 6, 9, 12], dtype="<f8")),
        )
        fitted = fit_denoising_system(pca, training, context("p0", "p1", "p2"))
        transformed = transform_fitted_denoiser(fitted, spectrum("held", np.array([4, 8, 12, 16], dtype="<f8")))
        self.assertIs(transformed.status, DenoisingRunStatus.COMPLETE)
        assert transformed.denoised_intensity is not None
        np.testing.assert_allclose(
            transformed.denoised_intensity,
            np.array([4, 8, 12, 16], dtype="<f8"),
            rtol=0,
            atol=3e-14,
        )
        rebound = transform_fitted_denoiser(
            fitted,
            spectrum(
                "held-rebound",
                np.array([4, 8, 12, 16], dtype="<f8"),
                axis=np.array([0.0, 1.0, 2.0, 4.0], dtype="<f8"),
            ),
        )
        self.assertIs(rebound.status, DenoisingRunStatus.FAILED_DOMAIN)
        self.assertEqual(rebound.error_code, "axis_identity_mismatch")
        self.assertIsNone(rebound.denoised_intensity)
        with self.assertRaisesRegex(DenoisingFitError, "record IDs"):
            fit_denoising_system(pca, training, context("p0", "p1", "wrong"))

    def test_svd_fit_is_uncentered_seeded_and_byte_deterministic(self) -> None:
        svd = system("svd_reconstruction", n_components=2)
        training = (
            spectrum("s0", np.array([1, 0, 1, 0], dtype="<f8")),
            spectrum("s1", np.array([0, 1, 0, 1], dtype="<f8")),
            spectrum("s2", np.array([1, 1, 1, 1], dtype="<f8")),
        )
        fit_context = context("s0", "s1", "s2")
        first = fit_denoising_system(svd, training, fit_context)
        second = fit_denoising_system(svd, training, fit_context)
        self.assertEqual(first.state_sha256, second.state_sha256)
        np.testing.assert_array_equal(first.components, second.components)
        self.assertIsNone(first.mean)
        transformed = transform_fitted_denoiser(
            first, spectrum("held", np.array([2, 3, 2, 3], dtype="<f8"))
        )
        assert transformed.denoised_intensity is not None
        np.testing.assert_allclose(
            transformed.denoised_intensity,
            np.array([2, 3, 2, 3], dtype="<f8"),
            rtol=0,
            atol=2e-14,
        )

    def test_domain_statuses_never_clip_frozen_parameters(self) -> None:
        short = spectrum("short", np.array([1.0, 2.0], dtype="<f8"))
        cases = (
            system("savitzky_golay", window_length=5, polyorder=2),
            system("wavelet", wavelet="db6", threshold_mode="soft", threshold_strategy="universal_mad"),
            system("whittaker_smoothing", **{"lambda": 1.0}),
        )
        for method in cases:
            with self.subTest(family=method.family_id):
                result = run_stateless_denoising_system(method, short)
                self.assertIs(result.status, DenoisingRunStatus.NOT_APPLICABLE)
                self.assertIsNone(result.denoised_intensity)
        oversized = system("pca_reconstruction", n_components=96)
        training = (
            spectrum("o0", np.array([1, 2, 3], dtype="<f8")),
            spectrum("o1", np.array([2, 3, 4], dtype="<f8")),
        )
        with self.assertRaises(DenoisingFitError) as observed:
            fit_denoising_system(oversized, training, context("o0", "o1"))
        self.assertIs(observed.exception.status, DenoisingRunStatus.NOT_APPLICABLE)
        self.assertEqual(oversized.hyperparameters["n_components"], 96)

    def test_outputs_and_fitted_state_are_immutable_and_content_hashed(self) -> None:
        sg = system("savitzky_golay", window_length=5, polyorder=2)
        result = run_stateless_denoising_system(
            sg, spectrum("immutable", np.array([0, 1, 4, 9, 16], dtype="<f8"))
        )
        assert result.denoised_intensity is not None
        self.assertEqual(result.denoised_intensity.dtype, np.dtype("<f8"))
        self.assertFalse(result.denoised_intensity.flags.writeable)
        self.assertEqual(
            result.output_sha256,
            hashlib.sha256(result.denoised_intensity.tobytes()).hexdigest(),
        )
        with self.assertRaises(ValueError):
            result.denoised_intensity[0] = 999

        pca = system("pca_reconstruction", n_components=2)
        training = (
            spectrum("i0", np.array([1, 2, 3], dtype="<f8")),
            spectrum("i1", np.array([2, 3, 4], dtype="<f8")),
            spectrum("i2", np.array([3, 4, 5], dtype="<f8")),
        )
        fitted = fit_denoising_system(pca, training, context("i0", "i1", "i2"))
        self.assertFalse(fitted.components.flags.writeable)
        assert fitted.mean is not None
        self.assertFalse(fitted.mean.flags.writeable)
        self.assertRegex(fitted.state_sha256, r"^[0-9a-f]{64}$")


class DenoisingCanaryTest(unittest.TestCase):
    def test_real_inputs_reproduce_frozen_ledgers_hashes_and_roles(self) -> None:
        config_value = load_denoising_canary_config(CONFIG_PATH, project_root=ROOT)
        inputs = load_denoising_canary_inputs(config_value, project_root=ROOT)
        self.assertEqual(len(inputs.stateless_sources), 4)
        self.assertEqual(len(inputs.fit_sources), 120)
        self.assertEqual(len(inputs.transform_sources), 30)
        self.assertEqual(
            tuple(source.record_id for source in inputs.stateless_sources),
            (
                "raw-0003d1b556e9bc8434a8a7972d5f",
                "raw-002525ef577ac9ed746f80d1bfea",
                "raw-0005cc5ef4a23f3ad7c4dfa6ad90",
                "raw-016f962d269c54b622268d06b2d1",
            ),
        )
        self.assertEqual(inputs.axis_sha256, "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d")
        self.assertEqual(inputs.fit_ledger_sha256, "69a94dc2e5e3be177c1ea34fed8391e8dd8d6f585b4ceada8ad24baf91e58419")
        self.assertEqual(inputs.transform_ledger_sha256, "35721ab2f4319c038c010991655467fc93289c2868934a52ee64a9f2ae348198")
        self.assertEqual(inputs.fit_matrix_sha256, "fb2f7d431a04af042028be034c47aa4a8571087cb795aece3933c227e876e425")
        self.assertFalse(
            {source.record_id for source in inputs.fit_sources}
            & {source.record_id for source in inputs.transform_sources}
        )
        self.assertTrue(all(not source.spectrum.intensity.flags.writeable for source in inputs.fit_sources))
        identity = canary_module._actual_input_identity(inputs)
        self.assertTrue(canary_module._matches_frozen_input_identity(identity, config_value))
        original = inputs.stateless_sources[0]
        changed_intensity = np.array(original.spectrum.intensity, copy=True)
        changed_intensity[0] += 1.0
        changed_spectrum = Spectrum1D(
            original.spectrum.spectrum_id,
            original.spectrum.sample_id,
            original.spectrum.axis_cm1,
            changed_intensity,
        )
        rebound = replace(
            original,
            intensity_sha256=hashlib.sha256(changed_spectrum.intensity.tobytes()).hexdigest(),
            spectrum=changed_spectrum,
        )
        changed_inputs = replace(
            inputs,
            stateless_sources=(rebound, *inputs.stateless_sources[1:]),
        )
        self.assertFalse(
            canary_module._matches_frozen_input_identity(
                canary_module._actual_input_identity(changed_inputs),
                config_value,
            )
        )

    def test_reduced_artifact_is_verifiable_and_byte_deterministic(self) -> None:
        config_value = load_denoising_canary_config(CONFIG_PATH, project_root=ROOT)
        inputs = load_denoising_canary_inputs(config_value, project_root=ROOT)
        reduced = replace(
            inputs,
            stateless_sources=inputs.stateless_sources[:1],
            fit_sources=inputs.fit_sources[:4],
            transform_sources=inputs.transform_sources[:2],
        )
        systems = (
            system("savitzky_golay", window_length=5, polyorder=2),
            system("pca_reconstruction", n_components=2),
        )
        catalog = load_classical_catalog(CATALOG_PATH)
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = build_denoising_canary_artifact(
                reduced,
                systems,
                Path(first_root),
                config=config_value,
                catalog=catalog,
                project_root=ROOT,
            )
            second = build_denoising_canary_artifact(
                reduced,
                systems,
                Path(second_root),
                config=config_value,
                catalog=catalog,
                project_root=ROOT,
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertFalse(first.is_frozen_canary)
            self.assertFalse(first.canary_passed)
            first_files = {
                path.relative_to(first.path).as_posix(): path.read_bytes()
                for path in first.path.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second.path).as_posix(): path.read_bytes()
                for path in second.path.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            verified = verify_denoising_canary_artifact(first.path, project_root=ROOT)
            self.assertEqual(verified.run_id, first.run_id)
            self.assertEqual(verified.fit_receipt_count, 1)
            self.assertEqual(verified.transform_receipt_count, 3)

    def test_verifier_rejects_rebound_stream_and_summary_drift(self) -> None:
        config_value = load_denoising_canary_config(CONFIG_PATH, project_root=ROOT)
        inputs = load_denoising_canary_inputs(config_value, project_root=ROOT)
        reduced = replace(
            inputs,
            stateless_sources=inputs.stateless_sources[:1],
            fit_sources=inputs.fit_sources[:4],
            transform_sources=inputs.transform_sources[:1],
        )
        systems = (
            system("savitzky_golay", window_length=5, polyorder=2),
            system("pca_reconstruction", n_components=2),
        )
        catalog = load_classical_catalog(CATALOG_PATH)
        with tempfile.TemporaryDirectory() as temp_root:
            summary = build_denoising_canary_artifact(
                reduced,
                systems,
                Path(temp_root),
                config=config_value,
                catalog=catalog,
                project_root=ROOT,
            )
            checksum_path = summary.path / "SHA256SUMS"
            checksum_original = checksum_path.read_bytes()
            output_path = summary.path / "denoised_outputs.f64le"
            output_original = output_path.read_bytes()
            changed = bytes([output_original[0] ^ 1]) + output_original[1:]
            output_path.write_bytes(changed)
            checksum_path.write_text(
                checksum_original.decode().replace(
                    hashlib.sha256(output_original).hexdigest(),
                    hashlib.sha256(changed).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DenoisingCanaryError, "output stream"):
                verify_denoising_canary_artifact(summary.path, project_root=ROOT)
            output_path.write_bytes(output_original)
            checksum_path.write_bytes(checksum_original)

            fit_path = summary.path / "fit_receipts.jsonl"
            fit_original = fit_path.read_bytes()
            fit_rows = [json.loads(line) for line in fit_original.splitlines()]
            fit_rows[0]["context_sha256"] = "0" * 64
            fit_changed = b"".join(canonical(row) for row in fit_rows)
            fit_path.write_bytes(fit_changed)
            checksum_path.write_text(
                checksum_original.decode().replace(
                    hashlib.sha256(fit_original).hexdigest(),
                    hashlib.sha256(fit_changed).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DenoisingCanaryError, "fit receipt identity"):
                verify_denoising_canary_artifact(summary.path, project_root=ROOT)
            fit_path.write_bytes(fit_original)
            checksum_path.write_bytes(checksum_original)

            summary_path = summary.path / "summary.json"
            summary_original = summary_path.read_bytes()
            document = json.loads(summary_original)
            document["transform_receipt_count"] += 1
            changed_summary = canonical(document)
            summary_path.write_bytes(changed_summary)
            checksum_path.write_text(
                checksum_original.decode().replace(
                    hashlib.sha256(summary_original).hexdigest(),
                    hashlib.sha256(changed_summary).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DenoisingCanaryError, "summary"):
                verify_denoising_canary_artifact(summary.path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
