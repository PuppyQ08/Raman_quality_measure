from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.semisynth import (  # noqa: E402
    AttemptStatus,
    BackgroundExtractionAttempt,
    BackgroundModelError,
    BackgroundParameterAttempt,
    RruffEqualAxisPair,
    RruffPairAssignment,
    RruffPairRole,
    SemiSyntheticPairError,
    extract_pair_backgrounds,
    fit_background_parameter_model,
    load_phase2_config,
    load_rruff_equal_axis_pairs,
    parameterize_background,
    reconstruct_background,
    sample_background_parameters,
    build_background_fit_artifact,
)


CONFIG_PATH = ROOT / "experiments" / "phase2" / "configs" / "semisynth_v1.json"
PAIR_INDEX = ROOT / "data" / "unified" / "rruff_raman_pairs.jsonl"
RAW_DATASET = ROOT / "data" / "unified" / "rruff_raman_raw"
PROCESSED_DATASET = ROOT / "data" / "unified" / "rruff_raman_processed"


def synthetic_pair(
    *,
    pair_id: str = "synthetic-pair",
    excitation_stratum: str = "green_532",
    axis: np.ndarray | None = None,
    raw: np.ndarray | None = None,
    processed: np.ndarray | None = None,
) -> RruffEqualAxisPair:
    if axis is None:
        axis = np.linspace(100.0, 400.0, 32, dtype="<f8")
    if raw is None:
        raw = (
            0.01 * axis
            + 3.0
            + 20.0 * np.exp(-0.5 * ((axis - 250.0) / 12.0) ** 2)
        )
    if processed is None:
        processed = np.linspace(1.0, 32.0, 32, dtype="<f8")
    return RruffEqualAxisPair(
        pair_id=pair_id,
        group_key="R-SYNTHETIC",
        role=RruffPairRole.EXTRACTION_FIT,
        excitation_stratum=excitation_stratum,
        axis_relation="exact_equal",
        raw_record_id="raw-synthetic",
        processed_record_id="processed-synthetic",
        raw_slice=(0, axis.size),
        processed_slice=(0, axis.size),
        raw_source_member_sha256="1" * 64,
        processed_source_member_sha256="2" * 64,
        processed_template_qualifier=(
            "algorithmically_processed_signal_template_not_physical_clean_gt"
        ),
        axis_cm1=np.asarray(axis, dtype="<f8"),
        raw_intensity=np.asarray(raw, dtype="<f8"),
        processed_intensity=np.asarray(processed, dtype="<f8"),
    )


def valid_parameter_attempt(index: int, *, nrmse: float) -> BackgroundParameterAttempt:
    value = float(index)
    vector = np.array(
        [
            1.0 + 0.08 * np.sin(0.11 * value),
            0.25 * np.cos(0.07 * value),
            -0.15 * np.sin(0.13 * value + 0.2),
            0.10 * np.cos(0.17 * value + 0.3),
            0.07 * np.sin(0.19 * value + 0.5),
            -0.05 * np.cos(0.23 * value + 0.7),
            0.03 * np.sin(0.29 * value + 0.9),
            -1.0 + 0.2 * np.sin(0.05 * value),
        ],
        dtype="<f8",
    )
    pair_id = f"pair-{index:03d}"
    return BackgroundParameterAttempt(
        pair_id=pair_id,
        extractor_id="airpls",
        excitation_stratum="green_532",
        status=AttemptStatus.VALID,
        vector=vector,
        reconstruction_nrmse=nrmse,
        input_sha256=hashlib.sha256(pair_id.encode("utf-8")).hexdigest(),
        error_code=None,
        error_message=None,
    )


def failed_parameter_attempt(index: int) -> BackgroundParameterAttempt:
    pair_id = f"pair-{index:03d}"
    return BackgroundParameterAttempt(
        pair_id=pair_id,
        extractor_id="airpls",
        excitation_stratum="green_532",
        status=AttemptStatus.FAILED,
        vector=None,
        reconstruction_nrmse=None,
        input_sha256=hashlib.sha256(pair_id.encode("utf-8")).hexdigest(),
        error_code="synthetic_failure",
        error_message="synthetic invalid parameter",
    )


class RruffEqualAxisPairLoaderTest(unittest.TestCase):
    def test_real_loader_preserves_all_three_exact_slice_relations(self) -> None:
        assignments = (
            RruffPairAssignment(
                pair_id="a430abd8c9e167be77ce4f69ff524700a3cd1f3eeb3e49b132567067485a5270",
                group_key="R070037",
                role=RruffPairRole.EXTRACTION_FIT,
                excitation_stratum="green_532",
                raw_record_id="raw-547fc12f0b31f1f5a6205ac44872",
                processed_record_id="processed-db2de3a53da39189f1a8dfee39f1",
                axis_relation="processed_exact_contiguous_subset_of_raw",
            ),
            RruffPairAssignment(
                pair_id="f63e12da56d5fc2b753ed966b2fde269e3c278db57959e1681b95111f92c65a6",
                group_key="R250018",
                role=RruffPairRole.EXTRACTION_FIT,
                excitation_stratum="green_532",
                raw_record_id="raw-9fbe213ab52f2f6ea1d5cf0ab934",
                processed_record_id="processed-a4d621c87e7272733ddfb67bc7dd",
                axis_relation="exact_equal",
            ),
            RruffPairAssignment(
                pair_id="10323b09f79303fb0b5379adba1d2cf02fe9ae59d98d03290441e6d06d2d30a3",
                group_key="R060546",
                role=RruffPairRole.EXTRACTION_FIT,
                excitation_stratum="green_532",
                raw_record_id="raw-5bb936468e868f92eb574a91af6f",
                processed_record_id="processed-a741cde292e54f8aad272770f52e",
                axis_relation="raw_exact_contiguous_subset_of_processed",
            ),
        )
        loaded = load_rruff_equal_axis_pairs(
            PAIR_INDEX,
            RAW_DATASET,
            PROCESSED_DATASET,
            assignments,
            config=load_phase2_config(CONFIG_PATH),
            verify_checksums=False,
        )
        by_relation = {pair.axis_relation: pair for pair in loaded}
        expected = {
            "processed_exact_contiguous_subset_of_raw": (
                3359,
                (35, 3394),
                (0, 3359),
                "b2eea81dc04e7381476376103cea342b8b1772fc57a2e527f15f3a07e667b1d7",
                "4ab551e0ca1751b44986b13a9e6d43923da89fa9ddf1a742a0e848bb9f5ab36b",
                "4ab551e0ca1751b44986b13a9e6d43923da89fa9ddf1a742a0e848bb9f5ab36b",
            ),
            "exact_equal": (
                2806,
                (0, 2806),
                (0, 2806),
                "707ae2a8ccbebda9940c7dd201494ed652e1db4b71666cf9a482b33252dba078",
                "99a3e83879300a30adcf6875f4261cf8c89720568c11253af3194c1072186bcd",
                "5e416ccb165b25a15ee172ae4058eec31bad7b392fa0ee19f27433d2b4c8b896",
            ),
            "raw_exact_contiguous_subset_of_processed": (
                3344,
                (0, 3344),
                (50, 3394),
                "cb90711b85d37f882d49f363ce5353a3e8a9d06b2014e8810ac72b87918e2999",
                "7394a01116e9142325745e2d92c3a3c3c48bf6fda13c0fcdb694346272a79c6d",
                "7394a01116e9142325745e2d92c3a3c3c48bf6fda13c0fcdb694346272a79c6d",
            ),
        }
        self.assertEqual(set(by_relation), set(expected))
        for relation, oracle in expected.items():
            with self.subTest(relation=relation):
                pair = by_relation[relation]
                length, raw_slice, processed_slice, axis_sha, raw_sha, processed_sha = oracle
                self.assertEqual(pair.axis_cm1.size, length)
                self.assertEqual(pair.raw_slice, raw_slice)
                self.assertEqual(pair.processed_slice, processed_slice)
                self.assertEqual(hashlib.sha256(pair.axis_cm1.tobytes()).hexdigest(), axis_sha)
                self.assertEqual(hashlib.sha256(pair.raw_intensity.tobytes()).hexdigest(), raw_sha)
                self.assertEqual(
                    hashlib.sha256(pair.processed_intensity.tobytes()).hexdigest(),
                    processed_sha,
                )
                self.assertFalse(pair.axis_cm1.flags.writeable)
                self.assertFalse(pair.raw_intensity.flags.writeable)
                self.assertFalse(pair.processed_intensity.flags.writeable)
                self.assertEqual(
                    pair.processed_template_qualifier,
                    "algorithmically_processed_signal_template_not_physical_clean_gt",
                )

    def test_pair_contract_rejects_nonincreasing_axis(self) -> None:
        axis = np.linspace(100.0, 400.0, 32, dtype="<f8")
        axis[10] = axis[9]
        with self.assertRaisesRegex(SemiSyntheticPairError, "axis_cm1.increasing"):
            synthetic_pair(axis=axis)


class FrozenExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_phase2_config(CONFIG_PATH)
        self.pair = synthetic_pair()

    def test_three_frozen_extractors_have_literal_output_oracles(self) -> None:
        attempts = extract_pair_backgrounds(self.pair, self.config)
        self.assertEqual(
            tuple(attempt.extractor_id for attempt in attempts),
            ("airpls", "arpls", "mor"),
        )
        expected_sha = {
            "airpls": "262dae5c17a39073f551c6b56855c51e40fe1de09f549ad464ecd82c95265d42",
            "arpls": "f44e288fa6e690159f0b0ebb3d92a1f97abf62ffd23676f16ddd7cc3857c93e8",
            "mor": "60821b9b3ba1f5377fecb2a9d10ac721df10fa927996b5d213bbd533642902db",
        }
        for attempt in attempts:
            with self.subTest(extractor=attempt.extractor_id):
                self.assertIs(attempt.status, AttemptStatus.VALID)
                assert attempt.baseline_raw is not None
                self.assertEqual(
                    hashlib.sha256(attempt.baseline_raw.tobytes()).hexdigest(),
                    expected_sha[attempt.extractor_id],
                )
                self.assertFalse(attempt.baseline_raw.flags.writeable)
        self.assertEqual(attempts[0].diagnostics["iteration_count"], 3)
        self.assertEqual(attempts[1].diagnostics["iteration_count"], 9)
        self.assertEqual(attempts[2].diagnostics["half_window"], 11)

    def test_one_extractor_failure_does_not_block_other_ids(self) -> None:
        with patch(
            "rpe.semisynth.background_model.Baseline.airpls",
            side_effect=RuntimeError("injected airPLS failure"),
        ):
            attempts = extract_pair_backgrounds(self.pair, self.config)
        self.assertIs(attempts[0].status, AttemptStatus.FAILED)
        self.assertEqual(attempts[0].error_code, "RuntimeError")
        self.assertIs(attempts[1].status, AttemptStatus.VALID)
        self.assertIs(attempts[2].status, AttemptStatus.VALID)

    def test_extractors_reject_non_fit_pair_roles(self) -> None:
        for role in (RruffPairRole.SIGNAL_TEMPLATE, RruffPairRole.REAL_HOLDOUT):
            with self.subTest(role=role), self.assertRaisesRegex(
                BackgroundModelError, "pair.role"
            ):
                extract_pair_backgrounds(replace(self.pair, role=role), self.config)


class BackgroundParameterModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_phase2_config(CONFIG_PATH)

    def test_degree_six_parameterization_has_hand_derived_oracle(self) -> None:
        axis = np.linspace(100.0, 400.0, 32, dtype="<f8")
        normalized = np.linspace(-1.0, 1.0, 32, dtype="<f8")
        source_coefficients = np.array(
            [2.0, 0.5, -0.25, 0.125, 0.0, 0.0, 0.0], dtype="<f8"
        )
        q = np.polynomial.legendre.legval(normalized, source_coefficients)
        q -= q.min()
        pair = synthetic_pair(
            axis=axis,
            raw=q + 5.0,
            processed=np.linspace(1.0, 32.0, 32, dtype="<f8"),
        )
        extraction = BackgroundExtractionAttempt(
            pair_id=pair.pair_id,
            extractor_id="airpls",
            excitation_stratum="green_532",
            status=AttemptStatus.VALID,
            baseline_raw=np.asarray(q + 5.0, dtype="<f8"),
            diagnostics={"oracle": True},
            error_code=None,
            error_message=None,
        )
        attempt = parameterize_background(pair, extraction, self.config)
        self.assertIs(attempt.status, AttemptStatus.VALID)
        np.testing.assert_allclose(
            attempt.vector,
            np.array(
                [
                    0.9428114284734528,
                    0.5387493876991158,
                    -0.2693746938495579,
                    0.13468734692477896,
                    0.0,
                    0.0,
                    0.0,
                    -3.0142100875407887,
                ],
                dtype="<f8",
            ),
            rtol=0.0,
            atol=2e-15,
        )
        assert attempt.reconstruction_nrmse is not None
        self.assertLess(attempt.reconstruction_nrmse, 1e-14)
        assert attempt.vector is not None
        self.assertFalse(attempt.vector.flags.writeable)

    def test_ledoit_wolf_receipt_and_fit_gates_are_frozen(self) -> None:
        attempts = tuple(
            valid_parameter_attempt(index, nrmse=0.01 + 0.00001 * index)
            for index in range(200)
        )
        receipt = fit_background_parameter_model(
            attempts,
            extractor_id="airpls",
            excitation_stratum="green_532",
            config=self.config,
        )
        self.assertEqual(receipt.input_count, 200)
        self.assertEqual(receipt.valid_count, 200)
        self.assertEqual(receipt.invalid_count, 0)
        self.assertEqual(receipt.valid_fraction, 1.0)
        self.assertEqual(receipt.gate_failures, ())
        self.assertEqual(receipt.median_reconstruction_nrmse, 0.010995)
        self.assertEqual(receipt.p95_reconstruction_nrmse, 0.011890500000000002)
        np.testing.assert_allclose(
            receipt.mean,
            np.array(
                [
                    1.007267020416129,
                    0.018221732536002903,
                    -0.0026124224183975064,
                    0.0003607683293588439,
                    0.00029161394074457515,
                    0.000042502242059907804,
                    0.0006861330485391882,
                    -0.9629542219803663,
                ],
                dtype="<f8",
            ),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertAlmostEqual(receipt.shrinkage, 0.020711582226894926)
        self.assertEqual(
            receipt.log_amplitude_bounds,
            (-1.1998591882636642, -0.8000430388530121),
        )
        self.assertEqual(
            receipt.mahalanobis_squared_max, 21.95495499065953
        )
        self.assertRegex(receipt.model_id, r"^[0-9a-f]{64}$")
        reordered = fit_background_parameter_model(
            tuple(reversed(attempts)),
            extractor_id="airpls",
            excitation_stratum="green_532",
            config=self.config,
        )
        self.assertEqual(reordered.model_id, receipt.model_id)

        invalid_attempts = tuple(
            valid_parameter_attempt(index, nrmse=float(value))
            for index, value in enumerate(np.linspace(0.11, 0.31, 179))
        ) + tuple(failed_parameter_attempt(index) for index in range(179, 200))
        failed = fit_background_parameter_model(
            invalid_attempts,
            extractor_id="airpls",
            excitation_stratum="green_532",
            config=self.config,
        )
        self.assertEqual(failed.valid_fraction, 0.895)
        self.assertEqual(
            failed.gate_failures,
            (
                "valid_fraction",
                "valid_count",
                "median_reconstruction_nrmse",
                "p95_reconstruction_nrmse",
            ),
        )

    def test_sampling_rejection_and_reconstruction_are_deterministic(self) -> None:
        attempts = tuple(
            valid_parameter_attempt(index, nrmse=0.01 + 0.00001 * index)
            for index in range(200)
        )
        receipt = fit_background_parameter_model(
            attempts,
            extractor_id="airpls",
            excitation_stratum="green_532",
            config=self.config,
        )
        oracle_receipt = replace(
            receipt,
            model_id="a" * 64,
            mean=np.zeros(8, dtype="<f8"),
            covariance=np.eye(8, dtype="<f8") * 0.01,
            precision=np.eye(8, dtype="<f8") * 100.0,
            log_amplitude_bounds=(-0.05, 0.05),
            mahalanobis_squared_max=21.95495499065953,
            gate_failures=(),
        )
        first = sample_background_parameters(oracle_receipt, 1, self.config)
        second = sample_background_parameters(oracle_receipt, 1, self.config)
        self.assertEqual(
            first.seed_sha256,
            "151c139698eb3393a6fc69c9e501b058a0e73517771453765d60ae69e4135988",
        )
        self.assertEqual(first.rejected_attempts, 2)
        self.assertEqual(first, second)
        np.testing.assert_allclose(
            first.vector,
            np.array(
                [
                    -0.022660225171930785,
                    0.1134034945783984,
                    -0.16120158028940523,
                    0.06687354828433506,
                    0.02746103385672896,
                    0.07039140346225498,
                    0.062064330293688366,
                    0.010234518159147605,
                ],
                dtype="<f8",
            ),
            rtol=0.0,
            atol=0.0,
        )

        reconstruction_sample = replace(
            first,
            vector=np.array(
                [1.0, 0.5, -0.25, 0.125, 0.0, 0.0, 0.0, -0.7],
                dtype="<f8",
            ),
        )
        axis = np.linspace(100.0, 400.0, 32, dtype="<f8")
        template = np.linspace(1.0, 32.0, 32, dtype="<f8")
        background = reconstruct_background(
            reconstruction_sample, axis, template
        )
        self.assertGreaterEqual(float(background.min()), 0.0)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(background * background))),
            float(np.exp(-0.7) * np.sqrt(np.mean(template * template))),
        )
        self.assertFalse(background.flags.writeable)

        drifted_config = replace(self.config, sha256="b" * 64)
        with self.assertRaisesRegex(BackgroundModelError, "config identity"):
            sample_background_parameters(oracle_receipt, 1, drifted_config)


class BackgroundFitArtifactTest(unittest.TestCase):
    def test_four_stratum_fixture_writes_deterministic_failed_artifact(self) -> None:
        config = load_phase2_config(CONFIG_PATH)
        pairs = tuple(
            synthetic_pair(pair_id=f"fixture-{index}", excitation_stratum=stratum)
            for index, stratum in enumerate(
                ("green_514", "green_532", "nir_780", "nir_785")
            )
        )
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_background_fit_artifact(
                pairs, Path(first_dir), config=config, project_root=ROOT
            )
            second = build_background_fit_artifact(
                pairs, Path(second_dir), config=config, project_root=ROOT
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertFalse(first.gate_passed)
            self.assertEqual(first.pair_count, 4)
            self.assertEqual(first.attempt_count, 12)
            self.assertEqual(first.model_count, 12)
            self.assertFalse((first.path / "complete.json").exists())
            self.assertTrue((first.path / "failed.json").is_file())
            expected_names = {
                "manifest.json",
                "parameter_attempts.jsonl",
                "model_receipts.json",
                "gate.json",
                "SHA256SUMS",
                "failed.json",
            }
            self.assertEqual(
                {path.name for path in first.path.iterdir()}, expected_names
            )
            for name in expected_names:
                self.assertEqual(
                    (first.path / name).read_bytes(),
                    (second.path / name).read_bytes(),
                )
            checksum_lines = (first.path / "SHA256SUMS").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(checksum_lines), 4)
            for line in checksum_lines:
                digest, name = line.split("  ")
                self.assertEqual(
                    hashlib.sha256((first.path / name).read_bytes()).hexdigest(),
                    digest,
                )


if __name__ == "__main__":
    unittest.main()
