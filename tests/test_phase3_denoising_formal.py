from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.methods import TaskLine, load_classical_catalog  # noqa: E402
from rpe.methods.classical import DenoisingFitContext, DenoisingRunStatus  # noqa: E402
import rpe.runner.phase3_denoising_formal as formal_module  # noqa: E402
from rpe.runner import (  # noqa: E402
    DenoisingFormalCoveragePolicy,
    DenoisingFormalError,
    DenoisingFormalFitReceipt,
    DenoisingFormalSummary,
    DenoisingFormalTransformReceipt,
    build_phase3_denoising_formal,
    evaluate_denoising_coverage,
    load_phase3_denoising_formal_config,
    load_phase3_denoising_formal_inputs,
    verify_phase3_denoising_formal,
)
import tools.run_phase3_denoising_formal as cli_module  # noqa: E402


CONFIG = ROOT / "experiments" / "phase3" / "configs" / "denoising_v1_formal_coverage.json"
CATALOG = ROOT / "experiments" / "phase3" / "configs" / "classical_system_catalog_v1.json"
CLI = ROOT / "tools" / "run_phase3_denoising_formal.py"


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return sha256_bytes(np.asarray(array, dtype="<f8").tobytes())


def report_record_ids_sha256(record_ids: tuple[str, ...]) -> str:
    return sha256_bytes(("\n".join(record_ids) + "\n").encode("utf-8"))


def canonical_record_ledger_sha256(record_ids: tuple[str, ...]) -> str:
    return sha256_bytes(b"".join(canonical({"record_id": record_id}) for record_id in record_ids))


def matrix_sha256(spectra: tuple[Spectrum1D, ...]) -> str:
    matrix = np.ascontiguousarray([spectrum.intensity for spectrum in spectra], dtype="<f8")
    return array_sha(matrix)


def spectrum(record_id: str, values: np.ndarray, *, axis: np.ndarray | None = None) -> Spectrum1D:
    intensity = np.asarray(values, dtype="<f8")
    actual_axis = (
        np.linspace(100.0, 100.0 + float(intensity.size - 1), intensity.size, dtype="<f8")
        if axis is None
        else np.asarray(axis, dtype="<f8")
    )
    return Spectrum1D(
        spectrum_id=f"fixture::{record_id}",
        sample_id=record_id,
        axis_cm1=actual_axis,
        intensity=intensity,
    )


def stateless_receipt(
    *,
    system,
    record_id: str,
    source_order: int,
    status: DenoisingRunStatus,
    point_count: int = 4,
) -> DenoisingFormalTransformReceipt:
    axis = np.linspace(100.0, 103.0, point_count, dtype="<f8")
    values = np.linspace(0.0, 1.0, point_count, dtype="<f8") + source_order
    output_sha = array_sha(values) if status in {DenoisingRunStatus.COMPLETE, DenoisingRunStatus.COMPLETE_WITH_WARNING} else None
    return DenoisingFormalTransformReceipt(
        cohort="stateless",
        source_role="stateless",
        source_order=source_order,
        record_id=record_id,
        system_id=system.system_id,
        family_id=system.family_id,
        method_id=system.method_id,
        status=status,
        point_count=point_count,
        axis_sha256=array_sha(axis),
        output_sha256=output_sha,
        output_offset_bytes=(source_order * point_count * 8 if output_sha is not None else None),
        output_byte_count=(point_count * 8 if output_sha is not None else None),
        fitted_state_sha256=None,
        diagnostics={},
        warnings=(),
        error_code=None if output_sha is not None else "failed_domain",
        error_message=None if output_sha is not None else "fixture failure",
    )


def fitted_receipt(
    *,
    system,
    status: DenoisingRunStatus,
    train_record_ids: tuple[str, ...],
    fit_state_offset_bytes: int = 0,
) -> DenoisingFormalFitReceipt:
    context = DenoisingFitContext(
        split_id="d1_seed0_train",
        representation_id="bacteria_id_reference_increasing_float64_v1",
        record_ids=train_record_ids,
    )
    components = np.eye(2, 4, dtype="<f8")
    fit_state_sha = array_sha(components) if status in {DenoisingRunStatus.COMPLETE, DenoisingRunStatus.COMPLETE_WITH_WARNING} else None
    return DenoisingFormalFitReceipt(
        system_id=system.system_id,
        family_id=system.family_id,
        method_id=system.method_id,
        status=status,
        split_id=context.split_id,
        representation_id=context.representation_id,
        training_record_ledger_sha256=sha256_bytes(
            b"".join(canonical({"record_id": record_id}) for record_id in train_record_ids)
        ),
        training_matrix_sha256="a" * 64,
        axis_sha256="b" * 64,
        context_sha256=context.context_sha256,
        fit_state_sha256=fit_state_sha,
        fit_state_offset_bytes=(fit_state_offset_bytes if fit_state_sha is not None else None),
        fit_state_byte_count=(components.nbytes if fit_state_sha is not None else None),
        warnings=(),
        diagnostics={},
        error_code=None if fit_state_sha is not None else "failed_fit",
        error_message=None if fit_state_sha is not None else "fixture fit failure",
    )


def fitted_transform_receipt(
    *,
    system,
    record_id: str,
    source_order: int,
    status: DenoisingRunStatus,
    source_role: str,
    fitted_state_sha256: str | None,
    point_count: int = 4,
) -> DenoisingFormalTransformReceipt:
    axis = np.linspace(500.0, 503.0, point_count, dtype="<f8")
    values = np.linspace(2.0, 3.0, point_count, dtype="<f8") + source_order
    output_sha = array_sha(values) if status in {DenoisingRunStatus.COMPLETE, DenoisingRunStatus.COMPLETE_WITH_WARNING} else None
    return DenoisingFormalTransformReceipt(
        cohort="fitted",
        source_role=source_role,
        source_order=source_order,
        record_id=record_id,
        system_id=system.system_id,
        family_id=system.family_id,
        method_id=system.method_id,
        status=status,
        point_count=point_count,
        axis_sha256=array_sha(axis),
        output_sha256=output_sha,
        output_offset_bytes=(source_order * point_count * 8 if output_sha is not None else None),
        output_byte_count=(point_count * 8 if output_sha is not None else None),
        fitted_state_sha256=fitted_state_sha256,
        diagnostics=(
            {"fitted_state_sha256": fitted_state_sha256}
            if fitted_state_sha256 is not None
            else {}
        ),
        warnings=(),
        error_code=None if output_sha is not None else "failed_runtime",
        error_message=None if output_sha is not None else "fixture transform failure",
    )


def formal_source(
    record_id: str,
    *,
    class_label: int,
    source_order: int,
    values: np.ndarray,
    axis: np.ndarray | None = None,
) -> formal_module.DenoisingFormalSource:
    current = spectrum(record_id, values, axis=axis)
    return formal_module.DenoisingFormalSource(
        cohort_id="fixture",
        record_id=record_id,
        class_label=class_label,
        source_order=source_order,
        point_count=current.intensity.size,
        axis_sha256=array_sha(current.axis_cm1),
        intensity_sha256=array_sha(current.intensity),
        spectrum=current,
        sample_id=record_id,
        mineral_name=f"fixture-{class_label}",
    )


class Phase3DenoisingFormalConfigTest(unittest.TestCase):
    def test_config_is_canonical_and_binds_frozen_formal_contract(self) -> None:
        raw = CONFIG.read_bytes()
        document = json.loads(raw)
        self.assertEqual(raw, canonical(document))
        config = load_phase3_denoising_formal_config(CONFIG)
        self.assertEqual(config.schema_version, "phase3-denoising-formal-coverage-v1")
        self.assertEqual(config.catalog.catalog_id, "519d12da585aa87b00d121799b012a3d7b0686cc85455db05f4fb4fc2e0d7b7e")
        self.assertEqual(config.authorities.parent_plan_sha256, "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5")
        self.assertEqual(config.authorities.step5_design_sha256, "c4a2280d8545bd3f5270d11f1af679d4d0504bca00cf4cc8d2dd11312cbe0ee6")
        self.assertEqual(config.authorities.step6_denoising_report_sha256, "c1e7caa85abe27e240d808d5b6c98efc6c5640b99a57b22e1a420cb2a45cc514")
        self.assertEqual(config.authorities.step7_peak_report_sha256, "75616c89017771a5c57ee1c5a3ffaddc142c1209a0b6c8041f7dde29bf1cc484")
        self.assertEqual(config.authorities.step8_protocol_sha256, "b9a8e170077b6224c305d053a80b6ac3fde20e264be1cdef6fa0d24eb94bcc2e")
        self.assertEqual(config.expected.system_count, 60)
        self.assertEqual(config.expected.stateless_system_count, 36)
        self.assertEqual(config.expected.fitted_system_count, 24)
        self.assertEqual(config.expected.stateless_source_count, 10000)
        self.assertEqual(config.expected.fit_source_count, 62700)
        self.assertEqual(config.expected.validation_source_count, 300)
        self.assertEqual(config.expected.test_source_count, 3000)
        self.assertEqual(config.expected.fit_receipt_count, 24)
        self.assertEqual(config.expected.stateless_transform_receipt_count, 360000)
        self.assertEqual(config.expected.fitted_transform_receipt_count, 79200)
        self.assertEqual(config.expected.total_transform_receipt_count, 439200)
        self.assertEqual(config.operational.worker_processes, 8)
        self.assertEqual(config.operational.blas_threads_per_process, 1)
        self.assertEqual(config.claim_boundary, "formal_coverage_only_not_denoising_quality_or_phase5")
        self.assertEqual(config.phase5_power_status, "not_powered_for_phase5")
        self.assertEqual(config.rruff.source_ledger_sha256, "818bfddd9eb94cced9d486f4232a3e8b7e8f9505e8df6218e143b354b89a663f")
        self.assertEqual(config.bacteria.dataset_sha256sums_sha256, "6d6d1399ac0e5197a9e51a1a0cf32edf51c924ce8d7c12aeb9d58f9af93c7f3e")
        self.assertEqual(config.bacteria.axis_sha256, "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d")
        self.assertEqual(config.bacteria.d1_config_sha256, "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046")
        self.assertEqual(config.bacteria.d1_seed0_result_sha256, "3f48553f755851bdcd4b9fe387e452955b2e60298f14838d3fb010feec798974")
        self.assertEqual(config.bacteria.train_record_ids_sha256, "ed3ea7448b5737b4122f76cb4b60f95487780c5a68e03f64d48a7b36d39e5433")
        self.assertEqual(config.bacteria.validation_record_ids_sha256, "aa2cf7c7950b877493114d0492410474e0d21998e1081363d3499969092ab7a5")
        self.assertEqual(config.bacteria.test_record_ids_sha256, "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd")
        self.assertEqual(config.bacteria.train_record_ledger_sha256, "c70a80f100338da3267693b71944c7a262a7f32df0db38b7ed14ace38bb1760a")
        self.assertEqual(config.bacteria.validation_record_ledger_sha256, "3604165935b49233add09e59b63fb26ffe5cbdf1d4270a888b4b9cd79a1a1fa0")
        self.assertEqual(config.bacteria.test_record_ledger_sha256, "4132b5b9991e66cf6b83c707871d394492730d83453f0b9cacbdc3882d1b749c")
        self.assertEqual(config.bacteria.train_matrix_sha256, "6ccfa3a29600ebc0449057bd977d6b8a6679c9b5c5a28ee4218e12334600c7d9")
        self.assertEqual(config.bacteria.validation_matrix_sha256, "852a65c3922fecdbeac1dc5b005858d7872f316044333db0ff9be3b85686a621")
        self.assertEqual(config.bacteria.test_matrix_sha256, "971706d608dd23f49f2564676d7abe05108b4d97ee465a13728bb50e9c72dfa1")
        self.assertEqual(
            hashlib.sha256((ROOT / config.catalog.path).read_bytes()).hexdigest(),
            config.catalog.sha256,
        )

    def test_real_input_loader_reconstructs_exact_frozen_ledgers(self) -> None:
        config = load_phase3_denoising_formal_config(CONFIG)
        inputs = load_phase3_denoising_formal_inputs(config, project_root=ROOT)
        self.assertEqual(len(inputs.stateless_sources), 10000)
        self.assertEqual(len(inputs.fit_sources), 62700)
        self.assertEqual(len(inputs.validation_sources), 300)
        self.assertEqual(len(inputs.test_sources), 3000)
        self.assertEqual(inputs.axis_sha256, "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d")
        self.assertEqual(inputs.train_record_ids_sha256, "ed3ea7448b5737b4122f76cb4b60f95487780c5a68e03f64d48a7b36d39e5433")
        self.assertEqual(inputs.validation_record_ids_sha256, "aa2cf7c7950b877493114d0492410474e0d21998e1081363d3499969092ab7a5")
        self.assertEqual(inputs.test_record_ids_sha256, "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd")
        self.assertEqual(inputs.train_record_ledger_sha256, "c70a80f100338da3267693b71944c7a262a7f32df0db38b7ed14ace38bb1760a")
        self.assertEqual(inputs.validation_record_ledger_sha256, "3604165935b49233add09e59b63fb26ffe5cbdf1d4270a888b4b9cd79a1a1fa0")
        self.assertEqual(inputs.test_record_ledger_sha256, "4132b5b9991e66cf6b83c707871d394492730d83453f0b9cacbdc3882d1b749c")
        self.assertEqual(inputs.train_matrix_sha256, "6ccfa3a29600ebc0449057bd977d6b8a6679c9b5c5a28ee4218e12334600c7d9")
        self.assertEqual(inputs.validation_matrix_sha256, "852a65c3922fecdbeac1dc5b005858d7872f316044333db0ff9be3b85686a621")
        self.assertEqual(inputs.test_matrix_sha256, "971706d608dd23f49f2564676d7abe05108b4d97ee465a13728bb50e9c72dfa1")
        self.assertFalse({source.record_id for source in inputs.fit_sources} & {source.record_id for source in inputs.validation_sources})
        self.assertFalse({source.record_id for source in inputs.fit_sources} & {source.record_id for source in inputs.test_sources})
        self.assertFalse({source.record_id for source in inputs.validation_sources} & {source.record_id for source in inputs.test_sources})
        self.assertTrue(all(not source.spectrum.intensity.flags.writeable for source in inputs.fit_sources[:10]))


class Phase3DenoisingFormalCoverageTest(unittest.TestCase):
    def test_separate_cohort_denominators_and_strict_promotion(self) -> None:
        catalog = load_classical_catalog(CATALOG)
        stateless_systems = [
            system
            for system in catalog.systems
            if system.task_line is TaskLine.DENOISING and system.family_id == "savitzky_golay"
        ][:2]
        fitted_systems = [
            system
            for system in catalog.systems
            if system.task_line is TaskLine.DENOISING and system.family_id == "pca_reconstruction"
        ][:2]
        self.assertEqual(len(stateless_systems), 2)
        self.assertEqual(len(fitted_systems), 2)
        policy = DenoisingFormalCoveragePolicy(
            require_zero_unsuccessful=True,
            descriptive_minimum_successful_fraction=0.99,
            descriptive_minimum_common_record_fraction=0.95,
        )
        fit_success = fitted_receipt(
            system=fitted_systems[0],
            status=DenoisingRunStatus.COMPLETE,
            train_record_ids=("train-0", "train-1", "train-2", "train-3"),
        )
        fit_failure = fitted_receipt(
            system=fitted_systems[1],
            status=DenoisingRunStatus.FAILED_FIT,
            train_record_ids=("train-0", "train-1", "train-2", "train-3"),
        )
        transform_rows = (
            stateless_receipt(system=stateless_systems[0], record_id="s0", source_order=0, status=DenoisingRunStatus.COMPLETE),
            stateless_receipt(system=stateless_systems[0], record_id="s1", source_order=1, status=DenoisingRunStatus.COMPLETE_WITH_WARNING),
            stateless_receipt(system=stateless_systems[1], record_id="s0", source_order=0, status=DenoisingRunStatus.COMPLETE),
            stateless_receipt(system=stateless_systems[1], record_id="s1", source_order=1, status=DenoisingRunStatus.FAILED_DOMAIN),
            fitted_transform_receipt(
                system=fitted_systems[0],
                record_id="v0",
                source_order=0,
                source_role="validation",
                status=DenoisingRunStatus.COMPLETE,
                fitted_state_sha256=fit_success.fit_state_sha256,
            ),
            fitted_transform_receipt(
                system=fitted_systems[0],
                record_id="t0",
                source_order=1,
                source_role="test",
                status=DenoisingRunStatus.COMPLETE,
                fitted_state_sha256=fit_success.fit_state_sha256,
            ),
            fitted_transform_receipt(
                system=fitted_systems[1],
                record_id="v0",
                source_order=0,
                source_role="validation",
                status=DenoisingRunStatus.FAILED_RUNTIME,
                fitted_state_sha256=None,
            ),
            fitted_transform_receipt(
                system=fitted_systems[1],
                record_id="t0",
                source_order=1,
                source_role="test",
                status=DenoisingRunStatus.FAILED_RUNTIME,
                fitted_state_sha256=None,
            ),
        )
        result = evaluate_denoising_coverage(
            fit_receipts=(fit_success, fit_failure),
            transform_receipts=transform_rows,
            stateless_record_ids=("s0", "s1"),
            validation_record_ids=("v0",),
            test_record_ids=("t0",),
            systems=tuple(stateless_systems + fitted_systems),
            policy=policy,
        )
        self.assertEqual(result.stateless_promoted_system_ids, (stateless_systems[0].system_id,))
        self.assertEqual(result.fitted_promoted_system_ids, (fitted_systems[0].system_id,))
        self.assertEqual(result.promoted_system_ids, (fitted_systems[0].system_id, stateless_systems[0].system_id))
        self.assertEqual(result.stateless_common_successful_record_count, 2)
        self.assertEqual(result.fitted_validation_common_successful_record_count, 1)
        self.assertEqual(result.fitted_test_common_successful_record_count, 1)
        self.assertEqual(result.fitted_union_common_successful_record_count, 2)
        blocked = next(summary for summary in result.system_summaries if summary.system_id == stateless_systems[1].system_id)
        self.assertFalse(blocked.coverage_promoted)
        self.assertEqual(blocked.transform_status_counts["failed_domain"], 1)
        fit_blocked = next(summary for summary in result.system_summaries if summary.system_id == fitted_systems[1].system_id)
        self.assertFalse(fit_blocked.coverage_promoted)
        self.assertEqual(fit_blocked.fit_status_counts["failed_fit"], 1)


class Phase3DenoisingFormalArtifactTest(unittest.TestCase):
    def test_small_artifact_is_streamed_deterministic_and_independently_verifiable(self) -> None:
        config = load_phase3_denoising_formal_config(CONFIG)
        catalog = load_classical_catalog(CATALOG)
        systems = (
            next(
                system for system in catalog.systems
                if system.task_line is TaskLine.DENOISING and system.family_id == "savitzky_golay"
            ),
            next(
                system for system in catalog.systems
                if system.task_line is TaskLine.DENOISING and system.family_id == "pca_reconstruction"
                and system.hyperparameters["n_components"] == 2
            ),
        )
        axis = np.linspace(400.0, 403.0, 4, dtype="<f8")
        stateless_sources = (
            formal_source("stateless-0", class_label=0, source_order=0, values=np.array([0.0, 1.0, 2.0, 3.0], dtype="<f8"), axis=axis),
            formal_source("stateless-1", class_label=1, source_order=1, values=np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8"), axis=axis),
        )
        fit_sources = (
            formal_source("train-0", class_label=0, source_order=0, values=np.array([0.0, 1.0, 0.0, 1.0], dtype="<f8"), axis=axis),
            formal_source("train-1", class_label=0, source_order=1, values=np.array([0.1, 1.1, 0.1, 1.1], dtype="<f8"), axis=axis),
            formal_source("train-2", class_label=1, source_order=2, values=np.array([2.0, 3.0, 2.0, 3.0], dtype="<f8"), axis=axis),
            formal_source("train-3", class_label=1, source_order=3, values=np.array([2.1, 3.1, 2.1, 3.1], dtype="<f8"), axis=axis),
        )
        validation_sources = (
            formal_source("validation-0", class_label=0, source_order=0, values=np.array([0.2, 1.2, 0.2, 1.2], dtype="<f8"), axis=axis),
        )
        test_sources = (
            formal_source("test-0", class_label=1, source_order=0, values=np.array([2.2, 3.2, 2.2, 3.2], dtype="<f8"), axis=axis),
            formal_source("test-1", class_label=1, source_order=1, values=np.array([2.3, 3.3, 2.3, 3.3], dtype="<f8"), axis=axis),
        )
        fixture_inputs = formal_module.Phase3DenoisingFormalInputs(
            stateless_sources=stateless_sources,
            fit_sources=fit_sources,
            validation_sources=validation_sources,
            test_sources=test_sources,
            axis_sha256=array_sha(axis),
            train_record_ids_sha256=report_record_ids_sha256(tuple(source.record_id for source in fit_sources)),
            validation_record_ids_sha256=report_record_ids_sha256(tuple(source.record_id for source in validation_sources)),
            test_record_ids_sha256=report_record_ids_sha256(tuple(source.record_id for source in test_sources)),
            train_record_ledger_sha256=canonical_record_ledger_sha256(tuple(source.record_id for source in fit_sources)),
            validation_record_ledger_sha256=canonical_record_ledger_sha256(tuple(source.record_id for source in validation_sources)),
            test_record_ledger_sha256=canonical_record_ledger_sha256(tuple(source.record_id for source in test_sources)),
            train_matrix_sha256=matrix_sha256(tuple(source.spectrum for source in fit_sources)),
            validation_matrix_sha256=matrix_sha256(tuple(source.spectrum for source in validation_sources)),
            test_matrix_sha256=matrix_sha256(tuple(source.spectrum for source in test_sources)),
        )
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = formal_module._build_phase3_denoising_formal_fixture_artifact(
                fixture_inputs,
                systems,
                Path(first_root),
                config=config,
                catalog=catalog,
                worker_count=1,
                project_root=ROOT,
            )
            second = formal_module._build_phase3_denoising_formal_fixture_artifact(
                fixture_inputs,
                systems,
                Path(second_root),
                config=config,
                catalog=catalog,
                worker_count=2,
                project_root=ROOT,
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertIsInstance(first, DenoisingFormalSummary)
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
            self.assertIn("denoised_outputs.f64le", first_files)
            self.assertIn("fitted_states.f64le", first_files)
            self.assertNotIn(b"bytearray", first_files["manifest.json"])
            verified = verify_phase3_denoising_formal(first.path, worker_count=3, project_root=ROOT)
            self.assertEqual(verified.run_id, first.run_id)
            self.assertEqual(verified.path, first.path)

    def test_verifier_rejects_checksum_receipt_and_state_binding_drift(self) -> None:
        config = load_phase3_denoising_formal_config(CONFIG)
        catalog = load_classical_catalog(CATALOG)
        system = next(
            current for current in catalog.systems
            if current.task_line is TaskLine.DENOISING and current.family_id == "pca_reconstruction"
            and current.hyperparameters["n_components"] == 2
        )
        axis = np.linspace(500.0, 503.0, 4, dtype="<f8")
        fit_sources = (
            formal_source("train-0", class_label=0, source_order=0, values=np.array([0.0, 1.0, 0.0, 1.0], dtype="<f8"), axis=axis),
            formal_source("train-1", class_label=0, source_order=1, values=np.array([0.1, 1.1, 0.1, 1.1], dtype="<f8"), axis=axis),
            formal_source("train-2", class_label=1, source_order=2, values=np.array([2.0, 3.0, 2.0, 3.0], dtype="<f8"), axis=axis),
            formal_source("train-3", class_label=1, source_order=3, values=np.array([2.1, 3.1, 2.1, 3.1], dtype="<f8"), axis=axis),
        )
        validation_sources = (
            formal_source("validation-0", class_label=0, source_order=0, values=np.array([0.2, 1.2, 0.2, 1.2], dtype="<f8"), axis=axis),
        )
        test_sources = (
            formal_source("test-0", class_label=1, source_order=0, values=np.array([2.2, 3.2, 2.2, 3.2], dtype="<f8"), axis=axis),
        )
        fixture_inputs = formal_module.Phase3DenoisingFormalInputs(
            stateless_sources=(formal_source("stateless-0", class_label=0, source_order=0, values=np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8"), axis=axis),),
            fit_sources=fit_sources,
            validation_sources=validation_sources,
            test_sources=test_sources,
            axis_sha256=array_sha(axis),
            train_record_ids_sha256=report_record_ids_sha256(tuple(source.record_id for source in fit_sources)),
            validation_record_ids_sha256=report_record_ids_sha256(tuple(source.record_id for source in validation_sources)),
            test_record_ids_sha256=report_record_ids_sha256(tuple(source.record_id for source in test_sources)),
            train_record_ledger_sha256=canonical_record_ledger_sha256(tuple(source.record_id for source in fit_sources)),
            validation_record_ledger_sha256=canonical_record_ledger_sha256(tuple(source.record_id for source in validation_sources)),
            test_record_ledger_sha256=canonical_record_ledger_sha256(tuple(source.record_id for source in test_sources)),
            train_matrix_sha256=matrix_sha256(tuple(source.spectrum for source in fit_sources)),
            validation_matrix_sha256=matrix_sha256(tuple(source.spectrum for source in validation_sources)),
            test_matrix_sha256=matrix_sha256(tuple(source.spectrum for source in test_sources)),
        )
        with tempfile.TemporaryDirectory() as output_root:
            summary = formal_module._build_phase3_denoising_formal_fixture_artifact(
                fixture_inputs,
                (system,),
                Path(output_root),
                config=config,
                catalog=catalog,
                worker_count=1,
                project_root=ROOT,
            )
            output_path = summary.path / "denoised_outputs.f64le"
            checksum_path = summary.path / "SHA256SUMS"
            original_output = output_path.read_bytes()
            original_checksums = checksum_path.read_text(encoding="utf-8")
            changed_output = bytes([original_output[0] ^ 1]) + original_output[1:]
            output_path.write_bytes(changed_output)
            checksum_path.write_text(
                original_checksums.replace(sha256_bytes(original_output), sha256_bytes(changed_output)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DenoisingFormalError, "output stream"):
                verify_phase3_denoising_formal(summary.path, worker_count=2, project_root=ROOT)
            output_path.write_bytes(original_output)
            checksum_path.write_text(original_checksums, encoding="utf-8")

            fit_path = summary.path / "fit_receipts.jsonl"
            fit_original = fit_path.read_bytes()
            fit_rows = [json.loads(line) for line in fit_original.splitlines()]
            fit_rows[0]["training_matrix_sha256"] = "0" * 64
            fit_changed = b"".join(canonical(row) for row in fit_rows)
            fit_path.write_bytes(fit_changed)
            checksum_path.write_text(
                original_checksums.replace(sha256_bytes(fit_original), sha256_bytes(fit_changed)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DenoisingFormalError, "fit receipt identity"):
                verify_phase3_denoising_formal(summary.path, worker_count=2, project_root=ROOT)


class Phase3DenoisingFormalCliTest(unittest.TestCase):
    def test_cli_supports_build_and_verify_only(self) -> None:
        build_summary = mock.Mock(path=Path("/tmp/phase3-denoising-formal"), run_id="run-1")
        verify_summary = mock.Mock(path=Path("/tmp/phase3-denoising-formal"), run_id="run-1")
        with mock.patch.object(cli_module, "build_phase3_denoising_formal", return_value=build_summary) as build_mock:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli_module.main(["build", "--output-root", "/tmp/out", "--worker-count", "1"])
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            build_payload = json.loads(stdout.getvalue())
            self.assertEqual(build_payload["status"], "built")
            self.assertEqual(build_payload["path"], str(build_summary.path))
            build_mock.assert_called_once_with(Path("/tmp/out"), worker_count=1)

        with mock.patch.object(cli_module, "verify_phase3_denoising_formal", return_value=verify_summary) as verify_mock:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli_module.main(["verify", "--run-path", "/tmp/run", "--worker-count", "2"])
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            verify_payload = json.loads(stdout.getvalue())
            self.assertEqual(verify_payload["status"], "verified")
            self.assertEqual(verify_payload["path"], str(verify_summary.path))
            verify_mock.assert_called_once_with(
                Path("/tmp/run"), worker_count=2, project_root=ROOT
            )

    def test_cli_rejects_unknown_mode(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            cli_module.main(["unknown"])
        self.assertNotEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
