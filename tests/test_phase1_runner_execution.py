from __future__ import annotations

import re
import struct
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from phase1_runner_helpers import (  # noqa: E402
    phase1_fixture_config,
    write_phase1_fixture_dataset,
)
from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.io.store import UnifiedDataset  # noqa: E402
from rpe.perturb import (  # noqa: E402
    BaselineDistortionError,
    CorrelatedNoiseError,
    GaussianNoiseError,
    P11AxisTransformError,
    P12AxisTransformError,
    PeakFamilyError,
    Perturbation,
    PerturbationContractError,
    PerturbationContext,
    PerturbationResult,
    PerturbationState,
    load_perturbation_sweep_config,
    validate_perturbation_result,
)
import rpe.runner.phase1_perturbations as phase1_perturbations  # noqa: E402
from rpe.runner.phase1_selection import SelectedSourceRow, load_phase1_source  # noqa: E402
from rpe.runner.phase1_types import (  # noqa: E402
    CellEvidence,
    CellStatus,
    PerturbedRecord,
    Phase1Cell,
    Phase1RunnerError,
    Phase1ShardPayload,
)
from rpe.runner.phase1_perturbations import (  # noqa: E402
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    operator_for,
    run_perturbation_cell,
    run_shard_payload,
    run_source_cells,
)


SWEEP_CONFIG = (
    ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"
)


def fixture_source(
    *,
    record_id: str = "a-00",
    selection_rank: int = 0,
) -> object:
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = write_phase1_fixture_dataset(Path(tmp))
        with UnifiedDataset.open(dataset_path, verify_checksums=True) as dataset:
            return load_phase1_source(
                dataset,
                SelectedSourceRow(
                    selection_rank=selection_rank,
                    record_id=record_id,
                    sample_id="sample-a" if record_id != "decreasing-record" else "sample-b",
                    class_label=0,
                    mineral_name="Mineral-A",
                    axis_id=f"axis-{selection_rank}",
                ),
            )


def fixture_config(*, peak_reason_codes: tuple[str, ...] | None = None) -> object:
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = write_phase1_fixture_dataset(Path(tmp))
        config = phase1_fixture_config(
            dataset_path,
            subset_size=8,
            shard_source_count=4,
        )
    if peak_reason_codes is None:
        return config
    return replace(config, peak_not_applicable_reason_codes=peak_reason_codes)


def fixture_sweep():
    return load_perturbation_sweep_config(SWEEP_CONFIG)


def perturbation_context(sweep) -> PerturbationContext:
    return PerturbationContext(
        sweep_id=sweep.sweep_id,
        sweep_config_sha256=sweep.sha256,
        global_seed=sweep.global_seed,
    )


class CountingOperator:
    axis_behavior = None

    def __init__(self, inner: Perturbation) -> None:
        self.inner = inner
        self.perturbation_id = inner.perturbation_id
        self.axis_behavior = inner.axis_behavior
        self.prepare_calls = 0
        self.applied_alphas: list[float] = []

    def prepare(self, spectrum, context):
        self.prepare_calls += 1
        return self.inner.prepare(spectrum, context)

    def apply(self, spectrum, alpha, state):
        self.applied_alphas.append(alpha)
        return self.inner.apply(spectrum, alpha, state)


class RaisingFactory:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls: list[str] = []

    def __call__(self, perturbation_id, sweep):
        self.calls.append(perturbation_id)
        raise self.error


class ScriptedOperator:
    def __init__(self, perturbation_id: str, source, sweep, failure=None) -> None:
        self.perturbation_id = perturbation_id
        self._inner = operator_for(perturbation_id, sweep)
        self.axis_behavior = self._inner.axis_behavior
        self._failure = failure
        self.prepare_calls = 0
        self.apply_calls = 0

    def prepare(self, spectrum, context):
        self.prepare_calls += 1
        if isinstance(self._failure, Exception) and self._failure.args == ("prepare",):
            raise self._failure
        return self._inner.prepare(spectrum, context)

    def apply(self, spectrum, alpha, state):
        self.apply_calls += 1
        if callable(self._failure):
            maybe = self._failure(alpha, self.apply_calls)
            if maybe is not None:
                raise maybe
        return self._inner.apply(spectrum, alpha, state)


class DelayedP10Operator:
    axis_behavior = operator_for("p10", fixture_sweep()).axis_behavior

    def __init__(self, source, sweep, started, active, peak_active, release_event):
        self.perturbation_id = "p10"
        self._inner = operator_for("p10", sweep)
        self._source = source
        self._sweep = sweep
        self._started = started
        self._active = active
        self._peak_active = peak_active
        self._release_event = release_event

    def prepare(self, spectrum, context):
        with self._active["lock"]:
            self._active["value"] += 1
            self._peak_active["value"] = max(
                self._peak_active["value"],
                self._active["value"],
            )
        self._started.append(spectrum.spectrum_id)
        self._release_event.wait(timeout=5.0)
        return self._inner.prepare(spectrum, context)

    def apply(self, spectrum, alpha, state):
        try:
            return self._inner.apply(spectrum, alpha, state)
        finally:
            if alpha == self._sweep.alpha_grid[-1]:
                with self._active["lock"]:
                    self._active["value"] -= 1


class RunnerExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = fixture_source()
        self.other_source = fixture_source(record_id="a-01", selection_rank=1)
        self.third_source = fixture_source(record_id="a-02", selection_rank=2)
        self.config = fixture_config(
            peak_reason_codes=(
                "false_peak_placement_impossible",
                "insufficient_points_for_peak_model",
                "invalid_peak_component",
                "no_detected_peak",
                "nonpositive_intensity_range",
                "zero_false_peak_insertion_capacity",
            )
        )
        self.sweep = fixture_sweep()

    def test_p09_and_p10_prepare_once_apply_nine_and_validate_every_result(self) -> None:
        for perturbation_id in ("p09", "p10"):
            with self.subTest(perturbation_id=perturbation_id):
                holder: dict[str, CountingOperator] = {}

                def factory(name, sweep):
                    operator = CountingOperator(operator_for(name, sweep))
                    holder["operator"] = operator
                    return operator

                cell = run_perturbation_cell(
                    self.source,
                    perturbation_id,
                    self.config,
                    self.sweep,
                    operator_factory=factory,
                    p10_admission=P10MemoryAdmission(estimate_p10_peak_bytes(self.source.spectrum.axis_cm1.size))
                    if perturbation_id == "p10"
                    else None,
                )
                self.assertEqual(cell.status, CellStatus.COMPLETE)
                self.assertEqual(holder["operator"].prepare_calls, 1)
                self.assertEqual(tuple(holder["operator"].applied_alphas), self.sweep.alpha_grid)
                self.assertEqual(tuple(record.alpha for record in cell.records), self.sweep.alpha_grid)
                self.assertEqual(len(cell.records), 9)
                for record in cell.records:
                    validate_perturbation_result(
                        self.source.spectrum,
                        cell.state,
                        record.result,
                        self.sweep,
                    )

    def test_p06_and_p07_return_deferred_without_calling_factory(self) -> None:
        for perturbation_id in ("p06", "p07"):
            with self.subTest(perturbation_id=perturbation_id):
                calls: list[str] = []

                def factory(name, sweep):
                    calls.append(name)
                    return operator_for(name, sweep)

                cell = run_perturbation_cell(
                    self.source,
                    perturbation_id,
                    self.config,
                    self.sweep,
                    operator_factory=factory,
                )
                self.assertEqual(calls, [])
                self.assertEqual(cell.status, CellStatus.NOT_APPLICABLE)
                self.assertEqual(cell.reason_code, self.config.deferred_reason_code)
                self.assertIsNone(cell.state)
                self.assertEqual(cell.records, ())

    def test_operator_for_returns_exact_public_classes_and_rejects_deferred_or_unknown(self) -> None:
        expected = {
            "p01": "P1GlobalPeakAttenuation",
            "p02": "P2SelectiveWeakPeakAttenuation",
            "p03": "P3PeakBroadening",
            "p04": "P4WeakPeakDeletion",
            "p05": "P5FalsePeakInsertion",
            "p08": "P8LowOrderBaselineDistortion",
            "p09": "P9GaussianWhiteNoise",
            "p10": "P10CorrelatedNoise",
            "p11": "P11GlobalWavenumberShift",
            "p12": "P12QuadraticWavenumberWarp",
        }
        for perturbation_id, class_name in expected.items():
            with self.subTest(perturbation_id=perturbation_id):
                self.assertEqual(
                    operator_for(perturbation_id, self.sweep).__class__.__name__,
                    class_name,
                )
        for perturbation_id in ("p06", "p07", "p99"):
            with self.subTest(rejected=perturbation_id):
                with self.assertRaisesRegex(Phase1RunnerError, re.escape(f"perturbation_id: unsupported {perturbation_id!r}")):
                    operator_for(perturbation_id, self.sweep)

    def test_peak_family_path_mappings_and_typed_failures_are_classified_exactly(self) -> None:
        mapping_cases = (
            ("point count", "insufficient_points_for_peak_model"),
            ("intensity range", "nonpositive_intensity_range"),
            ("detected peaks", "no_detected_peak"),
            ("median_fwhm_cm1", "invalid_peak_component"),
            ("component validity", "invalid_peak_component"),
            ("false peak candidates", "false_peak_placement_impossible"),
        )
        for path, expected_reason in mapping_cases:
            with self.subTest(path=path):
                cell = run_perturbation_cell(
                    self.source,
                    "p01",
                    self.config,
                    self.sweep,
                    operator_factory=RaisingFactory(PeakFamilyError(path, "fixture domain")).__call__,
                )
                self.assertEqual(cell.status, CellStatus.NOT_APPLICABLE)
                self.assertEqual(cell.reason_code, expected_reason)
                self.assertEqual(cell.evidence.exception_type, "PeakFamilyError")
                self.assertEqual(cell.evidence.exception_path, path)
                self.assertEqual(cell.records, ())

        failed_peak = run_perturbation_cell(
            self.source,
            "p01",
            self.config,
            self.sweep,
            operator_factory=RaisingFactory(PeakFamilyError("other path", "fixture")).__call__,
        )
        self.assertEqual(failed_peak.status, CellStatus.FAILED)
        self.assertIsNone(failed_peak.reason_code)

        typed_errors = (
            BaselineDistortionError("baseline", "fixture"),
            GaussianNoiseError("noise", "fixture"),
            CorrelatedNoiseError("correlated", "fixture"),
            P11AxisTransformError("axis", "fixture"),
            P12AxisTransformError("axis", "fixture"),
            PerturbationContractError("result", "fixture"),
        )
        for error in typed_errors:
            with self.subTest(error=type(error).__name__):
                cell = run_perturbation_cell(
                    self.source,
                    "p08" if isinstance(error, BaselineDistortionError) else "p09",
                    self.config,
                    self.sweep,
                    operator_factory=RaisingFactory(error).__call__,
                )
                self.assertEqual(cell.status, CellStatus.FAILED)
                self.assertEqual(cell.evidence.exception_type, type(error).__name__)
                self.assertEqual(cell.records, ())

    def test_peak_family_allowlist_is_scoped_to_p01_p05_only(self) -> None:
        cell = run_perturbation_cell(
            self.source,
            "p08",
            self.config,
            self.sweep,
            operator_factory=RaisingFactory(PeakFamilyError("point count", "fixture")).__call__,
        )
        self.assertEqual(cell.status, CellStatus.FAILED)
        self.assertIsNone(cell.reason_code)
        self.assertEqual(cell.evidence.exception_type, "PeakFamilyError")

    def test_p05_real_zero_insertion_capacity_is_not_applicable(self) -> None:
        single_peak_source = replace(
            self.source,
            spectrum=Spectrum1D(
                spectrum_id=self.source.spectrum.spectrum_id,
                sample_id=self.source.spectrum.sample_id,
                axis_cm1=np.array(
                    (100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0),
                    dtype="<f8",
                ),
                intensity=np.array(
                    (0.2, 0.5, 1.5, 4.0, 1.5, 0.5, 0.2),
                    dtype="<f8",
                ),
            ),
        )
        cell = run_perturbation_cell(
            single_peak_source,
            "p05",
            self.config,
            self.sweep,
        )
        self.assertEqual(cell.status, CellStatus.NOT_APPLICABLE)
        self.assertEqual(cell.reason_code, "zero_false_peak_insertion_capacity")
        self.assertIsNotNone(cell.state)
        self.assertEqual(len(cell.state.peak_indices), 1)
        self.assertEqual(cell.records, ())

    def test_typed_error_after_successful_alpha_emits_no_partial_records(self) -> None:
        def fail_on_second_alpha(alpha, apply_calls):
            if apply_calls == 2:
                return GaussianNoiseError("noise", "explode on second alpha")
            return None

        def factory(name, sweep):
            return ScriptedOperator(name, self.source, sweep, failure=fail_on_second_alpha)

        cell = run_perturbation_cell(
            self.source,
            "p09",
            self.config,
            self.sweep,
            operator_factory=factory,
        )
        self.assertEqual(cell.status, CellStatus.FAILED)
        self.assertEqual(cell.records, ())

    def test_programming_and_system_errors_propagate(self) -> None:
        for error in (
            ValueError("plain"),
            TypeError("plain"),
            RuntimeError("plain"),
            AssertionError("plain"),
            MemoryError("plain"),
            KeyboardInterrupt(),
        ):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    run_perturbation_cell(
                        self.source,
                        "p09",
                        self.config,
                        self.sweep,
                        operator_factory=RaisingFactory(error).__call__,
                    )

    def test_immutable_constructors_enforce_invariants_and_frozen_evidence(self) -> None:
        cell = run_perturbation_cell(self.source, "p09", self.config, self.sweep)
        record = cell.records[0]
        self.assertEqual(record.alpha, record.result.alpha)
        self.assertEqual(
            record.alpha_float64_le_hex,
            struct.pack("<d", record.result.alpha).hex(),
        )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("alpha_float64_le_hex: must equal little-endian float64 hex")):
            PerturbedRecord(
                source=self.source,
                result=record.result,
                alpha_float64_le_hex="00",
            )
        alias_result = replace(
            record.result,
            output=self._forged_output_with_alias(
                axis_cm1=self.source.spectrum.axis_cm1,
                intensity=record.result.output.intensity,
                spectrum_id=record.result.output.spectrum_id,
                sample_id=record.result.output.sample_id,
            ),
        )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("result.output.axis_cm1: must not share memory with source spectrum")):
            PerturbedRecord(
                source=self.source,
                result=alias_result,
                alpha_float64_le_hex=struct.pack("<d", alias_result.alpha).hex(),
            )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("records: must be empty unless status is complete")):
            Phase1Cell(
                source=self.source,
                perturbation_id="p09",
                state=None,
                records=cell.records,
                evidence=CellEvidence(
                    status=CellStatus.FAILED,
                    reason_code=None,
                    exception_type="GaussianNoiseError",
                    exception_path="noise",
                    exception_message="fixture",
                    native_gate=MappingProxyType({}),
                ),
            )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("sources[1].selection.selection_rank: must be strictly increasing and unique")):
            Phase1ShardPayload(
                shard_index=0,
                sources=(self.other_source, self.source),
                cells=run_source_cells(self.source, self.config, self.sweep)
                + run_source_cells(self.other_source, self.config, self.sweep),
            )
        evidence = cell.evidence
        self.assertIsInstance(evidence.native_gate, MappingProxyType)
        with self.assertRaises(TypeError):
            evidence.native_gate["x"] = 1

    def test_cell_evidence_status_matrix_rejects_invalid_reason_and_exception_combinations(self) -> None:
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("reason_code: must be absent when status is complete")):
            CellEvidence(
                status=CellStatus.COMPLETE,
                reason_code="unexpected",
                exception_type="PeakFamilyError",
                exception_path="point count",
                exception_message="fixture",
                native_gate={},
            )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("reason_code: must be present when status is not_applicable")):
            CellEvidence(
                status=CellStatus.NOT_APPLICABLE,
                reason_code=None,
                exception_type="PeakFamilyError",
                exception_path="point count",
                exception_message="fixture",
                native_gate={},
            )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("reason_code: must be absent when status is failed")):
            CellEvidence(
                status=CellStatus.FAILED,
                reason_code="unexpected",
                exception_type="PeakFamilyError",
                exception_path="point count",
                exception_message="fixture",
                native_gate={},
            )

    def test_complete_cell_rejects_negative_zero_alpha_bytes(self) -> None:
        cell = run_perturbation_cell(self.source, "p09", self.config, self.sweep)
        first = cell.records[0]
        neg_zero_result = replace(first.result, alpha=-0.0)
        neg_zero_record = PerturbedRecord(
            source=self.source,
            result=neg_zero_result,
            alpha_float64_le_hex=struct.pack("<d", -0.0).hex(),
        )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("records: must follow the frozen alpha order")):
            Phase1Cell(
                source=self.source,
                perturbation_id="p09",
                state=cell.state,
                records=(neg_zero_record, *cell.records[1:]),
                evidence=cell.evidence,
            )

    def test_phase1_shard_payload_rejects_duplicate_record_ids_even_with_unique_ranks(self) -> None:
        duplicate_source = replace(
            self.other_source,
            selection=replace(
                self.other_source.selection,
                record_id=self.source.selection.record_id,
            ),
        )
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("sources[1].selection.record_id: must be globally unique")):
            Phase1ShardPayload(
                shard_index=0,
                sources=(self.source, duplicate_source),
                cells=run_source_cells(self.source, self.config, self.sweep)
                + run_source_cells(duplicate_source, self.config, self.sweep),
            )

    def test_estimate_p10_peak_bytes_and_preflight_budget_rejection(self) -> None:
        self.assertEqual(
            estimate_p10_peak_bytes(23_775),
            32 * 23_775**2 + 64 * 23_775 + 2**30,
        )
        estimate = estimate_p10_peak_bytes(self.source.spectrum.axis_cm1.size)
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("memory_budget_bytes: must admit every source p10 estimate")):
            run_shard_payload(
                (self.source,),
                self.config,
                self.sweep,
                shard_index=0,
                worker_count=1,
                memory_budget_bytes=estimate - 1,
            )

    def test_p10_admission_caps_parallelism_and_payload_order_is_deterministic(self) -> None:
        estimate = estimate_p10_peak_bytes(self.source.spectrum.axis_cm1.size)
        started: list[str] = []
        active = {"value": 0, "lock": threading.Lock()}
        peak_active = {"value": 0}
        release_event = threading.Event()

        def factory(name, sweep):
            if name == "p10":
                return DelayedP10Operator(
                    self.source,
                    sweep,
                    started,
                    active,
                    peak_active,
                    release_event,
                )
            return operator_for(name, sweep)

        result_holder = {}
        error_holder = {}

        def run_payload():
            try:
                with patch.object(phase1_perturbations, "operator_for", new=factory):
                    result_holder["payload"] = run_shard_payload(
                        (self.third_source, self.source, self.other_source),
                        self.config,
                        self.sweep,
                        shard_index=0,
                        worker_count=3,
                        memory_budget_bytes=2 * estimate,
                    )
            except BaseException as error:  # pragma: no cover - captured for assertion
                error_holder["error"] = error

        thread = threading.Thread(target=run_payload)
        thread.start()
        deadline = time.time() + 5.0
        while len(started) < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(len(started), 2)
        self.assertLessEqual(peak_active["value"], 2)
        release_event.set()
        thread.join(timeout=5.0)
        if "error" in error_holder:
            raise error_holder["error"]
        payload = result_holder["payload"]
        self.assertEqual(
            tuple(source.selection.selection_rank for source in payload.sources),
            (0, 1, 2),
        )
        expected_order = tuple(
            perturbation_id
            for _source in payload.sources
            for perturbation_id in self.sweep.perturbation_ids
        )
        self.assertEqual(tuple(cell.perturbation_id for cell in payload.cells), expected_order)
        self.assertEqual(len(payload.cells), 36)

    def test_run_shard_payload_rejects_duplicate_ranks_and_records_before_submission(self) -> None:
        worker_calls: list[int] = []

        def unexpected_worker(*args, **kwargs):
            worker_calls.append(1)
            raise AssertionError("worker should not run")

        duplicate_rank = replace(
            self.other_source,
            selection=replace(self.other_source.selection, selection_rank=self.source.selection.selection_rank),
        )
        with patch.object(phase1_perturbations, "run_source_cells", new=unexpected_worker):
            with self.assertRaisesRegex(Phase1RunnerError, re.escape("sources[1].selection.selection_rank: must be strictly increasing and unique")):
                run_shard_payload(
                    (self.source, duplicate_rank),
                    self.config,
                    self.sweep,
                    shard_index=0,
                    worker_count=2,
                    memory_budget_bytes=2 * estimate_p10_peak_bytes(self.source.spectrum.axis_cm1.size),
                )
        self.assertEqual(worker_calls, [])

        duplicate_record = replace(
            self.other_source,
            selection=replace(self.other_source.selection, record_id=self.source.selection.record_id),
        )
        with patch.object(phase1_perturbations, "run_source_cells", new=unexpected_worker):
            with self.assertRaisesRegex(Phase1RunnerError, re.escape("sources[1].selection.record_id: must be globally unique")):
                run_shard_payload(
                    (self.source, duplicate_record),
                    self.config,
                    self.sweep,
                    shard_index=0,
                    worker_count=2,
                    memory_budget_bytes=2 * estimate_p10_peak_bytes(self.source.spectrum.axis_cm1.size),
                )
        self.assertEqual(worker_calls, [])

        payload = run_shard_payload(
            (self.third_source, self.source, self.other_source),
            self.config,
            self.sweep,
            shard_index=0,
            worker_count=2,
            memory_budget_bytes=3 * estimate_p10_peak_bytes(self.source.spectrum.axis_cm1.size),
        )
        self.assertEqual(
            tuple(source.selection.selection_rank for source in payload.sources),
            (0, 1, 2),
        )

    @staticmethod
    def _forged_output_with_alias(
        *,
        axis_cm1: np.ndarray,
        intensity: np.ndarray,
        spectrum_id: str,
        sample_id: str | None,
    ) -> Spectrum1D:
        spectrum = object.__new__(Spectrum1D)
        object.__setattr__(spectrum, "spectrum_id", spectrum_id)
        object.__setattr__(spectrum, "sample_id", sample_id)
        object.__setattr__(spectrum, "axis_cm1", axis_cm1)
        object.__setattr__(spectrum, "intensity", intensity)
        return spectrum


class MemoryAdmissionTest(unittest.TestCase):
    def test_memory_admission_validates_and_blocks_release_mismatch(self) -> None:
        admission = P10MemoryAdmission(10)
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("memory_budget_bytes: must be a positive integer")):
            P10MemoryAdmission(True)
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("estimated_bytes: must be a positive integer")):
            admission.acquire(False)
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("estimated_bytes: must not exceed budget")):
            admission.acquire(11)
        admission.acquire(4)
        admission.release(4)
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("estimated_bytes: weight was not acquired")):
            admission.release(1)

    def test_memory_admission_rejects_releasing_unacquired_weight_even_if_total_is_large_enough(self) -> None:
        admission = P10MemoryAdmission(10)
        admission.acquire(4)
        admission.acquire(6)
        with self.assertRaisesRegex(Phase1RunnerError, re.escape("estimated_bytes: weight was not acquired")):
            admission.release(5)
        admission.release(4)
        admission.release(6)


if __name__ == "__main__":
    unittest.main()
