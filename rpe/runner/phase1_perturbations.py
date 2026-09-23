from __future__ import annotations

import math
import struct
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from rpe.perturb import (
    BaselineDistortionError,
    CorrelatedNoiseError,
    GaussianNoiseError,
    P1GlobalPeakAttenuation,
    P2SelectiveWeakPeakAttenuation,
    P3PeakBroadening,
    P4WeakPeakDeletion,
    P5FalsePeakInsertion,
    P8LowOrderBaselineDistortion,
    P9GaussianWhiteNoise,
    P10CorrelatedNoise,
    P11AxisTransformError,
    P11GlobalWavenumberShift,
    P12AxisTransformError,
    P12QuadraticWavenumberWarp,
    PeakFamilyError,
    Perturbation,
    PerturbationContractError,
    PerturbationContext,
    PerturbationSweepConfig,
    PerturbationState,
    validate_perturbation_result,
)
from rpe.runner.phase1_config import Phase1CoreConfig
from rpe.runner.phase1_gates import (
    Phase1GateError,
    native_cell_view,
    validate_cell_native_gate,
)
from rpe.runner.phase1_selection import Phase1Source
from rpe.runner.phase1_types import (
    CellEvidence,
    CellStatus,
    PerturbedRecord,
    Phase1Cell,
    Phase1RunnerError,
    Phase1ShardPayload,
    _validate_sources_sequence,
)


_OPERATOR_CLASSES: dict[str, type[Perturbation]] = {
    "p01": P1GlobalPeakAttenuation,
    "p02": P2SelectiveWeakPeakAttenuation,
    "p03": P3PeakBroadening,
    "p04": P4WeakPeakDeletion,
    "p05": P5FalsePeakInsertion,
    "p08": P8LowOrderBaselineDistortion,
    "p09": P9GaussianWhiteNoise,
    "p10": P10CorrelatedNoise,
    "p11": P11GlobalWavenumberShift,
    "p12": P12QuadraticWavenumberWarp,
}
_DEFERRED_IDS = frozenset({"p06", "p07"})
_PEAK_REASON_BY_PATH = {
    "point count": "insufficient_points_for_peak_model",
    "intensity range": "nonpositive_intensity_range",
    "detected peaks": "no_detected_peak",
    "median_fwhm_cm1": "invalid_peak_component",
    "component validity": "invalid_peak_component",
    "false peak candidates": "false_peak_placement_impossible",
}
_FAILED_TYPED_ERRORS = (
    BaselineDistortionError,
    GaussianNoiseError,
    CorrelatedNoiseError,
    P11AxisTransformError,
    P12AxisTransformError,
    PerturbationContractError,
)


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise Phase1RunnerError(path, "must be a nonempty string")
    return value


def _positive_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase1RunnerError(path, "must be a positive integer")
    return value


def _nonnegative_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase1RunnerError(path, "must be a nonnegative integer")
    return value


def _perturbation_context(sweep: PerturbationSweepConfig) -> PerturbationContext:
    return PerturbationContext(
        sweep_id=sweep.sweep_id,
        sweep_config_sha256=sweep.sha256,
        global_seed=sweep.global_seed,
    )


def _complete_evidence() -> CellEvidence:
    return CellEvidence(
        status=CellStatus.COMPLETE,
        reason_code=None,
        exception_type=None,
        exception_path=None,
        exception_message=None,
        native_gate={},
    )


def _complete_evidence_with_witness(native_gate: dict[str, object]) -> CellEvidence:
    return CellEvidence(
        status=CellStatus.COMPLETE,
        reason_code=None,
        exception_type=None,
        exception_path=None,
        exception_message=None,
        native_gate=native_gate,
    )


def _not_applicable_evidence(
    *,
    reason_code: str,
    exception: BaseException | None = None,
) -> CellEvidence:
    return CellEvidence(
        status=CellStatus.NOT_APPLICABLE,
        reason_code=reason_code,
        exception_type=None if exception is None else type(exception).__name__,
        exception_path=None if exception is None else getattr(exception, "path", None),
        exception_message=None if exception is None else str(exception),
        native_gate={},
    )


def _failed_evidence(error: BaseException) -> CellEvidence:
    return CellEvidence(
        status=CellStatus.FAILED,
        reason_code=None,
        exception_type=type(error).__name__,
        exception_path=getattr(error, "path", None),
        exception_message=str(error),
        native_gate={},
    )


class P10MemoryAdmission:
    def __init__(self, memory_budget_bytes: int) -> None:
        self._budget = _positive_int("memory_budget_bytes", memory_budget_bytes)
        self._active = 0
        self._outstanding_weights: dict[int, int] = {}
        self._condition = threading.Condition()

    def acquire(self, estimated_bytes: int) -> None:
        weight = _positive_int("estimated_bytes", estimated_bytes)
        if weight > self._budget:
            raise Phase1RunnerError("estimated_bytes", "must not exceed budget")
        with self._condition:
            while self._active + weight > self._budget:
                self._condition.wait()
            self._active += weight
            self._outstanding_weights[weight] = self._outstanding_weights.get(weight, 0) + 1

    def release(self, estimated_bytes: int) -> None:
        weight = _positive_int("estimated_bytes", estimated_bytes)
        with self._condition:
            outstanding = self._outstanding_weights.get(weight, 0)
            if outstanding == 0:
                raise Phase1RunnerError(
                    "estimated_bytes",
                    "weight was not acquired",
                )
            if weight > self._active:
                raise Phase1RunnerError(
                    "estimated_bytes",
                    "release exceeds acquired total",
                )
            self._active -= weight
            if outstanding == 1:
                del self._outstanding_weights[weight]
            else:
                self._outstanding_weights[weight] = outstanding - 1
            self._condition.notify_all()

    @contextmanager
    def admit(self, estimated_bytes: int):
        self.acquire(estimated_bytes)
        try:
            yield
        finally:
            self.release(estimated_bytes)


def operator_for(
    perturbation_id: str,
    sweep: PerturbationSweepConfig,
) -> Perturbation:
    perturbation_name = _nonempty_string("perturbation_id", perturbation_id)
    operator_class = _OPERATOR_CLASSES.get(perturbation_name)
    if operator_class is None:
        raise Phase1RunnerError(
            "perturbation_id",
            f"unsupported {perturbation_name!r}",
        )
    return operator_class(sweep)


def estimate_p10_peak_bytes(point_count: int) -> int:
    count = _positive_int("point_count", point_count)
    return 32 * count * count + 64 * count + 2**30


def _deferred_cell(
    source: Phase1Source,
    perturbation_id: str,
    config: Phase1CoreConfig,
) -> Phase1Cell:
    return Phase1Cell(
        source=source,
        perturbation_id=perturbation_id,
        state=None,
        records=(),
        evidence=_not_applicable_evidence(reason_code=config.deferred_reason_code),
    )


def _classify_peak_family_error(
    perturbation_id: str,
    error: PeakFamilyError,
    config: Phase1CoreConfig,
) -> CellEvidence:
    if perturbation_id not in {"p01", "p02", "p03", "p04", "p05"}:
        return _failed_evidence(error)
    reason_code = _PEAK_REASON_BY_PATH.get(error.path)
    if (
        reason_code is not None
        and reason_code in config.peak_not_applicable_reason_codes
    ):
        return _not_applicable_evidence(reason_code=reason_code, exception=error)
    return _failed_evidence(error)


def _operator_for_runtime(
    perturbation_id: str,
    sweep: PerturbationSweepConfig,
    operator_factory: Callable[[str, PerturbationSweepConfig], Perturbation],
) -> Perturbation:
    operator = operator_factory(perturbation_id, sweep)
    if not isinstance(operator, Perturbation):
        raise Phase1RunnerError(
            "operator_factory",
            "must return an object implementing Perturbation",
        )
    return operator


def _p10_capacity_not_applicable(
    state: PerturbationState,
    sweep: PerturbationSweepConfig,
) -> bool:
    peak_indices = getattr(state, "peak_indices", None)
    if not isinstance(peak_indices, tuple):
        return False
    positive_alphas = tuple(alpha for alpha in sweep.alpha_grid if alpha > 0.0)
    if not positive_alphas:
        return False
    capacity = int(math.floor(max(positive_alphas) * len(peak_indices)))
    return capacity == 0


def run_perturbation_cell(
    source: Phase1Source,
    perturbation_id: str,
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    *,
    operator_factory: Callable[[str, PerturbationSweepConfig], Perturbation] = operator_for,
    p10_admission: P10MemoryAdmission | None = None,
) -> Phase1Cell:
    if not isinstance(source, Phase1Source):
        raise Phase1RunnerError("source", "must be Phase1Source")
    perturbation_name = _nonempty_string("perturbation_id", perturbation_id)
    if perturbation_name in _DEFERRED_IDS:
        return _deferred_cell(source, perturbation_name, config)

    context = _perturbation_context(sweep)
    estimated_p10_bytes = None
    if perturbation_name == "p10":
        estimated_p10_bytes = estimate_p10_peak_bytes(source.spectrum.axis_cm1.size)

    def execute() -> Phase1Cell:
        try:
            operator = _operator_for_runtime(
                perturbation_name,
                sweep,
                operator_factory,
            )
            state = operator.prepare(source.spectrum, context)
            if (
                perturbation_name == "p05"
                and _p10_capacity_not_applicable(state, sweep)
            ):
                return Phase1Cell(
                    source=source,
                    perturbation_id=perturbation_name,
                    state=state,
                    records=(),
                    evidence=_not_applicable_evidence(
                        reason_code="zero_false_peak_insertion_capacity"
                    ),
                )
            records: list[PerturbedRecord] = []
            for alpha in sweep.alpha_grid:
                result = operator.apply(source.spectrum, alpha, state)
                validate_perturbation_result(
                    source.spectrum,
                    state,
                    result,
                    sweep,
                )
                records.append(
                    PerturbedRecord(
                        source=source,
                        result=result,
                        alpha_float64_le_hex=struct.pack("<d", result.alpha).hex(),
                    )
                )
            return Phase1Cell(
                source=source,
                perturbation_id=perturbation_name,
                state=state,
                records=tuple(records),
                evidence=_complete_evidence(),
            )
        except PeakFamilyError as error:
            return Phase1Cell(
                source=source,
                perturbation_id=perturbation_name,
                state=locals().get("state"),
                records=(),
                evidence=_classify_peak_family_error(
                    perturbation_name,
                    error,
                    config,
                ),
            )
        except _FAILED_TYPED_ERRORS as error:
            return Phase1Cell(
                source=source,
                perturbation_id=perturbation_name,
                state=locals().get("state"),
                records=(),
                evidence=_failed_evidence(error),
            )
        except Phase1GateError as error:
            return Phase1Cell(
                source=source,
                perturbation_id=perturbation_name,
                state=locals().get("state"),
                records=(),
                evidence=_failed_evidence(error),
            )

    if estimated_p10_bytes is not None:
        admission = (
            p10_admission
            if p10_admission is not None
            else P10MemoryAdmission(estimated_p10_bytes)
        )
        with admission.admit(estimated_p10_bytes):
            cell = execute()
    else:
        cell = execute()
    if cell.status is not CellStatus.COMPLETE:
        return cell
    try:
        gate = validate_cell_native_gate(
            native_cell_view(cell),
            sweep,
            tolerance=config.core_gate["float_relative_tolerance"],
        )
    except Phase1GateError as error:
        return Phase1Cell(
            source=cell.source,
            perturbation_id=cell.perturbation_id,
            state=cell.state,
            records=(),
            evidence=_failed_evidence(error),
        )
    return Phase1Cell(
        source=cell.source,
        perturbation_id=cell.perturbation_id,
        state=cell.state,
        records=cell.records,
        evidence=_complete_evidence_with_witness(dict(gate.witness)),
    )


def run_source_cells(
    source: Phase1Source,
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    *,
    p10_admission: P10MemoryAdmission | None = None,
) -> tuple[Phase1Cell, ...]:
    return tuple(
        run_perturbation_cell(
            source,
            perturbation_id,
            config,
            sweep,
            operator_factory=operator_for,
            p10_admission=p10_admission,
        )
        for perturbation_id in sweep.perturbation_ids
    )


def run_shard_payload(
    sources: tuple[Phase1Source, ...],
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    *,
    shard_index: int,
    worker_count: int,
    memory_budget_bytes: int,
) -> Phase1ShardPayload:
    _nonnegative_int("shard_index", shard_index)
    _positive_int("worker_count", worker_count)
    budget = _positive_int("memory_budget_bytes", memory_budget_bytes)
    sorted_sources = tuple(
        sorted(sources, key=lambda source: source.selection.selection_rank)
    )
    _validate_sources_sequence(sorted_sources)
    estimates = tuple(
        estimate_p10_peak_bytes(source.spectrum.axis_cm1.size)
        for source in sorted_sources
    )
    if any(estimate > budget for estimate in estimates):
        raise Phase1RunnerError(
            "memory_budget_bytes",
            "must admit every source p10 estimate",
        )

    admission = P10MemoryAdmission(budget)
    source_cells: dict[int, tuple[Phase1Cell, ...]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_by_rank = {
            source.selection.selection_rank: executor.submit(
                run_source_cells,
                source,
                config,
                sweep,
                p10_admission=admission,
            )
            for source in sorted_sources
        }
        for rank, future in future_by_rank.items():
            source_cells[rank] = future.result()

    ordered_cells = tuple(
        cell
        for source in sorted_sources
        for cell in source_cells[source.selection.selection_rank]
    )
    return Phase1ShardPayload(
        shard_index=shard_index,
        sources=sorted_sources,
        cells=ordered_cells,
    )
