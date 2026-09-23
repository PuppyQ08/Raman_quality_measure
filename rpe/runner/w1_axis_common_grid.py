from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from rpe.evaluation import Peak1D, PeakPairInput, SingleSpectrumInput, Spectrum1D, SpectrumPairInput, evaluate_metric
from rpe.metrics import ISLikeStructureToNoiseMetric, MAEMetric, MSEMetric, NMSEMetric, PeakDetectionCurvesMetric, PearsonRMetric, RMSEMetric, SAMMetric, Wasserstein1Metric
from rpe.runner.phase6_appendix_audits import AppendixAuditsError, resample_spectrum
from rpe.runner.phase6_appendix_metric_core import METRIC_IDS as METRIC_OUTPUT_IDS
from rpe.runner.w1_axis_robustness import NativePanelTable


ROOT = Path(__file__).resolve().parents[2]
_SUPPORT_CONFIG_PATHS = {
    "bacteria": ROOT / "experiments" / "phase4" / "configs" / "d2_protocol_a_full_domain_eligibility_v1.json",
    "sugar": ROOT / "experiments" / "phase4" / "configs" / "d4_protocol_a_full_domain_eligibility_v1.json",
    "d5": ROOT / "experiments" / "phase4" / "configs" / "d5_eligibility_v1.json",
}
_EXPECTED_SUPPORT_HASHES = {
    "bacteria": "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6",
    "sugar": "db910b11f92151db481391e96b8a06e246140596cfd4abad64764b87d2d84ee5",
    "d5": "23a29616dbac14efd7de072b9b84ee82c6740cf46ddd89e644819967f4648422",
}
_LOWER_IS_BETTER = {
    "mse",
    "rmse",
    "mae",
    "sam",
    "nmse",
    "wasserstein_1_cm1",
    "artifact_peak_ratio",
    "missing_peak_ratio",
}
_SUPPORT_CACHE: dict[str, "CommonGridSupportSpec"] | None = None
_NUMERICAL_CODE_DEPENDENCY_PATHS = (
    "rpe/runner/w1_axis_common_grid.py",
    "rpe/runner/phase6_appendix_audits.py",
    "rpe/runner/phase6_appendix_metric_core.py",
    "rpe/runner/phase4_d2_protocol_a.py",
    "rpe/runner/phase4_d4_eligibility.py",
    "rpe/runner/phase4_d5_protocol_a.py",
    "rpe/metrics/__init__.py",
    "rpe/metrics/fidelity.py",
    "rpe/metrics/transport.py",
    "rpe/metrics/reference_free.py",
    "rpe/metrics/peak.py",
    "rpe/evaluation/__init__.py",
    "rpe/evaluation/contracts.py",
    "rpe/methods/catalog.py",
    "rpe/methods/classical/peaks.py",
    "rpe/perturb/__init__.py",
)
_PROCESS_DETECT_PEAKS: Callable[[Spectrum1D], Sequence[object] | None] | None = None
_PROCESS_BLAS_LIMIT = None
_PROCESS_PHASE1 = None
_PROCESS_SWEEP = None


class W1AxisCommonGridError(ValueError):
    pass


@dataclass(frozen=True)
class CommonGridSupportSpec:
    dataset_id: str
    axis_cm1: np.ndarray
    point_count: int
    axis_sha256_f64: str
    max_in_range_native_gap_cm1: float | None

    def __post_init__(self) -> None:
        axis = np.ascontiguousarray(np.asarray(self.axis_cm1, dtype="<f8"))
        if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all():
            raise W1AxisCommonGridError(f"{self.dataset_id}: support axis must be finite 1D")
        if not np.all(np.diff(axis) > 0.0):
            raise W1AxisCommonGridError(f"{self.dataset_id}: support axis must be strictly increasing")
        if axis.size != int(self.point_count):
            raise W1AxisCommonGridError(f"{self.dataset_id}: support point count mismatch")
        digest = _array_sha(axis, dtype="<f8")
        if digest != self.axis_sha256_f64:
            raise W1AxisCommonGridError(f"{self.dataset_id}: support-axis digest mismatch")
        axis.setflags(write=False)
        object.__setattr__(self, "axis_cm1", axis)


@dataclass(frozen=True)
class CommonGridPanelResult:
    status: str
    table: NativePanelTable | None
    failure_rows: tuple[Mapping[str, object], ...]
    complete_metric_output_ids: tuple[str, ...] = ()
    incomplete_metric_output_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommonGridCondition:
    perturbation_id: str
    alpha: float
    spectrum: Spectrum1D


@dataclass(frozen=True)
class CommonGridRecord:
    record_id: str
    cluster_id: str
    source_spectrum: Spectrum1D
    condition_spectra: tuple[CommonGridCondition, ...]
    aggregation_weight: int = 1
    source_order: int = 0
    source_adapter_id: str | None = None
    source_adapter_metadata: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommonGridEvaluationResult:
    aggregated_rows: tuple[Mapping[str, object], ...]
    failure_rows: tuple[Mapping[str, object], ...]
    processed_record_count: int
    effective_worker_count: int = 1
    p10_worker_capacity: int = 1
    worker_process_ids: tuple[int, ...] = ()
    worker_blas_thread_limits: tuple[int, ...] = ()
    requested_worker_count: int = 1


@dataclass(frozen=True)
class CommonGridWorkerConfig:
    """Pickle-safe receipt used to rebuild a CWT detector inside each worker."""

    catalog_path: str
    cwt_system_id: str
    phase1_config_path: str
    sweep_config_path: str
    # Test-only synchronization seam.  Production configurations leave this
    # unset; a manager-backed barrier is pickle-safe under spawn.
    process_start_gate: object | None = None
    source_adapter_id: str | None = None
    p10_memory_budget_bytes: int = 0


@dataclass(frozen=True)
class CommonGridDatasetInput:
    """The retained source records and frozen Phase-1 objects for one dataset."""

    dataset_id: str
    records: tuple[CommonGridRecord, ...]
    planned_record_count: int
    support_spec: CommonGridSupportSpec
    detect_peaks: Callable[[Spectrum1D], Sequence[object] | None]
    sweep_sha256: str
    frozen_authority_sha256s: Mapping[str, str] = field(default_factory=dict)
    p10_memory_budget_bytes: int = 0
    worker_config: CommonGridWorkerConfig | None = None


@dataclass(frozen=True)
class CommonGridDatasetResult:
    dataset_id: str
    aggregated_rows: tuple[Mapping[str, object], ...]
    failure_rows: tuple[Mapping[str, object], ...]
    processed_record_count: int
    planned_record_count: int
    status: str
    elapsed_seconds: float
    cache_hit_count: int
    effective_worker_count: int = 1
    p10_worker_capacity: int = 1
    requested_worker_count: int = 1
    worker_process_ids: tuple[int, ...] = ()
    worker_blas_thread_limits: tuple[int, ...] = ()
    cache_hit: bool = False


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _array_sha(value: np.ndarray, *, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def aggregate_numerical_dependency_identity(dependency_sha256s: Mapping[str, str]) -> str:
    """Canonical aggregate cache key for the declared numerical dependency receipt."""
    expected = set(_NUMERICAL_CODE_DEPENDENCY_PATHS)
    observed = set(dependency_sha256s)
    if observed != expected:
        raise W1AxisCommonGridError("numerical dependency receipt does not match declared dependency set")
    if any(not isinstance(value, str) or value == "" for value in dependency_sha256s.values()):
        raise W1AxisCommonGridError("numerical dependency digest must be non-empty")
    return hashlib.sha256(_canonical_json_bytes(dict(sorted(dependency_sha256s.items())))).hexdigest()


def common_grid_numerical_code_identity(root: Path = ROOT) -> tuple[str, Mapping[str, str]]:
    """Hash all direct execution-semantics modules for cache binding."""
    hashes = {
        relative: hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest()
        for relative in _NUMERICAL_CODE_DEPENDENCY_PATHS
    }
    return aggregate_numerical_dependency_identity(hashes), hashes


def _authority_hashes(root: Path, relative_paths: Sequence[str]) -> Mapping[str, str]:
    return {
        relative: hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _load_json(path: Path) -> Mapping[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W1AxisCommonGridError(f"cannot read support config: {path}") from error
    if not isinstance(document, dict):
        raise W1AxisCommonGridError(f"{path}: support config must be a JSON object")
    return document


def _support_axis_from_config(dataset_id: str, document: Mapping[str, object]) -> tuple[np.ndarray, float | None]:
    support_grid = document.get("support_grid")
    if not isinstance(support_grid, AbcMapping):
        raise W1AxisCommonGridError(f"{dataset_id}: support_grid missing")
    if dataset_id in {"bacteria", "sugar"}:
        raw = support_grid.get("coordinates_cm1")
        axis = np.ascontiguousarray(np.asarray(raw, dtype="<f8"))
    else:
        try:
            start = float(support_grid["start_cm1"])
            stop = float(support_grid["stop_cm1"])
            step = float(support_grid["step_cm1"])
            point_count = int(support_grid["point_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise W1AxisCommonGridError(f"{dataset_id}: invalid support-grid contract") from error
        axis = np.ascontiguousarray(start + step * np.arange(point_count, dtype="<f8"), dtype="<f8")
        if point_count != axis.size or (point_count > 0 and float(axis[-1]) != stop):
            raise W1AxisCommonGridError(f"{dataset_id}: D5 support endpoint drift")
    max_gap = support_grid.get("max_in_range_native_gap_cm1")
    return axis, None if max_gap is None else float(max_gap)


def load_common_grid_support_specs() -> dict[str, CommonGridSupportSpec]:
    global _SUPPORT_CACHE
    if _SUPPORT_CACHE is not None:
        return dict(_SUPPORT_CACHE)
    specs: dict[str, CommonGridSupportSpec] = {}
    for dataset_id, path in _SUPPORT_CONFIG_PATHS.items():
        axis, max_gap = _support_axis_from_config(dataset_id, _load_json(path))
        specs[dataset_id] = CommonGridSupportSpec(
            dataset_id=dataset_id,
            axis_cm1=axis,
            point_count=int(axis.size),
            axis_sha256_f64=_EXPECTED_SUPPORT_HASHES[dataset_id],
            max_in_range_native_gap_cm1=max_gap,
        )
    _SUPPORT_CACHE = dict(specs)
    return dict(specs)


def _peak_detector(system: object) -> Callable[[Spectrum1D], Sequence[object] | None]:
    from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system

    def detect(spectrum: Spectrum1D) -> tuple[object, ...]:
        outcome = run_peak_detection_system(system, spectrum)
        if getattr(outcome, "status", None) not in {
            PeakRunStatus.COMPLETE,
            PeakRunStatus.COMPLETE_WITH_WARNING,
            "complete",
            "complete_with_warning",
        }:
            raise W1AxisCommonGridError(
                f"detector failure: {getattr(outcome, 'status', None)}"
            )
        return tuple(outcome.peaks)

    return detect


def _worker_initializer(worker_config: CommonGridWorkerConfig | None) -> None:
    """Set numerical limits and independently rebuild frozen detector state."""
    global _PROCESS_BLAS_LIMIT, _PROCESS_DETECT_PEAKS, _PROCESS_PHASE1, _PROCESS_SWEEP
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_limits
    _PROCESS_BLAS_LIMIT = threadpool_limits(limits=1, user_api="blas")
    _PROCESS_BLAS_LIMIT.__enter__()
    if worker_config is None:
        _PROCESS_DETECT_PEAKS = None
        return
    from rpe.methods.catalog import load_classical_catalog
    from rpe.perturb import load_perturbation_sweep_config
    from rpe.runner.phase1_config import load_phase1_core_config
    # Re-load frozen evaluator receipts in every isolated process before the
    # first numerical record operation; conditions themselves remain frozen.
    _PROCESS_PHASE1 = load_phase1_core_config(Path(worker_config.phase1_config_path))
    _PROCESS_SWEEP = load_perturbation_sweep_config(Path(worker_config.sweep_config_path))
    catalog = load_classical_catalog(Path(worker_config.catalog_path))
    matches = tuple(system for system in catalog.systems if system.system_id == worker_config.cwt_system_id)
    if len(matches) != 1:
        raise W1AxisCommonGridError("worker CWT system did not resolve exactly once")
    _PROCESS_DETECT_PEAKS = _peak_detector(matches[0])


def _process_materialize_conditions(record: CommonGridRecord, worker_config: CommonGridWorkerConfig) -> tuple[CommonGridCondition, ...]:
    if _PROCESS_PHASE1 is None or _PROCESS_SWEEP is None:
        raise W1AxisCommonGridError("worker frozen phase1/sweep configuration was not reconstructed")
    if record.source_adapter_id == "d2":
        from rpe.runner.phase4_d2_protocol_a import _phase1_source
        source = _phase1_source(record.source_order, record.record_id, int(record.cluster_id), record.source_spectrum)
    elif record.source_adapter_id == "d4":
        from rpe.runner import phase4_d4_eligibility as d4
        source = d4._phase1_source_for_spectrum(record.source_spectrum, order=record.source_order)
    elif record.source_adapter_id == "d5":
        from rpe.runner.phase4_d5_protocol_a import _phase1_source
        if len(record.source_adapter_metadata) != 1:
            raise W1AxisCommonGridError("d5 worker record is missing mineral-name metadata")
        source = _phase1_source(record.source_order, record.record_id, int(record.cluster_id), record.source_adapter_metadata[0], record.source_spectrum)
    else:
        raise W1AxisCommonGridError(f"unknown fused source adapter: {record.source_adapter_id}")
    return _conditions_for_source(source, phase1=_PROCESS_PHASE1, sweep=_PROCESS_SWEEP, p10_memory_budget_bytes=worker_config.p10_memory_budget_bytes)


def _worker_blas_thread_limit() -> int:
    from threadpoolctl import threadpool_info
    limits = [int(item["num_threads"]) for item in threadpool_info() if item.get("user_api") == "blas"]
    return max(limits) if limits else 1


def _evaluate_common_grid_record_process(
    payload: tuple[CommonGridRecord, CommonGridSupportSpec, str, object | None, CommonGridWorkerConfig],
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    record, support_spec, interpolator, process_start_gate, worker_config = payload
    if process_start_gate is not None:
        process_start_gate.wait(timeout=30)
    if _PROCESS_DETECT_PEAKS is None:
        raise W1AxisCommonGridError("process worker detector was not reconstructed")
    detector = _PROCESS_DETECT_PEAKS
    if is_fused_raw_common_grid_record(record):
        successes, failures = _evaluate_fused_common_grid_record(record, support_spec=support_spec, detect_peaks=detector, interpolator=interpolator, materialize_conditions=lambda raw: _process_materialize_conditions(raw, worker_config))
    else:
        successes, failures = _evaluate_common_grid_record(record, support_spec=support_spec, detect_peaks=detector, interpolator=interpolator)
    return successes, failures, os.getpid(), _worker_blas_thread_limit()


def collect_indexed_future_results(
    futures_by_index: Mapping[object, int], *, on_completed: Callable[[int], None] | None = None,
) -> tuple[object, ...]:
    """Collect completion-order futures while returning their canonical input order."""
    results: list[object | None] = [None] * len(futures_by_index)
    for completed, future in enumerate(as_completed(futures_by_index), start=1):
        results[futures_by_index[future]] = future.result()
        if on_completed is not None:
            on_completed(completed)
    return tuple(results)


def _conditions_for_source(
    source: object,
    *,
    phase1: object,
    sweep: object,
    p10_memory_budget_bytes: int,
) -> tuple[CommonGridCondition, ...]:
    from rpe.runner.phase6_appendix_metric_core import rematerialize_p8_p12_conditions

    identifiers, spectra = rematerialize_p8_p12_conditions(
        source, phase1, sweep, p10_memory_budget_bytes=p10_memory_budget_bytes
    )
    conditions: list[CommonGridCondition] = []
    for identifier, spectrum in zip(identifiers, spectra, strict=True):
        perturbation_id, alpha_hex = identifier.split(":", 1)
        alpha = float(np.frombuffer(bytes.fromhex(alpha_hex), dtype="<f8")[0])
        conditions.append(CommonGridCondition(perturbation_id, alpha, spectrum))
    if len(conditions) != 40:
        raise W1AxisCommonGridError("frozen P08-P12 condition count drift")
    return tuple(conditions)


def is_fused_raw_common_grid_record(record: CommonGridRecord) -> bool:
    return record.source_adapter_id is not None and not record.condition_spectra


def _evaluate_fused_common_grid_record(
    record: CommonGridRecord, *, support_spec: CommonGridSupportSpec,
    detect_peaks: Callable[[Spectrum1D], Sequence[object] | None], interpolator: str,
    materialize_conditions: Callable[[CommonGridRecord], tuple[CommonGridCondition, ...]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conditions = materialize_conditions(record)
    return _evaluate_common_grid_record(
        replace(record, condition_spectra=conditions), support_spec=support_spec, detect_peaks=detect_peaks, interpolator=interpolator,
    )


def _load_bacteria_dataset_input(root: Path, max_records: int | None) -> CommonGridDatasetInput:
    from rpe.methods.catalog import load_classical_catalog
    from rpe.runner.phase4_d2_protocol_a import (
        _CWT_SYSTEM_ID,
        load_phase4_d2_protocol_a_config,
        reconstruct_d2_protocol_a_inputs,
    )

    inputs = reconstruct_d2_protocol_a_inputs(
        root / "data/unified/bacteria_id_reference",
        root / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json",
        load_phase4_d2_protocol_a_config(
            root / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json"
        ),
    )
    catalog = load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json")
    systems = tuple(item for item in catalog.systems if item.system_id == _CWT_SYSTEM_ID)
    if len(systems) != 1:
        raise W1AxisCommonGridError("bacteria CWT system did not resolve exactly once")
    selected = zip(inputs.record_ids, inputs.native_test_labels, inputs.native_test_spectra, strict=True)
    if max_records is not None:
        selected = tuple(selected)[:max_records]
    records = tuple(
        CommonGridRecord(
            record_id=str(record_id), cluster_id=str(int(label)), source_spectrum=spectrum, condition_spectra=(),
            source_order=order, source_adapter_id="d2",
        )
        for order, (record_id, label, spectrum) in enumerate(selected)
    )
    return CommonGridDatasetInput(
        dataset_id="bacteria", records=records, planned_record_count=len(inputs.record_ids),
        support_spec=load_common_grid_support_specs()["bacteria"],
        detect_peaks=_peak_detector(systems[0]), sweep_sha256=hashlib.sha256((root / "experiments/shared/raman_perturbation_sweep_v1.json").read_bytes()).hexdigest(),
        frozen_authority_sha256s=_authority_hashes(root, (
            "experiments/phase1/configs/rruff_raw_core10k_v1.json", "experiments/shared/raman_perturbation_sweep_v1.json",
            "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json", "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json",
            "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json",
            "experiments/phase3/configs/classical_system_catalog_v1.json",
        )), p10_memory_budget_bytes=64 * 1024**3, worker_config=CommonGridWorkerConfig(
            str(root / "experiments/phase3/configs/classical_system_catalog_v1.json"), _CWT_SYSTEM_ID,
            str(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json"), str(root / "experiments/shared/raman_perturbation_sweep_v1.json"),
            p10_memory_budget_bytes=64 * 1024**3, source_adapter_id="d2",
        ),
    )


def _load_sugar_dataset_input(root: Path, max_records: int | None) -> CommonGridDatasetInput:
    from rpe.downstream.sugar_quantitative import load_d4_sugar_cohort
    from rpe.methods.catalog import load_classical_catalog
    from rpe.runner import phase4_d4_eligibility as d4

    cohort = load_d4_sugar_cohort(
        root / "experiments/phase05/configs/d4_sugar_protocol.json",
        root / "data/raw/ramanbench/cache/10779223/Raw data.zip",
    )
    eligibility = d4.load_phase4_d4_eligibility_config(
        root / "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
    )
    catalog = load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json")
    systems = tuple(item for item in catalog.systems if item.system_id == eligibility.cwt_system_id)
    if len(systems) != 1:
        raise W1AxisCommonGridError("sugar CWT system did not resolve exactly once")
    source_rows = zip(cohort.record_ids, cohort.well_ids, cohort.intensity, strict=True)
    if max_records is not None:
        source_rows = tuple(source_rows)[:max_records]
    axis = np.asarray(cohort.wavenumber, dtype="<f8")
    records = tuple(
        CommonGridRecord(
            record_id=str(record_id), cluster_id=str(well_id), source_spectrum=(spectrum := Spectrum1D(
                f"d4_sugar_low_snr::{record_id}", str(well_id), axis, np.asarray(intensity, dtype="<f8")
            )), condition_spectra=(), source_order=order, source_adapter_id="d4",
        )
        for order, (record_id, well_id, intensity) in enumerate(source_rows)
    )
    return CommonGridDatasetInput(
        dataset_id="sugar", records=records, planned_record_count=len(cohort.record_ids),
        support_spec=load_common_grid_support_specs()["sugar"], detect_peaks=_peak_detector(systems[0]),
        sweep_sha256=hashlib.sha256((root / "experiments/shared/raman_perturbation_sweep_v1.json").read_bytes()).hexdigest(),
        frozen_authority_sha256s=_authority_hashes(root, (
            "experiments/phase1/configs/rruff_raw_core10k_v1.json", "experiments/shared/raman_perturbation_sweep_v1.json",
            "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json", "experiments/phase05/configs/d4_sugar_protocol.json",
            "experiments/phase3/configs/classical_system_catalog_v1.json",
        )), p10_memory_budget_bytes=eligibility.p10_memory_budget_bytes, worker_config=CommonGridWorkerConfig(
            str(root / "experiments/phase3/configs/classical_system_catalog_v1.json"), eligibility.cwt_system_id,
            str(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json"), str(root / "experiments/shared/raman_perturbation_sweep_v1.json"),
            p10_memory_budget_bytes=eligibility.p10_memory_budget_bytes, source_adapter_id="d4",
        ),
    )


def _load_d5_dataset_input(root: Path, max_records: int | None) -> CommonGridDatasetInput:
    from collections import Counter
    from rpe.downstream.rruff import load_d5_native_spectra, load_d5_raw_cohort
    from rpe.methods.catalog import load_classical_catalog
    from rpe.runner.phase4_d5_protocol_a import _resolve_cwt

    cohort = load_d5_raw_cohort(root / "experiments/phase05/configs/d5_rruff_protocol.json", root / "data/unified/rruff_raman_raw")
    occurrences = Counter(int(index) for split in cohort.splits for index in split.query_indices)
    indices = tuple(sorted(occurrences))
    if max_records is not None:
        indices = indices[:max_records]
    spectra = load_d5_native_spectra(root / "data/unified/rruff_raman_raw", tuple(cohort.record_ids[index] for index in indices))
    cwt = _resolve_cwt(load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json"))
    records = tuple(
        CommonGridRecord(
            record_id=str(cohort.record_ids[index]), cluster_id=str(int(cohort.class_labels[index])),
            source_spectrum=spectrum, aggregation_weight=int(occurrences[index]), condition_spectra=(),
            source_order=order, source_adapter_id="d5", source_adapter_metadata=(str(cohort.mineral_names[index]),),
        )
        for order, (index, spectrum) in enumerate(zip(indices, spectra, strict=True))
    )
    return CommonGridDatasetInput(
        dataset_id="d5", records=records, planned_record_count=len(occurrences),
        support_spec=load_common_grid_support_specs()["d5"], detect_peaks=_peak_detector(cwt),
        sweep_sha256=hashlib.sha256((root / "experiments/shared/raman_perturbation_sweep_v1.json").read_bytes()).hexdigest(),
        frozen_authority_sha256s=_authority_hashes(root, (
            "experiments/phase1/configs/rruff_raw_core10k_v1.json", "experiments/shared/raman_perturbation_sweep_v1.json",
            "experiments/phase4/configs/d5_eligibility_v1.json", "experiments/phase05/configs/d5_rruff_protocol.json",
            "experiments/phase3/configs/classical_system_catalog_v1.json",
        )), p10_memory_budget_bytes=64 * 1024**3, worker_config=CommonGridWorkerConfig(
            str(root / "experiments/phase3/configs/classical_system_catalog_v1.json"), cwt.system_id,
            str(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json"), str(root / "experiments/shared/raman_perturbation_sweep_v1.json"),
            p10_memory_budget_bytes=64 * 1024**3, source_adapter_id="d5",
        ),
    )


def materialize_real_common_grid_records(
    *, root: Path = ROOT, max_records_per_dataset: int | None = None
) -> Mapping[str, CommonGridDatasetInput]:
    """Load real retained sources and regenerate all frozen P08-P12 cells once."""
    project_root = Path(root)
    return {
        "bacteria": _load_bacteria_dataset_input(project_root, max_records_per_dataset),
        "sugar": _load_sugar_dataset_input(project_root, max_records_per_dataset),
        "d5": _load_d5_dataset_input(project_root, max_records_per_dataset),
    }


def materialize_real_common_grid_dataset(
    dataset_id: str, *, root: Path = ROOT, max_records_per_dataset: int | None = None,
) -> CommonGridDatasetInput:
    """Load one raw-source dataset only; fused workers materialize P08--P12."""
    loaders = {
        "bacteria": _load_bacteria_dataset_input,
        "sugar": _load_sugar_dataset_input,
        "d5": _load_d5_dataset_input,
    }
    try:
        loader = loaders[dataset_id]
    except KeyError as error:
        raise W1AxisCommonGridError(f"unknown common-grid dataset: {dataset_id}") from error
    return loader(Path(root), max_records_per_dataset)


def evaluate_real_common_grid_dataset(
    dataset: CommonGridDatasetInput, *, worker_count: int = 1, progress_log_path: Path | None = None,
    progress_started_at: float | None = None,
) -> CommonGridDatasetResult:
    started = time.monotonic()
    result = evaluate_common_grid_records(
        dataset.records, support_spec=dataset.support_spec, detect_peaks=dataset.detect_peaks, interpolator="linear",
        worker_count=worker_count, p10_memory_budget_bytes=dataset.p10_memory_budget_bytes, worker_config=dataset.worker_config,
        progress_log_path=progress_log_path, progress_started_at=progress_started_at,
    )
    is_canary = result.processed_record_count < dataset.planned_record_count
    return CommonGridDatasetResult(
        dataset_id=dataset.dataset_id, aggregated_rows=result.aggregated_rows, failure_rows=result.failure_rows,
        processed_record_count=result.processed_record_count, planned_record_count=dataset.planned_record_count,
        status="canary_partial" if is_canary else ("complete" if not result.failure_rows else "incomplete_grid"),
        elapsed_seconds=time.monotonic() - started, cache_hit_count=0,
        effective_worker_count=result.effective_worker_count,
        p10_worker_capacity=result.p10_worker_capacity,
        requested_worker_count=result.requested_worker_count,
        worker_process_ids=result.worker_process_ids,
        worker_blas_thread_limits=result.worker_blas_thread_limits,
        cache_hit=False,
    )


def common_grid_dataset_source_sha256(dataset: CommonGridDatasetInput) -> str:
    digest = hashlib.sha256()
    for record in dataset.records:
        digest.update(str(record.record_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.cluster_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(record.aggregation_weight)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record.source_order).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record.source_adapter_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_canonical_json_bytes(record.source_adapter_metadata))
        digest.update(np.ascontiguousarray(record.source_spectrum.axis_cm1, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(record.source_spectrum.intensity, dtype="<f8").tobytes())
    return digest.hexdigest()


def common_grid_dataset_cache_identity(
    dataset: CommonGridDatasetInput, code_sha256: str
) -> dict[str, object]:
    frozen_authorities = dict(sorted(dataset.frozen_authority_sha256s.items()))
    record_provenance_sha256 = common_grid_dataset_source_sha256(dataset)
    frozen_sha = hashlib.sha256(_canonical_json_bytes(frozen_authorities)).hexdigest()
    identity: dict[str, object] = build_common_grid_cache_identity(
        panel_id=dataset.dataset_id, source_spectrum_sha256=common_grid_dataset_source_sha256(dataset),
        sweep_sha256=dataset.sweep_sha256, representation_sha256=dataset.support_spec.axis_sha256_f64,
        code_sha256=hashlib.sha256(
            _canonical_json_bytes({
                "common_grid_code_sha256": code_sha256,
                "frozen_authorities_sha256": frozen_sha,
                "p10_memory_budget_bytes": dataset.p10_memory_budget_bytes,
            })
        ).hexdigest(),
    )
    # Keep the individual authority and record-provenance receipts visible in
    # the on-disk identity, rather than making a later audit reverse-engineer
    # them from the derived code digest alone.
    identity.update({
        "record_provenance_sha256": record_provenance_sha256,
        "frozen_authority_sha256s": frozen_authorities,
        "p10_memory_budget_bytes": int(dataset.p10_memory_budget_bytes),
    })
    return identity


def evaluate_or_load_real_common_grid_dataset(
    dataset: CommonGridDatasetInput, *, cache_root: Path, resume: bool, code_sha256: str, worker_count: int = 1,
    progress_started_at: float | None = None,
) -> CommonGridDatasetResult:
    identity = common_grid_dataset_cache_identity(dataset, code_sha256)
    def build_payload() -> Mapping[str, object]:
        progress_root = Path(tempfile.gettempdir()) / "w1-axis-common-grid-progress" / dataset.dataset_id
        value = evaluate_real_common_grid_dataset(
            dataset, worker_count=worker_count, progress_log_path=progress_root / f"{cache_key}.jsonl",
            progress_started_at=progress_started_at,
        )
        return {
            "dataset_id": value.dataset_id, "aggregated_rows": list(value.aggregated_rows),
            "failure_rows": list(value.failure_rows), "processed_record_count": value.processed_record_count,
            "planned_record_count": value.planned_record_count, "status": value.status,
            "elapsed_seconds": value.elapsed_seconds,
            "effective_worker_count": value.effective_worker_count,
            "p10_worker_capacity": value.p10_worker_capacity,
            "requested_worker_count": value.requested_worker_count,
            "worker_process_ids": list(value.worker_process_ids),
            "worker_blas_thread_limits": list(value.worker_blas_thread_limits),
        }
    # Cache directories are keyed by the complete identity.  This lets a
    # canary and a later full dataset coexist while resume still rejects a
    # tampered identity inside its own directory.
    cache_key = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
    payload, cache_hit = load_or_build_common_grid_cache(
        Path(cache_root) / cache_key, identity=identity, resume=resume, build_payload=build_payload
    )
    return CommonGridDatasetResult(
        dataset_id=str(payload["dataset_id"]),
        aggregated_rows=tuple(payload["aggregated_rows"]), failure_rows=tuple(payload["failure_rows"]),
        processed_record_count=int(payload["processed_record_count"]),
        planned_record_count=int(payload["planned_record_count"]), status=str(payload["status"]),
        elapsed_seconds=float(payload["elapsed_seconds"]), cache_hit_count=1 if cache_hit else 0,
        effective_worker_count=int(payload.get("effective_worker_count", 1)),
        p10_worker_capacity=int(payload.get("p10_worker_capacity", 1)),
        requested_worker_count=int(payload.get("requested_worker_count", 1)),
        worker_process_ids=tuple(int(value) for value in payload.get("worker_process_ids", ())),
        worker_blas_thread_limits=tuple(int(value) for value in payload.get("worker_blas_thread_limits", ())),
        cache_hit=bool(cache_hit),
    )


def _common_grid_view(spectrum: Spectrum1D, axis_cm1: np.ndarray, intensity: np.ndarray) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=f"{spectrum.spectrum_id}:common_grid",
        sample_id=spectrum.sample_id,
        axis_cm1=np.ascontiguousarray(axis_cm1, dtype="<f8"),
        intensity=np.ascontiguousarray(intensity, dtype="<f8"),
    )


def resample_pair_to_common_grid(
    source: Spectrum1D,
    candidate: Spectrum1D,
    support_axis: np.ndarray,
    *,
    interpolator: str,
    max_in_range_native_gap_cm1: float | None,
) -> tuple[Spectrum1D, Spectrum1D]:
    axis = np.ascontiguousarray(np.asarray(support_axis, dtype="<f8"))
    try:
        source_values = resample_spectrum(
            source.axis_cm1,
            source.intensity,
            axis,
            interpolator=interpolator,
            max_in_range_native_gap_cm1=max_in_range_native_gap_cm1,
        )
        candidate_values = resample_spectrum(
            candidate.axis_cm1,
            candidate.intensity,
            axis,
            interpolator=interpolator,
            max_in_range_native_gap_cm1=max_in_range_native_gap_cm1,
        )
    except AppendixAuditsError as error:
        raise W1AxisCommonGridError(str(error)) from error
    return (
        _common_grid_view(source, axis, source_values),
        _common_grid_view(candidate, axis, candidate_values),
    )


def _metric_scalar(result, output_id: str) -> float:
    rows = [row for row in result.outputs if getattr(row, "output_id", None) == output_id]
    if len(rows) != 1:
        raise W1AxisCommonGridError(f"metric output missing: {output_id}")
    value = float(rows[0].value)
    if not np.isfinite(value):
        raise W1AxisCommonGridError(f"metric output non-finite: {output_id}")
    return value


_SCALAR_METRICS = (
    ("mse", MSEMetric, False),
    ("rmse", RMSEMetric, False),
    ("mae", MAEMetric, False),
    ("sam", SAMMetric, False),
    ("pearson_r", PearsonRMetric, False),
    ("nmse", NMSEMetric, False),
    ("wasserstein_1_cm1", Wasserstein1Metric, False),
    ("is_like_structure_to_noise", ISLikeStructureToNoiseMetric, True),
)
_PEAK_METRIC_IDS = METRIC_OUTPUT_IDS[8:]


def _failure_reason(scope: str, metric_output_id: str, error: Exception) -> str:
    return f"{scope}:{metric_output_id}:{type(error).__name__}:{error}"


def _evaluate_scalar_metrics(
    reference: Spectrum1D, candidate: Spectrum1D,
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    failures: dict[str, str] = {}
    pair_request = SpectrumPairInput(reference, candidate)
    for metric_output_id, metric_type, is_single_spectrum in _SCALAR_METRICS:
        try:
            request = SingleSpectrumInput(candidate) if is_single_spectrum else pair_request
            values[metric_output_id] = _metric_scalar(
                evaluate_metric(metric_type(), request), metric_output_id
            )
        except Exception as error:
            failures[metric_output_id] = f"{type(error).__name__}:{error}"
    return values, failures


def _evaluate_peak_metrics(
    reference_peaks: tuple[Peak1D, ...], candidate_peaks: tuple[Peak1D, ...],
) -> tuple[dict[str, float], dict[str, str]]:
    try:
        peak_result = evaluate_metric(
            PeakDetectionCurvesMetric(),
            PeakPairInput(reference_peaks, candidate_peaks, 2.0, (0.0,)),
        )
        return (
            {metric_id: _metric_scalar(peak_result, metric_id) for metric_id in _PEAK_METRIC_IDS},
            {},
        )
    except Exception as error:
        reason = f"{type(error).__name__}:{error}"
        return {}, {metric_id: reason for metric_id in _PEAK_METRIC_IDS}


def _normalize_peaks(peaks: Sequence[object] | None) -> tuple[Peak1D, ...]:
    if peaks is None:
        return ()
    result: list[Peak1D] = []
    for peak in peaks:
        item = peak.to_peak1d() if hasattr(peak, "to_peak1d") else peak
        if not isinstance(item, Peak1D):
            raise W1AxisCommonGridError("detect_peaks must yield Peak1D values")
        result.append(item)
    return tuple(result)


def compute_common_grid_metric_vector(
    reference: Spectrum1D,
    candidate: Spectrum1D,
    *,
    detect_peaks: Callable[[Spectrum1D], Sequence[object] | None],
    reference_peaks: tuple[Peak1D, ...] | None = None,
    candidate_peaks: tuple[Peak1D, ...] | None = None,
) -> dict[str, float]:
    values, scalar_failures = _evaluate_scalar_metrics(reference, candidate)
    if scalar_failures:
        metric_output_id, reason = next(iter(scalar_failures.items()))
        raise W1AxisCommonGridError(f"metric failure: {metric_output_id}: {reason}")
    reference_peaks = _normalize_peaks(detect_peaks(reference)) if reference_peaks is None else reference_peaks
    candidate_peaks = _normalize_peaks(detect_peaks(candidate)) if candidate_peaks is None else candidate_peaks
    peak_values, peak_failures = _evaluate_peak_metrics(reference_peaks, candidate_peaks)
    if peak_failures:
        metric_output_id, reason = next(iter(peak_failures.items()))
        raise W1AxisCommonGridError(f"metric failure: {metric_output_id}: {reason}")
    values.update(peak_values)
    return {metric_id: float(values[metric_id]) for metric_id in METRIC_OUTPUT_IDS}


def orient_metric_harms_from_alpha_zero(
    current_values: Mapping[str, float],
    baseline_values: Mapping[str, float],
) -> dict[str, float]:
    harms: dict[str, float] = {}
    for metric_id in METRIC_OUTPUT_IDS:
        try:
            current = float(current_values[metric_id])
            baseline = float(baseline_values[metric_id])
        except (KeyError, TypeError, ValueError) as error:
            raise W1AxisCommonGridError(f"missing metric value: {metric_id}") from error
        harm = current - baseline if metric_id in _LOWER_IS_BETTER else baseline - current
        if not np.isfinite(harm):
            raise W1AxisCommonGridError(f"non-finite harm: {metric_id}")
        harms[metric_id] = harm
    return harms


def _p10_worker_capacity(records: Sequence[CommonGridRecord], budget: int | None) -> int:
    if budget is None or budget <= 0 or not records:
        return max(1, len(records))
    largest = max(int(record.source_spectrum.axis_cm1.size) for record in records)
    estimated_peak = 32 * largest * largest + 64 * largest + 2**30
    return max(1, int(budget) // estimated_peak)


def _evaluate_common_grid_record(
    record: CommonGridRecord, *, support_spec: CommonGridSupportSpec,
    detect_peaks: Callable[[Spectrum1D], Sequence[object] | None], interpolator: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    success_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    def failure_row(condition: CommonGridCondition, metric_output_id: str, reason_code: str) -> dict[str, object]:
        return {
            "record_id": record.record_id, "cluster_id": record.cluster_id,
            "perturbation_id": condition.perturbation_id, "alpha": float(condition.alpha),
            "metric_output_id": metric_output_id, "reason_code": reason_code,
        }

    try:
        source_values = resample_spectrum(
            record.source_spectrum.axis_cm1, record.source_spectrum.intensity, support_spec.axis_cm1,
            interpolator=interpolator, max_in_range_native_gap_cm1=support_spec.max_in_range_native_gap_cm1,
        )
        baseline_source = _common_grid_view(record.source_spectrum, support_spec.axis_cm1, source_values)
    except Exception as error:
        reason = f"source_resample_failure:{type(error).__name__}:{error}"
        for condition in record.condition_spectra:
            failure_rows.extend(
                failure_row(condition, metric_output_id, reason)
                for metric_output_id in METRIC_OUTPUT_IDS
            )
        return success_rows, failure_rows

    baseline_values, baseline_failures = _evaluate_scalar_metrics(baseline_source, baseline_source)
    try:
        source_peaks = _normalize_peaks(detect_peaks(baseline_source))
    except Exception as error:
        source_peaks = ()
        baseline_failures.update({
            metric_output_id: f"{type(error).__name__}:{error}"
            for metric_output_id in _PEAK_METRIC_IDS
        })
    else:
        baseline_peak_values, baseline_peak_failures = _evaluate_peak_metrics(source_peaks, source_peaks)
        baseline_values.update(baseline_peak_values)
        baseline_failures.update(baseline_peak_failures)

    for condition in record.condition_spectra:
        try:
            candidate_values = resample_spectrum(
                condition.spectrum.axis_cm1, condition.spectrum.intensity, support_spec.axis_cm1,
                interpolator=interpolator, max_in_range_native_gap_cm1=support_spec.max_in_range_native_gap_cm1,
            )
            resampled_candidate = _common_grid_view(condition.spectrum, support_spec.axis_cm1, candidate_values)
        except Exception as error:
            reason = f"candidate_resample_failure:{type(error).__name__}:{error}"
            failure_rows.extend(
                failure_row(condition, metric_output_id, reason)
                for metric_output_id in METRIC_OUTPUT_IDS
            )
            continue

        current_values, current_failures = _evaluate_scalar_metrics(baseline_source, resampled_candidate)
        if any(metric_output_id not in baseline_failures for metric_output_id in _PEAK_METRIC_IDS):
            try:
                candidate_peaks = _normalize_peaks(detect_peaks(resampled_candidate))
            except Exception as error:
                current_failures.update({
                    metric_output_id: f"{type(error).__name__}:{error}"
                    for metric_output_id in _PEAK_METRIC_IDS
                })
            else:
                current_peak_values, current_peak_failures = _evaluate_peak_metrics(source_peaks, candidate_peaks)
                current_values.update(current_peak_values)
                current_failures.update(current_peak_failures)

        for metric_output_id in METRIC_OUTPUT_IDS:
            if metric_output_id in baseline_failures:
                failure_rows.append(failure_row(
                    condition, metric_output_id,
                    f"baseline_metric_failure:{metric_output_id}:{baseline_failures[metric_output_id]}",
                ))
                continue
            if metric_output_id in current_failures:
                failure_rows.append(failure_row(
                    condition, metric_output_id,
                    f"metric_failure:{metric_output_id}:{current_failures[metric_output_id]}",
                ))
                continue
            metric_harm = (
                current_values[metric_output_id] - baseline_values[metric_output_id]
                if metric_output_id in _LOWER_IS_BETTER
                else baseline_values[metric_output_id] - current_values[metric_output_id]
            )
            if not np.isfinite(metric_harm):
                failure_rows.append(failure_row(
                    condition, metric_output_id,
                    f"metric_failure:{metric_output_id}:W1AxisCommonGridError:non-finite harm",
                ))
                continue
            success_rows.append({
                "record_id": record.record_id, "cluster_id": record.cluster_id,
                "perturbation_id": condition.perturbation_id, "alpha": float(condition.alpha),
                "metric_output_id": metric_output_id, "metric_harm": float(metric_harm),
                "aggregation_weight": int(record.aggregation_weight),
            })
    return success_rows, failure_rows


def evaluate_common_grid_records(
    records: Sequence[CommonGridRecord],
    *,
    support_spec: CommonGridSupportSpec,
    detect_peaks: Callable[[Spectrum1D], Sequence[object] | None],
    interpolator: str,
    worker_count: int = 1,
    p10_memory_budget_bytes: int | None = None,
    worker_config: CommonGridWorkerConfig | None = None,
    progress_log_path: Path | None = None,
    progress_started_at: float | None = None,
) -> CommonGridEvaluationResult:
    if worker_count < 1:
        raise W1AxisCommonGridError("worker_count must be positive")
    requested_worker_count = min(worker_count, max(1, len(records)))
    p10_capacity = _p10_worker_capacity(records, p10_memory_budget_bytes)
    effective_worker_count = min(requested_worker_count, p10_capacity)
    if effective_worker_count > 1 and worker_config is None:
        raise W1AxisCommonGridError(
            "worker_count>1 requires reconstructible worker_config; caller detect_peaks cannot be serialized safely"
        )
    progress_path = None if progress_log_path is None else Path(progress_log_path)

    def write_progress(event: str, completed: int) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "event": event, "completed_record_count": int(completed),
            "planned_record_count": len(records), "elapsed_seconds": time.monotonic() - started,
            "requested_worker_count": int(worker_count), "effective_worker_count": int(effective_worker_count),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()

    started = time.monotonic() if progress_started_at is None else float(progress_started_at)
    write_progress("dataset_start", 0)
    worker_process_ids: tuple[int, ...] = ()
    worker_blas_thread_limits: tuple[int, ...] = ()
    if effective_worker_count == 1:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            if any(is_fused_raw_common_grid_record(record) for record in records):
                if worker_config is None:
                    raise W1AxisCommonGridError("fused raw record requires worker_config")
                _worker_initializer(worker_config)
            record_results_list = []
            for completed, record in enumerate(records, start=1):
                if is_fused_raw_common_grid_record(record):
                    record_results_list.append(_evaluate_fused_common_grid_record(
                        record, support_spec=support_spec, detect_peaks=detect_peaks, interpolator=interpolator,
                        materialize_conditions=lambda raw: _process_materialize_conditions(raw, worker_config),
                    ))
                else:
                    record_results_list.append(_evaluate_common_grid_record(
                        record, support_spec=support_spec, detect_peaks=detect_peaks, interpolator=interpolator,
                    ))
                if completed % 100 == 0 or completed == len(records):
                    write_progress("checkpoint", completed)
            record_results = tuple(record_results_list)
    else:
        payloads = tuple((record, support_spec, interpolator, worker_config.process_start_gate, worker_config) for record in records)
        # Completion-order collection keeps progress live; index assembly keeps
        # all downstream rows deterministic in canonical input order.
        with ProcessPoolExecutor(
            max_workers=effective_worker_count, initializer=_worker_initializer, initargs=(worker_config,),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            future_index = {
                executor.submit(_evaluate_common_grid_record_process, payload): index
                for index, payload in enumerate(payloads)
            }
            process_results = collect_indexed_future_results(
                future_index,
                on_completed=lambda completed: write_progress("checkpoint", completed)
                if completed % 100 == 0 or completed == len(records) else None,
            )
        record_results = tuple((successes, failures) for successes, failures, _, _ in process_results)
        worker_process_ids = tuple(sorted({pid for _, _, pid, _ in process_results}))
        worker_blas_thread_limits = tuple(sorted({limit for _, _, _, limit in process_results}))
    success_rows = [row for successes, _ in record_results for row in successes]
    failure_rows = [row for _, failures in record_results for row in failures]
    aggregated = aggregate_record_metric_rows(
        success_rows,
        cluster_field="cluster_id",
        weight_field="aggregation_weight",
    )
    write_progress("dataset_end", len(records))
    return CommonGridEvaluationResult(
        aggregated_rows=aggregated,
        failure_rows=tuple(failure_rows),
        processed_record_count=len(records),
        effective_worker_count=effective_worker_count,
        p10_worker_capacity=p10_capacity,
        worker_process_ids=worker_process_ids,
        worker_blas_thread_limits=worker_blas_thread_limits,
        requested_worker_count=requested_worker_count,
    )


def aggregate_record_metric_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    cluster_field: str,
    weight_field: str | None = None,
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, float, str], dict[str, float | int | str]] = {}
    for row in rows:
        try:
            cluster_value = str(row[cluster_field])
            perturbation_id = str(row["perturbation_id"])
            alpha = float(row["alpha"])
            metric_output_id = str(row["metric_output_id"])
            metric_harm = float(row["metric_harm"])
        except (KeyError, TypeError, ValueError) as error:
            raise W1AxisCommonGridError("invalid record metric row") from error
        if not np.isfinite(metric_harm):
            raise W1AxisCommonGridError("metric_harm must be finite")
        if weight_field is None:
            weight = 1
            within_cluster_count = 1
        else:
            try:
                weight = int(row[weight_field])
            except (KeyError, TypeError, ValueError) as error:
                raise W1AxisCommonGridError("invalid aggregation weight") from error
            if weight <= 0:
                raise W1AxisCommonGridError("aggregation weights must be positive")
            within_cluster_count = weight
        key = (cluster_value, perturbation_id, alpha, metric_output_id)
        bucket = grouped.setdefault(
            key,
            {
                cluster_field: cluster_value,
                "perturbation_id": perturbation_id,
                "alpha": alpha,
                "metric_output_id": metric_output_id,
                "weighted_sum": 0.0,
                "total_weight": 0,
                "within_cluster_count": 0,
            },
        )
        bucket["weighted_sum"] = float(bucket["weighted_sum"]) + metric_harm * weight
        bucket["total_weight"] = int(bucket["total_weight"]) + weight
        bucket["within_cluster_count"] = int(bucket["within_cluster_count"]) + within_cluster_count
    aggregated: list[dict[str, object]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        total_weight = int(bucket["total_weight"])
        aggregated.append(
            {
                cluster_field: bucket[cluster_field],
                "perturbation_id": bucket["perturbation_id"],
                "alpha": float(bucket["alpha"]),
                "metric_output_id": bucket["metric_output_id"],
                "metric_harm": float(bucket["weighted_sum"]) / total_weight,
                "within_cluster_count": int(bucket["within_cluster_count"]),
            }
        )
    return tuple(aggregated)


def replace_metric_harm(parent_table: NativePanelTable, replacement_metric_harm: np.ndarray) -> NativePanelTable:
    return replace(
        parent_table,
        metric_harm=np.ascontiguousarray(replacement_metric_harm, dtype="<f8"),
        common_grid_ready=True,
    )


def _row_cluster_id(row: Mapping[str, object]) -> str:
    for field in ("cluster_id", "class_label", "well_id"):
        value = row.get(field)
        if value is not None:
            return str(value)
    raise W1AxisCommonGridError("aggregated row missing cluster identifier")


def finalize_common_grid_panel_result(
    *,
    panel_id: str,
    parent_table: NativePanelTable,
    aggregated_rows: Sequence[Mapping[str, object]],
    failure_rows: Sequence[Mapping[str, object]],
) -> CommonGridPanelResult:
    failure_tuple = tuple(failure_rows)
    metric_index = {value: index for index, value in enumerate(parent_table.metric_output_ids)}
    cluster_index = {value: index for index, value in enumerate(parent_table.cluster_ids)}
    perturbation_index = {value: index for index, value in enumerate(parent_table.perturbation_ids)}
    alpha_index = {float(value).hex(): index for index, value in enumerate(parent_table.alpha_grid)}
    expected_per_metric = (
        len(parent_table.cluster_ids)
        * len(parent_table.perturbation_ids)
        * len(parent_table.alpha_grid)
    )
    failed_metric_ids: set[str] = set()
    for failure in failure_tuple:
        metric_output_id = failure.get("metric_output_id")
        if metric_output_id is None:
            failed_metric_ids.update(metric_index)
        elif str(metric_output_id) in metric_index:
            failed_metric_ids.add(str(metric_output_id))
        else:
            raise W1AxisCommonGridError(f"{panel_id}: failure outside frozen metric grid")

    replacement = np.empty_like(parent_table.metric_harm)
    seen: set[tuple[str, str, str, str]] = set()
    for row in aggregated_rows:
        cluster_id = _row_cluster_id(row)
        try:
            metric_output_id = str(row["metric_output_id"])
            perturbation_id = str(row["perturbation_id"])
            alpha_key = float(row["alpha"]).hex()
            metric_harm = float(row["metric_harm"])
        except (KeyError, TypeError, ValueError) as error:
            raise W1AxisCommonGridError(f"{panel_id}: invalid aggregated row") from error
        key = (cluster_id, metric_output_id, perturbation_id, alpha_key)
        if key in seen:
            raise W1AxisCommonGridError(f"{panel_id}: duplicate aggregated row")
        if (
            cluster_id not in cluster_index
            or metric_output_id not in metric_index
            or perturbation_id not in perturbation_index
            or alpha_key not in alpha_index
        ):
            raise W1AxisCommonGridError(f"{panel_id}: aggregated row outside frozen grid")
        replacement[
            metric_index[metric_output_id],
            cluster_index[cluster_id],
            perturbation_index[perturbation_id],
            alpha_index[alpha_key],
        ] = metric_harm
        seen.add(key)
    complete_metric_output_ids = tuple(
        metric_output_id
        for metric_output_id in parent_table.metric_output_ids
        if metric_output_id not in failed_metric_ids
        and sum(key[1] == metric_output_id for key in seen) == expected_per_metric
    )
    incomplete_metric_output_ids = tuple(
        metric_output_id
        for metric_output_id in parent_table.metric_output_ids
        if metric_output_id not in complete_metric_output_ids
    )
    if not complete_metric_output_ids:
        return CommonGridPanelResult(
            status="incomplete_grid",
            table=None,
            failure_rows=failure_tuple,
            complete_metric_output_ids=(),
            incomplete_metric_output_ids=incomplete_metric_output_ids,
        )
    selected_indices = [metric_index[metric_output_id] for metric_output_id in complete_metric_output_ids]
    table = replace(
        parent_table,
        metric_output_ids=complete_metric_output_ids,
        metric_harm=np.ascontiguousarray(replacement[selected_indices, :, :, :], dtype="<f8"),
    )
    return CommonGridPanelResult(
        status="complete" if not incomplete_metric_output_ids else "partial_metric_grid",
        table=table,
        failure_rows=failure_tuple,
        complete_metric_output_ids=complete_metric_output_ids,
        incomplete_metric_output_ids=incomplete_metric_output_ids,
    )


def build_common_grid_cache_identity(
    *,
    panel_id: str,
    source_spectrum_sha256: str,
    sweep_sha256: str,
    representation_sha256: str,
    code_sha256: str,
) -> dict[str, str]:
    identity = {
        "panel_id": str(panel_id),
        "source_spectrum_sha256": str(source_spectrum_sha256),
        "sweep_sha256": str(sweep_sha256),
        "representation_sha256": str(representation_sha256),
        "code_sha256": str(code_sha256),
    }
    if any(len(value) == 0 for value in identity.values()):
        raise W1AxisCommonGridError("cache identity fields must be non-empty")
    return identity


def write_complete_common_grid_cache(
    cache_root: Path,
    *,
    identity: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    root = Path(cache_root)
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        raise W1AxisCommonGridError("cache root already exists")
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(parent)))
    staging_root = staging_parent / root.name
    staging_root.mkdir()
    try:
        identity_bytes = _canonical_json_bytes(identity)
        payload_bytes = _canonical_json_bytes(payload)
        completion = {
            "schema_version": "w1-axis-common-grid-cache-v1",
            "identity": dict(identity),
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "status": "complete",
        }
        (staging_root / "identity.json").write_bytes(identity_bytes)
        (staging_root / "payload.json").write_bytes(payload_bytes)
        (staging_root / "complete.json").write_bytes(_canonical_json_bytes(completion))
        staging_root.rename(root)
    except Exception:
        if staging_parent.exists():
            for child in sorted(staging_parent.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            if staging_parent.exists():
                staging_parent.rmdir()
        raise
    if staging_parent.exists():
        staging_parent.rmdir()


def load_complete_common_grid_cache(
    cache_root: Path,
    *,
    expected_identity: Mapping[str, object],
) -> dict[str, object]:
    root = Path(cache_root)
    try:
        identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        payload = json.loads((root / "payload.json").read_text(encoding="utf-8"))
        completion = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W1AxisCommonGridError("cannot read complete common-grid cache") from error
    if dict(identity) != dict(expected_identity) or dict(completion.get("identity", {})) != dict(expected_identity):
        raise W1AxisCommonGridError("cache identity mismatch")
    payload_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if completion.get("payload_sha256") != payload_sha256:
        raise W1AxisCommonGridError("cache payload digest mismatch")
    if not isinstance(payload, dict):
        raise W1AxisCommonGridError("cache payload must be a JSON object")
    return payload


def load_or_build_common_grid_cache(
    cache_root: Path,
    *,
    identity: Mapping[str, object],
    resume: bool,
    build_payload: Callable[[], Mapping[str, object]],
) -> tuple[dict[str, object], bool]:
    root = Path(cache_root)
    if resume and root.is_dir():
        return load_complete_common_grid_cache(root, expected_identity=identity), True
    payload = dict(build_payload())
    write_complete_common_grid_cache(root, identity=identity, payload=payload)
    return payload, False


__all__ = [
    "CommonGridCondition",
    "CommonGridDatasetInput",
    "CommonGridDatasetResult",
    "CommonGridEvaluationResult",
    "CommonGridPanelResult",
    "CommonGridRecord",
    "CommonGridSupportSpec",
    "METRIC_OUTPUT_IDS",
    "W1AxisCommonGridError",
    "aggregate_record_metric_rows",
    "build_common_grid_cache_identity",
    "compute_common_grid_metric_vector",
    "common_grid_dataset_source_sha256",
    "evaluate_or_load_real_common_grid_dataset",
    "evaluate_real_common_grid_dataset",
    "evaluate_common_grid_records",
    "finalize_common_grid_panel_result",
    "load_common_grid_support_specs",
    "load_complete_common_grid_cache",
    "load_or_build_common_grid_cache",
    "materialize_real_common_grid_records",
    "orient_metric_harms_from_alpha_zero",
    "replace_metric_harm",
    "resample_pair_to_common_grid",
    "write_complete_common_grid_cache",
]
