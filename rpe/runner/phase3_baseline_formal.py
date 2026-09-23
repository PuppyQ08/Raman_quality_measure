from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

from rpe.methods import Phase3ClassicalCatalog, Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.baseline import BaselineRunStatus, run_baseline_system

if TYPE_CHECKING:
    from rpe.methods.classical.baseline_canary import BaselineCanarySource


CONFIG_BYTE_COUNT = 1717
CONFIG_SHA256 = "0c3d31f12d5169cda9f92b1aef6569638b3055d216c7bb173e3ca519a435cdee"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUN_DOMAIN = b"rpe-phase3-baseline-formal10k-v1\0"
_SUCCESS_STATUSES = (BaselineRunStatus.COMPLETE, BaselineRunStatus.COMPLETE_WITH_WARNING)
_TERMINAL_STATUSES = tuple(BaselineRunStatus)
_WORKER_SYSTEMS: tuple[Phase3System, ...] = ()


class BaselineFormalError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class CatalogIdentity(ArtifactIdentity):
    catalog_id: str


@dataclass(frozen=True)
class BaselineCoveragePolicy:
    minimum_successful_fraction: float
    minimum_phase5_systems: int
    minimum_phase5_families: int
    minimum_promoted_per_family: int
    minimum_common_record_fraction: float
    require_zero_failed_runtime: bool
    require_zero_not_applicable: bool


@dataclass(frozen=True)
class Phase3BaselineFormalConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    catalog: CatalogIdentity
    expected_source_count: int
    expected_system_count: int
    expected_attempt_count: int
    worker_processes: int
    blas_threads_per_process: int
    shard_source_count: int
    storage_mode: str
    success_statuses: tuple[BaselineRunStatus, ...]
    policy: BaselineCoveragePolicy
    phase1: Mapping[str, object]
    claim_boundary: str


@dataclass(frozen=True)
class BaselineFormalReceipt:
    selection_rank: int
    record_id: str
    system_id: str
    family_id: str
    method_id: str
    status: BaselineRunStatus
    baseline_sha256: str | None
    corrected_sha256: str | None
    warnings: tuple[object, ...]
    diagnostics: Mapping[str, object]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, BaselineRunStatus):
            raise BaselineFormalError("receipt.status", "must be BaselineRunStatus")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(sorted(self.diagnostics.items()))))


@dataclass(frozen=True)
class SystemCoverageSummary:
    system_id: str
    family_id: str
    terminal_count: int
    successful_count: int
    successful_fraction: float
    status_counts: Mapping[str, int]
    coverage_promoted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_counts", MappingProxyType(dict(sorted(self.status_counts.items()))))


@dataclass(frozen=True)
class BaselineCoverageResult:
    system_summaries: tuple[SystemCoverageSummary, ...]
    promoted_system_ids: tuple[str, ...]
    qualifying_family_ids: tuple[str, ...]
    phase5_eligible_system_ids: tuple[str, ...]
    phase5_eligible_system_count: int
    common_successful_record_count: int
    common_record_fraction: float
    gate_passed: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class BaselineFormalSummary:
    path: Path
    run_id: str
    source_count: int
    system_count: int
    attempt_count: int
    gate_passed: bool
    is_formal_run: bool
    status_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_counts", MappingProxyType(dict(sorted(self.status_counts.items()))))


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: object) -> str:
    import numpy as np

    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _freeze(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    raise BaselineFormalError("json", "unsupported value")


def load_phase3_baseline_formal_config(path: Path) -> Phase3BaselineFormalConfig:
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical(document):
        raise BaselineFormalError("config canonical", "must use canonical JSON")
    observed_sha = hashlib.sha256(raw).hexdigest()
    if CONFIG_SHA256 and (len(raw) != CONFIG_BYTE_COUNT or observed_sha != CONFIG_SHA256):
        raise BaselineFormalError("config identity", "does not match")
    catalog_value = document["catalog"]
    catalog = CatalogIdentity(
        path=str(catalog_value["path"]),
        byte_count=int(catalog_value["byte_count"]),
        sha256=str(catalog_value["sha256"]),
        catalog_id=str(catalog_value["catalog_id"]),
    )
    catalog_path = _PROJECT_ROOT / catalog.path
    if catalog_path.stat().st_size != catalog.byte_count or _sha(catalog_path) != catalog.sha256:
        raise BaselineFormalError("catalog", "identity mismatch")
    policy_value = document["policy"]
    policy = BaselineCoveragePolicy(
        minimum_successful_fraction=float(policy_value["minimum_successful_fraction"]),
        minimum_phase5_systems=int(policy_value["minimum_phase5_systems"]),
        minimum_phase5_families=int(policy_value["minimum_phase5_families"]),
        minimum_promoted_per_family=int(policy_value["minimum_promoted_per_family"]),
        minimum_common_record_fraction=float(policy_value["minimum_common_record_fraction"]),
        require_zero_failed_runtime=bool(policy_value["require_zero_failed_runtime"]),
        require_zero_not_applicable=bool(policy_value["require_zero_not_applicable"]),
    )
    operational = document["operational"]
    storage = document["storage"]
    config = Phase3BaselineFormalConfig(
        path=path, byte_count=len(raw), sha256=observed_sha,
        schema_version=str(document["schema_version"]), catalog=catalog,
        expected_source_count=int(document["expected_source_count"]),
        expected_system_count=int(document["expected_system_count"]),
        expected_attempt_count=int(document["expected_attempt_count"]),
        worker_processes=int(operational["worker_processes"]),
        blas_threads_per_process=int(operational["blas_threads_per_process"]),
        shard_source_count=int(operational["shard_source_count"]),
        storage_mode=str(storage["storage_mode"]),
        success_statuses=tuple(BaselineRunStatus(value) for value in document["success_statuses"]),
        policy=policy, phase1=_freeze(document["phase1"]),
        claim_boundary=str(document["claim_boundary"]),
    )
    assert isinstance(config.phase1, Mapping)
    return config


def load_rruff_baseline_formal_sources(
    config: Phase3BaselineFormalConfig, *, project_root: Path
) -> tuple["BaselineCanarySource", ...]:
    import numpy as np

    from rpe.evaluation import Spectrum1D
    from rpe.io.store import UnifiedDataset
    from rpe.methods.classical.baseline_canary import BaselineCanarySource

    root = Path(project_root)
    phase1 = config.phase1
    subset_path = root / str(phase1["selected_source_subset_path"])
    manifest_path = root / str(phase1["run_manifest_path"])
    if _sha(subset_path) != phase1["selected_source_subset_sha256"]:
        raise BaselineFormalError("phase1.source_subset", "SHA256 mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("selected_source_subset_sha256") != phase1["selected_source_subset_sha256"]:
        raise BaselineFormalError("phase1.manifest.source_subset", "does not match")
    if manifest.get("scientific_config", {}).get("sha256") != phase1["scientific_config_sha256"]:
        raise BaselineFormalError("phase1.manifest.scientific_config", "does not match")
    if manifest.get("source_snapshot_sha256") != phase1["source_snapshot_sha256"]:
        raise BaselineFormalError("phase1.manifest.source_snapshot", "does not match")
    rows = [json.loads(line) for line in subset_path.read_bytes().splitlines()]
    if len(rows) != config.expected_source_count:
        raise BaselineFormalError("phase1.source_subset", "source count mismatch")
    if tuple(int(row["selection_rank"]) for row in rows) != tuple(range(len(rows))):
        raise BaselineFormalError("phase1.source_subset", "ranks must be consecutive")
    sources: list[BaselineCanarySource] = []
    with UnifiedDataset.open(root / str(phase1["source_dataset_path"]), verify_checksums=True) as dataset:
        for row in rows:
            record = dataset.get(str(row["record_id"]))
            axis = np.asarray(record.wavenumber, dtype="<f8")
            intensity = np.asarray(record.intensity, dtype="<f8")
            if np.all(np.diff(axis) < 0.0):
                axis = np.ascontiguousarray(axis[::-1])
                intensity = np.ascontiguousarray(intensity[::-1])
            spectrum = Spectrum1D(
                spectrum_id=f"{record.meta.dataset_id}::{record.record_id}",
                sample_id=record.meta.sample_id,
                axis_cm1=axis,
                intensity=intensity,
            )
            axis_sha = _array_sha(spectrum.axis_cm1)
            intensity_sha = _array_sha(spectrum.intensity)
            if axis_sha != row.get("normalized_axis_float64_sha256"):
                raise BaselineFormalError("source.normalized_axis_float64_sha256", "does not match ledger")
            if intensity_sha != row.get("normalized_intensity_float64_sha256"):
                raise BaselineFormalError("source.normalized_intensity_float64_sha256", "does not match ledger")
            point_count = int(spectrum.intensity.size)
            sources.append(
                BaselineCanarySource(
                    band_id="formal10k",
                    selection_rank=int(row["selection_rank"]),
                    record_id=str(row["record_id"]),
                    sample_id=str(row["sample_id"]),
                    class_label=int(row["class_label"]),
                    mineral_name=str(row["mineral_name"]),
                    point_count=point_count,
                    axis_sha256=axis_sha,
                    intensity_sha256=intensity_sha,
                    spectrum=spectrum,
                )
            )
    return tuple(sources)


def evaluate_baseline_coverage(
    receipts: Sequence[BaselineFormalReceipt],
    *,
    source_ids: Sequence[str],
    systems: Sequence[Phase3System],
    policy: BaselineCoveragePolicy,
) -> BaselineCoverageResult:
    frozen_receipts = tuple(receipts)
    frozen_sources = tuple(source_ids)
    frozen_systems = tuple(systems)
    expected_pairs = {(record_id, system.system_id) for record_id in frozen_sources for system in frozen_systems}
    observed_pairs = {(row.record_id, row.system_id) for row in frozen_receipts}
    if len(frozen_receipts) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise BaselineFormalError("receipt Cartesian product", "is incomplete or duplicated")
    by_system: dict[str, list[BaselineFormalReceipt]] = defaultdict(list)
    system_by_id = {system.system_id: system for system in frozen_systems}
    for row in frozen_receipts:
        by_system[row.system_id].append(row)
    summaries = []
    promoted = []
    success_values = {status.value for status in _SUCCESS_STATUSES}
    for system in sorted(frozen_systems, key=lambda value: value.system_id):
        rows = by_system[system.system_id]
        counts = Counter(row.status.value for row in rows)
        successful = sum(counts[value] for value in success_values)
        fraction = successful / len(frozen_sources)
        promoted_status = (
            len(rows) == len(frozen_sources)
            and fraction >= policy.minimum_successful_fraction
            and (not policy.require_zero_failed_runtime or counts[BaselineRunStatus.FAILED_RUNTIME.value] == 0)
            and (not policy.require_zero_not_applicable or counts[BaselineRunStatus.NOT_APPLICABLE.value] == 0)
        )
        if promoted_status:
            promoted.append(system.system_id)
        summaries.append(SystemCoverageSummary(system.system_id, system.family_id, len(rows), successful, fraction, counts, promoted_status))
    promoted_by_family: dict[str, list[str]] = defaultdict(list)
    for system_id in promoted:
        promoted_by_family[system_by_id[system_id].family_id].append(system_id)
    qualifying_families = tuple(sorted(family for family, ids in promoted_by_family.items() if len(ids) >= policy.minimum_promoted_per_family))
    eligible_ids = tuple(sorted(system_id for family in qualifying_families for system_id in promoted_by_family[family]))
    successful_pairs = {(row.record_id, row.system_id) for row in frozen_receipts if row.status.value in success_values}
    common_count = sum(all((record_id, system_id) in successful_pairs for system_id in eligible_ids) for record_id in frozen_sources) if eligible_ids else 0
    common_fraction = common_count / len(frozen_sources)
    failures = []
    if len(eligible_ids) < policy.minimum_phase5_systems:
        failures.append("eligible_system_count")
    if len(qualifying_families) < policy.minimum_phase5_families:
        failures.append("eligible_family_count")
    if common_fraction < policy.minimum_common_record_fraction:
        failures.append("common_record_fraction")
    return BaselineCoverageResult(
        system_summaries=tuple(summaries), promoted_system_ids=tuple(sorted(promoted)),
        qualifying_family_ids=qualifying_families, phase5_eligible_system_ids=eligible_ids,
        phase5_eligible_system_count=len(eligible_ids), common_successful_record_count=common_count,
        common_record_fraction=common_fraction, gate_passed=not failures, gate_failures=tuple(failures),
    )


def _receipt_from_result(source: BaselineCanarySource, system: Phase3System) -> BaselineFormalReceipt:
    result = run_baseline_system(system, source.spectrum)
    return BaselineFormalReceipt(
        selection_rank=source.selection_rank, record_id=source.record_id,
        system_id=system.system_id, family_id=system.family_id, method_id=system.method_id,
        status=result.status, baseline_sha256=result.baseline_sha256,
        corrected_sha256=result.corrected_sha256, warnings=result.warnings,
        diagnostics=result.diagnostics, error_code=result.error_code, error_message=result.error_message,
    )


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _receipt_document(row: BaselineFormalReceipt) -> dict[str, object]:
    return {
        "baseline_sha256": row.baseline_sha256, "corrected_sha256": row.corrected_sha256,
        "diagnostics": _json_ready(row.diagnostics), "error_code": row.error_code,
        "error_message": row.error_message, "family_id": row.family_id,
        "method_id": row.method_id, "record_id": row.record_id,
        "selection_rank": row.selection_rank, "status": row.status.value,
        "system_id": row.system_id,
        "warnings": [{"category": item.category, "message": item.message} for item in row.warnings],
    }


def _receipt_payload_for_shard(
    sources: Sequence["BaselineCanarySource"],
    systems: Sequence[Phase3System],
) -> bytes:
    receipts = (
        _receipt_from_result(source, system)
        for source in sources
        for system in systems
    )
    return b"".join(_canonical(_receipt_document(row)) for row in receipts)


def _initialize_worker(
    catalog_path: str, project_root: str, system_ids: tuple[str, ...]
) -> None:
    global _WORKER_SYSTEMS

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    catalog = load_classical_catalog(
        Path(catalog_path), project_root=Path(project_root)
    )
    by_id = {system.system_id: system for system in catalog.systems}
    _WORKER_SYSTEMS = tuple(by_id[system_id] for system_id in system_ids)


def _worker_shard(
    task: tuple[int, tuple["BaselineCanarySource", ...]]
) -> tuple[int, bytes]:
    shard_index, sources = task
    if not _WORKER_SYSTEMS:
        raise BaselineFormalError("worker", "systems were not initialized")
    return shard_index, _receipt_payload_for_shard(sources, _WORKER_SYSTEMS)


def _source_document(source: BaselineCanarySource) -> dict[str, object]:
    return {
        "axis_sha256": source.axis_sha256, "class_label": source.class_label,
        "intensity_sha256": source.intensity_sha256, "mineral_name": source.mineral_name,
        "point_count": source.point_count, "record_id": source.record_id,
        "sample_id": source.sample_id, "selection_rank": source.selection_rank,
    }


def _summary_documents(result: BaselineCoverageResult) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    systems = [
        {
            "coverage_promoted": summary.coverage_promoted, "family_id": summary.family_id,
            "status_counts": dict(summary.status_counts), "successful_count": summary.successful_count,
            "successful_fraction": summary.successful_fraction, "system_id": summary.system_id,
            "terminal_count": summary.terminal_count,
        }
        for summary in result.system_summaries
    ]
    families: dict[str, dict[str, object]] = {}
    for summary in result.system_summaries:
        current = families.setdefault(summary.family_id, {"family_id": summary.family_id, "promoted_system_count": 0, "system_count": 0})
        current["system_count"] = int(current["system_count"]) + 1
        current["promoted_system_count"] = int(current["promoted_system_count"]) + int(summary.coverage_promoted)
    for family, current in families.items():
        current["qualifying_family"] = family in result.qualifying_family_ids
    return systems, [families[key] for key in sorted(families)]


def _code_identity(root: Path) -> dict[str, str]:
    paths = (
        "rpe/methods/catalog.py", "rpe/methods/classical/baseline.py",
        "rpe/runner/phase3_baseline_formal.py",
    )
    return {path: _sha(root / path) for path in paths}


def build_baseline_receipt_artifact(
    sources: Sequence[BaselineCanarySource],
    systems: Sequence[Phase3System],
    output_root: Path,
    *,
    catalog: Phase3ClassicalCatalog,
    config: Phase3BaselineFormalConfig,
    project_root: Path,
    worker_processes: int = 1,
    shard_source_count: int | None = None,
) -> BaselineFormalSummary:
    frozen_sources = tuple(sorted(sources, key=lambda value: value.selection_rank))
    frozen_systems = tuple(sorted(systems, key=lambda value: value.system_id))
    if not frozen_sources or not frozen_systems:
        raise BaselineFormalError("inputs", "must not be empty")
    if worker_processes <= 0:
        raise BaselineFormalError("worker_processes", "must be positive")
    if any(system.task_line is not TaskLine.BASELINE_CORRECTION for system in frozen_systems):
        raise BaselineFormalError("systems", "must all be baseline correction")
    catalog_ids = {system.system_id for system in catalog.systems}
    if any(system.system_id not in catalog_ids for system in frozen_systems):
        raise BaselineFormalError("catalog membership", "system is not in catalog")
    if len({source.record_id for source in frozen_sources}) != len(frozen_sources):
        raise BaselineFormalError("sources", "record IDs must be unique")
    if len({system.system_id for system in frozen_systems}) != len(frozen_systems):
        raise BaselineFormalError("systems", "system IDs must be unique")
    for source in frozen_sources:
        if source.point_count != source.spectrum.intensity.size:
            raise BaselineFormalError("source.point_count", "does not match spectrum")
        if source.axis_sha256 != _array_sha(source.spectrum.axis_cm1):
            raise BaselineFormalError("source.axis_sha256", "does not match spectrum")
        if source.intensity_sha256 != _array_sha(source.spectrum.intensity):
            raise BaselineFormalError("source.intensity_sha256", "does not match spectrum")
    source_payload = b"".join(_canonical(_source_document(source)) for source in frozen_sources)
    identity = {
        "catalog_id": catalog.catalog_id, "catalog_sha256": catalog.sha256,
        "code_identity": _code_identity(Path(project_root)), "config_sha256": config.sha256,
        "source_subset_sha256": hashlib.sha256(source_payload).hexdigest(),
        "system_ids": [system.system_id for system in frozen_systems],
    }
    run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()
    run_path = Path(output_root) / f"phase3-baseline-formal10k-{run_id}"
    if run_path.exists():
        raise BaselineFormalError("output", "run path already exists")
    (run_path / "shards").mkdir(parents=True)
    actual_shard_size = len(frozen_sources) if shard_source_count is None else shard_source_count
    if actual_shard_size <= 0:
        raise BaselineFormalError("shard_source_count", "must be positive")
    source_shards = tuple(
        tuple(frozen_sources[start : start + actual_shard_size])
        for start in range(0, len(frozen_sources), actual_shard_size)
    )
    shard_tasks = tuple(enumerate(source_shards))
    if worker_processes == 1:
        shard_outputs = tuple(
            (index, _receipt_payload_for_shard(shard_sources, frozen_systems))
            for index, shard_sources in shard_tasks
        )
    else:
        with ProcessPoolExecutor(
            max_workers=worker_processes,
            initializer=_initialize_worker,
            initargs=(
                str(catalog.path.resolve()),
                str(Path(project_root).resolve()),
                tuple(system.system_id for system in frozen_systems),
            ),
        ) as executor:
            shard_outputs = tuple(executor.map(_worker_shard, shard_tasks))
    for shard_index, shard_payload in sorted(shard_outputs):
        (run_path / "shards" / f"{shard_index:05d}.jsonl").write_bytes(
            shard_payload
        )
    (run_path / "source_subset.jsonl").write_bytes(source_payload)
    receipts = tuple(
        _read_receipt(json.loads(line))
        for _, shard_payload in sorted(shard_outputs)
        for line in shard_payload.splitlines()
    )
    coverage = evaluate_baseline_coverage(
        receipts,
        source_ids=tuple(source.record_id for source in frozen_sources),
        systems=frozen_systems,
        policy=config.policy,
    )
    system_docs, family_docs = _summary_documents(coverage)
    (run_path / "system_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in system_docs))
    (run_path / "family_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in family_docs))
    promotion = {
        "common_record_fraction": coverage.common_record_fraction,
        "phase5_eligible_system_ids": list(coverage.phase5_eligible_system_ids),
        "promoted_system_ids": list(coverage.promoted_system_ids),
        "qualifying_family_ids": list(coverage.qualifying_family_ids),
        "run_id": run_id,
    }
    gate = {
        "common_successful_record_count": coverage.common_successful_record_count,
        "gate_failures": list(coverage.gate_failures), "gate_passed": coverage.gate_passed,
        "phase5_eligible_system_count": coverage.phase5_eligible_system_count, "run_id": run_id,
    }
    is_formal = len(frozen_sources) == config.expected_source_count and len(frozen_systems) == config.expected_system_count
    manifest = {
        **identity, "attempt_count": len(receipts), "claim_boundary": config.claim_boundary,
        "is_formal_run": is_formal, "run_id": run_id,
        "schema_version": "phase3-baseline-formal10k-run-v1",
        "source_count": len(frozen_sources), "system_count": len(frozen_systems),
        "storage_mode": config.storage_mode,
    }
    (run_path / "manifest.json").write_bytes(_canonical(manifest))
    (run_path / "promotion.json").write_bytes(_canonical(promotion))
    (run_path / "gate.json").write_bytes(_canonical(gate))
    marker_name = "complete.json" if coverage.gate_passed and is_formal else "failed.json"
    (run_path / marker_name).write_bytes(_canonical({"gate_passed": coverage.gate_passed, "is_formal_run": is_formal, "run_id": run_id, "status": "complete" if marker_name == "complete.json" else "failed"}))
    members = sorted(path.relative_to(run_path).as_posix() for path in run_path.rglob("*") if path.is_file())
    (run_path / "SHA256SUMS").write_text("".join(f"{_sha(run_path/member)}  {member}\n" for member in members), encoding="utf-8")
    status_counts = Counter(row.status.value for row in receipts)
    return BaselineFormalSummary(run_path, run_id, len(frozen_sources), len(frozen_systems), len(receipts), coverage.gate_passed, is_formal, status_counts)


def _read_receipt(row: Mapping[str, object]) -> BaselineFormalReceipt:
    from rpe.methods.classical.baseline import CapturedWarning
    return BaselineFormalReceipt(
        selection_rank=int(row["selection_rank"]), record_id=str(row["record_id"]),
        system_id=str(row["system_id"]), family_id=str(row["family_id"]),
        method_id=str(row["method_id"]), status=BaselineRunStatus(str(row["status"])),
        baseline_sha256=None if row["baseline_sha256"] is None else str(row["baseline_sha256"]),
        corrected_sha256=None if row["corrected_sha256"] is None else str(row["corrected_sha256"]),
        warnings=tuple(CapturedWarning(str(value["category"]), str(value["message"])) for value in row["warnings"]),
        diagnostics=_freeze(row["diagnostics"]), error_code=None if row["error_code"] is None else str(row["error_code"]),
        error_message=None if row["error_message"] is None else str(row["error_message"]),
    )


def _read_canonical_json(path: Path, label: str) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BaselineFormalError(label, "is not valid JSON") from error
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise BaselineFormalError(label, "must be canonical JSON object")
    return value


def _read_canonical_jsonl(path: Path, label: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BaselineFormalError(
                    label, f"line {line_number} is not valid JSON"
                ) from error
            if not isinstance(value, Mapping) or raw != _canonical(value):
                raise BaselineFormalError(
                    label, f"line {line_number} must be canonical JSON object"
                )
            rows.append(value)
    return rows


def verify_phase3_baseline_formal(path: Path, *, project_root: Path) -> BaselineFormalSummary:
    path = Path(path)
    root = Path(project_root)
    checksum_raw = (path / "SHA256SUMS").read_text(encoding="utf-8")
    checks: dict[str, str] = {}
    for line in checksum_raw.splitlines():
        try:
            digest, member = line.split("  ")
        except ValueError as error:
            raise BaselineFormalError("checksum", "invalid line") from error
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not member
            or member in checks
            or member == "SHA256SUMS"
        ):
            raise BaselineFormalError("checksum", "invalid entry")
        checks[member] = digest
        member_path = path / member
        if not member_path.is_file() or _sha(member_path) != digest:
            raise BaselineFormalError("checksum", f"{member} mismatch")
    expected_checksum_text = "".join(
        f"{checks[member]}  {member}\n" for member in sorted(checks)
    )
    if checksum_raw != expected_checksum_text:
        raise BaselineFormalError("checksum", "entries must be sorted and canonical")
    actual = {file.relative_to(path).as_posix() for file in path.rglob("*") if file.is_file() and file.name != "SHA256SUMS"}
    if set(checks) != actual:
        raise BaselineFormalError("checksum inventory", "does not match files")

    manifest = _read_canonical_json(path / "manifest.json", "manifest")
    config = load_phase3_baseline_formal_config(
        root / "experiments/phase3/configs/baseline_formal10k_v1.json"
    )
    catalog = load_classical_catalog(
        root / "experiments/phase3/configs/classical_system_catalog_v1.json",
        project_root=root,
    )
    source_rows = _read_canonical_jsonl(
        path / "source_subset.jsonl", "source subset"
    )
    source_keys = {
        "axis_sha256", "class_label", "intensity_sha256",
        "mineral_name", "point_count", "record_id", "sample_id",
        "selection_rank",
    }
    if any(set(row) != source_keys for row in source_rows):
        raise BaselineFormalError("source subset", "has an unexpected key set")
    source_ids = tuple(str(row["record_id"]) for row in source_rows)
    source_ranks = tuple(int(row["selection_rank"]) for row in source_rows)
    if (
        len(source_ids) != len(set(source_ids))
        or len(source_ranks) != len(set(source_ranks))
        or source_ranks != tuple(sorted(source_ranks))
    ):
        raise BaselineFormalError("source subset", "IDs and ranks must be unique and ordered")
    source_by_id = {record_id: row for record_id, row in zip(source_ids, source_rows)}

    if not isinstance(manifest.get("system_ids"), list):
        raise BaselineFormalError("manifest.system_ids", "must be an array")
    system_ids = tuple(str(value) for value in manifest["system_ids"])
    if system_ids != tuple(sorted(set(system_ids))):
        raise BaselineFormalError("manifest.system_ids", "must be unique and sorted")
    systems_by_id = {system.system_id: system for system in catalog.systems}
    try:
        systems = tuple(systems_by_id[value] for value in system_ids)
    except KeyError as error:
        raise BaselineFormalError("manifest.system_ids", "contains an unknown system") from error
    if any(system.task_line is not TaskLine.BASELINE_CORRECTION for system in systems):
        raise BaselineFormalError("manifest.system_ids", "must contain baseline systems only")

    shard_paths = sorted((path / "shards").glob("*.jsonl"))
    expected_shard_names = [f"{index:05d}.jsonl" for index in range(len(shard_paths))]
    if [shard.name for shard in shard_paths] != expected_shard_names:
        raise BaselineFormalError("shards", "must be consecutively numbered")
    marker_names = sorted(name for name in actual if name in {"complete.json", "failed.json"})
    if len(marker_names) != 1:
        raise BaselineFormalError("marker", "must contain exactly one terminal marker")
    expected_inventory = {
        "manifest.json", "source_subset.jsonl", "system_summary.jsonl",
        "family_summary.jsonl", "promotion.json", "gate.json",
        marker_names[0],
        *(f"shards/{shard.name}" for shard in shard_paths),
    }
    if actual != expected_inventory:
        raise BaselineFormalError("artifact inventory", "does not match protocol")

    receipts: list[BaselineFormalReceipt] = []
    expected_row_index = 0
    for shard_path in shard_paths:
        shard_rows = _read_canonical_jsonl(
            shard_path, f"receipt shard {shard_path.name}"
        )
        for row in shard_rows:
            receipt = _read_receipt(row)
            if _canonical(_receipt_document(receipt)) != _canonical(row):
                raise BaselineFormalError("receipt schema", "has an unexpected value or key")
            if expected_row_index >= len(source_rows) * len(systems):
                raise BaselineFormalError("row order", "contains extra receipt rows")
            expected_source = source_rows[expected_row_index // len(systems)]
            expected_system = systems[expected_row_index % len(systems)]
            if (
                receipt.selection_rank != int(expected_source["selection_rank"])
                or receipt.record_id != str(expected_source["record_id"])
                or receipt.system_id != expected_system.system_id
            ):
                raise BaselineFormalError(
                    "row order", "must be source order then system_id"
                )
            if (
                receipt.family_id != expected_system.family_id
                or receipt.method_id != expected_system.method_id
                or source_by_id[receipt.record_id] is not expected_source
            ):
                raise BaselineFormalError("receipt identity", "does not match source or catalog")
            receipts.append(receipt)
            expected_row_index += 1
    if expected_row_index != len(source_rows) * len(systems):
        raise BaselineFormalError("receipt Cartesian product", "is incomplete")

    coverage = evaluate_baseline_coverage(receipts, source_ids=source_ids, systems=systems, policy=config.policy)
    system_docs, family_docs = _summary_documents(coverage)
    if _read_canonical_jsonl(path / "system_summary.jsonl", "system summary") != system_docs:
        raise BaselineFormalError("system summary", "does not match receipts")
    if _read_canonical_jsonl(path / "family_summary.jsonl", "family summary") != family_docs:
        raise BaselineFormalError("family summary", "does not match receipts")

    identity = {
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": catalog.sha256,
        "code_identity": _code_identity(root),
        "config_sha256": config.sha256,
        "source_subset_sha256": _sha(path / "source_subset.jsonl"),
        "system_ids": list(system_ids),
    }
    expected_run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()

    baseline_ids = tuple(
        sorted(
            system.system_id
            for system in catalog.systems
            if system.task_line is TaskLine.BASELINE_CORRECTION
        )
    )
    formal_source_match = False
    if len(source_rows) == config.expected_source_count and system_ids == baseline_ids:
        formal_sources = load_rruff_baseline_formal_sources(config, project_root=root)
        formal_source_match = source_rows == [
            _source_document(source) for source in formal_sources
        ]
    expected_is_formal = (
        formal_source_match
        and len(receipts) == config.expected_attempt_count
        and len(systems) == config.expected_system_count
    )
    expected_manifest = {
        **identity,
        "attempt_count": len(receipts),
        "claim_boundary": config.claim_boundary,
        "is_formal_run": expected_is_formal,
        "run_id": expected_run_id,
        "schema_version": "phase3-baseline-formal10k-run-v1",
        "source_count": len(source_rows),
        "system_count": len(systems),
        "storage_mode": config.storage_mode,
    }
    if manifest != expected_manifest:
        raise BaselineFormalError("manifest", "does not match verified identities and counts")

    if expected_is_formal:
        expected_shard_count = math.ceil(
            config.expected_source_count / config.shard_source_count
        )
        if len(shard_paths) != expected_shard_count:
            raise BaselineFormalError("shards", "formal shard count does not match config")
        for index, shard_path in enumerate(shard_paths):
            expected_sources_in_shard = min(
                config.shard_source_count,
                config.expected_source_count - index * config.shard_source_count,
            )
            line_count = sum(1 for _ in shard_path.open("rb"))
            if line_count != expected_sources_in_shard * config.expected_system_count:
                raise BaselineFormalError("shards", "formal shard range does not match config")

    promotion = _read_canonical_json(path / "promotion.json", "promotion")
    expected_promotion = {
        "common_record_fraction": coverage.common_record_fraction,
        "phase5_eligible_system_ids": list(coverage.phase5_eligible_system_ids),
        "promoted_system_ids": list(coverage.promoted_system_ids),
        "qualifying_family_ids": list(coverage.qualifying_family_ids),
        "run_id": expected_run_id,
    }
    if promotion != expected_promotion:
        raise BaselineFormalError("promotion", "does not match receipts")

    gate = _read_canonical_json(path / "gate.json", "gate")
    expected_gate = {
        "common_successful_record_count": coverage.common_successful_record_count,
        "gate_failures": list(coverage.gate_failures), "gate_passed": coverage.gate_passed,
        "phase5_eligible_system_count": coverage.phase5_eligible_system_count, "run_id": expected_run_id,
    }
    if gate != expected_gate:
        raise BaselineFormalError("gate", "does not match receipts")

    marker_name = "complete.json" if expected_is_formal and coverage.gate_passed else "failed.json"
    if marker_names != [marker_name]:
        raise BaselineFormalError("marker", "does not match gate")
    marker = _read_canonical_json(path / marker_name, "marker")
    expected_marker = {
        "gate_passed": coverage.gate_passed,
        "is_formal_run": expected_is_formal,
        "run_id": expected_run_id,
        "status": "complete" if marker_name == "complete.json" else "failed",
    }
    if marker != expected_marker:
        raise BaselineFormalError("marker", "does not match verified status")
    return BaselineFormalSummary(path, expected_run_id, len(source_ids), len(systems), len(receipts), coverage.gate_passed, expected_is_formal, Counter(row.status.value for row in receipts))


__all__ = [
    "BaselineCoveragePolicy", "BaselineCoverageResult", "BaselineFormalError",
    "BaselineFormalReceipt", "BaselineFormalSummary", "BaselineRunStatus",
    "SystemCoverageSummary",
    "build_baseline_receipt_artifact", "evaluate_baseline_coverage",
    "load_phase3_baseline_formal_config", "verify_phase3_baseline_formal",
    "load_rruff_baseline_formal_sources",
]
