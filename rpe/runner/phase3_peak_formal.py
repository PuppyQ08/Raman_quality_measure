from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.evaluation import Spectrum1D
from rpe.io.store import UnifiedDataset
from rpe.methods.catalog import Phase3ClassicalCatalog, Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.peaks import PeakRunStatus, PeakWarning, peak_document, run_peak_detection_system


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase3/configs/peak_detection_v1_formal_coverage.json"
RUN_DOMAIN = b"rpe-phase3-peak-detection-v1-formal-coverage-v1\0"
CONFIG_BYTES = 3081
CONFIG_SHA256 = "22bf21749a1dcab7cf88e850f689790b9e8d375a5885d8b69991913de33397d1"
SCIPY_FAMILIES = frozenset({"find_peaks", "find_peaks_cwt"})
SUCCESS = frozenset({PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING})


class PeakFormalError(ValueError):
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
class PeakFormalExpected:
    planned_k: int
    runnable_k: int
    find_peaks_system_count: int
    find_peaks_cwt_system_count: int
    mspd_unavailable_system_count: int
    source_count: int
    source_point_count: int
    receipt_count: int


@dataclass(frozen=True)
class PeakFormalOperational:
    worker_processes: int
    verifier_worker_processes: int
    blas_threads_per_process: int
    source_block_size: int
    parallel_start_method: str


@dataclass(frozen=True)
class PeakFormalCoveragePolicy:
    require_zero_unsuccessful: bool
    descriptive_minimum_successful_fraction: float
    descriptive_minimum_common_record_fraction: float
    minimum_promoted_per_family: int
    success_statuses: tuple[PeakRunStatus, ...] = (
        PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING,
    )
    subset_full_tau_status: str = "not_evaluable_no_quality_endpoint"


@dataclass(frozen=True)
class Phase3PeakFormalConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    authorities: Mapping[str, ArtifactIdentity]
    catalog: ArtifactIdentity
    catalog_id: str
    phase3_lock: ArtifactIdentity
    expected: PeakFormalExpected
    operational: PeakFormalOperational
    policy: PeakFormalCoveragePolicy
    rruff: Mapping[str, object]
    system_ids: tuple[str, ...]
    mspd_system_ids: tuple[str, ...]
    system_ids_sha256: str
    mspd_system_ids_sha256: str
    phase5_power_status: str
    claim_boundary: str
    document: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "authorities", MappingProxyType(dict(self.authorities)))
        object.__setattr__(self, "rruff", _freeze_mapping(self.rruff))
        object.__setattr__(self, "document", _freeze_mapping(self.document))


@dataclass(frozen=True)
class PeakFormalSource:
    cohort_id: str
    record_id: str
    class_label: int
    source_order: int
    point_count: int
    axis_sha256: str
    intensity_sha256: str
    spectrum: Spectrum1D
    sample_id: str | None = None
    mineral_name: str | None = None


@dataclass(frozen=True)
class PeakFormalReceipt:
    system_id: str
    family_id: str
    method_id: str
    record_id: str
    source_order: int
    status: PeakRunStatus
    peak_count: int
    peak_start: int
    peaks_sha256: str | None
    empty_output: bool
    warnings: tuple[PeakWarning, ...]
    diagnostics: Mapping[str, object]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class PeakSystemCoverageSummary:
    system_id: str
    family_id: str
    receipt_count: int
    successful_count: int
    successful_fraction: float
    status_counts: Mapping[str, int]
    peak_count: int
    empty_receipt_count: int
    warning_receipt_count: int
    rejection_totals: Mapping[str, int]
    coverage_promoted: bool


@dataclass(frozen=True)
class PeakCoverageResult:
    system_summaries: tuple[PeakSystemCoverageSummary, ...]
    promoted_system_ids: tuple[str, ...]
    qualifying_family_ids: tuple[str, ...]
    common_successful_record_count: int
    common_successful_record_fraction: float


@dataclass(frozen=True)
class PeakFormalSummary:
    path: Path
    run_id: str
    status: str
    is_formal_run: bool
    system_count: int
    source_count: int
    receipt_count: int
    peak_count: int
    empty_receipt_count: int
    promoted_system_count: int
    warning_count: int


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _freeze(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    raise PeakFormalError("mapping", "contains unsupported value")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _identity(value: Mapping[str, object]) -> ArtifactIdentity:
    return ArtifactIdentity(str(value["path"]), int(value["byte_count"]), str(value["sha256"]))


def _verify_identity(root: Path, value: ArtifactIdentity, label: str) -> None:
    path = root / value.path
    if not path.is_file() or path.stat().st_size != value.byte_count or _sha(path) != value.sha256:
        raise PeakFormalError(label, "identity mismatch")


def _ids_sha(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def load_phase3_peak_formal_config(path: Path) -> Phase3PeakFormalConfig:
    path = Path(path); raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise PeakFormalError("config", "invalid JSON") from error
    if not isinstance(document, Mapping) or raw != _canonical(document):
        raise PeakFormalError("config", "must be canonical finite JSON")
    observed_sha = hashlib.sha256(raw).hexdigest()
    if path.resolve() == DEFAULT_CONFIG.resolve() and (len(raw) != CONFIG_BYTES or observed_sha != CONFIG_SHA256):
        raise PeakFormalError("config", "frozen identity mismatch")
    root = ROOT
    authorities_raw = document["authorities"]
    assert isinstance(authorities_raw, Mapping)
    authorities = {str(key): _identity(value) for key, value in authorities_raw.items()}
    for key, value in authorities.items(): _verify_identity(root, value, f"authorities.{key}")
    catalog_raw = document["catalog"]; assert isinstance(catalog_raw, Mapping)
    catalog_identity = ArtifactIdentity(str(catalog_raw["path"]), int(catalog_raw["byte_count"]), str(catalog_raw["sha256"]))
    _verify_identity(root, catalog_identity, "catalog")
    phase3_lock = _identity(document["phase3_lock"]); _verify_identity(root, phase3_lock, "phase3_lock")  # type: ignore[arg-type]
    catalog = load_classical_catalog(root / catalog_identity.path, project_root=root)
    scipy_ids = tuple(sorted(value.system_id for value in catalog.systems if value.task_line is TaskLine.PEAK_DETECTION and value.family_id in SCIPY_FAMILIES))
    mspd_ids = tuple(sorted(value.system_id for value in catalog.systems if value.task_line is TaskLine.PEAK_DETECTION and value.family_id == "mspd"))
    if _ids_sha(scipy_ids) != catalog_raw["scipy_system_ids_sha256"] or _ids_sha(mspd_ids) != catalog_raw["mspd_system_ids_sha256"]:
        raise PeakFormalError("catalog", "peak system projection mismatch")
    expected = PeakFormalExpected(**{key: int(value) for key, value in document["expected"].items()})  # type: ignore[union-attr]
    if expected != PeakFormalExpected(36, 24, 12, 12, 12, 10000, 22053403, 240000):
        raise PeakFormalError("expected", "does not match Step 8")
    operational = PeakFormalOperational(**document["operational"])  # type: ignore[arg-type]
    if (operational.worker_processes, operational.verifier_worker_processes, operational.blas_threads_per_process, operational.source_block_size, operational.parallel_start_method) != (8, 7, 1, 100, "fork"):
        raise PeakFormalError("operational", "does not match frozen execution")
    policy_raw = document["policy"]; assert isinstance(policy_raw, Mapping)
    policy = PeakFormalCoveragePolicy(
        bool(policy_raw["require_zero_unsuccessful"]), float(policy_raw["descriptive_minimum_successful_fraction"]),
        float(policy_raw["descriptive_minimum_common_record_fraction"]), int(policy_raw["minimum_promoted_per_family"]),
        tuple(PeakRunStatus(value) for value in policy_raw["success_statuses"]), str(policy_raw["subset_full_tau_status"]),
    )
    rruff = document["rruff"]; assert isinstance(rruff, Mapping)
    phase1 = _identity(rruff["phase1_manifest"])  # type: ignore[arg-type]
    _verify_identity(root, phase1, "rruff.phase1_manifest")
    if len(scipy_ids) != 24 or len(mspd_ids) != 12:
        raise PeakFormalError("catalog", "peak K mismatch")
    return Phase3PeakFormalConfig(
        path, len(raw), observed_sha, str(document["schema_version"]), authorities, catalog_identity,
        str(catalog_raw["catalog_id"]), phase3_lock, expected, operational, policy, rruff, scipy_ids, mspd_ids,
        str(catalog_raw["scipy_system_ids_sha256"]), str(catalog_raw["mspd_system_ids_sha256"]),
        str(document["phase5_power_status"]), str(document["claim_boundary"]), document,
    )


def _source_row(source: PeakFormalSource) -> dict[str, object]:
    return {
        "axis_sha256": source.axis_sha256, "class_label": source.class_label, "cohort_id": source.cohort_id,
        "intensity_sha256": source.intensity_sha256, "mineral_name": source.mineral_name,
        "point_count": source.point_count, "record_id": source.record_id, "sample_id": source.sample_id,
        "source_order": source.source_order,
    }


def load_phase3_peak_formal_sources(config: Phase3PeakFormalConfig, *, project_root: Path) -> tuple[PeakFormalSource, ...]:
    root = Path(project_root); rruff = config.rruff
    ledger_path = root / str(rruff["source_ledger_path"])
    if ledger_path.stat().st_size != int(rruff["source_ledger_size"]) or _sha(ledger_path) != rruff["source_ledger_sha256"]:
        raise PeakFormalError("rruff.source_ledger", "identity mismatch")
    manifest_identity = rruff["phase1_manifest"]; assert isinstance(manifest_identity, Mapping)
    manifest = json.loads((root / str(manifest_identity["path"])).read_bytes())
    if (manifest.get("selected_source_subset_sha256") != rruff["source_ledger_sha256"]
            or manifest.get("scientific_config", {}).get("sha256") != rruff["scientific_config_sha256"]
            or manifest.get("source_snapshot_sha256") != rruff["source_snapshot_sha256"]):
        raise PeakFormalError("rruff.phase1_manifest", "authority mismatch")
    rows = [json.loads(line) for line in ledger_path.read_bytes().splitlines()]
    if len(rows) != config.expected.source_count:
        raise PeakFormalError("rruff.source_ledger", "source count mismatch")
    sources = []
    with UnifiedDataset.open(root / str(rruff["dataset_path"]), verify_checksums=True) as dataset:
        for order, row in enumerate(rows):
            if int(row["selection_rank"]) != order: raise PeakFormalError("source_order", "must be consecutive")
            record = dataset.get(str(row["record_id"])); axis = np.asarray(record.wavenumber, dtype="<f8"); intensity = np.asarray(record.intensity, dtype="<f8")
            if np.all(np.diff(axis) < 0): axis = np.ascontiguousarray(axis[::-1]); intensity = np.ascontiguousarray(intensity[::-1])
            spectrum = Spectrum1D(str(row["record_id"]), record.meta.sample_id, axis, intensity)
            axis_sha, intensity_sha = _array_sha(axis), _array_sha(intensity)
            if axis_sha != row["normalized_axis_float64_sha256"] or intensity_sha != row["normalized_intensity_float64_sha256"]:
                raise PeakFormalError("source", "normalized hash mismatch")
            sources.append(PeakFormalSource("rruff_core10k", str(row["record_id"]), int(row["class_label"]), order, intensity.size, axis_sha, intensity_sha, spectrum, str(row["sample_id"]), str(row["mineral_name"])))
    if sum(value.point_count for value in sources) != config.expected.source_point_count:
        raise PeakFormalError("source points", "count mismatch")
    return tuple(sources)


def _rejection_totals(diagnostics: Mapping[str, object]) -> dict[str, int]:
    result: Counter[str] = Counter()
    characterization = diagnostics.get("characterization_rejections", {})
    if isinstance(characterization, Mapping): result.update({f"characterization:{key}": int(value) for key, value in characterization.items()})
    for key in ("rejected_duplicate", "rejected_non_admissible", "rejected_out_of_range"):
        if key in diagnostics: result[f"selection:{key}"] += int(diagnostics[key])
    return dict(sorted(result.items()))


def evaluate_peak_formal_coverage(*, receipts: Sequence[PeakFormalReceipt], source_ids: Sequence[str], systems: Sequence[Phase3System], policy: PeakFormalCoveragePolicy) -> PeakCoverageResult:
    source_ids = tuple(source_ids); systems = tuple(sorted(systems, key=lambda value: value.system_id)); rows = tuple(receipts)
    expected_pairs = [(system.system_id, record_id) for system in systems for record_id in source_ids]
    if [(row.system_id, row.record_id) for row in rows] != expected_pairs:
        raise PeakFormalError("receipt Cartesian product", "incomplete, duplicate, or reordered")
    by_system: dict[str, list[PeakFormalReceipt]] = defaultdict(list)
    for row in rows: by_system[row.system_id].append(row)
    summaries = []; promoted = []; successful_by_system: dict[str, set[str]] = {}
    for system in systems:
        current = by_system[system.system_id]; counts = Counter(row.status.value for row in current)
        successful = sum(row.status in policy.success_statuses for row in current)
        is_promoted = successful == len(source_ids) and (not policy.require_zero_unsuccessful or successful == len(current))
        if is_promoted: promoted.append(system.system_id)
        successful_by_system[system.system_id] = {row.record_id for row in current if row.status in policy.success_statuses}
        rejection = Counter()
        for row in current: rejection.update(_rejection_totals(row.diagnostics))
        summaries.append(PeakSystemCoverageSummary(system.system_id, system.family_id, len(current), successful, successful / len(current), dict(counts), sum(row.peak_count for row in current), sum(row.empty_output for row in current), sum(bool(row.warnings) for row in current), dict(sorted(rejection.items())), is_promoted))
    family_counts = Counter(next(system.family_id for system in systems if system.system_id == sid) for sid in promoted)
    qualifying = tuple(sorted(family for family, count in family_counts.items() if count >= policy.minimum_promoted_per_family))
    common = sum(all(record_id in successful_by_system[sid] for sid in promoted) for record_id in source_ids) if promoted else 0
    return PeakCoverageResult(tuple(summaries), tuple(promoted), qualifying, common, common / len(source_ids))


_WORKER_SOURCES: tuple[PeakFormalSource, ...] = ()
_WORKER_SYSTEMS: tuple[Phase3System, ...] = ()


def _worker_block(task: tuple[int, int, int]) -> tuple[int, int, list[dict[str, object]]]:
    system_index, start, stop = task; system = _WORKER_SYSTEMS[system_index]; output = []
    with threadpool_limits(limits=1):
        for source in _WORKER_SOURCES[start:stop]:
            result = run_peak_detection_system(system, source.spectrum)
            output.append({
                "diagnostics": _json_ready(result.diagnostics), "error_code": result.error_code, "error_message": result.error_message,
                "peaks": [peak_document(value) for value in result.peaks], "peaks_sha256": result.peaks_sha256,
                "status": result.status.value, "warnings": [(value.category, value.message) for value in result.warnings],
            })
    return system_index, start, output


def _bounded_ordered_map(executor: ProcessPoolExecutor, tasks: Sequence[tuple[int, int, int]], *, window: int) -> Iterable[tuple[int, int, list[dict[str, object]]]]:
    iterator = iter(tasks); pending = deque()
    for _ in range(min(window, len(tasks))):
        try: pending.append(executor.submit(_worker_block, next(iterator)))
        except StopIteration: break
    while pending:
        yield pending.popleft().result()
        try: pending.append(executor.submit(_worker_block, next(iterator)))
        except StopIteration: pass


def _system_row(system: Phase3System) -> dict[str, object]:
    return {"family_id": system.family_id, "hyperparameters": _json_ready(system.hyperparameters), "method_id": system.method_id, "system_id": system.system_id}


def _code_identity(root: Path) -> dict[str, dict[str, object]]:
    paths = ("rpe/io/store.py", "rpe/methods/catalog.py", "rpe/methods/classical/peaks.py", "rpe/runner/phase3_peak_formal.py", "rpe/runner/phase3_peak_formal_verifier.py", "tools/run_phase3_peak_formal.py")
    return {name: {"byte_count": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in paths}


def _software_identity() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py")}


def _build_phase3_peak_formal_fixture_artifact(sources: Sequence[PeakFormalSource], systems: Sequence[Phase3System], output_root: Path, *, config: Phase3PeakFormalConfig, catalog: Phase3ClassicalCatalog, project_root: Path, worker_count: int) -> PeakFormalSummary:
    global _WORKER_SOURCES, _WORKER_SYSTEMS
    sources = tuple(sources); systems = tuple(sorted(systems, key=lambda value: value.system_id)); root = Path(project_root)
    if worker_count <= 0 or not sources or not systems: raise PeakFormalError("inputs", "worker count, sources, and systems must be positive/nonempty")
    if any(value.task_line is not TaskLine.PEAK_DETECTION or value.family_id not in SCIPY_FAMILIES for value in systems): raise PeakFormalError("systems", "must be SciPy peak systems")
    source_payload = b"".join(_canonical(_source_row(value)) for value in sources)
    identity = {"catalog_id": catalog.catalog_id, "catalog_sha256": catalog.sha256, "code_identity": _code_identity(root), "config_sha256": config.sha256,
                "phase3_lock_sha256": config.phase3_lock.sha256, "software_identity": _software_identity(), "source_projection_sha256": hashlib.sha256(source_payload).hexdigest(),
                "system_ids": [value.system_id for value in systems]}
    run_id = hashlib.sha256(RUN_DOMAIN + _canonical(identity)).hexdigest(); path = Path(output_root) / f"phase3-peak-detection-v1-formal-{run_id}"
    if path.exists(): raise PeakFormalError("output", "run path already exists")
    path.mkdir(parents=True); (path / "config.json").write_bytes(config.path.read_bytes()); (path / "sources.jsonl").write_bytes(source_payload)
    (path / "systems.jsonl").write_bytes(b"".join(_canonical(_system_row(value)) for value in systems))
    _WORKER_SOURCES, _WORKER_SYSTEMS = sources, systems
    tasks = [(si, start, min(start + config.operational.source_block_size, len(sources))) for si in range(len(systems)) for start in range(0, len(sources), config.operational.source_block_size)]
    records_for_gate = []; warning_rows = []; global_peak_start = 0; status_by_system = defaultdict(Counter); success_masks = {value.system_id: np.zeros(len(sources), dtype=bool) for value in systems}
    peak_counts = Counter(); empty_counts = Counter(); warning_receipts = Counter(); rejection_counts: dict[str, Counter[str]] = defaultdict(Counter)
    if worker_count == 1: chunks: Iterable[tuple[int, int, list[dict[str, object]]]] = map(_worker_block, tasks); executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("fork"))
        chunks = _bounded_ordered_map(executor, tasks, window=2 * worker_count)
    with (path / "records.jsonl").open("wb") as record_stream, (path / "peaks.jsonl").open("wb") as peak_stream, (path / "warnings.jsonl").open("wb") as warning_stream:
        try:
            for system_index, start, results in chunks:
                system = systems[system_index]
                for relative, result in enumerate(results):
                    source = sources[start + relative]; peaks = result["peaks"]; status = PeakRunStatus(str(result["status"])); warnings = tuple(PeakWarning(a, b) for a, b in result["warnings"])
                    for sequence, peak in enumerate(peaks): peak_stream.write(_canonical({**peak, "record_id": source.record_id, "sequence": sequence, "system_id": system.system_id}))
                    empty = status in SUCCESS and len(peaks) == 0
                    diagnostics = result["diagnostics"]
                    selection_rejections = {key: int(diagnostics[key]) for key in ("rejected_duplicate", "rejected_non_admissible", "rejected_out_of_range") if key in diagnostics}
                    characterization_rejections = dict(diagnostics.get("characterization_rejections", {}))
                    record = {"axis_sha256": source.axis_sha256, "characterization_rejections": characterization_rejections, "cohort_id": source.cohort_id,
                              "diagnostics": diagnostics, "empty_output": empty,
                              "error_code": result["error_code"], "error_message": result["error_message"], "family_id": system.family_id, "intensity_sha256": source.intensity_sha256,
                              "method_id": system.method_id, "peak_count": len(peaks), "peak_start": global_peak_start, "peaks_sha256": result["peaks_sha256"], "point_count": source.point_count,
                              "raw_candidate_count": int(diagnostics.get("candidate_count", 0)), "record_id": source.record_id,
                              "selection_rejections": selection_rejections, "source_order": source.source_order, "status": status.value, "system_id": system.system_id,
                              "warnings": [{"category": value.category, "message": value.message} for value in warnings]}
                    record_stream.write(_canonical(record)); records_for_gate.append(PeakFormalReceipt(system.system_id, system.family_id, system.method_id, source.record_id, source.source_order, status, len(peaks), global_peak_start, result["peaks_sha256"], empty, warnings, result["diagnostics"], result["error_code"], result["error_message"]))
                    global_peak_start += len(peaks); status_by_system[system.system_id][status.value] += 1; success_masks[system.system_id][source.source_order] = status in SUCCESS
                    peak_counts[system.system_id] += len(peaks); empty_counts[system.system_id] += empty; warning_receipts[system.system_id] += bool(warnings); rejection_counts[system.system_id].update(_rejection_totals(result["diagnostics"]))
                    for sequence, warning in enumerate(warnings):
                        row = {"category": warning.category, "message": warning.message, "record_id": source.record_id, "sequence": sequence, "system_id": system.system_id}
                        warning_rows.append(row); warning_stream.write(_canonical(row))
        finally:
            if executor is not None: executor.shutdown()
    coverage = evaluate_peak_formal_coverage(receipts=records_for_gate, source_ids=tuple(value.record_id for value in sources), systems=systems, policy=config.policy)
    system_rows = [{"coverage_promoted": row.coverage_promoted, "empty_receipt_count": row.empty_receipt_count, "family_id": row.family_id,
                    "peak_count": row.peak_count, "receipt_count": row.receipt_count, "rejection_totals": dict(row.rejection_totals), "status_counts": dict(row.status_counts),
                    "successful_count": row.successful_count, "successful_fraction": row.successful_fraction, "system_id": row.system_id, "warning_receipt_count": row.warning_receipt_count}
                   for row in coverage.system_summaries]
    (path / "system_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in system_rows))
    family_rows = []
    for family, planned, runnable in (("find_peaks", 12, 12), ("find_peaks_cwt", 12, 12), ("mspd", 12, 0)):
        family_rows.append({"family_id": family, "planned_system_count": planned, "promoted_system_count": sum(value.family_id == family and value.coverage_promoted for value in coverage.system_summaries),
                            "qualifying_family": family in coverage.qualifying_family_ids, "runnable_system_count": runnable,
                            "state": "implementation_missing_algorithm_contract_mismatch" if family == "mspd" else "executed"})
    (path / "family_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in family_rows))
    warning_summary = [{"family_id": system.family_id, "receipt_count": len(sources), "system_id": system.system_id, "warning_count": sum(1 for row in warning_rows if row["system_id"] == system.system_id),
                        "warning_receipt_count": warning_receipts[system.system_id], "warning_receipt_fraction": warning_receipts[system.system_id] / len(sources)} for system in systems]
    (path / "warning_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in warning_summary))
    is_formal = tuple(value.system_id for value in systems) == config.system_ids and len(sources) == config.expected.source_count and len(records_for_gate) == config.expected.receipt_count
    cohort = {"common_successful_record_count": coverage.common_successful_record_count, "common_successful_record_fraction": coverage.common_successful_record_fraction,
              "empty_receipt_count": sum(empty_counts.values()), "peak_count": global_peak_start, "promoted_system_count": len(coverage.promoted_system_ids),
              "receipt_count": len(records_for_gate), "source_count": len(sources), "warning_count": len(warning_rows)}
    (path / "cohort_summary.json").write_bytes(_canonical(cohort))
    promotion = {"coverage_promoted_K": len(coverage.promoted_system_ids), "mspd_state": "implementation_missing_algorithm_contract_mismatch", "phase5_eligible_K": "not_evaluated",
                 "phase5_power_status": config.phase5_power_status, "planned_K": 36, "promoted_system_ids": list(coverage.promoted_system_ids), "qualifying_family_ids": list(coverage.qualifying_family_ids),
                 "runnable_K": 24, "subset_full_tau_status": config.policy.subset_full_tau_status}
    (path / "promotion.json").write_bytes(_canonical(promotion))
    gate = {"formal_execution_complete": is_formal, "phase5_power_status": config.phase5_power_status, "strict_promotion": "complete_cartesian_and_100_percent_success",
            "subset_full_tau_status": config.policy.subset_full_tau_status}; (path / "gate.json").write_bytes(_canonical(gate))
    status = "complete" if is_formal else "fixture_complete"; manifest = {**identity, "claim_boundary": config.claim_boundary, "empty_receipt_count": sum(empty_counts.values()),
        "is_formal_run": is_formal, "peak_count": global_peak_start, "receipt_count": len(records_for_gate), "run_id": run_id,
        "schema_version": "phase3-peak-detection-v1-formal-artifact-v1", "source_count": len(sources), "status": status, "system_count": len(systems), "warning_count": len(warning_rows)}
    (path / "manifest.json").write_bytes(_canonical(manifest)); marker_name = "complete.json" if is_formal else "failed.json"
    (path / marker_name).write_bytes(_canonical({"is_formal_run": is_formal, "run_id": run_id, "status": status}))
    names = sorted(value.name for value in path.iterdir() if value.is_file()); (path / "SHA256SUMS").write_text("".join(f"{_sha(path / name)}  {name}\n" for name in names))
    return PeakFormalSummary(path, run_id, status, is_formal, len(systems), len(sources), len(records_for_gate), global_peak_start, sum(empty_counts.values()), len(coverage.promoted_system_ids), len(warning_rows))


def build_phase3_peak_formal(output_root: Path, *, worker_count: int = 8) -> PeakFormalSummary:
    config = load_phase3_peak_formal_config(DEFAULT_CONFIG)
    if worker_count != config.operational.worker_processes: raise PeakFormalError("worker_count", "formal authority requires 8")
    catalog = load_classical_catalog(ROOT / config.catalog.path, project_root=ROOT); by_id = {value.system_id: value for value in catalog.systems}
    systems = tuple(by_id[value] for value in config.system_ids); sources = load_phase3_peak_formal_sources(config, project_root=ROOT)
    return _build_phase3_peak_formal_fixture_artifact(sources, systems, Path(output_root), config=config, catalog=catalog, project_root=ROOT, worker_count=worker_count)


def verify_phase3_peak_formal(path: Path, *, worker_count: int, project_root: Path = ROOT) -> PeakFormalSummary:
    from rpe.runner.phase3_peak_formal_verifier import verify_phase3_peak_formal as independent
    try: return independent(path, worker_count=worker_count, project_root=project_root)
    except ValueError as error: raise PeakFormalError("independent verifier", str(error)) from error


__all__ = ["PeakFormalCoveragePolicy", "PeakFormalError", "PeakFormalReceipt", "PeakFormalSource", "PeakFormalSummary", "Phase3PeakFormalConfig",
           "build_phase3_peak_formal", "evaluate_peak_formal_coverage", "load_phase3_peak_formal_config", "load_phase3_peak_formal_sources", "verify_phase3_peak_formal"]
