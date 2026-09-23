from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.evaluation import Spectrum1D
from rpe.io.store import UnifiedDataset
from rpe.methods.catalog import Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.peaks import (
    DetectedPeak1D,
    PeakDetectionRunResult,
    PeakRunStatus,
    PeakWarning,
    peak_document,
    run_peak_detection_system,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/phase3/configs/peak_detection_v1_formal_coverage.json"
CONFIG_BYTES = 3081
CONFIG_SHA256 = "22bf21749a1dcab7cf88e850f689790b9e8d375a5885d8b69991913de33397d1"
RUN_DOMAIN = b"rpe-phase3-peak-detection-v1-formal-coverage-v1\0"
SCIPY_FAMILIES = frozenset({"find_peaks", "find_peaks_cwt"})
SUCCESS = frozenset({"complete", "complete_with_warning"})


class PeakFormalVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class _Summary:
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


@dataclass(frozen=True)
class _Source:
    cohort_id: str
    record_id: str
    class_label: int
    source_order: int
    point_count: int
    axis_sha256: str
    intensity_sha256: str
    spectrum: Spectrum1D
    sample_id: str | None
    mineral_name: str | None


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping): return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_ready(item) for item in value]
    return value


def _read_json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes(); value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != _canonical(value): raise PeakFormalVerificationError(f"{path.name} is not canonical JSON")
    return value


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            value = json.loads(raw)
            if not isinstance(value, Mapping) or raw != _canonical(value): raise PeakFormalVerificationError(f"{path.name} line {line_number} is not canonical")
            yield value


def _verify_identity(root: Path, value: Mapping[str, object], label: str) -> None:
    path = root / str(value["path"])
    if not path.is_file() or path.stat().st_size != int(value["byte_count"]) or _sha(path) != str(value["sha256"]):
        raise PeakFormalVerificationError(f"{label} identity mismatch")


def _read_config(root: Path) -> tuple[Mapping[str, object], bytes]:
    path = root / CONFIG_PATH.relative_to(ROOT); raw = path.read_bytes(); value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != _canonical(value) or len(raw) != CONFIG_BYTES or hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise PeakFormalVerificationError("frozen config identity mismatch")
    authorities = value["authorities"]; assert isinstance(authorities, Mapping)
    for name, identity in authorities.items(): assert isinstance(identity, Mapping); _verify_identity(root, identity, f"authorities.{name}")
    for name in ("catalog", "phase3_lock"):
        identity = value[name]; assert isinstance(identity, Mapping); _verify_identity(root, identity, name)
    rruff = value["rruff"]; assert isinstance(rruff, Mapping); phase1 = rruff["phase1_manifest"]; assert isinstance(phase1, Mapping); _verify_identity(root, phase1, "rruff.phase1_manifest")
    manifest = json.loads((root / str(phase1["path"])).read_bytes())
    if (manifest.get("selected_source_subset_sha256") != rruff["source_ledger_sha256"] or manifest.get("scientific_config", {}).get("sha256") != rruff["scientific_config_sha256"] or manifest.get("source_snapshot_sha256") != rruff["source_snapshot_sha256"]):
        raise PeakFormalVerificationError("Phase 1 manifest identity mismatch")
    return value, raw


def _source_row(source: _Source) -> dict[str, object]:
    return {"axis_sha256": source.axis_sha256, "class_label": source.class_label, "cohort_id": source.cohort_id, "intensity_sha256": source.intensity_sha256,
            "mineral_name": source.mineral_name, "point_count": source.point_count, "record_id": source.record_id, "sample_id": source.sample_id, "source_order": source.source_order}


def _system_row(system: Phase3System) -> dict[str, object]:
    return {"family_id": system.family_id, "hyperparameters": _json_ready(system.hyperparameters), "method_id": system.method_id, "system_id": system.system_id}


def _code_identity(root: Path) -> dict[str, dict[str, object]]:
    names = ("rpe/io/store.py", "rpe/methods/catalog.py", "rpe/methods/classical/peaks.py", "rpe/runner/phase3_peak_formal.py", "rpe/runner/phase3_peak_formal_verifier.py", "tools/run_phase3_peak_formal.py")
    return {name: {"byte_count": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names}


def _software_identity() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py")}


def _load_sources(config: Mapping[str, object], root: Path) -> tuple[_Source, ...]:
    rruff = config["rruff"]; expected = config["expected"]; assert isinstance(rruff, Mapping); assert isinstance(expected, Mapping)
    ledger = root / str(rruff["source_ledger_path"])
    if ledger.stat().st_size != int(rruff["source_ledger_size"]) or _sha(ledger) != rruff["source_ledger_sha256"]: raise PeakFormalVerificationError("RRUFF ledger mismatch")
    rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]; sources = []
    with UnifiedDataset.open(root / str(rruff["dataset_path"]), verify_checksums=True) as dataset:
        for order, row in enumerate(rows):
            if int(row["selection_rank"]) != order: raise PeakFormalVerificationError("RRUFF order mismatch")
            record = dataset.get(str(row["record_id"])); axis = np.asarray(record.wavenumber, dtype="<f8"); intensity = np.asarray(record.intensity, dtype="<f8")
            if np.all(np.diff(axis) < 0): axis = np.ascontiguousarray(axis[::-1]); intensity = np.ascontiguousarray(intensity[::-1])
            spectrum = Spectrum1D(str(row["record_id"]), record.meta.sample_id, axis, intensity); axis_sha = _array_sha(axis); intensity_sha = _array_sha(intensity)
            if axis_sha != row["normalized_axis_float64_sha256"] or intensity_sha != row["normalized_intensity_float64_sha256"]: raise PeakFormalVerificationError("RRUFF normalized hash mismatch")
            sources.append(_Source("rruff_core10k", str(row["record_id"]), int(row["class_label"]), order, intensity.size, axis_sha, intensity_sha, spectrum, str(row["sample_id"]), str(row["mineral_name"])))
    if len(sources) != int(expected["source_count"]) or sum(value.point_count for value in sources) != int(expected["source_point_count"]): raise PeakFormalVerificationError("RRUFF count mismatch")
    return tuple(sources)


def _peak_from_row(row: Mapping[str, object]) -> DetectedPeak1D:
    return DetectedPeak1D(index=int(row["index"]), position_cm1=float(row["position_cm1"]), height=float(row["height"]), prominence=float(row["prominence"]),
        fwhm_cm1=float(row["fwhm_cm1"]), area=float(row["area"]), left_base_index=int(row["left_base_index"]), right_base_index=int(row["right_base_index"]),
        contour_height=float(row["contour_height"]), area_left_cm1=float(row["area_left_cm1"]), area_right_cm1=float(row["area_right_cm1"]))


def _rejections(diagnostics: Mapping[str, object]) -> dict[str, int]:
    result: Counter[str] = Counter(); characterize = diagnostics.get("characterization_rejections", {})
    if isinstance(characterize, Mapping): result.update({f"characterization:{key}": int(value) for key, value in characterize.items()})
    for key in ("rejected_duplicate", "rejected_non_admissible", "rejected_out_of_range"):
        if key in diagnostics: result[f"selection:{key}"] += int(diagnostics[key])
    return dict(sorted(result.items()))


def _validate_artifact(path: Path, root: Path) -> _Summary:
    raw_checks = (path / "SHA256SUMS").read_text(); parsed = []
    for line in raw_checks.splitlines():
        digest, name = line.split("  ")
        if not (path / name).is_file() or _sha(path / name) != digest: raise PeakFormalVerificationError(f"checksum mismatch: {name}")
        parsed.append((name, digest))
    if raw_checks != "".join(f"{digest}  {name}\n" for name, digest in sorted(parsed)): raise PeakFormalVerificationError("checksum order mismatch")
    names = {name for name, _ in parsed}; actual = {value.name for value in path.iterdir() if value.is_file() and value.name != "SHA256SUMS"}
    markers = names & {"complete.json", "failed.json"}; required = {"config.json", "sources.jsonl", "systems.jsonl", "records.jsonl", "peaks.jsonl", "warnings.jsonl",
        "system_summary.jsonl", "family_summary.jsonl", "warning_summary.jsonl", "cohort_summary.json", "promotion.json", "gate.json", "manifest.json", *markers}
    if names != actual or names != required or len(markers) != 1: raise PeakFormalVerificationError("artifact inventory mismatch")
    config, config_raw = _read_config(root)
    if (path / "config.json").read_bytes() != config_raw: raise PeakFormalVerificationError("artifact config mismatch")
    catalog_value = config["catalog"]; assert isinstance(catalog_value, Mapping); catalog = load_classical_catalog(root / str(catalog_value["path"]), project_root=root)
    all_systems = {value.system_id: value for value in catalog.systems}; system_rows = list(_iter_jsonl(path / "systems.jsonl")); source_rows = list(_iter_jsonl(path / "sources.jsonl"))
    try: systems = tuple(all_systems[str(row["system_id"])] for row in system_rows)
    except KeyError as error: raise PeakFormalVerificationError("unknown peak system") from error
    if system_rows != [_system_row(value) for value in systems]: raise PeakFormalVerificationError("system rows mismatch catalog")
    source_ids = tuple(str(value["record_id"]) for value in source_rows); source_by_id = {str(value["record_id"]): value for value in source_rows}
    warning_iter = iter(_iter_jsonl(path / "warnings.jsonl")); expected_warning_index = 0; next_warning = next(warning_iter, None)
    peak_iter = iter(_iter_jsonl(path / "peaks.jsonl")); global_peak = 0; record_count = 0; empty_count = 0; warning_count = 0
    system_stats = {value.system_id: {"status": Counter(), "peak": 0, "empty": 0, "warning_receipts": 0, "rejection": Counter(), "success": set()} for value in systems}
    expected_pairs = ((system, source) for system in systems for source in source_rows)
    for record, (system, source) in zip(_iter_jsonl(path / "records.jsonl"), expected_pairs, strict=True):
        if record["system_id"] != system.system_id or record["record_id"] != source["record_id"] or int(record["source_order"]) != int(source["source_order"]): raise PeakFormalVerificationError("record Cartesian order mismatch")
        if record["axis_sha256"] != source["axis_sha256"] or record["intensity_sha256"] != source["intensity_sha256"] or int(record["point_count"]) != int(source["point_count"]): raise PeakFormalVerificationError("record source identity mismatch")
        count = int(record["peak_count"])
        if int(record["peak_start"]) != global_peak: raise PeakFormalVerificationError("peak ledger slice mismatch")
        peak_values = []
        for sequence in range(count):
            row = next(peak_iter, None)
            if row is None or row["system_id"] != system.system_id or row["record_id"] != source["record_id"] or int(row["sequence"]) != sequence: raise PeakFormalVerificationError("peak ledger key/order mismatch")
            peak_values.append(_peak_from_row(row))
        warnings = tuple(PeakWarning(str(value["category"]), str(value["message"])) for value in record["warnings"])
        result = PeakDetectionRunResult(system.system_id, system.family_id, system.method_id, str(source["record_id"]), PeakRunStatus(str(record["status"])), tuple(peak_values), None, warnings, record["diagnostics"], record["error_code"], record["error_message"])
        if result.peaks_sha256 != record["peaks_sha256"]: raise PeakFormalVerificationError("peak ledger digest mismatch")
        successful = str(record["status"]) in SUCCESS
        if bool(record["empty_output"]) != (successful and count == 0): raise PeakFormalVerificationError("peak ledger empty-output mismatch")
        diagnostics = record["diagnostics"]; expected_selection = {key: int(diagnostics[key]) for key in ("rejected_duplicate", "rejected_non_admissible", "rejected_out_of_range") if key in diagnostics}
        if record["selection_rejections"] != expected_selection or record["characterization_rejections"] != dict(diagnostics.get("characterization_rejections", {})) or int(record["raw_candidate_count"]) != int(diagnostics.get("candidate_count", 0)): raise PeakFormalVerificationError("peak ledger diagnostic projection mismatch")
        for sequence, warning in enumerate(warnings):
            expected_warning = {"category": warning.category, "message": warning.message, "record_id": source["record_id"], "sequence": sequence, "system_id": system.system_id}
            if next_warning != expected_warning: raise PeakFormalVerificationError("warning ledger mismatch")
            warning_count += 1; next_warning = next(warning_iter, None)
        stats = system_stats[system.system_id]; stats["status"][str(record["status"])] += 1; stats["peak"] += count; stats["empty"] += bool(record["empty_output"]); stats["warning_receipts"] += bool(warnings); stats["rejection"].update(_rejections(diagnostics))
        if successful: stats["success"].add(str(source["record_id"]))
        global_peak += count; record_count += 1
    if next(peak_iter, None) is not None or next_warning is not None: raise PeakFormalVerificationError("peak ledger contains unattributed rows")
    success_statuses = set(config["policy"]["success_statuses"]); summaries = []; promoted = []
    for system in systems:
        stats = system_stats[system.system_id]; success_count = sum(stats["status"][value] for value in success_statuses); is_promoted = success_count == len(source_rows)
        if is_promoted: promoted.append(system.system_id)
        summaries.append({"coverage_promoted": is_promoted, "empty_receipt_count": stats["empty"], "family_id": system.family_id, "peak_count": stats["peak"], "receipt_count": len(source_rows),
            "rejection_totals": dict(sorted(stats["rejection"].items())), "status_counts": dict(stats["status"]), "successful_count": success_count,
            "successful_fraction": success_count / len(source_rows), "system_id": system.system_id, "warning_receipt_count": stats["warning_receipts"]})
    if list(_iter_jsonl(path / "system_summary.jsonl")) != summaries: raise PeakFormalVerificationError("system summary mismatch")
    family_rows = []
    minimum_family = int(config["policy"]["minimum_promoted_per_family"])
    for family, planned, runnable in (("find_peaks", 12, 12), ("find_peaks_cwt", 12, 12), ("mspd", 12, 0)):
        count = sum(value.family_id == family and value.system_id in promoted for value in systems); family_rows.append({"family_id": family, "planned_system_count": planned,
            "promoted_system_count": count, "qualifying_family": count >= minimum_family, "runnable_system_count": runnable,
            "state": "implementation_missing_algorithm_contract_mismatch" if family == "mspd" else "executed"})
    if list(_iter_jsonl(path / "family_summary.jsonl")) != family_rows: raise PeakFormalVerificationError("family summary mismatch")
    common = sum(all(record_id in system_stats[system_id]["success"] for system_id in promoted) for record_id in source_ids) if promoted else 0
    cohort = {"common_successful_record_count": common, "common_successful_record_fraction": common / len(source_rows), "empty_receipt_count": empty_count,
        "peak_count": global_peak, "promoted_system_count": len(promoted), "receipt_count": record_count, "source_count": len(source_rows), "warning_count": warning_count}
    stored_cohort = _read_json(path / "cohort_summary.json")
    if stored_cohort["peak_count"] != global_peak or stored_cohort["receipt_count"] != record_count or stored_cohort["promoted_system_count"] != len(promoted) or stored_cohort["common_successful_record_count"] != common:
        raise PeakFormalVerificationError("cohort summary mismatch")
    manifest = _read_json(path / "manifest.json"); identity = {"catalog_id": catalog.catalog_id, "catalog_sha256": catalog.sha256, "code_identity": _code_identity(root), "config_sha256": CONFIG_SHA256,
        "phase3_lock_sha256": config["phase3_lock"]["sha256"], "software_identity": _software_identity(), "source_projection_sha256": hashlib.sha256((path / "sources.jsonl").read_bytes()).hexdigest(),
        "system_ids": [value.system_id for value in systems]}
    run_id = hashlib.sha256(RUN_DOMAIN + _canonical(identity)).hexdigest()
    if manifest["run_id"] != run_id or manifest["code_identity"] != identity["code_identity"] or manifest["peak_count"] != global_peak or manifest["receipt_count"] != record_count: raise PeakFormalVerificationError("manifest mismatch")
    marker = _read_json(path / next(iter(markers)))
    if marker["run_id"] != run_id: raise PeakFormalVerificationError("terminal marker mismatch")
    promotion = _read_json(path / "promotion.json")
    return _Summary(path, run_id, str(manifest["status"]), bool(manifest["is_formal_run"]), len(systems), len(source_rows), record_count, global_peak, int(stored_cohort["empty_receipt_count"]), len(promotion["promoted_system_ids"]), warning_count)


_SOURCES: tuple[_Source, ...] = ()
_SYSTEMS: tuple[Phase3System, ...] = ()


def _worker(task: tuple[int, int, int]) -> tuple[int, int, list[dict[str, object]]]:
    system_index, start, stop = task; system = _SYSTEMS[system_index]; rows = []
    with threadpool_limits(limits=1):
        for source in _SOURCES[start:stop]:
            result = run_peak_detection_system(system, source.spectrum)
            rows.append({"diagnostics": _json_ready(result.diagnostics), "error_code": result.error_code, "error_message": result.error_message,
                "peaks": [peak_document(value) for value in result.peaks], "peaks_sha256": result.peaks_sha256, "status": result.status.value,
                "warnings": [(value.category, value.message) for value in result.warnings]})
    return system_index, start, rows


def _bounded_map(executor: ProcessPoolExecutor, tasks: Sequence[tuple[int, int, int]], window: int) -> Iterable[tuple[int, int, list[dict[str, object]]]]:
    iterator = iter(tasks); pending = deque()
    for _ in range(min(window, len(tasks))):
        try: pending.append(executor.submit(_worker, next(iterator)))
        except StopIteration: break
    while pending:
        yield pending.popleft().result()
        try: pending.append(executor.submit(_worker, next(iterator)))
        except StopIteration: pass


def _build_reconstruction(config: Mapping[str, object], config_raw: bytes, sources: tuple[_Source, ...], systems: tuple[Phase3System, ...], output_root: Path, worker_count: int, root: Path) -> _Summary:
    global _SOURCES, _SYSTEMS
    _SOURCES, _SYSTEMS = sources, systems; catalog = config["catalog"]; lock = config["phase3_lock"]; operational = config["operational"]; expected = config["expected"]; policy = config["policy"]
    assert isinstance(catalog, Mapping) and isinstance(lock, Mapping) and isinstance(operational, Mapping) and isinstance(expected, Mapping) and isinstance(policy, Mapping)
    source_payload = b"".join(_canonical(_source_row(value)) for value in sources)
    identity = {"catalog_id": catalog["catalog_id"], "catalog_sha256": catalog["sha256"], "code_identity": _code_identity(root), "config_sha256": CONFIG_SHA256,
        "phase3_lock_sha256": lock["sha256"], "software_identity": _software_identity(), "source_projection_sha256": hashlib.sha256(source_payload).hexdigest(), "system_ids": [value.system_id for value in systems]}
    run_id = hashlib.sha256(RUN_DOMAIN + _canonical(identity)).hexdigest(); path = output_root / f"phase3-peak-detection-v1-formal-{run_id}"; path.mkdir(parents=True)
    (path / "config.json").write_bytes(config_raw); (path / "sources.jsonl").write_bytes(source_payload); (path / "systems.jsonl").write_bytes(b"".join(_canonical(_system_row(value)) for value in systems))
    tasks = [(si, start, min(start + int(operational["source_block_size"]), len(sources))) for si in range(len(systems)) for start in range(0, len(sources), int(operational["source_block_size"]))]
    executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("fork")); chunks = _bounded_map(executor, tasks, 2 * worker_count)
    global_peak = 0; record_count = 0; warning_count = 0; empty_count = 0; stats = {value.system_id: {"status": Counter(), "peak": 0, "empty": 0, "warning_receipts": 0, "rejection": Counter(), "success": set()} for value in systems}
    with (path / "records.jsonl").open("wb") as records, (path / "peaks.jsonl").open("wb") as peaks, (path / "warnings.jsonl").open("wb") as warnings:
        try:
            for system_index, start, results in chunks:
                system = systems[system_index]
                for relative, result in enumerate(results):
                    source = sources[start + relative]; current_peaks = result["peaks"]; status = str(result["status"]); current_warnings = result["warnings"]; diagnostics = result["diagnostics"]
                    for sequence, peak in enumerate(current_peaks): peaks.write(_canonical({**peak, "record_id": source.record_id, "sequence": sequence, "system_id": system.system_id}))
                    empty = status in SUCCESS and not current_peaks; selection = {key: int(diagnostics[key]) for key in ("rejected_duplicate", "rejected_non_admissible", "rejected_out_of_range") if key in diagnostics}
                    record = {"axis_sha256": source.axis_sha256, "characterization_rejections": dict(diagnostics.get("characterization_rejections", {})), "cohort_id": source.cohort_id,
                        "diagnostics": diagnostics, "empty_output": empty, "error_code": result["error_code"], "error_message": result["error_message"], "family_id": system.family_id,
                        "intensity_sha256": source.intensity_sha256, "method_id": system.method_id, "peak_count": len(current_peaks), "peak_start": global_peak,
                        "peaks_sha256": result["peaks_sha256"], "point_count": source.point_count, "raw_candidate_count": int(diagnostics.get("candidate_count", 0)),
                        "record_id": source.record_id, "selection_rejections": selection, "source_order": source.source_order, "status": status, "system_id": system.system_id,
                        "warnings": [{"category": category, "message": message} for category, message in current_warnings]}
                    records.write(_canonical(record)); current_stats = stats[system.system_id]; current_stats["status"][status] += 1; current_stats["peak"] += len(current_peaks); current_stats["empty"] += empty
                    current_stats["warning_receipts"] += bool(current_warnings); current_stats["rejection"].update(_rejections(diagnostics));
                    if status in SUCCESS: current_stats["success"].add(source.record_id)
                    for sequence, (category, message) in enumerate(current_warnings): warnings.write(_canonical({"category": category, "message": message, "record_id": source.record_id, "sequence": sequence, "system_id": system.system_id})); warning_count += 1
                    global_peak += len(current_peaks); record_count += 1; empty_count += empty
        finally: executor.shutdown()
    promoted = []; system_rows = []
    for system in systems:
        value = stats[system.system_id]; successful = sum(value["status"][status] for status in policy["success_statuses"]); promotion = successful == len(sources)
        if promotion: promoted.append(system.system_id)
        system_rows.append({"coverage_promoted": promotion, "empty_receipt_count": value["empty"], "family_id": system.family_id, "peak_count": value["peak"], "receipt_count": len(sources),
            "rejection_totals": dict(sorted(value["rejection"].items())), "status_counts": dict(value["status"]), "successful_count": successful, "successful_fraction": successful / len(sources),
            "system_id": system.system_id, "warning_receipt_count": value["warning_receipts"]})
    (path / "system_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in system_rows))
    qualifying = []; family_rows = []
    for family, planned, runnable in (("find_peaks", 12, 12), ("find_peaks_cwt", 12, 12), ("mspd", 12, 0)):
        count = sum(value.family_id == family and value.system_id in promoted for value in systems); qualified = count >= int(policy["minimum_promoted_per_family"])
        if qualified: qualifying.append(family)
        family_rows.append({"family_id": family, "planned_system_count": planned, "promoted_system_count": count, "qualifying_family": qualified,
            "runnable_system_count": runnable, "state": "implementation_missing_algorithm_contract_mismatch" if family == "mspd" else "executed"})
    (path / "family_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in family_rows))
    warning_rows = [{"family_id": system.family_id, "receipt_count": len(sources), "system_id": system.system_id, "warning_count": 0,
        "warning_receipt_count": stats[system.system_id]["warning_receipts"], "warning_receipt_fraction": stats[system.system_id]["warning_receipts"] / len(sources)} for system in systems]
    (path / "warning_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in warning_rows))
    common = sum(all(source.record_id in stats[system_id]["success"] for system_id in promoted) for source in sources) if promoted else 0
    (path / "cohort_summary.json").write_bytes(_canonical({"common_successful_record_count": common, "common_successful_record_fraction": common / len(sources),
        "empty_receipt_count": empty_count, "peak_count": global_peak, "promoted_system_count": len(promoted), "receipt_count": record_count, "source_count": len(sources), "warning_count": warning_count}))
    (path / "promotion.json").write_bytes(_canonical({"coverage_promoted_K": len(promoted), "mspd_state": "implementation_missing_algorithm_contract_mismatch", "phase5_eligible_K": "not_evaluated",
        "phase5_power_status": config["phase5_power_status"], "planned_K": 36, "promoted_system_ids": promoted, "qualifying_family_ids": qualifying, "runnable_K": 24,
        "subset_full_tau_status": policy["subset_full_tau_status"]}))
    is_formal = len(systems) == int(expected["runnable_k"]) and len(sources) == int(expected["source_count"]) and record_count == int(expected["receipt_count"])
    (path / "gate.json").write_bytes(_canonical({"formal_execution_complete": is_formal, "phase5_power_status": config["phase5_power_status"],
        "strict_promotion": "complete_cartesian_and_100_percent_success", "subset_full_tau_status": policy["subset_full_tau_status"]}))
    status = "complete" if is_formal else "fixture_complete"; manifest = {**identity, "claim_boundary": config["claim_boundary"], "empty_receipt_count": empty_count, "is_formal_run": is_formal,
        "peak_count": global_peak, "receipt_count": record_count, "run_id": run_id, "schema_version": "phase3-peak-detection-v1-formal-artifact-v1", "source_count": len(sources),
        "status": status, "system_count": len(systems), "warning_count": warning_count}
    (path / "manifest.json").write_bytes(_canonical(manifest)); marker_name = "complete.json" if is_formal else "failed.json"; (path / marker_name).write_bytes(_canonical({"is_formal_run": is_formal, "run_id": run_id, "status": status}))
    names = sorted(value.name for value in path.iterdir() if value.is_file()); (path / "SHA256SUMS").write_text("".join(f"{_sha(path / name)}  {name}\n" for name in names))
    return _Summary(path, run_id, status, is_formal, len(systems), len(sources), record_count, global_peak, empty_count, len(promoted), warning_count)


def verify_phase3_peak_formal(path: Path, *, worker_count: int, project_root: Path = ROOT) -> _Summary:
    if worker_count <= 0: raise PeakFormalVerificationError("worker_count must be positive")
    root = Path(project_root); authority = _validate_artifact(Path(path), root)
    if not authority.is_formal_run: return authority
    config, config_raw = _read_config(root); operational = config["operational"]; assert isinstance(operational, Mapping)
    if worker_count == int(operational["worker_processes"]): raise PeakFormalVerificationError("independent worker count must differ")
    catalog_value = config["catalog"]; assert isinstance(catalog_value, Mapping); catalog = load_classical_catalog(root / str(catalog_value["path"]), project_root=root)
    systems = tuple(sorted((value for value in catalog.systems if value.task_line is TaskLine.PEAK_DETECTION and value.family_id in SCIPY_FAMILIES), key=lambda value: value.system_id))
    if hashlib.sha256(("\n".join(value.system_id for value in systems) + "\n").encode()).hexdigest() != catalog_value["scipy_system_ids_sha256"]: raise PeakFormalVerificationError("catalog projection mismatch")
    sources = _load_sources(config, root); temporary = Path(tempfile.mkdtemp(prefix=".peak-formal-verify-", dir=Path(path).parent))
    try:
        rebuilt = _build_reconstruction(config, config_raw, sources, systems, temporary, worker_count, root)
        left_names = sorted(value.relative_to(path).as_posix() for value in Path(path).rglob("*") if value.is_file()); right_names = sorted(value.relative_to(rebuilt.path).as_posix() for value in rebuilt.path.rglob("*") if value.is_file())
        if left_names != right_names: raise PeakFormalVerificationError("independent inventory mismatch")
        for name in left_names:
            left, right = Path(path) / name, rebuilt.path / name
            if left.stat().st_size != right.stat().st_size: raise PeakFormalVerificationError(f"independent size mismatch: {name}")
            with left.open("rb") as a, right.open("rb") as b:
                while True:
                    x, y = a.read(4 * 1024 * 1024), b.read(4 * 1024 * 1024)
                    if x != y: raise PeakFormalVerificationError(f"independent byte mismatch: {name}")
                    if not x: break
    finally: shutil.rmtree(temporary)
    return authority


__all__ = ["verify_phase3_peak_formal"]
