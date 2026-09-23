"""Independent full-reexecution verifier for Phase 4 D5 eligibility.

This module deliberately does not call the production ledger reconstruction,
gate aggregation, receipt serialization, or artifact builder.  It starts from
the frozen D5 loader and Phase 1 operator/native-gate primitives, constructs a
second artifact, and compares every byte with the candidate artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import h5py
import numpy as np
import scipy
import threadpoolctl
from threadpoolctl import threadpool_limits

from rpe.downstream.rruff import (
    D5RawCohort,
    load_d5_native_spectra,
    load_d5_raw_cohort,
)
from rpe.evaluation import Spectrum1D
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus, Phase1Cell
from rpe.runner.phase4_d5_eligibility import (
    ACTIVE_PERTURBATIONS,
    ALL_PERTURBATIONS,
    ALPHA_GRID,
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_STATIC_FILES,
    CODE_RELATIVE_PATHS,
    CONFIG_AUTHORITY_RELATIVE_PATH,
    DATASET_RELATIVE_PATH,
    D5_CONFIG_RELATIVE_PATH,
    EXPERIMENT_ID,
    FORBIDDEN_EXACT_KEYS,
    FORBIDDEN_KEY_FRAGMENTS,
    FULL_DOMAIN_PERTURBATIONS,
    INACTIVE_PERTURBATIONS,
    NOT_APPLICABLE_REASONS,
    PAYLOAD_PREFIX,
    PEAK_PERTURBATIONS,
    PHASE1_CONFIG_RELATIVE_PATH,
    ROOT,
    RUN_PREFIX,
    STRUCTURAL_REASON,
    SWEEP_RELATIVE_PATH,
    TERMINAL_STATES,
    Phase4D5EligibilityConfig,
    Phase4D5EligibilityError,
    Phase4D5EligibilitySummary,
    parse_phase4_d5_eligibility_config,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D5EligibilityError("independent verifier JSON", "unsupported value")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, *, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _environment() -> dict[str, object]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _code() -> dict[str, object]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _config_authority() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _load_artifact_config(path: Path) -> Phase4D5EligibilityConfig:
    raw = (path / "config.json").read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5EligibilityError("config.json", str(error)) from error
    if raw != _canonical(document):
        raise Phase4D5EligibilityError("config.json", "must be canonical JSON")
    synthetic = bool(document.get("synthetic_fixture", False))
    return parse_phase4_d5_eligibility_config(
        path / "config.json", raw, require_frozen_identity=not synthetic
    )


def _validate_authorities(config: Phase4D5EligibilityConfig) -> tuple[dict[str, object], dict[str, object]]:
    if config.synthetic_fixture:
        code = {str(key): dict(value) for key, value in config.code_authority.items()}
        environment = (
            dict(config.environment_authority)
            if config.environment_authority
            else _environment()
        )
        return code, environment
    code = _code()
    if code != {str(key): dict(value) for key, value in config.code_authority.items()}:
        raise Phase4D5EligibilityError("independent code authority", "mismatch")
    environment = _environment()
    if environment != dict(config.environment_authority):
        raise Phase4D5EligibilityError("independent environment authority", "mismatch")
    return code, environment


def _literal_ledgers(
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    config: Phase4D5EligibilityConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(native_spectra) != len(cohort.record_ids):
        raise Phase4D5EligibilityError("independent native spectra", "count mismatch")
    if tuple(spectrum.spectrum_id for spectrum in native_spectra) != tuple(
        f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids
    ):
        raise Phase4D5EligibilityError("independent native spectra", "order mismatch")
    if cohort.protocol_config_sha256 != config.authorities["d5_config_sha256"]:
        raise Phase4D5EligibilityError("independent D5 config", "identity mismatch")
    if tuple(split.split_sha256 for split in cohort.splits) != tuple(
        config.frozen_identities["split_sha256"]
    ):
        raise Phase4D5EligibilityError("independent splits", "identity mismatch")

    positions: dict[int, list[dict[str, int]]] = defaultdict(list)
    occurrences: list[dict[str, object]] = []
    for split in cohort.splits:
        query = tuple(int(value) for value in split.query_indices)
        for query_order, cohort_index in enumerate(query):
            positions[cohort_index].append(
                {"query_order": query_order, "split_seed": int(split.seed)}
            )
            occurrences.append(
                {
                    "class_label": int(cohort.class_labels[cohort_index]),
                    "cohort_index": cohort_index,
                    "group_id": cohort.group_ids[cohort_index],
                    "query_order": query_order,
                    "record_id": cohort.record_ids[cohort_index],
                    "split_query_count": len(query),
                    "split_seed": int(split.seed),
                    "split_sha256": split.split_sha256,
                }
            )
    unique_indices = sorted(positions, key=lambda index: cohort.record_ids[index])
    order = {cohort_index: index for index, cohort_index in enumerate(unique_indices)}
    for row in occurrences:
        row["unique_record_order"] = order[int(row["cohort_index"])]
    occurrences.sort(key=lambda row: (int(row["split_seed"]), int(row["query_order"])))

    records: list[dict[str, object]] = []
    for record_order, cohort_index in enumerate(unique_indices):
        spectrum = native_spectra[cohort_index]
        query_positions = sorted(
            positions[cohort_index],
            key=lambda row: (row["split_seed"], row["query_order"]),
        )
        records.append(
            {
                "class_label": int(cohort.class_labels[cohort_index]),
                "cohort_index": cohort_index,
                "group_id": cohort.group_ids[cohort_index],
                "mineral_name": cohort.mineral_names[cohort_index],
                "native_axis_sha256": _array_sha(spectrum.axis_cm1),
                "native_intensity_sha256": _array_sha(spectrum.intensity),
                "occurrence_count": len(query_positions),
                "pin_id": cohort.pin_ids[cohort_index],
                "point_count": int(spectrum.axis_cm1.size),
                "provenance": {
                    "d5_protocol_config_sha256": cohort.protocol_config_sha256,
                    "dataset_id": cohort.dataset_id,
                    "dataset_sha256sums_sha256": config.authorities.get(
                        "dataset_sha256sums_sha256", "synthetic"
                    ),
                },
                "query_positions": query_positions,
                "record_id": cohort.record_ids[cohort_index],
                "record_order": record_order,
                "rruff_id": cohort.rruff_ids[cohort_index],
                "split_seeds": sorted({row["split_seed"] for row in query_positions}),
            }
        )
    record_ids = [str(row["record_id"]) for row in records]
    group_ids = sorted({str(row["group_id"]) for row in records})
    class_ids = sorted({str(row["class_label"]) for row in records})
    observed = {
        "query_record_ids_sha256": _ids_digest(record_ids),
        "query_group_ids_sha256": _ids_digest(group_ids),
        "query_class_labels_sha256": _ids_digest(class_ids),
    }
    for key, digest in observed.items():
        if digest != config.frozen_identities[key]:
            raise Phase4D5EligibilityError(f"independent {key}", "mismatch")
    if (len(records), len(occurrences), len(group_ids), len(class_ids)) != (
        config.unique_record_count,
        config.split_occurrence_count,
        config.group_count,
        config.class_count,
    ):
        raise Phase4D5EligibilityError("independent ledgers", "count mismatch")
    return records, occurrences


def _phase1_source(record: Mapping[str, object], spectrum: Spectrum1D) -> Phase1Source:
    axis_f4 = np.ascontiguousarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.ascontiguousarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=int(record["record_order"]),
            record_id=str(record["record_id"]),
            sample_id=spectrum.sample_id or str(record["rruff_id"]),
            class_label=int(record["class_label"]),
            mineral_name=str(record["mineral_name"]),
            axis_id=f"native::{record['native_axis_sha256']}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]),
        normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance=MappingProxyType(dict(record["provenance"])),
    )


def _grid(config: Phase4D5EligibilityConfig) -> np.ndarray:
    values = np.arange(
        config.support_start_cm1,
        config.support_stop_cm1 + config.support_step_cm1 / 2.0,
        config.support_step_cm1,
        dtype="<f8",
    )
    if values.size != config.support_point_count:
        raise Phase4D5EligibilityError("independent support grid", "count mismatch")
    return values


def _project_sha(spectrum: Spectrum1D, grid: np.ndarray, max_gap: float) -> str:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5EligibilityError("independent support", "extrapolation required")
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    if left < 0 or right >= axis.size:
        raise Phase4D5EligibilityError("independent support", "invalid bounds")
    support = axis[left : right + 1]
    if support.size < 2 or float(np.max(np.diff(support))) > max_gap:
        raise Phase4D5EligibilityError("independent support", "native gap too large")
    projected = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(projected).all():
        raise Phase4D5EligibilityError("independent support", "nonfinite projection")
    return _array_sha(projected, dtype="<f4")


def _exception(cell: Phase1Cell) -> dict[str, object] | None:
    evidence = cell.evidence
    if evidence.exception_type is None:
        return None
    return {
        "message": evidence.exception_message or "unspecified",
        "path": evidence.exception_path or "unspecified",
        "type": evidence.exception_type,
    }


def _runtime_failure(
    record: Mapping[str, object],
    perturbation_id: str,
    error: BaseException,
    p10_estimate: int | None,
    state_digest: str | None = None,
) -> dict[str, object]:
    return {
        "class_label": int(record["class_label"]),
        "exception": {
            "message": str(error),
            "path": str(getattr(error, "path", "unexpected_exception")),
            "type": type(error).__name__,
        },
        "group_id": str(record["group_id"]),
        "native_gate": {},
        "output_count": 0,
        "outputs": [],
        "p10_estimated_peak_bytes": p10_estimate,
        "perturbation_id": perturbation_id,
        "reason_code": None,
        "record_id": str(record["record_id"]),
        "state": "failed_runtime",
        "state_digest": state_digest,
    }


def _receipt(
    record: Mapping[str, object],
    cell: Phase1Cell,
    config: Phase4D5EligibilityConfig,
    grid: np.ndarray,
) -> dict[str, object]:
    estimate = (
        estimate_p10_peak_bytes(int(record["point_count"]))
        if cell.perturbation_id == "p10"
        else None
    )
    state_digest = None if cell.state is None else cell.state.state_digest
    if cell.status is CellStatus.COMPLETE:
        try:
            outputs = []
            for item in cell.records:
                output = item.result.output
                support_sha = (
                    _project_sha(output, grid, config.support_max_gap_cm1)
                    if cell.perturbation_id in {"p11", "p12"}
                    else None
                )
                outputs.append(
                    {
                        "alpha": float(item.result.alpha),
                        "alpha_float64_le_hex": item.alpha_float64_le_hex,
                        "axis_changed": bool(item.result.axis_changed),
                        "diagnostics": _json_ready(item.result.diagnostics),
                        "intensity_changed": bool(item.result.intensity_changed),
                        "output_axis_sha256": _array_sha(output.axis_cm1),
                        "output_intensity_sha256": _array_sha(output.intensity),
                        "output_spectrum_id": output.spectrum_id,
                        "support_projection_sha256": support_sha,
                    }
                )
        except Exception as error:
            return _runtime_failure(
                record, cell.perturbation_id, error, estimate, state_digest
            )
        return {
            "class_label": int(record["class_label"]),
            "exception": None,
            "group_id": str(record["group_id"]),
            "native_gate": _json_ready(cell.evidence.native_gate),
            "output_count": len(outputs),
            "outputs": outputs,
            "p10_estimated_peak_bytes": estimate,
            "perturbation_id": cell.perturbation_id,
            "reason_code": None,
            "record_id": str(record["record_id"]),
            "state": "complete",
            "state_digest": state_digest,
        }
    if cell.status is CellStatus.NOT_APPLICABLE:
        if (
            cell.perturbation_id not in PEAK_PERTURBATIONS
            or cell.reason_code not in NOT_APPLICABLE_REASONS
        ):
            return _runtime_failure(
                record,
                cell.perturbation_id,
                Phase4D5EligibilityError("independent not-applicable", "taxonomy mismatch"),
                estimate,
                state_digest,
            )
        state = "not_applicable"
        reason = cell.reason_code
    else:
        state = "failed_runtime"
        reason = None
    return {
        "class_label": int(record["class_label"]),
        "exception": _exception(cell),
        "group_id": str(record["group_id"]),
        "native_gate": {},
        "output_count": 0,
        "outputs": [],
        "p10_estimated_peak_bytes": estimate,
        "perturbation_id": cell.perturbation_id,
        "reason_code": reason,
        "record_id": str(record["record_id"]),
        "state": state,
        "state_digest": state_digest,
    }


def _record_cells(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
    phase1_config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    config: Phase4D5EligibilityConfig,
    admission: P10MemoryAdmission,
    grid: np.ndarray,
) -> tuple[dict[str, object], ...]:
    source = _phase1_source(record, spectrum)
    rows = []
    for perturbation_id in ALL_PERTURBATIONS:
        if perturbation_id in INACTIVE_PERTURBATIONS:
            rows.append(
                {
                    "class_label": int(record["class_label"]),
                    "exception": None,
                    "group_id": str(record["group_id"]),
                    "native_gate": {},
                    "output_count": 0,
                    "outputs": [],
                    "p10_estimated_peak_bytes": None,
                    "perturbation_id": perturbation_id,
                    "reason_code": STRUCTURAL_REASON,
                    "record_id": str(record["record_id"]),
                    "state": "structurally_ineligible",
                    "state_digest": None,
                }
            )
            continue
        estimate = (
            estimate_p10_peak_bytes(int(record["point_count"]))
            if perturbation_id == "p10"
            else None
        )
        try:
            cell = run_perturbation_cell(
                source,
                perturbation_id,
                phase1_config,
                sweep,
                p10_admission=admission,
            )
            rows.append(_receipt(record, cell, config, grid))
        except Exception as error:
            rows.append(_runtime_failure(record, perturbation_id, error, estimate))
    return tuple(rows)


def _required(total: int, fraction: float) -> int:
    return int(math.ceil(total * fraction - 1e-15))


def _derived_ledgers(
    records: Sequence[Mapping[str, object]],
    cells: Sequence[Mapping[str, object]],
    config: Phase4D5EligibilityConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if len(records) != config.unique_record_count or len(cells) != config.expected_cell_count:
        raise Phase4D5EligibilityError("independent derived ledgers", "count mismatch")
    by_record = {str(row["record_id"]): row for row in records}
    by_cell = {
        (str(row["record_id"]), str(row["perturbation_id"])): row
        for row in cells
    }
    if len(by_record) != len(records) or len(by_cell) != len(records) * 12:
        raise Phase4D5EligibilityError("independent cell grid", "duplicate or missing key")
    class_records: dict[int, list[str]] = defaultdict(list)
    group_records: dict[str, list[str]] = defaultdict(list)
    names: dict[int, set[str]] = defaultdict(set)
    for record_id, row in by_record.items():
        label = int(row["class_label"] )
        class_records[label].append(record_id)
        group_records[str(row["group_id"] )].append(record_id)
        names[label].add(str(row["mineral_name"] ))
    if len(class_records) != config.class_count or len(group_records) != config.group_count:
        raise Phase4D5EligibilityError("independent denominators", "class/group mismatch")

    classes: list[dict[str, object]] = []
    operators: dict[str, dict[str, object]] = {}
    for perturbation_id in ALL_PERTURBATIONS:
        states = Counter(str(by_cell[(record_id, perturbation_id)]["state"]) for record_id in by_record)
        complete_records = {
            record_id
            for record_id in by_record
            if by_cell[(record_id, perturbation_id)]["state"] == "complete"
        }
        complete_class_count = 0
        for label in sorted(class_records):
            required_records = sorted(class_records[label])
            complete_count = sum(record_id in complete_records for record_id in required_records)
            complete = complete_count == len(required_records)
            complete_class_count += int(complete)
            local_states = Counter(
                str(by_cell[(record_id, perturbation_id)]["state"])
                for record_id in required_records
            )
            classes.append(
                {
                    "class_label": label,
                    "complete": complete,
                    "complete_record_count": complete_count,
                    "mineral_name": sorted(names[label])[0],
                    "perturbation_id": perturbation_id,
                    "required_record_count": len(required_records),
                    "state": (
                        "complete"
                        if complete
                        else (
                            "structurally_ineligible"
                            if perturbation_id in INACTIVE_PERTURBATIONS
                            else "closed_incomplete"
                        )
                    ),
                    "state_counts": {state: local_states.get(state, 0) for state in TERMINAL_STATES},
                }
            )
        complete_groups = sum(
            all(record_id in complete_records for record_id in record_ids)
            for record_ids in group_records.values()
        )
        complete_occurrences = sum(
            int(by_record[record_id]["occurrence_count"]) for record_id in complete_records
        )
        if perturbation_id in ("p01", "p02", "p03", "p04"):
            record_threshold = config.gates["p01_p04_record_fraction"]
            class_threshold = config.gates["p01_p04_class_fraction"]
        elif perturbation_id == "p05":
            record_threshold = config.gates["p05_record_fraction"]
            class_threshold = config.gates["p05_class_fraction"]
        elif perturbation_id in FULL_DOMAIN_PERTURBATIONS:
            record_threshold = config.gates["full_domain_record_fraction"]
            class_threshold = config.gates["full_domain_class_fraction"]
        else:
            record_threshold = class_threshold = 1.0
        required_records = _required(config.unique_record_count, record_threshold)
        required_classes = _required(config.class_count, class_threshold)
        runtime = states.get("failed_runtime", 0)
        evaluable = (
            perturbation_id not in INACTIVE_PERTURBATIONS
            and len(complete_records) >= required_records
            and complete_class_count >= required_classes
            and runtime == 0
            and (perturbation_id in PEAK_PERTURBATIONS or states.get("not_applicable", 0) == 0)
        )
        reasons = Counter(
            str(by_cell[(record_id, perturbation_id)]["reason_code"])
            for record_id in by_record
            if by_cell[(record_id, perturbation_id)]["reason_code"] is not None
        )
        operators[perturbation_id] = {
            "class_fraction": complete_class_count / config.class_count,
            "complete_class_count": complete_class_count,
            "complete_group_count": complete_groups,
            "complete_occurrence_count": complete_occurrences,
            "complete_record_count": len(complete_records),
            "group_denominator": config.group_count,
            "not_applicable_count": states.get("not_applicable", 0),
            "occurrence_denominator": config.split_occurrence_count,
            "reason_counts": dict(sorted(reasons.items())),
            "record_fraction": len(complete_records) / config.unique_record_count,
            "required_class_count": required_classes,
            "required_record_count": required_records,
            "runtime_failure_count": runtime,
            "state": (
                "evaluable"
                if evaluable
                else (
                    "structurally_ineligible"
                    if perturbation_id in INACTIVE_PERTURBATIONS
                    else "not_evaluable_coverage"
                )
            ),
            "state_counts": {state: states.get(state, 0) for state in TERMINAL_STATES},
        }

    common_records: list[dict[str, object]] = []
    common_ids: set[str] = set()
    for record_id in sorted(by_record):
        peak_states = {
            perturbation_id: str(by_cell[(record_id, perturbation_id)]["state"])
            for perturbation_id in PEAK_PERTURBATIONS
        }
        complete = all(state == "complete" for state in peak_states.values())
        if complete:
            common_ids.add(record_id)
        row = by_record[record_id]
        common_records.append(
            {
                "class_label": int(row["class_label"]),
                "complete": complete,
                "group_id": str(row["group_id"]),
                "peak_states": peak_states,
                "record_id": record_id,
                "scope": "record",
            }
        )
    common_classes: list[dict[str, object]] = []
    common_class_count = 0
    for label in sorted(class_records):
        required_records = sorted(class_records[label])
        peak_states = {
            perturbation_id: (
                "complete"
                if all(
                    by_cell[(record_id, perturbation_id)]["state"] == "complete"
                    for record_id in required_records
                )
                else "closed_incomplete"
            )
            for perturbation_id in PEAK_PERTURBATIONS
        }
        complete = all(record_id in common_ids for record_id in required_records)
        common_class_count += int(complete)
        common_classes.append(
            {
                "class_label": label,
                "complete": complete,
                "peak_states": peak_states,
                "required_record_count": len(required_records),
                "scope": "class",
            }
        )
    required_common_records = _required(
        config.unique_record_count, config.gates["peak_common_record_fraction"]
    )
    required_common_classes = _required(
        config.class_count, config.gates["peak_common_class_fraction"]
    )
    peak_operators_pass = all(operators[value]["state"] == "evaluable" for value in PEAK_PERTURBATIONS)
    peak_pass = (
        peak_operators_pass
        and len(common_ids) >= required_common_records
        and common_class_count >= required_common_classes
    )
    full_pass = all(operators[value]["state"] == "evaluable" for value in FULL_DOMAIN_PERTURBATIONS)
    gate = {
        "class_denominator": config.class_count,
        "full_domain_core": {
            "perturbation_ids": list(FULL_DOMAIN_PERTURBATIONS),
            "state": "evaluable" if full_pass else "not_evaluable_coverage",
        },
        "group_audit_denominator": config.group_count,
        "overall_status": "pass" if full_pass and peak_pass else "fail",
        "operators": operators,
        "peak_common_support": {
            "complete_class_count": common_class_count,
            "complete_record_count": len(common_ids),
            "operator_gates_passed": peak_operators_pass,
            "record_fraction": len(common_ids) / config.unique_record_count,
            "class_fraction": common_class_count / config.class_count,
            "required_class_count": required_common_classes,
            "required_record_count": required_common_records,
            "state": "evaluable" if peak_pass else "not_evaluable_coverage",
        },
        "record_denominator": config.unique_record_count,
        "split_occurrence_audit_denominator": config.split_occurrence_count,
    }
    return classes, common_records + common_classes, gate


def _run_identity(
    config: Phase4D5EligibilityConfig, code: Mapping[str, object], environment: Mapping[str, object]
) -> tuple[str, dict[str, object]]:
    identity = {
        "authorities": config.authorities,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config_authority": _config_authority(),
        "config_sha256": config.sha256,
        "denominators": {
            "classes": config.class_count,
            "groups_audit": config.group_count,
            "split_occurrences_audit": config.split_occurrence_count,
            "unique_records": config.unique_record_count,
        },
        "environment": environment,
        "frozen_identities": config.frozen_identities,
        "p10_memory_budget_bytes": config.p10_memory_budget_bytes,
        "support_grid": {
            "max_gap_cm1": config.support_max_gap_cm1,
            "point_count": config.support_point_count,
            "start_cm1": config.support_start_cm1,
            "step_cm1": config.support_step_cm1,
            "stop_cm1": config.support_stop_cm1,
        },
        "terminal_taxonomy": list(TERMINAL_STATES),
    }
    return RUN_PREFIX + _sha_bytes(_canonical(identity)), identity


def _forbidden(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_EXACT_KEYS or any(part in lowered for part in FORBIDDEN_KEY_FRAGMENTS):
                raise Phase4D5EligibilityError("independent outcome-blind boundary", f"forbidden field {key!r}")
            _forbidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _forbidden(item)


def _second_build(
    output_dir: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D5EligibilityConfig,
    worker_count: int,
) -> Phase4D5EligibilitySummary:
    if sweep.sha256 != config.authorities["sweep_sha256"] or tuple(sweep.alpha_grid) != ALPHA_GRID:
        raise Phase4D5EligibilityError("independent sweep", "identity mismatch")
    if phase1_config.file_sha256 != config.authorities["phase1_core_config_sha256"]:
        raise Phase4D5EligibilityError("independent Phase 1 config", "identity mismatch")
    code, environment = _validate_authorities(config)
    records, occurrences = _literal_ledgers(cohort, native_spectra, config)
    estimates = [estimate_p10_peak_bytes(int(row["point_count"])) for row in records]
    if max(estimates) > config.p10_memory_budget_bytes:
        raise Phase4D5EligibilityError("independent P10 admission", "record exceeds budget")
    admission = P10MemoryAdmission(config.p10_memory_budget_bytes)
    grid = _grid(config)
    native_by_index = dict(enumerate(native_spectra))
    results: dict[int, tuple[dict[str, object], ...]] = {}
    with threadpool_limits(limits=1, user_api="blas"):
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                int(record["record_order"]): executor.submit(
                    _record_cells,
                    record,
                    native_by_index[int(record["cohort_index"])],
                    phase1_config,
                    sweep,
                    config,
                    admission,
                    grid,
                )
                for record in records
            }
            for order, future in futures.items():
                results[order] = future.result()
    cells = [row for order in range(len(records)) for row in results[order]]
    classes, common, gate = _derived_ledgers(records, cells, config)
    run_id, run_identity = _run_identity(config, code, environment)
    state_counts = Counter(str(row["state"]) for row in cells)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "cells": len(cells),
            "class_summaries": len(classes),
            "common_support": len(common),
            "split_occurrences": len(occurrences),
            "unique_records": len(records),
        },
        "environment": environment,
        "experiment_id": EXPERIMENT_ID,
        "gate_status": gate["overall_status"],
        "perturbation_ids": list(ALL_PERTURBATIONS),
        "run_id": run_id,
        "run_identity": run_identity,
        "state_counts": {state: state_counts.get(state, 0) for state in TERMINAL_STATES},
        "storage": {
            "numerical_perturbation_arrays_stored": False,
            "receipts_only": True,
        },
        "synthetic_fixture": config.synthetic_fixture,
    }
    marker_name = "complete.json" if gate["overall_status"] == "pass" else "failed.json"
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": gate["overall_status"],
    }
    for value in (config.document, records, occurrences, cells, classes, common, gate, manifest, marker):
        _forbidden(value)
    payloads = {
        "config.json": config.raw_bytes,
        "unique_records.jsonl": b"".join(_canonical(row) for row in records),
        "split_occurrences.jsonl": b"".join(_canonical(row) for row in occurrences),
        "cells.jsonl": b"".join(_canonical(row) for row in cells),
        "classes.jsonl": b"".join(_canonical(row) for row in classes),
        "common_support.jsonl": b"".join(_canonical(row) for row in common),
        "gate.json": _canonical(gate),
        "manifest.json": _canonical(manifest),
        marker_name: _canonical(marker),
    }
    output_dir.mkdir(parents=True)
    members = (*PAYLOAD_PREFIX, marker_name)
    for name in members:
        (output_dir / name).write_bytes(payloads[name])
    (output_dir / "SHA256SUMS").write_bytes(
        "".join(f"{_sha_bytes(payloads[name])}  {name}\n" for name in members).encode("utf-8")
    )
    return Phase4D5EligibilitySummary(
        path=output_dir,
        run_id=run_id,
        status=str(gate["overall_status"]),
        unique_record_count=len(records),
        split_occurrence_count=len(occurrences),
        cell_count=len(cells),
        class_summary_count=len(classes),
    )


def _verify_file_set_and_checksums(path: Path) -> tuple[str, tuple[str, ...]]:
    observed = {item.name for item in path.iterdir() if item.is_file()}
    markers = observed & {"complete.json", "failed.json"}
    if len(markers) != 1:
        raise Phase4D5EligibilityError("independent marker", "must contain exactly one")
    marker = next(iter(markers))
    if observed != set(ARTIFACT_STATIC_FILES) | {marker}:
        raise Phase4D5EligibilityError("independent file set", "missing or extra file")
    members = (*PAYLOAD_PREFIX, marker)
    lines = (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if lines != [f"{_sha_file(path / name)}  {name}" for name in members]:
        raise Phase4D5EligibilityError("independent SHA256SUMS", "mismatch")
    return marker, members


def verify_phase4_d5_eligibility_from_inputs(
    path: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    worker_count: int,
) -> Phase4D5EligibilitySummary:
    path = Path(path)
    _, members = _verify_file_set_and_checksums(path)
    config = _load_artifact_config(path)
    with tempfile.TemporaryDirectory() as temporary:
        rebuilt = _second_build(
            Path(temporary) / "independent-second-build",
            cohort=cohort,
            native_spectra=native_spectra,
            sweep=sweep,
            phase1_config=phase1_config,
            config=config,
            worker_count=worker_count,
        )
        for name in (*members, "SHA256SUMS"):
            if (path / name).read_bytes() != (rebuilt.path / name).read_bytes():
                raise Phase4D5EligibilityError("independent second build", f"byte mismatch for {name}")
    return Phase4D5EligibilitySummary(
        path=path,
        run_id=rebuilt.run_id,
        status=rebuilt.status,
        unique_record_count=rebuilt.unique_record_count,
        split_occurrence_count=rebuilt.split_occurrence_count,
        cell_count=rebuilt.cell_count,
        class_summary_count=rebuilt.class_summary_count,
    )


def verify_phase4_d5_eligibility(
    path: Path,
    *,
    worker_count: int = 16,
) -> Phase4D5EligibilitySummary:
    cohort = load_d5_raw_cohort(
        ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH
    )
    native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    return verify_phase4_d5_eligibility_from_inputs(
        Path(path),
        cohort=cohort,
        native_spectra=native,
        sweep=sweep,
        phase1_config=phase1_config,
        worker_count=worker_count,
    )


__all__ = [
    "verify_phase4_d5_eligibility",
    "verify_phase4_d5_eligibility_from_inputs",
]
