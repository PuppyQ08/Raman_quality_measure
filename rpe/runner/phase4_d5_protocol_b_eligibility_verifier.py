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
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d5_protocol_b_eligibility import (
    ACTIVE_PERTURBATIONS,
    ARTIFACT_PAYLOAD_FILES,
    ARTIFACT_SCHEMA_VERSION,
    CODE_RELATIVE_PATHS,
    CONFIG_AUTHORITY_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    D5_CONFIG_RELATIVE_PATH,
    DATASET_RELATIVE_PATH,
    EXPERIMENT_ID,
    FORBIDDEN_EXACT_KEYS,
    FORBIDDEN_KEY_FRAGMENTS,
    PHASE1_CONFIG_RELATIVE_PATH,
    PROTOCOL,
    ROOT,
    RUN_PREFIX,
    STRUCTURAL_REASON,
    SWEEP_RELATIVE_PATH,
    TERMINAL_STATES,
    Phase4D5ProtocolBEligibilityError,
    Phase4D5ProtocolBEligibilitySummary,
    load_phase4_d5_protocol_b_eligibility_config,
    parse_phase4_d5_protocol_b_eligibility_config,
)


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }


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
    raise Phase4D5ProtocolBEligibilityError(
        "independent verifier JSON", "unsupported value"
    )


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray, *, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    return _sha256_bytes(array.tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _class_labels_digest(values: Sequence[int]) -> str:
    ordered = sorted({int(value) for value in values})
    return _sha256_bytes(
        ("\n".join(str(value) for value in ordered) + "\n").encode("utf-8")
    )


def _environment_document() -> dict[str, object]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _code_document() -> dict[str, Mapping[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha256_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _config_authority_document() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _validate_code_and_environment(config) -> tuple[dict[str, object], dict[str, object]]:
    if config.synthetic_fixture:
        code = {str(key): dict(value) for key, value in config.code_authority.items()}
        environment = (
            dict(config.environment_authority)
            if config.environment_authority
            else _environment_document()
        )
        return code, environment
    code = _code_document()
    expected_code = {str(key): dict(value) for key, value in config.code_authority.items()}
    if code != expected_code:
        raise Phase4D5ProtocolBEligibilityError(
            "independent code_authority", "mismatch"
        )
    environment = _environment_document()
    if environment != dict(config.environment_authority):
        raise Phase4D5ProtocolBEligibilityError(
            "independent environment_authority", "mismatch"
        )
    return code, environment


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
        source_axis_float32_sha256=_array_sha256(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha256(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]),
        normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance=MappingProxyType(
            {
                "d5_protocol_config_sha256": str(record.get("d5_protocol_config_sha256", "")),
                "dataset_id": "rruff_raman_raw",
                "record_id": str(record["record_id"]),
            }
        ),
    )


def _support_grid(config) -> np.ndarray:
    grid = np.arange(
        config.support_start_cm1,
        config.support_stop_cm1 + config.support_step_cm1 / 2.0,
        config.support_step_cm1,
        dtype="<f8",
    )
    if grid.size != config.support_point_count:
        raise Phase4D5ProtocolBEligibilityError(
            "independent support_grid", "point count mismatch"
        )
    return grid


def _project_support(
    spectrum: Spectrum1D,
    grid: np.ndarray,
    max_gap_cm1: float,
) -> tuple[str, int, float]:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5ProtocolBEligibilityError(
            "independent support projection",
            f"{spectrum.spectrum_id} would require extrapolation",
        )
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    if left < 0 or right >= axis.size:
        raise Phase4D5ProtocolBEligibilityError(
            "independent support projection", "invalid bounds"
        )
    support = axis[left : right + 1]
    max_gap = float(np.max(np.diff(support))) if support.size > 1 else math.inf
    if support.size < 2 or max_gap > max_gap_cm1:
        raise Phase4D5ProtocolBEligibilityError(
            "independent support projection", "native gap exceeds maximum"
        )
    values = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(values).all():
        raise Phase4D5ProtocolBEligibilityError(
            "independent support projection", "contains nonfinite values"
        )
    norm = float(np.linalg.norm(values.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise Phase4D5ProtocolBEligibilityError(
            "independent support projection", "zero or nonfinite norm"
        )
    return _array_sha256(values, dtype="<f4"), int(values.size), max_gap


def _reconstruct_role_ledgers(
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    config,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if len(native_spectra) != len(cohort.record_ids):
        raise Phase4D5ProtocolBEligibilityError(
            "independent native_spectra", "must align with cohort"
        )
    expected_spectrum_ids = tuple(
        f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids
    )
    if tuple(spectrum.spectrum_id for spectrum in native_spectra) != expected_spectrum_ids:
        raise Phase4D5ProtocolBEligibilityError(
            "independent native_spectra", "record order mismatch"
        )
    if cohort.protocol_config_sha256 != config.authorities["d5_config_sha256"]:
        raise Phase4D5ProtocolBEligibilityError(
            "independent cohort", "D5 config identity mismatch"
        )
    observed_splits = tuple(split.split_sha256 for split in cohort.splits)
    configured_splits = tuple(config.frozen_identities["split_sha256"])
    if observed_splits != configured_splits:
        raise Phase4D5ProtocolBEligibilityError(
            "independent splits", "frozen SHA-256 sequence mismatch"
        )

    role_occurrences: list[dict[str, object]] = []
    record_roles: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"query": [], "library": []}
    )
    for split in cohort.splits:
        query_indices = tuple(int(index) for index in split.query_indices)
        library_indices = tuple(int(index) for index in split.library_indices)
        overlap = set(query_indices) & set(library_indices)
        if overlap:
            raise Phase4D5ProtocolBEligibilityError(
                "independent splits", "query/library overlap"
            )
        if sorted(query_indices + library_indices) != list(range(len(cohort.record_ids))):
            raise Phase4D5ProtocolBEligibilityError(
                "independent splits", "must partition full cohort"
            )
        query_classes = {int(cohort.class_labels[index]) for index in query_indices}
        library_classes = {int(cohort.class_labels[index]) for index in library_indices}
        if len(query_classes) != config.class_count or len(library_classes) != config.class_count:
            raise Phase4D5ProtocolBEligibilityError(
                "independent splits", "each role must cover all classes"
            )
        for role, indices in (("query", query_indices), ("library", library_indices)):
            for role_order, cohort_index in enumerate(indices):
                record_roles[cohort_index][role].append(int(split.seed))
                role_occurrences.append(
                    {
                        "class_label": int(cohort.class_labels[cohort_index]),
                        "cohort_index": cohort_index,
                        "group_id": cohort.group_ids[cohort_index],
                        "record_id": cohort.record_ids[cohort_index],
                        "role": role,
                        "role_order": role_order,
                        "split_role_count": len(indices),
                        "split_seed": int(split.seed),
                        "split_sha256": split.split_sha256,
                    }
                )

    unique_records: list[dict[str, object]] = []
    for record_order, cohort_index in enumerate(range(len(cohort.record_ids))):
        spectrum = native_spectra[cohort_index]
        query_split_seeds = sorted(record_roles[cohort_index]["query"])
        library_split_seeds = sorted(record_roles[cohort_index]["library"])
        unique_records.append(
            {
                "class_label": int(cohort.class_labels[cohort_index]),
                "cohort_index": cohort_index,
                "group_id": cohort.group_ids[cohort_index],
                "library_occurrence_count": len(library_split_seeds),
                "library_split_seeds": library_split_seeds,
                "mineral_name": cohort.mineral_names[cohort_index],
                "native_axis_sha256": _array_sha256(spectrum.axis_cm1),
                "native_intensity_sha256": _array_sha256(spectrum.intensity),
                "pin_id": cohort.pin_ids[cohort_index],
                "point_count": int(spectrum.axis_cm1.size),
                "query_occurrence_count": len(query_split_seeds),
                "query_split_seeds": query_split_seeds,
                "record_id": cohort.record_ids[cohort_index],
                "record_order": record_order,
                "role_occurrence_count": len(query_split_seeds) + len(library_split_seeds),
                "rruff_id": cohort.rruff_ids[cohort_index],
            }
        )

    order_by_index = {
        int(row["cohort_index"]): int(row["record_order"]) for row in unique_records
    }
    for row in role_occurrences:
        row["unique_record_order"] = order_by_index[int(row["cohort_index"])]
    role_occurrences.sort(
        key=lambda row: (
            int(row["split_seed"]),
            0 if str(row["role"]) == "query" else 1,
            int(row["role_order"]),
        )
    )

    group_to_records: dict[str, list[str]] = defaultdict(list)
    groups: list[dict[str, object]] = []
    for row in unique_records:
        group_to_records[str(row["group_id"])].append(str(row["record_id"]))
    for group_id in sorted(group_to_records):
        records = sorted(group_to_records[group_id])
        query_count = sum(
            1
            for row in unique_records
            if row["group_id"] == group_id
            for _ in range(int(row["query_occurrence_count"]))
        )
        library_count = sum(
            1
            for row in unique_records
            if row["group_id"] == group_id
            for _ in range(int(row["library_occurrence_count"]))
        )
        groups.append(
            {
                "group_id": group_id,
                "library_role_occurrence_count": library_count,
                "query_role_occurrence_count": query_count,
                "record_count": len(records),
                "record_ids": records,
                "role_occurrence_count": query_count + library_count,
            }
        )

    record_ids = [str(row["record_id"]) for row in unique_records]
    group_ids = [str(row["group_id"]) for row in groups]
    class_labels = sorted({int(row["class_label"]) for row in unique_records})
    if _ids_digest(record_ids) != config.frozen_identities["record_ids_sha256"]:
        raise Phase4D5ProtocolBEligibilityError("independent record_ids", "digest mismatch")
    if _ids_digest(group_ids) != config.frozen_identities["group_ids_sha256"]:
        raise Phase4D5ProtocolBEligibilityError("independent group_ids", "digest mismatch")
    if _class_labels_digest(class_labels) != config.frozen_identities["class_labels_sha256"]:
        raise Phase4D5ProtocolBEligibilityError(
            "independent class_labels", "digest mismatch"
        )

    query_total = sum(int(row["query_occurrence_count"]) for row in unique_records)
    library_total = sum(int(row["library_occurrence_count"]) for row in unique_records)
    if (
        len(unique_records),
        query_total,
        library_total,
        len(role_occurrences),
        len(groups),
        len(class_labels),
    ) != (
        config.unique_record_count,
        config.query_role_occurrence_count,
        config.library_role_occurrence_count,
        config.role_occurrence_count,
        config.group_count,
        config.class_count,
    ):
        raise Phase4D5ProtocolBEligibilityError(
            "independent ledger counts", "do not match config"
        )
    return unique_records, role_occurrences, groups


def _cell_exception(cell) -> dict[str, object] | None:
    evidence = cell.evidence
    if evidence.exception_type is None:
        return None
    return {
        "message": evidence.exception_message or "unspecified",
        "path": evidence.exception_path or "unspecified",
        "type": evidence.exception_type,
    }


def _failed_runtime_receipt(
    record: Mapping[str, object],
    perturbation_id: str,
    error: BaseException,
    *,
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


def _cell_receipt(
    record: Mapping[str, object],
    cell,
    config,
    grid: np.ndarray,
) -> dict[str, object]:
    p10_estimate = (
        estimate_p10_peak_bytes(int(record["point_count"]))
        if cell.perturbation_id == "p10"
        else None
    )
    state_digest = None if cell.state is None else cell.state.state_digest
    if cell.status is CellStatus.COMPLETE:
        try:
            outputs = []
            for perturbed in cell.records:
                output = perturbed.result.output
                support_hash, support_point_count, support_max_gap = _project_support(
                    output,
                    grid,
                    config.support_max_gap_cm1,
                )
                outputs.append(
                    {
                        "alpha": float(perturbed.result.alpha),
                        "alpha_float64_le_hex": perturbed.alpha_float64_le_hex,
                        "axis_changed": bool(perturbed.result.axis_changed),
                        "diagnostics": _json_ready(perturbed.result.diagnostics),
                        "intensity_changed": bool(perturbed.result.intensity_changed),
                        "output_axis_sha256": _array_sha256(output.axis_cm1),
                        "output_intensity_sha256": _array_sha256(output.intensity),
                        "output_spectrum_id": output.spectrum_id,
                        "support_max_in_range_gap_cm1": support_max_gap,
                        "support_point_count": support_point_count,
                        "support_projection_sha256": support_hash,
                    }
                )
        except Exception as error:
            return _failed_runtime_receipt(
                record,
                cell.perturbation_id,
                error,
                p10_estimate=p10_estimate,
                state_digest=state_digest,
            )
        return {
            "class_label": int(record["class_label"]),
            "exception": None,
            "group_id": str(record["group_id"]),
            "native_gate": _json_ready(cell.evidence.native_gate),
            "output_count": len(outputs),
            "outputs": outputs,
            "p10_estimated_peak_bytes": p10_estimate,
            "perturbation_id": cell.perturbation_id,
            "reason_code": None,
            "record_id": str(record["record_id"]),
            "state": "complete",
            "state_digest": state_digest,
        }
    if cell.status is CellStatus.NOT_APPLICABLE:
        state = "not_applicable"
        reason_code = cell.reason_code
    else:
        state = "failed_runtime"
        reason_code = None
    return {
        "class_label": int(record["class_label"]),
        "exception": _cell_exception(cell),
        "group_id": str(record["group_id"]),
        "native_gate": {},
        "output_count": 0,
        "outputs": [],
        "p10_estimated_peak_bytes": p10_estimate,
        "perturbation_id": cell.perturbation_id,
        "reason_code": reason_code,
        "record_id": str(record["record_id"]),
        "state": state,
        "state_digest": state_digest,
    }


def _run_record_operator_cells(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
    phase1_config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    config,
    admission: P10MemoryAdmission,
    grid: np.ndarray,
) -> tuple[dict[str, object], ...]:
    source = _phase1_source(record, spectrum)
    receipts: list[dict[str, object]] = []
    for perturbation_id in ACTIVE_PERTURBATIONS:
        p10_estimate = (
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
            receipts.append(_cell_receipt(record, cell, config, grid))
        except Exception as error:
            receipts.append(
                _failed_runtime_receipt(
                    record,
                    perturbation_id,
                    error,
                    p10_estimate=p10_estimate,
                )
            )
    return tuple(receipts)


def _condition_id(perturbation_id: str, alpha: float) -> str:
    return f"{perturbation_id}:{np.float64(alpha).tobytes().hex()}"


def _build_record_conditions(
    records: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    native_spectra: tuple[Spectrum1D, ...],
    config,
    grid: np.ndarray,
) -> list[dict[str, object]]:
    native_by_record = {
        str(row["record_id"]): native_spectra[int(row["cohort_index"])] for row in records
    }
    cell_by_key = {
        (str(cell["record_id"]), str(cell["perturbation_id"])): cell
        for cell in operator_cells
    }
    conditions: list[dict[str, object]] = []
    positive_alphas = tuple(alpha for alpha in config.alpha_grid if alpha > 0.0)
    for record in records:
        record_id = str(record["record_id"])
        alpha_zero_hash, point_count, max_gap = _project_support(
            native_by_record[record_id],
            grid,
            config.support_max_gap_cm1,
        )
        conditions.append(
            {
                "alpha": 0.0,
                "alpha_float64_le_hex": np.float64(0.0).tobytes().hex(),
                "axis_sha256": str(record["native_axis_sha256"]),
                "class_label": int(record["class_label"]),
                "condition_id": "alpha0",
                "condition_kind": "alpha0",
                "group_id": str(record["group_id"]),
                "intensity_sha256": str(record["native_intensity_sha256"]),
                "perturbation_id": None,
                "record_id": record_id,
                "record_order": int(record["record_order"]),
                "state": "complete",
                "support_max_in_range_gap_cm1": max_gap,
                "support_point_count": point_count,
                "support_projection_sha256": alpha_zero_hash,
            }
        )
        alpha_zero_axis: str | None = None
        alpha_zero_intensity: str | None = None
        for perturbation_id in ACTIVE_PERTURBATIONS:
            cell = cell_by_key[(record_id, perturbation_id)]
            if str(cell["state"]) == "complete":
                outputs = list(cell["outputs"])
                alpha0 = outputs[0]
                alpha0_axis = str(alpha0["output_axis_sha256"])
                alpha0_intensity = str(alpha0["output_intensity_sha256"])
                if (
                    alpha0_axis != str(record["native_axis_sha256"])
                    or alpha0_intensity != str(record["native_intensity_sha256"])
                ):
                    raise Phase4D5ProtocolBEligibilityError(
                        "independent alpha-zero collapse",
                        "all active operators must preserve native alpha-zero bytes",
                    )
                if alpha_zero_axis is None:
                    alpha_zero_axis = alpha0_axis
                    alpha_zero_intensity = alpha0_intensity
                elif (
                    alpha_zero_axis != alpha0_axis
                    or alpha_zero_intensity != alpha0_intensity
                ):
                    raise Phase4D5ProtocolBEligibilityError(
                        "independent alpha-zero collapse",
                        "all active operators must share identical alpha-zero bytes",
                    )
                for output in outputs[1:]:
                    alpha = float(output["alpha"])
                    conditions.append(
                        {
                            "alpha": alpha,
                            "alpha_float64_le_hex": str(output["alpha_float64_le_hex"]),
                            "axis_sha256": str(output["output_axis_sha256"]),
                            "class_label": int(record["class_label"]),
                            "condition_id": _condition_id(perturbation_id, alpha),
                            "condition_kind": "positive",
                            "group_id": str(record["group_id"]),
                            "intensity_sha256": str(output["output_intensity_sha256"]),
                            "perturbation_id": perturbation_id,
                            "record_id": record_id,
                            "record_order": int(record["record_order"]),
                            "state": "complete",
                            "support_max_in_range_gap_cm1": float(
                                output["support_max_in_range_gap_cm1"]
                            ),
                            "support_point_count": int(output["support_point_count"]),
                            "support_projection_sha256": str(
                                output["support_projection_sha256"]
                            ),
                        }
                    )
            else:
                for alpha in positive_alphas:
                    conditions.append(
                        {
                            "alpha": float(alpha),
                            "alpha_float64_le_hex": np.float64(alpha).tobytes().hex(),
                            "axis_sha256": None,
                            "class_label": int(record["class_label"]),
                            "condition_id": _condition_id(perturbation_id, alpha),
                            "condition_kind": "positive",
                            "group_id": str(record["group_id"]),
                            "intensity_sha256": None,
                            "perturbation_id": perturbation_id,
                            "record_id": record_id,
                            "record_order": int(record["record_order"]),
                            "state": str(cell["state"]),
                            "support_max_in_range_gap_cm1": None,
                            "support_point_count": None,
                            "support_projection_sha256": None,
                        }
                    )
    return conditions


def _evaluate_all_role_gates(
    unique_records: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
    role_occurrences: Sequence[Mapping[str, object]],
    config,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(unique_records) != config.unique_record_count:
        raise Phase4D5ProtocolBEligibilityError(
            "independent unique_records", "denominator mismatch"
        )
    if len(operator_cells) != config.expected_operator_cell_count:
        raise Phase4D5ProtocolBEligibilityError(
            "independent operator_cells", "count mismatch"
        )
    records_by_id = {str(row["record_id"]): row for row in unique_records}
    cell_by_key = {
        (str(row["record_id"]), str(row["perturbation_id"])): row
        for row in operator_cells
    }
    expected_keys = {
        (record_id, perturbation_id)
        for record_id in records_by_id
        for perturbation_id in ACTIVE_PERTURBATIONS
    }
    if set(cell_by_key) != expected_keys:
        raise Phase4D5ProtocolBEligibilityError(
            "independent operator_cells", "grid keys mismatch"
        )
    class_to_records: dict[int, list[str]] = defaultdict(list)
    group_to_records: dict[str, list[str]] = defaultdict(list)
    for row in unique_records:
        class_to_records[int(row["class_label"])].append(str(row["record_id"]))
        group_to_records[str(row["group_id"])].append(str(row["record_id"]))

    class_rows: list[dict[str, object]] = []
    operators: dict[str, dict[str, object]] = {}
    group_audits: dict[str, dict[str, object]] = {}
    role_audits: dict[str, dict[str, object]] = {}
    for perturbation_id in ACTIVE_PERTURBATIONS:
        states = Counter(
            str(cell_by_key[(record_id, perturbation_id)]["state"])
            for record_id in records_by_id
        )
        complete_records = {
            record_id
            for record_id in records_by_id
            if cell_by_key[(record_id, perturbation_id)]["state"] == "complete"
        }
        failed_records = set(records_by_id) - complete_records
        complete_classes = 0
        for class_label in sorted(class_to_records):
            record_ids = sorted(class_to_records[class_label])
            complete_count = sum(record_id in complete_records for record_id in record_ids)
            complete = complete_count == len(record_ids)
            complete_classes += int(complete)
            class_rows.append(
                {
                    "class_label": class_label,
                    "complete": complete,
                    "complete_record_count": complete_count,
                    "perturbation_id": perturbation_id,
                    "required_record_count": len(record_ids),
                    "state": "complete" if complete else "closed_incomplete",
                    "state_counts": {
                        state: sum(
                            cell_by_key[(record_id, perturbation_id)]["state"] == state
                            for record_id in record_ids
                        )
                        for state in TERMINAL_STATES
                    },
                }
            )
        failed_groups = sum(
            not all(record_id in complete_records for record_id in group_to_records[group_id])
            for group_id in group_to_records
        )
        query_failed_count = sum(
            1
            for row in role_occurrences
            if str(row["role"]) == "query" and str(row["record_id"]) in failed_records
        )
        library_failed_count = sum(
            1
            for row in role_occurrences
            if str(row["role"]) == "library" and str(row["record_id"]) in failed_records
        )
        operators[perturbation_id] = {
            "complete_record_count": len(complete_records),
            "required_record_count": config.unique_record_count,
            "complete_class_count": complete_classes,
            "required_class_count": config.class_count,
            "failed_runtime_count": states.get("failed_runtime", 0),
            "not_applicable_count": states.get("not_applicable", 0),
            "state": (
                "evaluable"
                if len(complete_records) == config.unique_record_count
                and complete_classes == config.class_count
                and states.get("failed_runtime_count", 0) == 0
                and states.get("not_applicable", 0) == 0
                else "not_evaluable_coverage"
            ),
            "state_counts": {state: states.get(state, 0) for state in TERMINAL_STATES},
        }
        group_audits[perturbation_id] = {
            "complete_group_count": len(groups) - failed_groups,
            "failed_group_count": failed_groups,
            "required_group_count": len(groups),
        }
        role_audits[perturbation_id] = {
            "query_complete_count": config.query_role_occurrence_count - query_failed_count,
            "query_failed_count": query_failed_count,
            "library_complete_count": config.library_role_occurrence_count - library_failed_count,
            "library_failed_count": library_failed_count,
        }
    overall = all(
        operators[perturbation_id]["state"] == "evaluable"
        for perturbation_id in ACTIVE_PERTURBATIONS
    )
    gate = {
        "protocol": PROTOCOL,
        "record_denominator": config.unique_record_count,
        "class_denominator": config.class_count,
        "group_denominator": config.group_count,
        "role_occurrence_denominator": config.role_occurrence_count,
        "operators": operators,
        "group_audits": group_audits,
        "role_audits": role_audits,
        "full_domain_core": {
            "perturbation_ids": list(ACTIVE_PERTURBATIONS),
            "state": "evaluable" if overall else "not_evaluable_coverage",
        },
        "inherited_rulings": _json_ready(config.inherited_rulings),
        "overall_status": "pass" if overall else "fail",
    }
    return class_rows, gate


def _run_identity(
    config,
    code: Mapping[str, object],
    environment: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    identity = {
        "authorities": config.authorities,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config_authority": _config_authority_document(),
        "config_sha256": config.sha256,
        "denominators": {
            "classes": config.class_count,
            "groups": config.group_count,
            "library_role_occurrences": config.library_role_occurrence_count,
            "query_role_occurrences": config.query_role_occurrence_count,
            "records": config.unique_record_count,
            "role_occurrences": config.role_occurrence_count,
        },
        "environment": environment,
        "frozen_identities": config.frozen_identities,
        "support_grid": {
            "max_gap_cm1": config.support_max_gap_cm1,
            "point_count": config.support_point_count,
            "start_cm1": config.support_start_cm1,
            "step_cm1": config.support_step_cm1,
            "stop_cm1": config.support_stop_cm1,
        },
        "terminal_taxonomy": list(TERMINAL_STATES),
        "trust_anchor": config.trust_anchor,
    }
    return RUN_PREFIX + _sha256_bytes(_canonical_json_bytes(identity)), identity


def _manifest_document(
    config,
    code: Mapping[str, object],
    environment: Mapping[str, object],
    run_id: str,
    run_identity: Mapping[str, object],
    unique_records: Sequence[Mapping[str, object]],
    role_occurrences: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    record_conditions: Sequence[Mapping[str, object]],
    class_summaries: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> dict[str, object]:
    operator_states = Counter(str(row["state"]) for row in operator_cells)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_order": list(ARTIFACT_PAYLOAD_FILES),
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "class_summaries": len(class_summaries),
            "groups": len(groups),
            "operator_cells": len(operator_cells),
            "record_conditions": len(record_conditions),
            "role_occurrences": len(role_occurrences),
            "unique_records": len(unique_records),
        },
        "environment": environment,
        "experiment_id": EXPERIMENT_ID,
        "inherited_rulings": _json_ready(config.inherited_rulings),
        "protocol": PROTOCOL,
        "run_id": run_id,
        "run_identity": run_identity,
        "state_counts": {
            state: operator_states.get(state, 0) for state in TERMINAL_STATES
        },
        "synthetic_fixture": config.synthetic_fixture,
    }


def validate_protocol_b_outcome_blind_payload(
    value: object, *, path: str = "artifact"
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_EXACT_KEYS or any(
                fragment in key_text for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise Phase4D5ProtocolBEligibilityError(
                    "outcome-blind boundary",
                    f"forbidden field {key!r} at {path}",
                )
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}[{index}]")


def _compare_artifacts(observed_path: Path, rebuilt_path: Path) -> None:
    if _tree(observed_path) != _tree(rebuilt_path):
        raise Phase4D5ProtocolBEligibilityError(
            "independent verifier", "rebuilt artifact bytes differ"
        )


def verify_phase4_d5_protocol_b_eligibility_from_inputs(
    path: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    worker_count: int,
) -> Phase4D5ProtocolBEligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D5ProtocolBEligibilityError(
            "worker_count", "must be a positive integer"
        )
    path = Path(path)
    if not path.is_dir():
        raise Phase4D5ProtocolBEligibilityError("path", "must be an existing run directory")
    raw = (path / "config.json").read_bytes()
    document = json.loads(raw)
    config = parse_phase4_d5_protocol_b_eligibility_config(
        path / "config.json",
        raw,
        require_frozen_identity=not bool(document.get("synthetic_fixture", False)),
    )
    if not config.synthetic_fixture:
        frozen_path = ROOT / CONFIG_RELATIVE_PATH
        if frozen_path.is_file() and raw != frozen_path.read_bytes():
            raise Phase4D5ProtocolBEligibilityError(
                "config.json", "does not match frozen experiment config"
            )
    if sweep.sha256 != config.authorities["sweep_sha256"] or tuple(sweep.alpha_grid) != config.alpha_grid:
        raise Phase4D5ProtocolBEligibilityError(
            "independent sweep", "identity or alpha grid mismatch"
        )
    if phase1_config.file_sha256 != config.authorities["phase1_core_config_sha256"]:
        raise Phase4D5ProtocolBEligibilityError(
            "independent phase1_config", "identity mismatch"
        )
    if phase1_config.core_gate["float_relative_tolerance"] != config.native_gate_relative_tolerance:
        raise Phase4D5ProtocolBEligibilityError(
            "independent phase1_config", "native gate tolerance mismatch"
        )

    code, environment = _validate_code_and_environment(config)
    unique_records, role_occurrences, groups = _reconstruct_role_ledgers(
        cohort, native_spectra, config
    )
    estimates = [estimate_p10_peak_bytes(int(row["point_count"])) for row in unique_records]
    if max(estimates) > config.p10_memory_budget_bytes:
        raise Phase4D5ProtocolBEligibilityError(
            "independent p10 admission",
            "at least one record exceeds the frozen 64 GiB budget",
        )
    admission = P10MemoryAdmission(config.p10_memory_budget_bytes)
    native_by_index = {index: spectrum for index, spectrum in enumerate(native_spectra)}
    grid = _support_grid(config)
    results: dict[int, tuple[dict[str, object], ...]] = {}
    with threadpool_limits(limits=1, user_api="blas"):
        if worker_count == 1:
            for record in unique_records:
                order = int(record["record_order"])
                results[order] = _run_record_operator_cells(
                    record,
                    native_by_index[int(record["cohort_index"])],
                    phase1_config,
                    sweep,
                    config,
                    admission,
                    grid,
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    int(record["record_order"]): executor.submit(
                        _run_record_operator_cells,
                        record,
                        native_by_index[int(record["cohort_index"])],
                        phase1_config,
                        sweep,
                        config,
                        admission,
                        grid,
                    )
                    for record in unique_records
                }
                for order, future in futures.items():
                    results[order] = future.result()
    operator_cells = [
        row for order in range(len(unique_records)) for row in results[order]
    ]
    class_summaries, gate = _evaluate_all_role_gates(
        unique_records,
        operator_cells,
        groups,
        role_occurrences,
        config,
    )
    record_conditions = _build_record_conditions(
        unique_records,
        operator_cells,
        native_spectra,
        config,
        grid,
    )
    if len(record_conditions) != config.expected_canonical_record_condition_count and record_conditions:
        raise Phase4D5ProtocolBEligibilityError(
            "independent record_conditions", "count mismatch"
        )
    if len(class_summaries) != config.expected_class_summary_count:
        raise Phase4D5ProtocolBEligibilityError(
            "independent class_summaries", "count mismatch"
        )

    run_id, run_identity = _run_identity(config, code, environment)
    manifest = _manifest_document(
        config,
        code,
        environment,
        run_id,
        run_identity,
        unique_records,
        role_occurrences,
        groups,
        operator_cells,
        record_conditions,
        class_summaries,
        gate,
    )
    marker_name = "complete.json" if gate["overall_status"] == "pass" else "failed.json"
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": gate["overall_status"],
    }

    payload_objects = [
        config.document,
        unique_records,
        role_occurrences,
        groups,
        operator_cells,
        record_conditions,
        class_summaries,
        gate,
        manifest,
        marker,
    ]
    for value in payload_objects:
        validate_protocol_b_outcome_blind_payload(value)

    payloads = {
        "config.json": config.raw_bytes,
        "unique_records.jsonl": b"".join(
            _canonical_json_bytes(row) for row in unique_records
        ),
        "role_occurrences.jsonl": b"".join(
            _canonical_json_bytes(row) for row in role_occurrences
        ),
        "groups.jsonl": b"".join(_canonical_json_bytes(row) for row in groups),
        "operator_cells.jsonl": b"".join(
            _canonical_json_bytes(row) for row in operator_cells
        ),
        "record_conditions.jsonl": b"".join(
            _canonical_json_bytes(row) for row in record_conditions
        ),
        "class_summaries.jsonl": b"".join(
            _canonical_json_bytes(row) for row in class_summaries
        ),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker_name: _canonical_json_bytes(marker),
    }
    ordered_payloads = (*ARTIFACT_PAYLOAD_FILES, marker_name)

    with tempfile.TemporaryDirectory(prefix="phase4-d5-protocol-b-verify-") as temporary:
        rebuilt_path = Path(temporary) / run_id
        rebuilt_path.mkdir(parents=True)
        for name in ordered_payloads:
            rebuilt_path.joinpath(name).write_bytes(payloads[name])
        checksum = "".join(
            f"{_sha256_bytes(payloads[name])}  {name}\n" for name in ordered_payloads
        ).encode("utf-8")
        rebuilt_path.joinpath("SHA256SUMS").write_bytes(checksum)
        _compare_artifacts(path, rebuilt_path)

    return Phase4D5ProtocolBEligibilitySummary(
        path=path,
        run_id=run_id,
        status=str(gate["overall_status"]),
        unique_record_count=len(unique_records),
        role_occurrence_count=len(role_occurrences),
        operator_cell_count=len(operator_cells),
        record_condition_count=len(record_conditions),
        class_summary_count=len(class_summaries),
    )


def verify_phase4_d5_protocol_b_eligibility(
    path: Path,
    *,
    worker_count: int = 12,
) -> Phase4D5ProtocolBEligibilitySummary:
    load_phase4_d5_protocol_b_eligibility_config(ROOT / CONFIG_RELATIVE_PATH)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    cohort = load_d5_raw_cohort(ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH)
    native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids)
    return verify_phase4_d5_protocol_b_eligibility_from_inputs(
        Path(path),
        cohort=cohort,
        native_spectra=native,
        sweep=sweep,
        phase1_config=phase1_config,
        worker_count=worker_count,
    )


__all__ = [
    "verify_phase4_d5_protocol_b_eligibility",
    "verify_phase4_d5_protocol_b_eligibility_from_inputs",
]
