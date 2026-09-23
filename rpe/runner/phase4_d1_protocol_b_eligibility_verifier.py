from __future__ import annotations

"""Independent verifier for Phase 4 D1 Protocol-B eligibility artifacts."""

import ast
import hashlib
import json
import multiprocessing
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from typing import Mapping, Sequence

import numpy as np
import h5py
import scipy
import sklearn
import threadpoolctl

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import Spectrum1D
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = (
    "experiments/phase4/configs/d1_protocol_b_all_role_eligibility_v1.json"
)
AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d1_protocol_b_eligibility_authority.py"
DATASET_RELATIVE_PATH = "data/unified/bacteria_id_reference"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
D1_PROTOCOL_A_CONFIG_RELATIVE_PATH = (
    "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
)
CHECKSUM_FILENAME = "SHA256SUMS"
MARKER_FILENAMES = ("complete.json", "failed.json")
PROTOCOL = "B"
RUN_PREFIX = "phase4-d1-protocol-b-all-role-eligibility-"
ARTIFACT_SCHEMA_VERSION = "phase4-d1-protocol-b-all-role-eligibility-artifact-v1"
MODEL_SEEDS = (0, 1, 2, 3, 4)
ACTIVE_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
CONDITION_COUNT = 1 + len(ACTIVE_PERTURBATION_IDS) * len(POSITIVE_ALPHAS)
TERMINAL_STATES = ("complete", "not_applicable", "failed_runtime")


class Phase4D1ProtocolBEligibilityVerifierError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D1ProtocolBEligibilitySummary:
    path: Path
    run_id: str
    status: str
    endpoint_count: int
    model_seed_count: int
    source_record_count: int
    model_cell_count: int
    role_occurrence_count: int
    operator_cell_count: int
    record_condition_count: int
    class_summary_count: int
    condition_matrix_shard_count: int
    condition_matrix_bytes: int


@dataclass(frozen=True)
class _RealInputs:
    source_records: tuple[dict[str, object], ...]
    model_cells: tuple[dict[str, object], ...]
    model_role_occurrences: tuple[dict[str, object], ...]
    source_spectra: tuple[Spectrum1D, ...]
    source_projections: tuple[np.ndarray, ...]
    source_record_count: int
    model_cell_count: int
    role_occurrence_count: int
    source_ledger_sha256: str
    model_ledger_sha256: str
    role_ledger_sha256: str
    test_record_ids_sha256: str
    support_axis_f64_sha256: str
    support_axis_f32_sha256: str
    native_axis_f64_sha256: str
    native_axis_point_count: int
    support_axis_point_count: int
    role_overlap_detected: bool


@dataclass(frozen=True)
class _ExecutedSource:
    operator_cells: tuple[dict[str, object], ...]
    condition_projections: tuple[np.ndarray | None, ...]


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise Phase4D1ProtocolBEligibilityVerifierError(
        "json",
        f"unsupported value type {type(value).__name__}",
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


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(str(value) for value in values) + "\n").encode("utf-8"))


def _canonical_ledger_digest(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json_bytes(row))
    return digest.hexdigest()


def _environment_document() -> dict[str, object]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), str(error)) from error
    if not isinstance(value, dict):
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), "must be a JSON object")
    if raw != _canonical_json_bytes(value):
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), "must use canonical JSON")
    return value


def _load_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Phase4D1ProtocolBEligibilityVerifierError(
                        f"{path}:{line_number}",
                        f"invalid JSONL: {error}",
                    ) from error
                if not isinstance(value, dict):
                    raise Phase4D1ProtocolBEligibilityVerifierError(
                        f"{path}:{line_number}",
                        "JSONL rows must be objects",
                    )
                rows.append(value)
    except OSError as error:
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), str(error)) from error
    return rows


def _count_jsonl(path: Path) -> int:
    count = 0
    try:
        with path.open("rb") as stream:
            for line in stream:
                if line.strip():
                    count += 1
    except OSError as error:
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), str(error)) from error
    return count


def _require_object(path: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "must be an object")
    return value


def _require_string(path: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "must be a nonempty string")
    return value


def _require_int(path: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "must be an integer")
    if value < minimum:
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "is outside the allowed range")
    return int(value)


def _require_number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "must be a finite number")
    converted = float(value)
    if not np.isfinite(converted):
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "must be a finite number")
    return converted


def _require_float_sequence(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "must be an array")
    converted = tuple(float(item) for item in value)
    if any(not np.isfinite(item) for item in converted):
        raise Phase4D1ProtocolBEligibilityVerifierError(path, "must contain finite numbers")
    return converted


def _artifact_file_set(path: Path) -> set[str]:
    return {item.name for item in path.iterdir() if item.is_file()}


def _load_checksum_lines(path: Path) -> list[tuple[str, str]]:
    checksum_path = path / CHECKSUM_FILENAME
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Phase4D1ProtocolBEligibilityVerifierError(str(checksum_path), str(error)) from error
    parsed: list[tuple[str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not parts[1]:
            raise Phase4D1ProtocolBEligibilityVerifierError(
                f"{checksum_path}:{line_number}",
                "must be '<sha256><two spaces><filename>'",
            )
        parsed.append((parts[0], parts[1]))
    return parsed


def _validate_checksum_tree(path: Path, expected_order: list[str]) -> None:
    lines = _load_checksum_lines(path)
    observed_order = [name for _, name in lines]
    if observed_order != expected_order:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            CHECKSUM_FILENAME,
            "payload order mismatch",
        )
    for digest, name in lines:
        file_path = path / name
        if not file_path.is_file():
            raise Phase4D1ProtocolBEligibilityVerifierError(name, "listed file is missing")
        if _sha_file(file_path) != digest:
            raise Phase4D1ProtocolBEligibilityVerifierError(name, "checksum mismatch")


def _inject_shards(payload_order: list[str], shard_names: Sequence[str]) -> list[str]:
    ordered_shards = sorted(str(name) for name in shard_names)
    if any(name in payload_order for name in ordered_shards):
        return list(payload_order)
    index = payload_order.index("condition_matrix_shards.jsonl")
    return payload_order[:index] + ordered_shards + payload_order[index:]


def _artifact_orders(
    config: dict[str, object],
    manifest: dict[str, object],
    shard_names: Sequence[str],
) -> tuple[list[str], list[str]]:
    config_order_raw = config.get("artifact_payload_files")
    if not isinstance(config_order_raw, list) or not config_order_raw:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "config.artifact_payload_files",
            "must be a nonempty string array",
        )
    compact_order = [
        _require_string("config.artifact_payload_files[]", item)
        for item in config_order_raw
    ]
    dynamic_order = _inject_shards(compact_order, shard_names)
    manifest_order = manifest.get("artifact_order")
    if manifest_order is not None:
        if not isinstance(manifest_order, list):
            raise Phase4D1ProtocolBEligibilityVerifierError(
                "manifest.artifact_order",
                "must be a string array",
            )
        manifest_values = [str(item) for item in manifest_order]
        if manifest_values not in (compact_order, dynamic_order):
            raise Phase4D1ProtocolBEligibilityVerifierError(
                "manifest.artifact_order",
                "artifact order mismatch",
            )
    return compact_order, dynamic_order


def _expected_counts_from_inputs(inputs: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    fields = (
        ("source_record_count", ("source_record_count", "source_records")),
        ("model_cell_count", ("model_cell_count", "model_cells")),
        ("role_occurrence_count", ("role_occurrence_count", "model_role_occurrences")),
    )
    for target, names in fields:
        for name in names:
            if not hasattr(inputs, name):
                continue
            value = getattr(inputs, name)
            if isinstance(value, int) and not isinstance(value, bool):
                counts[target] = value
                break
            try:
                counts[target] = len(value)
                break
            except TypeError:
                continue
    return counts


def _support_projection(
    spectrum: Spectrum1D,
    support_coordinates_cm1: Sequence[float],
    support_max_gap_cm1: float,
) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f8")
    support = np.asarray(tuple(float(value) for value in support_coordinates_cm1), dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size:
        raise Phase4D1ProtocolBEligibilityVerifierError("support_projection", "axis/intensity mismatch")
    if not np.all(np.diff(axis) > 0.0):
        raise Phase4D1ProtocolBEligibilityVerifierError("support_projection", "axis must increase")
    if float(axis[0]) > float(support[0]) or float(axis[-1]) < float(support[-1]):
        raise Phase4D1ProtocolBEligibilityVerifierError("support_projection", "extrapolation required")
    left = int(np.searchsorted(axis, support[0], side="left"))
    right = int(np.searchsorted(axis, support[-1], side="right"))
    native_in_range = axis[left:right]
    if native_in_range.size < 2:
        raise Phase4D1ProtocolBEligibilityVerifierError("support_projection", "insufficient native support")
    if float(np.max(np.diff(native_in_range))) > float(support_max_gap_cm1):
        raise Phase4D1ProtocolBEligibilityVerifierError("support_projection", "native gap exceeds gate")
    projected = np.interp(support, axis, intensity)
    output = np.ascontiguousarray(projected, dtype="<f4")
    if not np.isfinite(output).all() or float(np.linalg.norm(output.astype(np.float64))) <= 0.0:
        raise Phase4D1ProtocolBEligibilityVerifierError("support_projection", "invalid projected row")
    return output


def _synthetic_base_projection(inputs: object, spectrum: Spectrum1D) -> np.ndarray:
    point_count = _require_int(
        "inputs.support_axis_point_count",
        getattr(inputs, "support_axis_point_count"),
        minimum=1,
    )
    output = np.ascontiguousarray(np.asarray(spectrum.intensity, dtype="<f8")[:point_count], dtype="<f4")
    if not np.isfinite(output).all() or float(np.linalg.norm(output.astype(np.float64))) <= 0.0:
        raise Phase4D1ProtocolBEligibilityVerifierError("support_projection", "invalid synthetic projected row")
    return output


def _alpha_hex(alpha: float) -> str:
    return np.float64(alpha).tobytes().hex()


def _condition_id(perturbation_id: str | None, alpha: float) -> str:
    if perturbation_id is None:
        return "alpha0"
    return f"{perturbation_id}:{_alpha_hex(alpha)}"


def _condition_specs(
    active_perturbation_ids: Sequence[str],
    alpha_grid: Sequence[float],
) -> tuple[tuple[int, str | None, float], ...]:
    specs: list[tuple[int, str | None, float]] = [(0, None, 0.0)]
    order = 1
    for perturbation_id in active_perturbation_ids:
        for alpha in alpha_grid[1:]:
            specs.append((order, str(perturbation_id), float(alpha)))
            order += 1
    return tuple(specs)


def _synthetic_condition_row(
    base_projection: np.ndarray,
    perturbation_id: str | None,
    alpha: float,
) -> np.ndarray:
    if perturbation_id is None or alpha == 0.0:
        return np.ascontiguousarray(base_projection, dtype="<f4")
    perturbation_index = ACTIVE_PERTURBATION_IDS.index(perturbation_id) + 1
    return np.ascontiguousarray(
        np.asarray(base_projection, dtype=np.float32)
        + np.float32(float(alpha) * perturbation_index / 100.0),
        dtype="<f4",
    )


def _run_identity(config_raw: bytes, config: dict[str, object], inputs: object) -> tuple[str, dict[str, object]]:
    storage = _require_object("config.condition_matrix_store", config.get("condition_matrix_store"))
    identity = {
        "claim_boundary": _require_string("config.claim_boundary", config.get("claim_boundary")),
        "code_authority": _require_object("config.code_authority", config.get("code_authority", {})),
        "config_sha256": _sha_bytes(config_raw),
        "ledgers": {
            "model_ledger_sha256": _require_string("inputs.model_ledger_sha256", getattr(inputs, "model_ledger_sha256")),
            "role_ledger_sha256": _require_string("inputs.role_ledger_sha256", getattr(inputs, "role_ledger_sha256")),
            "source_ledger_sha256": _require_string("inputs.source_ledger_sha256", getattr(inputs, "source_ledger_sha256")),
            "test_record_ids_sha256": _require_string("inputs.test_record_ids_sha256", getattr(inputs, "test_record_ids_sha256")),
        },
        "native": {
            "f64_sha256": _require_string("inputs.native_axis_f64_sha256", getattr(inputs, "native_axis_f64_sha256")),
            "point_count": _require_int("inputs.native_axis_point_count", getattr(inputs, "native_axis_point_count"), minimum=1),
        },
        "protocol": PROTOCOL,
        "shard_layout": {
            "condition_count": _require_int("condition_matrix_store.condition_count", storage.get("condition_count"), minimum=1),
            "dtype": _require_string("condition_matrix_store.dtype", storage.get("dtype")),
            "files": [str(item) for item in storage.get("shard_files", [])],
            "layout": _require_string("condition_matrix_store.layout", storage.get("layout")),
            "shard_source_count": _require_int("condition_matrix_store.shard_source_count", storage.get("shard_source_count"), minimum=1),
        },
        "support": {
            "f32_sha256": _require_string("inputs.support_axis_f32_sha256", getattr(inputs, "support_axis_f32_sha256")),
            "f64_sha256": _require_string("inputs.support_axis_f64_sha256", getattr(inputs, "support_axis_f64_sha256")),
            "point_count": _require_int("inputs.support_axis_point_count", getattr(inputs, "support_axis_point_count"), minimum=1),
        },
    }
    return RUN_PREFIX + _sha_bytes(_canonical_json_bytes(identity)), identity


def _synthetic_operator_cells(
    inputs: object,
    base_projections: Sequence[np.ndarray],
) -> list[dict[str, object]]:
    source_records = [dict(row) for row in getattr(inputs, "source_records")]
    source_spectra = tuple(getattr(inputs, "source_spectra"))
    native_axis_point_count = _require_int(
        "inputs.native_axis_point_count",
        getattr(inputs, "native_axis_point_count"),
        minimum=1,
    )
    cells: list[dict[str, object]] = []
    for source_record, projection, spectrum in zip(
        source_records,
        base_projections,
        source_spectra,
        strict=True,
    ):
        record_id = _require_string("source_record.record_id", source_record.get("record_id"))
        class_label = _require_int("source_record.class_label", source_record.get("class_label"), minimum=0)
        for perturbation_id in ACTIVE_PERTURBATION_IDS:
            outputs = []
            for alpha in ALPHA_GRID:
                row = _synthetic_condition_row(projection, perturbation_id, float(alpha))
                native_output = _synthetic_condition_row(
                    np.asarray(spectrum.intensity, dtype="<f4"),
                    perturbation_id,
                    float(alpha),
                )
                outputs.append(
                    {
                        "alpha": float(alpha),
                        "alpha_float64_le_hex": _alpha_hex(float(alpha)),
                        "axis_changed": False,
                        "intensity_changed": bool(alpha > 0.0),
                        "output_axis_sha256": str(source_record["native_axis_sha256"]),
                        "output_intensity_sha256": _array_sha(native_output, "<f8"),
                        "output_spectrum_id": (
                            spectrum.spectrum_id
                            if alpha == 0.0
                            else f"{spectrum.spectrum_id}::{perturbation_id}::{_alpha_hex(float(alpha))}"
                        ),
                        "support_max_in_range_gap_cm1": 0.0,
                        "support_point_count": int(row.size),
                        "support_projection_sha256": _array_sha(row, "<f4"),
                    }
                )
            cells.append(
                {
                    "class_label": class_label,
                    "exception": None,
                    "native_gate": {},
                    "output_count": len(outputs),
                    "outputs": outputs,
                    "p10_estimated_peak_bytes": (
                        estimate_p10_peak_bytes(native_axis_point_count)
                        if perturbation_id == "p10"
                        else None
                    ),
                    "perturbation_id": perturbation_id,
                    "reason_code": None,
                    "record_id": record_id,
                    "state": "complete",
                    "state_digest": _sha_bytes(f"{record_id}:{perturbation_id}".encode("utf-8")),
                }
            )
    return cells


def _evaluate_gate(
    config: dict[str, object],
    source_records: Sequence[dict[str, object]],
    model_role_occurrences: Sequence[dict[str, object]],
    operator_cells: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    endpoint_id = _require_string("config.endpoint_id", config.get("endpoint_id"))
    source_by_id = {
        _require_string("source_record.record_id", row.get("record_id")): row
        for row in source_records
    }
    cell_by_key = {
        (
            _require_string("operator_cell.record_id", row.get("record_id")),
            _require_string("operator_cell.perturbation_id", row.get("perturbation_id")),
        ): row
        for row in operator_cells
    }
    class_to_records: dict[int, list[str]] = {}
    for record_id, source_record in source_by_id.items():
        class_to_records.setdefault(
            _require_int("source_record.class_label", source_record.get("class_label"), minimum=0),
            [],
        ).append(record_id)
    operators: dict[str, object] = {}
    class_rows: list[dict[str, object]] = []
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        state_counts = {state: 0 for state in TERMINAL_STATES}
        complete_records: set[str] = set()
        for record_id in source_by_id:
            state = str(cell_by_key[(record_id, perturbation_id)]["state"])
            if state in state_counts:
                state_counts[state] += 1
            if state == "complete":
                complete_records.add(record_id)
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
                }
            )
        operators[perturbation_id] = {
            "complete_class_count": complete_classes,
            "complete_record_count": len(complete_records),
            "failed_runtime_count": state_counts["failed_runtime"],
            "not_applicable_count": state_counts["not_applicable"],
            "required_class_count": len(class_to_records),
            "required_record_count": len(source_by_id),
            "state": (
                "evaluable"
                if len(complete_records) == len(source_by_id)
                and complete_classes == len(class_to_records)
                and state_counts["failed_runtime"] == 0
                and state_counts["not_applicable"] == 0
                else "not_evaluable_coverage"
            ),
            "state_counts": dict(state_counts),
        }
    endpoint_state = (
        "evaluable"
        if all(operators[perturbation_id]["state"] == "evaluable" for perturbation_id in ACTIVE_PERTURBATION_IDS)
        else "not_evaluable_coverage"
    )
    overall_status = "pass" if endpoint_state == "evaluable" else "fail"
    gate = {
        endpoint_id: {
            "class_denominator": len(class_to_records),
            "operators": operators,
            "record_denominator": len(source_by_id),
            "role_occurrence_denominator": len(model_role_occurrences),
            "state": endpoint_state,
        },
        "inherited_rulings": config["inherited_rulings"],
        "marker_filename": "complete.json" if overall_status == "pass" else "failed.json",
        "overall_status": overall_status,
        "protocol": PROTOCOL,
    }
    return class_rows, gate


def _synthetic_expected_artifact(
    config: dict[str, object],
    config_raw: bytes,
    inputs: object,
) -> tuple[dict[str, bytes], dict[str, bytes], list[str]]:
    source_records = [dict(row) for row in getattr(inputs, "source_records")]
    model_cells = [dict(row) for row in getattr(inputs, "model_cells")]
    model_role_occurrences = [dict(row) for row in getattr(inputs, "model_role_occurrences")]
    support_grid = _require_object("config.support_grid", config.get("support_grid"))
    base_projections = [
        _synthetic_base_projection(inputs, spectrum)
        for spectrum in tuple(getattr(inputs, "source_spectra"))
    ]
    operator_cells = _synthetic_operator_cells(inputs, base_projections)
    class_summaries, gate = _evaluate_gate(config, source_records, model_role_occurrences, operator_cells)
    marker_name = _require_string("gate.marker_filename", gate.get("marker_filename"))
    run_id, run_identity = _run_identity(config_raw, config, inputs)
    condition_specs = _condition_specs(ACTIVE_PERTURBATION_IDS, ALPHA_GRID)
    shard_source_count = _require_int(
        "config.condition_matrix_store.shard_source_count",
        _require_object("config.condition_matrix_store", config.get("condition_matrix_store")).get("shard_source_count"),
        minimum=1,
    )
    support_point_count = _require_int("config.support_grid.point_count", support_grid.get("point_count"), minimum=1)
    cell_by_key = {
        (
            _require_string("operator_cell.record_id", row.get("record_id")),
            _require_string("operator_cell.perturbation_id", row.get("perturbation_id")),
        ): row
        for row in operator_cells
    }
    record_conditions: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []
    shard_payloads: dict[str, bytes] = {}
    total_bytes = 0
    for shard_id, start in enumerate(range(0, len(source_records), shard_source_count)):
        end = min(start + shard_source_count, len(source_records)) - 1
        rows_per_shard = end - start + 1
        filename = f"condition_matrices_{start:05d}_{end:05d}.f32le"
        rows_by_source: list[list[tuple[dict[str, object], bytes]]] = []
        for source_order in range(start, end + 1):
            source_rows: list[tuple[dict[str, object], bytes]] = []
            for condition_order, perturbation_id, alpha in condition_specs:
                source_record = source_records[source_order]
                record_id = _require_string("source_record.record_id", source_record.get("record_id"))
                state = "complete"
                if perturbation_id is not None:
                    state = str(cell_by_key[(record_id, perturbation_id)]["state"])
                complete = state == "complete"
                row_bytes = (
                    _synthetic_condition_row(base_projections[source_order], perturbation_id, alpha).tobytes(order="C")
                    if complete else bytes(support_point_count * 4)
                )
                projection_sha256 = _sha_bytes(row_bytes) if complete else None
                if perturbation_id is None:
                    output_axis_sha256 = str(source_record["native_axis_sha256"])
                    output_intensity_sha256 = str(source_record["native_intensity_sha256"])
                    output_spectrum_id = tuple(getattr(inputs, "source_spectra"))[
                        source_order
                    ].spectrum_id
                elif complete:
                    detail = cell_by_key[(record_id, perturbation_id)]["outputs"][
                        ALPHA_GRID.index(alpha)
                    ]
                    output_axis_sha256 = str(detail["output_axis_sha256"])
                    output_intensity_sha256 = str(detail["output_intensity_sha256"])
                    output_spectrum_id = str(detail["output_spectrum_id"])
                else:
                    output_axis_sha256 = None
                    output_intensity_sha256 = None
                    output_spectrum_id = None
                local_order = source_order - start
                offset = ((condition_order * rows_per_shard + local_order) * support_point_count * 4)
                receipt = {
                        "alpha": float(alpha),
                        "alpha_float64_le_hex": _alpha_hex(alpha),
                        "class_label": _require_int("source_record.class_label", source_record.get("class_label"), minimum=0),
                        "condition_id": _condition_id(perturbation_id, alpha),
                        "condition_order": condition_order,
                        "filename": filename,
                        "native_axis_sha256": str(source_record["native_axis_sha256"]),
                        "native_intensity_sha256": str(source_record["native_intensity_sha256"]),
                        "output_axis_sha256": output_axis_sha256,
                        "output_intensity_sha256": output_intensity_sha256,
                        "output_spectrum_id": output_spectrum_id,
                        "padding_not_scientific_output": not complete,
                        "perturbation_id": perturbation_id,
                        "record_id": record_id,
                        "record_order": source_order,
                        "shard_byte_offset": offset,
                        "shard_id": shard_id,
                        "shard_local_source_order": local_order,
                        "source_row": _require_int("source_record.source_row", source_record.get("source_row"), minimum=0),
                        "source_split": _require_string("source_record.source_split", source_record.get("source_split")),
                        "state": state,
                        "support_point_count": support_point_count,
                        "support_projection_sha256": projection_sha256,
                }
                record_conditions.append(receipt)
                source_rows.append((receipt, row_bytes))
            rows_by_source.append(source_rows)
        parts = [
            rows_by_source[local_order][condition_order][1]
            for condition_order in range(len(condition_specs))
            for local_order in range(rows_per_shard)
        ]
        shard_bytes = b"".join(parts)
        shard_payloads[filename] = shard_bytes
        shard_rows.append(
            {
                "byte_count": len(shard_bytes),
                "condition_count": len(condition_specs),
                "dtype": "little_endian_float32",
                "end_source_order": end,
                "filename": filename,
                "layout": "condition_major_source_major_support",
                "rows_per_shard": rows_per_shard,
                "sha256": _sha_bytes(shard_bytes),
                "shard_id": shard_id,
                "start_source_order": start,
                "support_point_count": support_point_count,
            }
        )
        total_bytes += len(shard_bytes)
    compact_order = [
        _require_string("config.artifact_payload_files[]", item)
        for item in config["artifact_payload_files"]
    ]
    manifest = {
        "artifact_order": list(compact_order),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": _require_object("config.authorities", config.get("authorities", {})),
        "claim_boundary": _require_string("config.claim_boundary", config.get("claim_boundary")),
        "code_authority": _require_object("config.code_authority", config.get("code_authority", {})),
        "condition_matrix_bytes": total_bytes,
        "condition_matrix_shard_count": len(shard_rows),
        "counts": {
            "class_summaries": len(class_summaries),
            "model_cells": len(model_cells),
            "model_role_occurrences": len(model_role_occurrences),
            "operator_cells": len(operator_cells),
            "record_conditions": len(record_conditions),
            "source_records": len(source_records),
        },
        "endpoint_count": _require_int("config.denominators.endpoint_count", config["denominators"]["endpoint_count"], minimum=1),
        "environment_authority": _require_object("config.environment_authority", config.get("environment_authority", {})),
        "protocol": PROTOCOL,
        "run_id": run_id,
        "run_identity": run_identity,
        "status": _require_string("gate.overall_status", gate.get("overall_status")),
        "synthetic_fixture": bool(config.get("synthetic_fixture", False)),
        "trust_anchor": _require_object("config.trust_anchor", config.get("trust_anchor", {})),
    }
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": _require_string("gate.overall_status", gate.get("overall_status")),
    }
    payloads = {
        "config.json": config_raw,
        "source_records.jsonl": _jsonl_bytes(source_records),
        "model_cells.jsonl": _jsonl_bytes(model_cells),
        "model_role_occurrences.jsonl": _jsonl_bytes(model_role_occurrences),
        "operator_cells.jsonl": _jsonl_bytes(operator_cells),
        "record_conditions.jsonl": _jsonl_bytes(record_conditions),
        "class_summaries.jsonl": _jsonl_bytes(class_summaries),
        "condition_matrix_shards.jsonl": _jsonl_bytes(shard_rows),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker_name: _canonical_json_bytes(marker),
    }
    checksum_order = _inject_shards(compact_order, sorted(shard_payloads)) + [marker_name]
    checksum_lines = []
    for name in checksum_order:
        digest = _sha_bytes(payloads[name]) if name in payloads else _sha_bytes(shard_payloads[name])
        checksum_lines.append(f"{digest}  {name}")
    payloads[CHECKSUM_FILENAME] = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    payloads.update(shard_payloads)
    return payloads, shard_payloads, checksum_order


def _load_anchor_constants(path: Path) -> tuple[int, str]:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), str(error)) from error
    config_bytes: int | None = None
    config_sha256: str | None = None
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "CONFIG_BYTES":
            config_bytes = ast.literal_eval(statement.value)
        elif target.id == "CONFIG_SHA256":
            config_sha256 = ast.literal_eval(statement.value)
    if not isinstance(config_bytes, int) or not isinstance(config_sha256, str):
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), "authority must define CONFIG_BYTES and CONFIG_SHA256")
    return config_bytes, config_sha256


def _support_coordinates_from_config(
    config: dict[str, object],
    *,
    synthetic_fixture: bool,
) -> tuple[float, ...]:
    support_grid = _require_object("support_grid", config.get("support_grid"))
    coordinates = support_grid.get("coordinates_cm1")
    if coordinates is not None:
        return _require_float_sequence("support_grid.coordinates_cm1", coordinates)
    if synthetic_fixture:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "support_grid.coordinates_cm1",
            "synthetic fixtures must provide explicit support coordinates",
        )
    bound_config = _load_json(ROOT / D1_PROTOCOL_A_CONFIG_RELATIVE_PATH)
    bound_support = _require_object("d1_protocol_a.support_grid", bound_config.get("support_grid"))
    return _require_float_sequence(
        "d1_protocol_a.support_grid.coordinates_cm1",
        bound_support.get("coordinates_cm1"),
    )


def _validate_authority_receipts(config: dict[str, object], *, synthetic_fixture: bool) -> None:
    authorities = _require_object("authorities", config.get("authorities", {}))
    code_authority = _require_object("code_authority", config.get("code_authority", {}))
    environment_authority = _require_object("environment_authority", config.get("environment_authority", {}))
    trust_anchor = _require_object("trust_anchor", config.get("trust_anchor", {}))
    if synthetic_fixture:
        if authorities or code_authority or environment_authority or trust_anchor:
            raise Phase4D1ProtocolBEligibilityVerifierError(
                "synthetic authority",
                "synthetic fixtures may only use empty authority mappings",
            )
        return
    if trust_anchor:
        authority_rel = _require_string(
            "trust_anchor.config_authority_relative_path",
            trust_anchor.get("config_authority_relative_path"),
        )
        if authority_rel != AUTHORITY_RELATIVE_PATH:
            raise Phase4D1ProtocolBEligibilityVerifierError("trust_anchor", "authority path mismatch")
        if trust_anchor.get("config_binds_authority") not in (False, None):
            raise Phase4D1ProtocolBEligibilityVerifierError("trust_anchor", "config must not bind authority module")
        if trust_anchor.get("direction") not in ("authority_to_config_only", None):
            raise Phase4D1ProtocolBEligibilityVerifierError("trust_anchor", "unsupported authority direction")
        _load_anchor_constants(ROOT / authority_rel)
    for key, value in authorities.items():
        authority = _require_object(f"authorities.{key}", value)
        receipt_path = ROOT / _require_string(f"authorities.{key}.path", authority.get("path"))
        receipt_bytes = _require_int(f"authorities.{key}.bytes", authority.get("bytes"), minimum=1)
        receipt_sha256 = _require_string(f"authorities.{key}.sha256", authority.get("sha256"))
        if not receipt_path.is_file():
            raise Phase4D1ProtocolBEligibilityVerifierError(f"authorities.{key}", "authority path is missing")
        if receipt_path.stat().st_size != receipt_bytes or _sha_file(receipt_path) != receipt_sha256:
            raise Phase4D1ProtocolBEligibilityVerifierError(f"authorities.{key}", "live authority mismatch")
    if code_authority:
        observed = {
            relative_path: {
                "bytes": (ROOT / relative_path).stat().st_size,
                "sha256": _sha_file(ROOT / relative_path),
            }
            for relative_path in code_authority
        }
        if code_authority != observed:
            raise Phase4D1ProtocolBEligibilityVerifierError("code_authority", "mismatch")
    if environment_authority and environment_authority != _environment_document():
        raise Phase4D1ProtocolBEligibilityVerifierError("environment_authority", "mismatch")


def _validate_frozen_config_anchor(config_path: Path, config_raw: bytes, *, synthetic_fixture: bool) -> None:
    if synthetic_fixture:
        return
    trust_anchor = _require_object("trust_anchor", _load_json(config_path).get("trust_anchor", {}))
    if not trust_anchor:
        authority_path = ROOT / AUTHORITY_RELATIVE_PATH
    else:
        authority_path = ROOT / _require_string(
            "trust_anchor.config_authority_relative_path",
            trust_anchor.get("config_authority_relative_path"),
        )
    config_bytes, config_sha256 = _load_anchor_constants(authority_path)
    if config_bytes <= 0 or set(config_sha256) == {"0"}:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "frozen config identity",
            "config unavailable or untrusted",
        )
    if len(config_raw) != config_bytes or _sha_bytes(config_raw) != config_sha256:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "frozen config identity",
            "bytes or SHA-256 mismatch",
        )


def _split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    class_labels = sorted({int(value) for value in labels.tolist()})
    for class_label in class_labels:
        indices = np.flatnonzero(labels == class_label)
        if len(indices) != 100:
            raise Phase4D1ProtocolBEligibilityVerifierError(
                "D1 inputs",
                f"finetune class {class_label} must contain 100 records",
            )
        shuffled = generator.permutation(indices)
        validation.extend(shuffled[:10].tolist())
        train.extend(shuffled[10:].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def _reconstruct_real_inputs(config: dict[str, object]) -> _RealInputs:
    support_grid = _require_object("support_grid", config.get("support_grid"))
    support_coordinates = _support_coordinates_from_config(config, synthetic_fixture=False)
    support_point_count = _require_int(
        "support_grid.point_count",
        support_grid.get("point_count"),
        minimum=1,
    )
    if len(support_coordinates) != support_point_count:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "support_grid",
            "point count mismatch",
        )
    support_max_gap = _require_number(
        "support_grid.max_in_range_native_gap_cm1",
        support_grid.get("max_in_range_native_gap_cm1"),
    )
    frozen_identities = _require_object("frozen_identities", config.get("frozen_identities", {}))
    dataset_path = ROOT / DATASET_RELATIVE_PATH
    if not dataset_path.is_dir():
        raise Phase4D1ProtocolBEligibilityVerifierError(str(dataset_path), "retained dataset directory required")

    split_records: dict[str, list[dict[str, object]]] = {"finetune": [], "reference": [], "test": []}
    split_spectra: dict[str, list[Spectrum1D]] = {"finetune": [], "reference": [], "test": []}
    split_projections: dict[str, list[np.ndarray]] = {"finetune": [], "reference": [], "test": []}
    split_native_f32: dict[str, list[np.ndarray]] = {"finetune": [], "reference": [], "test": []}
    split_labels: dict[str, list[int]] = {"finetune": [], "reference": [], "test": []}
    split_ids: dict[str, list[str]] = {"finetune": [], "reference": [], "test": []}
    split_rows: dict[str, list[int]] = {"finetune": [], "reference": [], "test": []}
    source_spectra: list[Spectrum1D] = []
    source_projections: list[np.ndarray] = []
    native_axis: np.ndarray | None = None
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            split = str(batch.source_split)
            if split not in split_records:
                raise Phase4D1ProtocolBEligibilityVerifierError("BacteriaIdBatchLoader", "unexpected split")
            stored_axis = np.asarray(batch.wavenumber, dtype="<f4")
            axis = np.ascontiguousarray(stored_axis[::-1], dtype="<f8")
            if not np.all(np.diff(axis) > 0.0):
                raise Phase4D1ProtocolBEligibilityVerifierError("D1 inputs", "native axis reversal failed")
            if native_axis is None:
                native_axis = axis
            elif not np.array_equal(native_axis, axis):
                raise Phase4D1ProtocolBEligibilityVerifierError("D1 inputs", "shared native axis mismatch")
            for record_id, label, source_row, intensity in zip(
                batch.record_ids,
                batch.class_labels,
                batch.source_rows,
                batch.intensity,
                strict=True,
            ):
                record_id = str(record_id)
                native_intensity = np.ascontiguousarray(np.asarray(intensity, dtype="<f4")[::-1], dtype="<f8")
                spectrum = Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{record_id}",
                    sample_id=None,
                    axis_cm1=axis,
                    intensity=native_intensity,
                )
                projected = _support_projection(spectrum, support_coordinates, support_max_gap)
                split_records[split].append(
                    {
                        "class_label": int(label),
                        "native_axis_sha256": _array_sha(axis, "<f8"),
                        "native_intensity_sha256": _array_sha(native_intensity, "<f8"),
                        "record_id": record_id,
                        "record_order": 0,
                        "scope": split,
                        "source_row": int(source_row),
                        "source_split": split,
                        "support_projection_sha256": _array_sha(projected, "<f4"),
                    }
                )
                split_spectra[split].append(spectrum)
                split_projections[split].append(projected)
                split_native_f32[split].append(np.ascontiguousarray(intensity, dtype="<f4"))
                split_labels[split].append(int(label))
                split_ids[split].append(record_id)
                split_rows[split].append(int(source_row))
    if native_axis is None:
        raise Phase4D1ProtocolBEligibilityVerifierError("D1 inputs", "dataset is empty")
    support_axis_f64_sha256 = _array_sha(np.asarray(support_coordinates, dtype="<f8"), "<f8")
    support_axis_f32_sha256 = _array_sha(np.asarray(support_coordinates, dtype="<f8"), "<f4")
    if (
        "support_axis_f64_sha256" in frozen_identities
        and support_axis_f64_sha256 != str(frozen_identities["support_axis_f64_sha256"])
    ):
        raise Phase4D1ProtocolBEligibilityVerifierError("support_axis_f64_sha256", "support-axis digest mismatch")
    if (
        "support_axis_f32_sha256" in frozen_identities
        and support_axis_f32_sha256 != str(frozen_identities["support_axis_f32_sha256"])
    ):
        raise Phase4D1ProtocolBEligibilityVerifierError("support_axis_f32_sha256", "support-axis digest mismatch")
    expected_split_counts = {"finetune": 3000, "reference": 60000, "test": 3000}
    observed_split_counts = {key: len(value) for key, value in split_records.items()}
    if observed_split_counts != expected_split_counts:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "source_records",
            "source split cardinality mismatch",
        )
    test_digest = _ids_digest(split_ids["test"])
    if (
        "test_record_ids_sha256" in frozen_identities
        and test_digest != str(frozen_identities["test_record_ids_sha256"])
    ):
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "test_record_ids_sha256",
            "test identity mismatch",
        )

    source_records: list[dict[str, object]] = []
    for split in ("finetune", "reference", "test"):
        for row, spectrum, projected in zip(
            split_records[split],
            split_spectra[split],
            split_projections[split],
            strict=True,
        ):
            updated = {**row, "record_order": len(source_records)}
            source_records.append(updated)
            source_spectra.append(spectrum)
            source_projections.append(projected)

    source_rows_for_digest = [
        {
            "class_label": int(row["class_label"]),
            "record_id": str(row["record_id"]),
            "source_row": int(row["source_row"]),
            "source_split": str(row["source_split"]),
        }
        for row in source_records
    ]
    source_digest = _canonical_ledger_digest(source_rows_for_digest)
    if "source_ledger_sha256" in frozen_identities and source_digest != str(frozen_identities["source_ledger_sha256"]):
        raise Phase4D1ProtocolBEligibilityVerifierError("source_ledger_sha256", "frozen source ledger mismatch")

    model_rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []
    role_overlap = False
    reference_native = np.asarray(split_native_f32["reference"], dtype="<f4")
    finetune_native = np.asarray(split_native_f32["finetune"], dtype="<f4")
    test_native = np.asarray(split_native_f32["test"], dtype="<f4")
    reference_projected = np.asarray(split_projections["reference"], dtype="<f4")
    finetune_projected = np.asarray(split_projections["finetune"], dtype="<f4")
    test_projected = np.asarray(split_projections["test"], dtype="<f4")
    reference_labels = np.asarray(split_labels["reference"], dtype="<i8")
    finetune_labels = np.asarray(split_labels["finetune"], dtype="<i8")
    test_labels = np.asarray(split_labels["test"], dtype="<i8")
    for seed in MODEL_SEEDS:
        train_idx, valid_idx = _split_indices(finetune_labels, seed)
        train_ids = tuple(split_ids["reference"] + [split_ids["finetune"][int(index)] for index in train_idx])
        valid_ids = tuple(split_ids["finetune"][int(index)] for index in valid_idx)
        test_ids = tuple(split_ids["test"])
        if any(set(a) & set(b) for a, b in ((train_ids, valid_ids), (train_ids, test_ids), (valid_ids, test_ids))):
            role_overlap = True
        source_hashes = {
            "train": {
                _sha_bytes(row.tobytes())
                for row in np.concatenate((reference_native, finetune_native[train_idx]), axis=0)
            },
            "validation": {
                _sha_bytes(row.tobytes())
                for row in finetune_native[valid_idx]
            },
            "test": {_sha_bytes(row.tobytes()) for row in test_native},
        }
        projected_hashes = {
            "train": {
                _sha_bytes(row.tobytes())
                for row in np.concatenate((reference_projected, finetune_projected[train_idx]), axis=0)
            },
            "validation": {
                _sha_bytes(row.tobytes())
                for row in finetune_projected[valid_idx]
            },
            "test": {_sha_bytes(row.tobytes()) for row in test_projected},
        }
        if any(
            left & right
            for groups in (source_hashes, projected_hashes)
            for left, right in (
                (groups["train"], groups["validation"]),
                (groups["train"], groups["test"]),
                (groups["validation"], groups["test"]),
            )
        ):
            raise Phase4D1ProtocolBEligibilityVerifierError(
                "D1 inputs",
                "exact cross-role duplicate detected",
            )
        model_rows.append(
            {
                "model_seed": int(seed),
                "test_count": len(test_ids),
                "test_record_ids_sha256": _ids_digest(test_ids),
                "train_count": len(train_ids),
                "train_record_ids_sha256": _ids_digest(train_ids),
                "validation_count": len(valid_ids),
                "validation_record_ids_sha256": _ids_digest(valid_ids),
            }
        )
        role_specs = (
            (
                "train",
                train_ids,
                tuple(reference_labels.tolist()) + tuple(finetune_labels[train_idx].tolist()),
                tuple(split_rows["reference"]) + tuple(split_rows["finetune"][int(index)] for index in train_idx),
                ("reference",) * len(split_rows["reference"]) + ("finetune",) * len(train_idx),
            ),
            (
                "validation",
                valid_ids,
                tuple(finetune_labels[valid_idx].tolist()),
                tuple(split_rows["finetune"][int(index)] for index in valid_idx),
                ("finetune",) * len(valid_idx),
            ),
            ("test", test_ids, tuple(test_labels.tolist()), tuple(split_rows["test"]), ("test",) * len(split_rows["test"])),
        )
        for role, ids, labels, rows, splits in role_specs:
            for order, (record_id, label, source_row, split) in enumerate(zip(ids, labels, rows, splits, strict=True)):
                role_rows.append(
                    {
                        "class_label": int(label),
                        "model_seed": int(seed),
                        "record_id": str(record_id),
                        "role": str(role),
                        "role_order": int(order),
                        "source_row": int(source_row),
                        "source_split": str(split),
                    }
                )
    model_digest = _canonical_ledger_digest(model_rows)
    role_digest = _canonical_ledger_digest(role_rows)
    if "model_ledger_sha256" in frozen_identities and model_digest != str(frozen_identities["model_ledger_sha256"]):
        raise Phase4D1ProtocolBEligibilityVerifierError("model_ledger_sha256", "frozen model ledger mismatch")
    if "role_ledger_sha256" in frozen_identities and role_digest != str(frozen_identities["role_ledger_sha256"]):
        raise Phase4D1ProtocolBEligibilityVerifierError("role_ledger_sha256", "frozen role ledger mismatch")
    native_digest = _array_sha(native_axis, "<f8")
    if "native_axis_increasing_f64_sha256" in frozen_identities and native_digest != str(frozen_identities["native_axis_increasing_f64_sha256"]):
        raise Phase4D1ProtocolBEligibilityVerifierError("native_axis_increasing_f64_sha256", "native axis mismatch")
    return _RealInputs(
        source_records=tuple(source_records),
        model_cells=tuple(model_rows),
        model_role_occurrences=tuple(role_rows),
        source_spectra=tuple(source_spectra),
        source_projections=tuple(source_projections),
        source_record_count=len(source_records),
        model_cell_count=len(model_rows),
        role_occurrence_count=len(role_rows),
        source_ledger_sha256=source_digest,
        model_ledger_sha256=model_digest,
        role_ledger_sha256=role_digest,
        test_record_ids_sha256=test_digest,
        support_axis_f64_sha256=support_axis_f64_sha256,
        support_axis_f32_sha256=support_axis_f32_sha256,
        native_axis_f64_sha256=native_digest,
        native_axis_point_count=int(native_axis.size),
        support_axis_point_count=support_point_count,
        role_overlap_detected=role_overlap,
    )


def _compare_bytes(path: Path, expected: bytes) -> None:
    actual = path.read_bytes()
    if actual != expected:
        raise Phase4D1ProtocolBEligibilityVerifierError(str(path), "rebuild payload mismatch")


def _compare_stream_line(
    stream: BinaryIO,
    *,
    expected: bytes,
    label: str,
    index: int,
) -> None:
    actual = stream.readline()
    if actual != expected:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            f"{label}:{index}",
            "rebuild payload mismatch",
        )


def _ensure_stream_eof(stream: BinaryIO, label: str) -> None:
    if stream.readline():
        raise Phase4D1ProtocolBEligibilityVerifierError(
            label,
            "unexpected trailing payload rows",
        )


def _compare_jsonl_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    with path.open("rb") as stream:
        for index, row in enumerate(rows, start=1):
            _compare_stream_line(
                stream,
                expected=_canonical_json_bytes(row),
                label=path.name,
                index=index,
            )
        _ensure_stream_eof(stream, path.name)


def _phase1_source(record: Mapping[str, object], spectrum: Spectrum1D) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=int(record["record_order"]),
            record_id=str(record["record_id"]),
            sample_id=str(record["record_id"]),
            class_label=int(record["class_label"]),
            mineral_name=f"bacteria-{record['class_label']}",
            axis_id=f"native::{record['native_axis_sha256']}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, "<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, "<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]),
        normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance={"dataset_id": "bacteria_id_reference", "record_id": str(record["record_id"])},
    )


def _cell_exception(cell: object) -> dict[str, object] | None:
    evidence = getattr(cell, "evidence")
    exception_type = getattr(evidence, "exception_type")
    if exception_type is None:
        return None
    return {
        "message": getattr(evidence, "exception_message") or "unspecified",
        "path": getattr(evidence, "exception_path") or "unspecified",
        "type": exception_type,
    }


def _cell_receipt(
    record: Mapping[str, object],
    cell: object,
    support_coordinates_cm1: Sequence[float],
    support_max_gap_cm1: float,
    *,
    p10_estimate: int | None,
) -> tuple[dict[str, object], tuple[np.ndarray, ...]]:
    if getattr(cell, "status") is CellStatus.COMPLETE:
        outputs: list[dict[str, object]] = []
        projections: list[np.ndarray] = []
        for perturbed in getattr(cell, "records"):
            result = getattr(perturbed, "result")
            output = getattr(result, "output")
            projection = _support_projection(
                output,
                support_coordinates_cm1,
                support_max_gap_cm1,
            )
            projections.append(projection)
            axis = np.asarray(output.axis_cm1, dtype="<f8")
            left = int(np.searchsorted(axis, support_coordinates_cm1[0], side="left"))
            right = int(np.searchsorted(axis, support_coordinates_cm1[-1], side="right"))
            in_range = axis[left:right]
            outputs.append(
                {
                    "alpha": float(getattr(result, "alpha")),
                    "alpha_float64_le_hex": getattr(perturbed, "alpha_float64_le_hex"),
                    "axis_changed": bool(getattr(result, "axis_changed")),
                    "intensity_changed": bool(getattr(result, "intensity_changed")),
                    "output_axis_sha256": _array_sha(output.axis_cm1, "<f8"),
                    "output_intensity_sha256": _array_sha(output.intensity, "<f8"),
                    "output_spectrum_id": output.spectrum_id,
                    "support_max_in_range_gap_cm1": float(np.max(np.diff(in_range))),
                    "support_point_count": int(projection.size),
                    "support_projection_sha256": _array_sha(projection, "<f4"),
                }
            )
        evidence = getattr(cell, "evidence")
        state = getattr(cell, "state")
        return (
            {
                "class_label": int(record["class_label"]),
                "exception": None,
                "native_gate": _json_ready(getattr(evidence, "native_gate")),
                "output_count": len(outputs),
                "outputs": tuple(outputs),
                "p10_estimated_peak_bytes": p10_estimate,
                "perturbation_id": getattr(cell, "perturbation_id"),
                "reason_code": None,
                "record_id": str(record["record_id"]),
                "state": "complete",
                "state_digest": None if state is None else getattr(state, "state_digest"),
            },
            tuple(projections),
        )
    state_name = "not_applicable" if getattr(cell, "status") is CellStatus.NOT_APPLICABLE else "failed_runtime"
    state = getattr(cell, "state")
    return (
        {
            "class_label": int(record["class_label"]),
            "exception": _cell_exception(cell),
            "native_gate": {},
            "output_count": 0,
            "outputs": (),
            "p10_estimated_peak_bytes": p10_estimate,
            "perturbation_id": getattr(cell, "perturbation_id"),
            "reason_code": getattr(cell, "reason_code") if state_name == "not_applicable" else None,
            "record_id": str(record["record_id"]),
            "state": state_name,
            "state_digest": None if state is None else getattr(state, "state_digest"),
        },
        (),
    )


_PROCESS_SWEEP = None
_PROCESS_PHASE1_CONFIG = None
_PROCESS_SUPPORT_COORDINATES: tuple[float, ...] | None = None
_PROCESS_SUPPORT_MAX_GAP_CM1: float | None = None
_PROCESS_P10_MEMORY_BUDGET_BYTES: int | None = None


def _initialize_process(
    sweep_path: str,
    phase1_config_path: str,
    config_path: str,
) -> None:
    global _PROCESS_SWEEP
    global _PROCESS_PHASE1_CONFIG
    global _PROCESS_SUPPORT_COORDINATES
    global _PROCESS_SUPPORT_MAX_GAP_CM1
    global _PROCESS_P10_MEMORY_BUDGET_BYTES
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    _PROCESS_SWEEP = load_perturbation_sweep_config(Path(sweep_path))
    _PROCESS_PHASE1_CONFIG = load_phase1_core_config(Path(phase1_config_path))
    config = _load_json(Path(config_path))
    support_grid = _require_object("support_grid", config.get("support_grid"))
    _PROCESS_SUPPORT_COORDINATES = _support_coordinates_from_config(
        config,
        synthetic_fixture=False,
    )
    _PROCESS_SUPPORT_MAX_GAP_CM1 = _require_number(
        "support_grid.max_in_range_native_gap_cm1",
        support_grid.get("max_in_range_native_gap_cm1"),
    )
    _PROCESS_P10_MEMORY_BUDGET_BYTES = _require_int(
        "p10.memory_budget_bytes",
        _require_object("p10", config.get("p10")).get("memory_budget_bytes"),
        minimum=1,
    )


def _execute_initialized_record(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
) -> _ExecutedSource:
    if (
        _PROCESS_SWEEP is None
        or _PROCESS_PHASE1_CONFIG is None
        or _PROCESS_SUPPORT_COORDINATES is None
        or _PROCESS_SUPPORT_MAX_GAP_CM1 is None
        or _PROCESS_P10_MEMORY_BUDGET_BYTES is None
    ):
        raise Phase4D1ProtocolBEligibilityVerifierError("worker", "process not initialized")
    source = _phase1_source(record, spectrum)
    admission = P10MemoryAdmission(_PROCESS_P10_MEMORY_BUDGET_BYTES)
    base_projection = _support_projection(
        spectrum,
        _PROCESS_SUPPORT_COORDINATES,
        _PROCESS_SUPPORT_MAX_GAP_CM1,
    )
    receipts: list[dict[str, object]] = []
    projections_by_perturbation: dict[str, tuple[np.ndarray, ...]] = {}
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        p10_estimate = estimate_p10_peak_bytes(int(spectrum.axis_cm1.size)) if perturbation_id == "p10" else None
        try:
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                cell = run_perturbation_cell(
                    source,
                    perturbation_id,
                    _PROCESS_PHASE1_CONFIG,
                    _PROCESS_SWEEP,
                    p10_admission=admission if perturbation_id == "p10" else None,
                )
            receipt, projections = _cell_receipt(
                record,
                cell,
                _PROCESS_SUPPORT_COORDINATES,
                _PROCESS_SUPPORT_MAX_GAP_CM1,
                p10_estimate=p10_estimate,
            )
            if projections:
                if len(projections) != len(ALPHA_GRID):
                    raise Phase4D1ProtocolBEligibilityVerifierError(
                        "operator output",
                        "must contain the complete frozen alpha grid",
                    )
                if not np.array_equal(projections[0], base_projection):
                    raise Phase4D1ProtocolBEligibilityVerifierError(
                        "operator alpha zero",
                        "must equal the source projection exactly",
                    )
            receipts.append(receipt)
            projections_by_perturbation[perturbation_id] = projections
        except Exception as error:
            receipts.append(
                {
                    "class_label": int(record["class_label"]),
                    "exception": {
                        "message": str(error),
                        "path": str(getattr(error, "path", "unexpected_exception")),
                        "type": type(error).__name__,
                    },
                    "native_gate": {},
                    "output_count": 0,
                    "outputs": (),
                    "p10_estimated_peak_bytes": p10_estimate,
                    "perturbation_id": perturbation_id,
                    "reason_code": None,
                    "record_id": str(record["record_id"]),
                    "state": "failed_runtime",
                    "state_digest": None,
                }
            )
            projections_by_perturbation[perturbation_id] = ()
    condition_projections: list[np.ndarray | None] = [base_projection]
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        projections = projections_by_perturbation[perturbation_id]
        if projections:
            condition_projections.extend(projections[1:])
        else:
            condition_projections.extend([None] * len(POSITIVE_ALPHAS))
    return _ExecutedSource(
        operator_cells=tuple(receipts),
        condition_projections=tuple(condition_projections),
    )


def _bounded_process_count(
    requested_workers: int,
    point_counts: Sequence[int],
    memory_budget_bytes: int,
) -> int:
    if isinstance(requested_workers, bool) or not isinstance(requested_workers, int) or requested_workers < 1:
        raise Phase4D1ProtocolBEligibilityVerifierError("worker_count", "must be a positive integer")
    if not point_counts:
        return 1
    estimates = [estimate_p10_peak_bytes(int(point_count)) for point_count in point_counts]
    peak_estimate = max(estimates)
    if peak_estimate > memory_budget_bytes:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "p10 admission",
            "at least one record exceeds the frozen 64 GiB budget",
        )
    by_memory = max(1, memory_budget_bytes // peak_estimate)
    return max(1, min(int(requested_workers), int(by_memory), len(point_counts)))


def _iter_executed_batches(
    inputs: object,
    *,
    config_path: Path,
    worker_count: int,
) -> Sequence[tuple[int, tuple[_ExecutedSource, ...]]]:
    config = _load_json(config_path)
    support_grid = _require_object("support_grid", config.get("support_grid"))
    support_coordinates = _support_coordinates_from_config(config, synthetic_fixture=False)
    support_max_gap = _require_number(
        "support_grid.max_in_range_native_gap_cm1",
        support_grid.get("max_in_range_native_gap_cm1"),
    )
    point_counts = tuple(int(spectrum.axis_cm1.size) for spectrum in getattr(inputs, "source_spectra"))
    memory_budget = _require_int(
        "p10.memory_budget_bytes",
        _require_object("p10", config.get("p10")).get("memory_budget_bytes"),
        minimum=1,
    )
    process_count = _bounded_process_count(worker_count, point_counts, memory_budget)
    if process_count == 1:
        global _PROCESS_SWEEP
        global _PROCESS_PHASE1_CONFIG
        global _PROCESS_SUPPORT_COORDINATES
        global _PROCESS_SUPPORT_MAX_GAP_CM1
        global _PROCESS_P10_MEMORY_BUDGET_BYTES
        _PROCESS_SWEEP = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
        _PROCESS_PHASE1_CONFIG = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
        _PROCESS_SUPPORT_COORDINATES = support_coordinates
        _PROCESS_SUPPORT_MAX_GAP_CM1 = support_max_gap
        _PROCESS_P10_MEMORY_BUDGET_BYTES = memory_budget
        try:
            shard_source_count = _require_int(
                "condition_matrix_store.shard_source_count",
                _require_object("condition_matrix_store", config.get("condition_matrix_store")).get("shard_source_count"),
                minimum=1,
            )
            for start in range(0, getattr(inputs, "source_record_count"), shard_source_count):
                end = min(start + shard_source_count, getattr(inputs, "source_record_count"))
                yield (
                    start,
                    tuple(
                        _execute_initialized_record(record, spectrum)
                        for record, spectrum in zip(
                            getattr(inputs, "source_records")[start:end],
                            getattr(inputs, "source_spectra")[start:end],
                            strict=True,
                        )
                    ),
                )
        finally:
            _PROCESS_SWEEP = None
            _PROCESS_PHASE1_CONFIG = None
            _PROCESS_SUPPORT_COORDINATES = None
            _PROCESS_SUPPORT_MAX_GAP_CM1 = None
            _PROCESS_P10_MEMORY_BUDGET_BYTES = None
        return
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_process,
        initargs=(
            str(ROOT / SWEEP_RELATIVE_PATH),
            str(ROOT / PHASE1_CONFIG_RELATIVE_PATH),
            str(config_path),
        ),
    ) as executor:
        shard_source_count = _require_int(
            "condition_matrix_store.shard_source_count",
            _require_object("condition_matrix_store", config.get("condition_matrix_store")).get("shard_source_count"),
            minimum=1,
        )
        for start in range(0, getattr(inputs, "source_record_count"), shard_source_count):
            end = min(start + shard_source_count, getattr(inputs, "source_record_count"))
            records = tuple(dict(row) for row in getattr(inputs, "source_records")[start:end])
            spectra = getattr(inputs, "source_spectra")[start:end]
            yield (
                start,
                tuple(
                    executor.map(
                        _execute_initialized_record,
                        records,
                        spectra,
                        chunksize=1,
                    )
                ),
            )


def _condition_receipt(
    *,
    source_record: Mapping[str, object],
    spectrum: Spectrum1D,
    executed: _ExecutedSource,
    condition_order: int,
    perturbation_id: str | None,
    alpha: float,
    filename: str,
    shard_id: int,
    shard_local_source_order: int,
    rows_per_shard: int,
    support_point_count: int,
) -> tuple[dict[str, object], bytes]:
    projection = executed.condition_projections[condition_order]
    state = "complete"
    if perturbation_id is None:
        output_axis_sha256: str | None = str(source_record["native_axis_sha256"])
        output_intensity_sha256: str | None = str(source_record["native_intensity_sha256"])
        output_spectrum_id: str | None = spectrum.spectrum_id
    else:
        perturbation_order = ACTIVE_PERTURBATION_IDS.index(perturbation_id)
        cell = executed.operator_cells[perturbation_order]
        state = str(cell["state"])
        if state == "complete":
            alpha_order = ALPHA_GRID.index(alpha)
            outputs = tuple(_require_object("operator_cell", row) for row in cell["outputs"])
            if len(outputs) != len(ALPHA_GRID):
                raise Phase4D1ProtocolBEligibilityVerifierError(
                    "operator outputs",
                    "must contain the complete frozen alpha grid",
                )
            detail = outputs[alpha_order]
            output_axis_sha256 = str(detail["output_axis_sha256"])
            output_intensity_sha256 = str(detail["output_intensity_sha256"])
            output_spectrum_id = str(detail["output_spectrum_id"])
        else:
            output_axis_sha256 = None
            output_intensity_sha256 = None
            output_spectrum_id = None
    if state == "complete":
        if projection is None:
            raise Phase4D1ProtocolBEligibilityVerifierError(
                "condition projection",
                "complete condition is missing its numerical row",
            )
        projected = np.ascontiguousarray(projection, dtype="<f4")
        if (
            projected.size != support_point_count
            or not np.isfinite(projected).all()
            or float(np.linalg.norm(projected.astype(np.float64))) <= 0.0
        ):
            raise Phase4D1ProtocolBEligibilityVerifierError(
                "condition projection",
                "complete condition has an invalid numerical row",
            )
        row_bytes = projected.tobytes(order="C")
        projection_sha256: str | None = _sha_bytes(row_bytes)
        if perturbation_id is None:
            if projection_sha256 != str(source_record["support_projection_sha256"]):
                raise Phase4D1ProtocolBEligibilityVerifierError(
                    "condition projection",
                    "alpha-zero projection differs from source receipt",
                )
        else:
            if projection_sha256 != str(detail["support_projection_sha256"]):
                raise Phase4D1ProtocolBEligibilityVerifierError(
                    "condition projection",
                    "projected row differs from operator receipt",
                )
    else:
        row_bytes = bytes(support_point_count * 4)
        projection_sha256 = None
    offset = (
        (condition_order * rows_per_shard + shard_local_source_order)
        * support_point_count
        * 4
    )
    receipt = {
        "alpha": float(alpha),
        "alpha_float64_le_hex": _alpha_hex(alpha),
        "class_label": int(source_record["class_label"]),
        "condition_id": _condition_id(perturbation_id, alpha),
        "condition_order": condition_order,
        "filename": filename,
        "native_axis_sha256": str(source_record["native_axis_sha256"]),
        "native_intensity_sha256": str(source_record["native_intensity_sha256"]),
        "output_axis_sha256": output_axis_sha256,
        "output_intensity_sha256": output_intensity_sha256,
        "output_spectrum_id": output_spectrum_id,
        "padding_not_scientific_output": state != "complete",
        "perturbation_id": perturbation_id,
        "record_id": str(source_record["record_id"]),
        "record_order": int(source_record["record_order"]),
        "shard_byte_offset": offset,
        "shard_id": shard_id,
        "shard_local_source_order": shard_local_source_order,
        "source_row": int(source_record["source_row"]),
        "source_split": str(source_record["source_split"]),
        "state": state,
        "support_point_count": support_point_count,
        "support_projection_sha256": projection_sha256,
    }
    return receipt, row_bytes


def _compare_reexecuted_real_payloads(
    path: Path,
    *,
    config_path: Path,
    config: dict[str, object],
    inputs: object,
    worker_count: int,
) -> None:
    if getattr(inputs, "role_overlap_detected", False):
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "role_ledger_sha256",
            "unexpected role overlap detected",
        )
    _compare_jsonl_rows(path / "source_records.jsonl", tuple(getattr(inputs, "source_records")))
    _compare_jsonl_rows(path / "model_cells.jsonl", tuple(getattr(inputs, "model_cells")))
    _compare_jsonl_rows(
        path / "model_role_occurrences.jsonl",
        tuple(getattr(inputs, "model_role_occurrences")),
    )
    config_raw = config_path.read_bytes()
    run_id, run_identity = _run_identity(config_raw, config, inputs)
    storage = _require_object("condition_matrix_store", config.get("condition_matrix_store"))
    shard_files_raw = storage.get("shard_files")
    if not isinstance(shard_files_raw, list) or not shard_files_raw:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "condition_matrix_store.shard_files",
            "must be a nonempty string array",
        )
    shard_files = [
        _require_string("condition_matrix_store.shard_files[]", item)
        for item in shard_files_raw
    ]
    support_point_count = _require_int(
        "support_grid.point_count",
        _require_object("support_grid", config.get("support_grid")).get("point_count"),
        minimum=1,
    )
    operator_path = path / "operator_cells.jsonl"
    condition_path = path / "record_conditions.jsonl"
    shard_rows_path = path / "condition_matrix_shards.jsonl"
    gate_cells: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []
    total_bytes = 0
    operator_cell_count = 0
    record_condition_count = 0
    with (
        operator_path.open("rb") as operator_stream,
        condition_path.open("rb") as condition_stream,
        shard_rows_path.open("rb") as shard_rows_stream,
    ):
        for shard_id, (start, executed_sources) in enumerate(
            _iter_executed_batches(inputs, config_path=config_path, worker_count=worker_count)
        ):
            filename = shard_files[shard_id]
            end = start + len(executed_sources) - 1
            rows_per_shard = len(executed_sources)
            expected_rows: list[list[bytes]] = []
            for local_order, executed in enumerate(executed_sources):
                source_order = start + local_order
                source_record = getattr(inputs, "source_records")[source_order]
                spectrum = getattr(inputs, "source_spectra")[source_order]
                if len(executed.operator_cells) != len(ACTIVE_PERTURBATION_IDS):
                    raise Phase4D1ProtocolBEligibilityVerifierError(
                        "operator_cells",
                        "one source must yield exactly five operator receipts",
                    )
                if len(executed.condition_projections) != CONDITION_COUNT:
                    raise Phase4D1ProtocolBEligibilityVerifierError(
                        "condition projections",
                        "one source must yield exactly 41 condition slots",
                    )
                for cell in executed.operator_cells:
                    operator_cell_count += 1
                    _compare_stream_line(
                        operator_stream,
                        expected=_canonical_json_bytes(cell),
                        label="operator_cells.jsonl",
                        index=operator_cell_count,
                    )
                    gate_cells.append(
                        {
                            "record_id": str(cell["record_id"]),
                            "perturbation_id": str(cell["perturbation_id"]),
                            "state": str(cell["state"]),
                        }
                    )
                source_rows: list[bytes] = []
                for condition_order, perturbation_id, alpha in _condition_specs(
                    ACTIVE_PERTURBATION_IDS,
                    ALPHA_GRID,
                ):
                    receipt, row_bytes = _condition_receipt(
                        source_record=source_record,
                        spectrum=spectrum,
                        executed=executed,
                        condition_order=condition_order,
                        perturbation_id=perturbation_id,
                        alpha=alpha,
                        filename=filename,
                        shard_id=shard_id,
                        shard_local_source_order=local_order,
                        rows_per_shard=rows_per_shard,
                        support_point_count=support_point_count,
                    )
                    record_condition_count += 1
                    _compare_stream_line(
                        condition_stream,
                        expected=_canonical_json_bytes(receipt),
                        label="record_conditions.jsonl",
                        index=record_condition_count,
                    )
                    source_rows.append(row_bytes)
                expected_rows.append(source_rows)
            digest = hashlib.sha256()
            byte_count = 0
            with (path / filename).open("rb") as shard_stream:
                for condition_order in range(CONDITION_COUNT):
                    for local_order in range(rows_per_shard):
                        expected_row = expected_rows[local_order][condition_order]
                        actual_row = shard_stream.read(len(expected_row))
                        if actual_row != expected_row:
                            raise Phase4D1ProtocolBEligibilityVerifierError(
                                filename,
                                "rebuild payload mismatch",
                            )
                        digest.update(expected_row)
                        byte_count += len(expected_row)
                if shard_stream.read(1):
                    raise Phase4D1ProtocolBEligibilityVerifierError(
                        filename,
                        "unexpected trailing payload bytes",
                    )
            shard_row = {
                "byte_count": byte_count,
                "condition_count": CONDITION_COUNT,
                "dtype": "little_endian_float32",
                "end_source_order": end,
                "filename": filename,
                "layout": "condition_major_source_major_support",
                "rows_per_shard": rows_per_shard,
                "sha256": digest.hexdigest(),
                "shard_id": shard_id,
                "start_source_order": start,
                "support_point_count": support_point_count,
            }
            _compare_stream_line(
                shard_rows_stream,
                expected=_canonical_json_bytes(shard_row),
                label="condition_matrix_shards.jsonl",
                index=shard_id + 1,
            )
            shard_rows.append(shard_row)
            total_bytes += byte_count
        _ensure_stream_eof(operator_stream, "operator_cells.jsonl")
        _ensure_stream_eof(condition_stream, "record_conditions.jsonl")
        _ensure_stream_eof(shard_rows_stream, "condition_matrix_shards.jsonl")
    class_summaries, gate = _evaluate_gate(
        config,
        tuple(getattr(inputs, "source_records")),
        tuple(getattr(inputs, "model_role_occurrences")),
        gate_cells,
    )
    _compare_jsonl_rows(path / "class_summaries.jsonl", class_summaries)
    _compare_bytes(path / "gate.json", _canonical_json_bytes(gate))
    marker_name = _require_string("gate.marker_filename", gate.get("marker_filename"))
    artifact_payload_files_raw = config.get("artifact_payload_files")
    if not isinstance(artifact_payload_files_raw, list) or not artifact_payload_files_raw:
        raise Phase4D1ProtocolBEligibilityVerifierError(
            "config.artifact_payload_files",
            "must be a nonempty string array",
        )
    manifest = {
        "artifact_order": [
            _require_string("config.artifact_payload_files[]", item)
            for item in artifact_payload_files_raw
        ],
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": _require_object(
            "config.authorities",
            config.get("authorities", {}),
        ),
        "claim_boundary": _require_string("config.claim_boundary", config.get("claim_boundary")),
        "code_authority": _require_object(
            "config.code_authority",
            config.get("code_authority", {}),
        ),
        "condition_matrix_bytes": total_bytes,
        "condition_matrix_shard_count": len(shard_rows),
        "counts": {
            "class_summaries": len(class_summaries),
            "model_cells": len(getattr(inputs, "model_cells")),
            "model_role_occurrences": len(getattr(inputs, "model_role_occurrences")),
            "operator_cells": operator_cell_count,
            "record_conditions": record_condition_count,
            "source_records": len(getattr(inputs, "source_records")),
        },
        "endpoint_count": _require_int(
            "config.denominators.endpoint_count",
            _require_object("config.denominators", config.get("denominators")).get("endpoint_count"),
            minimum=1,
        ),
        "environment_authority": _require_object(
            "config.environment_authority",
            config.get("environment_authority", {}),
        ),
        "protocol": PROTOCOL,
        "run_id": run_id,
        "run_identity": run_identity,
        "status": _require_string("gate.overall_status", gate.get("overall_status")),
        "synthetic_fixture": bool(config.get("synthetic_fixture", False)),
        "trust_anchor": _require_object(
            "config.trust_anchor",
            config.get("trust_anchor", {}),
        ),
    }
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": _require_string("gate.overall_status", gate.get("overall_status")),
    }
    _compare_bytes(path / "manifest.json", _canonical_json_bytes(manifest))
    _compare_bytes(path / marker_name, _canonical_json_bytes(marker))
    _, dynamic_order = _artifact_orders(config, manifest, shard_files)
    checksum_lines = [
        f"{_sha_file(path / name)}  {name}"
        for name in dynamic_order + [marker_name]
    ]
    _compare_bytes(
        path / CHECKSUM_FILENAME,
        ("\n".join(checksum_lines) + "\n").encode("utf-8"),
    )


def _validate_shard_rows_and_matrix(
    path: Path,
    record_conditions: list[dict[str, object]],
) -> tuple[int, int]:
    shard_rows = _load_jsonl_rows(path / "condition_matrix_shards.jsonl")
    if not shard_rows:
        raise Phase4D1ProtocolBEligibilityVerifierError("condition_matrix_shards.jsonl", "must contain at least one row")
    shard_meta = {
        _require_string(f"condition_matrix_shards.jsonl:{index}.filename", row.get("filename")): row
        for index, row in enumerate(shard_rows, start=1)
    }
    total_bytes = 0
    for filename, row in shard_meta.items():
        byte_count = _require_int(f"condition_matrix_shards.{filename}.byte_count", row.get("byte_count"), minimum=0)
        rows_per_shard = _require_int(f"condition_matrix_shards.{filename}.rows_per_shard", row.get("rows_per_shard"), minimum=1)
        condition_count = _require_int(f"condition_matrix_shards.{filename}.condition_count", row.get("condition_count"), minimum=1)
        support_point_count = _require_int(f"condition_matrix_shards.{filename}.support_point_count", row.get("support_point_count"), minimum=1)
        if byte_count != condition_count * rows_per_shard * support_point_count * 4:
            raise Phase4D1ProtocolBEligibilityVerifierError(filename, "declared shard byte_count disagrees with matrix shape")
        shard_path = path / filename
        if not shard_path.is_file():
            raise Phase4D1ProtocolBEligibilityVerifierError(filename, "shard file is missing")
        if shard_path.stat().st_size != byte_count:
            raise Phase4D1ProtocolBEligibilityVerifierError(filename, "shard size mismatch")
        if "sha256" in row and _sha_file(shard_path) != str(row["sha256"]):
            raise Phase4D1ProtocolBEligibilityVerifierError(filename, "shard receipt mismatch")
        total_bytes += byte_count
    for index, row in enumerate(record_conditions, start=1):
        filename = _require_string(f"record_conditions.jsonl:{index}.filename", row.get("filename"))
        offset = _require_int(f"record_conditions.jsonl:{index}.shard_byte_offset", row.get("shard_byte_offset"), minimum=0)
        point_count = _require_int(f"record_conditions.jsonl:{index}.support_point_count", row.get("support_point_count"), minimum=1)
        state = str(row.get("state"))
        read_size = point_count * 4
        byte_count = _require_int(f"condition_matrix_shards.{filename}.byte_count", shard_meta[filename].get("byte_count"), minimum=0)
        if offset + read_size > byte_count:
            raise Phase4D1ProtocolBEligibilityVerifierError(f"record_conditions.jsonl:{index}", "matrix byte offset falls outside shard bounds")
        with (path / filename).open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(read_size)
        if len(payload) != read_size:
            raise Phase4D1ProtocolBEligibilityVerifierError(f"record_conditions.jsonl:{index}", "truncated shard payload")
        expected_hash = row.get("support_projection_sha256")
        if expected_hash is not None:
            if state == "complete":
                if payload == b"\x00" * read_size or _sha_bytes(payload) != str(expected_hash):
                    raise Phase4D1ProtocolBEligibilityVerifierError(f"record_conditions.jsonl:{index}", "condition matrix semantic mismatch with recorded payload hash")
            elif payload != b"\x00" * read_size:
                raise Phase4D1ProtocolBEligibilityVerifierError(f"record_conditions.jsonl:{index}", "noncomplete row must remain zero-padded in the shard matrix")
    return len(shard_rows), total_bytes


def _synthetic_verify_against_inputs(path: Path, config_path: Path, inputs: object) -> None:
    config = _load_json(config_path)
    config_raw = config_path.read_bytes()
    expected_payloads, expected_shards, checksum_order = _synthetic_expected_artifact(config, config_raw, inputs)
    for name, payload in expected_payloads.items():
        if name == CHECKSUM_FILENAME:
            continue
        _compare_bytes(path / name, payload)
    _compare_bytes(path / CHECKSUM_FILENAME, expected_payloads[CHECKSUM_FILENAME])
    for name, payload in expected_shards.items():
        _compare_bytes(path / name, payload)
    _validate_checksum_tree(path, checksum_order)
    _validate_shard_rows_and_matrix(path, _load_jsonl_rows(path / "record_conditions.jsonl"))


def _validate_common_run_structure(
    path: Path,
    *,
    config_path: Path,
    expected_counts: dict[str, int] | None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, int, int]:
    run_config = path / "config.json"
    if config_path != run_config and config_path.read_bytes() != run_config.read_bytes():
        raise Phase4D1ProtocolBEligibilityVerifierError("config.json", "external config_path bytes differ from authoritative payload")
    config_raw = run_config.read_bytes()
    config = _load_json(run_config)
    synthetic_fixture = bool(config.get("synthetic_fixture", False))
    _validate_authority_receipts(config, synthetic_fixture=synthetic_fixture)
    _validate_frozen_config_anchor(run_config, config_raw, synthetic_fixture=synthetic_fixture)
    manifest = _load_json(path / "manifest.json")
    gate = _load_json(path / "gate.json")
    marker_names = [name for name in MARKER_FILENAMES if (path / name).is_file()]
    if len(marker_names) != 1:
        raise Phase4D1ProtocolBEligibilityVerifierError("terminal marker", "must contain exactly one of complete.json or failed.json")
    marker_name = marker_names[0]
    marker = _load_json(path / marker_name)
    shard_rows = _load_jsonl_rows(path / "condition_matrix_shards.jsonl")
    shard_names = [
        _require_string(f"condition_matrix_shards.jsonl:{index}.filename", row.get("filename"))
        for index, row in enumerate(shard_rows, start=1)
    ]
    _compact_order, dynamic_order = _artifact_orders(config, manifest, shard_names)
    expected_files = set(dynamic_order) | {CHECKSUM_FILENAME, marker_name}
    if _artifact_file_set(path) != expected_files:
        raise Phase4D1ProtocolBEligibilityVerifierError("artifact inventory", "payload set mismatch")
    if str(gate.get("marker_filename")) != marker_name:
        raise Phase4D1ProtocolBEligibilityVerifierError("gate.json", "marker_filename mismatch")
    if str(marker.get("status")) != str(gate.get("overall_status")):
        raise Phase4D1ProtocolBEligibilityVerifierError(marker_name, "status mismatch")
    if str(marker.get("run_id")) != str(manifest.get("run_id")):
        raise Phase4D1ProtocolBEligibilityVerifierError(marker_name, "run_id mismatch")
    _validate_checksum_tree(path, dynamic_order + [marker_name])
    if synthetic_fixture:
        record_conditions = _load_jsonl_rows(path / "record_conditions.jsonl")
        record_condition_count = len(record_conditions)
    else:
        record_conditions = []
        record_condition_count = _count_jsonl(path / "record_conditions.jsonl")
    condition_matrix_shard_count, condition_matrix_bytes = _validate_shard_rows_and_matrix(path, record_conditions)
    source_record_count = _count_jsonl(path / "source_records.jsonl")
    model_cell_count = _count_jsonl(path / "model_cells.jsonl")
    role_occurrence_count = _count_jsonl(path / "model_role_occurrences.jsonl")
    operator_cell_count = _count_jsonl(path / "operator_cells.jsonl")
    class_summary_count = _count_jsonl(path / "class_summaries.jsonl")
    if expected_counts is not None:
        if expected_counts.get("source_record_count", source_record_count) != source_record_count:
            raise Phase4D1ProtocolBEligibilityVerifierError("inputs.source_record_count", "mismatch with authoritative source_records.jsonl")
        if expected_counts.get("model_cell_count", model_cell_count) != model_cell_count:
            raise Phase4D1ProtocolBEligibilityVerifierError("inputs.model_cell_count", "mismatch with authoritative model_cells.jsonl")
        if expected_counts.get("role_occurrence_count", role_occurrence_count) != role_occurrence_count:
            raise Phase4D1ProtocolBEligibilityVerifierError("inputs.role_occurrence_count", "mismatch with authoritative model_role_occurrences.jsonl")
    counts = _require_object("manifest.counts", manifest.get("counts"))
    expected_manifest_counts = {
        "class_summaries": class_summary_count,
        "model_cells": model_cell_count,
        "model_role_occurrences": role_occurrence_count,
        "operator_cells": operator_cell_count,
        "record_conditions": record_condition_count,
        "source_records": source_record_count,
    }
    if counts != expected_manifest_counts:
        raise Phase4D1ProtocolBEligibilityVerifierError("manifest.counts", "count mismatch")
    if _require_int("manifest.condition_matrix_shard_count", manifest.get("condition_matrix_shard_count"), minimum=1) != condition_matrix_shard_count:
        raise Phase4D1ProtocolBEligibilityVerifierError("manifest.condition_matrix_shard_count", "mismatch")
    if _require_int("manifest.condition_matrix_bytes", manifest.get("condition_matrix_bytes"), minimum=0) != condition_matrix_bytes:
        raise Phase4D1ProtocolBEligibilityVerifierError("manifest.condition_matrix_bytes", "mismatch")
    if _require_string("manifest.protocol", manifest.get("protocol")) != PROTOCOL:
        raise Phase4D1ProtocolBEligibilityVerifierError("manifest.protocol", "mismatch")
    if _require_string("manifest.status", manifest.get("status")) != _require_string("gate.overall_status", gate.get("overall_status")):
        raise Phase4D1ProtocolBEligibilityVerifierError("manifest.status", "mismatch")
    run_id = _require_string("manifest.run_id", manifest.get("run_id"))
    return config, manifest, gate, run_id, condition_matrix_shard_count, condition_matrix_bytes


def _summary_from_artifact(
    path: Path,
    *,
    config_path: Path,
    expected_counts: dict[str, int] | None,
) -> Phase4D1ProtocolBEligibilitySummary:
    config, manifest, gate, run_id, condition_matrix_shard_count, condition_matrix_bytes = _validate_common_run_structure(
        path,
        config_path=config_path,
        expected_counts=expected_counts,
    )
    endpoint_count = _require_int(
        "manifest.endpoint_count",
        manifest.get("endpoint_count"),
        minimum=1,
    )
    model_seed_count = len(config.get("model_seeds", ()))
    return Phase4D1ProtocolBEligibilitySummary(
        path=path,
        run_id=run_id,
        status=_require_string("gate.overall_status", gate.get("overall_status")),
        endpoint_count=endpoint_count,
        model_seed_count=model_seed_count,
        source_record_count=_count_jsonl(path / "source_records.jsonl"),
        model_cell_count=_count_jsonl(path / "model_cells.jsonl"),
        role_occurrence_count=_count_jsonl(path / "model_role_occurrences.jsonl"),
        operator_cell_count=_count_jsonl(path / "operator_cells.jsonl"),
        record_condition_count=_count_jsonl(path / "record_conditions.jsonl"),
        class_summary_count=_count_jsonl(path / "class_summaries.jsonl"),
        condition_matrix_shard_count=condition_matrix_shard_count,
        condition_matrix_bytes=condition_matrix_bytes,
    )


def verify_phase4_d1_protocol_b_eligibility_from_inputs(
    path: Path,
    *,
    inputs: object,
    config_path: Path,
    worker_count: int,
) -> Phase4D1ProtocolBEligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D1ProtocolBEligibilityVerifierError("worker_count", "must be a positive integer")
    run_path = Path(path)
    if not run_path.is_dir():
        raise Phase4D1ProtocolBEligibilityVerifierError("path", "must be an existing run directory")
    config_file = Path(config_path)
    config = _load_json(config_file)
    if bool(config.get("synthetic_fixture", False)):
        _synthetic_verify_against_inputs(run_path, config_file, inputs)
    else:
        _compare_reexecuted_real_payloads(
            run_path,
            config_path=config_file,
            config=config,
            inputs=inputs,
            worker_count=worker_count,
        )
    return _summary_from_artifact(
        run_path,
        config_path=config_file,
        expected_counts=_expected_counts_from_inputs(inputs),
    )


def verify_phase4_d1_protocol_b_eligibility(
    path: Path,
    *,
    worker_count: int = 12,
) -> Phase4D1ProtocolBEligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D1ProtocolBEligibilityVerifierError("worker_count", "must be a positive integer")
    run_path = Path(path)
    if not run_path.is_dir():
        raise Phase4D1ProtocolBEligibilityVerifierError("path", "must be an existing run directory")
    config_path = run_path / "config.json"
    if not config_path.is_file():
        raise Phase4D1ProtocolBEligibilityVerifierError("config.json", "run directory is missing config.json")
    config = _load_json(config_path)
    if not bool(config.get("synthetic_fixture", False)):
        real_inputs = _reconstruct_real_inputs(config)
        _compare_reexecuted_real_payloads(
            run_path,
            config_path=config_path,
            config=config,
            inputs=real_inputs,
            worker_count=worker_count,
        )
        expected_counts = _expected_counts_from_inputs(real_inputs)
    else:
        expected_counts = None
    summary = _summary_from_artifact(
        run_path,
        config_path=config_path,
        expected_counts=expected_counts,
    )
    frozen_config_path = ROOT / CONFIG_RELATIVE_PATH
    if frozen_config_path.is_file() and config_path.read_bytes() != frozen_config_path.read_bytes():
        raise Phase4D1ProtocolBEligibilityVerifierError("config.json", "does not match frozen experiment config")
    return summary


__all__ = [
    "Phase4D1ProtocolBEligibilitySummary",
    "Phase4D1ProtocolBEligibilityVerifierError",
    "verify_phase4_d1_protocol_b_eligibility",
    "verify_phase4_d1_protocol_b_eligibility_from_inputs",
]

