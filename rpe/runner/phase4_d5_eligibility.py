from __future__ import annotations

import hashlib
import json
import math
import platform
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from rpe.runner.phase4_d5_eligibility_authority import (
    CONFIG_BYTES,
    CONFIG_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d5_eligibility_v1.json"
PROTOCOL_RELATIVE_PATH = "reports/phase4/step04_d5_eligibility_protocol.md"
D5_CONFIG_RELATIVE_PATH = "experiments/phase05/configs/d5_rruff_protocol.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
DATASET_RELATIVE_PATH = "data/unified/rruff_raman_raw"
SCHEMA_VERSION = "phase4-d5-eligibility-config-v1"
EXPERIMENT_ID = "phase4-d5-eligibility-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d5-eligibility-artifact-v1"
RUN_PREFIX = "phase4-d5-eligibility-"
ACTIVE_PERTURBATIONS = (
    "p01",
    "p02",
    "p03",
    "p04",
    "p05",
    "p08",
    "p09",
    "p10",
    "p11",
    "p12",
)
INACTIVE_PERTURBATIONS = ("p06", "p07")
ALL_PERTURBATIONS = tuple(f"p{index:02d}" for index in range(1, 13))
PEAK_PERTURBATIONS = ("p01", "p02", "p03", "p04", "p05")
FULL_DOMAIN_PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
STRUCTURAL_REASON = "structurally_ineligible_missing_explicit_baseline"
NOT_APPLICABLE_REASONS = (
    "false_peak_placement_impossible",
    "insufficient_points_for_peak_model",
    "invalid_peak_component",
    "no_detected_peak",
    "nonpositive_intensity_range",
    "zero_false_peak_insertion_capacity",
)
TERMINAL_STATES = (
    "complete",
    "not_applicable",
    "failed_runtime",
    "structurally_ineligible",
)
CODE_RELATIVE_PATHS = (
    "rpe/downstream/rruff.py",
    "rpe/evaluation/contracts.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/baseline_distortion.py",
    "rpe/perturb/contracts.py",
    "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/perturb/peak_family.py",
    "rpe/perturb/sweep.py",
    "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_gates.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py",
    "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d5_eligibility.py",
    "rpe/runner/phase4_d5_eligibility_verifier.py",
    "tools/run_phase4_d5_eligibility.py",
)
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d5_eligibility_authority.py"
ARTIFACT_STATIC_FILES = (
    "config.json",
    "unique_records.jsonl",
    "split_occurrences.jsonl",
    "cells.jsonl",
    "classes.jsonl",
    "common_support.jsonl",
    "gate.json",
    "manifest.json",
    "SHA256SUMS",
)
PAYLOAD_PREFIX = (
    "config.json",
    "unique_records.jsonl",
    "split_occurrences.jsonl",
    "cells.jsonl",
    "classes.jsonl",
    "common_support.jsonl",
    "gate.json",
    "manifest.json",
)
FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "accuracy",
        "accuracy_delta",
        "alignment_gap",
        "bootstrap",
        "correct",
        "downstream_harm",
        "loss",
        "metric_id",
        "metric_value",
        "mse",
        "p_value",
        "permutation",
        "prediction",
        "score",
        "top1_correct",
        "top5_correct",
        "wasserstein_1_cm1",
    }
)
FORBIDDEN_KEY_FRAGMENTS = (
    "alignment_",
    "metric_value",
    "predicted_",
    "prediction_",
    "top1_",
    "top5_",
)
RECORD_KEYS = {
    "class_label",
    "cohort_index",
    "group_id",
    "mineral_name",
    "native_axis_sha256",
    "native_intensity_sha256",
    "occurrence_count",
    "pin_id",
    "point_count",
    "provenance",
    "query_positions",
    "record_id",
    "record_order",
    "rruff_id",
    "split_seeds",
}
OCCURRENCE_KEYS = {
    "class_label",
    "cohort_index",
    "group_id",
    "record_id",
    "split_query_count",
    "query_order",
    "split_seed",
    "split_sha256",
    "unique_record_order",
}
CELL_KEYS = {
    "class_label",
    "exception",
    "group_id",
    "native_gate",
    "output_count",
    "outputs",
    "p10_estimated_peak_bytes",
    "perturbation_id",
    "reason_code",
    "record_id",
    "state",
    "state_digest",
}
OUTPUT_KEYS = {
    "alpha",
    "alpha_float64_le_hex",
    "axis_changed",
    "diagnostics",
    "intensity_changed",
    "output_axis_sha256",
    "output_intensity_sha256",
    "output_spectrum_id",
    "support_projection_sha256",
}
CLASS_KEYS = {
    "class_label",
    "complete",
    "complete_record_count",
    "mineral_name",
    "perturbation_id",
    "required_record_count",
    "state",
    "state_counts",
}
COMMON_RECORD_KEYS = {
    "class_label",
    "complete",
    "group_id",
    "peak_states",
    "record_id",
    "scope",
}
COMMON_CLASS_KEYS = {
    "class_label",
    "complete",
    "peak_states",
    "required_record_count",
    "scope",
}


class Phase4D5EligibilityError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D5EligibilityConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    active_perturbation_ids: tuple[str, ...]
    inactive_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    unique_record_count: int
    split_occurrence_count: int
    group_count: int
    class_count: int
    expected_cell_count: int
    expected_active_cell_count: int
    expected_inactive_cell_count: int
    expected_class_summary_count: int
    expected_apply_call_count: int
    gates: Mapping[str, float]
    not_applicable_reason_codes: tuple[str, ...]
    p10_memory_budget_bytes: int
    native_gate_relative_tolerance: float
    support_start_cm1: float
    support_stop_cm1: float
    support_step_cm1: float
    support_point_count: int
    support_max_gap_cm1: float
    authorities: Mapping[str, str]
    frozen_identities: Mapping[str, object]
    code_authority: Mapping[str, Mapping[str, object]]
    environment_authority: Mapping[str, object]
    claim_boundary: str


@dataclass(frozen=True)
class D5EligibilityLedgers:
    unique_records: tuple[Mapping[str, object], ...]
    split_occurrences: tuple[Mapping[str, object], ...]
    unique_group_count: int
    unique_class_count: int


@dataclass(frozen=True)
class Phase4D5EligibilitySummary:
    path: Path
    run_id: str
    status: str
    unique_record_count: int
    split_occurrence_count: int
    cell_count: int
    class_summary_count: int


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
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase4D5EligibilityError("json", "contains nonfinite value")
        return value
    raise Phase4D5EligibilityError(
        "json", f"unsupported value type {type(value).__name__}"
    )


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray, *, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    return _sha256_bytes(array.tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D5EligibilityError(path, "must be an object")
    return value


def _strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D5EligibilityError(path, "must be an array")
    converted = tuple(str(item) for item in value)
    if any(not item for item in converted) or len(set(converted)) != len(converted):
        raise Phase4D5EligibilityError(path, "must contain unique nonempty strings")
    return converted


def _floats(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D5EligibilityError(path, "must be an array")
    converted = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in converted):
        raise Phase4D5EligibilityError(path, "must contain finite numbers")
    return converted


def _integer(path: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4D5EligibilityError(path, "must be an integer")
    if value < (1 if positive else 0):
        raise Phase4D5EligibilityError(path, "is outside the allowed range")
    return value


def _number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D5EligibilityError(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise Phase4D5EligibilityError(path, "must be finite")
    return result


def _lower_hex(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase4D5EligibilityError(path, "must be lowercase SHA-256 hex")
    return value


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


def _validate_code_and_environment(
    config: Phase4D5EligibilityConfig,
) -> tuple[dict[str, Mapping[str, object]], dict[str, object]]:
    if config.synthetic_fixture:
        code = {str(key): dict(value) for key, value in config.code_authority.items()}
        environment = (
            dict(config.environment_authority)
            if config.environment_authority
            else _environment_document()
        )
        return code, environment
    if tuple(config.code_authority) != CODE_RELATIVE_PATHS:
        raise Phase4D5EligibilityError(
            "code_authority", "must contain the exact ordered frozen paths"
        )
    code = _code_document()
    expected_code = {key: dict(value) for key, value in config.code_authority.items()}
    if code != expected_code:
        raise Phase4D5EligibilityError(
            "code_authority", "current bytes or SHA-256 differ from config"
        )
    environment = _environment_document()
    if environment != dict(config.environment_authority):
        raise Phase4D5EligibilityError(
            "environment_authority", "current environment differs from config"
        )
    return code, environment


def parse_phase4_d5_eligibility_config(
    path: Path,
    raw: bytes,
    *,
    require_frozen_identity: bool,
) -> Phase4D5EligibilityConfig:
    try:
        document_value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5EligibilityError("config", str(error)) from error
    document = _object("config", document_value)
    if raw != _canonical_json_bytes(document):
        raise Phase4D5EligibilityError("config", "must use canonical JSON")
    if require_frozen_identity and (
        len(raw) != CONFIG_BYTES or _sha256_bytes(raw) != CONFIG_SHA256
    ):
        raise Phase4D5EligibilityError(
            "frozen config identity", "bytes or SHA-256 mismatch"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D5EligibilityError("schema_version", "mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D5EligibilityError("experiment_id", "mismatch")
    active = _strings("active_perturbation_ids", document.get("active_perturbation_ids"))
    inactive = _strings("inactive_perturbation_ids", document.get("inactive_perturbation_ids"))
    alpha_grid = _floats("alpha_grid", document.get("alpha_grid"))
    if active != ACTIVE_PERTURBATIONS or inactive != INACTIVE_PERTURBATIONS:
        raise Phase4D5EligibilityError("perturbation_ids", "frozen set mismatch")
    if alpha_grid != ALPHA_GRID:
        raise Phase4D5EligibilityError("alpha_grid", "frozen grid mismatch")

    denominators = _object("denominators", document.get("denominators"))
    unique_records = _integer("denominators.unique_record_count", denominators.get("unique_record_count"), positive=True)
    occurrences = _integer("denominators.split_occurrence_count", denominators.get("split_occurrence_count"), positive=True)
    groups = _integer("denominators.group_count", denominators.get("group_count"), positive=True)
    classes = _integer("denominators.class_count", denominators.get("class_count"), positive=True)
    if occurrences < unique_records or groups > unique_records or classes > unique_records:
        raise Phase4D5EligibilityError("denominators", "structural ordering mismatch")

    expected = _object("expected", document.get("expected"))
    expected_cells = _integer("expected.cell_count", expected.get("cell_count"), positive=True)
    expected_active = _integer("expected.active_cell_count", expected.get("active_cell_count"), positive=True)
    expected_inactive = _integer("expected.inactive_cell_count", expected.get("inactive_cell_count"), positive=True)
    expected_classes = _integer("expected.class_summary_count", expected.get("class_summary_count"), positive=True)
    expected_calls = _integer("expected.apply_call_count", expected.get("apply_call_count"), positive=True)
    expected_tuple = (expected_cells, expected_active, expected_inactive, expected_classes, expected_calls)
    derived_tuple = (
        unique_records * len(ALL_PERTURBATIONS),
        unique_records * len(ACTIVE_PERTURBATIONS),
        unique_records * len(INACTIVE_PERTURBATIONS),
        classes * len(ALL_PERTURBATIONS),
        unique_records * len(ACTIVE_PERTURBATIONS) * len(ALPHA_GRID),
    )
    if expected_tuple != derived_tuple:
        raise Phase4D5EligibilityError("expected", "counts do not match denominators")

    gate_document = _object("gates", document.get("gates"))
    gates = {str(key): _number(f"gates.{key}", value) for key, value in gate_document.items()}
    frozen_gates = {
        "full_domain_class_fraction": 1.0,
        "full_domain_record_fraction": 1.0,
        "p01_p04_class_fraction": 0.95,
        "p01_p04_record_fraction": 0.95,
        "p05_class_fraction": 0.9,
        "p05_record_fraction": 0.9,
        "peak_common_class_fraction": 0.9,
        "peak_common_record_fraction": 0.9,
    }
    if gates != frozen_gates:
        raise Phase4D5EligibilityError("gates", "frozen thresholds mismatch")
    reasons = _strings("not_applicable_reason_codes", document.get("not_applicable_reason_codes"))
    if reasons != NOT_APPLICABLE_REASONS:
        raise Phase4D5EligibilityError("not_applicable_reason_codes", "frozen taxonomy mismatch")

    p10 = _object("p10", document.get("p10"))
    budget = _integer("p10.memory_budget_bytes", p10.get("memory_budget_bytes"), positive=True)
    if (
        budget != 64 * 2**30
        or _number("p10.correlation_length_cm1", p10.get("correlation_length_cm1")) != 20.0
        or p10.get("peak_estimate_formula") != "32*N^2+64*N+2^30"
    ):
        raise Phase4D5EligibilityError("p10", "frozen resource contract mismatch")
    tolerance = _number("phase1_native_gate_relative_tolerance", document.get("phase1_native_gate_relative_tolerance"))
    if tolerance != 1e-12:
        raise Phase4D5EligibilityError("phase1_native_gate_relative_tolerance", "must equal 1e-12")

    support = _object("support_grid", document.get("support_grid"))
    support_values = (
        _number("support_grid.start_cm1", support.get("start_cm1")),
        _number("support_grid.stop_cm1", support.get("stop_cm1")),
        _number("support_grid.step_cm1", support.get("step_cm1")),
        _integer("support_grid.point_count", support.get("point_count"), positive=True),
        _number("support_grid.max_in_range_native_gap_cm1", support.get("max_in_range_native_gap_cm1")),
    )
    if support_values != (204.0, 1800.0, 2.0, 799, 3.0):
        raise Phase4D5EligibilityError("support_grid", "frozen support mismatch")

    authorities_document = _object("authorities", document.get("authorities"))
    authorities = {str(key): _lower_hex(f"authorities.{key}", value) for key, value in authorities_document.items()}
    identities_document = _object("frozen_identities", document.get("frozen_identities"))
    identities = dict(identities_document)
    for key, value in identities.items():
        if key == "split_sha256":
            for index, digest in enumerate(_strings("frozen_identities.split_sha256", value)):
                _lower_hex(f"frozen_identities.split_sha256[{index}]", digest)
        elif key.endswith("sha256"):
            _lower_hex(f"frozen_identities.{key}", value)

    code_document = _object("code_authority", document.get("code_authority", {}))
    code_authority: dict[str, Mapping[str, object]] = {}
    for relative, value in code_document.items():
        identity = _object(f"code_authority.{relative}", value)
        code_authority[str(relative)] = MappingProxyType(
            {
                "bytes": _integer(f"code_authority.{relative}.bytes", identity.get("bytes"), positive=True),
                "sha256": _lower_hex(f"code_authority.{relative}.sha256", identity.get("sha256")),
            }
        )
    environment = dict(_object("environment_authority", document.get("environment_authority", {})))
    claim_boundary = document.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary:
        raise Phase4D5EligibilityError("claim_boundary", "must be nonempty")
    synthetic = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity:
        frozen_authorities = {
            "d5_config_sha256": "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009",
            "dataset_sha256sums_sha256": "f0cb09af8d80cdf3d52af9ed01bc1ae48abba94fefdff712c30d6e8c8bfe4995",
            "phase1_core_config_sha256": "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd",
            "phase4_step1_sha256": "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f",
            "phase4_step2_sha256": "4b9c784476d5add2f8551b928fc1d47055945c9584a47b2478996529a0c6dfe6",
            "phase4_step3_sha256": "76b1ffdc3544673cebbd6520803ae72aeff610a6a66a2306dd317c9402fdcbbe",
            "protocol_sha256": "2ae40b991d79935f173b16cf59ad8d7f531ad6a0f5717373712da96a802279e5",
            "sweep_sha256": "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186",
        }
        if synthetic or authorities != frozen_authorities:
            raise Phase4D5EligibilityError("authorities", "frozen authorities mismatch")
        if (unique_records, occurrences, groups, classes) != (3012, 6621, 1550, 681):
            raise Phase4D5EligibilityError("denominators", "frozen D5 values mismatch")

    config = Phase4D5EligibilityConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=_sha256_bytes(raw),
        document=_freeze(document),
        synthetic_fixture=synthetic,
        active_perturbation_ids=active,
        inactive_perturbation_ids=inactive,
        alpha_grid=alpha_grid,
        unique_record_count=unique_records,
        split_occurrence_count=occurrences,
        group_count=groups,
        class_count=classes,
        expected_cell_count=expected_cells,
        expected_active_cell_count=expected_active,
        expected_inactive_cell_count=expected_inactive,
        expected_class_summary_count=expected_classes,
        expected_apply_call_count=expected_calls,
        gates=MappingProxyType(gates),
        not_applicable_reason_codes=reasons,
        p10_memory_budget_bytes=budget,
        native_gate_relative_tolerance=tolerance,
        support_start_cm1=support_values[0],
        support_stop_cm1=support_values[1],
        support_step_cm1=support_values[2],
        support_point_count=support_values[3],
        support_max_gap_cm1=support_values[4],
        authorities=MappingProxyType(authorities),
        frozen_identities=_freeze(identities),
        code_authority=MappingProxyType(code_authority),
        environment_authority=_freeze(environment),
        claim_boundary=claim_boundary,
    )
    if require_frozen_identity:
        _validate_code_and_environment(config)
    return config


def load_phase4_d5_eligibility_config(path: Path) -> Phase4D5EligibilityConfig:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise Phase4D5EligibilityError("config", str(error)) from error
    return parse_phase4_d5_eligibility_config(
        Path(path), raw, require_frozen_identity=True
    )


def _support_grid(config: Phase4D5EligibilityConfig) -> np.ndarray:
    grid = np.arange(
        config.support_start_cm1,
        config.support_stop_cm1 + config.support_step_cm1 / 2.0,
        config.support_step_cm1,
        dtype="<f8",
    )
    if grid.size != config.support_point_count:
        raise Phase4D5EligibilityError("support_grid", "point count mismatch")
    return grid


def _support_projection_sha256(
    spectrum: Spectrum1D,
    grid: np.ndarray,
    max_gap_cm1: float,
) -> str:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5EligibilityError(
            "support projection", f"{spectrum.spectrum_id} would require extrapolation"
        )
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    if left < 0 or right >= axis.size:
        raise Phase4D5EligibilityError("support projection", "invalid bounds")
    support = axis[left : right + 1]
    if support.size < 2 or float(np.max(np.diff(support))) > max_gap_cm1:
        raise Phase4D5EligibilityError("support projection", "native gap exceeds maximum")
    values = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(values).all():
        raise Phase4D5EligibilityError("support projection", "contains nonfinite values")
    return _array_sha256(values, dtype="<f4")


def reconstruct_d5_eligibility_ledgers(
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    config: Phase4D5EligibilityConfig,
) -> D5EligibilityLedgers:
    if not isinstance(cohort, D5RawCohort):
        raise Phase4D5EligibilityError("cohort", "must be D5RawCohort")
    if len(native_spectra) != len(cohort.record_ids):
        raise Phase4D5EligibilityError("native_spectra", "must align with cohort")
    expected_spectrum_ids = tuple(
        f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids
    )
    if tuple(spectrum.spectrum_id for spectrum in native_spectra) != expected_spectrum_ids:
        raise Phase4D5EligibilityError("native_spectra", "record order mismatch")
    if cohort.protocol_config_sha256 != config.authorities.get("d5_config_sha256"):
        raise Phase4D5EligibilityError("cohort", "D5 config identity mismatch")

    configured_splits = tuple(config.frozen_identities.get("split_sha256", ()))
    observed_splits = tuple(split.split_sha256 for split in cohort.splits)
    if observed_splits != configured_splits:
        raise Phase4D5EligibilityError("splits", "frozen SHA-256 sequence mismatch")

    occurrence_rows: list[dict[str, object]] = []
    positions: dict[int, list[dict[str, int]]] = defaultdict(list)
    for split in cohort.splits:
        query_indices = tuple(int(index) for index in split.query_indices)
        for query_order, cohort_index in enumerate(query_indices):
            positions[cohort_index].append(
                {"query_order": query_order, "split_seed": int(split.seed)}
            )
            occurrence_rows.append(
                {
                    "class_label": int(cohort.class_labels[cohort_index]),
                    "cohort_index": cohort_index,
                    "group_id": cohort.group_ids[cohort_index],
                    "query_order": query_order,
                    "record_id": cohort.record_ids[cohort_index],
                    "split_query_count": len(query_indices),
                    "split_seed": int(split.seed),
                    "split_sha256": split.split_sha256,
                }
            )

    unique_indices = sorted(positions, key=lambda index: cohort.record_ids[index])
    order_by_index = {cohort_index: order for order, cohort_index in enumerate(unique_indices)}
    for row in occurrence_rows:
        row["unique_record_order"] = order_by_index[int(row["cohort_index"])]
    occurrence_rows.sort(key=lambda row: (int(row["split_seed"]), int(row["query_order"])))

    record_rows: list[dict[str, object]] = []
    for record_order, cohort_index in enumerate(unique_indices):
        spectrum = native_spectra[cohort_index]
        query_positions = sorted(
            positions[cohort_index],
            key=lambda item: (item["split_seed"], item["query_order"]),
        )
        record_rows.append(
            {
                "class_label": int(cohort.class_labels[cohort_index]),
                "cohort_index": cohort_index,
                "group_id": cohort.group_ids[cohort_index],
                "mineral_name": cohort.mineral_names[cohort_index],
                "native_axis_sha256": _array_sha256(spectrum.axis_cm1),
                "native_intensity_sha256": _array_sha256(spectrum.intensity),
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
                "split_seeds": sorted({item["split_seed"] for item in query_positions}),
            }
        )

    record_ids = [str(row["record_id"]) for row in record_rows]
    group_ids = sorted({str(row["group_id"]) for row in record_rows})
    class_labels = sorted({str(row["class_label"]) for row in record_rows})
    identities = config.frozen_identities
    checks = {
        "query_record_ids_sha256": _ids_digest(record_ids),
        "query_group_ids_sha256": _ids_digest(group_ids),
        "query_class_labels_sha256": _ids_digest(class_labels),
    }
    for key, observed in checks.items():
        expected = identities.get(key)
        if expected is not None and observed != expected:
            raise Phase4D5EligibilityError(key, "literal union digest mismatch")
    counts = (len(record_rows), len(occurrence_rows), len(group_ids), len(class_labels))
    expected_counts = (
        config.unique_record_count,
        config.split_occurrence_count,
        config.group_count,
        config.class_count,
    )
    if counts != expected_counts:
        raise Phase4D5EligibilityError("ledger counts", "do not match config")
    return D5EligibilityLedgers(
        unique_records=tuple(MappingProxyType(row) for row in record_rows),
        split_occurrences=tuple(MappingProxyType(row) for row in occurrence_rows),
        unique_group_count=len(group_ids),
        unique_class_count=len(class_labels),
    )


def _phase1_source(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
) -> Phase1Source:
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
        provenance=MappingProxyType(dict(_object("record.provenance", record["provenance"]))),
    )


def _exception_document(
    exception_type: str | None,
    exception_path: str | None,
    exception_message: str | None,
) -> Mapping[str, object] | None:
    if exception_type is None and exception_path is None and exception_message is None:
        return None
    return {
        "message": exception_message or "unspecified",
        "path": exception_path or "unspecified",
        "type": exception_type or "unspecified",
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
    cell: Phase1Cell,
    config: Phase4D5EligibilityConfig,
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
                support_hash = None
                if cell.perturbation_id in {"p11", "p12"}:
                    support_hash = _support_projection_sha256(
                        output, grid, config.support_max_gap_cm1
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
        reason = cell.reason_code
        if cell.perturbation_id not in PEAK_PERTURBATIONS or reason not in config.not_applicable_reason_codes:
            error = Phase4D5EligibilityError(
                "not_applicable", "unexpected perturbation or reason code"
            )
            return _failed_runtime_receipt(
                record,
                cell.perturbation_id,
                error,
                p10_estimate=p10_estimate,
                state_digest=state_digest,
            )
    else:
        state = "failed_runtime"
        reason = None
    return {
        "class_label": int(record["class_label"]),
        "exception": _exception_document(
            cell.evidence.exception_type,
            cell.evidence.exception_path,
            cell.evidence.exception_message,
        ),
        "group_id": str(record["group_id"]),
        "native_gate": {},
        "output_count": 0,
        "outputs": [],
        "p10_estimated_peak_bytes": p10_estimate,
        "perturbation_id": cell.perturbation_id,
        "reason_code": reason,
        "record_id": str(record["record_id"]),
        "state": state,
        "state_digest": state_digest,
    }


def _run_record_cells(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
    phase1_config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    config: Phase4D5EligibilityConfig,
    admission: P10MemoryAdmission,
    grid: np.ndarray,
) -> tuple[dict[str, object], ...]:
    source = _phase1_source(record, spectrum)
    receipts: list[dict[str, object]] = []
    for perturbation_id in ALL_PERTURBATIONS:
        if perturbation_id in INACTIVE_PERTURBATIONS:
            receipts.append(
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


def _required_count(total: int, fraction: float) -> int:
    return int(math.ceil(total * fraction - 1e-15))


def evaluate_d5_eligibility_gates(
    record_rows: Sequence[Mapping[str, object]],
    cell_rows: Sequence[Mapping[str, object]],
    config: Phase4D5EligibilityConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if len(record_rows) != config.unique_record_count:
        raise Phase4D5EligibilityError("record_rows", "denominator mismatch")
    if len(cell_rows) != config.expected_cell_count:
        raise Phase4D5EligibilityError("cell_rows", "count mismatch")
    records_by_id = {str(row["record_id"]): row for row in record_rows}
    if len(records_by_id) != len(record_rows):
        raise Phase4D5EligibilityError("record_rows", "record IDs must be unique")
    cell_by_key = {
        (str(row["record_id"]), str(row["perturbation_id"])): row
        for row in cell_rows
    }
    if len(cell_by_key) != len(record_rows) * len(ALL_PERTURBATIONS):
        raise Phase4D5EligibilityError("cell_rows", "grid is incomplete or duplicated")
    expected_keys = {
        (record_id, perturbation_id)
        for record_id in records_by_id
        for perturbation_id in ALL_PERTURBATIONS
    }
    if set(cell_by_key) != expected_keys:
        raise Phase4D5EligibilityError("cell_rows", "grid keys mismatch")

    class_to_records: dict[int, list[str]] = defaultdict(list)
    group_to_records: dict[str, list[str]] = defaultdict(list)
    class_names: dict[int, set[str]] = defaultdict(set)
    for record_id, row in records_by_id.items():
        label = int(row["class_label"])
        class_to_records[label].append(record_id)
        group_to_records[str(row["group_id"])].append(record_id)
        if "mineral_name" in row:
            class_names[label].add(str(row["mineral_name"]))
    if len(class_to_records) != config.class_count:
        raise Phase4D5EligibilityError("classes", "denominator mismatch")

    class_rows: list[dict[str, object]] = []
    operator_rows: dict[str, dict[str, object]] = {}
    for perturbation_id in ALL_PERTURBATIONS:
        states = Counter(
            str(cell_by_key[(record_id, perturbation_id)]["state"])
            for record_id in records_by_id
        )
        if set(states) - set(TERMINAL_STATES):
            raise Phase4D5EligibilityError("cell state", "unknown terminal state")
        complete_records = {
            record_id
            for record_id in records_by_id
            if cell_by_key[(record_id, perturbation_id)]["state"] == "complete"
        }
        complete_classes = 0
        for class_label in sorted(class_to_records):
            required = sorted(class_to_records[class_label])
            complete_count = sum(record_id in complete_records for record_id in required)
            complete = complete_count == len(required)
            complete_classes += int(complete)
            names = class_names[class_label]
            mineral_name = sorted(names)[0] if names else f"class-{class_label}"
            class_state_counts = Counter(
                str(cell_by_key[(record_id, perturbation_id)]["state"])
                for record_id in required
            )
            class_rows.append(
                {
                    "class_label": class_label,
                    "complete": complete,
                    "complete_record_count": complete_count,
                    "mineral_name": mineral_name,
                    "perturbation_id": perturbation_id,
                    "required_record_count": len(required),
                    "state": (
                        "complete"
                        if complete
                        else (
                            "structurally_ineligible"
                            if perturbation_id in INACTIVE_PERTURBATIONS
                            else "closed_incomplete"
                        )
                    ),
                    "state_counts": {
                        state: class_state_counts.get(state, 0) for state in TERMINAL_STATES
                    },
                }
            )
        complete_groups = sum(
            all(record_id in complete_records for record_id in record_ids)
            for record_ids in group_to_records.values()
        )
        complete_occurrences = sum(
            int(records_by_id[record_id].get("occurrence_count", 1))
            for record_id in complete_records
        )
        if perturbation_id in {"p01", "p02", "p03", "p04"}:
            record_fraction = config.gates["p01_p04_record_fraction"]
            class_fraction = config.gates["p01_p04_class_fraction"]
        elif perturbation_id == "p05":
            record_fraction = config.gates["p05_record_fraction"]
            class_fraction = config.gates["p05_class_fraction"]
        elif perturbation_id in FULL_DOMAIN_PERTURBATIONS:
            record_fraction = config.gates["full_domain_record_fraction"]
            class_fraction = config.gates["full_domain_class_fraction"]
        else:
            record_fraction = 1.0
            class_fraction = 1.0
        required_records = _required_count(config.unique_record_count, record_fraction)
        required_classes = _required_count(config.class_count, class_fraction)
        runtime_failures = states.get("failed_runtime", 0)
        evaluable = (
            perturbation_id not in INACTIVE_PERTURBATIONS
            and len(complete_records) >= required_records
            and complete_classes >= required_classes
            and runtime_failures == 0
            and (
                perturbation_id in PEAK_PERTURBATIONS
                or states.get("not_applicable", 0) == 0
            )
        )
        operator_rows[perturbation_id] = {
            "class_fraction": complete_classes / config.class_count,
            "complete_class_count": complete_classes,
            "complete_group_count": complete_groups,
            "complete_occurrence_count": complete_occurrences,
            "complete_record_count": len(complete_records),
            "group_denominator": config.group_count,
            "not_applicable_count": states.get("not_applicable", 0),
            "occurrence_denominator": config.split_occurrence_count,
            "reason_counts": dict(
                sorted(
                    Counter(
                        str(cell_by_key[(record_id, perturbation_id)].get("reason_code"))
                        for record_id in records_by_id
                        if cell_by_key[(record_id, perturbation_id)].get("reason_code") is not None
                    ).items()
                )
            ),
            "record_fraction": len(complete_records) / config.unique_record_count,
            "required_class_count": required_classes,
            "required_record_count": required_records,
            "runtime_failure_count": runtime_failures,
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

    common_record_rows: list[dict[str, object]] = []
    common_complete_records: set[str] = set()
    for record_id in sorted(records_by_id):
        peak_states = {
            perturbation_id: str(cell_by_key[(record_id, perturbation_id)]["state"])
            for perturbation_id in PEAK_PERTURBATIONS
        }
        complete = all(state == "complete" for state in peak_states.values())
        if complete:
            common_complete_records.add(record_id)
        record = records_by_id[record_id]
        common_record_rows.append(
            {
                "class_label": int(record["class_label"]),
                "complete": complete,
                "group_id": str(record["group_id"]),
                "peak_states": peak_states,
                "record_id": record_id,
                "scope": "record",
            }
        )
    common_class_rows: list[dict[str, object]] = []
    common_complete_classes = 0
    for class_label in sorted(class_to_records):
        required = sorted(class_to_records[class_label])
        peak_states = {
            perturbation_id: (
                "complete"
                if all(
                    cell_by_key[(record_id, perturbation_id)]["state"] == "complete"
                    for record_id in required
                )
                else "closed_incomplete"
            )
            for perturbation_id in PEAK_PERTURBATIONS
        }
        complete = all(record_id in common_complete_records for record_id in required)
        common_complete_classes += int(complete)
        common_class_rows.append(
            {
                "class_label": class_label,
                "complete": complete,
                "peak_states": peak_states,
                "required_record_count": len(required),
                "scope": "class",
            }
        )
    common_rows = common_record_rows + common_class_rows
    common_required_records = _required_count(
        config.unique_record_count, config.gates["peak_common_record_fraction"]
    )
    common_required_classes = _required_count(
        config.class_count, config.gates["peak_common_class_fraction"]
    )
    peak_operator_pass = all(
        operator_rows[perturbation_id]["state"] == "evaluable"
        for perturbation_id in PEAK_PERTURBATIONS
    )
    peak_common_pass = (
        peak_operator_pass
        and len(common_complete_records) >= common_required_records
        and common_complete_classes >= common_required_classes
    )
    full_domain_pass = all(
        operator_rows[perturbation_id]["state"] == "evaluable"
        for perturbation_id in FULL_DOMAIN_PERTURBATIONS
    )
    gate = {
        "class_denominator": config.class_count,
        "full_domain_core": {
            "perturbation_ids": list(FULL_DOMAIN_PERTURBATIONS),
            "state": "evaluable" if full_domain_pass else "not_evaluable_coverage",
        },
        "group_audit_denominator": config.group_count,
        "overall_status": "pass" if full_domain_pass and peak_common_pass else "fail",
        "operators": operator_rows,
        "peak_common_support": {
            "complete_class_count": common_complete_classes,
            "complete_record_count": len(common_complete_records),
            "operator_gates_passed": peak_operator_pass,
            "record_fraction": len(common_complete_records) / config.unique_record_count,
            "class_fraction": common_complete_classes / config.class_count,
            "required_class_count": common_required_classes,
            "required_record_count": common_required_records,
            "state": "evaluable" if peak_common_pass else "not_evaluable_coverage",
        },
        "record_denominator": config.unique_record_count,
        "split_occurrence_audit_denominator": config.split_occurrence_count,
    }
    return class_rows, common_rows, gate


def validate_outcome_blind_payload(value: object, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_EXACT_KEYS or any(
                fragment in key_text for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise Phase4D5EligibilityError(
                    "outcome-blind boundary", f"forbidden field {key!r} at {path}"
                )
            validate_outcome_blind_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_outcome_blind_payload(item, path=f"{path}[{index}]")


def _run_identity(
    config: Phase4D5EligibilityConfig,
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
    return RUN_PREFIX + _sha256_bytes(_canonical_json_bytes(identity)), identity


def _manifest_document(
    config: Phase4D5EligibilityConfig,
    code: Mapping[str, object],
    environment: Mapping[str, object],
    run_id: str,
    run_identity: Mapping[str, object],
    record_rows: Sequence[Mapping[str, object]],
    occurrence_rows: Sequence[Mapping[str, object]],
    cell_rows: Sequence[Mapping[str, object]],
    class_rows: Sequence[Mapping[str, object]],
    common_rows: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> dict[str, object]:
    cell_states = Counter(str(row["state"]) for row in cell_rows)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "cells": len(cell_rows),
            "class_summaries": len(class_rows),
            "common_support": len(common_rows),
            "split_occurrences": len(occurrence_rows),
            "unique_records": len(record_rows),
        },
        "environment": environment,
        "experiment_id": EXPERIMENT_ID,
        "gate_status": gate["overall_status"],
        "perturbation_ids": list(ALL_PERTURBATIONS),
        "run_id": run_id,
        "run_identity": run_identity,
        "state_counts": {state: cell_states.get(state, 0) for state in TERMINAL_STATES},
        "storage": {
            "numerical_perturbation_arrays_stored": False,
            "receipts_only": True,
        },
        "synthetic_fixture": config.synthetic_fixture,
    }


def build_phase4_d5_eligibility_from_inputs(
    output_dir: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D5EligibilityConfig,
    worker_count: int,
) -> Phase4D5EligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D5EligibilityError("worker_count", "must be a positive integer")
    if sweep.sha256 != config.authorities.get("sweep_sha256") or tuple(sweep.alpha_grid) != config.alpha_grid:
        raise Phase4D5EligibilityError("sweep", "identity or alpha grid mismatch")
    if phase1_config.file_sha256 != config.authorities.get("phase1_core_config_sha256"):
        raise Phase4D5EligibilityError("phase1_config", "identity mismatch")
    if phase1_config.core_gate["float_relative_tolerance"] != config.native_gate_relative_tolerance:
        raise Phase4D5EligibilityError("phase1_config", "native gate tolerance mismatch")
    code, environment = _validate_code_and_environment(config)
    ledgers = reconstruct_d5_eligibility_ledgers(cohort, native_spectra, config)
    record_rows = [dict(row) for row in ledgers.unique_records]
    occurrence_rows = [dict(row) for row in ledgers.split_occurrences]
    estimates = [estimate_p10_peak_bytes(int(row["point_count"])) for row in record_rows]
    if max(estimates) > config.p10_memory_budget_bytes:
        raise Phase4D5EligibilityError(
            "p10 admission", "at least one record exceeds the frozen 64 GiB budget"
        )
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Phase4D5EligibilityError("output_dir", "must not already exist")

    native_by_index = {index: spectrum for index, spectrum in enumerate(native_spectra)}
    admission = P10MemoryAdmission(config.p10_memory_budget_bytes)
    grid = _support_grid(config)
    results: dict[int, tuple[dict[str, object], ...]] = {}
    with threadpool_limits(limits=1, user_api="blas"):
        if worker_count == 1:
            for record in record_rows:
                order = int(record["record_order"] )
                results[order] = _run_record_cells(
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
                        _run_record_cells,
                        record,
                        native_by_index[int(record["cohort_index"])],
                        phase1_config,
                        sweep,
                        config,
                        admission,
                        grid,
                    )
                    for record in record_rows
                }
                for order, future in futures.items():
                    results[order] = future.result()
    cell_rows = [
        row
        for order in range(len(record_rows))
        for row in results[order]
    ]
    class_rows, common_rows, gate = evaluate_d5_eligibility_gates(
        record_rows, cell_rows, config
    )
    run_id, run_identity = _run_identity(config, code, environment)
    manifest = _manifest_document(
        config,
        code,
        environment,
        run_id,
        run_identity,
        record_rows,
        occurrence_rows,
        cell_rows,
        class_rows,
        common_rows,
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
        record_rows,
        occurrence_rows,
        cell_rows,
        class_rows,
        common_rows,
        gate,
        manifest,
        marker,
    ]
    for value in payload_objects:
        validate_outcome_blind_payload(value)
    payloads = {
        "config.json": config.raw_bytes,
        "unique_records.jsonl": b"".join(_canonical_json_bytes(row) for row in record_rows),
        "split_occurrences.jsonl": b"".join(_canonical_json_bytes(row) for row in occurrence_rows),
        "cells.jsonl": b"".join(_canonical_json_bytes(row) for row in cell_rows),
        "classes.jsonl": b"".join(_canonical_json_bytes(row) for row in class_rows),
        "common_support.jsonl": b"".join(_canonical_json_bytes(row) for row in common_rows),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker_name: _canonical_json_bytes(marker),
    }
    output_dir.mkdir(parents=True)
    ordered_payloads = (*PAYLOAD_PREFIX, marker_name)
    for name in ordered_payloads:
        (output_dir / name).write_bytes(payloads[name])
    checksum = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n" for name in ordered_payloads
    ).encode("utf-8")
    (output_dir / "SHA256SUMS").write_bytes(checksum)
    return Phase4D5EligibilitySummary(
        path=output_dir,
        run_id=run_id,
        status=str(gate["overall_status"]),
        unique_record_count=len(record_rows),
        split_occurrence_count=len(occurrence_rows),
        cell_count=len(cell_rows),
        class_summary_count=len(class_rows),
    )


def build_phase4_d5_eligibility(
    output_root: Path,
    *,
    worker_count: int = 16,
) -> Phase4D5EligibilitySummary:
    config = load_phase4_d5_eligibility_config(ROOT / CONFIG_RELATIVE_PATH)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    cohort = load_d5_raw_cohort(
        ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH
    )
    native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids)
    code, environment = _validate_code_and_environment(config)
    run_id, _ = _run_identity(config, code, environment)
    return build_phase4_d5_eligibility_from_inputs(
        Path(output_root) / run_id,
        cohort=cohort,
        native_spectra=native,
        sweep=sweep,
        phase1_config=phase1_config,
        config=config,
        worker_count=worker_count,
    )


def _read_json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5EligibilityError(path.name, str(error)) from error
    if raw != _canonical_json_bytes(value):
        raise Phase4D5EligibilityError(path.name, "must use canonical JSON")
    return _object(path.name, value)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = []
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Phase4D5EligibilityError(f"{path.name}[{index}]", str(error)) from error
        if line != _canonical_json_bytes(value):
            raise Phase4D5EligibilityError(f"{path.name}[{index}]", "must use canonical JSON")
        rows.append(_object(f"{path.name}[{index}]", value))
    return rows


def _require_keys(path: str, row: Mapping[str, object], keys: set[str]) -> None:
    if set(row) != keys:
        raise Phase4D5EligibilityError(
            f"{path} schema",
            f"missing={sorted(keys-set(row))!r}; extra={sorted(set(row)-keys)!r}",
        )


def _validate_cell_rows(
    records: Sequence[Mapping[str, object]],
    cells: Sequence[Mapping[str, object]],
    config: Phase4D5EligibilityConfig,
) -> None:
    expected_order = [
        (str(record["record_id"]), perturbation_id)
        for record in records
        for perturbation_id in ALL_PERTURBATIONS
    ]
    observed_order = []
    point_counts = {str(row["record_id"]): int(row["point_count"]) for row in records}
    for index, row in enumerate(cells):
        _require_keys(f"cells[{index}]", row, CELL_KEYS)
        record_id = str(row["record_id"] )
        perturbation_id = str(row["perturbation_id"] )
        observed_order.append((record_id, perturbation_id))
        state = str(row["state"] )
        if state not in TERMINAL_STATES:
            raise Phase4D5EligibilityError(f"cells[{index}].state", "unknown")
        outputs = row["outputs"]
        if not isinstance(outputs, list):
            raise Phase4D5EligibilityError(f"cells[{index}].outputs", "must be an array")
        if int(row["output_count"]) != len(outputs):
            raise Phase4D5EligibilityError(f"cells[{index}].output_count", "mismatch")
        if state == "complete":
            if len(outputs) != len(ALPHA_GRID) or row["reason_code"] is not None or row["exception"] is not None:
                raise Phase4D5EligibilityError(f"cells[{index}]", "complete receipt mismatch")
            for output_index, output in enumerate(outputs):
                output_row = _object(f"cells[{index}].outputs[{output_index}]", output)
                _require_keys(f"cells[{index}].outputs[{output_index}]", output_row, OUTPUT_KEYS)
                if float(output_row["alpha"]) != ALPHA_GRID[output_index]:
                    raise Phase4D5EligibilityError(f"cells[{index}].outputs", "alpha order mismatch")
                support_hash = output_row["support_projection_sha256"]
                if perturbation_id in {"p11", "p12"}:
                    _lower_hex("support_projection_sha256", support_hash)
                elif support_hash is not None:
                    raise Phase4D5EligibilityError("support_projection_sha256", "must be null outside P11/P12")
        elif outputs or int(row["output_count"]) != 0:
            raise Phase4D5EligibilityError(f"cells[{index}]", "noncomplete receipt owns output")
        if perturbation_id in INACTIVE_PERTURBATIONS:
            if state != "structurally_ineligible" or row["reason_code"] != STRUCTURAL_REASON:
                raise Phase4D5EligibilityError(f"cells[{index}]", "inactive semantics mismatch")
        if state == "not_applicable" and (
            perturbation_id not in PEAK_PERTURBATIONS
            or row["reason_code"] not in NOT_APPLICABLE_REASONS
        ):
            raise Phase4D5EligibilityError(f"cells[{index}]", "not-applicable taxonomy mismatch")
        expected_estimate = (
            estimate_p10_peak_bytes(point_counts[record_id])
            if perturbation_id == "p10"
            else None
        )
        if row["p10_estimated_peak_bytes"] != expected_estimate:
            raise Phase4D5EligibilityError(f"cells[{index}].p10_estimated_peak_bytes", "mismatch")
    if observed_order != expected_order:
        raise Phase4D5EligibilityError("cells order", "does not match record by perturbation order")


def verify_phase4_d5_eligibility(
    path: Path,
    *,
    worker_count: int = 16,
) -> Phase4D5EligibilitySummary:
    from rpe.runner.phase4_d5_eligibility_verifier import (
        verify_phase4_d5_eligibility as independent_verify,
    )

    return independent_verify(Path(path), worker_count=worker_count)


__all__ = [
    "ACTIVE_PERTURBATIONS",
    "ALL_PERTURBATIONS",
    "ARTIFACT_STATIC_FILES",
    "D5EligibilityLedgers",
    "Phase4D5EligibilityConfig",
    "Phase4D5EligibilityError",
    "Phase4D5EligibilitySummary",
    "build_phase4_d5_eligibility",
    "build_phase4_d5_eligibility_from_inputs",
    "evaluate_d5_eligibility_gates",
    "load_phase4_d5_eligibility_config",
    "parse_phase4_d5_eligibility_config",
    "reconstruct_d5_eligibility_ledgers",
    "validate_outcome_blind_payload",
    "verify_phase4_d5_eligibility",
]
