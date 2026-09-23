"""Independent verifier for Phase 4 D4 Protocol-A eligibility."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
import struct
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from rpe.downstream.sugar_quantitative import (
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    README_BYTES,
    README_MEMBER,
    README_SHA256,
    TARGET_MEMBER,
    TARGET_MEMBER_BYTES,
    TARGET_MEMBER_SHA256,
    D4SugarCohort,
    load_d4_sugar_cohort,
)
from rpe.evaluation import PeakPairInput, SingleSpectrumInput, Spectrum1D, SpectrumPairInput, evaluate_metric
from rpe.methods import load_classical_catalog
from rpe.methods.catalog import Phase3System
from rpe.methods.classical.peaks import run_peak_detection_system
from rpe.metrics import (
    ISLikeStructureToNoiseMetric,
    MAEMetric,
    MSEMetric,
    NMSEMetric,
    PeakDetectionCurvesMetric,
    PearsonRMetric,
    RMSEMetric,
    SAMMetric,
    Wasserstein1Metric,
)
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus, Phase1Cell
from rpe.runner.phase4_d4_eligibility_authority import CONFIG_BYTES, CONFIG_SHA256


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-d4-protocol-a-full-domain-eligibility-config-v1"
EXPERIMENT_ID = "phase4-d4-protocol-a-full-domain-eligibility-v1"
RUN_PREFIX = "phase4-d4-protocol-a-full-domain-eligibility-"
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
PHASE05_D4_PROTOCOL_RELATIVE_PATH = "experiments/phase05/configs/d4_sugar_protocol.json"
ARCHIVE_RELATIVE_PATH = "data/raw/ramanbench/cache/10779223/Raw data.zip"
CATALOG_RELATIVE_PATH = "experiments/phase3/configs/classical_system_catalog_v1.json"
AUTHORITY_FILE_RECEIPTS = MappingProxyType(
    {
        "archive": (ARCHIVE_RELATIVE_PATH, ARCHIVE_BYTES, ARCHIVE_SHA256),
        "classical_catalog": (CATALOG_RELATIVE_PATH, 376665, "8ad40b08df78b8905d75a67c84a2bb328531ef17cb12704ffad04f0a8f925d8f"),
        "parent_plan": ("raman_preproc_benchmark_plan_v2.md", 46363, "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5"),
        "phase05_d4_loader_report": ("reports/phase05/d4_step02_read_only_loader.md", 9203, "a522dc33d810c61edf3e33d94fc628b8d9a6f703fae8fc15f0fe4dc1918e19c1"),
        "phase05_d4_protocol": (PHASE05_D4_PROTOCOL_RELATIVE_PATH, 4799, "69a5dba7ab45cbc421f988439e4f3b3aa9b99505d6ff13cbb3d3b89389d071c8"),
        "phase05_d4_protocol_audit": ("reports/phase05/d4_step01_protocol_audit.json", 7421, "7d6ea22b7a96a5e7546d6fd6bc51e68024b59a5214888b0d83c61908aafbd506"),
        "phase05_d4_protocol_report": ("reports/phase05/d4_step01_protocol.md", 11265, "246b8bc86bd8b454a51b48fae9953683d555fc0d09d729c4baaa1102f1eb7304"),
        "phase05_screening_decision": ("reports/phase05/phase05_internal_screening_decision.md", 8997, "de9fc2c13ea211e57982f64a413614b5040c8ab8b5c3961622741d741ab8830b"),
        "phase1_core_config": (PHASE1_CONFIG_RELATIVE_PATH, 2350, "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd"),
        "phase4_preregistration": ("reports/phase4/step01_phase4_feasibility_preregistration.md", 31265, "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f"),
        "phase4_step03": ("reports/phase4/step03_alignment_core.md", 8513, "76b1ffdc3544673cebbd6520803ae72aeff610a6a66a2306dd317c9402fdcbbe"),
        "phase4_step25": ("reports/phase4/step25_d1_protocol_b_full_domain.md", 9063, "1f6da0e89baaafd77883c590a9296f5b82a11863eb849bcb117ee4099cca358b"),
        "phase4_step26_design": ("reports/phase4/step26_d4_protocol_a_full_domain_eligibility_design.md", 29610, "5158cc1904a69b7b00607495fd457ed180802a9df79690749235b70e345b8c49"),
        "sugar_quantitative": ("rpe/downstream/sugar_quantitative.py", 25666, "a6bfd6a3556b64a3014adef6e567115e2f525c3bf9f03d67bfab628d0f6aba56"),
        "sweep": (SWEEP_RELATIVE_PATH, 559, "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"),
    }
)
AUTHORITY_MEMBER_RECEIPTS = MappingProxyType(
    {
        "source_readme_member": (README_MEMBER, README_BYTES, README_SHA256),
        "target_member": (TARGET_MEMBER, TARGET_MEMBER_BYTES, TARGET_MEMBER_SHA256),
    }
)
CODE_AUTHORITY_PATHS = (
    "rpe/downstream/sugar_quantitative.py",
    "rpe/evaluation/contracts.py",
    "rpe/methods/catalog.py",
    "rpe/methods/classical/peaks.py",
    "rpe/metrics/fidelity.py",
    "rpe/metrics/peak.py",
    "rpe/metrics/reference_free.py",
    "rpe/metrics/transport.py",
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
    "rpe/runner/phase4_d4_eligibility.py",
    "rpe/runner/phase4_d4_eligibility_verifier.py",
    "tests/test_phase4_d4_eligibility.py",
    "tools/run_phase4_d4_eligibility.py",
)
TRUST_ANCHOR = MappingProxyType(
    {
        "config_authority_relative_path": "rpe/runner/phase4_d4_eligibility_authority.py",
        "config_binds_authority": False,
        "direction": "authority_to_config_only",
    }
)

ACTIVE_PERTURBATION_IDS = (
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
INACTIVE_PERTURBATION_IDS = ("p06", "p07")
ALL_PERTURBATION_IDS = tuple(f"p{index:02d}" for index in range(1, 13))
FULL_DOMAIN_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
PEAK_PERTURBATION_IDS = ("p01", "p02", "p03", "p04", "p05")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
METRIC_OUTPUT_IDS = (
    "mse",
    "rmse",
    "mae",
    "sam",
    "pearson_r",
    "nmse",
    "wasserstein_1_cm1",
    "is_like_structure_to_noise",
    "precision",
    "recall",
    "f1",
    "artifact_peak_ratio",
    "missing_peak_ratio",
)
CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
STRUCTURAL_REASON = "structurally_ineligible_missing_explicit_baseline"
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "acc_cross",
        "accuracy",
        "alignment",
        "alignment_gap",
        "bootstrap",
        "figure",
        "lod",
        "loq",
        "metric_value",
        "peak_count",
        "peak_list",
        "peak_lists",
        "peaks",
        "prediction",
        "predictions",
        "selected_c",
    }
)
FORBIDDEN_KEY_FRAGMENTS = (
    "prediction",
    "metric_value",
    "peak_count",
    "peak_list",
    "bootstrap",
    "alignment",
    "figure",
    "lod",
    "loq",
)
ARTIFACT_STATIC_FILES = (
    "config.json",
    "source_records.jsonl",
    "well_folds.jsonl",
    "model_cells.jsonl",
    "model_role_occurrences.jsonl",
    "operator_cells.jsonl",
    "blank_cells.jsonl",
    "record_conditions.jsonl",
    "blank_conditions.jsonl",
    "well_summaries.jsonl",
    "common_support.jsonl",
    "gate.json",
    "manifest.json",
    "SHA256SUMS",
)


class Phase4D4EligibilityVerifierError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D4EligibilityConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    schema_version: str
    experiment_id: str
    synthetic_fixture: bool
    active_perturbation_ids: tuple[str, ...]
    inactive_perturbation_ids: tuple[str, ...]
    full_domain_perturbation_ids: tuple[str, ...]
    peak_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_output_ids: tuple[str, ...]
    cwt_system_id: str
    mixture_record_count: int
    blank_record_count: int
    mixture_well_count: int
    blank_well_count: int
    model_cell_count: int
    model_role_occurrence_count: int
    records_per_well: int
    support_coordinates_cm1: tuple[float, ...]
    support_point_count: int
    support_max_gap_cm1: float
    support_first_cm1: float
    support_last_cm1: float
    p10_memory_budget_bytes: int
    phase1_native_gate_relative_tolerance: float
    gates: Mapping[str, float]
    authorities: Mapping[str, object]
    code_authority: Mapping[str, Mapping[str, object]]
    environment_authority: Mapping[str, object]
    trust_anchor: Mapping[str, object]
    frozen_identities: Mapping[str, object]
    claim_boundary: str
    native_axis_float32_sha256: str
    support_axis_float64_sha256: str


@dataclass(frozen=True)
class D4EligibilityInputs:
    mixture_spectra: tuple[Spectrum1D, ...]
    blank_spectra: tuple[Spectrum1D, ...]
    mixture_record_ids: tuple[str, ...]
    blank_record_ids: tuple[str, ...]
    well_ids: tuple[str, ...]
    blank_well_ids: tuple[str, ...]
    support_axis_cm1: np.ndarray
    well_folds: tuple[Mapping[str, object], ...]
    model_cells: tuple[Mapping[str, object], ...]
    model_role_occurrences: tuple[Mapping[str, object], ...]
    source_records: tuple[Mapping[str, object], ...]
    mixture_record_ids_sha256: str
    well_ids_sha256: str


@dataclass(frozen=True)
class Phase4D4EligibilitySummary:
    path: Path
    run_id: str
    status: str
    marker_filename: str
    mixture_record_count: int
    model_cell_count: int
    model_role_occurrence_count: int


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
    if is_dataclass(value):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D4EligibilityVerifierError(
        "json", f"unsupported value {type(value).__name__}"
    )


def _reject_nonfinite(value: str) -> None:
    raise Phase4D4EligibilityVerifierError("config", f"unsupported JSON constant {value!r}")


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D4EligibilityVerifierError(path, "must be an object")
    return value


def _float(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D4EligibilityVerifierError(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise Phase4D4EligibilityVerifierError(path, "must be finite")
    return result


def _int(path: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4D4EligibilityVerifierError(path, "must be an integer")
    if minimum is not None and value < minimum:
        raise Phase4D4EligibilityVerifierError(path, f"must be >= {minimum}")
    return value


def _string_tuple(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise Phase4D4EligibilityVerifierError(path, "must be an array of strings")
    return tuple(value)


def _float_tuple(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise Phase4D4EligibilityVerifierError(path, "must be an array")
    return tuple(_float(f"{path}[{index}]", item) for index, item in enumerate(value))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _authority_document() -> dict[str, Mapping[str, object]]:
    document = {
        key: {"path": path, "bytes": byte_count, "sha256": sha256}
        for key, (path, byte_count, sha256) in AUTHORITY_FILE_RECEIPTS.items()
    }
    document.update(
        {
            key: {
                "archive_path": ARCHIVE_RELATIVE_PATH,
                "member_path": member_path,
                "bytes": byte_count,
                "sha256": sha256,
            }
            for key, (member_path, byte_count, sha256) in AUTHORITY_MEMBER_RECEIPTS.items()
        }
    )
    return document


def _code_authority_document() -> dict[str, Mapping[str, object]]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha_file(ROOT / relative_path),
        }
        for relative_path in CODE_AUTHORITY_PATHS
    }


def _environment_authority_document() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _validate_real_authorities(
    *,
    authorities: Mapping[str, object],
    code_authority: Mapping[str, object],
    environment_authority: Mapping[str, object],
    trust_anchor: Mapping[str, object],
    verify_live_files: bool,
) -> None:
    if dict(authorities) != _authority_document():
        raise Phase4D4EligibilityVerifierError(
            "authorities", "must contain exact frozen receipts"
        )
    if tuple(code_authority) != CODE_AUTHORITY_PATHS:
        raise Phase4D4EligibilityVerifierError(
            "code_authority", "path set or order mismatch"
        )
    for relative_path, receipt_value in code_authority.items():
        receipt = _object(f"code_authority.{relative_path}", receipt_value)
        if (
            set(receipt) != {"bytes", "sha256"}
            or _int(f"code_authority.{relative_path}.bytes", receipt.get("bytes"), minimum=1) < 1
            or len(str(receipt.get("sha256", ""))) != 64
        ):
            raise Phase4D4EligibilityVerifierError("code_authority", "invalid receipt")
    if dict(environment_authority) != _environment_authority_document():
        raise Phase4D4EligibilityVerifierError("environment_authority", "mismatch")
    if dict(trust_anchor) != dict(TRUST_ANCHOR):
        raise Phase4D4EligibilityVerifierError("trust_anchor", "mismatch")
    if not verify_live_files:
        return
    if dict(code_authority) != _code_authority_document():
        raise Phase4D4EligibilityVerifierError(
            "code_authority", "bytes or SHA-256 mismatch"
        )
    for key, (relative_path, expected_bytes, expected_sha256) in AUTHORITY_FILE_RECEIPTS.items():
        target = ROOT / relative_path
        if (
            not target.is_file()
            or target.stat().st_size != expected_bytes
            or _sha_file(target) != expected_sha256
        ):
            raise Phase4D4EligibilityVerifierError(
                f"authorities.{key}", "live file bytes or SHA-256 mismatch"
            )
    with zipfile.ZipFile(ROOT / ARCHIVE_RELATIVE_PATH) as archive:
        for key, (member_path, expected_bytes, expected_sha256) in AUTHORITY_MEMBER_RECEIPTS.items():
            try:
                info = archive.getinfo(member_path)
                payload = archive.read(member_path)
            except (KeyError, OSError, zipfile.BadZipFile) as error:
                raise Phase4D4EligibilityVerifierError(
                    f"authorities.{key}", str(error)
                ) from error
            if info.file_size != expected_bytes or _sha_bytes(payload) != expected_sha256:
                raise Phase4D4EligibilityVerifierError(
                    f"authorities.{key}", "archive member bytes or SHA-256 mismatch"
                )
def _ids_sha(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _array_sha(value: np.ndarray, *, dtype: str) -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _hex_alpha(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def _support_axis(config: Phase4D4EligibilityConfig) -> np.ndarray:
    values = np.asarray(config.support_coordinates_cm1, dtype="<f8")
    if values.size != config.support_point_count:
        raise Phase4D4EligibilityVerifierError("support_grid", "point count mismatch")
    if config.support_axis_float64_sha256 and _array_sha(values, dtype="<f8") != config.support_axis_float64_sha256:
        raise Phase4D4EligibilityVerifierError("support_grid", "support axis float64 hash mismatch")
    return values


def _validate_native_axis(native_axis: np.ndarray, config: Phase4D4EligibilityConfig) -> None:
    axis_f4 = np.ascontiguousarray(np.asarray(native_axis, dtype="<f4"))
    axis_f8 = np.ascontiguousarray(np.asarray(axis_f4, dtype="<f8"))
    if axis_f8.ndim != 1 or axis_f8.size < config.support_point_count:
        raise Phase4D4EligibilityVerifierError("support_projection", "native support too short")
    if not np.isfinite(axis_f8).all():
        raise Phase4D4EligibilityVerifierError("support_projection", "native axis must be finite")
    if not np.all(np.diff(axis_f8) > 0.0):
        raise Phase4D4EligibilityVerifierError("support_projection", "native axis must be strictly increasing")
    if config.native_axis_float32_sha256 and _array_sha(axis_f4, dtype="<f4") != config.native_axis_float32_sha256:
        raise Phase4D4EligibilityVerifierError("support_projection", "native axis float32 hash mismatch")
    native_axis_f64_sha256 = str(config.frozen_identities.get("native_axis_f64_sha256", ""))
    if native_axis_f64_sha256 and _array_sha(axis_f8, dtype="<f8") != native_axis_f64_sha256:
        raise Phase4D4EligibilityVerifierError("support_projection", "native axis float64 hash mismatch")
    support = _support_axis(config)
    if support.size != config.support_point_count:
        raise Phase4D4EligibilityVerifierError("support_projection", "point count mismatch")
    if not np.isclose(float(support[0]), config.support_first_cm1, rtol=0.0, atol=1e-12):
        raise Phase4D4EligibilityVerifierError("support_projection", "support first point mismatch")
    if not np.isclose(float(support[-1]), config.support_last_cm1, rtol=0.0, atol=1e-12):
        raise Phase4D4EligibilityVerifierError("support_projection", "support last point mismatch")
    if config.support_axis_float64_sha256 and _array_sha(support, dtype="<f8") != config.support_axis_float64_sha256:
        raise Phase4D4EligibilityVerifierError("support_projection", "support axis float64 hash mismatch")
    support_axis_f32_sha256 = str(config.frozen_identities.get("support_axis_f32_sha256", ""))
    if support_axis_f32_sha256 and _array_sha(support, dtype="<f4") != support_axis_f32_sha256:
        raise Phase4D4EligibilityVerifierError("support_projection", "support axis float32 hash mismatch")
    if float(np.max(np.diff(support))) > config.support_max_gap_cm1:
        raise Phase4D4EligibilityVerifierError("support_projection", "support gap exceeds the frozen gate")


def parse_phase4_d4_eligibility_config(
    path: Path,
    raw_bytes: bytes,
    *,
    require_frozen_identity: bool = True,
) -> Phase4D4EligibilityConfig:
    try:
        document = json.loads(raw_bytes, parse_constant=_reject_nonfinite)
    except Phase4D4EligibilityVerifierError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D4EligibilityVerifierError(str(path), str(error)) from error
    if raw_bytes != _canonical_json_bytes(document):
        raise Phase4D4EligibilityVerifierError(str(path), "must use canonical JSON")
    sha256 = _sha_bytes(raw_bytes)
    if require_frozen_identity and (len(raw_bytes) != CONFIG_BYTES or sha256 != CONFIG_SHA256):
        raise Phase4D4EligibilityVerifierError(str(path), "frozen config identity mismatch")
    root = _object("config", document)
    schema_version = str(root.get("schema_version"))
    experiment_id = str(root.get("experiment_id"))
    if schema_version != SCHEMA_VERSION or experiment_id != EXPERIMENT_ID:
        raise Phase4D4EligibilityVerifierError("config", "schema or experiment identity mismatch")

    active = _string_tuple("active_perturbation_ids", root.get("active_perturbation_ids"))
    inactive = _string_tuple("inactive_perturbation_ids", root.get("inactive_perturbation_ids"))
    full_domain = _string_tuple("full_domain_perturbation_ids", root.get("full_domain_perturbation_ids"))
    peak = _string_tuple("peak_perturbation_ids", root.get("peak_perturbation_ids"))
    alpha_grid = _float_tuple("alpha_grid", root.get("alpha_grid"))
    metric_ids = _string_tuple("metric_output_ids", root.get("metric_output_ids"))
    if active != ACTIVE_PERTURBATION_IDS:
        raise Phase4D4EligibilityVerifierError("active_perturbation_ids", "must equal the frozen perturbation set")
    if inactive != INACTIVE_PERTURBATION_IDS:
        raise Phase4D4EligibilityVerifierError("inactive_perturbation_ids", "must equal the frozen perturbation set")
    if full_domain != FULL_DOMAIN_PERTURBATION_IDS:
        raise Phase4D4EligibilityVerifierError("full_domain_perturbation_ids", "must equal the frozen perturbation set")
    if peak != PEAK_PERTURBATION_IDS:
        raise Phase4D4EligibilityVerifierError("peak_perturbation_ids", "must equal the frozen perturbation set")
    if alpha_grid != ALPHA_GRID:
        raise Phase4D4EligibilityVerifierError("alpha_grid", "must equal the frozen alpha grid")
    if metric_ids != METRIC_OUTPUT_IDS:
        raise Phase4D4EligibilityVerifierError("metric_output_ids", "must equal the frozen metric output set")

    denominators = _object("denominators", root.get("denominators"))
    support_grid = _object("support_grid", root.get("support_grid"))
    p10 = _object("p10", root.get("p10"))
    gates_raw = _object("gates", root.get("gates"))
    synthetic_fixture = bool(root.get("synthetic_fixture", False))
    authorities_value = root.get("authorities")
    if authorities_value is None:
        if synthetic_fixture:
            authorities = MappingProxyType({})
        else:
            raise Phase4D4EligibilityVerifierError("authorities", "must be an object")
    else:
        authorities = MappingProxyType(dict(_object("authorities", authorities_value)))
    code_authority = MappingProxyType(
        {
            str(relative_path): MappingProxyType(
                dict(_object(f"code_authority.{relative_path}", receipt))
            )
            for relative_path, receipt in _object(
                "code_authority", root.get("code_authority", {})
            ).items()
        }
    )
    environment_authority = MappingProxyType(
        dict(_object("environment_authority", root.get("environment_authority", {})))
    )
    trust_anchor = MappingProxyType(
        dict(_object("trust_anchor", root.get("trust_anchor", {})))
    )
    if synthetic_fixture:
        if authorities or code_authority or environment_authority or trust_anchor:
            raise Phase4D4EligibilityVerifierError(
                "synthetic authority", "synthetic fixtures require empty authority mappings"
            )
    else:
        _validate_real_authorities(
            authorities=authorities,
            code_authority=code_authority,
            environment_authority=environment_authority,
            trust_anchor=trust_anchor,
            verify_live_files=require_frozen_identity,
        )
    frozen_identities = MappingProxyType(dict(_object("frozen_identities", root.get("frozen_identities"))))
    gates = MappingProxyType({key: _float(f"gates.{key}", value) for key, value in gates_raw.items()})
    coordinates = support_grid.get("coordinates_cm1")
    if coordinates is not None:
        support_coordinates = _float_tuple("support_grid.coordinates_cm1", coordinates)
        if len(support_coordinates) == 0:
            raise Phase4D4EligibilityVerifierError("support_grid.coordinates_cm1", "must not be empty")
        support_first_cm1 = support_coordinates[0]
        support_last_cm1 = support_coordinates[-1]
    else:
        if synthetic_fixture:
            support_first_cm1 = _float("support_grid.first_cm1", support_grid.get("first_cm1"))
            support_last_cm1 = _float("support_grid.last_cm1", support_grid.get("last_cm1"))
            support_coordinates = ()
        else:
            raise Phase4D4EligibilityVerifierError(
                "support_grid.coordinates_cm1",
                "real frozen config must store literal support coordinates",
            )

    frozen_identity_dict = dict(frozen_identities)
    if coordinates is not None:
        frozen_identity_dict["support_coordinates_cm1"] = list(support_coordinates)
    frozen_identities = MappingProxyType(frozen_identity_dict)

    return Phase4D4EligibilityConfig(
        path=Path(path),
        raw_bytes=bytes(raw_bytes),
        sha256=sha256,
        schema_version=schema_version,
        experiment_id=experiment_id,
        synthetic_fixture=synthetic_fixture,
        active_perturbation_ids=active,
        inactive_perturbation_ids=inactive,
        full_domain_perturbation_ids=full_domain,
        peak_perturbation_ids=peak,
        alpha_grid=alpha_grid,
        metric_output_ids=metric_ids,
        cwt_system_id=str(root.get("cwt_system_id")),
        mixture_record_count=_int("denominators.mixture_record_count", denominators.get("mixture_record_count"), minimum=1),
        blank_record_count=_int("denominators.blank_record_count", denominators.get("blank_record_count"), minimum=0),
        mixture_well_count=_int("denominators.mixture_well_count", denominators.get("mixture_well_count"), minimum=1),
        blank_well_count=_int("denominators.blank_well_count", denominators.get("blank_well_count"), minimum=0),
        model_cell_count=_int("denominators.model_cell_count", denominators.get("model_cell_count"), minimum=1),
        model_role_occurrence_count=_int("denominators.model_role_occurrence_count", denominators.get("model_role_occurrence_count"), minimum=1),
        records_per_well=_int("denominators.records_per_well", denominators.get("records_per_well", 32), minimum=1),
        support_coordinates_cm1=tuple(support_coordinates),
        support_point_count=_int("support_grid.point_count", support_grid.get("point_count"), minimum=1),
        support_max_gap_cm1=_float("support_grid.max_in_range_native_gap_cm1", support_grid.get("max_in_range_native_gap_cm1")),
        support_first_cm1=support_first_cm1,
        support_last_cm1=support_last_cm1,
        p10_memory_budget_bytes=_int("p10.memory_budget_bytes", p10.get("memory_budget_bytes"), minimum=1),
        phase1_native_gate_relative_tolerance=_float(
            "phase1_native_gate_relative_tolerance",
            root.get("phase1_native_gate_relative_tolerance"),
        ),
        gates=gates,
        authorities=authorities,
        code_authority=code_authority,
        environment_authority=environment_authority,
        trust_anchor=trust_anchor,
        frozen_identities=frozen_identities,
        claim_boundary=str(root.get("claim_boundary")),
        native_axis_float32_sha256=str(frozen_identities.get("native_axis_f32_sha256", "")),
        support_axis_float64_sha256=str(frozen_identities.get("support_axis_f64_sha256", "")),
    )


def load_phase4_d4_eligibility_config(path: Path) -> Phase4D4EligibilityConfig:
    return parse_phase4_d4_eligibility_config(Path(path), Path(path).read_bytes(), require_frozen_identity=True)


def reconstruct_d4_eligibility_inputs(cohort: D4SugarCohort, config: Phase4D4EligibilityConfig) -> D4EligibilityInputs:
    mixture_axis = np.asarray(cohort.wavenumber, dtype="<f4")
    _validate_native_axis(mixture_axis, config)
    support_axis = _support_axis(config)
    mixture_wells = tuple(sorted(set(cohort.well_ids)))
    blank_wells = tuple(sorted(set(cohort.blank_well_ids)))
    if len(cohort.record_ids) != config.mixture_record_count:
        raise Phase4D4EligibilityVerifierError("cohort.record_ids", "mixture record count mismatch")
    if len(cohort.blank_record_ids) != config.blank_record_count:
        raise Phase4D4EligibilityVerifierError("cohort.blank_record_ids", "blank record count mismatch")
    if len(mixture_wells) != config.mixture_well_count:
        raise Phase4D4EligibilityVerifierError("cohort.well_ids", "mixture well count mismatch")
    if len(blank_wells) != config.blank_well_count:
        raise Phase4D4EligibilityVerifierError("cohort.blank_well_ids", "blank well count mismatch")

    mixture_spectra = tuple(
        Spectrum1D(
            spectrum_id=f"d4_sugar_low_snr::{record_id}",
            sample_id=well_id,
            axis_cm1=np.asarray(mixture_axis, dtype="<f8"),
            intensity=np.asarray(cohort.intensity[index], dtype="<f8"),
        )
        for index, (record_id, well_id) in enumerate(zip(cohort.record_ids, cohort.well_ids, strict=True))
    )
    blank_spectra = tuple(
        Spectrum1D(
            spectrum_id=f"d4_blank::{record_id}",
            sample_id=well_id,
            axis_cm1=np.asarray(mixture_axis, dtype="<f8"),
            intensity=np.asarray(cohort.blank_intensity[index], dtype="<f8"),
        )
        for index, (record_id, well_id) in enumerate(zip(cohort.blank_record_ids, cohort.blank_well_ids, strict=True))
    )

    well_folds = tuple(
        {
            "fold_index": fold_index,
            "record_ids_sha256": str(config.frozen_identities.get("fold_record_ids_sha256", [""] * len(cohort.folds))[fold_index]),
            "source_members_sha256": str(config.frozen_identities.get("fold_source_members_sha256", [""] * len(cohort.folds))[fold_index]),
            "well_ids_sha256": str(config.frozen_identities.get("fold_well_ids_sha256", [""] * len(cohort.folds))[fold_index]),
        }
        for fold_index in range(len(cohort.folds))
    )

    model_cells: list[Mapping[str, object]] = []
    model_role_occurrences: list[Mapping[str, object]] = []
    for split in cohort.splits:
        train_records = tuple(sorted(cohort.record_ids[index] for index in split.train_indices.tolist()))
        validation_records = tuple(sorted(cohort.record_ids[index] for index in split.validation_indices.tolist()))
        test_records = tuple(sorted(cohort.record_ids[index] for index in split.test_indices.tolist()))
        cell = {
            "seed": int(split.seed),
            "train_folds": tuple(int(value) for value in split.train_folds),
            "validation_fold": int(split.validation_fold),
            "test_fold": int(split.test_fold),
            "train_record_ids": train_records,
            "validation_record_ids": validation_records,
            "test_record_ids": test_records,
        }
        model_cells.append(cell)
        for record_id in train_records:
            model_role_occurrences.append({"record_id": record_id, "role": "train", "seed": int(split.seed)})
        for record_id in validation_records:
            model_role_occurrences.append({"record_id": record_id, "role": "validation", "seed": int(split.seed)})
        for record_id in test_records:
            model_role_occurrences.append({"record_id": record_id, "role": "test", "seed": int(split.seed)})
        for record_id in cohort.blank_record_ids:
            model_role_occurrences.append({"record_id": record_id, "role": "blank_auxiliary", "seed": int(split.seed)})
    if len(model_cells) != config.model_cell_count:
        raise Phase4D4EligibilityVerifierError("model_cells", "count mismatch")
    if len(model_role_occurrences) != config.model_role_occurrence_count:
        raise Phase4D4EligibilityVerifierError("model_role_occurrences", "count mismatch")

    source_records = tuple(
        {
            "record_id": record_id,
            "scope": "mixture",
            "well_id": well_id,
            "support_axis_sha256": _array_sha(support_axis, dtype="<f8"),
        }
        for record_id, well_id in zip(cohort.record_ids, cohort.well_ids, strict=True)
    ) + tuple(
        {
            "record_id": record_id,
            "scope": "blank_auxiliary",
            "well_id": well_id,
            "support_axis_sha256": _array_sha(support_axis, dtype="<f8"),
        }
        for record_id, well_id in zip(cohort.blank_record_ids, cohort.blank_well_ids, strict=True)
    )

    return D4EligibilityInputs(
        mixture_spectra=mixture_spectra,
        blank_spectra=blank_spectra,
        mixture_record_ids=tuple(cohort.record_ids),
        blank_record_ids=tuple(cohort.blank_record_ids),
        well_ids=mixture_wells,
        blank_well_ids=blank_wells,
        support_axis_cm1=np.ascontiguousarray(support_axis, dtype="<f8"),
        well_folds=well_folds,
        model_cells=tuple(model_cells),
        model_role_occurrences=tuple(model_role_occurrences),
        source_records=source_records,
        mixture_record_ids_sha256=_ids_sha(cohort.record_ids),
        well_ids_sha256=_ids_sha(mixture_wells),
    )


def project_d4_support(spectrum: Spectrum1D, config: Phase4D4EligibilityConfig) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size:
        raise Phase4D4EligibilityVerifierError("spectrum", "axis and intensity must be aligned one-dimensional arrays")
    support = _support_axis(config)
    output = _project_to_support_axis(axis=axis, intensity=intensity, support=support, config=config)
    if output.size != config.support_point_count or not np.isfinite(output).all():
        raise Phase4D4EligibilityVerifierError("support_projection", "support projection invalid")
    return output


def _project_to_support_axis(
    *,
    axis: np.ndarray,
    intensity: np.ndarray,
    support: np.ndarray,
    config: Phase4D4EligibilityConfig,
) -> np.ndarray:
    left = int(np.searchsorted(axis, support[0], side="left"))
    if left < axis.size:
        candidate_axis = np.asarray(axis[left:], dtype="<f8")
        if candidate_axis.size == support.size and np.array_equal(candidate_axis, np.asarray(support, dtype="<f8")):
            return np.ascontiguousarray(intensity[left:], dtype="<f4")
    if float(axis[0]) > float(support[0]) or float(axis[-1]) < float(support[-1]):
        raise Phase4D4EligibilityVerifierError("support_projection", "extrapolation required")
    right = int(np.searchsorted(axis, support[-1], side="right"))
    native_in_range = axis[left:right]
    if native_in_range.size < 2:
        raise Phase4D4EligibilityVerifierError("support_projection", "insufficient in-range native support")
    if float(np.max(np.diff(native_in_range))) > config.support_max_gap_cm1:
        raise Phase4D4EligibilityVerifierError("support_projection", "support gap exceeds the frozen gate")
    return np.ascontiguousarray(np.interp(support, axis, intensity), dtype="<f4")


def _metric_objects() -> Mapping[str, object]:
    return {
        "mse": MSEMetric(),
        "rmse": RMSEMetric(),
        "mae": MAEMetric(),
        "sam": SAMMetric(),
        "pearson_r": PearsonRMetric(),
        "nmse": NMSEMetric(),
        "wasserstein_1_cm1": Wasserstein1Metric(),
        "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
    }


def _metric_state(state: str) -> str:
    if state == "complete":
        return "complete"
    if state == "not_applicable":
        return "metric_not_applicable"
    return "runtime_failure"


def _receipt_hash(value: object) -> str:
    return _sha_bytes(_canonical_json_bytes(value))


def _cwt_receipt_from_spectrum(cwt_system: Phase3System, spectrum: Spectrum1D) -> Mapping[str, object]:
    try:
        peak_receipt = run_peak_detection_system(cwt_system, spectrum)
    except Exception as error:
        return {
            "state": "runtime_failure",
            "diagnostics_sha256": _receipt_hash({"message": str(error), "type": type(error).__name__}),
            "peak_list_sha256": None,
            "warning_sha256": None,
        }
    return {
        "state": "complete",
        "diagnostics_sha256": _receipt_hash(getattr(peak_receipt, "diagnostics", None)),
        "peak_list_sha256": str(getattr(peak_receipt, "peaks_sha256", _receipt_hash(getattr(peak_receipt, "peaks", None)))),
        "warning_sha256": _receipt_hash(getattr(peak_receipt, "warnings", None)),
        "_peak_receipt": peak_receipt,
    }


def _scalar_metric_rows(
    *,
    source: Spectrum1D,
    output: Spectrum1D | None,
    state: str,
) -> Mapping[str, Mapping[str, str | None]]:
    rows: dict[str, Mapping[str, str | None]] = {}
    for metric_id, metric in _metric_objects().items():
        metric_state = _metric_state(state)
        result_sha256 = None
        diagnostics_sha256 = None
        if output is not None and metric_state == "complete":
            request = SingleSpectrumInput(output) if metric_id == "is_like_structure_to_noise" else SpectrumPairInput(source, output)
            try:
                result = evaluate_metric(metric, request)
                result_sha256 = _receipt_hash(getattr(result, "outputs", None))
                diagnostics_sha256 = _receipt_hash(getattr(result, "diagnostics", None))
            except Exception as error:
                metric_state = "runtime_failure"
                diagnostics_sha256 = _receipt_hash({"message": str(error), "type": type(error).__name__})
        rows[metric_id] = {
            "state": metric_state,
            "result_sha256": result_sha256,
            "diagnostics_sha256": diagnostics_sha256,
        }
    return rows


def _structure_metric_rows(
    *,
    source_cwt: Mapping[str, object] | None,
    output_cwt: Mapping[str, object] | None,
    state: str,
) -> Mapping[str, Mapping[str, str | None]]:
    rows: dict[str, Mapping[str, str | None]] = {}
    for metric_id in METRIC_OUTPUT_IDS[8:]:
        metric_state = _metric_state(state)
        result_sha256 = None
        diagnostics_sha256 = None
        if metric_state == "complete":
            if (
                source_cwt is None
                or output_cwt is None
                or str(source_cwt.get("state")) != "complete"
                or str(output_cwt.get("state")) != "complete"
            ):
                metric_state = "runtime_failure"
            else:
                try:
                    metric = PeakDetectionCurvesMetric()
                    result = evaluate_metric(
                        metric,
                        PeakPairInput(
                            reference_peaks=tuple(peak.to_peak1d() for peak in source_cwt["_peak_receipt"].peaks),
                            candidate_peaks=tuple(peak.to_peak1d() for peak in output_cwt["_peak_receipt"].peaks),
                            position_tolerance_cm1=2.0,
                            prominence_thresholds=(0.0,),
                        ),
                    )
                    result_sha256 = _receipt_hash(getattr(result, "outputs", None))
                    diagnostics_sha256 = _receipt_hash(getattr(result, "diagnostics", None))
                except Exception as error:
                    metric_state = "runtime_failure"
                    diagnostics_sha256 = _receipt_hash({"message": str(error), "type": type(error).__name__})
        rows[metric_id] = {
            "state": metric_state,
            "result_sha256": result_sha256,
            "diagnostics_sha256": diagnostics_sha256,
        }
    return rows


def evaluate_d4_eligibility_gates(
    *,
    cells: Sequence[Mapping[str, object]],
    mixture_record_ids: Sequence[str],
    mixture_well_ids: Sequence[str],
    config: Phase4D4EligibilityConfig,
    record_conditions: Sequence[Mapping[str, object]],
    blank_conditions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    per_perturbation_records: dict[str, set[str]] = {pid: set() for pid in ALL_PERTURBATION_IDS}
    per_perturbation_wells: dict[str, set[str]] = {pid: set() for pid in ALL_PERTURBATION_IDS}
    per_perturbation_states: dict[str, dict[str, int]] = {
        pid: {"complete": 0, "not_applicable": 0, "failed_runtime": 0, "structurally_ineligible": 0}
        for pid in ALL_PERTURBATION_IDS
    }
    for row in cells:
        perturbation_id = str(row["perturbation_id"])
        state = str(row["state"])
        record_id = str(row["record_id"])
        well_id = str(row["well_id"])
        per_perturbation_states[perturbation_id][state] += 1
        if state == "complete":
            per_perturbation_records[perturbation_id].add(record_id)

    expected_records_by_well: dict[str, set[str]] = {
        str(well_id): set() for well_id in mixture_well_ids
    }
    record_to_well: dict[str, str] = {}
    for row in cells:
        record_id = str(row["record_id"])
        well_id = str(row["well_id"])
        previous = record_to_well.setdefault(record_id, well_id)
        if previous != well_id:
            raise Phase4D4EligibilityVerifierError(
                "cells", f"record {record_id!r} maps to multiple wells"
            )
        if well_id in expected_records_by_well:
            expected_records_by_well[well_id].add(record_id)
    for perturbation_id in ALL_PERTURBATION_IDS:
        rows_by_well: dict[str, list[Mapping[str, object]]] = {
            str(well_id): [] for well_id in mixture_well_ids
        }
        for row in cells:
            if str(row["perturbation_id"]) == perturbation_id:
                well_id = str(row["well_id"])
                if well_id in rows_by_well:
                    rows_by_well[well_id].append(row)
        for well_id, expected_record_ids in expected_records_by_well.items():
            rows = rows_by_well[well_id]
            if (
                len(expected_record_ids) == config.records_per_well
                and len(rows) == config.records_per_well
                and {str(row["record_id"]) for row in rows} == expected_record_ids
                and all(str(row["state"]) == "complete" for row in rows)
            ):
                per_perturbation_wells[perturbation_id].add(well_id)

    full_domain = {}
    for perturbation_id in FULL_DOMAIN_PERTURBATION_IDS:
        state = "evaluable"
        if (
            len(per_perturbation_records[perturbation_id]) != len(mixture_record_ids)
            or len(per_perturbation_wells[perturbation_id]) != len(mixture_well_ids)
            or per_perturbation_states[perturbation_id]["not_applicable"] != 0
            or per_perturbation_states[perturbation_id]["failed_runtime"] != 0
        ):
            state = "not_evaluable_coverage"
        full_domain[perturbation_id] = {
            "complete_record_count": len(per_perturbation_records[perturbation_id]),
            "complete_well_count": len(per_perturbation_wells[perturbation_id]),
            "state": state,
            "state_counts": dict(per_perturbation_states[perturbation_id]),
        }

    peak_complete_records = []
    peak_complete_wells = []
    peak_summary = {}
    for perturbation_id in PEAK_PERTURBATION_IDS:
        record_threshold = config.gates["p05_record_fraction"] if perturbation_id == "p05" else config.gates["p01_p04_record_fraction"]
        well_threshold = config.gates["p05_well_fraction"] if perturbation_id == "p05" else config.gates["p01_p04_well_fraction"]
        complete_records = per_perturbation_records[perturbation_id]
        complete_wells = per_perturbation_wells[perturbation_id]
        peak_complete_records.append(complete_records)
        peak_complete_wells.append(complete_wells)
        minimum_records = math.ceil(len(mixture_record_ids) * record_threshold)
        minimum_wells = math.ceil(len(mixture_well_ids) * well_threshold)
        state = "evaluable"
        if (
            len(complete_records) < minimum_records
            or len(complete_wells) < minimum_wells
            or per_perturbation_states[perturbation_id]["failed_runtime"] != 0
        ):
            state = "not_evaluable_coverage"
        peak_summary[perturbation_id] = {
            "complete_record_count": len(complete_records),
            "complete_well_count": len(complete_wells),
            "state": state,
            "state_counts": dict(per_perturbation_states[perturbation_id]),
        }

    common_records = set.intersection(*peak_complete_records) if peak_complete_records else set()
    common_wells = set.intersection(*peak_complete_wells) if peak_complete_wells else set()
    peak_common_state = "evaluable"
    if (
        len(common_records) < math.ceil(len(mixture_record_ids) * config.gates["peak_common_record_fraction"])
        or len(common_wells) < math.ceil(len(mixture_well_ids) * config.gates["peak_common_well_fraction"])
    ):
        peak_common_state = "not_evaluable_coverage"

    def _grid_complete(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
        complete = sum(1 for row in rows if str(row.get("state")) == "complete")
        planned = len(rows)
        return {
            "planned": planned,
            "complete": complete,
            "failed": planned - complete,
            "state": "evaluable" if complete == planned else "not_evaluable_incomplete_grid",
        }

    execution_conditions = tuple(
        row for row in record_conditions
        if (
            str(row.get("condition_id")) == "alpha0"
            or str(row.get("condition_id")).split(":", 1)[0] in FULL_DOMAIN_PERTURBATION_IDS
        )
    )
    support_grid = _grid_complete(execution_conditions)
    metric_outputs = {
        metric_id: _grid_complete(
            tuple(
                metric_row
                for condition in execution_conditions
                for metric_row in (condition.get("metrics", {}) or {}).values()
                if metric_row is condition["metrics"][metric_id]
            )
        )
        for metric_id in METRIC_OUTPUT_IDS
    }
    cwt = _grid_complete(tuple(condition.get("cwt", {}) for condition in execution_conditions))
    blank_auxiliary = _grid_complete(blank_conditions)
    full_domain_state = "evaluable" if (
        all(value["state"] == "evaluable" for value in full_domain.values())
        and support_grid["state"] == "evaluable"
        and metric_outputs["mse"]["state"] == "evaluable"
        and cwt["state"] == "evaluable"
    ) else "not_evaluable_coverage"
    overall_status = "pass" if full_domain_state == "evaluable" and peak_common_state == "evaluable" else "fail"
    return {
        "blank_auxiliary": blank_auxiliary,
        "cwt": cwt,
        "full_domain_core": {"by_perturbation": full_domain, "state": full_domain_state},
        "marker_filename": "complete.json" if overall_status == "pass" else "failed.json",
        "metric_outputs": metric_outputs,
        "overall_status": overall_status,
        "peak_common_support": {
            "common_record_count": len(common_records),
            "common_well_count": len(common_wells),
            "p01": peak_summary["p01"],
            "p02": peak_summary["p02"],
            "p03": peak_summary["p03"],
            "p04": peak_summary["p04"],
            "p05": peak_summary["p05"],
            "state": peak_common_state,
        },
        "support_grid": support_grid,
    }


def validate_outcome_blind_d4_payload(value: object) -> None:
    def walk(path: str, current: object) -> None:
        if isinstance(current, Mapping):
            for key, item in current.items():
                name = str(key)
                if name in FORBIDDEN_PAYLOAD_KEYS or (
                    not name.endswith("_sha256") and any(fragment in name for fragment in FORBIDDEN_KEY_FRAGMENTS)
                ):
                    raise Phase4D4EligibilityVerifierError(path, name)
                walk(f"{path}.{name}", item)
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                walk(f"{path}[{index}]", item)

    walk("payload", value)


def _phase1_source_for_spectrum(spectrum: Spectrum1D, *, order: int) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    record_id = spectrum.spectrum_id.split("::", 1)[1]
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=order,
            record_id=record_id,
            sample_id=str(spectrum.sample_id or record_id),
            class_label=0,
            mineral_name="d4-sugar",
            axis_id=f"native::{_array_sha(spectrum.axis_cm1, dtype='<f8')}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1, dtype="<f8"),
        normalized_intensity_float64_sha256=_array_sha(spectrum.intensity, dtype="<f8"),
        provenance=MappingProxyType(
            {
                "license": None,
                "license_status": "not_stated",
                "retrieved_date": "2026-08-24",
                "sha256": "0" * 64,
                "source_artifact": "d4_sugar_low_snr",
                "source_url": "local://d4_sugar_low_snr",
            }
        ),
    )


def _exception_payload(cell: Phase1Cell) -> Mapping[str, object] | None:
    evidence = cell.evidence
    if evidence.status is CellStatus.COMPLETE:
        return None
    return {
        "message": evidence.exception_message,
        "path": evidence.exception_path,
        "type": evidence.exception_type,
    }


def _classify_cell(cell: Phase1Cell) -> str:
    if cell.status is CellStatus.COMPLETE:
        return "complete"
    if cell.status is CellStatus.NOT_APPLICABLE:
        return "not_applicable"
    return "failed_runtime"


def _condition_row(
    *,
    record_id: str,
    well_id: str,
    condition_id: str,
    state: str,
    metrics: Mapping[str, Mapping[str, str | None]],
    cwt: Mapping[str, str | None],
) -> Mapping[str, object]:
    row = {
        "record_id": record_id,
        "well_id": well_id,
        "condition_id": condition_id,
        "state": state,
        "metrics": metrics,
        "cwt": cwt,
    }
    validate_outcome_blind_d4_payload(row)
    return row


def _blank_condition_row(*, record_id: str, well_id: str, condition_id: str, state: str) -> Mapping[str, object]:
    row = {
        "record_id": record_id,
        "well_id": well_id,
        "condition_id": condition_id,
        "state": state,
    }
    validate_outcome_blind_d4_payload(row)
    return row


def _execute_d4_record(
    *,
    record_order: int,
    spectrum: Spectrum1D,
    sweep,
    phase1_config: Phase1CoreConfig,
    config: Phase4D4EligibilityConfig,
    cwt_system: Phase3System,
) -> Mapping[str, object]:
    source = _phase1_source_for_spectrum(spectrum, order=record_order)
    record_id = spectrum.spectrum_id.split("::", 1)[1]
    well_id = str(spectrum.sample_id)
    cells: list[Mapping[str, object]] = []
    support_axis = _support_axis(config)
    alpha0_projected = project_d4_support(spectrum, config)
    alpha0_output = Spectrum1D(
        spectrum_id=spectrum.spectrum_id + "::alpha0_support",
        sample_id=spectrum.sample_id,
        axis_cm1=np.asarray(support_axis, dtype="<f8"),
        intensity=np.asarray(alpha0_projected, dtype="<f8"),
    )
    alpha0_cwt = _cwt_receipt_from_spectrum(cwt_system, spectrum)
    record_conditions: list[Mapping[str, object]] = [
        _condition_row(
            record_id=record_id,
            well_id=well_id,
            condition_id="alpha0",
            state="complete",
            metrics={
                **_scalar_metric_rows(source=spectrum, output=spectrum, state="complete"),
                **_structure_metric_rows(source_cwt=alpha0_cwt, output_cwt=alpha0_cwt, state="complete"),
            },
            cwt={
                "state": str(alpha0_cwt["state"]),
                "diagnostics_sha256": alpha0_cwt["diagnostics_sha256"],
                "peak_list_sha256": alpha0_cwt["peak_list_sha256"],
                "warning_sha256": alpha0_cwt["warning_sha256"],
            },
        )
    ]
    for perturbation_id in ALL_PERTURBATION_IDS:
        if perturbation_id in INACTIVE_PERTURBATION_IDS:
            row = {
                "record_id": record_id,
                "well_id": well_id,
                "perturbation_id": perturbation_id,
                "reason_code": STRUCTURAL_REASON,
                "state": "structurally_ineligible",
            }
            validate_outcome_blind_d4_payload(row)
            cells.append(row)
            continue
        with threadpool_limits(limits=1, user_api="blas"):
            cell = run_perturbation_cell(
                source,
                perturbation_id,
                phase1_config,
                sweep,
                p10_admission=P10MemoryAdmission(config.p10_memory_budget_bytes) if perturbation_id == "p10" else None,
            )
        state = _classify_cell(cell)
        row = {
            "exception": _exception_payload(cell),
            "native_gate": dict(cell.evidence.native_gate),
            "p10_estimated_peak_bytes": estimate_p10_peak_bytes(spectrum.axis_cm1.size) if perturbation_id == "p10" else None,
            "perturbation_id": perturbation_id,
            "reason_code": cell.reason_code,
            "record_id": record_id,
            "state": state,
            "well_id": well_id,
        }
        validate_outcome_blind_d4_payload(row)
        cells.append(row)
        if perturbation_id in FULL_DOMAIN_PERTURBATION_IDS:
            records_by_alpha = {float(record.alpha): record for record in cell.records}
            for alpha in POSITIVE_ALPHAS:
                cwt_row = {"state": _metric_state(state), "diagnostics_sha256": None, "peak_list_sha256": None, "warning_sha256": None}
                metrics = {
                    metric_id: {"state": _metric_state(state), "result_sha256": None, "diagnostics_sha256": None}
                    for metric_id in METRIC_OUTPUT_IDS
                }
                perturbed = records_by_alpha.get(float(alpha))
                if state == "complete" and perturbed is not None:
                    native_output = perturbed.result.output
                    projected = project_d4_support(perturbed.result.output, config)
                    output_cwt = _cwt_receipt_from_spectrum(cwt_system, native_output)
                    metrics = {
                        **_scalar_metric_rows(source=spectrum, output=native_output, state=state),
                        **_structure_metric_rows(source_cwt=alpha0_cwt, output_cwt=output_cwt, state=state),
                    }
                    cwt_row = {
                        "state": str(output_cwt["state"]),
                        "diagnostics_sha256": output_cwt["diagnostics_sha256"],
                        "peak_list_sha256": _receipt_hash(
                            {
                                "cwt_peak_list_sha256": output_cwt["peak_list_sha256"],
                                "support_intensity_sha256": _sha_bytes(np.asarray(projected, dtype="<f4").tobytes(order="C")),
                            }
                        ),
                        "warning_sha256": output_cwt["warning_sha256"],
                    }
                record_conditions.append(
                    _condition_row(
                        record_id=record_id,
                        well_id=well_id,
                        condition_id=f"{perturbation_id}:{_hex_alpha(alpha)}",
                        state=state,
                        metrics=metrics,
                        cwt=cwt_row,
                    )
                )
    return {
        "cells": tuple(cells),
        "record_conditions": tuple(record_conditions),
        "record_id": record_id,
        "record_order": record_order,
        "well_id": well_id,
        "_worker_pid": os.getpid(),
    }


_PROCESS_SWEEP = None
_PROCESS_PHASE1_CONFIG: Phase1CoreConfig | None = None
_PROCESS_D4_CONFIG: Phase4D4EligibilityConfig | None = None
_PROCESS_CWT_SYSTEM: Phase3System | None = None


def _bounded_process_count(*, requested_workers: int, point_counts: Sequence[int], memory_budget_bytes: int) -> int:
    max_p10_estimate = max(estimate_p10_peak_bytes(int(value)) for value in point_counts)
    capacity = int(memory_budget_bytes) // max_p10_estimate
    if capacity < 1:
        raise Phase4D4EligibilityVerifierError("p10.memory_budget_bytes", "cannot admit the largest source record")
    return min(int(requested_workers), capacity)


def _initialize_d4_process(sweep_path: str, phase1_config_path: str, config_path: str, config_raw: bytes) -> None:
    global _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_D4_CONFIG, _PROCESS_CWT_SYSTEM
    _PROCESS_SWEEP = load_perturbation_sweep_config(Path(sweep_path))
    _PROCESS_PHASE1_CONFIG = load_phase1_core_config(Path(phase1_config_path))
    _PROCESS_D4_CONFIG = parse_phase4_d4_eligibility_config(Path(config_path), config_raw, require_frozen_identity=False)
    catalog = load_classical_catalog(ROOT / CATALOG_RELATIVE_PATH)
    matches = [system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D4EligibilityVerifierError("CWT system", "must resolve exactly once")
    _PROCESS_CWT_SYSTEM = matches[0]


def _execute_initialized_d4_record(record_order: int, spectrum: Spectrum1D) -> Mapping[str, object]:
    if _PROCESS_SWEEP is None or _PROCESS_PHASE1_CONFIG is None or _PROCESS_D4_CONFIG is None or _PROCESS_CWT_SYSTEM is None:
        raise Phase4D4EligibilityVerifierError("process worker", "was not initialized")
    return _execute_d4_record(
        record_order=record_order,
        spectrum=spectrum,
        sweep=_PROCESS_SWEEP,
        phase1_config=_PROCESS_PHASE1_CONFIG,
        config=_PROCESS_D4_CONFIG,
        cwt_system=_PROCESS_CWT_SYSTEM,
    )


def _run_d4_jobs(
    jobs: Sequence[tuple[int, Spectrum1D]],
    *,
    sweep_path: Path,
    phase1_config_path: Path,
    config: Phase4D4EligibilityConfig,
    worker_count: int,
) -> tuple[Mapping[str, object], ...]:
    process_count = min(
        len(jobs),
        _bounded_process_count(
            requested_workers=worker_count,
            point_counts=tuple(spectrum.axis_cm1.size for _, spectrum in jobs),
            memory_budget_bytes=config.p10_memory_budget_bytes,
        ),
    )
    if process_count <= 1:
        sweep = load_perturbation_sweep_config(sweep_path)
        phase1_config = load_phase1_core_config(phase1_config_path)
        catalog = load_classical_catalog(ROOT / CATALOG_RELATIVE_PATH)
        matches = [system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID]
        if len(matches) != 1:
            raise Phase4D4EligibilityVerifierError("CWT system", "must resolve exactly once")
        cwt_system = matches[0]
        return tuple(
            _execute_d4_record(
                record_order=record_order,
                spectrum=spectrum,
                sweep=sweep,
                phase1_config=phase1_config,
                config=config,
                cwt_system=cwt_system,
            )
            for record_order, spectrum in jobs
        )
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_d4_process,
        initargs=(str(sweep_path), str(phase1_config_path), str(config.path), config.raw_bytes),
    ) as executor:
        futures = [executor.submit(_execute_initialized_d4_record, record_order, spectrum) for record_order, spectrum in jobs]
        return tuple(sorted((future.result() for future in futures), key=lambda row: int(row["record_order"])))


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _build_run_id(config: Phase4D4EligibilityConfig, inputs: D4EligibilityInputs) -> str:
    payload = _canonical_json_bytes(
        {
            "config_sha256": config.sha256,
            "mixture_record_ids_sha256": inputs.mixture_record_ids_sha256,
            "support_axis_sha256": _array_sha(inputs.support_axis_cm1, dtype="<f8"),
            "well_ids_sha256": inputs.well_ids_sha256,
        }
    )
    return RUN_PREFIX + _sha_bytes(payload)


def build_phase4_d4_eligibility_from_inputs(
    output_dir: Path,
    *,
    inputs: D4EligibilityInputs,
    sweep_path: Path,
    phase1_config_path: Path,
    config: Phase4D4EligibilityConfig,
    worker_count: int,
) -> Phase4D4EligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise Phase4D4EligibilityVerifierError("worker_count", "must be a positive integer")
    output_path = Path(output_dir)
    if output_path.exists():
        raise Phase4D4EligibilityVerifierError(str(output_path), "append-only output target already exists")
    output_path.mkdir(parents=True)
    max_p10_estimate = max(
        estimate_p10_peak_bytes(int(spectrum.axis_cm1.size))
        for spectrum in inputs.mixture_spectra + inputs.blank_spectra
    )
    capacity = config.p10_memory_budget_bytes // max_p10_estimate
    run_id = _build_run_id(config, inputs)
    sweep = load_perturbation_sweep_config(Path(sweep_path))
    phase1_config = load_phase1_core_config(Path(phase1_config_path))

    source_records = tuple(sorted(inputs.source_records, key=lambda row: (str(row["scope"]), str(row["record_id"]))))
    well_folds = tuple(sorted(inputs.well_folds, key=lambda row: int(row["fold_index"])))
    model_cells = tuple(sorted(inputs.model_cells, key=lambda row: int(row["seed"])))
    model_role_occurrences = tuple(
        sorted(inputs.model_role_occurrences, key=lambda row: (int(row["seed"]), str(row["role"]), str(row["record_id"])))
    )

    mixture_jobs = tuple(enumerate(inputs.mixture_spectra))
    results = _run_d4_jobs(
        mixture_jobs,
        sweep_path=Path(sweep_path),
        phase1_config_path=Path(phase1_config_path),
        config=config,
        worker_count=worker_count,
    )

    operator_cells: list[Mapping[str, object]] = []
    record_conditions: list[Mapping[str, object]] = []
    for result in results:
        operator_cells.extend(result["cells"])
        record_conditions.extend(result["record_conditions"])

    blank_cells: list[Mapping[str, object]] = []
    blank_conditions: list[Mapping[str, object]] = []
    catalog = load_classical_catalog(ROOT / CATALOG_RELATIVE_PATH)
    matches = [system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D4EligibilityVerifierError("CWT system", "must resolve exactly once")
    cwt_system = matches[0]
    for spectrum in inputs.blank_spectra:
        record_id = spectrum.spectrum_id.split("::", 1)[1]
        well_id = str(spectrum.sample_id)
        alpha0_projected = project_d4_support(spectrum, config)
        alpha0_output = Spectrum1D(
            spectrum_id=spectrum.spectrum_id + "::alpha0_support",
            sample_id=spectrum.sample_id,
            axis_cm1=_support_axis(config),
            intensity=np.asarray(alpha0_projected, dtype="<f8"),
        )
        alpha0_cwt = _cwt_receipt_from_spectrum(cwt_system, spectrum)
        alpha0_state = (
            "complete"
            if str(alpha0_cwt.get("state")) == "complete"
            else "runtime_failure"
        )
        blank_conditions.append(
            {
                **_blank_condition_row(
                    record_id=record_id,
                    well_id=well_id,
                    condition_id="alpha0",
                    state=alpha0_state,
                ),
                "diagnostics_sha256": alpha0_cwt["diagnostics_sha256"],
                "result_sha256": _receipt_hash(
                    {
                        "cwt_peak_list_sha256": alpha0_cwt["peak_list_sha256"],
                        "support_intensity_sha256": _sha_bytes(
                            np.asarray(alpha0_projected, dtype="<f4").tobytes(order="C")
                        ),
                    }
                ),
                "warning_sha256": alpha0_cwt["warning_sha256"],
            }
        )
        for perturbation_id in FULL_DOMAIN_PERTURBATION_IDS:
            with threadpool_limits(limits=1, user_api="blas"):
                cell = run_perturbation_cell(
                    _phase1_source_for_spectrum(spectrum, order=0),
                    perturbation_id,
                    phase1_config,
                    sweep,
                    p10_admission=P10MemoryAdmission(config.p10_memory_budget_bytes) if perturbation_id == "p10" else None,
                )
            state = _classify_cell(cell)
            row = {"record_id": record_id, "well_id": well_id, "perturbation_id": perturbation_id, "state": state}
            validate_outcome_blind_d4_payload(row)
            blank_cells.append(row)
            for alpha in POSITIVE_ALPHAS:
                cwt_row = None
                if state == "complete":
                    records_by_alpha = {float(record.alpha): record for record in cell.records}
                    perturbed = records_by_alpha.get(float(alpha))
                    if perturbed is not None:
                        projected = project_d4_support(perturbed.result.output, config)
                        native_cwt = _cwt_receipt_from_spectrum(cwt_system, perturbed.result.output)
                        cwt_row = {
                            "state": str(native_cwt["state"]),
                            "diagnostics_sha256": native_cwt["diagnostics_sha256"],
                            "peak_list_sha256": _receipt_hash(
                                {
                                    "cwt_peak_list_sha256": native_cwt["peak_list_sha256"],
                                    "support_intensity_sha256": _sha_bytes(np.asarray(projected, dtype="<f4").tobytes(order="C")),
                                }
                            ),
                            "warning_sha256": native_cwt["warning_sha256"],
                        }
                blank_state = state
                if cwt_row is not None and str(cwt_row.get("state")) != "complete":
                    blank_state = "runtime_failure"
                blank_conditions.append(
                    {
                        **_blank_condition_row(
                            record_id=record_id,
                            well_id=well_id,
                            condition_id=f"{perturbation_id}:{_hex_alpha(alpha)}",
                            state=blank_state,
                        ),
                        "diagnostics_sha256": None if cwt_row is None else cwt_row["diagnostics_sha256"],
                        "result_sha256": None if cwt_row is None else cwt_row["peak_list_sha256"],
                        "warning_sha256": None if cwt_row is None else cwt_row["warning_sha256"],
                    }
                )

    expected_well_records = {
        well_id: {str(row["record_id"]) for row in operator_cells if str(row["well_id"]) == well_id}
        for well_id in inputs.well_ids
    }
    peak_complete_by_well = {}
    for perturbation_id in PEAK_PERTURBATION_IDS:
        complete_wells = set()
        for well_id, expected_record_ids in expected_well_records.items():
            matching = [
                row
                for row in operator_cells
                if row["perturbation_id"] == perturbation_id
                and str(row["well_id"]) == well_id
            ]
            if (
                len(expected_record_ids) == config.records_per_well
                and len(matching) == config.records_per_well
                and {str(row["record_id"]) for row in matching} == expected_record_ids
                and all(str(row["state"]) == "complete" for row in matching)
            ):
                complete_wells.add(well_id)
        peak_complete_by_well[perturbation_id] = complete_wells
    common_wells = set.intersection(*peak_complete_by_well.values()) if peak_complete_by_well else set()
    well_summaries: list[Mapping[str, object]] = []
    for perturbation_id in ALL_PERTURBATION_IDS:
        for well_id in inputs.well_ids:
            matching = [row for row in operator_cells if row["perturbation_id"] == perturbation_id and row["well_id"] == well_id]
            complete_ids = {str(row["record_id"]) for row in matching if str(row["state"]) == "complete"}
            complete = (
                len(matching) == len(expected_well_records[well_id])
                and complete_ids == expected_well_records[well_id]
            )
            row = {
                "perturbation_id": perturbation_id,
                "state": "complete" if complete else "not_evaluable_coverage",
                "well_id": well_id,
            }
            validate_outcome_blind_d4_payload(row)
            well_summaries.append(row)

    common_support: list[Mapping[str, object]] = []
    record_to_well = {spectrum.spectrum_id.split("::", 1)[1]: str(spectrum.sample_id) for spectrum in inputs.mixture_spectra}
    for record_id in inputs.mixture_record_ids:
        row = {"complete": record_to_well[record_id] in common_wells, "record_id": record_id, "scope": "record", "well_id": record_to_well[record_id]}
        validate_outcome_blind_d4_payload(row)
        common_support.append(row)
    for well_id in inputs.well_ids:
        row = {"complete": well_id in common_wells, "scope": "whole_well", "well_id": well_id}
        validate_outcome_blind_d4_payload(row)
        common_support.append(row)

    gate = evaluate_d4_eligibility_gates(
        cells=operator_cells,
        mixture_record_ids=inputs.mixture_record_ids,
        mixture_well_ids=inputs.well_ids,
        config=config,
        record_conditions=record_conditions,
        blank_conditions=blank_conditions,
    )

    files: dict[str, bytes] = {
        "config.json": config.raw_bytes,
        "source_records.jsonl": _jsonl_bytes(source_records),
        "well_folds.jsonl": _jsonl_bytes(well_folds),
        "model_cells.jsonl": _jsonl_bytes(model_cells),
        "model_role_occurrences.jsonl": _jsonl_bytes(model_role_occurrences),
        "operator_cells.jsonl": _jsonl_bytes(operator_cells),
        "blank_cells.jsonl": _jsonl_bytes(blank_cells),
        "record_conditions.jsonl": _jsonl_bytes(record_conditions),
        "blank_conditions.jsonl": _jsonl_bytes(blank_conditions),
        "well_summaries.jsonl": _jsonl_bytes(well_summaries),
        "common_support.jsonl": _jsonl_bytes(common_support),
        "gate.json": _canonical_json_bytes(gate),
    }
    manifest = {
        "artifact_schema_version": "phase4-d4-protocol-a-full-domain-eligibility-artifact-v1",
        "files": {name: {"bytes": len(payload), "sha256": _sha_bytes(payload)} for name, payload in files.items()},
        "resource_policy": {
            "blas_threads": 1,
            "capacity": capacity,
            "correlation_length_cm1": 20.0,
            "memory_budget_bytes": config.p10_memory_budget_bytes,
            "p10_estimate_bytes": 1201869824,
            "p10_estimate_formula": "32*N^2+64*N+2^30",
            "point_count": 2000,
        },
        "run_id": run_id,
        "status": gate["overall_status"],
    }
    files["manifest.json"] = _canonical_json_bytes(manifest)
    marker_name = str(gate["marker_filename"])
    files[marker_name] = _canonical_json_bytes(
        {
            "run_id": run_id,
            "schema_version": "phase4-d4-protocol-a-full-domain-eligibility-marker-v1",
            "status": gate["overall_status"],
        }
    )
    sha_lines = [f"{_sha_bytes(files[name])}  {name}\n" for name in ARTIFACT_STATIC_FILES if name != "SHA256SUMS"]
    sha_lines.append(f"{_sha_bytes(files[marker_name])}  {marker_name}\n")
    files["SHA256SUMS"] = "".join(sha_lines).encode("utf-8")
    for name, payload in files.items():
        (output_path / name).write_bytes(payload)
    return Phase4D4EligibilitySummary(
        path=output_path,
        run_id=run_id,
        status=str(gate["overall_status"]),
        marker_filename=marker_name,
        mixture_record_count=config.mixture_record_count,
        model_cell_count=config.model_cell_count,
        model_role_occurrence_count=len(model_role_occurrences),
    )


def _compare_tree(candidate_path: Path, rebuilt_path: Path) -> None:
    candidate_files = sorted(path.name for path in candidate_path.iterdir() if path.is_file())
    rebuilt_files = sorted(path.name for path in rebuilt_path.iterdir() if path.is_file())
    if candidate_files != rebuilt_files:
        raise Phase4D4EligibilityVerifierError("artifact inventory", "candidate and rebuilt file sets differ")
    for name in candidate_files:
        if (candidate_path / name).read_bytes() != (rebuilt_path / name).read_bytes():
            raise Phase4D4EligibilityVerifierError(name, "candidate and rebuilt bytes differ")


def verify_phase4_d4_eligibility_from_inputs(
    path: Path,
    *,
    inputs: D4EligibilityInputs,
    config: Phase4D4EligibilityConfig,
    sweep_path: Path,
    phase1_config_path: Path,
    worker_count: int,
) -> Phase4D4EligibilitySummary:
    with tempfile.TemporaryDirectory() as temporary_directory:
        rebuilt_path = Path(temporary_directory) / "rebuilt"
        summary = build_phase4_d4_eligibility_from_inputs(
            rebuilt_path,
            inputs=inputs,
            sweep_path=sweep_path,
            phase1_config_path=phase1_config_path,
            config=config,
            worker_count=worker_count,
        )
        _compare_tree(Path(path), rebuilt_path)
        return Phase4D4EligibilitySummary(
            path=Path(path),
            run_id=summary.run_id,
            status=summary.status,
            marker_filename=summary.marker_filename,
            mixture_record_count=summary.mixture_record_count,
            model_cell_count=summary.model_cell_count,
            model_role_occurrence_count=summary.model_role_occurrence_count,
        )


def verify_phase4_d4_eligibility(path: Path, *, worker_count: int = 12) -> Phase4D4EligibilitySummary:
    config = load_phase4_d4_eligibility_config(ROOT / CONFIG_RELATIVE_PATH)
    cohort = load_d4_sugar_cohort(ROOT / PHASE05_D4_PROTOCOL_RELATIVE_PATH, ROOT / ARCHIVE_RELATIVE_PATH)
    inputs = reconstruct_d4_eligibility_inputs(cohort, config)
    return verify_phase4_d4_eligibility_from_inputs(
        path,
        inputs=inputs,
        config=config,
        sweep_path=ROOT / SWEEP_RELATIVE_PATH,
        phase1_config_path=ROOT / PHASE1_CONFIG_RELATIVE_PATH,
        worker_count=worker_count,
    )
