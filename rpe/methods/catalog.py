from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


SCHEMA_VERSION = "phase3-classical-system-catalog-v1"
GLOBAL_SEED = 20260819
CATALOG_BYTE_COUNT = 376665
CATALOG_SHA256 = "8ad40b08df78b8905d75a67c84a2bb328531ef17cb12704ffad04f0a8f925d8f"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HEX = frozenset("0123456789abcdef")
_SYSTEM_DOMAIN = b"rpe-phase3-system-v1\0"
_CATALOG_DOMAIN = b"rpe-phase3-catalog-v1\0"
_ROOT_KEYS = (
    "availability_audit",
    "catalog_id",
    "dependency_lock",
    "evidence",
    "global_seed",
    "identity_domains",
    "phase2_fallback_binding",
    "schema_version",
    "systems",
    "views",
)
_SYSTEM_KEYS = (
    "availability",
    "backend",
    "downstream_eligibility",
    "evidence",
    "family_id",
    "hyperparameters",
    "input_contract",
    "method_id",
    "method_seed",
    "metric_eligibility",
    "ordered_composition",
    "output_contract",
    "protocol_eligibility",
    "system_id",
    "task_line",
)
_VIEW_KEYS = (
    "family_count",
    "phase5_power_status",
    "system_count",
    "system_ids",
    "systems_per_family",
    "task_line",
)


class Phase3CatalogError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class TaskLine(str, Enum):
    BASELINE_CORRECTION = "baseline_correction"
    DENOISING = "denoising"
    PEAK_DETECTION = "peak_detection"


class AvailabilityStatus(str, Enum):
    RUNNABLE = "runnable"
    DEPENDENCY_MISSING = "dependency_missing"
    IMPLEMENTATION_MISSING = "implementation_missing"
    CODE_MISSING = "code_missing"
    WEIGHTS_MISSING = "weights_missing"
    DATA_MISSING = "data_missing"
    REPORTED_ONLY = "reported_only"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class Phase3System:
    system_id: str
    task_line: TaskLine
    family_id: str
    method_id: str
    backend: Mapping[str, object]
    hyperparameters: Mapping[str, object]
    input_contract: Mapping[str, object]
    output_contract: Mapping[str, object]
    ordered_composition: tuple[str, ...]
    method_seed: None
    metric_eligibility: tuple[str, ...]
    downstream_eligibility: tuple[str, ...]
    protocol_eligibility: tuple[str, ...]
    availability: AvailabilityStatus
    evidence: Mapping[str, object]


@dataclass(frozen=True)
class Phase3View:
    task_line: TaskLine
    system_count: int
    family_count: int
    systems_per_family: int
    phase5_power_status: str
    system_ids: tuple[str, ...]


@dataclass(frozen=True)
class Phase3ClassicalCatalog:
    path: Path
    byte_count: int
    sha256: str
    catalog_id: str
    global_seed: int
    dependency_lock: ArtifactIdentity
    evidence: Mapping[str, ArtifactIdentity]
    views: tuple[Phase3View, ...]
    systems: tuple[Phase3System, ...]
    availability_audit: Mapping[str, object]
    phase2_fallback_binding: Mapping[str, object]
    document: Mapping[str, object]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_catalog_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value)


def _reject_nonfinite(token: str) -> object:
    raise Phase3CatalogError("catalog", f"non-finite constant {token}")


def _nonempty(path: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise Phase3CatalogError(path, "must be a nonempty string")
    return value


def _lower_hex(path: str, value: object) -> str:
    parsed = _nonempty(path, value)
    if len(parsed) != 64 or any(character not in _HEX for character in parsed):
        raise Phase3CatalogError(path, "must be lowercase SHA256")
    return parsed


def _positive_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase3CatalogError(path, "must be a positive integer")
    return value


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase3CatalogError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise Phase3CatalogError(path, "keys must be strings")
    return value


def _exact_keys(path: str, value: object, expected: Sequence[str]) -> Mapping[str, object]:
    parsed = _object(path, value)
    if set(parsed) != set(expected):
        raise Phase3CatalogError(path, "has an unexpected key set")
    return parsed


def _string_tuple(path: str, value: object, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase3CatalogError(path, "must be an array")
    parsed = tuple(_nonempty(f"{path}[{index}]", item) for index, item in enumerate(value))
    if nonempty and not parsed:
        raise Phase3CatalogError(path, "must not be empty")
    if len(parsed) != len(set(parsed)):
        raise Phase3CatalogError(path, "must not contain duplicates")
    return parsed


def _freeze(path: str, value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase3CatalogError(path, "must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(f"{path}[{index}]", item) for index, item in enumerate(value))
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _nonempty(f"{path}.key", key): _freeze(f"{path}.{key}", item)
                for key, item in sorted(value.items())
            }
        )
    raise Phase3CatalogError(path, "must be JSON-compatible")


def _file_identity(root: Path, relative_path: str) -> dict[str, object]:
    path = root / relative_path
    raw = path.read_bytes()
    return {
        "path": relative_path,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _common_input_contract(coordinate_mode: str) -> dict[str, object]:
    return {
        "axis": "strictly_increasing_native_cm1",
        "coordinate_mode": coordinate_mode,
        "dimension": 1,
        "dtype": "little_endian_float64",
        "finite": True,
        "implicit_processing": "forbidden",
    }


def _output_contract(task_line: TaskLine) -> dict[str, object]:
    if task_line is TaskLine.BASELINE_CORRECTION:
        return {
            "axis_relation": "same_axis",
            "outputs": ["baseline_estimate", "corrected_intensity"],
            "corrected_formula": "input_minus_baseline_no_clipping",
        }
    if task_line is TaskLine.DENOISING:
        return {
            "axis_relation": "same_axis",
            "outputs": ["denoised_intensity"],
        }
    return {
        "axis_relation": "same_axis",
        "outputs": [
            "peak_index",
            "position_cm1",
            "height",
            "prominence",
            "fwhm_cm1",
            "area",
        ],
        "order": "position_ascending",
    }


def _metric_eligibility(task_line: TaskLine) -> list[str]:
    if task_line is TaskLine.BASELINE_CORRECTION:
        return ["downstream", "half_split_consistency", "reference_free"]
    if task_line is TaskLine.DENOISING:
        return ["direct_fidelity_when_target_eligible", "downstream", "snr"]
    return ["artifact_peak_ratio", "downstream", "missing_peak_ratio", "peak_gt_when_assignment_eligible"]


def _system_document(
    task_line: TaskLine,
    family_id: str,
    method_id: str,
    package: str,
    callable_name: str,
    hyperparameters: Mapping[str, object],
    *,
    coordinate_mode: str,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    backend = {"callable": callable_name, "package": package}
    input_contract = _common_input_contract(coordinate_mode)
    output_contract = _output_contract(task_line)
    identity = {
        "backend": backend,
        "family_id": family_id,
        "hyperparameters": dict(hyperparameters),
        "input_contract": input_contract,
        "method_id": method_id,
        "method_seed": None,
        "ordered_composition": [method_id],
        "output_contract": output_contract,
        "task_line": task_line.value,
    }
    system_id = hashlib.sha256(_SYSTEM_DOMAIN + _canonical_json_bytes(identity)).hexdigest()
    system_evidence: dict[str, object] = {
        "catalog_role": "planned_atomic_system",
        "wrapper": "implementation_missing",
    }
    if task_line is TaskLine.BASELINE_CORRECTION:
        system_evidence["phase2_fallback"] = "downstream_reference_free_half_split_only"
    if evidence:
        system_evidence.update(evidence)
    if task_line is TaskLine.PEAK_DETECTION:
        downstream_eligibility = ["peak_assignment_benchmark"]
        protocol_eligibility = ["main_table", "phase5"]
    else:
        downstream_eligibility = ["D1", "D2", "D3", "D4", "D5"]
        protocol_eligibility = ["main_table", "phase5", "protocol_a", "protocol_b"]
    return {
        "availability": AvailabilityStatus.IMPLEMENTATION_MISSING.value,
        **identity,
        "downstream_eligibility": downstream_eligibility,
        "evidence": system_evidence,
        "metric_eligibility": _metric_eligibility(task_line),
        "protocol_eligibility": protocol_eligibility,
        "system_id": system_id,
    }


def _baseline_systems() -> list[dict[str, object]]:
    systems: list[dict[str, object]] = []
    l5 = [1e3, 1e4, 1e5, 1e6, 1e7]
    l15 = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]

    def add(family: str, method: str, configs: Sequence[Mapping[str, object]]) -> None:
        for config in configs:
            systems.append(
                _system_document(
                    TaskLine.BASELINE_CORRECTION,
                    family,
                    method,
                    "pybaselines==1.2.1",
                    f"pybaselines.Baseline.{method}",
                    config,
                    coordinate_mode=("physical_axis" if family not in {"snip", "morphological"} else "physical_window_converted_by_median_spacing"),
                    evidence={"dependency": "available"},
                )
            )

    add("asls", "asls", [
        {"diff_order": 2, "lam": lam, "max_iter": 50, "p": p, "tol": 0.001, "weights": None}
        for lam in l5 for p in (0.001, 0.01, 0.1)
    ])
    add("iasls", "iasls", [
        {"diff_order": 2, "lam": lam, "lam_1": 0.0001, "max_iter": 50, "p": p, "tol": 0.001, "weights": None}
        for lam in l5 for p in (0.001, 0.01, 0.1)
    ])
    add("airpls", "airpls", [
        {"diff_order": 2, "lam": lam, "max_iter": 50, "normalize_weights": False, "tol": 0.001, "weights": None}
        for lam in l15
    ])
    add("arpls", "arpls", [
        {"diff_order": 2, "lam": lam, "max_iter": 50, "tol": 0.001, "weights": None}
        for lam in l15
    ])
    add("drpls", "drpls", [
        {"diff_order": 2, "eta": eta, "lam": lam, "max_iter": 50, "tol": 0.001, "weights": None}
        for lam in l5 for eta in (0.1, 0.5, 0.9)
    ])
    add("iarpls", "iarpls", [
        {"diff_order": 2, "lam": lam, "max_iter": 50, "tol": 0.001, "weights": None}
        for lam in l15
    ])
    add("aspls", "aspls", [
        {"alpha": None, "asymmetric_coef": coefficient, "diff_order": 2, "lam": lam, "max_iter": 100, "tol": 0.001, "weights": None}
        for lam in l5 for coefficient in (0.1, 0.5, 0.9)
    ])
    add("psalsa", "psalsa", [
        {"diff_order": 2, "k": None, "lam": lam, "max_iter": 50, "p": p, "tol": 0.001, "weights": None}
        for lam in l5 for p in (0.01, 0.1, 0.5)
    ])
    variants = ((False, False), (False, True), (True, False))
    add("modpoly", "modpoly", [
        {"mask_initial_peaks": mask, "max_iter": 250, "poly_order": order, "return_coef": False, "tol": 0.001, "use_original": original, "weights": None}
        for order in (2, 3, 4, 5, 6) for original, mask in variants
    ])
    add("imodpoly", "imodpoly", [
        {"mask_initial_peaks": True, "max_iter": 250, "num_std": num_std, "poly_order": order, "return_coef": False, "tol": 0.001, "use_original": False, "weights": None}
        for order in (2, 3, 4, 5, 6) for num_std in (0.5, 1.0, 2.0)
    ])
    add("penalized_poly", "penalized_poly", [
        {"alpha_factor": 0.99, "cost_function": cost, "max_iter": 250, "poly_order": order, "return_coef": False, "threshold": None, "tol": 0.001, "weights": None}
        for order in (2, 3, 4, 5, 6)
        for cost in ("asymmetric_truncated_quadratic", "asymmetric_huber", "asymmetric_indec")
    ])
    add("snip", "snip", [
        {"decreasing": False, "filter_order": filter_order, "max_half_window_cm1": width, "pad_kwargs": None, "smooth_half_window": None}
        for width in (10.0, 20.0, 40.0, 80.0, 160.0) for filter_order in (2, 4, 6)
    ])
    add("morphological", "mor", [
        {"half_window_cm1": width, "window_kwargs": None}
        for width in (8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 80.0, 96.0, 128.0, 160.0, 192.0, 256.0, 320.0, 400.0)
    ])
    add("beads", "beads", [
        {"asymmetry": asymmetry, "cost_function": 2, "eps_0": 1e-6, "eps_1": 1e-6, "filter_type": 1, "fit_parabola": True, "freq_cutoff": cutoff, "lam_0": 1.0, "lam_1": 1.0, "lam_2": 1.0, "max_iter": 50, "smooth_half_window": None, "tol": 0.01}
        for cutoff in (0.0025, 0.005, 0.01, 0.02, 0.04) for asymmetry in (2.0, 6.0, 10.0)
    ])
    return systems


def _denoising_systems() -> list[dict[str, object]]:
    systems: list[dict[str, object]] = []

    def add(family: str, method: str, package: str, callable_name: str, configs: Sequence[Mapping[str, object]], *, coordinate_mode: str, evidence: Mapping[str, object] | None = None) -> None:
        for config in configs:
            systems.append(_system_document(TaskLine.DENOISING, family, method, package, callable_name, config, coordinate_mode=coordinate_mode, evidence=evidence))

    add("savitzky_golay", "savgol_filter", "scipy==1.18.0", "scipy.signal.savgol_filter", [
        {"axis": -1, "deriv": 0, "mode": "interp", "polyorder": order, "window_length": window}
        for window in (5, 7, 9, 11, 15, 21) for order in (2, 3)
    ], coordinate_mode="index", evidence={"dependency": "available"})
    add("wavelet", "wavelet_threshold", "PyWavelets==1.9.0", "rpe.methods.classical.denoising.wavelet_threshold", [
        {"extension_mode": "symmetric", "max_level": 5, "noise_estimator": "finest_detail_mad", "threshold_mode": mode, "threshold_strategy": strategy, "wavelet": wavelet}
        for wavelet in ("db4", "db6", "sym8") for mode in ("soft", "hard") for strategy in ("universal_mad", "bayes_shrink")
    ], coordinate_mode="index", evidence={"dependency": "missing"})
    component_grid = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
    add("pca_reconstruction", "pca_reconstruction", "scikit-learn==1.9.0", "rpe.methods.classical.denoising.pca_reconstruction", [
        {"copy": True, "n_components": components, "svd_solver": "full", "whiten": False}
        for components in component_grid
    ], coordinate_mode="common_axis_train_partition_fit", evidence={"dependency": "available"})
    add("svd_reconstruction", "svd_reconstruction", "scikit-learn==1.9.0", "rpe.methods.classical.denoising.svd_reconstruction", [
        {"algorithm": "randomized", "n_components": components, "n_iter": 5, "random_state": GLOBAL_SEED}
        for components in component_grid
    ], coordinate_mode="common_axis_train_partition_fit", evidence={"dependency": "available"})
    add("whittaker_smoothing", "whittaker_smoothing", "scipy==1.18.0", "rpe.methods.classical.denoising.whittaker_smoothing", [
        {"difference_order": 2, "lambda": lam}
        for lam in (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
    ], coordinate_mode="index", evidence={"dependency": "available"})
    return systems


def _peak_systems() -> list[dict[str, object]]:
    systems: list[dict[str, object]] = []

    def add(family: str, method: str, package: str, callable_name: str, configs: Sequence[Mapping[str, object]], evidence: Mapping[str, object]) -> None:
        for config in configs:
            systems.append(_system_document(TaskLine.PEAK_DETECTION, family, method, package, callable_name, config, coordinate_mode="physical_width_converted_by_median_spacing", evidence=evidence))

    add("find_peaks", "find_peaks", "scipy==1.18.0", "scipy.signal.find_peaks", [
        {"distance": None, "height": None, "plateau_size": None, "prominence_fraction": value, "rel_height": 0.5, "threshold": None, "width": None, "wlen": None}
        for value in (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    ], {"dependency": "available"})
    width_banks = ((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 4.0, 8.0), (2.0, 4.0, 6.0, 8.0), (2.0, 4.0, 8.0, 12.0))
    add("find_peaks_cwt", "find_peaks_cwt", "scipy==1.18.0", "scipy.signal.find_peaks_cwt", [
        {"gap_thresh": None, "max_distances": None, "min_length": None, "min_snr": min_snr, "noise_perc": 10, "wavelet": None, "widths_cm1": list(widths), "window_size": None}
        for widths in width_banks for min_snr in (1, 2, 3)
    ], {"dependency": "available"})
    add("mspd", "mspd", "rpe", "rpe.methods.classical.peaks.mspd", [
        {"max_scale_cm1": scale, "ridge_vote_fraction": fraction, "tie_break": "position_ascending"}
        for scale in (4.0, 8.0, 12.0, 16.0) for fraction in (0.25, 0.5, 0.75)
    ], {"implementation": "missing"})
    return systems


def _view_documents(systems: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    expectations = {
        TaskLine.BASELINE_CORRECTION: (210, 14, 15, "planned_target_k210_not_yet_powered"),
        TaskLine.DENOISING: (60, 5, 12, "not_powered_for_phase5"),
        TaskLine.PEAK_DETECTION: (36, 3, 12, "not_powered_for_phase5"),
    }
    views = []
    for task_line in TaskLine:
        selected = [system for system in systems if system["task_line"] == task_line.value]
        system_count, family_count, systems_per_family, status = expectations[task_line]
        views.append(
            {
                "family_count": family_count,
                "phase5_power_status": status,
                "system_count": system_count,
                "system_ids": sorted(str(system["system_id"]) for system in selected),
                "systems_per_family": systems_per_family,
                "task_line": task_line.value,
            }
        )
    return sorted(views, key=lambda value: str(value["task_line"]).encode("utf-8"))


def _catalog_scientific_projection(document: Mapping[str, object]) -> dict[str, object]:
    systems = []
    for raw_system in document["systems"]:  # type: ignore[index]
        system = _object("system", raw_system)
        systems.append({key: system[key] for key in ("system_id", "task_line", "family_id", "method_id", "backend", "hyperparameters", "input_contract", "output_contract", "ordered_composition", "method_seed", "metric_eligibility", "downstream_eligibility", "protocol_eligibility")})
    return {
        "global_seed": document["global_seed"],
        "identity_domains": document["identity_domains"],
        "phase2_fallback_binding": document["phase2_fallback_binding"],
        "schema_version": document["schema_version"],
        "systems": systems,
        "views": document["views"],
    }


def build_classical_catalog_document(project_root: Path | None = None) -> dict[str, object]:
    root = _PROJECT_ROOT if project_root is None else Path(project_root)
    systems = _baseline_systems() + _denoising_systems() + _peak_systems()
    systems.sort(key=lambda value: str(value["system_id"]).encode("utf-8"))
    document: dict[str, object] = {
        "availability_audit": {
            "catalog_runnable_system_count": 0,
            "installed": {
                "PyWavelets": None,
                "joblib": "1.5.3",
                "optuna": None,
                "pybaselines": "1.2.1",
                "scikit-learn": "1.9.0",
                "scipy": "1.18.0",
            },
            "planned_system_count": 306,
            "snapshot_role": "design_time_availability_not_runtime_promotion",
            "wrappers": {"baseline": False, "denoising": False, "peaks": False},
        },
        "catalog_id": "",
        "dependency_lock": _file_identity(root, "env/phase3-requirements.lock"),
        "evidence": {
            "design": _file_identity(root, "reports/phase3/step01_classical_system_registry_design.md"),
            "phase2_fallback": _file_identity(root, "reports/phase2/step04_phase2_fallback_decision.md"),
        },
        "global_seed": GLOBAL_SEED,
        "identity_domains": {
            "catalog": "rpe-phase3-catalog-v1",
            "grid_sampler_seed": GLOBAL_SEED,
            "system": "rpe-phase3-system-v1",
        },
        "phase2_fallback_binding": {
            "baseline_clean_gt_fidelity": "unavailable",
            "baseline_evidence": ["downstream", "half_split_consistency", "reference_free"],
            "decision": "authorized_fallback_pre_generation_model_adequacy_failure",
        },
        "schema_version": SCHEMA_VERSION,
        "systems": systems,
        "views": _view_documents(systems),
    }
    document["catalog_id"] = hashlib.sha256(
        _CATALOG_DOMAIN + _canonical_json_bytes(_catalog_scientific_projection(document))
    ).hexdigest()
    return document


def _parse_artifact(path: str, value: object) -> ArtifactIdentity:
    parsed = _exact_keys(path, value, ("byte_count", "path", "sha256"))
    relative_path = _nonempty(f"{path}.path", parsed["path"])
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise Phase3CatalogError(f"{path}.path", "must be normalized and relative")
    return ArtifactIdentity(
        path=relative_path,
        byte_count=_positive_int(f"{path}.byte_count", parsed["byte_count"]),
        sha256=_lower_hex(f"{path}.sha256", parsed["sha256"]),
    )


def _verify_artifact(root: Path, label: str, identity: ArtifactIdentity) -> None:
    path = root / identity.path
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Phase3CatalogError(label, str(error)) from error
    if len(raw) != identity.byte_count:
        raise Phase3CatalogError(label, "byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != identity.sha256:
        raise Phase3CatalogError(label, "SHA256 mismatch")


def _system_identity_document(system: Mapping[str, object]) -> dict[str, object]:
    return {key: system[key] for key in ("backend", "family_id", "hyperparameters", "input_contract", "method_id", "method_seed", "ordered_composition", "output_contract", "task_line")}


def _parse_system(index: int, value: object) -> Phase3System:
    path = f"systems[{index}]"
    system = _exact_keys(path, value, _SYSTEM_KEYS)
    system_id = _lower_hex(f"{path}.system_id", system["system_id"])
    if system["method_seed"] is not None:
        raise Phase3CatalogError(f"{path}.method_seed", "must be null for classical systems")
    availability_value = _nonempty(f"{path}.availability", system["availability"])
    try:
        availability = AvailabilityStatus(availability_value)
    except ValueError as error:
        raise Phase3CatalogError(f"{path}.availability", "is not supported") from error
    if availability is not AvailabilityStatus.IMPLEMENTATION_MISSING:
        raise Phase3CatalogError(f"{path}.availability", "must be implementation_missing in v1 snapshot")
    task_value = _nonempty(f"{path}.task_line", system["task_line"])
    try:
        task_line = TaskLine(task_value)
    except ValueError as error:
        raise Phase3CatalogError(f"{path}.task_line", "is not supported") from error
    ordered = _string_tuple(f"{path}.ordered_composition", system["ordered_composition"], nonempty=True)
    if len(ordered) != 1:
        raise Phase3CatalogError(f"{path}.ordered_composition", "must contain one atomic operation")
    expected_id = hashlib.sha256(_SYSTEM_DOMAIN + _canonical_json_bytes(_system_identity_document(system))).hexdigest()
    if system_id != expected_id:
        raise Phase3CatalogError(f"{path}.system identity", "does not match scientific descriptor")
    hyperparameters = _freeze(f"{path}.hyperparameters", _object(f"{path}.hyperparameters", system["hyperparameters"]))
    backend = _freeze(f"{path}.backend", _exact_keys(f"{path}.backend", system["backend"], ("callable", "package")))
    input_contract = _freeze(f"{path}.input_contract", _object(f"{path}.input_contract", system["input_contract"]))
    output_contract = _freeze(f"{path}.output_contract", _object(f"{path}.output_contract", system["output_contract"]))
    evidence = _freeze(f"{path}.evidence", _object(f"{path}.evidence", system["evidence"]))
    assert isinstance(hyperparameters, Mapping)
    assert isinstance(backend, Mapping)
    assert isinstance(input_contract, Mapping)
    assert isinstance(output_contract, Mapping)
    assert isinstance(evidence, Mapping)
    metric_eligibility = _string_tuple(f"{path}.metric_eligibility", system["metric_eligibility"], nonempty=True)
    if task_line is TaskLine.BASELINE_CORRECTION:
        if "semisynthetic_clean_gt_fidelity" in metric_eligibility:
            raise Phase3CatalogError(f"{path}.metric_eligibility", "violates Phase 2 fallback")
        if evidence.get("phase2_fallback") != "downstream_reference_free_half_split_only":
            raise Phase3CatalogError(f"{path}.evidence.phase2_fallback", "does not bind fallback")
    return Phase3System(
        system_id=system_id,
        task_line=task_line,
        family_id=_nonempty(f"{path}.family_id", system["family_id"]),
        method_id=_nonempty(f"{path}.method_id", system["method_id"]),
        backend=backend,
        hyperparameters=hyperparameters,
        input_contract=input_contract,
        output_contract=output_contract,
        ordered_composition=ordered,
        method_seed=None,
        metric_eligibility=metric_eligibility,
        downstream_eligibility=_string_tuple(f"{path}.downstream_eligibility", system["downstream_eligibility"], nonempty=True),
        protocol_eligibility=_string_tuple(f"{path}.protocol_eligibility", system["protocol_eligibility"], nonempty=True),
        availability=availability,
        evidence=evidence,
    )


def _parse_views(value: object, systems: tuple[Phase3System, ...]) -> tuple[Phase3View, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise Phase3CatalogError("views", "must contain three views")
    expectations = {
        TaskLine.BASELINE_CORRECTION: (210, 14, 15, "planned_target_k210_not_yet_powered"),
        TaskLine.DENOISING: (60, 5, 12, "not_powered_for_phase5"),
        TaskLine.PEAK_DETECTION: (36, 3, 12, "not_powered_for_phase5"),
    }
    parsed_views = []
    seen: set[TaskLine] = set()
    for index, raw in enumerate(value):
        path = f"views[{index}]"
        view = _exact_keys(path, raw, _VIEW_KEYS)
        try:
            task_line = TaskLine(_nonempty(f"{path}.task_line", view["task_line"]))
        except ValueError as error:
            raise Phase3CatalogError(f"{path}.task_line", "is not supported") from error
        if task_line in seen:
            raise Phase3CatalogError("views.task_line", "is duplicated")
        seen.add(task_line)
        expected_count, expected_families, expected_per_family, expected_status = expectations[task_line]
        observed = (view["system_count"], view["family_count"], view["systems_per_family"], view["phase5_power_status"])
        if observed != (expected_count, expected_families, expected_per_family, expected_status):
            raise Phase3CatalogError(f"{path}.view counts", "do not match frozen design")
        expected_ids = tuple(sorted(system.system_id for system in systems if system.task_line is task_line))
        ids = _string_tuple(f"{path}.system_ids", view["system_ids"])
        if ids != expected_ids:
            raise Phase3CatalogError(f"{path}.system_ids", "do not match view systems")
        family_counts: dict[str, int] = {}
        for system in systems:
            if system.task_line is task_line:
                family_counts[system.family_id] = family_counts.get(system.family_id, 0) + 1
        if len(family_counts) != expected_families or set(family_counts.values()) != {expected_per_family}:
            raise Phase3CatalogError(f"{path}.family counts", "do not match frozen design")
        parsed_views.append(Phase3View(task_line, expected_count, expected_families, expected_per_family, expected_status, ids))
    if seen != set(TaskLine):
        raise Phase3CatalogError("views", "does not cover all task lines")
    return tuple(parsed_views)


def load_classical_catalog(
    path: Path,
    *,
    project_root: Path | None = None,
    enforce_frozen_identity: bool = True,
) -> Phase3ClassicalCatalog:
    path = Path(path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw, parse_constant=_reject_nonfinite)
    except Phase3CatalogError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase3CatalogError("catalog", str(error)) from error
    if not isinstance(document, Mapping) or raw != _canonical_json_bytes(document):
        raise Phase3CatalogError("catalog canonical", "must use canonical finite JSON")
    observed_sha = hashlib.sha256(raw).hexdigest()
    if enforce_frozen_identity and (len(raw) != CATALOG_BYTE_COUNT or observed_sha != CATALOG_SHA256):
        raise Phase3CatalogError("catalog identity", "bytes or SHA256 mismatch")
    root = _exact_keys("catalog.keys", document, _ROOT_KEYS)
    if root["schema_version"] != SCHEMA_VERSION:
        raise Phase3CatalogError("schema_version", "does not match")
    if root["global_seed"] != GLOBAL_SEED:
        raise Phase3CatalogError("global_seed", "does not match")
    dependency_lock = _parse_artifact("dependency_lock", root["dependency_lock"])
    evidence_raw = _exact_keys("evidence", root["evidence"], ("design", "phase2_fallback"))
    evidence = {name: _parse_artifact(f"evidence.{name}", value) for name, value in evidence_raw.items()}
    resolved_root = _PROJECT_ROOT if project_root is None else Path(project_root)
    _verify_artifact(resolved_root, "dependency_lock", dependency_lock)
    for name, identity in evidence.items():
        _verify_artifact(resolved_root, f"evidence.{name}", identity)
    if not isinstance(root["systems"], list):
        raise Phase3CatalogError("systems", "must be an array")
    systems = tuple(_parse_system(index, value) for index, value in enumerate(root["systems"]))
    ids = tuple(system.system_id for system in systems)
    if len(systems) != 306:
        raise Phase3CatalogError("system counts", "must equal 306")
    if len(ids) != len(set(ids)):
        raise Phase3CatalogError("system_id", "must be unique")
    if ids != tuple(sorted(ids)):
        raise Phase3CatalogError("systems.order", "must be sorted by system_id")
    descriptors = [_canonical_json_bytes(_system_identity_document(_object("system", value))) for value in root["systems"]]
    if len(descriptors) != len(set(descriptors)):
        raise Phase3CatalogError("systems.descriptor", "must be unique")
    views = _parse_views(root["views"], systems)
    catalog_id = _lower_hex("catalog_id", root["catalog_id"])
    expected_catalog_id = hashlib.sha256(_CATALOG_DOMAIN + _canonical_json_bytes(_catalog_scientific_projection(root))).hexdigest()
    if catalog_id != expected_catalog_id:
        raise Phase3CatalogError("catalog_id", "does not match scientific document")
    expected = build_classical_catalog_document(resolved_root)
    if _catalog_scientific_projection(root) != _catalog_scientific_projection(expected):
        raise Phase3CatalogError("catalog scientific document", "does not match frozen design")
    availability_audit = _freeze("availability_audit", _object("availability_audit", root["availability_audit"]))
    if root["availability_audit"] != expected["availability_audit"]:
        raise Phase3CatalogError(
            "availability_audit", "does not match frozen design-time snapshot"
        )
    fallback = _freeze("phase2_fallback_binding", _object("phase2_fallback_binding", root["phase2_fallback_binding"]))
    frozen_document = _freeze("document", root)
    assert isinstance(availability_audit, Mapping)
    assert isinstance(fallback, Mapping)
    assert isinstance(frozen_document, Mapping)
    return Phase3ClassicalCatalog(
        path=path, byte_count=len(raw), sha256=observed_sha, catalog_id=catalog_id,
        global_seed=GLOBAL_SEED, dependency_lock=dependency_lock,
        evidence=MappingProxyType(dict(sorted(evidence.items()))), views=views,
        systems=systems, availability_audit=availability_audit,
        phase2_fallback_binding=fallback, document=frozen_document,
    )


def audit_classical_catalog_availability(
    catalog: Phase3ClassicalCatalog, *, project_root: Path | None = None
) -> Mapping[str, object]:
    if not isinstance(catalog, Phase3ClassicalCatalog):
        raise Phase3CatalogError("catalog", "must be Phase3ClassicalCatalog")
    root = _PROJECT_ROOT if project_root is None else Path(project_root)
    installed = {
        "PyWavelets": _installed_version("PyWavelets"),
        "joblib": _installed_version("joblib"),
        "optuna": _installed_version("optuna"),
        "pybaselines": _installed_version("pybaselines"),
        "scikit-learn": _installed_version("scikit-learn"),
        "scipy": _installed_version("scipy"),
    }
    wrappers = {
        "baseline": (root / "rpe/methods/classical/baseline.py").is_file(),
        "denoising": (root / "rpe/methods/classical/denoising.py").is_file(),
        "peaks": (root / "rpe/methods/classical/peaks.py").is_file(),
    }
    value = {
        "catalog_runnable_system_count": sum(system.availability is AvailabilityStatus.RUNNABLE for system in catalog.systems),
        "installed": installed,
        "planned_system_count": len(catalog.systems),
        "wrappers": wrappers,
    }
    frozen = _freeze("availability", value)
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "AvailabilityStatus", "Phase3CatalogError", "Phase3ClassicalCatalog",
    "Phase3System", "Phase3View", "TaskLine",
    "audit_classical_catalog_availability", "build_classical_catalog_document",
    "canonical_catalog_bytes", "load_classical_catalog",
]
