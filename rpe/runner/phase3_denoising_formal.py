from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.evaluation import Spectrum1D
from rpe.io.store import UnifiedDataset
from rpe.methods.catalog import (
    Phase3ClassicalCatalog,
    Phase3System,
    TaskLine,
    load_classical_catalog,
)
from rpe.methods.classical.denoising import (
    DenoisingFitContext,
    DenoisingFitError,
    DenoisingRunResult,
    DenoisingRunStatus,
    DenoisingWarning,
    FittedDenoiser,
    fit_denoising_system,
    run_stateless_denoising_system,
    transform_fitted_denoiser,
)
from rpe.runner.d1_bacteria_id import _combine, _load_dataset, _stratified_split


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase3/configs/denoising_v1_formal_coverage.json"
RUN_DOMAIN = b"rpe-phase3-denoising-v1-formal-coverage-v1\0"
STATELESS_FAMILIES = frozenset({"savitzky_golay", "wavelet", "whittaker_smoothing"})
FITTED_FAMILIES = frozenset({"pca_reconstruction", "svd_reconstruction"})
SUCCESS = frozenset({DenoisingRunStatus.COMPLETE, DenoisingRunStatus.COMPLETE_WITH_WARNING})
CONFIG_BYTES = 4659
CONFIG_SHA256 = "248b3fcfce517ada2cd3536865d809b5be6f684981c612776b84ba22670406ab"


class DenoisingFormalError(ValueError):
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
    denoising_system_ids_sha256: str


@dataclass(frozen=True)
class DenoisingFormalAuthorities:
    parent_plan: ArtifactIdentity
    step5_design: ArtifactIdentity
    step6_denoising_report: ArtifactIdentity
    step7_peak_report: ArtifactIdentity
    step8_protocol: ArtifactIdentity

    @property
    def parent_plan_sha256(self) -> str:
        return self.parent_plan.sha256

    @property
    def step5_design_sha256(self) -> str:
        return self.step5_design.sha256

    @property
    def step6_denoising_report_sha256(self) -> str:
        return self.step6_denoising_report.sha256

    @property
    def step7_peak_report_sha256(self) -> str:
        return self.step7_peak_report.sha256

    @property
    def step8_protocol_sha256(self) -> str:
        return self.step8_protocol.sha256


@dataclass(frozen=True)
class DenoisingFormalExpected:
    system_count: int
    stateless_system_count: int
    fitted_system_count: int
    stateless_source_count: int
    stateless_source_point_count: int
    fit_source_count: int
    validation_source_count: int
    test_source_count: int
    fit_receipt_count: int
    stateless_transform_receipt_count: int
    fitted_transform_receipt_count: int
    total_transform_receipt_count: int


@dataclass(frozen=True)
class DenoisingFormalOperational:
    worker_processes: int
    verifier_worker_processes: int
    blas_threads_per_process: int
    parallel_start_method: str


@dataclass(frozen=True)
class DenoisingFormalCoveragePolicy:
    require_zero_unsuccessful: bool
    descriptive_minimum_successful_fraction: float
    descriptive_minimum_common_record_fraction: float
    minimum_promoted_per_family: int = 8
    success_statuses: tuple[DenoisingRunStatus, ...] = (
        DenoisingRunStatus.COMPLETE,
        DenoisingRunStatus.COMPLETE_WITH_WARNING,
    )
    subset_full_tau_status: str = "not_evaluable_no_quality_endpoint"


@dataclass(frozen=True)
class DenoisingFormalRruff:
    dataset_path: str
    source_ledger_path: str
    source_ledger_size: int
    source_ledger_sha256: str
    phase1_manifest: ArtifactIdentity
    scientific_config_sha256: str
    source_snapshot_sha256: str


@dataclass(frozen=True)
class DenoisingFormalBacteria:
    dataset_path: str
    dataset_sha256sums_sha256: str
    d1_config: ArtifactIdentity
    d1_seed0_result: ArtifactIdentity
    seed: int
    validation_per_class: int
    axis_sha256: str
    train_record_ids_sha256: str
    validation_record_ids_sha256: str
    test_record_ids_sha256: str
    train_record_ledger_sha256: str
    validation_record_ledger_sha256: str
    test_record_ledger_sha256: str
    train_matrix_sha256: str
    validation_matrix_sha256: str
    test_matrix_sha256: str

    @property
    def d1_config_sha256(self) -> str:
        return self.d1_config.sha256

    @property
    def d1_seed0_result_sha256(self) -> str:
        return self.d1_seed0_result.sha256


@dataclass(frozen=True)
class Phase3DenoisingFormalConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    authorities: DenoisingFormalAuthorities
    catalog: CatalogIdentity
    phase3_lock: ArtifactIdentity
    rruff: DenoisingFormalRruff
    bacteria: DenoisingFormalBacteria
    expected: DenoisingFormalExpected
    operational: DenoisingFormalOperational
    policy: DenoisingFormalCoveragePolicy
    claim_boundary: str
    phase5_power_status: str
    system_ids: tuple[str, ...]
    document: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "document", _freeze_mapping(self.document))


@dataclass(frozen=True)
class DenoisingFormalSource:
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
class Phase3DenoisingFormalInputs:
    stateless_sources: tuple[DenoisingFormalSource, ...]
    fit_sources: tuple[DenoisingFormalSource, ...]
    validation_sources: tuple[DenoisingFormalSource, ...]
    test_sources: tuple[DenoisingFormalSource, ...]
    axis_sha256: str
    train_record_ids_sha256: str
    validation_record_ids_sha256: str
    test_record_ids_sha256: str
    train_record_ledger_sha256: str
    validation_record_ledger_sha256: str
    test_record_ledger_sha256: str
    train_matrix_sha256: str
    validation_matrix_sha256: str
    test_matrix_sha256: str


@dataclass(frozen=True)
class DenoisingFormalFitReceipt:
    system_id: str
    family_id: str
    method_id: str
    status: DenoisingRunStatus
    split_id: str
    representation_id: str
    training_record_ledger_sha256: str | None
    training_matrix_sha256: str | None
    axis_sha256: str | None
    context_sha256: str | None
    fit_state_sha256: str | None
    fit_state_offset_bytes: int | None
    fit_state_byte_count: int | None
    warnings: tuple[DenoisingWarning, ...]
    diagnostics: Mapping[str, object]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class DenoisingFormalTransformReceipt:
    cohort: str
    source_role: str
    source_order: int
    record_id: str
    system_id: str
    family_id: str
    method_id: str
    status: DenoisingRunStatus
    point_count: int
    axis_sha256: str
    output_sha256: str | None
    output_offset_bytes: int | None
    output_byte_count: int | None
    fitted_state_sha256: str | None
    diagnostics: Mapping[str, object]
    warnings: tuple[DenoisingWarning, ...]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class DenoisingSystemCoverageSummary:
    system_id: str
    family_id: str
    kind: str
    fit_status_counts: Mapping[str, int]
    transform_status_counts: Mapping[str, int]
    validation_status_counts: Mapping[str, int]
    test_status_counts: Mapping[str, int]
    successful_transform_count: int
    transform_count: int
    successful_fraction: float
    coverage_promoted: bool


@dataclass(frozen=True)
class DenoisingCoverageResult:
    system_summaries: tuple[DenoisingSystemCoverageSummary, ...]
    stateless_promoted_system_ids: tuple[str, ...]
    fitted_promoted_system_ids: tuple[str, ...]
    promoted_system_ids: tuple[str, ...]
    qualifying_family_ids: tuple[str, ...]
    stateless_common_successful_record_count: int
    fitted_validation_common_successful_record_count: int
    fitted_test_common_successful_record_count: int
    fitted_union_common_successful_record_count: int


@dataclass(frozen=True)
class DenoisingFormalSummary:
    path: Path
    run_id: str
    status: str
    is_formal_run: bool
    system_count: int
    fit_receipt_count: int
    transform_receipt_count: int
    promoted_system_count: int
    warning_count: int


@dataclass(frozen=True)
class _PackedRunResult:
    system_id: str
    family_id: str
    method_id: str
    status: DenoisingRunStatus
    denoised_intensity: np.ndarray | None
    output_sha256: str | None
    warnings: tuple[DenoisingWarning, ...]
    diagnostics: dict[str, object]
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True)
class _PackedFitFailure:
    path: str
    reason: str
    status: DenoisingRunStatus


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


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
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    raise DenoisingFormalError("mapping", "contains unsupported value")


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
        raise DenoisingFormalError(label, "identity mismatch")


def load_phase3_denoising_formal_config(path: Path) -> Phase3DenoisingFormalConfig:
    path = Path(path)
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise DenoisingFormalError("config", "invalid JSON") from error
    if not isinstance(document, Mapping) or raw != _canonical(document):
        raise DenoisingFormalError("config", "must be canonical finite JSON")
    observed_sha = hashlib.sha256(raw).hexdigest()
    if path.resolve() == DEFAULT_CONFIG.resolve() and (len(raw) != CONFIG_BYTES or observed_sha != CONFIG_SHA256):
        raise DenoisingFormalError("config", "frozen identity mismatch")
    root = ROOT
    authorities_raw = document["authorities"]
    assert isinstance(authorities_raw, Mapping)
    authorities = DenoisingFormalAuthorities(**{str(key): _identity(value) for key, value in authorities_raw.items()})
    for key, value in authorities.__dict__.items():
        _verify_identity(root, value, f"authorities.{key}")
    catalog_raw = document["catalog"]
    assert isinstance(catalog_raw, Mapping)
    catalog_identity = CatalogIdentity(
        str(catalog_raw["path"]), int(catalog_raw["byte_count"]), str(catalog_raw["sha256"]),
        str(catalog_raw["catalog_id"]), str(catalog_raw["denoising_system_ids_sha256"]),
    )
    _verify_identity(root, catalog_identity, "catalog")
    catalog = load_classical_catalog(root / catalog_identity.path, project_root=root)
    system_ids = tuple(sorted(system.system_id for system in catalog.systems if system.task_line is TaskLine.DENOISING))
    ids_sha = hashlib.sha256(("\n".join(system_ids) + "\n").encode()).hexdigest()
    if catalog.catalog_id != catalog_identity.catalog_id or ids_sha != catalog_identity.denoising_system_ids_sha256:
        raise DenoisingFormalError("catalog", "denoising view mismatch")
    lock = _identity(document["phase3_lock"])  # type: ignore[arg-type]
    _verify_identity(root, lock, "phase3_lock")
    expected = DenoisingFormalExpected(**{key: int(value) for key, value in document["expected"].items()})  # type: ignore[union-attr]
    operational = DenoisingFormalOperational(**document["operational"])  # type: ignore[arg-type]
    policy_raw = document["policy"]
    assert isinstance(policy_raw, Mapping)
    policy = DenoisingFormalCoveragePolicy(
        require_zero_unsuccessful=bool(policy_raw["require_zero_unsuccessful"]),
        descriptive_minimum_successful_fraction=float(policy_raw["descriptive_minimum_successful_fraction"]),
        descriptive_minimum_common_record_fraction=float(policy_raw["descriptive_minimum_common_record_fraction"]),
        minimum_promoted_per_family=int(policy_raw["minimum_promoted_per_family"]),
        success_statuses=tuple(DenoisingRunStatus(value) for value in policy_raw["success_statuses"]),
        subset_full_tau_status=str(policy_raw["subset_full_tau_status"]),
    )
    rruff_raw = document["rruff"]
    assert isinstance(rruff_raw, Mapping)
    rruff = DenoisingFormalRruff(
        dataset_path=str(rruff_raw["dataset_path"]), source_ledger_path=str(rruff_raw["source_ledger_path"]),
        source_ledger_size=int(rruff_raw["source_ledger_size"]), source_ledger_sha256=str(rruff_raw["source_ledger_sha256"]),
        phase1_manifest=_identity(rruff_raw["phase1_manifest"]),  # type: ignore[arg-type]
        scientific_config_sha256=str(rruff_raw["scientific_config_sha256"]),
        source_snapshot_sha256=str(rruff_raw["source_snapshot_sha256"]),
    )
    _verify_identity(root, rruff.phase1_manifest, "rruff.phase1_manifest")
    bacteria_raw = document["bacteria"]
    assert isinstance(bacteria_raw, Mapping)
    bacteria = DenoisingFormalBacteria(
        dataset_path=str(bacteria_raw["dataset_path"]), dataset_sha256sums_sha256=str(bacteria_raw["dataset_sha256sums_sha256"]),
        d1_config=_identity(bacteria_raw["d1_config"]), d1_seed0_result=_identity(bacteria_raw["d1_seed0_result"]),  # type: ignore[arg-type]
        seed=int(bacteria_raw["seed"]), validation_per_class=int(bacteria_raw["validation_per_class"]),
        axis_sha256=str(bacteria_raw["axis_sha256"]), train_record_ids_sha256=str(bacteria_raw["train_record_ids_sha256"]),
        validation_record_ids_sha256=str(bacteria_raw["validation_record_ids_sha256"]), test_record_ids_sha256=str(bacteria_raw["test_record_ids_sha256"]),
        train_record_ledger_sha256=str(bacteria_raw["train_record_ledger_sha256"]), validation_record_ledger_sha256=str(bacteria_raw["validation_record_ledger_sha256"]),
        test_record_ledger_sha256=str(bacteria_raw["test_record_ledger_sha256"]), train_matrix_sha256=str(bacteria_raw["train_matrix_sha256"]),
        validation_matrix_sha256=str(bacteria_raw["validation_matrix_sha256"]), test_matrix_sha256=str(bacteria_raw["test_matrix_sha256"]),
    )
    _verify_identity(root, bacteria.d1_config, "bacteria.d1_config")
    _verify_identity(root, bacteria.d1_seed0_result, "bacteria.d1_seed0_result")
    if _sha(root / bacteria.dataset_path / "SHA256SUMS") != bacteria.dataset_sha256sums_sha256:
        raise DenoisingFormalError("bacteria.SHA256SUMS", "identity mismatch")
    if expected != DenoisingFormalExpected(60, 36, 24, 10000, 22053403, 62700, 300, 3000, 24, 360000, 79200, 439200):
        raise DenoisingFormalError("expected", "does not match Step 8")
    if (operational.worker_processes != 8 or operational.verifier_worker_processes != 7
            or operational.blas_threads_per_process != 1 or operational.parallel_start_method != "fork"):
        raise DenoisingFormalError("operational", "does not match frozen execution")
    if len(system_ids) != 60:
        raise DenoisingFormalError("catalog", "must contain 60 denoising systems")
    return Phase3DenoisingFormalConfig(
        path, len(raw), observed_sha, str(document["schema_version"]), authorities, catalog_identity, lock, rruff, bacteria,
        expected, operational, policy, str(document["claim_boundary"]), str(document["phase5_power_status"]), system_ids, document,
    )


def _report_ids_sha(ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _materialize_split(data: object, axis: np.ndarray, cohort_id: str) -> tuple[DenoisingFormalSource, ...]:
    values = data  # private D1 _SplitData, intentionally bound by D1 authority
    result = []
    for index, record_id in enumerate(values.record_ids):
        intensity = np.ascontiguousarray(values.intensity[index, ::-1], dtype="<f8")
        spectrum = Spectrum1D(record_id, None, axis, intensity)
        result.append(DenoisingFormalSource(
            cohort_id, record_id, int(values.labels[index]), index, intensity.size, _array_sha(axis), _array_sha(intensity), spectrum, None, None
        ))
    return tuple(result)


def load_phase3_denoising_formal_inputs(
    config: Phase3DenoisingFormalConfig, *, project_root: Path
) -> Phase3DenoisingFormalInputs:
    root = Path(project_root)
    ledger_path = root / config.rruff.source_ledger_path
    if ledger_path.stat().st_size != config.rruff.source_ledger_size or _sha(ledger_path) != config.rruff.source_ledger_sha256:
        raise DenoisingFormalError("rruff.source_ledger", "identity mismatch")
    phase1_manifest = json.loads((root / config.rruff.phase1_manifest.path).read_bytes())
    if (phase1_manifest.get("selected_source_subset_sha256") != config.rruff.source_ledger_sha256
            or phase1_manifest.get("scientific_config", {}).get("sha256") != config.rruff.scientific_config_sha256
            or phase1_manifest.get("source_snapshot_sha256") != config.rruff.source_snapshot_sha256):
        raise DenoisingFormalError("rruff.phase1_manifest", "authority mismatch")
    rows = [json.loads(line) for line in ledger_path.read_bytes().splitlines()]
    if len(rows) != config.expected.stateless_source_count:
        raise DenoisingFormalError("rruff.source_ledger", "count mismatch")
    stateless = []
    with UnifiedDataset.open(root / config.rruff.dataset_path, verify_checksums=True) as dataset:
        for order, row in enumerate(rows):
            if int(row["selection_rank"]) != order:
                raise DenoisingFormalError("rruff.source_order", "must be consecutive")
            record = dataset.get(str(row["record_id"]))
            axis = np.asarray(record.wavenumber, dtype="<f8")
            intensity = np.asarray(record.intensity, dtype="<f8")
            if np.all(np.diff(axis) < 0):
                axis = np.ascontiguousarray(axis[::-1])
                intensity = np.ascontiguousarray(intensity[::-1])
            spectrum = Spectrum1D(str(row["record_id"]), record.meta.sample_id, axis, intensity)
            axis_sha, intensity_sha = _array_sha(axis), _array_sha(intensity)
            if axis_sha != row["normalized_axis_float64_sha256"] or intensity_sha != row["normalized_intensity_float64_sha256"]:
                raise DenoisingFormalError("rruff.source", "normalized hash mismatch")
            stateless.append(DenoisingFormalSource(
                "rruff_core10k", str(row["record_id"]), int(row["class_label"]), order, intensity.size, axis_sha, intensity_sha, spectrum,
                str(row["sample_id"]), str(row["mineral_name"]),
            ))
    if sum(source.point_count for source in stateless) != config.expected.stateless_source_point_count:
        raise DenoisingFormalError("rruff.source_points", "count mismatch")
    split_data, _ = _load_dataset(root / config.bacteria.dataset_path, 4096)
    finetune_train, validation_data = _stratified_split(
        split_data["finetune"], seed=config.bacteria.seed, validation_per_class=config.bacteria.validation_per_class, expected_class_count=30
    )
    train_data = _combine(split_data["reference"], finetune_train)
    test_data = split_data["test"]
    with UnifiedDataset.open(root / config.bacteria.dataset_path, verify_checksums=True) as dataset:
        raw_axis = np.asarray(dataset[0].wavenumber, dtype="<f8")
    axis = np.ascontiguousarray(raw_axis[::-1]) if np.all(np.diff(raw_axis) < 0) else np.ascontiguousarray(raw_axis)
    if _array_sha(axis) != config.bacteria.axis_sha256:
        raise DenoisingFormalError("bacteria.axis", "identity mismatch")
    fit_sources = _materialize_split(train_data, axis, "bacteria_train")
    validation_sources = _materialize_split(validation_data, axis, "bacteria_validation")
    test_sources = _materialize_split(test_data, axis, "bacteria_test")
    identities = {
        "train_record_ids_sha256": _report_ids_sha(tuple(source.record_id for source in fit_sources)),
        "validation_record_ids_sha256": _report_ids_sha(tuple(source.record_id for source in validation_sources)),
        "test_record_ids_sha256": _report_ids_sha(tuple(source.record_id for source in test_sources)),
        "train_record_ledger_sha256": hashlib.sha256(
            b"".join(_canonical({"record_id": source.record_id}) for source in fit_sources)
        ).hexdigest(),
        "validation_record_ledger_sha256": hashlib.sha256(
            b"".join(_canonical({"record_id": source.record_id}) for source in validation_sources)
        ).hexdigest(),
        "test_record_ledger_sha256": hashlib.sha256(
            b"".join(_canonical({"record_id": source.record_id}) for source in test_sources)
        ).hexdigest(),
        "train_matrix_sha256": _array_sha(np.asarray([source.spectrum.intensity for source in fit_sources], dtype="<f8")),
        "validation_matrix_sha256": _array_sha(np.asarray([source.spectrum.intensity for source in validation_sources], dtype="<f8")),
        "test_matrix_sha256": _array_sha(np.asarray([source.spectrum.intensity for source in test_sources], dtype="<f8")),
    }
    for key, observed in identities.items():
        if observed != getattr(config.bacteria, key):
            raise DenoisingFormalError(f"bacteria.{key}", "identity mismatch")
    if set(source.record_id for source in fit_sources) & set(source.record_id for source in (*validation_sources, *test_sources)):
        raise DenoisingFormalError("bacteria.roles", "train overlaps holdout")
    if set(source.record_id for source in validation_sources) & set(source.record_id for source in test_sources):
        raise DenoisingFormalError("bacteria.roles", "validation overlaps test")
    return Phase3DenoisingFormalInputs(
        tuple(stateless), fit_sources, validation_sources, test_sources, config.bacteria.axis_sha256,
        identities["train_record_ids_sha256"], identities["validation_record_ids_sha256"], identities["test_record_ids_sha256"],
        identities["train_record_ledger_sha256"], identities["validation_record_ledger_sha256"], identities["test_record_ledger_sha256"],
        identities["train_matrix_sha256"], identities["validation_matrix_sha256"], identities["test_matrix_sha256"],
    )


def evaluate_denoising_coverage(
    *, fit_receipts: Sequence[DenoisingFormalFitReceipt], transform_receipts: Sequence[DenoisingFormalTransformReceipt],
    stateless_record_ids: Sequence[str], validation_record_ids: Sequence[str], test_record_ids: Sequence[str],
    systems: Sequence[Phase3System], policy: DenoisingFormalCoveragePolicy,
) -> DenoisingCoverageResult:
    frozen_systems = tuple(sorted(systems, key=lambda value: value.system_id))
    stateless_ids, validation_ids, test_ids = tuple(stateless_record_ids), tuple(validation_record_ids), tuple(test_record_ids)
    fit_by_system = {row.system_id: row for row in fit_receipts}
    if len(fit_by_system) != len(fit_receipts):
        raise DenoisingFormalError("fit receipts", "duplicate system")
    transforms: dict[str, list[DenoisingFormalTransformReceipt]] = defaultdict(list)
    for row in transform_receipts:
        transforms[row.system_id].append(row)
    success = set(policy.success_statuses)
    summaries = []
    stateless_promoted, fitted_promoted = [], []
    successful_by_system_role: dict[tuple[str, str], set[str]] = {}
    for system in frozen_systems:
        rows = transforms.get(system.system_id, [])
        kind = "fitted" if system.family_id in FITTED_FAMILIES else "stateless"
        expected = (("stateless", stateless_ids),) if kind == "stateless" else (("validation", validation_ids), ("test", test_ids))
        expected_pairs = [(role, record_id) for role, ids in expected for record_id in ids]
        observed_pairs = [(row.source_role, row.record_id) for row in rows]
        if observed_pairs != expected_pairs:
            raise DenoisingFormalError("transform receipt Cartesian product", f"{system.system_id} mismatch")
        fit_rows = [] if kind == "stateless" else [fit_by_system.get(system.system_id)]
        if kind == "fitted" and (fit_rows[0] is None or set(fit_by_system) - {s.system_id for s in frozen_systems if s.family_id in FITTED_FAMILIES}):
            raise DenoisingFormalError("fit receipts", "Cartesian product mismatch")
        fit_counts = Counter(row.status.value for row in fit_rows if row is not None)
        transform_counts = Counter(row.status.value for row in rows)
        validation_counts = Counter(row.status.value for row in rows if row.source_role == "validation")
        test_counts = Counter(row.status.value for row in rows if row.source_role == "test")
        successful = sum(1 for row in rows if row.status in success)
        promoted = (all(row.status in success for row in rows) and (kind == "stateless" or fit_rows[0].status in success))
        if policy.require_zero_unsuccessful and len(rows) != successful:
            promoted = False
        if promoted:
            (fitted_promoted if kind == "fitted" else stateless_promoted).append(system.system_id)
        for role, ids in expected:
            successful_by_system_role[(system.system_id, role)] = {row.record_id for row in rows if row.source_role == role and row.status in success}
        summaries.append(DenoisingSystemCoverageSummary(
            system.system_id, system.family_id, kind, dict(fit_counts), dict(transform_counts), dict(validation_counts), dict(test_counts),
            successful, len(rows), successful / len(rows), promoted,
        ))
    promoted = tuple(sorted((*stateless_promoted, *fitted_promoted)))
    by_family: dict[str, int] = Counter(next(system.family_id for system in frozen_systems if system.system_id == sid) for sid in promoted)
    qualifying = tuple(sorted(family for family, count in by_family.items() if count >= policy.minimum_promoted_per_family))
    def common(ids: Sequence[str], system_ids: Sequence[str], role: str) -> int:
        return sum(all(record_id in successful_by_system_role[(sid, role)] for sid in system_ids) for record_id in ids) if system_ids else 0
    return DenoisingCoverageResult(
        tuple(summaries), tuple(stateless_promoted), tuple(fitted_promoted), promoted, qualifying,
        common(stateless_ids, stateless_promoted, "stateless"), common(validation_ids, fitted_promoted, "validation"),
        common(test_ids, fitted_promoted, "test"),
        common(validation_ids, fitted_promoted, "validation") + common(test_ids, fitted_promoted, "test"),
    )


_WORKER_INPUTS: Phase3DenoisingFormalInputs | None = None
_WORKER_SYSTEMS: tuple[Phase3System, ...] = ()


def _worker_environment() -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _pack_run_result(result: DenoisingRunResult) -> _PackedRunResult:
    return _PackedRunResult(
        result.system_id, result.family_id, result.method_id, result.status,
        result.denoised_intensity, result.output_sha256, result.warnings,
        dict(result.diagnostics), result.error_code, result.error_message,
    )


def _stateless_task(task: tuple[int, int, int]) -> tuple[int, list[_PackedRunResult]]:
    system_index, start, stop = task
    if _WORKER_INPUTS is None:
        raise RuntimeError("formal worker inputs are not initialized")
    system = _WORKER_SYSTEMS[system_index]
    with threadpool_limits(limits=1):
        return start, [
            _pack_run_result(run_stateless_denoising_system(system, source.spectrum))
            for source in _WORKER_INPUTS.stateless_sources[start:stop]
        ]


def _fitted_task(system_index: int) -> tuple[FittedDenoiser | _PackedFitFailure, list[_PackedRunResult]]:
    if _WORKER_INPUTS is None:
        raise RuntimeError("formal worker inputs are not initialized")
    system = _WORKER_SYSTEMS[system_index]
    training = tuple(source.spectrum for source in _WORKER_INPUTS.fit_sources)
    context = DenoisingFitContext("d1_seed0_train", "bacteria_id_reference_increasing_float64_v1", tuple(value.spectrum.spectrum_id for value in _WORKER_INPUTS.fit_sources))
    try:
        with threadpool_limits(limits=1):
            fitted = fit_denoising_system(system, training, context)
            results = [
                _pack_run_result(transform_fitted_denoiser(fitted, source.spectrum))
                for source in (*_WORKER_INPUTS.validation_sources, *_WORKER_INPUTS.test_sources)
            ]
    except DenoisingFitError as error:
        return _PackedFitFailure(error.path, error.reason, error.status), []
    return fitted, results


def _warning_rows(
    warnings: Sequence[DenoisingWarning], kind: str, system_id: str,
    record_id: str | None, source_role: str,
) -> list[dict[str, object]]:
    return [{"category": value.category, "message": value.message, "receipt_kind": kind, "record_id": record_id,
             "sequence": index, "source_role": source_role, "system_id": system_id} for index, value in enumerate(warnings)]


def _source_row(source: DenoisingFormalSource, role: str) -> dict[str, object]:
    return {"axis_sha256": source.axis_sha256, "class_label": source.class_label, "cohort_id": source.cohort_id, "intensity_sha256": source.intensity_sha256,
            "mineral_name": source.mineral_name, "point_count": source.point_count, "record_id": source.record_id, "role": role, "sample_id": source.sample_id, "source_order": source.source_order}


def _system_row(system: Phase3System) -> dict[str, object]:
    return {"family_id": system.family_id, "hyperparameters": _json_ready(system.hyperparameters), "kind": "fitted" if system.family_id in FITTED_FAMILIES else "stateless",
            "method_id": system.method_id, "system_id": system.system_id}


def _fit_receipt_row(row: DenoisingFormalFitReceipt) -> dict[str, object]:
    return {"axis_sha256": row.axis_sha256, "context_sha256": row.context_sha256, "diagnostics": _json_ready(row.diagnostics), "error_code": row.error_code,
            "error_message": row.error_message, "family_id": row.family_id, "fit_state_byte_count": row.fit_state_byte_count, "fit_state_offset_bytes": row.fit_state_offset_bytes,
            "fit_state_sha256": row.fit_state_sha256, "method_id": row.method_id, "representation_id": row.representation_id, "split_id": row.split_id, "status": row.status.value,
            "system_id": row.system_id, "training_matrix_sha256": row.training_matrix_sha256, "training_record_ledger_sha256": row.training_record_ledger_sha256,
            "warnings": [{"category": value.category, "message": value.message} for value in row.warnings]}


def _transform_receipt_row(row: DenoisingFormalTransformReceipt) -> dict[str, object]:
    return {"axis_sha256": row.axis_sha256, "cohort": row.cohort, "diagnostics": _json_ready(row.diagnostics), "error_code": row.error_code, "error_message": row.error_message,
            "family_id": row.family_id, "fitted_state_sha256": row.fitted_state_sha256, "method_id": row.method_id, "output_byte_count": row.output_byte_count,
            "output_offset_bytes": row.output_offset_bytes, "output_sha256": row.output_sha256, "point_count": row.point_count, "record_id": row.record_id,
            "source_order": row.source_order, "source_role": row.source_role, "status": row.status.value, "system_id": row.system_id,
            "warnings": [{"category": value.category, "message": value.message} for value in row.warnings]}


def _code_identity(root: Path) -> dict[str, dict[str, object]]:
    paths = ("rpe/methods/catalog.py", "rpe/methods/classical/denoising.py", "rpe/runner/d1_bacteria_id.py", "rpe/runner/phase3_denoising_formal.py",
             "rpe/runner/phase3_denoising_formal_verifier.py", "tools/run_phase3_denoising_formal.py")
    return {path: {"byte_count": (root / path).stat().st_size, "sha256": _sha(root / path)} for path in paths}


def _software_identity() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "PyWavelets", "h5py")}


def _build_phase3_denoising_formal_fixture_artifact(
    inputs: Phase3DenoisingFormalInputs, systems: Sequence[Phase3System], output_root: Path, *, config: Phase3DenoisingFormalConfig,
    catalog: Phase3ClassicalCatalog, worker_count: int, project_root: Path,
) -> DenoisingFormalSummary:
    global _WORKER_INPUTS, _WORKER_SYSTEMS
    if worker_count <= 0:
        raise DenoisingFormalError("worker_count", "must be positive")
    frozen_systems = tuple(sorted(systems, key=lambda value: value.system_id))
    if not frozen_systems or len({value.system_id for value in frozen_systems}) != len(frozen_systems):
        raise DenoisingFormalError("systems", "must be nonempty and unique")
    if any(value.task_line is not TaskLine.DENOISING for value in frozen_systems):
        raise DenoisingFormalError("systems", "must all be denoising systems")
    root = Path(project_root)
    input_identity = {
        "axis_sha256": inputs.axis_sha256, "stateless_source_count": len(inputs.stateless_sources), "fit_source_count": len(inputs.fit_sources),
        "validation_source_count": len(inputs.validation_sources), "test_source_count": len(inputs.test_sources),
        "train_record_ids_sha256": inputs.train_record_ids_sha256, "validation_record_ids_sha256": inputs.validation_record_ids_sha256,
        "test_record_ids_sha256": inputs.test_record_ids_sha256, "train_matrix_sha256": inputs.train_matrix_sha256,
        "validation_matrix_sha256": inputs.validation_matrix_sha256, "test_matrix_sha256": inputs.test_matrix_sha256,
        "stateless_sources_sha256": hashlib.sha256(b"".join(_canonical(_source_row(value, "stateless")) for value in inputs.stateless_sources)).hexdigest(),
    }
    identity = {"catalog_id": catalog.catalog_id, "catalog_sha256": catalog.sha256, "code_identity": _code_identity(root), "config_sha256": config.sha256,
                "input_identity": input_identity, "phase3_lock_sha256": config.phase3_lock.sha256, "software_identity": _software_identity(),
                "system_ids": [value.system_id for value in frozen_systems]}
    run_id = hashlib.sha256(RUN_DOMAIN + _canonical(identity)).hexdigest()
    path = Path(output_root) / f"phase3-denoising-v1-formal-{run_id}"
    if path.exists():
        raise DenoisingFormalError("output", "run path already exists")
    path.mkdir(parents=True)
    config_payload = config.path.read_bytes()
    (path / "config.json").write_bytes(config_payload)
    sources = [*(_source_row(v, "stateless") for v in inputs.stateless_sources), *(_source_row(v, "fit") for v in inputs.fit_sources),
               *(_source_row(v, "validation") for v in inputs.validation_sources), *(_source_row(v, "test") for v in inputs.test_sources)]
    (path / "sources.jsonl").write_bytes(b"".join(_canonical(value) for value in sources))
    (path / "systems.jsonl").write_bytes(b"".join(_canonical(_system_row(value)) for value in frozen_systems))
    stateless_systems = tuple(value for value in frozen_systems if value.family_id in STATELESS_FAMILIES)
    fitted_systems = tuple(value for value in frozen_systems if value.family_id in FITTED_FAMILIES)
    fit_receipts: list[DenoisingFormalFitReceipt] = []
    transform_receipts: list[DenoisingFormalTransformReceipt] = []
    warning_rows: list[dict[str, object]] = []
    _WORKER_INPUTS, _WORKER_SYSTEMS = inputs, frozen_systems
    _worker_environment()
    with (path / "denoised_outputs.f64le").open("wb") as output_stream, (path / "fitted_states.f64le").open("wb") as state_stream:
        for system in stateless_systems:
            system_index = frozen_systems.index(system)
            tasks = [(system_index, start, min(start + 16, len(inputs.stateless_sources))) for start in range(0, len(inputs.stateless_sources), 16)]
            if worker_count == 1:
                chunks: Iterable[tuple[int, list[_PackedRunResult]]] = map(_stateless_task, tasks)
                executor = None
            else:
                executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("fork"))
                chunks = executor.map(_stateless_task, tasks, chunksize=1)
            try:
                for start, results in chunks:
                    for relative, result in enumerate(results):
                        source = inputs.stateless_sources[start + relative]
                        offset = None
                        byte_count = None
                        if result.denoised_intensity is not None:
                            raw = np.ascontiguousarray(result.denoised_intensity, dtype="<f8").tobytes()
                            offset, byte_count = output_stream.tell(), len(raw)
                            output_stream.write(raw)
                        receipt = DenoisingFormalTransformReceipt(
                            "stateless", "stateless", source.source_order, source.record_id, system.system_id, system.family_id, system.method_id, result.status,
                            source.point_count, source.axis_sha256, result.output_sha256, offset, byte_count, None, result.diagnostics, result.warnings, result.error_code, result.error_message,
                        )
                        transform_receipts.append(receipt)
                        warning_rows.extend(_warning_rows(result.warnings, "transform", system.system_id, source.record_id, "stateless"))
            finally:
                if executor is not None:
                    executor.shutdown()
        fit_indices = [frozen_systems.index(value) for value in fitted_systems]
        if worker_count == 1:
            fit_results: Iterable[tuple[FittedDenoiser | _PackedFitFailure, list[_PackedRunResult]]] = map(_fitted_task, fit_indices)
            fit_executor = None
        else:
            fit_executor = ProcessPoolExecutor(max_workers=min(worker_count, len(fit_indices)), mp_context=multiprocessing.get_context("fork"))
            fit_results = fit_executor.map(_fitted_task, fit_indices, chunksize=1)
        try:
            for system, (fitted_or_error, results) in zip(fitted_systems, fit_results, strict=True):
                if isinstance(fitted_or_error, _PackedFitFailure):
                    error = fitted_or_error
                    fit_receipts.append(DenoisingFormalFitReceipt(system.system_id, system.family_id, system.method_id, error.status, "d1_seed0_train",
                        "bacteria_id_reference_increasing_float64_v1", None, None, None, None, None, None, None, (), {}, error.path, error.reason))
                    for source_index, source in enumerate((*inputs.validation_sources, *inputs.test_sources)):
                        role = "validation" if source_index < len(inputs.validation_sources) else "test"
                        transform_receipts.append(DenoisingFormalTransformReceipt("fitted", role, source.source_order,
                            source.record_id, system.system_id, system.family_id, system.method_id, DenoisingRunStatus.FAILED_FIT, source.point_count, source.axis_sha256,
                            None, None, None, None, {}, (), error.path, error.reason))
                    continue
                fitted = fitted_or_error
                state_offset = state_stream.tell()
                descriptors = []
                for role, array in (("mean", fitted.mean), ("components", fitted.components), ("explained_variance", fitted.explained_variance)):
                    if array is None:
                        continue
                    raw = np.ascontiguousarray(array, dtype="<f8").tobytes()
                    descriptors.append({"byte_count": len(raw), "offset": state_stream.tell(), "role": role, "sha256": hashlib.sha256(raw).hexdigest(), "shape": list(array.shape)})
                    state_stream.write(raw)
                fit_status = DenoisingRunStatus.COMPLETE_WITH_WARNING if fitted.warnings else DenoisingRunStatus.COMPLETE
                fit_receipts.append(DenoisingFormalFitReceipt(system.system_id, system.family_id, system.method_id, fit_status, "d1_seed0_train",
                    "bacteria_id_reference_increasing_float64_v1", fitted.training_record_ledger_sha256, fitted.training_matrix_sha256, fitted.axis_sha256, fitted.context_sha256,
                    fitted.state_sha256, state_offset, state_stream.tell() - state_offset, fitted.warnings, {"arrays": descriptors, "n_components": fitted.n_components}, None, None))
                warning_rows.extend(_warning_rows(fitted.warnings, "fit", system.system_id, None, "fit"))
                ordered_sources = (*inputs.validation_sources, *inputs.test_sources)
                for index, (source, result) in enumerate(zip(ordered_sources, results, strict=True)):
                    offset = None
                    byte_count = None
                    if result.denoised_intensity is not None:
                        raw = np.ascontiguousarray(result.denoised_intensity, dtype="<f8").tobytes()
                        offset, byte_count = output_stream.tell(), len(raw)
                        output_stream.write(raw)
                    role = "validation" if index < len(inputs.validation_sources) else "test"
                    transform_receipts.append(DenoisingFormalTransformReceipt("fitted", role, source.source_order, source.record_id, system.system_id, system.family_id,
                        system.method_id, result.status, source.point_count, source.axis_sha256, result.output_sha256, offset, byte_count, fitted.state_sha256,
                        result.diagnostics, result.warnings, result.error_code, result.error_message))
                    warning_rows.extend(_warning_rows(result.warnings, "transform", system.system_id, source.record_id, role))
        finally:
            if fit_executor is not None:
                fit_executor.shutdown()
    (path / "fit_receipts.jsonl").write_bytes(b"".join(_canonical(_fit_receipt_row(value)) for value in fit_receipts))
    (path / "transform_receipts.jsonl").write_bytes(b"".join(_canonical(_transform_receipt_row(value)) for value in transform_receipts))
    (path / "warnings.jsonl").write_bytes(b"".join(_canonical(value) for value in warning_rows))
    coverage = evaluate_denoising_coverage(fit_receipts=fit_receipts, transform_receipts=transform_receipts, stateless_record_ids=tuple(v.record_id for v in inputs.stateless_sources),
        validation_record_ids=tuple(v.record_id for v in inputs.validation_sources), test_record_ids=tuple(v.record_id for v in inputs.test_sources), systems=frozen_systems, policy=config.policy)
    system_summary_rows = [{"coverage_promoted": value.coverage_promoted, "family_id": value.family_id, "fit_status_counts": dict(value.fit_status_counts), "kind": value.kind,
                            "successful_fraction": value.successful_fraction, "successful_transform_count": value.successful_transform_count, "system_id": value.system_id,
                            "test_status_counts": dict(value.test_status_counts), "transform_count": value.transform_count, "transform_status_counts": dict(value.transform_status_counts),
                            "validation_status_counts": dict(value.validation_status_counts)} for value in coverage.system_summaries]
    (path / "system_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in system_summary_rows))
    family_rows = []
    for family in sorted({value.family_id for value in coverage.system_summaries}):
        values = [value for value in coverage.system_summaries if value.family_id == family]
        family_rows.append({"family_id": family, "promoted_system_count": sum(value.coverage_promoted for value in values),
                            "qualifying_family": family in coverage.qualifying_family_ids, "system_count": len(values)})
    (path / "family_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in family_rows))
    warning_summary_rows = []
    warning_counter = Counter(
        (str(value["system_id"]), str(value["source_role"]), str(value["category"]))
        for value in warning_rows
    )
    for system in frozen_systems:
        roles = ("stateless",) if system.family_id in STATELESS_FAMILIES else ("fit", "validation", "test")
        denominators = {"stateless": len(inputs.stateless_sources), "fit": 1, "validation": len(inputs.validation_sources), "test": len(inputs.test_sources)}
        categories = sorted({category for sid, role, category in warning_counter if sid == system.system_id and role in roles})
        warning_summary_rows.append({
            "family_id": system.family_id,
            "role_receipt_counts": {role: denominators[role] for role in roles},
            "system_id": system.system_id,
            "warning_counts": {
                role: {category: warning_counter[(system.system_id, role, category)] for category in categories if warning_counter[(system.system_id, role, category)]}
                for role in roles
            },
            "warning_receipt_fractions": {
                role: sum(1 for row in transform_receipts if row.system_id == system.system_id and row.source_role == role and row.warnings) / denominators[role]
                if role != "fit" else float(bool(next((row.warnings for row in fit_receipts if row.system_id == system.system_id), ())))
                for role in roles
            },
        })
    (path / "warning_summary.jsonl").write_bytes(b"".join(_canonical(value) for value in warning_summary_rows))
    is_formal = (tuple(value.system_id for value in frozen_systems) == config.system_ids and len(inputs.stateless_sources) == config.expected.stateless_source_count
                 and len(inputs.fit_sources) == config.expected.fit_source_count and len(inputs.validation_sources) == config.expected.validation_source_count
                 and len(inputs.test_sources) == config.expected.test_source_count and inputs.train_matrix_sha256 == config.bacteria.train_matrix_sha256
                 and inputs.validation_matrix_sha256 == config.bacteria.validation_matrix_sha256 and inputs.test_matrix_sha256 == config.bacteria.test_matrix_sha256)
    cohort_summary = {"fitted": {"fit_source_count": len(inputs.fit_sources), "promoted_system_count": len(coverage.fitted_promoted_system_ids),
        "test_common_successful_record_count": coverage.fitted_test_common_successful_record_count, "test_source_count": len(inputs.test_sources),
        "union_common_successful_record_count": coverage.fitted_union_common_successful_record_count, "validation_common_successful_record_count": coverage.fitted_validation_common_successful_record_count,
        "validation_source_count": len(inputs.validation_sources),
        "validation_common_successful_fraction": coverage.fitted_validation_common_successful_record_count / len(inputs.validation_sources),
        "test_common_successful_fraction": coverage.fitted_test_common_successful_record_count / len(inputs.test_sources),
        "union_common_successful_fraction": coverage.fitted_union_common_successful_record_count / (len(inputs.validation_sources) + len(inputs.test_sources))},
        "stateless": {"common_successful_record_count": coverage.stateless_common_successful_record_count,
        "common_successful_fraction": coverage.stateless_common_successful_record_count / len(inputs.stateless_sources),
        "promoted_system_count": len(coverage.stateless_promoted_system_ids), "source_count": len(inputs.stateless_sources)}}
    (path / "cohort_summary.json").write_bytes(_canonical(cohort_summary))
    promotion = {"coverage_promoted_K": len(coverage.promoted_system_ids), "fitted_promoted_system_ids": list(coverage.fitted_promoted_system_ids),
        "phase5_eligible_K": "not_evaluated", "planned_K": 60, "runnable_K": 60,
        "phase5_power_status": config.phase5_power_status, "promoted_system_ids": list(coverage.promoted_system_ids), "qualifying_family_ids": list(coverage.qualifying_family_ids),
        "stateless_promoted_system_ids": list(coverage.stateless_promoted_system_ids), "subset_full_tau_status": config.policy.subset_full_tau_status}
    (path / "promotion.json").write_bytes(_canonical(promotion))
    is_formal = (is_formal and len(fit_receipts) == config.expected.fit_receipt_count
                 and len(transform_receipts) == config.expected.total_transform_receipt_count)
    gate = {"formal_execution_complete": is_formal, "phase5_power_status": config.phase5_power_status, "strict_promotion": "complete_cartesian_and_100_percent_success",
            "subset_full_tau_status": config.policy.subset_full_tau_status}
    (path / "gate.json").write_bytes(_canonical(gate))
    status = "complete" if is_formal else "fixture_complete"
    manifest = {**identity, "claim_boundary": config.claim_boundary, "fit_receipt_count": len(fit_receipts), "is_formal_run": is_formal, "run_id": run_id,
        "schema_version": "phase3-denoising-v1-formal-artifact-v1", "status": status, "system_count": len(frozen_systems),
        "transform_receipt_count": len(transform_receipts), "warning_count": len(warning_rows)}
    (path / "manifest.json").write_bytes(_canonical(manifest))
    marker_name = "complete.json" if is_formal else "failed.json"
    (path / marker_name).write_bytes(_canonical({"is_formal_run": is_formal, "run_id": run_id, "status": status}))
    names = sorted(value.relative_to(path).as_posix() for value in path.iterdir() if value.is_file())
    (path / "SHA256SUMS").write_text("".join(f"{_sha(path / name)}  {name}\n" for name in names), encoding="utf-8")
    return DenoisingFormalSummary(path, run_id, status, is_formal, len(frozen_systems), len(fit_receipts), len(transform_receipts), len(coverage.promoted_system_ids), len(warning_rows))


def build_phase3_denoising_formal(output_root: Path, *, worker_count: int = 8) -> DenoisingFormalSummary:
    config = load_phase3_denoising_formal_config(DEFAULT_CONFIG)
    if worker_count != config.operational.worker_processes:
        raise DenoisingFormalError(
            "worker_count",
            f"formal authority build requires {config.operational.worker_processes}",
        )
    catalog = load_classical_catalog(ROOT / config.catalog.path, project_root=ROOT)
    by_id = {system.system_id: system for system in catalog.systems}
    systems = tuple(by_id[value] for value in config.system_ids)
    inputs = load_phase3_denoising_formal_inputs(config, project_root=ROOT)
    return _build_phase3_denoising_formal_fixture_artifact(inputs, systems, Path(output_root), config=config, catalog=catalog, worker_count=worker_count, project_root=ROOT)


def _read_json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise DenoisingFormalError(path.name, "must be canonical JSON")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        value = json.loads(line)
        if not isinstance(value, Mapping) or line != _canonical(value):
            raise DenoisingFormalError(path.name, "must be canonical JSONL")
        rows.append(value)
    return rows


def _verify_checksums(path: Path) -> set[str]:
    raw = (path / "SHA256SUMS").read_text()
    rows = []
    for line in raw.splitlines():
        try:
            digest, name = line.split("  ")
        except ValueError as error:
            raise DenoisingFormalError("checksum", "invalid line") from error
        if not (path / name).is_file() or _sha(path / name) != digest:
            raise DenoisingFormalError("checksum", f"{name} mismatch")
        rows.append((name, digest))
    if raw != "".join(f"{digest}  {name}\n" for name, digest in sorted(rows)):
        raise DenoisingFormalError("checksum", "must be canonically sorted")
    names = {name for name, _ in rows}
    actual = {value.name for value in path.iterdir() if value.is_file() and value.name != "SHA256SUMS"}
    if names != actual:
        raise DenoisingFormalError("checksum", "inventory mismatch")
    return names


def _slice_hash(stream, offset: int, byte_count: int) -> str:
    stream.seek(offset)
    remaining = byte_count
    digest = hashlib.sha256()
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise DenoisingFormalError("output stream", "truncated slice")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def verify_phase3_denoising_formal_artifact(path: Path, *, project_root: Path = ROOT) -> DenoisingFormalSummary:
    path = Path(path)
    names = _verify_checksums(path)
    marker_names = names & {"complete.json", "failed.json"}
    if len(marker_names) != 1:
        raise DenoisingFormalError("marker", "must contain exactly one")
    required = {"config.json", "manifest.json", "systems.jsonl", "sources.jsonl", "fit_receipts.jsonl", "transform_receipts.jsonl",
                "denoised_outputs.f64le", "fitted_states.f64le", "warnings.jsonl", "warning_summary.jsonl", "system_summary.jsonl", "family_summary.jsonl",
                "cohort_summary.json", "promotion.json", "gate.json", *marker_names}
    if names != required:
        raise DenoisingFormalError("artifact inventory", "does not match protocol")
    manifest = _read_json(path / "manifest.json")
    if (path / "config.json").read_bytes() != DEFAULT_CONFIG.read_bytes():
        raise DenoisingFormalError("config", "does not match frozen config")
    fit_rows = _read_jsonl(path / "fit_receipts.jsonl")
    transform_rows = _read_jsonl(path / "transform_receipts.jsonl")
    sources = _read_jsonl(path / "sources.jsonl")
    systems = _read_jsonl(path / "systems.jsonl")
    source_by_role_id = {(str(row["role"]), str(row["record_id"])): row for row in sources}
    state_by_system: dict[str, str] = {}
    with (path / "fitted_states.f64le").open("rb") as state_stream:
        expected_offset = 0
        for row in fit_rows:
            if row["fit_state_sha256"] is None:
                continue
            if int(row["fit_state_offset_bytes"]) != expected_offset:
                raise DenoisingFormalError("fit receipt identity", "state offset mismatch")
            diagnostics = row["diagnostics"]
            arrays = diagnostics["arrays"]
            reconstructed = {}
            for descriptor in arrays:
                offset, count = int(descriptor["offset"]), int(descriptor["byte_count"])
                if offset != expected_offset or _slice_hash(state_stream, offset, count) != descriptor["sha256"]:
                    raise DenoisingFormalError("fit receipt identity", "state descriptor mismatch")
                state_stream.seek(offset)
                array = np.frombuffer(state_stream.read(count), dtype="<f8").reshape(descriptor["shape"])
                reconstructed[str(descriptor["role"])] = array
                expected_offset += count
            fitted = FittedDenoiser(
                system_id=str(row["system_id"]), family_id=str(row["family_id"]), method_id=str(row["method_id"]), n_components=int(diagnostics["n_components"]),
                axis_sha256=str(row["axis_sha256"]), training_record_ledger_sha256=str(row["training_record_ledger_sha256"]),
                training_matrix_sha256=str(row["training_matrix_sha256"]), context_sha256=str(row["context_sha256"]), mean=reconstructed.get("mean"),
                components=reconstructed["components"], explained_variance=reconstructed["explained_variance"],
                warnings=tuple(DenoisingWarning(str(value["category"]), str(value["message"])) for value in row["warnings"]),
            )
            if fitted.state_sha256 != row["fit_state_sha256"]:
                raise DenoisingFormalError("fit receipt identity", "state hash mismatch")
            state_by_system[str(row["system_id"])] = fitted.state_sha256
        if expected_offset != (path / "fitted_states.f64le").stat().st_size:
            raise DenoisingFormalError("fit receipt identity", "unindexed state bytes")
    with (path / "denoised_outputs.f64le").open("rb") as output_stream:
        expected_offset = 0
        last_key = None
        for row in transform_rows:
            key = ({"stateless": 0, "validation": 1, "test": 1}[str(row["source_role"])], str(row["system_id"]),
                   {"stateless": 0, "validation": 0, "test": 1}[str(row["source_role"])], int(row["source_order"]))
            if last_key is not None and key <= last_key:
                raise DenoisingFormalError("transform receipts", "canonical order mismatch")
            last_key = key
            source = source_by_role_id.get((str(row["source_role"]), str(row["record_id"])))
            if source is None or source["axis_sha256"] != row["axis_sha256"] or source["point_count"] != row["point_count"]:
                raise DenoisingFormalError("transform receipts", "source identity mismatch")
            if row["output_sha256"] is None:
                if row["output_offset_bytes"] is not None or row["output_byte_count"] is not None:
                    raise DenoisingFormalError("output stream", "failure carries output")
            else:
                offset, count = int(row["output_offset_bytes"]), int(row["output_byte_count"])
                if offset != expected_offset or count != int(row["point_count"]) * 8 or _slice_hash(output_stream, offset, count) != row["output_sha256"]:
                    raise DenoisingFormalError("output stream", "descriptor mismatch")
                expected_offset += count
            if row["source_role"] in {"validation", "test"} and row["fitted_state_sha256"] is not None:
                if state_by_system.get(str(row["system_id"])) != row["fitted_state_sha256"]:
                    raise DenoisingFormalError("fit receipt identity", "transform state binding mismatch")
        if expected_offset != (path / "denoised_outputs.f64le").stat().st_size:
            raise DenoisingFormalError("output stream", "unindexed bytes")
    run_id = str(manifest["run_id"]); marker = _read_json(path / next(iter(marker_names)))
    if marker["run_id"] != run_id:
        raise DenoisingFormalError("marker", "run identity mismatch")
    promotion = _read_json(path / "promotion.json")
    return DenoisingFormalSummary(path, run_id, str(manifest["status"]), bool(manifest["is_formal_run"]), len(systems), len(fit_rows), len(transform_rows),
        len(promotion["promoted_system_ids"]), len(_read_jsonl(path / "warnings.jsonl")))


def verify_phase3_denoising_formal(path: Path, *, worker_count: int, project_root: Path = ROOT) -> DenoisingFormalSummary:
    # Imported lazily to keep production build independent of verifier orchestration.
    from rpe.runner.phase3_denoising_formal_verifier import verify_phase3_denoising_formal as independent_verify
    try:
        return independent_verify(
            path, worker_count=worker_count, project_root=project_root
        )
    except ValueError as error:
        raise DenoisingFormalError("independent verifier", str(error)) from error


__all__ = [
    "DenoisingFormalCoveragePolicy", "DenoisingFormalError", "DenoisingFormalFitReceipt", "DenoisingFormalSummary",
    "DenoisingFormalTransformReceipt", "DenoisingFormalSource", "Phase3DenoisingFormalConfig", "Phase3DenoisingFormalInputs",
    "build_phase3_denoising_formal", "evaluate_denoising_coverage", "load_phase3_denoising_formal_config",
    "load_phase3_denoising_formal_inputs", "verify_phase3_denoising_formal", "verify_phase3_denoising_formal_artifact",
]
