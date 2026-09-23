from __future__ import annotations

import csv
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

from rpe.downstream.sugar_quantitative import TARGET_NAMES, load_d4_sugar_cohort
from rpe.evaluation import ReplicatePairInput, SingleSpectrumInput, Spectrum1D
from rpe.methods.catalog import Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.denoising import (
    DenoisingFitContext,
    DenoisingFitError,
    DenoisingRunResult,
    DenoisingRunStatus,
    fit_denoising_system,
    run_stateless_denoising_system,
    transform_fitted_denoiser,
)
from rpe.metrics.consistency import HalfSplitPearsonConsistencyMetric
from rpe.metrics.reference_free import ISLikeStructureToNoiseMetric


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/denoising_evidence_v1.json"
RUN_DOMAIN = b"rpe-phase6-denoising-evidence-v1\0"
SUCCESS = frozenset(
    {
        DenoisingRunStatus.COMPLETE,
        DenoisingRunStatus.COMPLETE_WITH_WARNING,
    }
)


class DenoisingEvidenceError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class DenoisingEvidenceConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    authorities: Mapping[str, Mapping[str, object]]
    artifact_contract: Mapping[str, object]
    bootstrap: Mapping[str, object]
    cohort: Mapping[str, object]
    direct_gt_state: str
    endpoint_manifest: tuple[Mapping[str, object], ...]
    expected: Mapping[str, int]
    model_recipe: Mapping[str, object]
    protocols: Mapping[str, object]
    source_revision_status: str
    system_ids: tuple[str, ...]
    document: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "authorities", _freeze_mapping(self.authorities))
        object.__setattr__(self, "artifact_contract", _freeze_mapping(self.artifact_contract))
        object.__setattr__(self, "bootstrap", _freeze_mapping(self.bootstrap))
        object.__setattr__(self, "cohort", _freeze_mapping(self.cohort))
        object.__setattr__(self, "expected", _freeze_mapping(self.expected))
        object.__setattr__(self, "model_recipe", _freeze_mapping(self.model_recipe))
        object.__setattr__(self, "protocols", _freeze_mapping(self.protocols))
        object.__setattr__(
            self,
            "endpoint_manifest",
            tuple(_freeze_mapping(item) for item in self.endpoint_manifest),
        )
        object.__setattr__(self, "document", _freeze_mapping(self.document))


@dataclass(frozen=True)
class DenoisingEvidenceInputs:
    synthetic_fixture: bool
    cohort_id: str
    axis_cm1: np.ndarray
    support_axis_cm1: np.ndarray
    native_matrix: np.ndarray
    support_matrix: np.ndarray
    targets: np.ndarray
    target_names: tuple[str, ...]
    record_ids: tuple[str, ...]
    well_ids: tuple[str, ...]
    unique_well_ids: tuple[str, ...]
    rounds: np.ndarray
    repetitions: np.ndarray
    fold_ids: tuple[int, ...]
    fold_by_well: Mapping[str, int]
    record_to_fold: Mapping[str, int]
    train_indices_by_fold: Mapping[int, np.ndarray]
    validation_indices_by_fold: Mapping[int, np.ndarray]
    test_indices_by_fold: Mapping[int, np.ndarray]
    native_spectra: tuple[Spectrum1D, ...]
    well_count: int
    cohort_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_cm1", _readonly_array(self.axis_cm1, "<f8"))
        object.__setattr__(self, "support_axis_cm1", _readonly_array(self.support_axis_cm1, "<f8"))
        object.__setattr__(self, "native_matrix", _readonly_array(self.native_matrix, "<f8"))
        object.__setattr__(self, "support_matrix", _readonly_array(self.support_matrix, "<f8"))
        object.__setattr__(self, "targets", _readonly_array(self.targets, "<f8"))
        object.__setattr__(self, "rounds", _readonly_array(self.rounds, "<i8"))
        object.__setattr__(self, "repetitions", _readonly_array(self.repetitions, "<i8"))
        object.__setattr__(
            self,
            "train_indices_by_fold",
            {int(key): _readonly_array(value, "<i8") for key, value in self.train_indices_by_fold.items()},
        )
        object.__setattr__(
            self,
            "validation_indices_by_fold",
            {int(key): _readonly_array(value, "<i8") for key, value in self.validation_indices_by_fold.items()},
        )
        object.__setattr__(
            self,
            "test_indices_by_fold",
            {int(key): _readonly_array(value, "<i8") for key, value in self.test_indices_by_fold.items()},
        )
        object.__setattr__(self, "fold_by_well", _freeze_mapping(self.fold_by_well))
        object.__setattr__(self, "record_to_fold", _freeze_mapping(self.record_to_fold))
        object.__setattr__(self, "cohort_identity", _freeze_mapping(self.cohort_identity))


@dataclass(frozen=True)
class DenoisingEvidenceSummary:
    path: Path
    run_id: str
    status: str
    system_count: int
    fit_receipt_count: int
    transform_receipt_count: int
    model_receipt_count: int
    downstream_well_row_count: int
    reference_free_well_row_count: int
    half_split_well_row_count: int
    bootstrap_row_count: int
    method_evidence_row_count: int


class DenoisingEvidenceVerificationError(ValueError):
    pass


def _freeze_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    raise DenoisingEvidenceError("freeze", f"unsupported value type {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = {str(key): _freeze_value(item) for key, item in sorted(value.items())}
    return MappingProxyType(frozen)


def _readonly_array(value: object, dtype: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


def _normalize_identity(path: str, identity: Mapping[str, object]) -> Mapping[str, object]:
    byte_count = identity.get("byte_count", identity.get("bytes"))
    if not isinstance(identity.get("path"), str) or not identity["path"]:
        raise DenoisingEvidenceError(path, "identity path must be nonempty")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise DenoisingEvidenceError(path, "identity byte count must be a nonnegative integer")
    sha256 = identity.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise DenoisingEvidenceError(path, "identity SHA256 must be a 64-char string")
    return MappingProxyType({"path": identity["path"], "byte_count": byte_count, "sha256": sha256})


def load_phase6_denoising_evidence_config(path: Path = DEFAULT_CONFIG) -> DenoisingEvidenceConfig:
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical(document):
        raise DenoisingEvidenceError("config", "must be canonical JSON")
    if document.get("schema_version") != "phase6-denoising-evidence-v1":
        raise DenoisingEvidenceError("schema_version", "unexpected config schema")
    authorities = {
        name: _normalize_identity(f"authorities.{name}", value)
        for name, value in dict(document["authorities"]).items()
    }
    artifact_contract = dict(document["artifact_contract"])
    if tuple(artifact_contract.get("payload_files", ())) != (
        "config.json",
        "authority_bridge.json",
        "preflight.json",
        "fit_receipts.jsonl",
        "transform_receipts.jsonl",
        "model_receipts.jsonl",
        "system_status.jsonl",
        "downstream_well_rows.jsonl",
        "reference_free_well_rows.jsonl",
        "half_split_well_rows.jsonl",
        "bootstrap_results.jsonl",
        "method_evidence_rows.csv",
        "family_projection.csv",
        "manifest.json",
    ):
        raise DenoisingEvidenceError("artifact_contract", "payload inventory mismatch")
    system_ids = tuple(str(value) for value in document["system_ids"])
    if len(system_ids) != 60 or len(set(system_ids)) != 60:
        raise DenoisingEvidenceError("system_ids", "must contain exactly 60 unique IDs")
    promotion = json.loads((ROOT / str(authorities["denoising_promotion"]["path"])).read_text(encoding="utf-8"))
    if tuple(promotion.get("promoted_system_ids", ())) != system_ids:
        raise DenoisingEvidenceError("system_ids", "must match promoted denoising IDs")
    return DenoisingEvidenceConfig(
        path=path,
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=str(document["schema_version"]),
        authorities=authorities,
        artifact_contract=artifact_contract,
        bootstrap=dict(document["bootstrap"]),
        cohort=dict(document["cohort"]),
        direct_gt_state=str(document["direct_gt_state"]),
        endpoint_manifest=tuple(dict(item) for item in document["endpoint_manifest"]),
        expected=dict(document["expected"]),
        model_recipe=dict(document["model_recipe"]),
        protocols=dict(document["protocols"]),
        source_revision_status=str(document["source_revision_status"]),
        system_ids=system_ids,
        document=document,
    )


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise DenoisingEvidenceVerificationError(f"{path.name} must be a JSON object")
    return value


def _artifact_files(path: Path) -> dict[str, bytes]:
    return {
        current.relative_to(path).as_posix(): current.read_bytes()
        for current in path.rglob("*")
        if current.is_file()
    }


def _canonical(value: object) -> bytes:
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


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sha(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(_json_ready(row)) for row in rows)


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
    return value


def _csv_bytes(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> bytes:
    output: list[str] = []

    class _Writer:
        def write(self, text: str) -> int:
            output.append(text)
            return len(text)

    writer = csv.DictWriter(_Writer(), fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_cell(row.get(name)) for name in fieldnames})
    return "".join(output).encode("utf-8")


def _unique_wells(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _spectra_from_matrix(
    axis: np.ndarray,
    matrix: np.ndarray,
    record_ids: Sequence[str],
    well_ids: Sequence[str],
) -> tuple[Spectrum1D, ...]:
    return tuple(
        Spectrum1D(
            spectrum_id=f"d4::{record_id}",
            sample_id=str(well_id),
            axis_cm1=axis,
            intensity=np.asarray(row, dtype="<f8"),
        )
        for record_id, well_id, row in zip(record_ids, well_ids, matrix, strict=True)
    )


def _make_inputs(
    *,
    synthetic_fixture: bool,
    cohort_id: str,
    axis_cm1: np.ndarray,
    native_matrix: np.ndarray,
    targets: np.ndarray,
    record_ids: Sequence[str],
    well_ids: Sequence[str],
    rounds: np.ndarray,
    repetitions: np.ndarray,
    fold_by_well: Mapping[str, int],
    train_indices_by_fold: Mapping[int, np.ndarray],
    validation_indices_by_fold: Mapping[int, np.ndarray],
    test_indices_by_fold: Mapping[int, np.ndarray],
    cohort_identity: Mapping[str, object],
) -> DenoisingEvidenceInputs:
    axis = np.asarray(axis_cm1, dtype="<f8")
    native = np.ascontiguousarray(native_matrix, dtype="<f8")
    support = np.ascontiguousarray(native[:, 1:], dtype="<f8")
    target_matrix = np.ascontiguousarray(targets, dtype="<f8")
    record_ids_tuple = tuple(str(value) for value in record_ids)
    well_ids_tuple = tuple(str(value) for value in well_ids)
    unique_well_ids = _unique_wells(well_ids_tuple)
    fold_ids = tuple(sorted(int(value) for value in train_indices_by_fold))
    record_to_fold = {
        record_id: int(fold_by_well[well_id])
        for record_id, well_id in zip(record_ids_tuple, well_ids_tuple, strict=True)
    }
    return DenoisingEvidenceInputs(
        synthetic_fixture=synthetic_fixture,
        cohort_id=cohort_id,
        axis_cm1=axis,
        support_axis_cm1=axis[1:],
        native_matrix=native,
        support_matrix=support,
        targets=target_matrix,
        target_names=tuple(TARGET_NAMES),
        record_ids=record_ids_tuple,
        well_ids=well_ids_tuple,
        unique_well_ids=unique_well_ids,
        rounds=np.asarray(rounds, dtype=np.int64),
        repetitions=np.asarray(repetitions, dtype=np.int64),
        fold_ids=fold_ids,
        fold_by_well=dict(fold_by_well),
        record_to_fold=record_to_fold,
        train_indices_by_fold={int(key): np.asarray(value, dtype=np.int64) for key, value in train_indices_by_fold.items()},
        validation_indices_by_fold={int(key): np.asarray(value, dtype=np.int64) for key, value in validation_indices_by_fold.items()},
        test_indices_by_fold={int(key): np.asarray(value, dtype=np.int64) for key, value in test_indices_by_fold.items()},
        native_spectra=_spectra_from_matrix(axis, native, record_ids_tuple, well_ids_tuple),
        well_count=len(unique_well_ids),
        cohort_identity=cohort_identity,
    )


def _make_synthetic_denoising_evidence_inputs() -> DenoisingEvidenceInputs:
    axis = np.linspace(120.0, 4118.0, 2000, dtype="<f8")
    x = np.linspace(0.0, 1.0, axis.size, dtype="<f8")
    bases = np.stack(
        (
            np.exp(-((x - 0.15) / 0.04) ** 2),
            np.exp(-((x - 0.38) / 0.06) ** 2),
            np.exp(-((x - 0.61) / 0.05) ** 2),
            np.exp(-((x - 0.84) / 0.07) ** 2),
        ),
        axis=0,
    )
    background = 0.18 + 0.05 * np.sin(2.0 * np.pi * x) + 0.02 * x
    record_ids: list[str] = []
    well_ids: list[str] = []
    rounds: list[int] = []
    repetitions: list[int] = []
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    fold_by_well = {f"W{well_index:02d}": well_index % 5 for well_index in range(10)}
    for well_index in range(10):
        well_id = f"W{well_index:02d}"
        target = np.array(
            [
                0.32 * ((well_index % 5) / 4.0),
                0.32 * (((well_index * 2) % 5) / 4.0),
                0.32 * (((well_index * 3) % 5) / 4.0),
                0.32 * (((well_index * 4) % 5) / 4.0),
            ],
            dtype="<f8",
        )
        clean = background + np.sum(
            (1.2 + np.arange(4, dtype="<f8"))[:, None] * target[:, None] * bases,
            axis=0,
        )
        for source_round in range(1, 9):
            for repetition in range(1, 5):
                phase = 0.13 * well_index + 0.19 * source_round + 0.23 * repetition
                harmonic = 0.014 * np.sin(85.0 * x + phase) + 0.011 * np.cos(137.0 * x + 0.5 * phase)
                drift = 0.004 * (source_round - 4.5) * x + 0.003 * (repetition - 2.5) * (x - 0.5)
                rows.append(np.asarray(clean + harmonic + drift, dtype="<f8"))
                targets.append(target)
                record_ids.append(f"{well_id}_R{source_round}_M{repetition}")
                well_ids.append(well_id)
                rounds.append(source_round)
                repetitions.append(repetition)
    native_matrix = np.ascontiguousarray(rows, dtype="<f8")
    target_matrix = np.ascontiguousarray(targets, dtype="<f8")
    rounds_array = np.asarray(rounds, dtype=np.int64)
    repetitions_array = np.asarray(repetitions, dtype=np.int64)
    train_indices_by_fold: dict[int, np.ndarray] = {}
    validation_indices_by_fold: dict[int, np.ndarray] = {}
    test_indices_by_fold: dict[int, np.ndarray] = {}
    for fold in range(5):
        train_folds = tuple(candidate for candidate in range(5) if candidate not in {fold, (fold + 1) % 5})
        validation_fold = (fold + 1) % 5
        train_indices = [
            index
            for index, well_id in enumerate(well_ids)
            if fold_by_well[well_id] in train_folds
        ]
        validation_indices = [
            index
            for index, well_id in enumerate(well_ids)
            if fold_by_well[well_id] == validation_fold
        ]
        test_indices = [
            index
            for index, well_id in enumerate(well_ids)
            if fold_by_well[well_id] == fold
        ]
        train_indices_by_fold[fold] = np.asarray(train_indices, dtype=np.int64)
        validation_indices_by_fold[fold] = np.asarray(validation_indices, dtype=np.int64)
        test_indices_by_fold[fold] = np.asarray(test_indices, dtype=np.int64)
    identity = {
        "axis_sha256": _array_sha(axis),
        "support_axis_sha256": _array_sha(axis[1:]),
        "matrix_sha256": _array_sha(native_matrix),
        "support_matrix_sha256": _array_sha(native_matrix[:, 1:]),
        "targets_sha256": _array_sha(target_matrix),
        "record_ids_sha256": _ordered_sha(record_ids),
        "well_ids_sha256": _ordered_sha(_unique_wells(well_ids)),
    }
    return _make_inputs(
        synthetic_fixture=True,
        cohort_id="synthetic_d4_fixture_v1",
        axis_cm1=axis,
        native_matrix=native_matrix,
        targets=target_matrix,
        record_ids=record_ids,
        well_ids=well_ids,
        rounds=rounds_array,
        repetitions=repetitions_array,
        fold_by_well=fold_by_well,
        train_indices_by_fold=train_indices_by_fold,
        validation_indices_by_fold=validation_indices_by_fold,
        test_indices_by_fold=test_indices_by_fold,
        cohort_identity=identity,
    )


def _load_real_inputs(config: DenoisingEvidenceConfig, project_root: Path) -> DenoisingEvidenceInputs:
    cohort = load_d4_sugar_cohort(
        project_root / str(config.authorities["phase05_d4_protocol"]["path"]),
        project_root / str(config.authorities["source_archive"]["path"]),
    )
    axis = np.asarray(cohort.wavenumber, dtype="<f8")
    native_matrix = np.asarray(cohort.intensity, dtype="<f8")
    unique_wells = _unique_wells(tuple(str(value) for value in cohort.well_ids))
    fold_by_well: dict[str, int] = {}
    train_indices_by_fold: dict[int, np.ndarray] = {}
    validation_indices_by_fold: dict[int, np.ndarray] = {}
    test_indices_by_fold: dict[int, np.ndarray] = {}
    for split in cohort.splits:
        fold = int(split.test_fold)
        train_indices_by_fold[fold] = np.asarray(split.train_indices, dtype=np.int64)
        validation_indices_by_fold[fold] = np.asarray(split.validation_indices, dtype=np.int64)
        test_indices_by_fold[fold] = np.asarray(split.test_indices, dtype=np.int64)
        fold_wells = _unique_wells(tuple(str(cohort.well_ids[int(index)]) for index in split.test_indices))
        for well_id in fold_wells:
            fold_by_well[well_id] = fold
    identity = {
        "axis_sha256": _array_sha(axis),
        "support_axis_sha256": _array_sha(axis[1:]),
        "matrix_sha256": _array_sha(native_matrix),
        "support_matrix_sha256": _array_sha(native_matrix[:, 1:]),
        "targets_sha256": _array_sha(np.asarray(cohort.targets, dtype="<f8")),
        "record_ids_sha256": _ordered_sha(tuple(str(value) for value in cohort.record_ids)),
        "well_ids_sha256": _ordered_sha(unique_wells),
    }
    expected = config.cohort
    checks = {
        "record_ids_sha256": str(expected["mixture_record_ids_sha256"]),
        "axis_sha256": str(expected["native_axis_f64_sha256"]),
        "support_axis_sha256": str(expected["support_axis_f64_sha256"]),
        "targets_sha256": str(expected["target_matrix_sha256"]),
    }
    for key, value in checks.items():
        if identity[key] != value:
            raise DenoisingEvidenceError("cohort", f"{key} mismatch")
    return _make_inputs(
        synthetic_fixture=False,
        cohort_id="d4_low_snr_sugar",
        axis_cm1=axis,
        native_matrix=native_matrix,
        targets=np.asarray(cohort.targets, dtype="<f8"),
        record_ids=tuple(str(value) for value in cohort.record_ids),
        well_ids=tuple(str(value) for value in cohort.well_ids),
        rounds=np.asarray(cohort.rounds, dtype=np.int64),
        repetitions=np.asarray(cohort.repetitions, dtype=np.int64),
        fold_by_well=fold_by_well,
        train_indices_by_fold=train_indices_by_fold,
        validation_indices_by_fold=validation_indices_by_fold,
        test_indices_by_fold=test_indices_by_fold,
        cohort_identity=identity,
    )


def _validate_authorities(config: DenoisingEvidenceConfig, project_root: Path) -> None:
    for name, identity in config.authorities.items():
        path = project_root / str(identity["path"])
        if not path.is_file():
            raise DenoisingEvidenceError(name, "authority path is missing")
        if path.stat().st_size != int(identity["byte_count"]):
            raise DenoisingEvidenceError(name, "authority byte count mismatch")
        if _sha_file(path) != str(identity["sha256"]):
            raise DenoisingEvidenceError(name, "authority SHA256 mismatch")


def _systems_from_catalog(
    config: DenoisingEvidenceConfig,
    system_ids: Sequence[str],
    project_root: Path,
) -> tuple[Phase3System, ...]:
    catalog = load_classical_catalog(project_root / str(config.authorities["classical_catalog"]["path"]))
    by_id = {system.system_id: system for system in catalog.systems}
    systems: list[Phase3System] = []
    for system_id in system_ids:
        system = by_id.get(system_id)
        if system is None:
            raise DenoisingEvidenceVerificationError("artifact system IDs are not in the frozen catalog")
        if system.task_line is not TaskLine.DENOISING:
            raise DenoisingEvidenceVerificationError(f"{system_id} is not a denoising system")
        systems.append(system)
    return tuple(systems)


def _assert_exact_system_status_catalog(
    *,
    config: DenoisingEvidenceConfig,
    system_rows: Sequence[Mapping[str, object]],
    project_root: Path,
) -> tuple[Phase3System, ...]:
    observed_system_ids = tuple(str(row.get("system_id", "")) for row in system_rows)
    if observed_system_ids != tuple(config.system_ids):
        raise DenoisingEvidenceVerificationError(
            "system_status rows must match config system IDs and frozen catalog order"
        )
    expected_row_count = int(config.expected["system_status_row_count"])
    if len(system_rows) != expected_row_count:
        raise DenoisingEvidenceVerificationError(
            "system_status row count does not match config.system_ids/system_status expectation"
        )
    return _systems_from_catalog(config, config.system_ids, project_root)


def _code_authority(project_root: Path) -> Mapping[str, Mapping[str, object]]:
    paths = (
        "experiments/phase6/configs/denoising_evidence_v1.json",
        "rpe/methods/classical/denoising.py",
        "rpe/metrics/reference_free.py",
        "rpe/metrics/consistency.py",
        "rpe/runner/phase6_denoising_evidence.py",
        "rpe/runner/phase6_denoising_evidence_verifier.py",
        "tools/run_phase6_denoising_evidence.py",
    )
    return {
        relative: {
            "byte_count": (project_root / relative).stat().st_size,
            "sha256": _sha_file(project_root / relative),
        }
        for relative in paths
        if (project_root / relative).is_file()
    }


def _pls_state_digest(estimator: PLSRegression, fold: int, selected_n_components: int) -> str:
    state_arrays = (
        estimator._x_mean,
        estimator._y_mean,
        estimator._x_std,
        estimator._y_std,
        estimator.x_weights_,
        estimator.y_weights_,
        estimator.x_loadings_,
        estimator.y_loadings_,
        estimator.x_rotations_,
        estimator.y_rotations_,
        estimator.coef_,
        estimator.intercept_,
    )
    payload = {
        "fold": fold,
        "selected_n_components": selected_n_components,
        "state_sha256": [_array_sha(np.asarray(value, dtype="<f8")) for value in state_arrays],
        "n_iter": [int(value) for value in estimator.n_iter_],
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _fit_selected_pls(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    n_components_grid: Sequence[int],
    fold: int,
) -> tuple[PLSRegression, tuple[Mapping[str, object], ...], int, str]:
    candidates: list[tuple[tuple[int, float, int], PLSRegression]] = []
    validation_rows: list[Mapping[str, object]] = []
    for n_components in n_components_grid:
        if n_components < 1 or n_components > min(x_train.shape[0] - 1, x_train.shape[1]):
            validation_rows.append(
                {
                    "fold": fold,
                    "n_components": int(n_components),
                    "macro_normalized_rmse": None,
                    "state": "not_evaluable_component_domain",
                }
            )
            continue
        try:
            estimator = PLSRegression(
                n_components=int(n_components),
                scale=True,
                max_iter=500,
                tol=1e-6,
                copy=True,
            )
            warning_messages: list[warnings.WarningMessage] = []
            with warnings.catch_warnings(record=True) as warning_messages:
                warnings.simplefilter("always")
                with threadpool_limits(limits=1, user_api="blas"):
                    estimator.fit(x_train, y_train)
                    predicted = np.asarray(estimator.predict(x_validation), dtype="<f8")
            if not np.isfinite(predicted).all():
                raise FloatingPointError("nonfinite validation prediction")
        except (Warning, ValueError, ArithmeticError, FloatingPointError, StopIteration) as error:
            validation_rows.append(
                {
                    "fold": fold,
                    "n_components": int(n_components),
                    "macro_normalized_rmse": None,
                    "state": "failed_model_lifecycle",
                    "failure_category": type(error).__name__,
                    "failure_message": str(error) or type(error).__name__,
                }
            )
            continue
        score = float(np.mean(np.sqrt(np.mean((predicted - y_validation) ** 2, axis=0)) / 0.32))
        validation_rows.append(
            {
                "fold": fold,
                "n_components": int(n_components),
                "macro_normalized_rmse": score,
                "state": "complete",
                "warning_category": (
                    None
                    if not warning_messages
                    else type(warning_messages[0].message).__name__
                ),
                "warning_message": (
                    None if not warning_messages else str(warning_messages[0].message)
                ),
            }
        )
        candidates.append(((1 if warning_messages else 0, score, int(n_components)), estimator))
    if not candidates:
        raise DenoisingEvidenceError("model selection", f"no valid components for fold {fold}")
    (warning_rank, _score, selected), estimator = min(candidates, key=lambda item: item[0])
    del warning_rank
    return estimator, tuple(validation_rows), selected, _pls_state_digest(estimator, fold, selected)


def _validation_rows_complete(
    validation_rows: Sequence[Mapping[str, object]],
    n_components_grid: Sequence[int],
) -> bool:
    return len(validation_rows) == len(tuple(n_components_grid)) and all(
        row.get("state") == "complete" for row in validation_rows
    )


def _validation_scores_sha256(validation_rows: Sequence[Mapping[str, object]]) -> str:
    normalized_rows = []
    for row in validation_rows:
        value = row.get("macro_normalized_rmse")
        normalized_rows.append(
            {
                "fold": int(row["fold"]),
                "n_components": int(row["n_components"]),
                "macro_normalized_rmse": (
                    None
                    if value is None
                    else float(format(float(value), ".15g"))
                ),
                "state": str(row["state"]),
                "failure_category": row.get("failure_category"),
                "warning_category": row.get("warning_category"),
            }
        )
    return hashlib.sha256(_jsonl_bytes(tuple(normalized_rows))).hexdigest()


def _well_record_indexes(inputs: DenoisingEvidenceInputs, well_id: str) -> tuple[int, ...]:
    return tuple(index for index, current in enumerate(inputs.well_ids) if current == well_id)


def _mean_spectrum(spectrum_id: str, axis_cm1: np.ndarray, intensity: np.ndarray) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id=spectrum_id,
        axis_cm1=np.asarray(axis_cm1, dtype="<f8"),
        intensity=np.asarray(intensity, dtype="<f8"),
    )


def _reference_free_well_value(inputs: DenoisingEvidenceInputs, output_rows: np.ndarray) -> float:
    metric = ISLikeStructureToNoiseMetric()
    values = []
    for order, row in enumerate(output_rows):
        value = metric.evaluate(
            SingleSpectrumInput(
                _mean_spectrum(
                    f"reference-free-{order}",
                    inputs.axis_cm1,
                    np.asarray(row, dtype="<f8"),
                )
            )
        ).outputs[0].value
        values.append(float(value))
    return float(np.mean(values))


def _half_split_well_value(inputs: DenoisingEvidenceInputs, well_id: str, output_rows: np.ndarray) -> float:
    metric = HalfSplitPearsonConsistencyMetric()
    indexes = _well_record_indexes(inputs, well_id)
    order = sorted(
        range(len(indexes)),
        key=lambda current: (
            int(inputs.rounds[indexes[current]]),
            int(inputs.repetitions[indexes[current]]),
            str(inputs.record_ids[indexes[current]]),
        ),
    )
    even_rows = np.asarray(output_rows[order[0::2]], dtype="<f8")
    odd_rows = np.asarray(output_rows[order[1::2]], dtype="<f8")
    left = _mean_spectrum(f"{well_id}-even", inputs.axis_cm1, np.mean(even_rows, axis=0))
    right = _mean_spectrum(f"{well_id}-odd", inputs.axis_cm1, np.mean(odd_rows, axis=0))
    return float(metric.evaluate(ReplicatePairInput(left, right)).outputs[0].value)


def _closed_downstream_rows(
    *,
    inputs: DenoisingEvidenceInputs,
    system: Phase3System,
    downstream_rows: Sequence[Mapping[str, object]],
    state: str,
) -> list[Mapping[str, object]]:
    existing = {
        (int(row["fold"]), str(row["protocol_id"]), str(row["well_id"]))
        for row in downstream_rows
    }
    closed = list(downstream_rows)
    for well_id in inputs.unique_well_ids:
        fold = int(inputs.fold_by_well[well_id])
        for protocol_id in ("D4-denoise-A", "D4-denoise-B"):
            key = (fold, protocol_id, well_id)
            if key in existing:
                continue
            closed.append(
                {
                    "fold": fold,
                    "protocol_id": protocol_id,
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "well_id": well_id,
                    "identity_loss": None,
                    "candidate_loss": None,
                    "effect": None,
                    "state": state,
                }
            )
    return closed


def _closed_reference_rows(
    *,
    inputs: DenoisingEvidenceInputs,
    system: Phase3System,
    identity_reference_free: Mapping[str, float],
    state: str,
) -> list[Mapping[str, object]]:
    return [
        {
            "system_id": system.system_id,
            "family_id": system.family_id,
            "method_id": system.method_id,
            "well_id": well_id,
            "identity_value": identity_reference_free[well_id],
            "candidate_value": None,
            "effect": None,
            "state": state,
        }
        for well_id in inputs.unique_well_ids
    ]


def _closed_half_split_rows(
    *,
    inputs: DenoisingEvidenceInputs,
    system: Phase3System,
    identity_half_split: Mapping[str, float],
    state: str,
) -> list[Mapping[str, object]]:
    return [
        {
            "system_id": system.system_id,
            "family_id": system.family_id,
            "method_id": system.method_id,
            "well_id": well_id,
            "identity_value": identity_half_split[well_id],
            "candidate_value": None,
            "effect": None,
            "state": state,
        }
        for well_id in inputs.unique_well_ids
    ]


def _identity_models(
    inputs: DenoisingEvidenceInputs,
    config: DenoisingEvidenceConfig,
) -> tuple[
    Mapping[int, Mapping[str, object]],
    Mapping[str, np.ndarray],
    Mapping[str, float],
    Mapping[str, float],
    Mapping[str, float],
    Mapping[str, object],
]:
    models: dict[int, Mapping[str, object]] = {}
    predictions_by_record: dict[str, np.ndarray] = {}
    losses_by_well: dict[str, float] = {}
    reference_by_well: dict[str, float] = {}
    half_split_by_well: dict[str, float] = {}
    for fold in inputs.fold_ids:
        train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
        validation = np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)
        test = np.asarray(inputs.test_indices_by_fold[fold], dtype=np.int64)
        estimator, validation_rows, selected, model_digest = _fit_selected_pls(
            inputs.support_matrix[train],
            inputs.targets[train],
            inputs.support_matrix[validation],
            inputs.targets[validation],
            tuple(int(value) for value in config.model_recipe["n_components_grid"]),
            fold,
        )
        if not _validation_rows_complete(
            validation_rows,
            tuple(int(value) for value in config.model_recipe["n_components_grid"]),
        ):
            raise DenoisingEvidenceError(
                "identity_preflight_grid",
                f"fold {fold} did not complete the frozen five-candidate selection grid",
            )
        with threadpool_limits(limits=1, user_api="blas"):
            predicted = np.asarray(estimator.predict(inputs.support_matrix[test]), dtype="<f8")
        for record_index, value in zip(test.tolist(), predicted, strict=True):
            predictions_by_record[inputs.record_ids[record_index]] = np.asarray(value, dtype="<f8")
        models[fold] = {
            "fold": fold,
            "selected_n_components": selected,
            "model_state_digest": model_digest,
            "estimator": estimator,
            "validation_scores": validation_rows,
        }
    for well_id in inputs.unique_well_ids:
        indexes = _well_record_indexes(inputs, well_id)
        predictions = np.asarray([predictions_by_record[inputs.record_ids[index]] for index in indexes], dtype="<f8")
        targets = np.asarray(inputs.targets[list(indexes)], dtype="<f8")
        losses_by_well[well_id] = float(np.mean(((predictions - targets) ** 2) / (0.32 ** 2)))
        reference_by_well[well_id] = _reference_free_well_value(inputs, inputs.native_matrix[list(indexes)])
        half_split_by_well[well_id] = _half_split_well_value(
            inputs,
            well_id,
            np.asarray(inputs.native_matrix[list(indexes)], dtype="<f8"),
        )
    prediction_rows = tuple(
        {
            "fold": int(inputs.record_to_fold[record_id]),
            "predicted_targets": [float(item) for item in predictions_by_record[record_id]],
            "record_id": record_id,
            "well_id": inputs.well_ids[inputs.record_ids.index(record_id)],
        }
        for record_id in inputs.record_ids
    )
    model_digest_rows = tuple(
        {
            "fold": int(fold),
            "model_state_digest": str(models[fold]["model_state_digest"]),
            "selected_n_components": int(models[fold]["selected_n_components"]),
            "validation_scores_sha256": _validation_scores_sha256(
                tuple(models[fold]["validation_scores"])
            ),
        }
        for fold in sorted(models)
    )
    return models, predictions_by_record, losses_by_well, reference_by_well, half_split_by_well, {
        "prediction_sha256": hashlib.sha256(_jsonl_bytes(prediction_rows)).hexdigest(),
        "model_digest_sha256": hashlib.sha256(_jsonl_bytes(model_digest_rows)).hexdigest(),
    }


def _evaluate_system(
    *,
    inputs: DenoisingEvidenceInputs,
    config: DenoisingEvidenceConfig,
    system: Phase3System,
    identity_models: Mapping[int, Mapping[str, object]],
    identity_predictions: Mapping[str, np.ndarray],
    identity_losses: Mapping[str, float],
    identity_reference_free: Mapping[str, float],
    identity_half_split: Mapping[str, float],
) -> tuple[
    Mapping[str, object],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
    list[Mapping[str, object]],
]:
    fit_receipts: list[Mapping[str, object]] = []
    transform_receipts: list[Mapping[str, object]] = []
    model_receipts: list[Mapping[str, object]] = []
    downstream_rows: list[Mapping[str, object]] = []
    reference_rows: list[Mapping[str, object]] = []
    half_split_rows: list[Mapping[str, object]] = []
    n_components_grid = tuple(int(value) for value in config.model_recipe["n_components_grid"])
    protocol_b_reason = ""
    if system.family_id in {"savitzky_golay", "wavelet", "whittaker_smoothing"}:
        transformed = np.empty_like(inputs.native_matrix, dtype="<f8")
        failed: DenoisingRunResult | None = None
        protocol_b_state = "complete"
        for index, spectrum in enumerate(inputs.native_spectra):
            result = run_stateless_denoising_system(system, spectrum)
            transform_receipts.append(
                {
                    "fold": None,
                    "source_role": "all_roles",
                    "record_id": inputs.record_ids[index],
                    "well_id": inputs.well_ids[index],
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "status": result.status.value,
                    "output_sha256": result.output_sha256,
                }
            )
            if result.status not in SUCCESS or result.denoised_intensity is None:
                failed = result
                break
            transformed[index] = np.asarray(result.denoised_intensity, dtype="<f8")
        if failed is None:
            for fold in inputs.fold_ids:
                test = np.asarray(inputs.test_indices_by_fold[fold], dtype=np.int64)
                with threadpool_limits(limits=1, user_api="blas"):
                    a_predictions = np.asarray(
                        identity_models[fold]["estimator"].predict(transformed[test, 1:]),
                        dtype="<f8",
                    )
                b_predictions: np.ndarray | None = None
                try:
                    estimator, validation_rows, selected, model_digest = _fit_selected_pls(
                        transformed[np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64), 1:],
                        inputs.targets[np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)],
                        transformed[np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64), 1:],
                        inputs.targets[np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)],
                        n_components_grid,
                        int(fold),
                    )
                    if not _validation_rows_complete(validation_rows, n_components_grid):
                        raise DenoisingEvidenceError(
                            "protocol_b_grid",
                            f"fold {fold} did not complete the frozen five-candidate selection grid",
                        )
                    with threadpool_limits(limits=1, user_api="blas"):
                        b_predictions = np.asarray(estimator.predict(transformed[test, 1:]), dtype="<f8")
                    model_receipts.append(
                        {
                            "fold": int(fold),
                            "protocol_id": "D4-denoise-B",
                            "system_id": system.system_id,
                            "selected_n_components": selected,
                            "model_state_digest": model_digest,
                            "identity_model_state_digest": identity_models[fold]["model_state_digest"],
                            "validation_scores_sha256": _validation_scores_sha256(validation_rows),
                            "status": "complete",
                        }
                    )
                except DenoisingEvidenceError as error:
                    protocol_b_state = f"not_evaluable_protocol_b_{error.path}"
                    if not protocol_b_reason:
                        protocol_b_reason = error.path
                    model_receipts.append(
                        {
                            "fold": int(fold),
                            "protocol_id": "D4-denoise-B",
                            "system_id": system.system_id,
                            "selected_n_components": None,
                            "model_state_digest": None,
                            "identity_model_state_digest": identity_models[fold]["model_state_digest"],
                            "validation_scores_sha256": None,
                            "status": protocol_b_state,
                            "reason_code": error.path,
                        }
                    )
                test_list = test.tolist()
                for well_id in _unique_wells(tuple(inputs.well_ids[index] for index in test_list)):
                    indexes = tuple(index for index in test_list if inputs.well_ids[index] == well_id)
                    targets = np.asarray(inputs.targets[list(indexes)], dtype="<f8")
                    identity = np.asarray([identity_predictions[inputs.record_ids[index]] for index in indexes], dtype="<f8")
                    candidate_a = np.asarray([a_predictions[test_list.index(index)] for index in indexes], dtype="<f8")
                    identity_loss = float(np.mean(((identity - targets) ** 2) / (0.32 ** 2)))
                    candidate_a_loss = float(np.mean(((candidate_a - targets) ** 2) / (0.32 ** 2)))
                    downstream_rows.append(
                        {
                            "fold": int(fold),
                            "protocol_id": "D4-denoise-A",
                            "system_id": system.system_id,
                            "family_id": system.family_id,
                            "method_id": system.method_id,
                            "well_id": well_id,
                            "identity_loss": identity_loss,
                            "candidate_loss": candidate_a_loss,
                            "effect": candidate_a_loss - identity_loss,
                            "state": "complete",
                        }
                    )
                    if b_predictions is None:
                        downstream_rows.append(
                            {
                                "fold": int(fold),
                                "protocol_id": "D4-denoise-B",
                                "system_id": system.system_id,
                                "family_id": system.family_id,
                                "method_id": system.method_id,
                                "well_id": well_id,
                                "identity_loss": identity_loss,
                                "candidate_loss": None,
                                "effect": None,
                                "state": protocol_b_state,
                            }
                        )
                    else:
                        candidate_b = np.asarray([b_predictions[test_list.index(index)] for index in indexes], dtype="<f8")
                        candidate_b_loss = float(np.mean(((candidate_b - targets) ** 2) / (0.32 ** 2)))
                        downstream_rows.append(
                            {
                                "fold": int(fold),
                                "protocol_id": "D4-denoise-B",
                                "system_id": system.system_id,
                                "family_id": system.family_id,
                                "method_id": system.method_id,
                                "well_id": well_id,
                                "identity_loss": identity_loss,
                                "candidate_loss": candidate_b_loss,
                                "effect": candidate_b_loss - identity_loss,
                                "state": "complete",
                            }
                        )
            try:
                for well_id in inputs.unique_well_ids:
                    indexes = _well_record_indexes(inputs, well_id)
                    candidate_reference = _reference_free_well_value(inputs, transformed[list(indexes)])
                    candidate_half_split = _half_split_well_value(inputs, well_id, transformed[list(indexes)])
                    reference_rows.append(
                        {
                            "system_id": system.system_id,
                            "family_id": system.family_id,
                            "method_id": system.method_id,
                            "well_id": well_id,
                            "identity_value": identity_reference_free[well_id],
                            "candidate_value": candidate_reference,
                            "effect": candidate_reference - identity_reference_free[well_id],
                            "state": "complete",
                        }
                    )
                    half_split_rows.append(
                        {
                            "system_id": system.system_id,
                            "family_id": system.family_id,
                            "method_id": system.method_id,
                            "well_id": well_id,
                            "identity_value": identity_half_split[well_id],
                            "candidate_value": candidate_half_split,
                            "effect": candidate_half_split - identity_half_split[well_id],
                            "state": "complete",
                        }
                    )
                status = {
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "status": "complete" if protocol_b_reason == "" else "complete_with_protocol_b_closure",
                    "reason_code": protocol_b_reason,
                    "fit_receipt_count": 0,
                    "transform_receipt_count": len(transform_receipts),
                    "model_receipt_count": len(model_receipts),
                }
                return status, fit_receipts, transform_receipts, model_receipts, downstream_rows, reference_rows + half_split_rows
            except Exception as error:
                status = {
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "status": "failed_consumer_closure",
                    "reason_code": f"consumer:{type(error).__name__}",
                    "fit_receipt_count": 0,
                    "transform_receipt_count": len(transform_receipts),
                    "model_receipt_count": len(model_receipts),
                }
                reference_rows = _closed_reference_rows(
                    inputs=inputs,
                    system=system,
                    identity_reference_free=identity_reference_free,
                    state="not_evaluable_incomplete_system_grid",
                )
                half_split_rows = _closed_half_split_rows(
                    inputs=inputs,
                    system=system,
                    identity_half_split=identity_half_split,
                    state="not_evaluable_incomplete_system_grid",
                )
                return status, fit_receipts, transform_receipts, model_receipts, downstream_rows, reference_rows + half_split_rows
        status_code = failed.status.value if failed is not None else "failed_runtime"
        reason = failed.error_code if failed is not None else "failed_runtime"
    else:
        test_outputs: dict[str, np.ndarray] = {}
        reason = "not_evaluable_fit_failure"
        status_code = "failed_fit"
        protocol_b_reason = ""
        for fold in inputs.fold_ids:
            train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
            validation = np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)
            test = np.asarray(inputs.test_indices_by_fold[fold], dtype=np.int64)
            try:
                with threadpool_limits(limits=1, user_api="blas"):
                    fitted = fit_denoising_system(
                        system,
                        tuple(inputs.native_spectra[index] for index in train.tolist()),
                        DenoisingFitContext(
                            split_id=f"d4_fold_{fold}",
                            representation_id="d4_native_float64_v1",
                            record_ids=tuple(inputs.native_spectra[index].spectrum_id for index in train.tolist()),
                        ),
                    )
            except DenoisingFitError as error:
                fit_receipts.append(
                    {
                        "fold": int(fold),
                        "system_id": system.system_id,
                        "family_id": system.family_id,
                        "method_id": system.method_id,
                        "status": error.status.value,
                        "fitted_state_sha256": None,
                        "train_well_ids": list(_unique_wells(tuple(inputs.well_ids[index] for index in train.tolist()))),
                        "validation_well_ids": list(_unique_wells(tuple(inputs.well_ids[index] for index in validation.tolist()))),
                        "test_well_ids": list(_unique_wells(tuple(inputs.well_ids[index] for index in test.tolist()))),
                        "reason_code": error.path,
                    }
                )
                status_code = error.status.value
                reason = error.path
                break
            fit_receipts.append(
                {
                    "fold": int(fold),
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "status": "complete",
                    "fitted_state_sha256": fitted.state_sha256,
                    "train_well_ids": list(_unique_wells(tuple(inputs.well_ids[index] for index in train.tolist()))),
                    "validation_well_ids": list(_unique_wells(tuple(inputs.well_ids[index] for index in validation.tolist()))),
                    "test_well_ids": list(_unique_wells(tuple(inputs.well_ids[index] for index in test.tolist()))),
                    "reason_code": "",
                }
            )
            fold_outputs = np.empty_like(inputs.native_matrix, dtype="<f8")
            failed_result: DenoisingRunResult | None = None
            train_set = set(int(value) for value in train.tolist())
            validation_set = set(int(value) for value in validation.tolist())
            for index, spectrum in enumerate(inputs.native_spectra):
                with threadpool_limits(limits=1, user_api="blas"):
                    result = transform_fitted_denoiser(fitted, spectrum)
                role = "train" if index in train_set else "validation" if index in validation_set else "test"
                transform_receipts.append(
                    {
                        "fold": int(fold),
                        "source_role": role,
                        "record_id": inputs.record_ids[index],
                        "well_id": inputs.well_ids[index],
                        "system_id": system.system_id,
                        "family_id": system.family_id,
                        "method_id": system.method_id,
                        "status": result.status.value,
                        "output_sha256": result.output_sha256,
                        "fitted_state_sha256": fitted.state_sha256,
                    }
                )
                if result.status not in SUCCESS or result.denoised_intensity is None:
                    failed_result = result
                    break
                fold_outputs[index] = np.asarray(result.denoised_intensity, dtype="<f8")
            if failed_result is not None:
                status_code = failed_result.status.value
                reason = failed_result.error_code or failed_result.status.value
                break
            with threadpool_limits(limits=1, user_api="blas"):
                a_predictions = np.asarray(
                    identity_models[fold]["estimator"].predict(fold_outputs[test, 1:]),
                    dtype="<f8",
                )
            b_predictions: np.ndarray | None = None
            protocol_b_state = "complete"
            try:
                estimator, validation_rows, selected, model_digest = _fit_selected_pls(
                    fold_outputs[train, 1:],
                    inputs.targets[train],
                    fold_outputs[validation, 1:],
                    inputs.targets[validation],
                    n_components_grid,
                    int(fold),
                )
                if not _validation_rows_complete(validation_rows, n_components_grid):
                    raise DenoisingEvidenceError(
                        "protocol_b_grid",
                        f"fold {fold} did not complete the frozen five-candidate selection grid",
                    )
                with threadpool_limits(limits=1, user_api="blas"):
                    b_predictions = np.asarray(estimator.predict(fold_outputs[test, 1:]), dtype="<f8")
                model_receipts.append(
                    {
                        "fold": int(fold),
                        "protocol_id": "D4-denoise-B",
                        "system_id": system.system_id,
                        "selected_n_components": selected,
                        "model_state_digest": model_digest,
                        "identity_model_state_digest": identity_models[fold]["model_state_digest"],
                        "validation_scores_sha256": _validation_scores_sha256(validation_rows),
                        "status": "complete",
                    }
                )
            except DenoisingEvidenceError as error:
                protocol_b_state = f"not_evaluable_protocol_b_{error.path}"
                if not protocol_b_reason:
                    protocol_b_reason = error.path
                model_receipts.append(
                    {
                        "fold": int(fold),
                        "protocol_id": "D4-denoise-B",
                        "system_id": system.system_id,
                        "selected_n_components": None,
                        "model_state_digest": None,
                        "identity_model_state_digest": identity_models[fold]["model_state_digest"],
                        "validation_scores_sha256": None,
                        "status": protocol_b_state,
                        "reason_code": error.path,
                    }
                )
            test_list = test.tolist()
            for offset, index in enumerate(test.tolist()):
                test_outputs[inputs.record_ids[index]] = np.asarray(fold_outputs[index], dtype="<f8")
            for well_id in _unique_wells(tuple(inputs.well_ids[index] for index in test_list)):
                indexes = tuple(index for index in test_list if inputs.well_ids[index] == well_id)
                targets = np.asarray(inputs.targets[list(indexes)], dtype="<f8")
                identity = np.asarray([identity_predictions[inputs.record_ids[index]] for index in indexes], dtype="<f8")
                candidate_a = np.asarray([a_predictions[test_list.index(index)] for index in indexes], dtype="<f8")
                identity_loss = float(np.mean(((identity - targets) ** 2) / (0.32 ** 2)))
                candidate_a_loss = float(np.mean(((candidate_a - targets) ** 2) / (0.32 ** 2)))
                downstream_rows.append(
                    {
                        "fold": int(fold),
                        "protocol_id": "D4-denoise-A",
                        "system_id": system.system_id,
                        "family_id": system.family_id,
                        "method_id": system.method_id,
                        "well_id": well_id,
                        "identity_loss": identity_loss,
                        "candidate_loss": candidate_a_loss,
                        "effect": candidate_a_loss - identity_loss,
                        "state": "complete",
                    }
                )
                if b_predictions is None:
                    downstream_rows.append(
                        {
                            "fold": int(fold),
                            "protocol_id": "D4-denoise-B",
                            "system_id": system.system_id,
                            "family_id": system.family_id,
                            "method_id": system.method_id,
                            "well_id": well_id,
                            "identity_loss": identity_loss,
                            "candidate_loss": None,
                            "effect": None,
                            "state": protocol_b_state,
                        }
                    )
                else:
                    candidate_b = np.asarray([b_predictions[test_list.index(index)] for index in indexes], dtype="<f8")
                    candidate_b_loss = float(np.mean(((candidate_b - targets) ** 2) / (0.32 ** 2)))
                    downstream_rows.append(
                        {
                            "fold": int(fold),
                            "protocol_id": "D4-denoise-B",
                            "system_id": system.system_id,
                            "family_id": system.family_id,
                            "method_id": system.method_id,
                            "well_id": well_id,
                            "identity_loss": identity_loss,
                            "candidate_loss": candidate_b_loss,
                            "effect": candidate_b_loss - identity_loss,
                            "state": "complete",
                        }
                    )
        else:
            try:
                for well_id in inputs.unique_well_ids:
                    indexes = _well_record_indexes(inputs, well_id)
                    candidate_rows = np.asarray([test_outputs[inputs.record_ids[index]] for index in indexes], dtype="<f8")
                    candidate_reference = _reference_free_well_value(inputs, candidate_rows)
                    candidate_half_split = _half_split_well_value(inputs, well_id, candidate_rows)
                    reference_rows.append(
                        {
                            "system_id": system.system_id,
                            "family_id": system.family_id,
                            "method_id": system.method_id,
                            "well_id": well_id,
                            "identity_value": identity_reference_free[well_id],
                            "candidate_value": candidate_reference,
                            "effect": candidate_reference - identity_reference_free[well_id],
                            "state": "complete",
                        }
                    )
                    half_split_rows.append(
                        {
                            "system_id": system.system_id,
                            "family_id": system.family_id,
                            "method_id": system.method_id,
                            "well_id": well_id,
                            "identity_value": identity_half_split[well_id],
                            "candidate_value": candidate_half_split,
                            "effect": candidate_half_split - identity_half_split[well_id],
                            "state": "complete",
                        }
                    )
                status = {
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "status": "complete" if protocol_b_reason == "" else "complete_with_protocol_b_closure",
                    "reason_code": protocol_b_reason,
                    "fit_receipt_count": len(fit_receipts),
                    "transform_receipt_count": len(transform_receipts),
                    "model_receipt_count": len(model_receipts),
                }
                return status, fit_receipts, transform_receipts, model_receipts, downstream_rows, reference_rows + half_split_rows
            except Exception as error:
                status = {
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "status": "failed_consumer_closure",
                    "reason_code": f"consumer:{type(error).__name__}",
                    "fit_receipt_count": len(fit_receipts),
                    "transform_receipt_count": len(transform_receipts),
                    "model_receipt_count": len(model_receipts),
                }
                reference_rows = _closed_reference_rows(
                    inputs=inputs,
                    system=system,
                    identity_reference_free=identity_reference_free,
                    state="not_evaluable_incomplete_system_grid",
                )
                half_split_rows = _closed_half_split_rows(
                    inputs=inputs,
                    system=system,
                    identity_half_split=identity_half_split,
                    state="not_evaluable_incomplete_system_grid",
                )
                return status, fit_receipts, transform_receipts, model_receipts, downstream_rows, reference_rows + half_split_rows
    status = {
        "system_id": system.system_id,
        "family_id": system.family_id,
        "method_id": system.method_id,
        "status": status_code,
        "reason_code": reason,
        "fit_receipt_count": len(fit_receipts),
        "transform_receipt_count": len(transform_receipts),
        "model_receipt_count": len(model_receipts),
    }
    downstream_rows = _closed_downstream_rows(
        inputs=inputs,
        system=system,
        downstream_rows=downstream_rows,
        state="not_evaluable_incomplete_system_grid",
    )
    reference_rows = _closed_reference_rows(
        inputs=inputs,
        system=system,
        identity_reference_free=identity_reference_free,
        state="not_evaluable_incomplete_system_grid",
    )
    half_split_rows = _closed_half_split_rows(
        inputs=inputs,
        system=system,
        identity_half_split=identity_half_split,
        state="not_evaluable_incomplete_system_grid",
    )
    return status, fit_receipts, transform_receipts, model_receipts, downstream_rows, reference_rows + half_split_rows


def _bootstrap_rows(
    *,
    config: DenoisingEvidenceConfig,
    systems: Sequence[Phase3System],
    downstream_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    half_split_rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    rng = np.random.default_rng(int(config.bootstrap["seed"]))
    rows: list[Mapping[str, object]] = []
    well_count = len({str(row["well_id"]) for row in downstream_rows})
    sample_indexes = rng.integers(0, well_count, size=(int(config.bootstrap["resamples"]), well_count))
    for system in systems:
        for endpoint_id, protocol_id, source_rows in (
            ("D4-denoise-A", "D4-denoise-A", [row for row in downstream_rows if row["system_id"] == system.system_id and row["protocol_id"] == "D4-denoise-A"]),
            ("D4-denoise-B", "D4-denoise-B", [row for row in downstream_rows if row["system_id"] == system.system_id and row["protocol_id"] == "D4-denoise-B"]),
            ("D4-reference-free", "method_native", [row for row in reference_rows if row["system_id"] == system.system_id]),
            ("D4-half-split", "method_native", [row for row in half_split_rows if row["system_id"] == system.system_id]),
        ):
            values = [row["effect"] for row in source_rows if row["effect"] is not None]
            if len(values) != well_count:
                rows.append(
                    {
                        "system_id": system.system_id,
                        "family_id": system.family_id,
                        "method_id": system.method_id,
                        "endpoint_id": endpoint_id,
                        "protocol_id": protocol_id,
                        "estimate": None,
                        "interval_lower": None,
                        "interval_upper": None,
                        "cluster_count": well_count,
                        "resamples": int(config.bootstrap["resamples"]),
                        "seed": int(config.bootstrap["seed"]),
                        "state": "not_evaluable_incomplete_system_grid",
                    }
                )
                continue
            array = np.asarray(values, dtype="<f8")
            bootstrap_estimates = np.mean(array[sample_indexes], axis=1)
            rows.append(
                {
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "endpoint_id": endpoint_id,
                    "protocol_id": protocol_id,
                    "estimate": float(np.mean(array)),
                    "interval_lower": float(np.quantile(bootstrap_estimates, 0.025)),
                    "interval_upper": float(np.quantile(bootstrap_estimates, 0.975)),
                    "cluster_count": well_count,
                    "resamples": int(config.bootstrap["resamples"]),
                    "seed": int(config.bootstrap["seed"]),
                    "state": "complete",
                }
            )
    return rows


def _method_evidence_rows(
    *,
    config: DenoisingEvidenceConfig,
    systems: Sequence[Phase3System],
    system_status_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    status_by_system = {str(row["system_id"]): row for row in system_status_rows}
    bootstrap_by_key = {
        (str(row["system_id"]), str(row["endpoint_id"])): row for row in bootstrap_rows
    }
    rows: list[Mapping[str, object]] = []
    for system in systems:
        status = status_by_system[system.system_id]
        for endpoint in config.endpoint_manifest:
            endpoint_id = str(endpoint["endpoint_id"])
            bootstrap = bootstrap_by_key.get((system.system_id, endpoint_id))
            if endpoint_id == "availability":
                rows.append(
                    {
                        "evidence_id": f"{system.system_id}:{endpoint_id}",
                        "task_line": "denoising",
                        "system_id": system.system_id,
                        "family_id": system.family_id,
                        "endpoint_id": endpoint_id,
                        "protocol_id": endpoint["protocol_id"],
                        "cohort_id": "d4_low_snr_sugar",
                        "evidence_component": endpoint["evidence_component"],
                        "metric_output_id": endpoint["metric_output_id"],
                        "estimate": None,
                        "interval_lower": None,
                        "interval_upper": None,
                        "preferred_direction": endpoint["preferred_direction"],
                        "state": status["status"],
                        "reason_code": status["reason_code"],
                        "phase5_power_state": "not_powered_for_phase5",
                        "source_path": "system_status.jsonl",
                        "source_sha256": None,
                    }
                )
                continue
            if endpoint_id == "direct_gt":
                rows.append(
                    {
                        "evidence_id": f"{system.system_id}:{endpoint_id}",
                        "task_line": "denoising",
                        "system_id": system.system_id,
                        "family_id": system.family_id,
                        "endpoint_id": endpoint_id,
                        "protocol_id": endpoint["protocol_id"],
                        "cohort_id": "d4_low_snr_sugar",
                        "evidence_component": endpoint["evidence_component"],
                        "metric_output_id": endpoint["metric_output_id"],
                        "estimate": None,
                        "interval_lower": None,
                        "interval_upper": None,
                        "preferred_direction": endpoint["preferred_direction"],
                        "state": config.direct_gt_state,
                        "reason_code": config.direct_gt_state,
                        "phase5_power_state": "not_powered_for_phase5",
                        "source_path": "system_status.jsonl",
                        "source_sha256": None,
                    }
                )
                continue
            if bootstrap is None:
                raise DenoisingEvidenceVerificationError(f"missing bootstrap row for {system.system_id}:{endpoint_id}")
            source_path = (
                "downstream_well_rows.jsonl"
                if endpoint_id in {"D4-denoise-A", "D4-denoise-B"}
                else "reference_free_well_rows.jsonl"
                if endpoint_id == "D4-reference-free"
                else "half_split_well_rows.jsonl"
            )
            rows.append(
                {
                    "evidence_id": f"{system.system_id}:{endpoint_id}",
                    "task_line": "denoising",
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "endpoint_id": endpoint_id,
                    "protocol_id": endpoint["protocol_id"],
                    "cohort_id": "d4_low_snr_sugar",
                    "evidence_component": endpoint["evidence_component"],
                    "metric_output_id": endpoint["metric_output_id"],
                    "estimate": bootstrap["estimate"],
                    "interval_lower": bootstrap["interval_lower"],
                    "interval_upper": bootstrap["interval_upper"],
                    "preferred_direction": endpoint["preferred_direction"],
                    "state": bootstrap["state"],
                    "reason_code": "" if bootstrap["state"] == "complete" else bootstrap["state"],
                    "phase5_power_state": "not_powered_for_phase5",
                    "source_path": source_path,
                    "source_sha256": None,
                }
            )
    return rows


def _family_projection_rows(method_rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in method_rows:
        groups.setdefault((str(row["family_id"]), str(row["endpoint_id"])), []).append(row)
    rows: list[Mapping[str, object]] = []
    for (family_id, endpoint_id), items in sorted(groups.items()):
        values = [float(item["estimate"]) for item in items if item["estimate"] is not None]
        rows.append(
            {
                "family_id": family_id,
                "endpoint_id": endpoint_id,
                "registered_system_count": len(items),
                "complete_system_count": len(values),
                "median_estimate": None if not values else float(np.median(values)),
                "min_estimate": None if not values else float(np.min(values)),
                "max_estimate": None if not values else float(np.max(values)),
            }
        )
    return rows


def _sha256sums(payloads: Mapping[str, bytes]) -> bytes:
    lines = [f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in sorted(payloads.items())]
    return "".join(lines).encode("utf-8")


def _artifact_payloads(
    *,
    config: DenoisingEvidenceConfig,
    inputs: DenoisingEvidenceInputs,
    systems: Sequence[Phase3System],
    system_status_rows: Sequence[Mapping[str, object]],
    fit_receipts: Sequence[Mapping[str, object]],
    transform_receipts: Sequence[Mapping[str, object]],
    model_receipts: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    half_split_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    method_rows: Sequence[Mapping[str, object]],
    family_rows: Sequence[Mapping[str, object]],
    identity_projection: Mapping[str, object],
    run_id: str,
    project_root: Path,
) -> Mapping[str, bytes]:
    authority_bridge = {
        "authorities": _json_ready(config.authorities),
        "code_authority": _json_ready(_code_authority(project_root)),
        "cohort_identity": _json_ready(inputs.cohort_identity),
        "source_revision_status": config.source_revision_status,
        "system_ids_sha256": _ordered_sha(tuple(system.system_id for system in systems)),
    }
    preflight = {
        "status": "passed",
        "cohort_identity": _json_ready(inputs.cohort_identity),
        "identity_equivalence": {
            "status": "passed",
            "protocol_a_prediction_sha256": identity_projection["prediction_sha256"],
            "protocol_b_prediction_sha256": identity_projection["prediction_sha256"],
            "protocol_a_model_digest_sha256": identity_projection.get(
                "model_digest_sha256",
                identity_projection["prediction_sha256"],
            ),
            "protocol_b_model_digest_sha256": identity_projection.get(
                "model_digest_sha256",
                identity_projection["prediction_sha256"],
            ),
        },
        "system_count": len(systems),
    }
    manifest = {
        "schema_version": "phase6-denoising-evidence-artifact-v1",
        "run_id": run_id,
        "status": "complete",
        "synthetic_fixture": bool(inputs.synthetic_fixture),
        "system_count": len(systems),
        "fit_receipt_count": len(fit_receipts),
        "transform_receipt_count": len(transform_receipts),
        "model_receipt_count": len(model_receipts),
        "downstream_well_row_count": len(downstream_rows),
        "reference_free_well_row_count": len(reference_rows),
        "half_split_well_row_count": len(half_split_rows),
        "bootstrap_row_count": len(bootstrap_rows),
        "method_evidence_row_count": len(method_rows),
        "configured_payload_count": int(config.artifact_contract["configured_payload_count"]),
        "payload_files": list(config.artifact_contract["payload_files"]),
        "source_revision_status": config.source_revision_status,
    }
    method_fieldnames = (
        "evidence_id",
        "task_line",
        "system_id",
        "family_id",
        "endpoint_id",
        "protocol_id",
        "cohort_id",
        "evidence_component",
        "metric_output_id",
        "estimate",
        "interval_lower",
        "interval_upper",
        "preferred_direction",
        "state",
        "reason_code",
        "phase5_power_state",
        "source_path",
        "source_sha256",
    )
    family_fieldnames = (
        "family_id",
        "endpoint_id",
        "registered_system_count",
        "complete_system_count",
        "median_estimate",
        "min_estimate",
        "max_estimate",
    )
    payloads: dict[str, bytes] = {
        "config.json": _canonical(_json_ready(config.document)),
        "authority_bridge.json": _canonical(authority_bridge),
        "preflight.json": _canonical(preflight),
        "fit_receipts.jsonl": _jsonl_bytes(fit_receipts),
        "transform_receipts.jsonl": _jsonl_bytes(transform_receipts),
        "model_receipts.jsonl": _jsonl_bytes(model_receipts),
        "system_status.jsonl": _jsonl_bytes(system_status_rows),
        "downstream_well_rows.jsonl": _jsonl_bytes(downstream_rows),
        "reference_free_well_rows.jsonl": _jsonl_bytes(reference_rows),
        "half_split_well_rows.jsonl": _jsonl_bytes(half_split_rows),
        "bootstrap_results.jsonl": _jsonl_bytes(bootstrap_rows),
        "method_evidence_rows.csv": _csv_bytes(method_rows, method_fieldnames),
        "family_projection.csv": _csv_bytes(family_rows, family_fieldnames),
        "manifest.json": _canonical(manifest),
        "complete.json": _canonical({"run_id": run_id, "status": "complete"}),
    }
    method_sha = {
        "system_status.jsonl": hashlib.sha256(payloads["system_status.jsonl"]).hexdigest(),
        "downstream_well_rows.jsonl": hashlib.sha256(payloads["downstream_well_rows.jsonl"]).hexdigest(),
        "reference_free_well_rows.jsonl": hashlib.sha256(payloads["reference_free_well_rows.jsonl"]).hexdigest(),
        "half_split_well_rows.jsonl": hashlib.sha256(payloads["half_split_well_rows.jsonl"]).hexdigest(),
    }
    method_rows_with_source = []
    for row in method_rows:
        source_path = str(row["source_path"])
        method_rows_with_source.append(
            {
                **row,
                "source_sha256": method_sha.get(source_path),
            }
        )
    payloads["method_evidence_rows.csv"] = _csv_bytes(method_rows_with_source, method_fieldnames)
    payloads["SHA256SUMS"] = _sha256sums(payloads)
    return payloads


def _run_id(
    *,
    config: DenoisingEvidenceConfig,
    inputs: DenoisingEvidenceInputs,
    systems: Sequence[Phase3System],
    project_root: Path,
) -> str:
    payload = {
        "config_sha256": config.sha256,
        "cohort_id": inputs.cohort_id,
        "cohort_identity": _json_ready(inputs.cohort_identity),
        "synthetic_fixture": bool(inputs.synthetic_fixture),
        "system_ids": [system.system_id for system in systems],
        "code_authority": _json_ready(_code_authority(project_root)),
    }
    digest = hashlib.sha256(RUN_DOMAIN + _canonical(payload)).hexdigest()
    return f"{config.artifact_contract['run_prefix']}{digest}"


def _reexecute_payloads(
    *,
    path: Path,
    project_root: Path,
) -> tuple[Mapping[str, bytes], DenoisingEvidenceSummary]:
    config = load_phase6_denoising_evidence_config(DEFAULT_CONFIG)
    _validate_authorities(config, project_root)
    manifest = _read_json(path / "manifest.json")
    system_rows = [
        json.loads(line)
        for line in (path / "system_status.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    is_synthetic = bool(manifest.get("synthetic_fixture"))
    if is_synthetic:
        system_ids = tuple(str(row["system_id"]) for row in system_rows)
        systems = _systems_from_catalog(config, system_ids, project_root)
        inputs = _make_synthetic_denoising_evidence_inputs()
    else:
        systems = _assert_exact_system_status_catalog(
            config=config,
            system_rows=system_rows,
            project_root=project_root,
        )
        inputs = _load_real_inputs(config, project_root)
    identity_models, identity_predictions, identity_losses, identity_reference, identity_half_split, identity_projection = _identity_models(
        inputs,
        config,
    )
    system_status_rows: list[Mapping[str, object]] = []
    fit_receipts: list[Mapping[str, object]] = []
    transform_receipts: list[Mapping[str, object]] = []
    model_receipts: list[Mapping[str, object]] = []
    downstream_rows: list[Mapping[str, object]] = []
    reference_rows: list[Mapping[str, object]] = []
    half_split_rows: list[Mapping[str, object]] = []
    for system in systems:
        status, fit_rows, transform_rows, model_rows, downstream, aux_rows = _evaluate_system(
            inputs=inputs,
            config=config,
            system=system,
            identity_models=identity_models,
            identity_predictions=identity_predictions,
            identity_losses=identity_losses,
            identity_reference_free=identity_reference,
            identity_half_split=identity_half_split,
        )
        system_status_rows.append(status)
        fit_receipts.extend(fit_rows)
        transform_receipts.extend(transform_rows)
        model_receipts.extend(model_rows)
        downstream_rows.extend(downstream)
        midpoint = len(aux_rows) // 2
        reference_rows.extend(aux_rows[:midpoint])
        half_split_rows.extend(aux_rows[midpoint:])
    bootstrap_rows = _bootstrap_rows(
        config=config,
        systems=systems,
        downstream_rows=downstream_rows,
        reference_rows=reference_rows,
        half_split_rows=half_split_rows,
    )
    run_id = _run_id(config=config, inputs=inputs, systems=systems, project_root=project_root)
    method_rows = _method_evidence_rows(
        config=config,
        systems=systems,
        system_status_rows=system_status_rows,
        bootstrap_rows=bootstrap_rows,
    )
    family_rows = _family_projection_rows(method_rows)
    payloads = _artifact_payloads(
        config=config,
        inputs=inputs,
        systems=systems,
        system_status_rows=system_status_rows,
        fit_receipts=fit_receipts,
        transform_receipts=transform_receipts,
        model_receipts=model_receipts,
        downstream_rows=downstream_rows,
        reference_rows=reference_rows,
        half_split_rows=half_split_rows,
        bootstrap_rows=bootstrap_rows,
        method_rows=method_rows,
        family_rows=family_rows,
        identity_projection=identity_projection,
        run_id=run_id,
        project_root=project_root,
    )
    summary = DenoisingEvidenceSummary(
        path=path,
        run_id=run_id,
        status="complete",
        system_count=len(systems),
        fit_receipt_count=len(fit_receipts),
        transform_receipt_count=len(transform_receipts),
        model_receipt_count=len(model_receipts),
        downstream_well_row_count=len(downstream_rows),
        reference_free_well_row_count=len(reference_rows),
        half_split_well_row_count=len(half_split_rows),
        bootstrap_row_count=len(bootstrap_rows),
        method_evidence_row_count=len(method_rows),
    )
    return payloads, summary


def verify_phase6_denoising_evidence(
    path: Path,
    *,
    project_root: Path = ROOT,
) -> DenoisingEvidenceSummary:
    path = Path(path)
    if not path.is_dir():
        raise DenoisingEvidenceVerificationError("run path must exist")
    try:
        expected_payloads, rebuilt = _reexecute_payloads(
            path=path,
            project_root=project_root,
        )
        observed = _artifact_files(path)
        if observed != expected_payloads:
            raise DenoisingEvidenceVerificationError("artifact byte mismatch after full reexecution")
        return rebuilt
    except DenoisingEvidenceError as error:
        raise DenoisingEvidenceVerificationError(str(error)) from error


__all__ = ["verify_phase6_denoising_evidence"]
