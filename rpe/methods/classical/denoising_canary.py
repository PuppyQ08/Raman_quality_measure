from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.evaluation import Spectrum1D
from rpe.io.store import UnifiedDataset
from rpe.methods.catalog import (
    Phase3ClassicalCatalog,
    Phase3System,
    TaskLine,
    load_classical_catalog,
)
from rpe.methods.classical.baseline_canary import (
    BaselineCanarySource,
    load_rruff_baseline_canary_sources,
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


CONFIG_BYTE_COUNT = 5860
CONFIG_SHA256 = "75c39c647ed3ab56ba51853ebc10d608509750e46e12f4f126c9febb6627e79b"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RUN_DOMAIN = b"rpe-phase3-denoising-v1-canary-v1\0"
_STATELESS_FAMILIES = frozenset(
    {"savitzky_golay", "wavelet", "whittaker_smoothing"}
)
_FITTED_FAMILIES = frozenset({"pca_reconstruction", "svd_reconstruction"})
_SUCCESS = frozenset(
    {DenoisingRunStatus.COMPLETE.value, DenoisingRunStatus.COMPLETE_WITH_WARNING.value}
)


class DenoisingCanaryError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class DenoisingCanaryConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    catalog_path: str
    catalog_sha256: str
    catalog_id: str
    phase3_lock_path: str
    phase3_lock_sha256: str
    design_path: str
    design_sha256: str
    system_ids: tuple[str, ...]
    expected: Mapping[str, int]
    rruff: Mapping[str, object]
    bacteria: Mapping[str, object]
    claim_boundary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", MappingProxyType(dict(self.expected)))
        object.__setattr__(self, "rruff", _freeze_mapping(self.rruff))
        object.__setattr__(self, "bacteria", _freeze_mapping(self.bacteria))


@dataclass(frozen=True)
class DenoisingCanarySource:
    cohort_id: str
    record_id: str
    class_label: int
    source_row: int
    point_count: int
    axis_sha256: str
    intensity_sha256: str
    spectrum: Spectrum1D
    sample_id: str | None = None
    mineral_name: str | None = None


@dataclass(frozen=True)
class DenoisingCanaryInputs:
    stateless_sources: tuple[DenoisingCanarySource, ...]
    fit_sources: tuple[DenoisingCanarySource, ...]
    transform_sources: tuple[DenoisingCanarySource, ...]
    axis_sha256: str
    fit_ledger_sha256: str
    transform_ledger_sha256: str
    fit_matrix_sha256: str


@dataclass(frozen=True)
class DenoisingCanarySummary:
    path: Path
    run_id: str
    system_count: int
    fit_receipt_count: int
    transform_receipt_count: int
    fit_status_counts: Mapping[str, int]
    transform_status_counts: Mapping[str, int]
    canary_passed: bool
    is_frozen_canary: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fit_status_counts", MappingProxyType(dict(sorted(self.fit_status_counts.items())))
        )
        object.__setattr__(
            self,
            "transform_status_counts",
            MappingProxyType(dict(sorted(self.transform_status_counts.items()))),
        )


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


def _reject_nonfinite(token: str) -> object:
    raise DenoisingCanaryError("JSON", f"non-finite constant {token}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _freeze(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DenoisingCanaryError("mapping", "contains a non-finite float")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    raise DenoisingCanaryError("mapping", "contains an unsupported value")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze(item) for key, item in sorted(value.items())}
    )


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _exact_identity(root: Path, document: Mapping[str, object], label: str) -> None:
    path = root / str(document["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(document["byte_count"])
        or _sha(path) != str(document["sha256"])
    ):
        raise DenoisingCanaryError(label, "identity mismatch")


def load_denoising_canary_config(
    path: Path, *, project_root: Path
) -> DenoisingCanaryConfig:
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw, parse_constant=_reject_nonfinite)
    if not isinstance(document, Mapping) or raw != _canonical(document):
        raise DenoisingCanaryError("config", "must be canonical finite JSON")
    observed_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTE_COUNT or observed_sha != CONFIG_SHA256:
        raise DenoisingCanaryError("config", "frozen identity mismatch")
    root = Path(project_root)
    for key in ("catalog", "phase3_lock", "design"):
        value = document[key]
        if not isinstance(value, Mapping):
            raise DenoisingCanaryError(key, "must be an object")
        _exact_identity(root, value, key)
    catalog_document = document["catalog"]
    lock_document = document["phase3_lock"]
    design_document = document["design"]
    assert isinstance(catalog_document, Mapping)
    assert isinstance(lock_document, Mapping)
    assert isinstance(design_document, Mapping)
    catalog = load_classical_catalog(
        root / str(catalog_document["path"]), project_root=root
    )
    denoising_ids = tuple(
        system.system_id
        for system in catalog.systems
        if system.task_line is TaskLine.DENOISING
    )
    configured_ids = tuple(str(value) for value in document["denoising_system_ids"])
    if configured_ids != tuple(sorted(denoising_ids)) or len(configured_ids) != 60:
        raise DenoisingCanaryError("denoising_system_ids", "do not match catalog view")
    if (
        str(catalog_document["catalog_id"]) != catalog.catalog_id
        or str(catalog_document["sha256"]) != catalog.sha256
    ):
        raise DenoisingCanaryError("catalog", "does not match loaded catalog")
    expected_raw = document["expected"]
    if not isinstance(expected_raw, Mapping):
        raise DenoisingCanaryError("expected", "must be an object")
    expected = {str(key): int(value) for key, value in expected_raw.items()}
    literal_expected = {
        "fit_receipt_count": 24,
        "fitted_system_count": 24,
        "stateless_source_count": 4,
        "stateless_system_count": 36,
        "system_count": 60,
        "transform_receipt_count": 864,
        "transform_source_count": 30,
    }
    if expected != literal_expected:
        raise DenoisingCanaryError("expected", "does not match frozen design")
    rruff = document["rruff"]
    bacteria = document["bacteria"]
    if not isinstance(rruff, Mapping) or not isinstance(bacteria, Mapping):
        raise DenoisingCanaryError("cohorts", "must be objects")
    return DenoisingCanaryConfig(
        path=path,
        byte_count=len(raw),
        sha256=observed_sha,
        schema_version=str(document["schema_version"]),
        catalog_path=str(catalog_document["path"]),
        catalog_sha256=str(catalog_document["sha256"]),
        catalog_id=str(catalog_document["catalog_id"]),
        phase3_lock_path=str(lock_document["path"]),
        phase3_lock_sha256=str(lock_document["sha256"]),
        design_path=str(design_document["path"]),
        design_sha256=str(design_document["sha256"]),
        system_ids=configured_ids,
        expected=expected,
        rruff=rruff,
        bacteria=bacteria,
        claim_boundary=str(document["claim_boundary"]),
    )


def _rruff_source_document(source: BaselineCanarySource) -> dict[str, object]:
    return {
        "axis_sha256": source.axis_sha256,
        "band_id": source.band_id,
        "class_label": source.class_label,
        "intensity_sha256": source.intensity_sha256,
        "mineral_name": source.mineral_name,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "sample_id": source.sample_id,
        "selection_rank": source.selection_rank,
    }


def _bacteria_ledger_row(source: DenoisingCanarySource) -> dict[str, object]:
    return {
        "class_label": source.class_label,
        "record_id": source.record_id,
        "source_row": source.source_row,
    }


def _materialize_bacteria_source(
    dataset: UnifiedDataset,
    row: Mapping[str, object],
    *,
    cohort_id: str,
) -> DenoisingCanarySource:
    record_id = str(row["record_id"])
    record = dataset.get(record_id)
    axis = np.asarray(record.wavenumber, dtype="<f8")
    intensity = np.asarray(record.intensity, dtype="<f8")
    if np.all(np.diff(axis) < 0.0):
        axis = np.ascontiguousarray(axis[::-1])
        intensity = np.ascontiguousarray(intensity[::-1])
    spectrum = Spectrum1D(record_id, None, axis, intensity)
    return DenoisingCanarySource(
        cohort_id=cohort_id,
        record_id=record_id,
        class_label=int(row["targets"]["class_label"]),  # type: ignore[index]
        source_row=int(row["meta"]["source_metadata"]["source_row"]),  # type: ignore[index]
        point_count=spectrum.intensity.size,
        axis_sha256=_array_sha(spectrum.axis_cm1),
        intensity_sha256=_array_sha(spectrum.intensity),
        spectrum=spectrum,
        sample_id=record.meta.sample_id,
        mineral_name=None,
    )


def load_denoising_canary_inputs(
    config: DenoisingCanaryConfig, *, project_root: Path
) -> DenoisingCanaryInputs:
    if not isinstance(config, DenoisingCanaryConfig):
        raise DenoisingCanaryError("config", "must be DenoisingCanaryConfig")
    root = Path(project_root)
    phase1_path = root / str(config.rruff["phase1_run_path"])
    rruff_dataset = root / str(config.rruff["dataset_path"])
    rruff_values = load_rruff_baseline_canary_sources(phase1_path, rruff_dataset)
    rruff_payload = b"".join(_canonical(_rruff_source_document(value)) for value in rruff_values)
    if hashlib.sha256(rruff_payload).hexdigest() != config.rruff["source_ledger_sha256"]:
        raise DenoisingCanaryError("RRUFF source ledger", "SHA256 mismatch")
    stateless = tuple(
        DenoisingCanarySource(
            cohort_id=value.band_id,
            record_id=value.record_id,
            class_label=value.class_label,
            source_row=value.selection_rank,
            point_count=value.point_count,
            axis_sha256=value.axis_sha256,
            intensity_sha256=value.intensity_sha256,
            spectrum=value.spectrum,
            sample_id=value.sample_id,
            mineral_name=value.mineral_name,
        )
        for value in rruff_values
    )

    bacteria_path = root / str(config.bacteria["dataset_path"])
    if _sha(bacteria_path / "SHA256SUMS") != config.bacteria["dataset_sha256sums_sha256"]:
        raise DenoisingCanaryError("Bacteria SHA256SUMS", "identity mismatch")
    rows = [
        json.loads(line, parse_constant=_reject_nonfinite)
        for line in (bacteria_path / "records.jsonl").read_bytes().splitlines()
    ]
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        metadata = row["meta"]["source_metadata"]
        split = str(metadata["source_split"])
        label = int(row["targets"]["class_label"])
        grouped.setdefault((split, label), []).append(row)
    for values in grouped.values():
        values.sort(
            key=lambda row: (
                int(row["meta"]["source_metadata"]["source_row"]),
                str(row["record_id"]),
            )
        )
    fit_rows: list[Mapping[str, object]] = []
    transform_rows: list[Mapping[str, object]] = []
    fit_selection = config.bacteria["fit_selection"]
    transform_selection = config.bacteria["transform_selection"]
    assert isinstance(fit_selection, Mapping)
    assert isinstance(transform_selection, Mapping)
    for label in range(30):
        fit_candidates = grouped.get((str(fit_selection["source_split"]), label), [])
        transform_candidates = grouped.get(
            (str(transform_selection["source_split"]), label), []
        )
        fit_count = int(fit_selection["per_class"])
        transform_count = int(transform_selection["per_class"])
        if len(fit_candidates) < fit_count or len(transform_candidates) < transform_count:
            raise DenoisingCanaryError("Bacteria selection", f"class {label} is incomplete")
        fit_rows.extend(fit_candidates[:fit_count])
        transform_rows.extend(transform_candidates[:transform_count])
    with UnifiedDataset.open(bacteria_path, verify_checksums=True) as dataset:
        fit_sources = tuple(
            _materialize_bacteria_source(dataset, row, cohort_id="bacteria_fit")
            for row in fit_rows
        )
        transform_sources = tuple(
            _materialize_bacteria_source(dataset, row, cohort_id="bacteria_transform")
            for row in transform_rows
        )
    axes = {source.axis_sha256 for source in (*fit_sources, *transform_sources)}
    if len(axes) != 1:
        raise DenoisingCanaryError("Bacteria axis", "must be exactly shared")
    axis_sha = next(iter(axes))
    fit_ledger = hashlib.sha256(
        b"".join(_canonical(_bacteria_ledger_row(value)) for value in fit_sources)
    ).hexdigest()
    transform_ledger = hashlib.sha256(
        b"".join(_canonical(_bacteria_ledger_row(value)) for value in transform_sources)
    ).hexdigest()
    fit_matrix = np.ascontiguousarray(
        [source.spectrum.intensity for source in fit_sources], dtype="<f8"
    )
    fit_matrix_sha = _array_sha(fit_matrix)
    expected_values = {
        "axis_sha256": axis_sha,
        "fit_ledger_sha256": fit_ledger,
        "transform_ledger_sha256": transform_ledger,
        "fit_matrix_sha256": fit_matrix_sha,
    }
    for key, observed in expected_values.items():
        if observed != config.bacteria[key]:
            raise DenoisingCanaryError(f"Bacteria {key}", "does not match config")
    return DenoisingCanaryInputs(
        stateless_sources=stateless,
        fit_sources=fit_sources,
        transform_sources=transform_sources,
        axis_sha256=axis_sha,
        fit_ledger_sha256=fit_ledger,
        transform_ledger_sha256=transform_ledger,
        fit_matrix_sha256=fit_matrix_sha,
    )


def _source_document(source: DenoisingCanarySource, *, role: str) -> dict[str, object]:
    return {
        "axis_sha256": source.axis_sha256,
        "class_label": source.class_label,
        "cohort_id": source.cohort_id,
        "intensity_sha256": source.intensity_sha256,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "role": role,
        "sample_id": source.sample_id,
        "source_row": source.source_row,
        "mineral_name": source.mineral_name,
    }


def _stateless_identity_document(source: DenoisingCanarySource) -> dict[str, object]:
    if source.sample_id is None or source.mineral_name is None:
        raise DenoisingCanaryError(
            "stateless source metadata", "must include sample and mineral identities"
        )
    return {
        "axis_sha256": source.axis_sha256,
        "band_id": source.cohort_id,
        "class_label": source.class_label,
        "intensity_sha256": source.intensity_sha256,
        "mineral_name": source.mineral_name,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "sample_id": source.sample_id,
        "selection_rank": source.source_row,
    }


def _system_document(system: Phase3System) -> dict[str, object]:
    return {
        "family_id": system.family_id,
        "hyperparameters": _json_ready(system.hyperparameters),
        "kind": "fitted" if system.family_id in _FITTED_FAMILIES else "stateless",
        "method_id": system.method_id,
        "system_id": system.system_id,
    }


def _code_identity(root: Path) -> dict[str, str]:
    paths = (
        "rpe/methods/catalog.py",
        "rpe/methods/classical/denoising.py",
        "rpe/methods/classical/denoising_canary.py",
    )
    return {path: _sha(root / path) for path in paths}


def _software_identity() -> dict[str, str]:
    names = ("numpy", "scipy", "scikit-learn", "PyWavelets")
    return {name: importlib.metadata.version(name) for name in names}


def _actual_input_identity(inputs: DenoisingCanaryInputs) -> dict[str, str]:
    stateless_payload = b"".join(
        _canonical(_stateless_identity_document(source))
        for source in inputs.stateless_sources
    )
    fit_payload = b"".join(
        _canonical(_bacteria_ledger_row(source)) for source in inputs.fit_sources
    )
    transform_payload = b"".join(
        _canonical(_bacteria_ledger_row(source)) for source in inputs.transform_sources
    )
    fit_matrix = np.ascontiguousarray(
        [source.spectrum.intensity for source in inputs.fit_sources], dtype="<f8"
    )
    axis_values = {source.axis_sha256 for source in (*inputs.fit_sources, *inputs.transform_sources)}
    axis_sha = next(iter(axis_values)) if len(axis_values) == 1 else "mixed"
    return {
        "fit_axis_sha256": axis_sha,
        "fit_ledger_sha256": hashlib.sha256(fit_payload).hexdigest(),
        "fit_matrix_sha256": _array_sha(fit_matrix),
        "stateless_source_sha256": hashlib.sha256(stateless_payload).hexdigest(),
        "transform_ledger_sha256": hashlib.sha256(transform_payload).hexdigest(),
    }


def _matches_frozen_input_identity(
    identity: Mapping[str, str], config: DenoisingCanaryConfig
) -> bool:
    return (
        identity.get("stateless_source_sha256")
        == config.rruff["source_ledger_sha256"]
        and identity.get("fit_axis_sha256") == config.bacteria["axis_sha256"]
        and identity.get("fit_ledger_sha256")
        == config.bacteria["fit_ledger_sha256"]
        and identity.get("transform_ledger_sha256")
        == config.bacteria["transform_ledger_sha256"]
        and identity.get("fit_matrix_sha256")
        == config.bacteria["fit_matrix_sha256"]
    )


def _append_array(stream: bytearray, value: np.ndarray, role: str) -> dict[str, object]:
    array = np.ascontiguousarray(value, dtype="<f8")
    raw = array.tobytes()
    descriptor = {
        "byte_count": len(raw),
        "offset": len(stream),
        "role": role,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(array.shape),
    }
    stream.extend(raw)
    return descriptor


def _warning_documents(
    warnings: Sequence[DenoisingWarning],
    *,
    receipt_kind: str,
    system_id: str,
    record_id: str | None,
) -> list[dict[str, object]]:
    return [
        {
            "category": warning.category,
            "message": warning.message,
            "record_id": record_id,
            "receipt_kind": receipt_kind,
            "sequence": index,
            "system_id": system_id,
        }
        for index, warning in enumerate(warnings)
    ]


def _transform_document(
    source: DenoisingCanarySource,
    result: DenoisingRunResult,
    output_stream: bytearray,
) -> dict[str, object]:
    output_descriptor = None
    if result.denoised_intensity is not None:
        output_descriptor = _append_array(
            output_stream, result.denoised_intensity, "denoised_intensity"
        )
    return {
        "cohort_id": source.cohort_id,
        "diagnostics": _json_ready(result.diagnostics),
        "error_code": result.error_code,
        "error_message": result.error_message,
        "family_id": result.family_id,
        "method_id": result.method_id,
        "output": output_descriptor,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "source_row": source.source_row,
        "status": result.status.value,
        "system_id": result.system_id,
        "warnings": [
            {"category": warning.category, "message": warning.message}
            for warning in result.warnings
        ],
    }


def _fit_success_document(
    system: Phase3System,
    fitted: FittedDenoiser,
    state_stream: bytearray,
) -> dict[str, object]:
    arrays = []
    if fitted.mean is not None:
        arrays.append(_append_array(state_stream, fitted.mean, "mean"))
    arrays.append(_append_array(state_stream, fitted.components, "components"))
    arrays.append(
        _append_array(state_stream, fitted.explained_variance, "explained_variance")
    )
    status = (
        DenoisingRunStatus.COMPLETE_WITH_WARNING
        if fitted.warnings
        else DenoisingRunStatus.COMPLETE
    )
    return {
        "arrays": arrays,
        "axis_sha256": fitted.axis_sha256,
        "context_sha256": fitted.context_sha256,
        "error_code": None,
        "error_message": None,
        "family_id": system.family_id,
        "method_id": system.method_id,
        "state_sha256": fitted.state_sha256,
        "status": status.value,
        "system_id": system.system_id,
        "training_matrix_sha256": fitted.training_matrix_sha256,
        "training_record_ledger_sha256": fitted.training_record_ledger_sha256,
        "warnings": [
            {"category": warning.category, "message": warning.message}
            for warning in fitted.warnings
        ],
    }


def _fit_failure_document(
    system: Phase3System, error: DenoisingFitError
) -> dict[str, object]:
    return {
        "arrays": [],
        "axis_sha256": None,
        "context_sha256": None,
        "error_code": error.path,
        "error_message": error.reason,
        "family_id": system.family_id,
        "method_id": system.method_id,
        "state_sha256": None,
        "status": error.status.value,
        "system_id": system.system_id,
        "training_matrix_sha256": None,
        "training_record_ledger_sha256": None,
        "warnings": [],
    }


def _failed_transform_document(
    system: Phase3System,
    source: DenoisingCanarySource,
    error: DenoisingFitError,
) -> dict[str, object]:
    return {
        "cohort_id": source.cohort_id,
        "diagnostics": {},
        "error_code": error.path,
        "error_message": error.reason,
        "family_id": system.family_id,
        "method_id": system.method_id,
        "output": None,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "source_row": source.source_row,
        "status": DenoisingRunStatus.FAILED_FIT.value,
        "system_id": system.system_id,
        "warnings": [],
    }


def build_denoising_canary_artifact(
    inputs: DenoisingCanaryInputs,
    systems: Sequence[Phase3System],
    output_root: Path,
    *,
    config: DenoisingCanaryConfig,
    catalog: Phase3ClassicalCatalog,
    project_root: Path,
) -> DenoisingCanarySummary:
    if not isinstance(inputs, DenoisingCanaryInputs):
        raise DenoisingCanaryError("inputs", "must be DenoisingCanaryInputs")
    if not isinstance(config, DenoisingCanaryConfig):
        raise DenoisingCanaryError("config", "must be DenoisingCanaryConfig")
    frozen_systems = tuple(sorted(systems, key=lambda value: value.system_id))
    if not frozen_systems or len({value.system_id for value in frozen_systems}) != len(frozen_systems):
        raise DenoisingCanaryError("systems", "must be nonempty and unique")
    if any(value.task_line is not TaskLine.DENOISING for value in frozen_systems):
        raise DenoisingCanaryError("systems", "must all be denoising systems")
    catalog_ids = {value.system_id for value in catalog.systems}
    if any(value.system_id not in catalog_ids for value in frozen_systems):
        raise DenoisingCanaryError("systems", "contain a noncatalog system")
    source_groups = (
        inputs.stateless_sources,
        inputs.fit_sources,
        inputs.transform_sources,
    )
    for sources in source_groups:
        if not sources or len({value.record_id for value in sources}) != len(sources):
            raise DenoisingCanaryError("sources", "each cohort must be nonempty and unique")
        for source in sources:
            if (
                source.point_count != source.spectrum.intensity.size
                or source.axis_sha256 != _array_sha(source.spectrum.axis_cm1)
                or source.intensity_sha256 != _array_sha(source.spectrum.intensity)
            ):
                raise DenoisingCanaryError("source identity", "does not match spectrum")

    root = Path(project_root)
    input_identity = _actual_input_identity(inputs)
    system_ids = tuple(value.system_id for value in frozen_systems)
    identity = {
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": catalog.sha256,
        "code_identity": _code_identity(root),
        "config_sha256": config.sha256,
        "input_identity": input_identity,
        "phase3_lock_sha256": config.phase3_lock_sha256,
        "software_identity": _software_identity(),
        "system_ids": list(system_ids),
    }
    run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()
    path = Path(output_root) / f"phase3-denoising-v1-canary-{run_id}"
    if path.exists():
        raise DenoisingCanaryError("output", "run path already exists")
    path.mkdir(parents=True)

    stateless_systems = tuple(
        value for value in frozen_systems if value.family_id in _STATELESS_FAMILIES
    )
    fitted_systems = tuple(
        value for value in frozen_systems if value.family_id in _FITTED_FAMILIES
    )
    fit_context = DenoisingFitContext(
        split_id="bacteria_reference_min4_per_class_v1",
        representation_id="bacteria_id_reference_increasing_float64_v1",
        record_ids=tuple(value.record_id for value in inputs.fit_sources),
    )
    training_spectra = tuple(value.spectrum for value in inputs.fit_sources)
    state_stream = bytearray()
    output_stream = bytearray()
    fit_documents: list[dict[str, object]] = []
    transform_documents: list[dict[str, object]] = []
    warning_documents: list[dict[str, object]] = []

    for current_system in stateless_systems:
        for source in inputs.stateless_sources:
            result = run_stateless_denoising_system(current_system, source.spectrum)
            transform_documents.append(
                _transform_document(source, result, output_stream)
            )
            warning_documents.extend(
                _warning_documents(
                    result.warnings,
                    receipt_kind="transform",
                    system_id=current_system.system_id,
                    record_id=source.record_id,
                )
            )
    for current_system in fitted_systems:
        try:
            fitted = fit_denoising_system(current_system, training_spectra, fit_context)
        except DenoisingFitError as error:
            fit_documents.append(_fit_failure_document(current_system, error))
            for source in inputs.transform_sources:
                transform_documents.append(
                    _failed_transform_document(current_system, source, error)
                )
            continue
        fit_documents.append(
            _fit_success_document(current_system, fitted, state_stream)
        )
        warning_documents.extend(
            _warning_documents(
                fitted.warnings,
                receipt_kind="fit",
                system_id=current_system.system_id,
                record_id=None,
            )
        )
        for source in inputs.transform_sources:
            result = transform_fitted_denoiser(fitted, source.spectrum)
            transform_documents.append(
                _transform_document(source, result, output_stream)
            )
            warning_documents.extend(
                _warning_documents(
                    result.warnings,
                    receipt_kind="transform",
                    system_id=current_system.system_id,
                    record_id=source.record_id,
                )
            )

    source_documents = [
        *(
            _source_document(value, role="stateless")
            for value in inputs.stateless_sources
        ),
        *(_source_document(value, role="fit") for value in inputs.fit_sources),
        *(
            _source_document(value, role="transform")
            for value in inputs.transform_sources
        ),
    ]
    system_documents = [_system_document(value) for value in frozen_systems]
    fit_counts = Counter(str(value["status"]) for value in fit_documents)
    transform_counts = Counter(str(value["status"]) for value in transform_documents)
    expected = config.expected
    is_frozen = (
        system_ids == config.system_ids
        and len(stateless_systems) == expected["stateless_system_count"]
        and len(fitted_systems) == expected["fitted_system_count"]
        and len(inputs.stateless_sources) == expected["stateless_source_count"]
        and len(inputs.transform_sources) == expected["transform_source_count"]
        and len(fit_documents) == expected["fit_receipt_count"]
        and len(transform_documents) == expected["transform_receipt_count"]
        and _matches_frozen_input_identity(input_identity, config)
    )
    all_success = set(fit_counts).issubset(_SUCCESS) and set(transform_counts).issubset(_SUCCESS)
    canary_passed = is_frozen and all_success
    summary = {
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "fit_receipt_count": len(fit_documents),
        "fit_status_counts": dict(sorted(fit_counts.items())),
        "is_frozen_canary": is_frozen,
        "run_id": run_id,
        "system_count": len(frozen_systems),
        "transform_receipt_count": len(transform_documents),
        "transform_status_counts": dict(sorted(transform_counts.items())),
        "warning_count": len(warning_documents),
    }
    manifest = {
        **identity,
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "fit_receipt_count": len(fit_documents),
        "is_frozen_canary": is_frozen,
        "run_id": run_id,
        "schema_version": "phase3-denoising-v1-canary-run-v1",
        "source_count": len(source_documents),
        "system_count": len(frozen_systems),
        "transform_receipt_count": len(transform_documents),
    }
    marker_name = "complete.json" if canary_passed else "failed.json"
    marker = {
        "canary_passed": canary_passed,
        "is_frozen_canary": is_frozen,
        "run_id": run_id,
        "status": "complete" if canary_passed else "failed",
    }
    payloads = {
        "denoised_outputs.f64le": bytes(output_stream),
        "fitted_states.f64le": bytes(state_stream),
        "fit_receipts.jsonl": b"".join(_canonical(value) for value in fit_documents),
        "manifest.json": _canonical(manifest),
        "sources.jsonl": b"".join(_canonical(value) for value in source_documents),
        "summary.json": _canonical(summary),
        "systems.jsonl": b"".join(_canonical(value) for value in system_documents),
        "transform_receipts.jsonl": b"".join(
            _canonical(value) for value in transform_documents
        ),
        "warnings.jsonl": b"".join(
            _canonical(value) for value in warning_documents
        ),
        marker_name: _canonical(marker),
    }
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
            for name in sorted(payloads)
        ),
        encoding="utf-8",
    )
    return DenoisingCanarySummary(
        path=path,
        run_id=run_id,
        system_count=len(frozen_systems),
        fit_receipt_count=len(fit_documents),
        transform_receipt_count=len(transform_documents),
        fit_status_counts=fit_counts,
        transform_status_counts=transform_counts,
        canary_passed=canary_passed,
        is_frozen_canary=is_frozen,
    )


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=_reject_nonfinite)
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise DenoisingCanaryError(label, "must be a canonical JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, object]]:
    result = []
    for index, raw in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
        value = json.loads(raw, parse_constant=_reject_nonfinite)
        if not isinstance(value, Mapping) or raw != _canonical(value):
            raise DenoisingCanaryError(label, f"line {index} is not canonical")
        result.append(value)
    return result


def _verify_stream(
    path: Path,
    descriptors: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> None:
    raw = path.read_bytes()
    expected_offset = 0
    for descriptor in descriptors:
        offset = int(descriptor["offset"])
        byte_count = int(descriptor["byte_count"])
        shape = tuple(int(value) for value in descriptor["shape"])
        if (
            offset != expected_offset
            or byte_count != math.prod(shape) * 8
            or byte_count <= 0
            or offset + byte_count > len(raw)
        ):
            raise DenoisingCanaryError(label, "offset or shape mismatch")
        current = raw[offset : offset + byte_count]
        if hashlib.sha256(current).hexdigest() != descriptor["sha256"]:
            raise DenoisingCanaryError(label, "array SHA256 mismatch")
        values = np.frombuffer(current, dtype="<f8")
        if not np.isfinite(values).all():
            raise DenoisingCanaryError(label, "contains non-finite values")
        expected_offset += byte_count
    if expected_offset != len(raw):
        raise DenoisingCanaryError(label, "contains unattributed bytes")


def _descriptor_array(
    raw: bytes, descriptor: Mapping[str, object], *, label: str
) -> np.ndarray:
    offset = int(descriptor["offset"])
    byte_count = int(descriptor["byte_count"])
    shape = tuple(int(value) for value in descriptor["shape"])
    if (
        byte_count != math.prod(shape) * 8
        or byte_count <= 0
        or offset < 0
        or offset + byte_count > len(raw)
    ):
        raise DenoisingCanaryError(label, "descriptor bounds or shape mismatch")
    values = np.frombuffer(
        raw[offset : offset + byte_count], dtype="<f8"
    ).reshape(shape).copy()
    if not np.isfinite(values).all():
        raise DenoisingCanaryError(label, "contains non-finite values")
    return values


def _receipt_warnings(
    value: object, *, label: str
) -> tuple[DenoisingWarning, ...]:
    if not isinstance(value, list):
        raise DenoisingCanaryError(label, "warnings must be an array")
    try:
        return tuple(
            DenoisingWarning(str(item["category"]), str(item["message"]))
            for item in value
        )
    except (TypeError, KeyError) as error:
        raise DenoisingCanaryError(label, "warning schema mismatch") from error


def _source_input_identity(
    sources: Sequence[Mapping[str, object]], *, fit_matrix_sha256: str
) -> dict[str, str]:
    roles = tuple(str(value.get("role")) for value in sources)
    if any(role not in {"stateless", "fit", "transform"} for role in roles):
        raise DenoisingCanaryError("sources", "contains an unknown role")
    if roles != tuple(sorted(roles, key={"stateless": 0, "fit": 1, "transform": 2}.get)):
        raise DenoisingCanaryError("sources", "roles are not canonically grouped")
    stateless = [value for value in sources if value["role"] == "stateless"]
    fit = [value for value in sources if value["role"] == "fit"]
    transform = [value for value in sources if value["role"] == "transform"]
    stateless_payload = b"".join(
        _canonical(
            {
                "axis_sha256": value["axis_sha256"],
                "band_id": value["cohort_id"],
                "class_label": value["class_label"],
                "intensity_sha256": value["intensity_sha256"],
                "mineral_name": value["mineral_name"],
                "point_count": value["point_count"],
                "record_id": value["record_id"],
                "sample_id": value["sample_id"],
                "selection_rank": value["source_row"],
            }
        )
        for value in stateless
    )
    fit_payload = b"".join(
        _canonical(
            {
                "class_label": value["class_label"],
                "record_id": value["record_id"],
                "source_row": value["source_row"],
            }
        )
        for value in fit
    )
    transform_payload = b"".join(
        _canonical(
            {
                "class_label": value["class_label"],
                "record_id": value["record_id"],
                "source_row": value["source_row"],
            }
        )
        for value in transform
    )
    axes = {
        str(value["axis_sha256"])
        for value in (*fit, *transform)
    }
    return {
        "fit_axis_sha256": next(iter(axes)) if len(axes) == 1 else "mixed",
        "fit_ledger_sha256": hashlib.sha256(fit_payload).hexdigest(),
        "fit_matrix_sha256": fit_matrix_sha256,
        "stateless_source_sha256": hashlib.sha256(stateless_payload).hexdigest(),
        "transform_ledger_sha256": hashlib.sha256(transform_payload).hexdigest(),
    }


def verify_denoising_canary_artifact(
    path: Path, *, project_root: Path
) -> DenoisingCanarySummary:
    path = Path(path)
    checksum_raw = (path / "SHA256SUMS").read_text(encoding="utf-8")
    checks: dict[str, str] = {}
    for line in checksum_raw.splitlines():
        try:
            digest, name = line.split("  ")
        except ValueError as error:
            raise DenoisingCanaryError("checksum", "invalid line") from error
        if name in checks or len(digest) != 64:
            raise DenoisingCanaryError("checksum", "invalid or duplicate entry")
        checks[name] = digest
        member = path / name
        if not member.is_file() or _sha(member) != digest:
            raise DenoisingCanaryError("checksum", f"{name} mismatch")
    if checksum_raw != "".join(
        f"{checks[name]}  {name}\n" for name in sorted(checks)
    ):
        raise DenoisingCanaryError("checksum", "must be canonically sorted")
    actual = {
        value.relative_to(path).as_posix()
        for value in path.rglob("*")
        if value.is_file() and value.name != "SHA256SUMS"
    }
    if actual != set(checks):
        raise DenoisingCanaryError("checksum inventory", "does not match files")

    manifest = _read_json(path / "manifest.json", "manifest")
    summary = _read_json(path / "summary.json", "summary")
    systems = _read_jsonl(path / "systems.jsonl", "systems")
    sources = _read_jsonl(path / "sources.jsonl", "sources")
    fit_receipts = _read_jsonl(path / "fit_receipts.jsonl", "fit receipts")
    transform_receipts = _read_jsonl(
        path / "transform_receipts.jsonl", "transform receipts"
    )
    warning_rows = _read_jsonl(path / "warnings.jsonl", "warnings")
    marker_names = sorted(actual & {"complete.json", "failed.json"})
    if len(marker_names) != 1:
        raise DenoisingCanaryError("marker", "must contain exactly one")
    expected_inventory = {
        "denoised_outputs.f64le",
        "fitted_states.f64le",
        "fit_receipts.jsonl",
        "manifest.json",
        "sources.jsonl",
        "summary.json",
        "systems.jsonl",
        "transform_receipts.jsonl",
        "warnings.jsonl",
        marker_names[0],
    }
    if actual != expected_inventory:
        raise DenoisingCanaryError("artifact inventory", "does not match protocol")

    root = Path(project_root)
    config = load_denoising_canary_config(
        root / "experiments/phase3/configs/denoising_v1_canary.json",
        project_root=root,
    )
    catalog = load_classical_catalog(root / config.catalog_path, project_root=root)
    catalog_by_id = {value.system_id: value for value in catalog.systems}
    system_ids = tuple(str(value["system_id"]) for value in systems)
    if system_ids != tuple(sorted(set(system_ids))):
        raise DenoisingCanaryError("systems", "must be unique and sorted")
    try:
        catalog_systems = tuple(catalog_by_id[value] for value in system_ids)
    except KeyError as error:
        raise DenoisingCanaryError("systems", "contains an unknown ID") from error
    expected_system_rows = [_system_document(value) for value in catalog_systems]
    if systems != expected_system_rows:
        raise DenoisingCanaryError("systems", "do not match catalog")
    stateless_ids = tuple(
        value.system_id
        for value in catalog_systems
        if value.family_id in _STATELESS_FAMILIES
    )
    fitted_ids = tuple(
        value.system_id for value in catalog_systems if value.family_id in _FITTED_FAMILIES
    )
    source_ids_by_role = {
        role: tuple(str(value["record_id"]) for value in sources if value["role"] == role)
        for role in ("stateless", "fit", "transform")
    }
    expected_fit_ids = fitted_ids
    observed_fit_ids = tuple(str(value["system_id"]) for value in fit_receipts)
    if observed_fit_ids != expected_fit_ids:
        raise DenoisingCanaryError("fit receipts", "do not match fitted systems")
    expected_transform_pairs = tuple(
        [
            (system_id, record_id)
            for system_id in stateless_ids
            for record_id in source_ids_by_role["stateless"]
        ]
        + [
            (system_id, record_id)
            for system_id in fitted_ids
            for record_id in source_ids_by_role["transform"]
        ]
    )
    observed_pairs = tuple(
        (str(value["system_id"]), str(value["record_id"]))
        for value in transform_receipts
    )
    if observed_pairs != expected_transform_pairs:
        raise DenoisingCanaryError("transform receipts", "Cartesian order mismatch")
    if set(source_ids_by_role["fit"]) & set(source_ids_by_role["transform"]):
        raise DenoisingCanaryError("source roles", "fit and transform overlap")

    fit_descriptors = [
        descriptor
        for receipt in fit_receipts
        for descriptor in receipt["arrays"]
    ]
    output_descriptors = [
        receipt["output"]
        for receipt in transform_receipts
        if receipt["output"] is not None
    ]
    _verify_stream(
        path / "fitted_states.f64le", fit_descriptors, label="fitted state stream"
    )
    _verify_stream(
        path / "denoised_outputs.f64le", output_descriptors, label="output stream"
    )
    state_raw = (path / "fitted_states.f64le").read_bytes()
    fit_record_ids = source_ids_by_role["fit"]
    fit_context = DenoisingFitContext(
        split_id="bacteria_reference_min4_per_class_v1",
        representation_id="bacteria_id_reference_increasing_float64_v1",
        record_ids=fit_record_ids,
    )
    training_record_ledger_sha256 = hashlib.sha256(
        b"".join(_canonical({"record_id": value}) for value in fit_record_ids)
    ).hexdigest()
    fitted_state_by_system: dict[str, str] = {}
    successful_training_matrix_hashes: set[str] = set()
    for system, receipt in zip(
        (value for value in catalog_systems if value.family_id in _FITTED_FAMILIES),
        fit_receipts,
        strict=True,
    ):
        status = str(receipt["status"])
        warnings = _receipt_warnings(receipt["warnings"], label="fit receipt identity")
        if status in _SUCCESS:
            if (
                receipt["family_id"] != system.family_id
                or receipt["method_id"] != system.method_id
                or receipt["axis_sha256"]
                != next(
                    iter(
                        {
                            str(value["axis_sha256"])
                            for value in sources
                            if value["role"] == "fit"
                        }
                    )
                )
                or receipt["context_sha256"] != fit_context.context_sha256
                or receipt["training_record_ledger_sha256"]
                != training_record_ledger_sha256
                or receipt["error_code"] is not None
                or receipt["error_message"] is not None
            ):
                raise DenoisingCanaryError(
                    "fit receipt identity", "does not match frozen fit context"
                )
            descriptors = receipt["arrays"]
            if not isinstance(descriptors, list):
                raise DenoisingCanaryError(
                    "fit receipt identity", "arrays must be a list"
                )
            by_role = {str(value["role"]): value for value in descriptors}
            expected_roles = (
                {"mean", "components", "explained_variance"}
                if system.family_id == "pca_reconstruction"
                else {"components", "explained_variance"}
            )
            if set(by_role) != expected_roles or len(by_role) != len(descriptors):
                raise DenoisingCanaryError(
                    "fit receipt identity", "array roles do not match family"
                )
            components = int(system.hyperparameters["n_components"])
            mean = (
                _descriptor_array(state_raw, by_role["mean"], label="fit receipt identity")
                if "mean" in by_role
                else None
            )
            component_array = _descriptor_array(
                state_raw, by_role["components"], label="fit receipt identity"
            )
            explained = _descriptor_array(
                state_raw,
                by_role["explained_variance"],
                label="fit receipt identity",
            )
            training_matrix_sha256 = str(receipt["training_matrix_sha256"])
            fitted = FittedDenoiser(
                system_id=system.system_id,
                family_id=system.family_id,
                method_id=system.method_id,
                n_components=components,
                axis_sha256=str(receipt["axis_sha256"]),
                training_record_ledger_sha256=str(
                    receipt["training_record_ledger_sha256"]
                ),
                training_matrix_sha256=training_matrix_sha256,
                context_sha256=str(receipt["context_sha256"]),
                mean=mean,
                components=component_array,
                explained_variance=explained,
                warnings=warnings,
            )
            if fitted.state_sha256 != receipt["state_sha256"]:
                raise DenoisingCanaryError(
                    "fit receipt identity", "state SHA256 does not match arrays"
                )
            fitted_state_by_system[system.system_id] = fitted.state_sha256
            successful_training_matrix_hashes.add(training_matrix_sha256)
        elif (
            receipt["arrays"]
            or receipt["state_sha256"] is not None
            or receipt["axis_sha256"] is not None
            or receipt["context_sha256"] is not None
        ):
            raise DenoisingCanaryError(
                "fit receipt identity", "failure carries fitted state"
            )
    if len(successful_training_matrix_hashes) > 1:
        raise DenoisingCanaryError(
            "fit receipt identity", "systems bind different training matrices"
        )
    for receipt in transform_receipts:
        successful = str(receipt["status"]) in _SUCCESS
        if successful != (receipt["output"] is not None):
            raise DenoisingCanaryError("transform receipt", "output/status mismatch")
        if receipt["output"] is not None:
            descriptor = receipt["output"]
            if descriptor["shape"] != [int(receipt["point_count"])]:
                raise DenoisingCanaryError("transform receipt", "output shape mismatch")
        if str(receipt["system_id"]) in fitted_state_by_system:
            diagnostics = receipt["diagnostics"]
            if (
                not isinstance(diagnostics, Mapping)
                or diagnostics.get("fitted_state_sha256")
                != fitted_state_by_system[str(receipt["system_id"])]
            ):
                raise DenoisingCanaryError(
                    "fit receipt identity", "transform is not bound to fitted state"
                )
    expected_warnings = []
    for receipt_kind, rows in (("fit", fit_receipts), ("transform", transform_receipts)):
        for receipt in rows:
            for sequence, warning in enumerate(receipt["warnings"]):
                expected_warnings.append(
                    {
                        "category": warning["category"],
                        "message": warning["message"],
                        "record_id": receipt.get("record_id"),
                        "receipt_kind": receipt_kind,
                        "sequence": sequence,
                        "system_id": receipt["system_id"],
                    }
                )
    if warning_rows != expected_warnings:
        raise DenoisingCanaryError("warnings", "do not match receipts")

    fit_counts = Counter(str(value["status"]) for value in fit_receipts)
    transform_counts = Counter(str(value["status"]) for value in transform_receipts)
    input_identity = manifest["input_identity"]
    if not isinstance(input_identity, Mapping):
        raise DenoisingCanaryError("manifest.input_identity", "must be an object")
    if len(successful_training_matrix_hashes) == 1:
        fit_matrix_sha256 = next(iter(successful_training_matrix_hashes))
    else:
        fit_matrix_sha256 = str(input_identity.get("fit_matrix_sha256", ""))
    recomputed_input_identity = _source_input_identity(
        sources, fit_matrix_sha256=fit_matrix_sha256
    )
    if dict(input_identity) != recomputed_input_identity:
        raise DenoisingCanaryError(
            "manifest.input_identity", "does not match source and fit receipts"
        )
    is_frozen = (
        system_ids == config.system_ids
        and len(stateless_ids) == config.expected["stateless_system_count"]
        and len(fitted_ids) == config.expected["fitted_system_count"]
        and len(source_ids_by_role["stateless"])
        == config.expected["stateless_source_count"]
        and len(source_ids_by_role["transform"])
        == config.expected["transform_source_count"]
        and len(fit_receipts) == config.expected["fit_receipt_count"]
        and len(transform_receipts) == config.expected["transform_receipt_count"]
        and _matches_frozen_input_identity(input_identity, config)
    )
    all_success = set(fit_counts).issubset(_SUCCESS) and set(transform_counts).issubset(_SUCCESS)
    canary_passed = is_frozen and all_success
    expected_summary = {
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "fit_receipt_count": len(fit_receipts),
        "fit_status_counts": dict(sorted(fit_counts.items())),
        "is_frozen_canary": is_frozen,
        "run_id": manifest["run_id"],
        "system_count": len(systems),
        "transform_receipt_count": len(transform_receipts),
        "transform_status_counts": dict(sorted(transform_counts.items())),
        "warning_count": len(warning_rows),
    }
    if summary != expected_summary:
        raise DenoisingCanaryError("summary", "does not match receipts")
    identity = {
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": catalog.sha256,
        "code_identity": _code_identity(root),
        "config_sha256": config.sha256,
        "input_identity": dict(input_identity),
        "phase3_lock_sha256": config.phase3_lock_sha256,
        "software_identity": _software_identity(),
        "system_ids": list(system_ids),
    }
    expected_run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()
    expected_manifest = {
        **identity,
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "fit_receipt_count": len(fit_receipts),
        "is_frozen_canary": is_frozen,
        "run_id": expected_run_id,
        "schema_version": "phase3-denoising-v1-canary-run-v1",
        "source_count": len(sources),
        "system_count": len(systems),
        "transform_receipt_count": len(transform_receipts),
    }
    if manifest != expected_manifest:
        raise DenoisingCanaryError("manifest", "does not match verified identities")
    marker_name = "complete.json" if canary_passed else "failed.json"
    if marker_names != [marker_name]:
        raise DenoisingCanaryError("marker", "does not match status")
    marker = _read_json(path / marker_name, "marker")
    expected_marker = {
        "canary_passed": canary_passed,
        "is_frozen_canary": is_frozen,
        "run_id": expected_run_id,
        "status": "complete" if canary_passed else "failed",
    }
    if marker != expected_marker:
        raise DenoisingCanaryError("marker", "does not match verified status")
    return DenoisingCanarySummary(
        path=path,
        run_id=expected_run_id,
        system_count=len(systems),
        fit_receipt_count=len(fit_receipts),
        transform_receipt_count=len(transform_receipts),
        fit_status_counts=fit_counts,
        transform_status_counts=transform_counts,
        canary_passed=canary_passed,
        is_frozen_canary=is_frozen,
    )


__all__ = [
    "DenoisingCanaryConfig",
    "DenoisingCanaryError",
    "DenoisingCanaryInputs",
    "DenoisingCanarySource",
    "DenoisingCanarySummary",
    "build_denoising_canary_artifact",
    "load_denoising_canary_config",
    "load_denoising_canary_inputs",
    "verify_denoising_canary_artifact",
]
