from __future__ import annotations

import hashlib
import math
import ntpath
import numbers
import posixpath
import re
import struct
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Mapping, TypeAlias
from urllib.parse import urlsplit

import numpy as np


SCHEMA_VERSION = "0.1.0"
DATASET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | Mapping[str, "JsonValue"]
)


class SchemaValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class PreprocessingStatus(str, Enum):
    KNOWN_RAW = "known_raw"
    KNOWN_CORRECTED = "known_corrected"
    UNKNOWN = "unknown"


class LicenseStatus(str, Enum):
    STANDARDIZED = "standardized"
    SOURCE_CLAIM = "source_claim"
    NOT_STATED = "not_stated"
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class PreprocessingStep:
    operation: str
    description: str
    evidence: str


@dataclass(frozen=True)
class PeakTarget:
    pos_cm1: float
    height: float | None
    fwhm: float | None
    assignment: str | None


@dataclass(frozen=True)
class Targets:
    clean: np.ndarray | None = None
    baseline: np.ndarray | None = None
    peaks: tuple[PeakTarget, ...] | None = None
    class_label: int | None = None
    concentration: float | None = None
    concentrations: Mapping[str, float] | None = None


@dataclass(frozen=True)
class Provenance:
    source_url: str
    license: str
    license_status: LicenseStatus
    sha256: str
    retrieved_date: date
    source_artifact: str | None


@dataclass(frozen=True)
class SpectrumMetadata:
    dataset_id: str
    sample_id: str | None
    instrument: str | None
    excitation_nm: float | None
    integration_time_s: float | None
    n_accumulations: int | None
    grating: str | None
    detector: str | None
    preprocessing_status: PreprocessingStatus
    preprocessing_steps: tuple[PreprocessingStep, ...]
    source_metadata: Mapping[str, JsonValue]

    @property
    def eligible_for_preprocessing_evaluation(self) -> bool:
        return self.preprocessing_status is PreprocessingStatus.KNOWN_RAW


@dataclass(frozen=True)
class RamanRecord:
    record_id: str
    intensity: np.ndarray
    wavenumber: np.ndarray
    meta: SpectrumMetadata
    targets: Targets
    provenance: Provenance


@dataclass(frozen=True)
class ArrayRef:
    axis_id: str
    row: int


@dataclass(frozen=True)
class RecordArrays:
    intensity: np.ndarray
    wavenumber: np.ndarray
    clean: np.ndarray | None
    baseline: np.ndarray | None


def _validate_float32_vector(path: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise SchemaValidationError(path, "must be a numpy.ndarray")
    if value.dtype.kind != "f" or value.dtype.itemsize != 4:
        raise SchemaValidationError(f"{path} dtype", "must be float32")
    if value.dtype.byteorder == ">":
        raise SchemaValidationError(
            f"{path} dtype",
            "big-endian float32 is invalid",
        )
    if value.ndim != 1:
        raise SchemaValidationError(
            f"{path} dimension",
            "must be one-dimensional",
        )
    if value.size == 0:
        raise SchemaValidationError(f"{path} empty", "must not be empty")
    if not np.isfinite(value).all():
        raise SchemaValidationError(
            f"{path} finite",
            "contains non-finite values",
        )


def _validate_nonempty_string(path: str, value: object) -> None:
    if not isinstance(value, str) or value == "":
        raise SchemaValidationError(path, "must be a non-empty string")


def _validate_optional_string(path: str, value: object) -> None:
    if value is not None:
        _validate_nonempty_string(path, value)


def _validate_positive_number(path: str, value: object) -> None:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{path} boolean", "must not be Boolean")
    if not isinstance(value, numbers.Real):
        raise SchemaValidationError(path, "must be a real number")
    if not math.isfinite(float(value)):
        raise SchemaValidationError(f"{path} finite", "must be finite")
    if value <= 0:
        raise SchemaValidationError(f"{path} positive", "must be positive")


def _validate_optional_positive_number(path: str, value: object) -> None:
    if value is not None:
        _validate_positive_number(path, value)


def _validate_finite_number(path: str, value: object) -> None:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{path} boolean", "must not be Boolean")
    if not isinstance(value, numbers.Real):
        raise SchemaValidationError(path, "must be a real number")
    if not math.isfinite(float(value)):
        raise SchemaValidationError(f"{path} finite", "must be finite")


def _validate_json_value(path: str, value: object) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise SchemaValidationError(path, "JSON number must be finite")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(f"{path}[{index}]", item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(
                    path,
                    "JSON object keys must be strings",
                )
            _validate_json_value(f"{path}.{key}", item)
        return
    raise SchemaValidationError(path, "value is not JSON-compatible")


def _validate_preprocessing_step(
    path: str,
    step: PreprocessingStep,
) -> None:
    if not isinstance(step, PreprocessingStep):
        raise SchemaValidationError(path, "must be a PreprocessingStep")
    _validate_nonempty_string(f"{path}.operation", step.operation)
    _validate_nonempty_string(f"{path}.description", step.description)
    _validate_nonempty_string(f"{path}.evidence", step.evidence)


def _validate_metadata(meta: SpectrumMetadata) -> None:
    if not isinstance(meta, SpectrumMetadata):
        raise SchemaValidationError("meta", "must be SpectrumMetadata")
    if (
        not isinstance(meta.dataset_id, str)
        or DATASET_ID_PATTERN.fullmatch(meta.dataset_id) is None
    ):
        if meta.dataset_id == "":
            path = "meta.dataset_id empty"
        elif isinstance(meta.dataset_id, str) and ".." in meta.dataset_id:
            path = "meta.dataset_id traversal"
        elif (
            isinstance(meta.dataset_id, str)
            and meta.dataset_id.lower() != meta.dataset_id
        ):
            path = "meta.dataset_id uppercase"
        else:
            path = "meta.dataset_id"
        raise SchemaValidationError(path, "invalid dataset identifier")

    _validate_optional_string("meta.sample_id", meta.sample_id)
    _validate_optional_string("meta.instrument", meta.instrument)
    _validate_optional_positive_number(
        "meta.excitation_nm",
        meta.excitation_nm,
    )
    _validate_optional_positive_number(
        "meta.integration_time_s",
        meta.integration_time_s,
    )
    if meta.n_accumulations is not None:
        if isinstance(meta.n_accumulations, bool):
            raise SchemaValidationError(
                "meta.n_accumulations boolean",
                "must not be Boolean",
            )
        if not isinstance(meta.n_accumulations, numbers.Integral):
            raise SchemaValidationError(
                "meta.n_accumulations",
                "must be an integer",
            )
        if meta.n_accumulations <= 0:
            raise SchemaValidationError(
                "meta.n_accumulations positive",
                "must be positive",
            )
    _validate_optional_string("meta.grating", meta.grating)
    _validate_optional_string("meta.detector", meta.detector)

    if not isinstance(meta.preprocessing_status, PreprocessingStatus):
        raise SchemaValidationError(
            "meta.preprocessing_status",
            "must be a PreprocessingStatus",
        )
    if not isinstance(meta.preprocessing_steps, tuple):
        raise SchemaValidationError(
            "meta.preprocessing_steps",
            "must be a tuple",
        )
    if (
        meta.preprocessing_status is PreprocessingStatus.KNOWN_RAW
        and meta.preprocessing_steps
    ):
        raise SchemaValidationError(
            "meta.preprocessing_steps",
            "known_raw records cannot contain preprocessing steps",
        )
    if (
        meta.preprocessing_status is PreprocessingStatus.KNOWN_CORRECTED
        and not meta.preprocessing_steps
    ):
        raise SchemaValidationError(
            "meta.preprocessing_steps",
            "known_corrected records require a documented step",
        )
    for index, step in enumerate(meta.preprocessing_steps):
        _validate_preprocessing_step(
            f"meta.preprocessing_steps[{index}]",
            step,
        )
    if not isinstance(meta.source_metadata, Mapping):
        raise SchemaValidationError(
            "meta.source_metadata",
            "must be a mapping",
        )
    _validate_json_value("meta.source_metadata", meta.source_metadata)


def _validate_peak(path: str, peak: PeakTarget) -> None:
    if not isinstance(peak, PeakTarget):
        raise SchemaValidationError(path, "must be a PeakTarget")
    _validate_finite_number(f"{path}.pos_cm1", peak.pos_cm1)
    if peak.height is not None:
        _validate_finite_number(f"{path}.height", peak.height)
    if peak.fwhm is not None:
        _validate_positive_number(f"{path}.fwhm", peak.fwhm)
    if peak.assignment is not None and not isinstance(peak.assignment, str):
        raise SchemaValidationError(
            f"{path}.assignment",
            "must be a string or None",
        )


def _validate_targets(targets: Targets, intensity_shape: tuple[int, ...]) -> None:
    if not isinstance(targets, Targets):
        raise SchemaValidationError("targets", "must be Targets")
    for name in ("clean", "baseline"):
        value = getattr(targets, name)
        if value is None:
            continue
        path = f"targets.{name}"
        _validate_float32_vector(path, value)
        if value.shape != intensity_shape:
            raise SchemaValidationError(
                f"{path} shape",
                "must match intensity shape",
            )

    if targets.peaks is not None:
        if not isinstance(targets.peaks, tuple):
            raise SchemaValidationError(
                "targets.peaks",
                "must be a tuple or None",
            )
        for index, peak in enumerate(targets.peaks):
            _validate_peak(f"targets.peaks[{index}]", peak)

    if targets.class_label is not None:
        if isinstance(targets.class_label, bool) or not isinstance(
            targets.class_label,
            numbers.Integral,
        ):
            raise SchemaValidationError(
                "targets.class_label",
                "must be an integer and not Boolean",
            )

    if targets.concentration is not None:
        _validate_finite_number(
            "targets.concentration",
            targets.concentration,
        )
    if (
        targets.concentration is not None
        and targets.concentrations is not None
    ):
        raise SchemaValidationError(
            "targets.concentrations",
            "cannot coexist with scalar concentration",
        )
    if targets.concentrations is not None:
        if not isinstance(targets.concentrations, Mapping):
            raise SchemaValidationError(
                "targets.concentrations",
                "must be a mapping",
            )
        for name, value in targets.concentrations.items():
            if not isinstance(name, str) or name == "":
                raise SchemaValidationError(
                    "targets.concentrations key",
                    "names must be non-empty strings",
                )
            _validate_finite_number(
                f"targets.concentrations.{name}",
                value,
            )


def _validate_sha256(path: str, value: object, *, detailed: bool) -> None:
    if not isinstance(value, str):
        raise SchemaValidationError(path, "must be a string")
    if len(value) != 64:
        suffix = " short" if detailed else ""
        raise SchemaValidationError(
            f"{path}{suffix}",
            "must contain 64 characters",
        )
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        suffix = " hexadecimal" if detailed else ""
        raise SchemaValidationError(
            f"{path}{suffix}",
            "must be hexadecimal",
        )
    if value != value.lower():
        suffix = " uppercase" if detailed else ""
        raise SchemaValidationError(
            f"{path}{suffix}",
            "must use lowercase hexadecimal",
        )


def _validate_provenance(provenance: Provenance) -> None:
    if not isinstance(provenance, Provenance):
        raise SchemaValidationError(
            "provenance",
            "must be Provenance",
        )
    _validate_nonempty_string(
        "provenance.source_url",
        provenance.source_url,
    )
    parsed_url = urlsplit(provenance.source_url)
    if not parsed_url.scheme or (
        parsed_url.scheme in {"http", "https"} and not parsed_url.netloc
    ):
        raise SchemaValidationError(
            "provenance.source_url",
            "must be an absolute URI",
        )

    _validate_nonempty_string("provenance.license", provenance.license)
    if not isinstance(provenance.license_status, LicenseStatus):
        raise SchemaValidationError(
            "provenance.license_status",
            "must be a LicenseStatus",
        )
    if (
        provenance.license == "not stated"
        and provenance.license_status is not LicenseStatus.NOT_STATED
    ):
        raise SchemaValidationError(
            "provenance.license not stated",
            "requires not_stated status",
        )
    if (
        provenance.license_status is LicenseStatus.NOT_STATED
        and provenance.license != "not stated"
    ):
        raise SchemaValidationError(
            "provenance.license_status",
            "not_stated status requires literal 'not stated' license",
        )

    _validate_sha256("provenance.sha256", provenance.sha256, detailed=True)
    if type(provenance.retrieved_date) is not date:
        raise SchemaValidationError(
            "provenance.retrieved_date",
            "must be an exact date",
        )

    if provenance.source_artifact is not None:
        _validate_nonempty_string(
            "provenance.source_artifact empty",
            provenance.source_artifact,
        )
        if "\x00" in provenance.source_artifact:
            raise SchemaValidationError(
                "provenance.source_artifact NUL",
                "must not contain NUL",
            )
        if posixpath.isabs(provenance.source_artifact):
            raise SchemaValidationError(
                "provenance.source_artifact POSIX",
                "must be source-relative",
            )
        if ntpath.isabs(provenance.source_artifact):
            raise SchemaValidationError(
                "provenance.source_artifact Windows",
                "must be source-relative",
            )
        if ntpath.splitdrive(provenance.source_artifact)[0]:
            raise SchemaValidationError(
                "provenance.source_artifact Windows drive",
                "must not contain a drive prefix",
            )
        if ".." in re.split(r"[\\/]", provenance.source_artifact):
            raise SchemaValidationError(
                "provenance.source_artifact traversal",
                "must not contain parent-directory traversal",
            )


def axis_id(wavenumber: np.ndarray) -> str:
    _validate_float32_vector("wavenumber", wavenumber)
    canonical = np.ascontiguousarray(wavenumber, dtype="<f4")
    payload = (
        b"rpe-axis-v1\0"
        + struct.pack("<Q", canonical.size)
        + canonical.tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def validate_record(record: RamanRecord) -> RamanRecord:
    if not isinstance(record, RamanRecord):
        raise SchemaValidationError("root", "must be a RamanRecord")
    _validate_nonempty_string("record_id", record.record_id)
    _validate_float32_vector("intensity", record.intensity)
    _validate_float32_vector("wavenumber", record.wavenumber)
    if record.intensity.shape != record.wavenumber.shape:
        raise SchemaValidationError(
            "array length",
            "intensity and wavenumber must have equal lengths",
        )

    differences = np.diff(record.wavenumber)
    if np.any(differences == 0):
        raise SchemaValidationError(
            "wavenumber duplicate",
            "axis values must be unique",
        )
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise SchemaValidationError(
            "wavenumber monotonic",
            "axis must be strictly increasing or decreasing",
        )
    _validate_metadata(record.meta)
    _validate_targets(record.targets, record.intensity.shape)
    _validate_provenance(record.provenance)
    return record


def _validate_array_ref(array_ref: ArrayRef) -> None:
    if not isinstance(array_ref, ArrayRef):
        raise SchemaValidationError(
            "array_ref",
            "must be an ArrayRef",
        )
    _validate_sha256("array_ref.axis_id", array_ref.axis_id, detailed=False)
    if isinstance(array_ref.row, bool):
        raise SchemaValidationError(
            "array_ref.row boolean",
            "must not be Boolean",
        )
    if not isinstance(array_ref.row, numbers.Integral) or array_ref.row < 0:
        raise SchemaValidationError(
            "array_ref.row",
            "must be a non-negative integer",
        )


def _json_copy(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    return value


def record_to_metadata(
    record: RamanRecord,
    array_ref: ArrayRef,
) -> dict[str, JsonValue]:
    validate_record(record)
    _validate_array_ref(array_ref)
    if array_ref.axis_id != axis_id(record.wavenumber):
        raise SchemaValidationError(
            "array_ref.axis_id",
            "does not match record wavenumber",
        )
    return {
        "record_id": record.record_id,
        "array_ref": {
            "axis_id": array_ref.axis_id,
            "row": int(array_ref.row),
        },
        "meta": {
            "dataset_id": record.meta.dataset_id,
            "sample_id": record.meta.sample_id,
            "instrument": record.meta.instrument,
            "excitation_nm": (
                None
                if record.meta.excitation_nm is None
                else float(record.meta.excitation_nm)
            ),
            "integration_time_s": (
                None
                if record.meta.integration_time_s is None
                else float(record.meta.integration_time_s)
            ),
            "n_accumulations": (
                None
                if record.meta.n_accumulations is None
                else int(record.meta.n_accumulations)
            ),
            "grating": record.meta.grating,
            "detector": record.meta.detector,
            "preprocessing_status": record.meta.preprocessing_status.value,
            "preprocessing_steps": [
                {
                    "operation": step.operation,
                    "description": step.description,
                    "evidence": step.evidence,
                }
                for step in record.meta.preprocessing_steps
            ],
            "source_metadata": _json_copy(record.meta.source_metadata),
        },
        "targets": {
            "clean_present": record.targets.clean is not None,
            "baseline_present": record.targets.baseline is not None,
            "peaks": (
                None
                if record.targets.peaks is None
                else [
                    {
                        "pos_cm1": float(peak.pos_cm1),
                        "height": (
                            None
                            if peak.height is None
                            else float(peak.height)
                        ),
                        "fwhm": (
                            None
                            if peak.fwhm is None
                            else float(peak.fwhm)
                        ),
                        "assignment": peak.assignment,
                    }
                    for peak in record.targets.peaks
                ]
            ),
            "class_label": (
                None
                if record.targets.class_label is None
                else int(record.targets.class_label)
            ),
            "concentration": (
                None
                if record.targets.concentration is None
                else float(record.targets.concentration)
            ),
            "concentrations": (
                None
                if record.targets.concentrations is None
                else {
                    name: float(value)
                    for name, value in record.targets.concentrations.items()
                }
            ),
        },
        "provenance": {
            "source_url": record.provenance.source_url,
            "license": record.provenance.license,
            "license_status": record.provenance.license_status.value,
            "sha256": record.provenance.sha256,
            "retrieved_date": record.provenance.retrieved_date.isoformat(),
            "source_artifact": record.provenance.source_artifact,
        },
    }


def _require_mapping(
    path: str,
    value: object,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise SchemaValidationError(path, "object keys must be strings")
    return value


def _require_exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: tuple[str, ...],
) -> None:
    actual_keys = set(value)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise SchemaValidationError(
            path,
            f"key mismatch: missing={missing}, unexpected={unexpected}",
        )


def _parse_preprocessing_steps(value: object) -> tuple[PreprocessingStep, ...]:
    path = "meta.preprocessing_steps"
    if not isinstance(value, list):
        raise SchemaValidationError(path, "must be a list")
    result = []
    expected_keys = ("operation", "description", "evidence")
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        parsed = _require_mapping(item_path, item)
        _require_exact_keys(item_path, parsed, expected_keys)
        result.append(
            PreprocessingStep(
                operation=parsed["operation"],
                description=parsed["description"],
                evidence=parsed["evidence"],
            )
        )
    return tuple(result)


def _parse_peaks(value: object) -> tuple[PeakTarget, ...] | None:
    if value is None:
        return None
    path = "targets.peaks"
    if not isinstance(value, list):
        raise SchemaValidationError(path, "must be a list or null")
    result = []
    expected_keys = ("pos_cm1", "height", "fwhm", "assignment")
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        parsed = _require_mapping(item_path, item)
        _require_exact_keys(item_path, parsed, expected_keys)
        result.append(
            PeakTarget(
                pos_cm1=parsed["pos_cm1"],
                height=parsed["height"],
                fwhm=parsed["fwhm"],
                assignment=parsed["assignment"],
            )
        )
    return tuple(result)


def _parse_enum(path: str, enum_type: type[Enum], value: object) -> Enum:
    if not isinstance(value, str):
        raise SchemaValidationError(path, "must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise SchemaValidationError(path, "unsupported value") from error


def _parse_date(path: str, value: object) -> date:
    if not isinstance(value, str):
        raise SchemaValidationError(path, "must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise SchemaValidationError(path, "invalid ISO date") from error


def metadata_to_record(
    metadata: Mapping[str, JsonValue],
    arrays: RecordArrays,
) -> RamanRecord:
    root = _require_mapping("root", metadata)
    _require_exact_keys(
        "root",
        root,
        ("record_id", "array_ref", "meta", "targets", "provenance"),
    )
    array_ref_data = _require_mapping("array_ref", root["array_ref"])
    _require_exact_keys(
        "array_ref",
        array_ref_data,
        ("axis_id", "row"),
    )
    array_ref = ArrayRef(
        axis_id=array_ref_data["axis_id"],
        row=array_ref_data["row"],
    )
    _validate_array_ref(array_ref)
    if not isinstance(arrays, RecordArrays):
        raise SchemaValidationError("arrays", "must be RecordArrays")
    if array_ref.axis_id != axis_id(arrays.wavenumber):
        raise SchemaValidationError(
            "array_ref.axis_id",
            "does not match stored wavenumber",
        )

    meta_data = _require_mapping("meta", root["meta"])
    _require_exact_keys(
        "meta",
        meta_data,
        (
            "dataset_id",
            "sample_id",
            "instrument",
            "excitation_nm",
            "integration_time_s",
            "n_accumulations",
            "grating",
            "detector",
            "preprocessing_status",
            "preprocessing_steps",
            "source_metadata",
        ),
    )
    source_metadata = _require_mapping(
        "meta.source_metadata",
        meta_data["source_metadata"],
    )
    meta = SpectrumMetadata(
        dataset_id=meta_data["dataset_id"],
        sample_id=meta_data["sample_id"],
        instrument=meta_data["instrument"],
        excitation_nm=meta_data["excitation_nm"],
        integration_time_s=meta_data["integration_time_s"],
        n_accumulations=meta_data["n_accumulations"],
        grating=meta_data["grating"],
        detector=meta_data["detector"],
        preprocessing_status=_parse_enum(
            "meta.preprocessing_status",
            PreprocessingStatus,
            meta_data["preprocessing_status"],
        ),
        preprocessing_steps=_parse_preprocessing_steps(
            meta_data["preprocessing_steps"]
        ),
        source_metadata=_json_copy(source_metadata),
    )

    targets_data = _require_mapping("targets", root["targets"])
    _require_exact_keys(
        "targets",
        targets_data,
        (
            "clean_present",
            "baseline_present",
            "peaks",
            "class_label",
            "concentration",
            "concentrations",
        ),
    )
    clean_present = targets_data["clean_present"]
    baseline_present = targets_data["baseline_present"]
    if not isinstance(clean_present, bool):
        raise SchemaValidationError(
            "targets.clean_present",
            "must be Boolean",
        )
    if not isinstance(baseline_present, bool):
        raise SchemaValidationError(
            "targets.baseline_present",
            "must be Boolean",
        )
    if clean_present != (arrays.clean is not None):
        raise SchemaValidationError(
            "targets.clean_present",
            "does not match stored arrays",
        )
    if baseline_present != (arrays.baseline is not None):
        raise SchemaValidationError(
            "targets.baseline_present",
            "does not match stored arrays",
        )
    concentrations_value = targets_data["concentrations"]
    concentrations = (
        None
        if concentrations_value is None
        else dict(
            _require_mapping(
                "targets.concentrations",
                concentrations_value,
            )
        )
    )
    targets = Targets(
        clean=arrays.clean,
        baseline=arrays.baseline,
        peaks=_parse_peaks(targets_data["peaks"]),
        class_label=targets_data["class_label"],
        concentration=targets_data["concentration"],
        concentrations=concentrations,
    )

    provenance_data = _require_mapping(
        "provenance",
        root["provenance"],
    )
    _require_exact_keys(
        "provenance",
        provenance_data,
        (
            "source_url",
            "license",
            "license_status",
            "sha256",
            "retrieved_date",
            "source_artifact",
        ),
    )
    provenance = Provenance(
        source_url=provenance_data["source_url"],
        license=provenance_data["license"],
        license_status=_parse_enum(
            "provenance.license_status",
            LicenseStatus,
            provenance_data["license_status"],
        ),
        sha256=provenance_data["sha256"],
        retrieved_date=_parse_date(
            "provenance.retrieved_date",
            provenance_data["retrieved_date"],
        ),
        source_artifact=provenance_data["source_artifact"],
    )
    record = RamanRecord(
        record_id=root["record_id"],
        intensity=arrays.intensity,
        wavenumber=arrays.wavenumber,
        meta=meta,
        targets=targets,
        provenance=provenance,
    )
    return validate_record(record)
