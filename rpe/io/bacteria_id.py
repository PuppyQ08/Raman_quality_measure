from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Iterator, Mapping

import h5py
import numpy as np

from rpe.io.schema import (
    SCHEMA_VERSION,
    LicenseStatus,
    PreprocessingStatus,
    PreprocessingStep,
    Provenance,
    RamanRecord,
    SpectrumMetadata,
    Targets,
    axis_id,
    validate_record,
)
from rpe.io.store import DATASET_FILES, validate_dataset, write_dataset


ADAPTER_VERSION = "0.1.0"
ARCHIVE_SHA256 = (
    "05f0978e2bcfe96f7f89667734a925f3d2dd261337e00a6d051df30e3006074f"
)
AXIS_ID = (
    "91e468d92cd4215f23c6c785b4611dc6f1ffe39cbb9134d11c60345954f80378"
)
REFERENCE_DATASET_ID = "bacteria_id_reference"
CLINICAL_DATASET_ID = "bacteria_id_clinical"
RECEIPT_NAME = "bacteria_id_conversion.json"

SOURCE_URL = (
    "https://www.dropbox.com/sh/gmgduvzyl5tken6/"
    "AABtSWXWPjoUBkKyC2e7Ag6Da?dl=1"
)
SOURCE_LICENSE = (
    "repository MIT; dataset-specific applicability not separately stated"
)
SOURCE_DATASET_ID = "bacteria-ID"
SOURCE_AXIS_FILE = "wavenumbers.npy"
REFERENCE_SPLITS = (
    ("reference", "pretraining", 1.0),
    ("finetune", "optical_system_adaptation", 2.0),
    ("test", "independent_test", 2.0),
)
CLINICAL_SPLITS = (
    ("clinical2018", 400, None),
    ("clinical2019", 100, 2.0),
)
REFERENCE_CLASS_LABELS = MappingProxyType(
    {
        0: "C. albicans",
        1: "C. glabrata",
        2: "K. aerogenes",
        3: "E. coli 1",
        4: "E. coli 2",
        5: "E. faecium",
        6: "E. faecalis 1",
        7: "E. faecalis 2",
        8: "E. cloacae",
        9: "K. pneumoniae 1",
        10: "K. pneumoniae 2",
        11: "P. mirabilis",
        12: "P. aeruginosa 1",
        13: "P. aeruginosa 2",
        14: "MSSA 1",
        15: "MSSA 3",
        16: "MRSA 1 (isogenic)",
        17: "MRSA 2",
        18: "MSSA 2",
        19: "S. enterica",
        20: "S. epidermidis",
        21: "S. lugdunensis",
        22: "S. marcescens",
        23: "S. pneumoniae 2",
        24: "S. pneumoniae 1",
        25: "S. sanguinis",
        26: "Group A Strep.",
        27: "Group B Strep.",
        28: "Group C Strep.",
        29: "Group G Strep.",
    }
)
CLINICAL_CLASS_LABELS = MappingProxyType(
    {
        0: "Meropenem",
        2: "TZP",
        3: "Vancomycin",
        5: "Penicillin",
        6: "Daptomycin",
    }
)
PUBLISHED_PREPROCESSING = (
    PreprocessingStep(
        operation="polynomial_background_correction",
        description=(
            "Individual fifth-order polynomial background correction using "
            "MATLAB subbackmod from the Biodata toolbox"
        ),
        evidence=(
            "Nature Communications DOI 10.1038/s41467-019-12898-9 Methods"
        ),
    ),
    PreprocessingStep(
        operation="min_max_normalization",
        description=(
            "Individual spectrum normalized to minimum 0 and maximum 1 over "
            "381.98–1792.4 cm^-1"
        ),
        evidence=(
            "Nature Communications DOI 10.1038/s41467-019-12898-9 Methods"
        ),
    ),
)

_AXIS_FLOAT32_MAX_ABS_ERROR = 5.0e-5
_INTENSITY_FLOAT32_MAX_ABS_ERROR = 3.0e-8
_FLOAT64_DTYPE = np.dtype(np.float64)
_ISOLATE_LABELS = tuple(range(30))
_CLINICAL_LABELS = (0, 2, 3, 5, 6)


class BacteriaIdValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class BacteriaIdSplitInspection:
    split: str
    spectra_file: str
    target_file: str
    spectra_shape: tuple[int, int]
    target_shape: tuple[int]
    spectra_dtype: str
    target_dtype: str
    row_count: int
    class_counts: Mapping[int, int]
    role: str
    label_space: str
    clinical_spectra_per_patient: int | None
    spectra_sha256: str
    target_sha256: str


@dataclass(frozen=True)
class BacteriaIdInspection:
    raw_root: Path
    archive_sha256: str
    axis_sha256: str
    axis_id: str
    axis_shape: tuple[int]
    axis_dtype: str
    axis_float32_max_abs_error: float
    intensity_float32_max_abs_error: float
    total_rows: int
    splits: Mapping[str, BacteriaIdSplitInspection]
    clinical_patient_blocks: Mapping[str, int]


@dataclass(frozen=True)
class BacteriaIdDatasetSummary:
    dataset_id: str
    record_count: int
    axis_id: str
    output_bytes: int
    files: Mapping[str, Mapping[str, int | str]]


@dataclass(frozen=True)
class BacteriaIdConversionSummary:
    reference: BacteriaIdDatasetSummary
    clinical: BacteriaIdDatasetSummary
    receipt_path: Path
    receipt_sha256: str
    eligible_records: int


@dataclass(frozen=True)
class BacteriaIdRuntimeMetrics:
    total_seconds: float
    inspection_seconds: float
    reference_build_seconds: float
    clinical_build_seconds: float
    validation_seconds: float
    peak_rss_raw: int
    peak_rss_bytes: int
    combined_output_bytes: int
    source_bytes: int
    compression_ratio: float


@dataclass(frozen=True)
class _PublicationArtifact:
    staged: Path
    final: Path
    backup_name: str


@dataclass(frozen=True)
class _SplitContract:
    split: str
    spectra_file: str
    target_file: str
    spectra_shape: tuple[int, int]
    target_shape: tuple[int]
    expected_labels: tuple[int, ...]
    expected_count_per_label: int
    role: str
    label_space: str
    clinical_spectra_per_patient: int | None


@dataclass(frozen=True)
class _BacteriaIdContract:
    archive_sha256: str
    axis_file: str
    axis_sha256: str
    member_sha256: Mapping[str, str]
    split_specs: Mapping[str, _SplitContract]


_MEMBER_SHA256 = MappingProxyType(
    {
        "wavenumbers.npy": (
            "22e11b897a4a7ddbf1dfe3a8067ff68906655cde569a667f7978636760d104f7"
        ),
        "X_reference.npy": (
            "80040a919a6346b410745322b5593227823c77fb9ea4ae9b4bf7f4aef2a07e45"
        ),
        "y_reference.npy": (
            "f7458e2a5256ae828c6c00d93cd5e6b9f7d04a455860c77382408b04943ac85f"
        ),
        "X_finetune.npy": (
            "361acaf2fa8fe868c90c33d3321a6045c1ea6be915c5dafae8a3d41a9bb900ad"
        ),
        "y_finetune.npy": (
            "aece34e11169d5cd9d13124b666d3e4d69f38f582817bcf66218abf6a9d55edb"
        ),
        "X_test.npy": (
            "ec43c85b44bf71ed6fabfcb4ddfde3d751d85e5a96be1fed9bb2b9ab1cd1f6e4"
        ),
        "y_test.npy": (
            "aece34e11169d5cd9d13124b666d3e4d69f38f582817bcf66218abf6a9d55edb"
        ),
        "X_2018clinical.npy": (
            "9e23eecf4fb43c32313c1111d144720764bb93ac7887fe9bcfd5223ec151db29"
        ),
        "y_2018clinical.npy": (
            "968f1eb87996ff36a2388f7df03e8fff4b177bc0a32e131cc4b9603ca4cc4ae6"
        ),
        "X_2019clinical.npy": (
            "79235c885d66f4013647458154387863e717c78d27082ee457444321b90dab86"
        ),
        "y_2019clinical.npy": (
            "705deee65ebf258582ff2dbe236a8145c8deb1ca5ebac9a53a9527fd174344dc"
        ),
    }
)

_PRODUCTION_SPLITS = MappingProxyType(
    {
        "reference": _SplitContract(
            split="reference",
            spectra_file="X_reference.npy",
            target_file="y_reference.npy",
            spectra_shape=(60000, 1000),
            target_shape=(60000,),
            expected_labels=_ISOLATE_LABELS,
            expected_count_per_label=2000,
            role="pretraining",
            label_space="isolate",
            clinical_spectra_per_patient=None,
        ),
        "finetune": _SplitContract(
            split="finetune",
            spectra_file="X_finetune.npy",
            target_file="y_finetune.npy",
            spectra_shape=(3000, 1000),
            target_shape=(3000,),
            expected_labels=_ISOLATE_LABELS,
            expected_count_per_label=100,
            role="optical_system_adaptation",
            label_space="isolate",
            clinical_spectra_per_patient=None,
        ),
        "test": _SplitContract(
            split="test",
            spectra_file="X_test.npy",
            target_file="y_test.npy",
            spectra_shape=(3000, 1000),
            target_shape=(3000,),
            expected_labels=_ISOLATE_LABELS,
            expected_count_per_label=100,
            role="independent_test",
            label_space="isolate",
            clinical_spectra_per_patient=None,
        ),
        "clinical2018": _SplitContract(
            split="clinical2018",
            spectra_file="X_2018clinical.npy",
            target_file="y_2018clinical.npy",
            spectra_shape=(10000, 1000),
            target_shape=(10000,),
            expected_labels=_CLINICAL_LABELS,
            expected_count_per_label=2000,
            role="clinical_test",
            label_space="treatment",
            clinical_spectra_per_patient=400,
        ),
        "clinical2019": _SplitContract(
            split="clinical2019",
            spectra_file="X_2019clinical.npy",
            target_file="y_2019clinical.npy",
            spectra_shape=(2500, 1000),
            target_shape=(2500,),
            expected_labels=_CLINICAL_LABELS,
            expected_count_per_label=500,
            role="clinical_test",
            label_space="treatment",
            clinical_spectra_per_patient=100,
        ),
    }
)

PRODUCTION_CONTRACT = _BacteriaIdContract(
    archive_sha256=ARCHIVE_SHA256,
    axis_file="wavenumbers.npy",
    axis_sha256=_MEMBER_SHA256["wavenumbers.npy"],
    member_sha256=_MEMBER_SHA256,
    split_specs=_PRODUCTION_SPLITS,
)


def _sha256_file(path: Path, error_path: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BacteriaIdValidationError(
            error_path,
            f"cannot read required file: {error}",
        ) from error
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, error_path: str) -> str:
    actual = _sha256_file(path, error_path)
    if actual != expected:
        raise BacteriaIdValidationError(
            f"{error_path}.sha256",
            f"expected {expected}, observed {actual}",
        )
    return actual


def _load_array(path: Path, error_path: str) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise BacteriaIdValidationError(
            f"{error_path}.load",
            f"cannot load required NumPy array: {error}",
        ) from error


def _require_shape(
    array: np.ndarray,
    expected: tuple[int, ...],
    error_path: str,
) -> None:
    if array.shape != expected:
        raise BacteriaIdValidationError(
            f"{error_path}.shape",
            f"expected {expected}, observed {array.shape}",
        )


def _require_float64(array: np.ndarray, error_path: str) -> None:
    if array.dtype != _FLOAT64_DTYPE:
        raise BacteriaIdValidationError(
            f"{error_path}.dtype",
            f"expected float64, observed {array.dtype}",
        )


def _require_finite(array: np.ndarray, error_path: str) -> None:
    if not np.isfinite(array).all():
        raise BacteriaIdValidationError(
            f"{error_path}.finite",
            "contains non-finite values",
        )


def _spectra_cast_error(spectra: np.ndarray, error_path: str) -> float:
    maximum = 0.0
    for start in range(0, spectra.shape[0], 1024):
        block = spectra[start : start + 1024]
        _require_finite(block, error_path)
        float32 = np.asarray(block, dtype=np.float32)
        block_error = float(
            np.max(np.abs(block - float32.astype(np.float64))),
        )
        maximum = max(maximum, block_error)
    return maximum


def _validated_labels(
    targets: np.ndarray,
    spec: _SplitContract,
    error_path: str,
) -> tuple[np.ndarray, dict[int, int], int | None]:
    _require_finite(targets, error_path)
    if not np.equal(targets, np.trunc(targets)).all():
        raise BacteriaIdValidationError(
            f"{error_path}.integral",
            "labels must be exact integer values",
        )

    labels = np.asarray(targets, dtype=np.int64)
    observed = set(int(value) for value in np.unique(labels))
    expected = set(spec.expected_labels)
    if observed != expected:
        raise BacteriaIdValidationError(
            f"{error_path}.labels",
            f"expected {sorted(expected)}, observed {sorted(observed)}",
        )

    class_counts = {
        label: int(np.count_nonzero(labels == label))
        for label in spec.expected_labels
    }
    block_size = spec.clinical_spectra_per_patient
    if block_size is None:
        expected_counts = {
            label: spec.expected_count_per_label
            for label in spec.expected_labels
        }
        if class_counts != expected_counts:
            raise BacteriaIdValidationError(
                f"{error_path}.class_counts",
                f"expected {expected_counts}, observed {class_counts}",
            )
        return labels, class_counts, None

    if labels.size % block_size != 0:
        raise BacteriaIdValidationError(
            f"{error_path}.patient_blocks.size",
            f"{labels.size} labels are not divisible into blocks of {block_size}",
        )
    blocks = labels.reshape(-1, block_size)
    if not np.equal(blocks, blocks[:, :1]).all():
        raise BacteriaIdValidationError(
            f"{error_path}.patient_blocks.uniform",
            "a patient block crosses treatment labels",
        )

    observed_blocks = Counter(int(label) for label in blocks[:, 0])
    expected_blocks_per_label = spec.expected_count_per_label // block_size
    expected_blocks = {
        label: expected_blocks_per_label
        for label in spec.expected_labels
    }
    if dict(observed_blocks) != expected_blocks:
        raise BacteriaIdValidationError(
            f"{error_path}.patient_blocks.counts",
            f"expected {expected_blocks}, observed {dict(observed_blocks)}",
        )
    return labels, class_counts, blocks.shape[0]


def inspect_bacteria_id(raw_root: Path) -> BacteriaIdInspection:
    return _inspect_bacteria_id(raw_root, PRODUCTION_CONTRACT)


def _shared_source_metadata(
    inspection: BacteriaIdInspection,
    split_inspection: BacteriaIdSplitInspection,
    row: int,
) -> dict[str, object]:
    return {
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_split": split_inspection.split,
        "source_row": row,
        "split_role": split_inspection.role,
        "label_space": split_inspection.label_space,
        "source_spectra_file": split_inspection.spectra_file,
        "source_target_file": split_inspection.target_file,
        "source_axis_file": SOURCE_AXIS_FILE,
        "source_spectra_sha256": split_inspection.spectra_sha256,
        "source_target_sha256": split_inspection.target_sha256,
        "source_axis_sha256": inspection.axis_sha256,
        "source_intensity_dtype": split_inspection.spectra_dtype,
        "stored_intensity_dtype": "float32",
        "source_axis_dtype": inspection.axis_dtype,
        "stored_axis_dtype": "float32",
        "measurement_mode": "SERS on gold-coated silica",
        "published_preprocessing": [
            "polynomial_background_correction",
            "min_max_normalization",
        ],
    }


def _provenance(inspection: BacteriaIdInspection) -> Provenance:
    return Provenance(
        source_url=SOURCE_URL,
        license=SOURCE_LICENSE,
        license_status=LicenseStatus.SOURCE_CLAIM,
        sha256=inspection.archive_sha256,
        retrieved_date=date(2026, 8, 14),
        source_artifact="data.zip",
    )


def _metadata(
    dataset_id: str,
    sample_id: str | None,
    integration_time_s: float | None,
    source_metadata: Mapping[str, object],
) -> SpectrumMetadata:
    return SpectrumMetadata(
        dataset_id=dataset_id,
        sample_id=sample_id,
        instrument="Horiba LabRAM HR Evolution",
        excitation_nm=633.0,
        integration_time_s=integration_time_s,
        n_accumulations=None,
        grating="300 lines/mm",
        detector=None,
        preprocessing_status=PreprocessingStatus.KNOWN_CORRECTED,
        preprocessing_steps=PUBLISHED_PREPROCESSING,
        source_metadata=source_metadata,
    )


def _load_record_arrays(
    raw_root: Path,
    split_inspection: BacteriaIdSplitInspection,
) -> tuple[np.ndarray, np.ndarray]:
    spectra = _load_array(
        raw_root / "extracted" / split_inspection.spectra_file,
        f"extracted/{split_inspection.spectra_file}",
    )
    targets = _load_array(
        raw_root / "extracted" / split_inspection.target_file,
        f"extracted/{split_inspection.target_file}",
    )
    return spectra, targets


def _require_matching_inspection_root(
    raw_root: Path,
    inspection: BacteriaIdInspection,
) -> None:
    if inspection.raw_root != raw_root:
        raise BacteriaIdValidationError(
            "inspection.raw_root",
            (
                f"expected {raw_root}, "
                f"observed {inspection.raw_root}"
            ),
        )


def iter_reference_records(
    raw_root: Path,
    inspection: BacteriaIdInspection | None = None,
) -> Iterator[RamanRecord]:
    raw_root = Path(raw_root)
    if inspection is None:
        inspection = inspect_bacteria_id(raw_root)
    _require_matching_inspection_root(raw_root, inspection)
    axis = np.array(
        _load_array(
            raw_root / "extracted" / SOURCE_AXIS_FILE,
            f"extracted/{SOURCE_AXIS_FILE}",
        ),
        dtype=np.float32,
        copy=True,
    )
    provenance = _provenance(inspection)

    for split, role, integration_time_s in REFERENCE_SPLITS:
        split_inspection = inspection.splits[split]
        spectra, targets = _load_record_arrays(raw_root, split_inspection)
        for row in range(split_inspection.row_count):
            label = int(targets[row])
            source_metadata = _shared_source_metadata(
                inspection,
                split_inspection,
                row,
            )
            source_metadata.update(
                {
                    "isolate_id": label,
                    "isolate_name": REFERENCE_CLASS_LABELS[label],
                }
            )
            if split == "reference":
                source_metadata.update(
                    {
                        "reference_collection_measurement_time_points": 3,
                        "per_row_measurement_time_point_available": False,
                    }
                )
            record = RamanRecord(
                record_id=f"{split}-{row:06d}",
                intensity=np.array(
                    spectra[row],
                    dtype=np.float32,
                    copy=True,
                ),
                wavenumber=axis.copy(),
                meta=_metadata(
                    REFERENCE_DATASET_ID,
                    None,
                    integration_time_s,
                    source_metadata,
                ),
                targets=Targets(class_label=label),
                provenance=provenance,
            )
            yield validate_record(record)


def iter_clinical_records(
    raw_root: Path,
    inspection: BacteriaIdInspection | None = None,
) -> Iterator[RamanRecord]:
    raw_root = Path(raw_root)
    if inspection is None:
        inspection = inspect_bacteria_id(raw_root)
    _require_matching_inspection_root(raw_root, inspection)
    axis = np.array(
        _load_array(
            raw_root / "extracted" / SOURCE_AXIS_FILE,
            f"extracted/{SOURCE_AXIS_FILE}",
        ),
        dtype=np.float32,
        copy=True,
    )
    provenance = _provenance(inspection)

    for split, _production_block_size, integration_time_s in CLINICAL_SPLITS:
        split_inspection = inspection.splits[split]
        block_size = split_inspection.clinical_spectra_per_patient
        if block_size is None:
            raise BacteriaIdValidationError(
                f"inspection.splits.{split}.clinical_spectra_per_patient",
                "clinical split requires a patient block size",
            )
        spectra, targets = _load_record_arrays(raw_root, split_inspection)
        for row in range(split_inspection.row_count):
            label = int(targets[row])
            patient_block_index = row // block_size
            blocks_per_treatment = (
                split_inspection.class_counts[label] // block_size
            )
            patient_index_within_treatment = (
                patient_block_index % blocks_per_treatment
            )
            spectrum_index_within_patient = row % block_size
            sample_id = (
                f"{split}-treatment-{label:02d}-patient-block-"
                f"{patient_index_within_treatment:02d}"
            )
            source_metadata = _shared_source_metadata(
                inspection,
                split_inspection,
                row,
            )
            source_metadata.update(
                {
                    "treatment_id": label,
                    "treatment_name": CLINICAL_CLASS_LABELS[label],
                    "patient_block_index": patient_block_index,
                    "patient_index_within_treatment": (
                        patient_index_within_treatment
                    ),
                    "spectrum_index_within_patient": (
                        spectrum_index_within_patient
                    ),
                    "sample_id_is_surrogate": True,
                    "sample_id_evidence": (
                        "pinned public notebook contiguous block indexing"
                    ),
                }
            )
            if split == "clinical2018":
                source_metadata.update(
                    {
                        "likely_integration_time_s": 1.0,
                        "canonical_integration_time_s_unknown_reason": (
                            "likely 1 s by contrast, but not uniquely asserted "
                            "for released X_2018clinical.npy"
                        ),
                    }
                )
            record = RamanRecord(
                record_id=f"{split}-{row:06d}",
                intensity=np.array(
                    spectra[row],
                    dtype=np.float32,
                    copy=True,
                ),
                wavenumber=axis.copy(),
                meta=_metadata(
                    CLINICAL_DATASET_ID,
                    sample_id,
                    integration_time_s,
                    source_metadata,
                ),
                targets=Targets(class_label=label),
                provenance=provenance,
            )
            yield validate_record(record)


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


def _read_canonical_json(path: Path, error_path: str) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BacteriaIdValidationError(error_path, str(error)) from error
    if not isinstance(value, Mapping):
        raise BacteriaIdValidationError(error_path, "must be a JSON object")
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise BacteriaIdValidationError(error_path, str(error)) from error
    if raw != canonical:
        raise BacteriaIdValidationError(
            error_path,
            "must use canonical JSON encoding",
        )
    return value


def _read_canonical_jsonl(path: Path, dataset_id: str) -> list[Mapping[str, object]]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise BacteriaIdValidationError(
            f"{dataset_id}.records.jsonl",
            str(error),
        ) from error
    records = []
    for index, line in enumerate(lines):
        error_path = f"{dataset_id}.records.jsonl[{index}]"
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {constant}")
                ),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise BacteriaIdValidationError(error_path, str(error)) from error
        if not isinstance(value, Mapping):
            raise BacteriaIdValidationError(
                error_path,
                "must be a JSON object",
            )
        if line != _canonical_json_bytes(value):
            raise BacteriaIdValidationError(
                error_path,
                "must use canonical JSON encoding",
            )
        records.append(value)
    return records


def _dataset_split_names(dataset_id: str) -> tuple[str, ...]:
    if dataset_id == REFERENCE_DATASET_ID:
        return tuple(split for split, _, _ in REFERENCE_SPLITS)
    if dataset_id == CLINICAL_DATASET_ID:
        return tuple(split for split, _, _ in CLINICAL_SPLITS)
    raise BacteriaIdValidationError(
        "dataset_id",
        f"unsupported Bacteria-ID dataset {dataset_id!r}",
    )


def _compare_bacteria_id_dataset(
    raw_root: Path,
    dataset_path: Path,
    inspection: BacteriaIdInspection,
    *,
    dataset_id: str,
) -> int:
    raw_root = Path(raw_root)
    dataset_path = Path(dataset_path)
    _require_matching_inspection_root(raw_root, inspection)
    if dataset_path.name != dataset_id:
        raise BacteriaIdValidationError(
            f"{dataset_id}.path",
            "dataset basename does not match dataset ID",
        )

    split_names = _dataset_split_names(dataset_id)
    expected_count = sum(
        inspection.splits[split].row_count for split in split_names
    )
    records = _read_canonical_jsonl(
        dataset_path / "records.jsonl",
        dataset_id,
    )
    if len(records) != expected_count:
        raise BacteriaIdValidationError(
            f"{dataset_id}.records.jsonl",
            f"expected {expected_count} records, observed {len(records)}",
        )

    source_spectra = {}
    source_targets = {}
    for split in split_names:
        split_inspection = inspection.splits[split]
        source_spectra[split], source_targets[split] = _load_record_arrays(
            raw_root,
            split_inspection,
        )
    source_axis = np.asarray(
        _load_array(
            raw_root / "extracted" / SOURCE_AXIS_FILE,
            f"extracted/{SOURCE_AXIS_FILE}",
        ),
        dtype=np.float32,
    )

    eligible_records = 0
    try:
        with h5py.File(dataset_path / "arrays.h5", "r") as arrays:
            axes = arrays["axes"]
            if set(axes) != {inspection.axis_id}:
                raise BacteriaIdValidationError(
                    f"{dataset_id}.arrays.h5.axes",
                    (
                        f"expected only {inspection.axis_id}, "
                        f"observed {sorted(axes)}"
                    ),
                )
            group = axes[inspection.axis_id]
            stored_axis = np.asarray(group["wavenumber"][:])
            if not np.array_equal(stored_axis, source_axis):
                raise BacteriaIdValidationError(
                    f"{dataset_id}.arrays.h5.wavenumber",
                    "stored axis differs from source float32 conversion",
                )
            intensity = group["intensity"]
            if intensity.shape != (expected_count, source_axis.size):
                raise BacteriaIdValidationError(
                    f"{dataset_id}.arrays.h5.intensity.shape",
                    (
                        f"expected {(expected_count, source_axis.size)}, "
                        f"observed {intensity.shape}"
                    ),
                )

            for start in range(0, expected_count, 512):
                stop = min(start + 512, expected_count)
                stored_chunk = np.asarray(
                    intensity[start:stop, :],
                    dtype=np.float32,
                )
                expected_chunk = np.empty_like(stored_chunk)
                for offset, record in enumerate(records[start:stop]):
                    record_index = start + offset
                    record_path = (
                        f"{dataset_id}.records.jsonl[{record_index}]"
                    )
                    try:
                        record_id = record["record_id"]
                        array_ref = record["array_ref"]
                        metadata = record["meta"]["source_metadata"]
                        preprocessing_status = record["meta"][
                            "preprocessing_status"
                        ]
                        class_label = record["targets"]["class_label"]
                        split = metadata["source_split"]
                        source_row = metadata["source_row"]
                        split_role = metadata["split_role"]
                        label_space = metadata["label_space"]
                    except (KeyError, TypeError) as error:
                        raise BacteriaIdValidationError(
                            record_path,
                            f"missing or invalid record field: {error}",
                        ) from error
                    if split not in split_names:
                        raise BacteriaIdValidationError(
                            f"{record_path}.meta.source_metadata.source_split",
                            f"unexpected split {split!r}",
                        )
                    if (
                        isinstance(source_row, bool)
                        or not isinstance(source_row, int)
                        or source_row < 0
                        or source_row >= inspection.splits[split].row_count
                    ):
                        raise BacteriaIdValidationError(
                            f"{record_path}.meta.source_metadata.source_row",
                            f"invalid source row {source_row!r}",
                        )
                    split_inspection = inspection.splits[split]
                    if split_role != split_inspection.role:
                        raise BacteriaIdValidationError(
                            f"{record_path}.meta.source_metadata.split_role",
                            (
                                f"expected {split_inspection.role}, "
                                f"observed {split_role}"
                            ),
                        )
                    if label_space != split_inspection.label_space:
                        raise BacteriaIdValidationError(
                            f"{record_path}.meta.source_metadata.label_space",
                            (
                                f"expected {split_inspection.label_space}, "
                                f"observed {label_space}"
                            ),
                        )
                    expected_id = f"{split}-{source_row:06d}"
                    if record_id != expected_id:
                        raise BacteriaIdValidationError(
                            f"{record_path}.record_id",
                            f"expected {expected_id}, observed {record_id}",
                        )
                    if (
                        not isinstance(array_ref, Mapping)
                        or array_ref.get("axis_id") != inspection.axis_id
                        or array_ref.get("row") != record_index
                    ):
                        raise BacteriaIdValidationError(
                            f"{record_path}.array_ref",
                            "does not match canonical HDF5 row",
                        )
                    expected_label = int(source_targets[split][source_row])
                    if class_label != expected_label:
                        raise BacteriaIdValidationError(
                            f"{record_path}.targets.class_label",
                            (
                                f"expected {expected_label}, "
                                f"observed {class_label}"
                            ),
                        )
                    if preprocessing_status == PreprocessingStatus.KNOWN_RAW.value:
                        eligible_records += 1
                    elif (
                        preprocessing_status
                        != PreprocessingStatus.KNOWN_CORRECTED.value
                    ):
                        raise BacteriaIdValidationError(
                            f"{record_path}.meta.preprocessing_status",
                            f"unexpected status {preprocessing_status!r}",
                        )
                    expected_chunk[offset] = np.asarray(
                        source_spectra[split][source_row],
                        dtype=np.float32,
                    )
                if not np.array_equal(stored_chunk, expected_chunk):
                    raise BacteriaIdValidationError(
                        f"{dataset_id}.arrays.h5.intensity",
                        f"stored rows {start}:{stop} differ from source",
                    )
    except BacteriaIdValidationError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise BacteriaIdValidationError(
            f"{dataset_id}.arrays.h5",
            str(error),
        ) from error

    if eligible_records != 0:
        raise BacteriaIdValidationError(
            f"{dataset_id}.eligible_records",
            f"expected 0, observed {eligible_records}",
        )
    return eligible_records


def _dataset_file_summary(dataset_path: Path) -> Mapping[str, Mapping[str, int | str]]:
    files = {}
    for name in DATASET_FILES:
        path = dataset_path / name
        files[name] = MappingProxyType(
            {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path, f"{dataset_path.name}/{name}"),
            }
        )
    return MappingProxyType(files)


def _dataset_summary(
    dataset_path: Path,
    *,
    dataset_id: str,
    record_count: int,
    axis_id_value: str,
) -> BacteriaIdDatasetSummary:
    files = _dataset_file_summary(dataset_path)
    return BacteriaIdDatasetSummary(
        dataset_id=dataset_id,
        record_count=record_count,
        axis_id=axis_id_value,
        output_bytes=sum(int(details["bytes"]) for details in files.values()),
        files=files,
    )


def _receipt_members(
    inspection: BacteriaIdInspection,
) -> dict[str, object]:
    members = {
        SOURCE_AXIS_FILE: {
            "sha256": inspection.axis_sha256,
            "shape": list(inspection.axis_shape),
            "dtype": inspection.axis_dtype,
        }
    }
    for split_inspection in inspection.splits.values():
        members[split_inspection.spectra_file] = {
            "sha256": split_inspection.spectra_sha256,
            "shape": list(split_inspection.spectra_shape),
            "dtype": split_inspection.spectra_dtype,
        }
        members[split_inspection.target_file] = {
            "sha256": split_inspection.target_sha256,
            "shape": list(split_inspection.target_shape),
            "dtype": split_inspection.target_dtype,
        }
    return members


def _receipt_splits(
    inspection: BacteriaIdInspection,
) -> dict[str, object]:
    return {
        split: {
            "records": split_inspection.row_count,
            "class_counts": {
                str(label): count
                for label, count in split_inspection.class_counts.items()
            },
            "role": split_inspection.role,
            "label_space": split_inspection.label_space,
        }
        for split, split_inspection in inspection.splits.items()
    }


def _receipt_output(summary: BacteriaIdDatasetSummary) -> dict[str, object]:
    return {
        "records": summary.record_count,
        "axis_id": summary.axis_id,
        "files": {
            name: dict(details) for name, details in summary.files.items()
        },
    }


def _receipt_document(
    raw_root: Path,
    inspection: BacteriaIdInspection,
    reference: BacteriaIdDatasetSummary,
    clinical: BacteriaIdDatasetSummary,
    *,
    eligible_records: int,
) -> dict[str, object]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "archive_file": "data.zip",
            "archive_url": SOURCE_URL,
            "archive_bytes": (raw_root / "data.zip").stat().st_size,
            "archive_sha256": inspection.archive_sha256,
            "retrieved_date": "2026-08-14",
        },
        "members": _receipt_members(inspection),
        "splits": _receipt_splits(inspection),
        "clinical_patient_blocks": dict(
            inspection.clinical_patient_blocks
        ),
        "cast_errors": {
            "axis_float32_max_abs_error_cm1": (
                inspection.axis_float32_max_abs_error
            ),
            "intensity_float32_max_abs_error": (
                inspection.intensity_float32_max_abs_error
            ),
        },
        "outputs": {
            REFERENCE_DATASET_ID: _receipt_output(reference),
            CLINICAL_DATASET_ID: _receipt_output(clinical),
        },
        "preprocessing": {
            "status": PreprocessingStatus.KNOWN_CORRECTED.value,
            "eligible_records": eligible_records,
            "total_records": reference.record_count + clinical.record_count,
            "steps": [step.operation for step in PUBLISHED_PREPROCESSING],
        },
        "license": {
            "text": SOURCE_LICENSE,
            "status": LicenseStatus.SOURCE_CLAIM.value,
            "redistribution_requires_review": True,
        },
        "known_discrepancies": [
            {
                "id": "clinical2018_patient_count",
                "paper_claim": 30,
                "released_patient_blocks": inspection.clinical_patient_blocks[
                    "clinical2018"
                ],
                "adapter_choice": (
                    "follow released arrays and pinned public notebook"
                ),
            }
        ],
    }


def _require_receipt_keys(
    path: str,
    value: object,
    expected: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BacteriaIdValidationError(path, "must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise BacteriaIdValidationError(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )
    return value


def _validate_bacteria_id_receipt(path: Path) -> Mapping[str, object]:
    receipt = _read_canonical_json(Path(path), "receipt")
    _require_receipt_keys(
        "receipt",
        receipt,
        {
            "adapter_version",
            "schema_version",
            "source",
            "members",
            "splits",
            "clinical_patient_blocks",
            "cast_errors",
            "outputs",
            "preprocessing",
            "license",
            "known_discrepancies",
        },
    )
    _require_receipt_keys(
        "receipt.source",
        receipt["source"],
        {
            "archive_file",
            "archive_url",
            "archive_bytes",
            "archive_sha256",
            "retrieved_date",
        },
    )
    members = receipt["members"]
    if not isinstance(members, Mapping):
        raise BacteriaIdValidationError(
            "receipt.members",
            "must be a JSON object",
        )
    expected_members = {
        SOURCE_AXIS_FILE,
        *(
            split_inspection.spectra_file
            for split_inspection in PRODUCTION_CONTRACT.split_specs.values()
        ),
        *(
            split_inspection.target_file
            for split_inspection in PRODUCTION_CONTRACT.split_specs.values()
        ),
    }
    if set(members) != expected_members:
        raise BacteriaIdValidationError(
            "receipt.members",
            (
                f"key mismatch: missing={sorted(expected_members - set(members))}, "
                f"unexpected={sorted(set(members) - expected_members)}"
            ),
        )
    for name, member in members.items():
        _require_receipt_keys(
            f"receipt.members.{name}",
            member,
            {"sha256", "shape", "dtype"},
        )
    splits = receipt["splits"]
    if not isinstance(splits, Mapping):
        raise BacteriaIdValidationError(
            "receipt.splits",
            "must be a JSON object",
        )
    expected_splits = set(PRODUCTION_CONTRACT.split_specs)
    if set(splits) != expected_splits:
        raise BacteriaIdValidationError(
            "receipt.splits",
            (
                f"key mismatch: missing={sorted(expected_splits - set(splits))}, "
                f"unexpected={sorted(set(splits) - expected_splits)}"
            ),
        )
    for name, split in splits.items():
        _require_receipt_keys(
            f"receipt.splits.{name}",
            split,
            {"records", "class_counts", "role", "label_space"},
        )
    _require_receipt_keys(
        "receipt.cast_errors",
        receipt["cast_errors"],
        {
            "axis_float32_max_abs_error_cm1",
            "intensity_float32_max_abs_error",
        },
    )
    clinical_patient_blocks = receipt["clinical_patient_blocks"]
    if not isinstance(clinical_patient_blocks, Mapping):
        raise BacteriaIdValidationError(
            "receipt.clinical_patient_blocks",
            "must be a JSON object",
        )
    expected_clinical_splits = {
        split for split, _, _ in CLINICAL_SPLITS
    }
    if set(clinical_patient_blocks) != expected_clinical_splits:
        raise BacteriaIdValidationError(
            "receipt.clinical_patient_blocks",
            (
                "key mismatch: "
                f"missing={sorted(expected_clinical_splits - set(clinical_patient_blocks))}, "
                f"unexpected={sorted(set(clinical_patient_blocks) - expected_clinical_splits)}"
            ),
        )
    for name, value in receipt["cast_errors"].items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise BacteriaIdValidationError(
                f"receipt.cast_errors.{name}",
                "must be finite",
            )
    outputs = receipt["outputs"]
    if not isinstance(outputs, Mapping):
        raise BacteriaIdValidationError(
            "receipt.outputs",
            "must be a JSON object",
        )
    expected_outputs = {REFERENCE_DATASET_ID, CLINICAL_DATASET_ID}
    if set(outputs) != expected_outputs:
        raise BacteriaIdValidationError(
            "receipt.outputs",
            (
                f"key mismatch: missing={sorted(expected_outputs - set(outputs))}, "
                f"unexpected={sorted(set(outputs) - expected_outputs)}"
            ),
        )
    for dataset_id, output in outputs.items():
        parsed_output = _require_receipt_keys(
            f"receipt.outputs.{dataset_id}",
            output,
            {"records", "axis_id", "files"},
        )
        files = parsed_output["files"]
        if not isinstance(files, Mapping):
            raise BacteriaIdValidationError(
                f"receipt.outputs.{dataset_id}.files",
                "must be a JSON object",
            )
        expected_files = set(DATASET_FILES)
        if set(files) != expected_files:
            raise BacteriaIdValidationError(
                f"receipt.outputs.{dataset_id}.files",
                (
                    f"key mismatch: missing={sorted(expected_files - set(files))}, "
                    f"unexpected={sorted(set(files) - expected_files)}"
                ),
            )
        for name, details in files.items():
            _require_receipt_keys(
                f"receipt.outputs.{dataset_id}.files.{name}",
                details,
                {"bytes", "sha256"},
            )
    _require_receipt_keys(
        "receipt.preprocessing",
        receipt["preprocessing"],
        {"status", "eligible_records", "total_records", "steps"},
    )
    _require_receipt_keys(
        "receipt.license",
        receipt["license"],
        {"text", "status", "redistribution_requires_review"},
    )
    discrepancies = receipt["known_discrepancies"]
    if not isinstance(discrepancies, list):
        raise BacteriaIdValidationError(
            "receipt.known_discrepancies",
            "must be a list",
        )
    for index, discrepancy in enumerate(discrepancies):
        _require_receipt_keys(
            f"receipt.known_discrepancies[{index}]",
            discrepancy,
            {
                "id",
                "paper_claim",
                "released_patient_blocks",
                "adapter_choice",
            },
        )
    return receipt


def _build_bacteria_id_staged(
    raw_root: Path,
    staging_parent: Path,
    inspection: BacteriaIdInspection,
    *,
    timings: dict[str, float] | None = None,
) -> BacteriaIdConversionSummary:
    raw_root = Path(raw_root)
    staging_parent = Path(staging_parent)
    _require_matching_inspection_root(raw_root, inspection)
    if not staging_parent.is_dir():
        raise BacteriaIdValidationError(
            "staging_parent",
            "must be an existing directory",
        )
    reference_path = staging_parent / REFERENCE_DATASET_ID
    clinical_path = staging_parent / CLINICAL_DATASET_ID
    receipt_path = staging_parent / RECEIPT_NAME
    for path in (reference_path, clinical_path, receipt_path):
        if path.exists():
            raise BacteriaIdValidationError(
                f"staging_parent/{path.name}",
                "staged artifact already exists",
            )

    reference_started = perf_counter()
    reference_records = list(
        iter_reference_records(raw_root, inspection),
    )
    write_dataset(
        reference_records,
        reference_path,
        dataset_id=REFERENCE_DATASET_ID,
        class_labels=REFERENCE_CLASS_LABELS,
        overwrite=False,
    )
    del reference_records
    reference_build_seconds = perf_counter() - reference_started
    validation_started = perf_counter()
    reference_validation = validate_dataset(reference_path)
    reference_eligible = _compare_bacteria_id_dataset(
        raw_root,
        reference_path,
        inspection,
        dataset_id=REFERENCE_DATASET_ID,
    )
    reference_summary = _dataset_summary(
        reference_path,
        dataset_id=REFERENCE_DATASET_ID,
        record_count=reference_validation.record_count,
        axis_id_value=inspection.axis_id,
    )
    validation_seconds = perf_counter() - validation_started

    clinical_started = perf_counter()
    clinical_records = list(
        iter_clinical_records(raw_root, inspection),
    )
    write_dataset(
        clinical_records,
        clinical_path,
        dataset_id=CLINICAL_DATASET_ID,
        class_labels=CLINICAL_CLASS_LABELS,
        overwrite=False,
    )
    del clinical_records
    clinical_build_seconds = perf_counter() - clinical_started
    validation_started = perf_counter()
    clinical_validation = validate_dataset(clinical_path)
    clinical_eligible = _compare_bacteria_id_dataset(
        raw_root,
        clinical_path,
        inspection,
        dataset_id=CLINICAL_DATASET_ID,
    )
    clinical_summary = _dataset_summary(
        clinical_path,
        dataset_id=CLINICAL_DATASET_ID,
        record_count=clinical_validation.record_count,
        axis_id_value=inspection.axis_id,
    )
    validation_seconds += perf_counter() - validation_started

    eligible_records = reference_eligible + clinical_eligible
    receipt = _receipt_document(
        raw_root,
        inspection,
        reference_summary,
        clinical_summary,
        eligible_records=eligible_records,
    )
    receipt_path.write_bytes(_canonical_json_bytes(receipt))
    validation_started = perf_counter()
    _validate_bacteria_id_receipt(receipt_path)
    validation_seconds += perf_counter() - validation_started
    receipt_sha256 = _sha256_file(receipt_path, RECEIPT_NAME)
    if timings is not None:
        timings.update(
            {
                "reference_build_seconds": reference_build_seconds,
                "clinical_build_seconds": clinical_build_seconds,
                "validation_seconds": validation_seconds,
            }
        )
    return BacteriaIdConversionSummary(
        reference=reference_summary,
        clinical=clinical_summary,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        eligible_records=eligible_records,
    )


def _remove_published_artifact(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _publish_conversion_artifacts(
    staging_parent: Path,
    artifacts: tuple[
        _PublicationArtifact,
        _PublicationArtifact,
        _PublicationArtifact,
    ],
    *,
    overwrite: bool,
) -> None:
    staging_parent = Path(staging_parent)
    if not staging_parent.is_dir():
        raise BacteriaIdValidationError(
            "staging_parent",
            "must be an existing directory",
        )
    if len(artifacts) != 3:
        raise BacteriaIdValidationError(
            "publication.artifacts",
            "must contain exactly three artifacts",
        )

    final_paths = [Path(artifact.final) for artifact in artifacts]
    if len(final_paths) != len(set(final_paths)):
        raise BacteriaIdValidationError(
            "publication.artifacts.final",
            "final paths must be unique",
        )
    backup_names = [artifact.backup_name for artifact in artifacts]
    if (
        len(backup_names) != len(set(backup_names))
        or any(
            not name
            or Path(name).name != name
            or name in {".", ".."}
            for name in backup_names
        )
    ):
        raise BacteriaIdValidationError(
            "publication.artifacts.backup_name",
            "backup names must be unique portable basenames",
        )
    for artifact in artifacts:
        staged = Path(artifact.staged)
        if staged.parent != staging_parent:
            raise BacteriaIdValidationError(
                "publication.artifacts.staged",
                "staged artifacts must be direct staging children",
            )
        if not staged.exists():
            raise BacteriaIdValidationError(
                f"publication.staged.{staged.name}",
                "staged artifact is missing",
            )
        if Path(artifact.final).parent == staging_parent:
            raise BacteriaIdValidationError(
                "publication.artifacts.final",
                "final artifacts must be outside staging",
            )
        backup = staging_parent / artifact.backup_name
        if backup.exists():
            raise BacteriaIdValidationError(
                f"publication.backup.{artifact.backup_name}",
                "backup path already exists",
            )
        if Path(artifact.final).exists() and not overwrite:
            raise BacteriaIdValidationError(
                f"output_root/{Path(artifact.final).name}",
                "already exists and overwrite is false",
            )

    backups: list[tuple[_PublicationArtifact, Path]] = []
    published: list[_PublicationArtifact] = []
    preserve_staging = False
    try:
        try:
            for artifact in artifacts:
                final = Path(artifact.final)
                if final.exists():
                    backup = staging_parent / artifact.backup_name
                    os.replace(final, backup)
                    backups.append((artifact, backup))

            for artifact in artifacts:
                os.replace(artifact.staged, artifact.final)
                published.append(artifact)
        except BaseException as publication_error:
            recovery_errors: list[str] = []
            for artifact in reversed(published):
                final = Path(artifact.final)
                if final.exists():
                    try:
                        _remove_published_artifact(final)
                    except BaseException as error:
                        recovery_errors.append(
                            f"remove {final}: {type(error).__name__}: {error}"
                        )
            for artifact, backup in backups:
                if not backup.exists():
                    continue
                final = Path(artifact.final)
                if final.exists():
                    recovery_errors.append(
                        f"restore {final}: destination still exists"
                    )
                    continue
                try:
                    os.replace(backup, final)
                except BaseException as error:
                    recovery_errors.append(
                        f"restore {final}: {type(error).__name__}: {error}"
                    )
            if recovery_errors:
                preserve_staging = True
                raise BacteriaIdValidationError(
                    "publication.restore",
                    (
                        f"manual recovery required from {staging_parent}: "
                        + "; ".join(recovery_errors)
                    ),
                ) from publication_error
            raise
    finally:
        if staging_parent.exists() and not preserve_staging:
            shutil.rmtree(staging_parent)


def _preflight_final_paths(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise BacteriaIdValidationError(
            "output_root",
            "must be a directory",
        )
    for name in (
        REFERENCE_DATASET_ID,
        CLINICAL_DATASET_ID,
        RECEIPT_NAME,
    ):
        if (output_root / name).exists() and not overwrite:
            raise BacteriaIdValidationError(
                f"output_root/{name}",
                "already exists and overwrite is false",
            )


def _run_bacteria_id_conversion(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    timings: dict[str, float] | None,
) -> BacteriaIdConversionSummary:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    _preflight_final_paths(output_root, overwrite=overwrite)
    output_root_existed = output_root.exists()

    inspection_started = perf_counter()
    inspection = inspect_bacteria_id(raw_root)
    inspection_seconds = perf_counter() - inspection_started
    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent: Path | None = None
    preserve_staging = False
    try:
        staging_parent = Path(
            tempfile.mkdtemp(
                prefix=".bacteria-id.staging-",
                dir=output_root,
            )
        )
        summary = _build_bacteria_id_staged(
            raw_root,
            staging_parent,
            inspection,
            timings=timings,
        )
        artifacts = (
            _PublicationArtifact(
                staged=staging_parent / REFERENCE_DATASET_ID,
                final=output_root / REFERENCE_DATASET_ID,
                backup_name="backup-reference",
            ),
            _PublicationArtifact(
                staged=staging_parent / CLINICAL_DATASET_ID,
                final=output_root / CLINICAL_DATASET_ID,
                backup_name="backup-clinical",
            ),
            _PublicationArtifact(
                staged=staging_parent / RECEIPT_NAME,
                final=output_root / RECEIPT_NAME,
                backup_name="backup-receipt",
            ),
        )
        try:
            _publish_conversion_artifacts(
                staging_parent,
                artifacts,
                overwrite=overwrite,
            )
        except BacteriaIdValidationError as error:
            preserve_staging = error.path == "publication.restore"
            raise
        if timings is not None:
            timings["inspection_seconds"] = inspection_seconds
        return replace(
            summary,
            receipt_path=output_root / RECEIPT_NAME,
        )
    finally:
        if (
            staging_parent is not None
            and staging_parent.exists()
            and not preserve_staging
        ):
            shutil.rmtree(staging_parent)
        if (
            not output_root_existed
            and output_root.is_dir()
            and not any(output_root.iterdir())
        ):
            output_root.rmdir()


def build_bacteria_id_unified(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> BacteriaIdConversionSummary:
    return _run_bacteria_id_conversion(
        raw_root,
        output_root,
        overwrite=overwrite,
        timings=None,
    )


def _linux_ru_maxrss_to_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Linux ru_maxrss must be a non-negative integer")
    return value * 1024


def _build_bacteria_id_unified_with_metrics(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> tuple[BacteriaIdConversionSummary, BacteriaIdRuntimeMetrics]:
    started = perf_counter()
    timings: dict[str, float] = {}
    summary = _run_bacteria_id_conversion(
        raw_root,
        output_root,
        overwrite=overwrite,
        timings=timings,
    )
    total_seconds = perf_counter() - started
    peak_rss_raw = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    combined_output_bytes = (
        summary.reference.output_bytes + summary.clinical.output_bytes
    )
    source_bytes = (Path(raw_root) / "data.zip").stat().st_size
    metrics = BacteriaIdRuntimeMetrics(
        total_seconds=total_seconds,
        inspection_seconds=timings["inspection_seconds"],
        reference_build_seconds=timings["reference_build_seconds"],
        clinical_build_seconds=timings["clinical_build_seconds"],
        validation_seconds=timings["validation_seconds"],
        peak_rss_raw=peak_rss_raw,
        peak_rss_bytes=_linux_ru_maxrss_to_bytes(peak_rss_raw),
        combined_output_bytes=combined_output_bytes,
        source_bytes=source_bytes,
        compression_ratio=combined_output_bytes / source_bytes,
    )
    return summary, metrics


def _inspect_bacteria_id(
    raw_root: Path,
    contract: _BacteriaIdContract,
) -> BacteriaIdInspection:
    raw_root = Path(raw_root)
    archive_sha256 = _require_hash(
        raw_root / "data.zip",
        contract.archive_sha256,
        "data.zip",
    )

    extracted = raw_root / "extracted"
    observed_hashes = {}
    for name, expected_hash in contract.member_sha256.items():
        observed_hashes[name] = _require_hash(
            extracted / name,
            expected_hash,
            f"extracted/{name}",
        )

    observed_axis_sha256 = observed_hashes.get(contract.axis_file)
    if observed_axis_sha256 != contract.axis_sha256:
        raise BacteriaIdValidationError(
            f"extracted/{contract.axis_file}.sha256",
            (
                f"expected {contract.axis_sha256}, "
                f"observed {observed_axis_sha256}"
            ),
        )

    axis_path = extracted / contract.axis_file
    axis_error_path = f"extracted/{contract.axis_file}"
    axis = _load_array(axis_path, axis_error_path)
    feature_counts = {
        spec.spectra_shape[1] for spec in contract.split_specs.values()
    }
    if len(feature_counts) != 1:
        raise BacteriaIdValidationError(
            "contract.split_specs.spectra_shape",
            "all splits must use one axis length",
        )
    _require_shape(axis, (feature_counts.pop(),), axis_error_path)
    _require_float64(axis, axis_error_path)
    _require_finite(axis, axis_error_path)

    float32_axis = np.asarray(axis, dtype=np.float32)
    if not np.all(np.diff(float32_axis) < 0):
        raise BacteriaIdValidationError(
            f"{axis_error_path}.order",
            "float32 axis must be strictly decreasing",
        )
    axis_float32_max_abs_error = float(
        np.max(np.abs(axis - float32_axis.astype(np.float64))),
    )
    if axis_float32_max_abs_error > _AXIS_FLOAT32_MAX_ABS_ERROR:
        raise BacteriaIdValidationError(
            f"{axis_error_path}.float32_cast_error",
            "exceeds 5.0e-05 cm^-1",
        )
    observed_axis_id = axis_id(float32_axis)

    splits = {}
    clinical_patient_blocks = {}
    total_rows = 0
    intensity_float32_max_abs_error = 0.0
    for split, spec in contract.split_specs.items():
        spectra_error_path = f"extracted/{spec.spectra_file}"
        target_error_path = f"extracted/{spec.target_file}"
        spectra = _load_array(extracted / spec.spectra_file, spectra_error_path)
        targets = _load_array(extracted / spec.target_file, target_error_path)

        _require_shape(spectra, spec.spectra_shape, spectra_error_path)
        _require_shape(targets, spec.target_shape, target_error_path)
        _require_float64(spectra, spectra_error_path)
        _require_float64(targets, target_error_path)

        split_cast_error = _spectra_cast_error(spectra, spectra_error_path)
        intensity_float32_max_abs_error = max(
            intensity_float32_max_abs_error,
            split_cast_error,
        )
        _, class_counts, patient_blocks = _validated_labels(
            targets,
            spec,
            target_error_path,
        )
        if patient_blocks is not None:
            clinical_patient_blocks[split] = patient_blocks

        row_count = spec.spectra_shape[0]
        total_rows += row_count
        splits[split] = BacteriaIdSplitInspection(
            split=spec.split,
            spectra_file=spec.spectra_file,
            target_file=spec.target_file,
            spectra_shape=spec.spectra_shape,
            target_shape=spec.target_shape,
            spectra_dtype=str(spectra.dtype),
            target_dtype=str(targets.dtype),
            row_count=row_count,
            class_counts=MappingProxyType(class_counts),
            role=spec.role,
            label_space=spec.label_space,
            clinical_spectra_per_patient=spec.clinical_spectra_per_patient,
            spectra_sha256=observed_hashes[spec.spectra_file],
            target_sha256=observed_hashes[spec.target_file],
        )

    if intensity_float32_max_abs_error > _INTENSITY_FLOAT32_MAX_ABS_ERROR:
        raise BacteriaIdValidationError(
            "spectra.float32_cast_error",
            "exceeds 3.0e-08",
        )

    return BacteriaIdInspection(
        raw_root=raw_root,
        archive_sha256=archive_sha256,
        axis_sha256=observed_hashes[contract.axis_file],
        axis_id=observed_axis_id,
        axis_shape=axis.shape,
        axis_dtype=str(axis.dtype),
        axis_float32_max_abs_error=axis_float32_max_abs_error,
        intensity_float32_max_abs_error=intensity_float32_max_abs_error,
        total_rows=total_rows,
        splits=MappingProxyType(splits),
        clinical_patient_blocks=MappingProxyType(clinical_patient_blocks),
    )


__all__ = [
    "BacteriaIdConversionSummary",
    "BacteriaIdDatasetSummary",
    "BacteriaIdInspection",
    "BacteriaIdRuntimeMetrics",
    "BacteriaIdSplitInspection",
    "BacteriaIdValidationError",
    "CLINICAL_CLASS_LABELS",
    "REFERENCE_CLASS_LABELS",
    "build_bacteria_id_unified",
    "inspect_bacteria_id",
    "iter_clinical_records",
    "iter_reference_records",
]
