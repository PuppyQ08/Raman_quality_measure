from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

import numpy as np

from rpe.io.schema import PreprocessingStatus, RamanRecord
from rpe.io.store import UnifiedDataset


DATASET_ID = "bacteria_id_reference"
SOURCE_SPLITS = ("finetune", "reference", "test")
SPLIT_ROLES = MappingProxyType(
    {
        "finetune": "optical_system_adaptation",
        "reference": "pretraining",
        "test": "independent_test",
    }
)


class BacteriaIdBatchValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class BacteriaIdBatch:
    source_split: str
    intensity: np.ndarray
    class_labels: np.ndarray
    record_ids: tuple[str, ...]
    source_rows: np.ndarray
    wavenumber: np.ndarray


def _positive_integer(path: str, value: object) -> int:
    if isinstance(value, bool):
        raise BacteriaIdBatchValidationError(path, "must not be Boolean")
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise BacteriaIdBatchValidationError(
            path,
            "must be an integer",
        ) from error
    if integer <= 0:
        raise BacteriaIdBatchValidationError(path, "must be positive")
    return integer


def _source_metadata(record: RamanRecord) -> Mapping[str, object]:
    metadata = record.meta.source_metadata
    if not isinstance(metadata, Mapping):
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.source_metadata",
            "must be a mapping",
        )
    return metadata


def _require_record(
    record: RamanRecord,
    *,
    expected_split: str | None,
    expected_source_row: int,
    expected_wavenumber: np.ndarray | None,
) -> tuple[str, int, int, np.ndarray]:
    if record.meta.dataset_id != DATASET_ID:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.dataset_id",
            f"must equal {DATASET_ID!r}",
        )
    if record.meta.sample_id is not None:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.sample_id",
            "must be null for the retained Bacteria-ID reference dataset",
        )
    if record.meta.preprocessing_status is not PreprocessingStatus.KNOWN_CORRECTED:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.preprocessing_status",
            "must equal known_corrected",
        )
    metadata = _source_metadata(record)
    source_split = metadata.get("source_split")
    if source_split not in SOURCE_SPLITS:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.source_split",
            f"must be one of {SOURCE_SPLITS!r}",
        )
    if expected_split is not None:
        expected_index = SOURCE_SPLITS.index(expected_split)
        observed_index = SOURCE_SPLITS.index(source_split)
        if observed_index < expected_index or observed_index > expected_index + 1:
            raise BacteriaIdBatchValidationError(
                f"{record.record_id}.source_split",
                "split order must be finetune, reference, test",
            )
        if observed_index == expected_index + 1 and expected_source_row != 0:
            raise BacteriaIdBatchValidationError(
                f"{record.record_id}.source_split",
                "previous split source rows are not contiguous",
            )
    expected_role = SPLIT_ROLES[source_split]
    if metadata.get("split_role") != expected_role:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.split_role",
            f"must equal {expected_role!r}",
        )
    if metadata.get("label_space") != "isolate":
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.label_space",
            "must equal 'isolate'",
        )
    source_row = metadata.get("source_row")
    if isinstance(source_row, bool) or not isinstance(source_row, int):
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.source_row",
            "must be an integer",
        )
    if source_row != expected_source_row:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.source_row",
            f"must equal contiguous row {expected_source_row}",
        )
    class_label = record.targets.class_label
    if (
        isinstance(class_label, bool)
        or not isinstance(class_label, int)
        or not 0 <= class_label < 30
    ):
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.class_label",
            "must be an integer in [0, 29]",
        )
    if record.intensity.dtype != np.dtype("<f4") or record.intensity.ndim != 1:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.intensity",
            "must be a one-dimensional little-endian float32 array",
        )
    if not np.isfinite(record.intensity).all():
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.intensity",
            "must contain only finite values",
        )
    wavenumber = record.wavenumber
    if wavenumber.dtype != np.dtype("<f4") or wavenumber.ndim != 1:
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.wavenumber",
            "must be a one-dimensional little-endian float32 array",
        )
    if not np.isfinite(wavenumber).all():
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.wavenumber",
            "must contain only finite values",
        )
    if expected_wavenumber is not None and not np.array_equal(
        wavenumber,
        expected_wavenumber,
    ):
        raise BacteriaIdBatchValidationError(
            f"{record.record_id}.wavenumber",
            "must equal the shared dataset axis",
        )
    return source_split, source_row, class_label, wavenumber


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class BacteriaIdBatchLoader:
    def __init__(self, dataset_path: Path, *, batch_size: int) -> None:
        self.dataset_path = Path(dataset_path)
        self.batch_size = _positive_integer("batch_size", batch_size)
        self._dataset: UnifiedDataset | None = None

    def __enter__(self) -> BacteriaIdBatchLoader:
        if self._dataset is not None:
            raise RuntimeError("Bacteria-ID batch loader is already open")
        self._dataset = UnifiedDataset.open(
            self.dataset_path,
            verify_checksums=True,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        dataset = self._dataset
        self._dataset = None
        if dataset is not None:
            dataset.close()

    def iter_batches(self) -> Iterator[BacteriaIdBatch]:
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Bacteria-ID batch loader is not open")

        split: str | None = None
        expected_source_row = 0
        wavenumber: np.ndarray | None = None
        intensity_rows: list[np.ndarray] = []
        labels: list[int] = []
        record_ids: list[str] = []
        source_rows: list[int] = []

        def batch() -> BacteriaIdBatch:
            if split is None or wavenumber is None:
                raise RuntimeError("cannot construct an empty Bacteria-ID batch")
            return BacteriaIdBatch(
                source_split=split,
                intensity=_read_only(
                    np.asarray(intensity_rows, dtype="<f4")
                ),
                class_labels=_read_only(
                    np.asarray(labels, dtype="<i8")
                ),
                record_ids=tuple(record_ids),
                source_rows=_read_only(
                    np.asarray(source_rows, dtype="<i8")
                ),
                wavenumber=wavenumber,
            )

        for record in dataset.iter_records():
            metadata = _source_metadata(record)
            observed_split = metadata.get("source_split")
            if split is not None and observed_split != split:
                if intensity_rows:
                    yield batch()
                    intensity_rows = []
                    labels = []
                    record_ids = []
                    source_rows = []
                expected_source_row = 0
            observed_split, source_row, class_label, observed_wavenumber = (
                _require_record(
                    record,
                    expected_split=split,
                    expected_source_row=expected_source_row,
                    expected_wavenumber=wavenumber,
                )
            )
            if split is None and observed_split != SOURCE_SPLITS[0]:
                raise BacteriaIdBatchValidationError(
                    f"{record.record_id}.source_split",
                    f"first split must equal {SOURCE_SPLITS[0]!r}",
                )
            if split != observed_split:
                split = observed_split
            if wavenumber is None:
                wavenumber = _read_only(
                    np.asarray(observed_wavenumber, dtype="<f4").copy()
                )
            intensity_rows.append(record.intensity)
            labels.append(class_label)
            record_ids.append(record.record_id)
            source_rows.append(source_row)
            expected_source_row += 1
            if len(intensity_rows) == self.batch_size:
                yield batch()
                intensity_rows = []
                labels = []
                record_ids = []
                source_rows = []

        if split != SOURCE_SPLITS[-1]:
            raise BacteriaIdBatchValidationError(
                "source_split",
                f"final split must equal {SOURCE_SPLITS[-1]!r}",
            )
        if intensity_rows:
            yield batch()


__all__ = [
    "BacteriaIdBatch",
    "BacteriaIdBatchLoader",
    "BacteriaIdBatchValidationError",
]
