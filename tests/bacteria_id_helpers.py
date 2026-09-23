from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


SYNTHETIC_AXIS = np.array(
    [1800.0, 1600.0, 1400.0, 1200.0, 1000.0, 800.0, 600.0, 400.0],
    dtype=np.float64,
)
SYNTHETIC_AXIS_FLOAT32_ID = (
    "7eeee35700e4daad1f8e385271b17f25048226ba0919a0211cf23d17a988ae8f"
)
ISOLATE_LABELS = tuple(range(30))
CLINICAL_LABELS = (0, 2, 3, 5, 6)
ZIP_MEMBER_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class SyntheticBacteriaIdSource:
    raw_root: Path
    archive_sha256: str
    member_sha256: Mapping[str, str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _intensities(row_count: int) -> np.ndarray:
    base = np.linspace(0.0, 1.0, SYNTHETIC_AXIS.size, dtype=np.float64)
    spectra = np.stack(
        [base + row_index * 1e-6 for row_index in range(row_count)],
    )
    minima = spectra.min(axis=1, keepdims=True)
    maxima = spectra.max(axis=1, keepdims=True)
    return (spectra - minima) / (maxima - minima)


def _clinical_labels() -> np.ndarray:
    return np.repeat(CLINICAL_LABELS, 2 * 5).astype(np.float64)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "wavenumbers.npy": SYNTHETIC_AXIS.copy(),
        "X_reference.npy": _intensities(60),
        "y_reference.npy": np.repeat(ISOLATE_LABELS, 2).astype(np.float64),
        "X_finetune.npy": _intensities(30),
        "y_finetune.npy": np.array(ISOLATE_LABELS, dtype=np.float64),
        "X_test.npy": _intensities(30),
        "y_test.npy": np.array(ISOLATE_LABELS, dtype=np.float64),
        "X_2018clinical.npy": _intensities(50),
        "y_2018clinical.npy": _clinical_labels(),
        "X_2019clinical.npy": _intensities(50),
        "y_2019clinical.npy": _clinical_labels(),
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def rebuild_archive(raw_root: Path) -> str:
    extracted = raw_root / "extracted"
    archive_path = raw_root / "data.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for path in sorted(extracted.glob("*.npy"), key=lambda item: item.name):
            info = zipfile.ZipInfo(path.name, date_time=ZIP_MEMBER_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return file_sha256(archive_path)


def create_synthetic_bacteria_id_source(
    raw_root: Path,
) -> SyntheticBacteriaIdSource:
    extracted = raw_root / "extracted"
    extracted.mkdir(parents=True)
    for name, array in _arrays().items():
        (extracted / name).write_bytes(_npy_bytes(array))

    archive_sha256 = rebuild_archive(raw_root)
    member_sha256 = {
        path.name: file_sha256(path)
        for path in sorted(extracted.glob("*.npy"), key=lambda item: item.name)
    }
    return SyntheticBacteriaIdSource(
        raw_root=raw_root,
        archive_sha256=archive_sha256,
        member_sha256=member_sha256,
    )


def mutate_npy(
    source: SyntheticBacteriaIdSource,
    name: str,
    mutation: Callable[[np.ndarray], np.ndarray],
    *,
    rebuild_zip: bool = True,
) -> SyntheticBacteriaIdSource:
    path = source.raw_root / "extracted" / name
    array = np.load(path, allow_pickle=False)
    path.write_bytes(_npy_bytes(mutation(array.copy())))
    archive_sha256 = (
        rebuild_archive(source.raw_root)
        if rebuild_zip
        else source.archive_sha256
    )
    member_sha256 = {
        member_path.name: file_sha256(member_path)
        for member_path in sorted(
            (source.raw_root / "extracted").glob("*.npy"),
            key=lambda item: item.name,
        )
    }
    return SyntheticBacteriaIdSource(
        raw_root=source.raw_root,
        archive_sha256=archive_sha256,
        member_sha256=member_sha256,
    )
