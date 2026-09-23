from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.io.schema import (
    LicenseStatus,
    PeakTarget,
    PreprocessingStatus,
    PreprocessingStep,
    Provenance,
    RamanRecord,
    SpectrumMetadata,
    Targets,
)
from rpe.io.store import write_dataset


DATASET_ID = "fixture_mixed_axes"
SOURCE_A_SHA256 = (
    "2e21e44d1bc1c4f6631499e6b5396f4f701f62ee5743227936d2ba83567660e1"
)
SOURCE_B_SHA256 = (
    "afe2261a4060a159bfbfebf1a1e0767f2d3366cb9c585c76b44a11bd5ee4a7e1"
)


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_records(source_root: Path) -> list[RamanRecord]:
    source_a = source_root / "source_a.txt"
    source_b = source_root / "source_b.txt"
    if _source_sha256(source_a) != SOURCE_A_SHA256:
        raise RuntimeError("source_a.txt: unexpected SHA256")
    if _source_sha256(source_b) != SOURCE_B_SHA256:
        raise RuntimeError("source_b.txt: unexpected SHA256")

    common_meta = SpectrumMetadata(
        dataset_id=DATASET_ID,
        sample_id="sample-a",
        instrument="fixture-scope",
        excitation_nm=785.0,
        integration_time_s=1.0,
        n_accumulations=2,
        grating="fixture-grating",
        detector="fixture-detector",
        preprocessing_status=PreprocessingStatus.KNOWN_RAW,
        preprocessing_steps=(),
        source_metadata={"split": "train", "replicate": 1},
    )
    common_provenance = Provenance(
        source_url="https://example.org/source",
        license="CC0-1.0",
        license_status=LicenseStatus.STANDARDIZED,
        sha256=SOURCE_A_SHA256,
        retrieved_date=date(2026, 8, 14),
        source_artifact="source_a.txt",
    )
    raw = RamanRecord(
        record_id="record_a_raw",
        intensity=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        wavenumber=np.array(
            [100.0, 200.0, 300.0, 400.0],
            dtype=np.float32,
        ),
        meta=common_meta,
        targets=Targets(
            peaks=(
                PeakTarget(
                    pos_cm1=200.0,
                    height=2.0,
                    fwhm=8.0,
                    assignment="fixture_peak",
                ),
            ),
            class_label=0,
        ),
        provenance=common_provenance,
    )
    corrected = RamanRecord(
        record_id="record_b_corrected",
        intensity=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        wavenumber=raw.wavenumber.copy(),
        meta=replace(
            common_meta,
            sample_id="sample-b",
            preprocessing_status=PreprocessingStatus.KNOWN_CORRECTED,
            preprocessing_steps=(
                PreprocessingStep(
                    operation="baseline_correction",
                    description="fixture baseline subtraction",
                    evidence="fixture protocol",
                ),
            ),
            source_metadata={"split": "validation", "replicate": 1},
        ),
        targets=Targets(
            clean=np.array([3.5, 2.5, 1.5, 0.5], dtype=np.float32),
            baseline=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            class_label=1,
            concentration=1.25,
        ),
        provenance=common_provenance,
    )
    unknown = RamanRecord(
        record_id="record_c_unknown",
        intensity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        wavenumber=np.array([900.0, 600.0, 300.0], dtype=np.float32),
        meta=replace(
            common_meta,
            sample_id=None,
            instrument=None,
            excitation_nm=None,
            integration_time_s=None,
            n_accumulations=None,
            grating=None,
            detector=None,
            preprocessing_status=PreprocessingStatus.UNKNOWN,
            preprocessing_steps=(
                PreprocessingStep(
                    operation="cosmic_ray_removal",
                    description="known step; remaining history unavailable",
                    evidence="fixture note",
                ),
            ),
            source_metadata={
                "split": "test",
                "nested": {"flags": [True, None]},
            },
        ),
        targets=Targets(
            concentrations={"acetate": 1.0, "glucose": 2.0},
        ),
        provenance=replace(
            common_provenance,
            sha256=SOURCE_B_SHA256,
            source_artifact="source_b.txt",
        ),
    )
    return [unknown, corrected, raw]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    source_root = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "unified"
    )
    summary = write_dataset(
        fixture_records(source_root),
        args.output,
        dataset_id=DATASET_ID,
        class_labels={0: "class_a", 1: "class_b"},
        concentration_unit="g/L",
        concentration_units={"acetate": "g/L", "glucose": "g/L"},
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "axis_group_count": summary.axis_group_count,
                "dataset_id": summary.dataset_id,
                "path": summary.path.as_posix(),
                "record_count": summary.record_count,
                "status": "written",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
