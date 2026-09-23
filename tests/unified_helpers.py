from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.io.schema import (  # noqa: E402
    LicenseStatus,
    PeakTarget,
    PreprocessingStatus,
    PreprocessingStep,
    Provenance,
    RamanRecord,
    SpectrumMetadata,
    Targets,
)


FIXTURE_DATASET_ID = "fixture_mixed_axes"
AXIS_INCREASING = np.array(
    [100.0, 200.0, 300.0, 400.0],
    dtype=np.float32,
)
AXIS_DECREASING = np.array(
    [900.0, 600.0, 300.0],
    dtype=np.float32,
)
AXIS_INCREASING_ID = (
    "73091c2a66b34a0de73655b45838bf1536689edfb9d38909d72702ec70acc387"
)
AXIS_DECREASING_ID = (
    "31863155122b36a76c9ead3635cfad532c44f41f6388e65181d20ec3f3983611"
)
SOURCE_A_SHA256 = (
    "2e21e44d1bc1c4f6631499e6b5396f4f701f62ee5743227936d2ba83567660e1"
)
SOURCE_B_SHA256 = (
    "afe2261a4060a159bfbfebf1a1e0767f2d3366cb9c585c76b44a11bd5ee4a7e1"
)


def raw_record() -> RamanRecord:
    return RamanRecord(
        record_id="record_a_raw",
        intensity=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        wavenumber=AXIS_INCREASING.copy(),
        meta=SpectrumMetadata(
            dataset_id=FIXTURE_DATASET_ID,
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
        ),
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
        provenance=Provenance(
            source_url="https://example.org/source",
            license="CC0-1.0",
            license_status=LicenseStatus.STANDARDIZED,
            sha256=SOURCE_A_SHA256,
            retrieved_date=date(2026, 8, 14),
            source_artifact="source_a.txt",
        ),
    )


def corrected_record() -> RamanRecord:
    base = raw_record()
    return replace(
        base,
        record_id="record_b_corrected",
        intensity=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        meta=replace(
            base.meta,
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
    )


def unknown_record() -> RamanRecord:
    base = raw_record()
    return replace(
        base,
        record_id="record_c_unknown",
        intensity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        wavenumber=AXIS_DECREASING.copy(),
        meta=replace(
            base.meta,
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
        targets=Targets(concentrations={"acetate": 1.0, "glucose": 2.0}),
        provenance=replace(
            base.provenance,
            sha256=SOURCE_B_SHA256,
            source_artifact="source_b.txt",
        ),
    )


def fixture_records() -> list[RamanRecord]:
    return [unknown_record(), corrected_record(), raw_record()]


def fixture_source_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "unified"
