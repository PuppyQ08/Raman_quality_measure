from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rpe.io as rpe_io  # noqa: E402
from bacteria_id_helpers import (  # noqa: E402
    CLINICAL_LABELS,
    ISOLATE_LABELS,
    SYNTHETIC_AXIS_FLOAT32_ID,
    SyntheticBacteriaIdSource,
    create_synthetic_bacteria_id_source,
    mutate_npy,
)
from rpe.io.bacteria_id import (  # noqa: E402
    BacteriaIdConversionSummary,
    BacteriaIdDatasetSummary,
    BacteriaIdInspection,
    BacteriaIdRuntimeMetrics,
    BacteriaIdSplitInspection,
    BacteriaIdValidationError,
    CLINICAL_CLASS_LABELS,
    REFERENCE_CLASS_LABELS,
    _PublicationArtifact,
    _build_bacteria_id_staged,
    _BacteriaIdContract,
    _SplitContract,
    _build_bacteria_id_unified_with_metrics,
    _compare_bacteria_id_dataset,
    _inspect_bacteria_id,
    _linux_ru_maxrss_to_bytes,
    _publish_conversion_artifacts,
    _validate_bacteria_id_receipt,
    build_bacteria_id_unified,
    inspect_bacteria_id,
    iter_clinical_records,
    iter_reference_records,
)
from rpe.io.schema import (  # noqa: E402
    LicenseStatus,
    PreprocessingStatus,
    PreprocessingStep,
    Targets,
    validate_record,
)
from rpe.io.store import validate_dataset  # noqa: E402


PRODUCTION_RAW_ROOT = ROOT / "data" / "raw" / "bacteria_id"
BACTERIA_BUILDER_PATH = ROOT / "tools" / "build_bacteria_id_unified.py"
UNIFIED_DATASET_FILES = {
    "SHA256SUMS",
    "SHA256SUMS.sha256",
    "arrays.h5",
    "dataset.json",
    "records.jsonl",
}
RECEIPT_KEYS = {
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
}
RECEIPT_SOURCE_KEYS = {
    "archive_file",
    "archive_url",
    "archive_bytes",
    "archive_sha256",
    "retrieved_date",
}
RECEIPT_MEMBER_KEYS = {"sha256", "shape", "dtype"}
RECEIPT_SPLIT_KEYS = {"records", "class_counts", "role", "label_space"}
RECEIPT_OUTPUT_KEYS = {"records", "axis_id", "files"}
RECEIPT_OUTPUT_FILE_KEYS = {"bytes", "sha256"}
RECEIPT_PREPROCESSING_KEYS = {
    "status",
    "eligible_records",
    "total_records",
    "steps",
}
RECEIPT_LICENSE_KEYS = {
    "text",
    "status",
    "redistribution_requires_review",
}
RECEIPT_DISCREPANCY_KEYS = {
    "id",
    "paper_claim",
    "released_patient_blocks",
    "adapter_choice",
}
EXPECTED_REFERENCE_CLASS_LABELS = {
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
EXPECTED_CLINICAL_CLASS_LABELS = {
    0: "Meropenem",
    2: "TZP",
    3: "Vancomycin",
    5: "Penicillin",
    6: "Daptomycin",
}
EXPECTED_PREPROCESSING_STEPS = (
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
COMMON_SOURCE_METADATA_KEYS = {
    "source_dataset_id",
    "source_split",
    "source_row",
    "split_role",
    "label_space",
    "source_spectra_file",
    "source_target_file",
    "source_axis_file",
    "source_spectra_sha256",
    "source_target_sha256",
    "source_axis_sha256",
    "source_intensity_dtype",
    "stored_intensity_dtype",
    "source_axis_dtype",
    "stored_axis_dtype",
    "measurement_mode",
    "published_preprocessing",
}
SPLIT_LAYOUT = {
    "reference": (
        "X_reference.npy",
        "y_reference.npy",
        (60, 8),
        2,
        "pretraining",
        "isolate",
        None,
    ),
    "finetune": (
        "X_finetune.npy",
        "y_finetune.npy",
        (30, 8),
        1,
        "optical_system_adaptation",
        "isolate",
        None,
    ),
    "test": (
        "X_test.npy",
        "y_test.npy",
        (30, 8),
        1,
        "independent_test",
        "isolate",
        None,
    ),
    "clinical2018": (
        "X_2018clinical.npy",
        "y_2018clinical.npy",
        (50, 8),
        10,
        "clinical_test",
        "treatment",
        5,
    ),
    "clinical2019": (
        "X_2019clinical.npy",
        "y_2019clinical.npy",
        (50, 8),
        10,
        "clinical_test",
        "treatment",
        5,
    ),
}


def synthetic_contract(
    source: SyntheticBacteriaIdSource,
) -> _BacteriaIdContract:
    split_specs = {}
    for split, (
        spectra_file,
        target_file,
        spectra_shape,
        expected_count,
        role,
        label_space,
        clinical_spectra_per_patient,
    ) in SPLIT_LAYOUT.items():
        expected_labels = (
            ISOLATE_LABELS if label_space == "isolate" else CLINICAL_LABELS
        )
        split_specs[split] = _SplitContract(
            split=split,
            spectra_file=spectra_file,
            target_file=target_file,
            spectra_shape=spectra_shape,
            target_shape=(spectra_shape[0],),
            expected_labels=expected_labels,
            expected_count_per_label=expected_count,
            role=role,
            label_space=label_space,
            clinical_spectra_per_patient=clinical_spectra_per_patient,
        )
    return _BacteriaIdContract(
        archive_sha256=source.archive_sha256,
        axis_file="wavenumbers.npy",
        axis_sha256=source.member_sha256["wavenumbers.npy"],
        member_sha256=MappingProxyType(dict(source.member_sha256)),
        split_specs=MappingProxyType(split_specs),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(record) for record in records),
    )


def load_bacteria_builder_module():
    specification = importlib.util.spec_from_file_location(
        "task5_build_bacteria_id_unified",
        BACTERIA_BUILDER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load Bacteria-ID builder")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def artifact_snapshot(path: Path):
    if path.is_dir():
        return {
            item.relative_to(path).as_posix(): (
                "dir" if item.is_dir() else item.read_bytes()
            )
            for item in sorted(path.rglob("*"))
        }
    return path.read_bytes()


class BacteriaIdInspectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.output_root = self.temporary_root / "unified"
        self.source = create_synthetic_bacteria_id_source(self.raw_root)

    def inspect(
        self,
        source: SyntheticBacteriaIdSource | None = None,
    ) -> BacteriaIdInspection:
        current_source = self.source if source is None else source
        return _inspect_bacteria_id(
            current_source.raw_root,
            synthetic_contract(current_source),
        )

    def assert_rejected(
        self,
        path: str,
        *,
        source: SyntheticBacteriaIdSource | None = None,
        contract: _BacteriaIdContract | None = None,
    ) -> None:
        current_source = self.source if source is None else source
        current_contract = (
            synthetic_contract(current_source)
            if contract is None
            else contract
        )
        with self.assertRaises(BacteriaIdValidationError) as caught:
            _inspect_bacteria_id(current_source.raw_root, current_contract)
        self.assertEqual(caught.exception.path, path)
        self.assertFalse(self.output_root.exists())

    def test_inspection_reports_exact_source_contract(self):
        inspection = self.inspect()

        self.assertIsInstance(inspection, BacteriaIdInspection)
        self.assertEqual(inspection.raw_root, self.raw_root)
        self.assertEqual(inspection.archive_sha256, self.source.archive_sha256)
        self.assertEqual(
            inspection.axis_sha256,
            self.source.member_sha256["wavenumbers.npy"],
        )
        self.assertEqual(inspection.axis_id, SYNTHETIC_AXIS_FLOAT32_ID)
        self.assertEqual(inspection.axis_shape, (8,))
        self.assertEqual(inspection.axis_dtype, "float64")
        self.assertEqual(inspection.axis_float32_max_abs_error, 0.0)
        self.assertLessEqual(
            inspection.intensity_float32_max_abs_error,
            3.0e-8,
        )
        self.assertGreater(
            inspection.intensity_float32_max_abs_error,
            0.0,
        )
        self.assertEqual(inspection.total_rows, 220)
        self.assertEqual(
            tuple(inspection.splits),
            (
                "reference",
                "finetune",
                "test",
                "clinical2018",
                "clinical2019",
            ),
        )
        self.assertEqual(
            dict(inspection.clinical_patient_blocks),
            {"clinical2018": 10, "clinical2019": 10},
        )

        expected_class_counts = {
            "reference": {label: 2 for label in ISOLATE_LABELS},
            "finetune": {label: 1 for label in ISOLATE_LABELS},
            "test": {label: 1 for label in ISOLATE_LABELS},
            "clinical2018": {label: 10 for label in CLINICAL_LABELS},
            "clinical2019": {label: 10 for label in CLINICAL_LABELS},
        }
        for split, expected_counts in expected_class_counts.items():
            with self.subTest(split=split):
                actual = inspection.splits[split]
                (
                    spectra_file,
                    target_file,
                    spectra_shape,
                    _,
                    role,
                    label_space,
                    clinical_spectra_per_patient,
                ) = SPLIT_LAYOUT[split]
                self.assertIsInstance(actual, BacteriaIdSplitInspection)
                self.assertEqual(actual.split, split)
                self.assertEqual(actual.spectra_file, spectra_file)
                self.assertEqual(actual.target_file, target_file)
                self.assertEqual(actual.spectra_shape, spectra_shape)
                self.assertEqual(actual.target_shape, (spectra_shape[0],))
                self.assertEqual(actual.spectra_dtype, "float64")
                self.assertEqual(actual.target_dtype, "float64")
                self.assertEqual(actual.row_count, spectra_shape[0])
                self.assertEqual(dict(actual.class_counts), expected_counts)
                self.assertEqual(actual.role, role)
                self.assertEqual(actual.label_space, label_space)
                self.assertEqual(
                    actual.clinical_spectra_per_patient,
                    clinical_spectra_per_patient,
                )
                self.assertEqual(
                    actual.spectra_sha256,
                    self.source.member_sha256[spectra_file],
                )
                self.assertEqual(
                    actual.target_sha256,
                    self.source.member_sha256[target_file],
                )

    def test_inspection_results_are_transitively_immutable(self):
        inspection = self.inspect()

        with self.assertRaises(FrozenInstanceError):
            inspection.total_rows = 0
        with self.assertRaises(TypeError):
            inspection.splits["missing"] = inspection.splits["reference"]
        with self.assertRaises(TypeError):
            inspection.splits["reference"].class_counts[0] = 999
        with self.assertRaises(TypeError):
            inspection.clinical_patient_blocks["clinical2018"] = 0

    def test_public_io_api_exports_inspection_and_record_mapping_boundary(self):
        self.assertIs(rpe_io.BacteriaIdInspection, BacteriaIdInspection)
        self.assertIs(
            rpe_io.BacteriaIdSplitInspection,
            BacteriaIdSplitInspection,
        )
        self.assertIs(
            rpe_io.BacteriaIdValidationError,
            BacteriaIdValidationError,
        )
        self.assertIs(rpe_io.inspect_bacteria_id, inspect_bacteria_id)
        self.assertIs(rpe_io.iter_reference_records, iter_reference_records)
        self.assertIs(rpe_io.iter_clinical_records, iter_clinical_records)
        self.assertIs(
            rpe_io.REFERENCE_CLASS_LABELS,
            REFERENCE_CLASS_LABELS,
        )
        self.assertIs(
            rpe_io.CLINICAL_CLASS_LABELS,
            CLINICAL_CLASS_LABELS,
        )
        self.assertIs(
            rpe_io.BacteriaIdDatasetSummary,
            BacteriaIdDatasetSummary,
        )
        self.assertIs(
            rpe_io.BacteriaIdConversionSummary,
            BacteriaIdConversionSummary,
        )
        self.assertIs(
            rpe_io.BacteriaIdRuntimeMetrics,
            BacteriaIdRuntimeMetrics,
        )
        self.assertIs(
            rpe_io.build_bacteria_id_unified,
            build_bacteria_id_unified,
        )

    def test_axis_hash_contract_is_enforced_independently(self):
        contract = synthetic_contract(self.source)
        invalid_contract = _BacteriaIdContract(
            archive_sha256=contract.archive_sha256,
            axis_file=contract.axis_file,
            axis_sha256="0" * 64,
            member_sha256=contract.member_sha256,
            split_specs=contract.split_specs,
        )

        self.assert_rejected(
            "extracted/wavenumbers.npy.sha256",
            contract=invalid_contract,
        )

    def test_inspection_loads_every_array_read_only_without_pickle(self):
        real_load = np.load
        calls = []

        def recording_load(path, *args, **kwargs):
            calls.append((Path(path), args, dict(kwargs)))
            return real_load(path, *args, **kwargs)

        with patch("numpy.load", side_effect=recording_load):
            self.inspect()

        self.assertEqual(
            {call[0].name for call in calls},
            set(self.source.member_sha256),
        )
        self.assertEqual(len(calls), 11)
        for path, args, kwargs in calls:
            with self.subTest(path=path.name):
                self.assertEqual(args, ())
                self.assertEqual(
                    kwargs,
                    {"mmap_mode": "r", "allow_pickle": False},
                )

    def test_archive_hash_mismatch_is_rejected_before_array_loading(self):
        (self.raw_root / "data.zip").write_bytes(
            (self.raw_root / "data.zip").read_bytes() + b"corrupt",
        )
        contract = synthetic_contract(self.source)

        with patch("numpy.load", wraps=np.load) as load:
            self.assert_rejected("data.zip.sha256", contract=contract)
        load.assert_not_called()

    def test_member_hash_mismatch_is_rejected_before_array_loading(self):
        mutate_npy(
            self.source,
            "X_reference.npy",
            lambda array: array.__setitem__((0, 0), 0.25) or array,
            rebuild_zip=False,
        )

        with patch("numpy.load", wraps=np.load) as load:
            self.assert_rejected(
                "extracted/X_reference.npy.sha256",
                contract=synthetic_contract(self.source),
            )
        load.assert_not_called()

    def test_missing_required_member_is_rejected(self):
        (self.raw_root / "extracted" / "X_test.npy").unlink()

        self.assert_rejected(
            "extracted/X_test.npy",
            contract=synthetic_contract(self.source),
        )

    def test_wrong_spectra_and_target_shapes_are_rejected(self):
        mutations = (
            (
                "X_finetune.npy",
                lambda array: array[:-1],
                "extracted/X_finetune.npy.shape",
            ),
            (
                "y_finetune.npy",
                lambda array: array[:-1],
                "extracted/y_finetune.npy.shape",
            ),
        )
        for index, (name, mutation, expected_path) in enumerate(mutations):
            with self.subTest(name=name):
                raw_root = self.temporary_root / f"shape-{index}"
                source = create_synthetic_bacteria_id_source(raw_root)
                source = mutate_npy(source, name, mutation)
                self.assert_rejected(expected_path, source=source)

    def test_wrong_source_dtypes_are_rejected(self):
        mutations = (
            (
                "wavenumbers.npy",
                lambda array: array.astype(np.float32),
                "extracted/wavenumbers.npy.dtype",
            ),
            (
                "X_test.npy",
                lambda array: array.astype(np.float32),
                "extracted/X_test.npy.dtype",
            ),
            (
                "y_test.npy",
                lambda array: array.astype(np.int64),
                "extracted/y_test.npy.dtype",
            ),
        )
        for index, (name, mutation, expected_path) in enumerate(mutations):
            with self.subTest(name=name):
                raw_root = self.temporary_root / f"dtype-{index}"
                source = create_synthetic_bacteria_id_source(raw_root)
                source = mutate_npy(source, name, mutation)
                self.assert_rejected(expected_path, source=source)

    def test_non_finite_axis_spectra_and_targets_are_rejected(self):
        def replace(index, value):
            def mutation(array):
                array[index] = value
                return array

            return mutation

        mutations = (
            (
                "wavenumbers.npy",
                replace(0, np.nan),
                "extracted/wavenumbers.npy.finite",
            ),
            (
                "X_reference.npy",
                replace((0, 0), np.inf),
                "extracted/X_reference.npy.finite",
            ),
            (
                "y_reference.npy",
                replace(0, np.nan),
                "extracted/y_reference.npy.finite",
            ),
        )
        for index, (name, mutation, expected_path) in enumerate(mutations):
            with self.subTest(name=name):
                raw_root = self.temporary_root / f"finite-{index}"
                source = create_synthetic_bacteria_id_source(raw_root)
                source = mutate_npy(source, name, mutation)
                self.assert_rejected(expected_path, source=source)

    def test_axis_must_be_strictly_decreasing_without_duplicates(self):
        def make_increasing(array):
            return array[::-1].copy()

        def duplicate(array):
            array[3] = array[2]
            return array

        for index, mutation in enumerate((make_increasing, duplicate)):
            with self.subTest(mutation=index):
                raw_root = self.temporary_root / f"axis-{index}"
                source = create_synthetic_bacteria_id_source(raw_root)
                source = mutate_npy(source, "wavenumbers.npy", mutation)
                self.assert_rejected(
                    "extracted/wavenumbers.npy.order",
                    source=source,
                )

    def test_labels_must_belong_to_the_fixed_label_space(self):
        def mutation(array):
            array[0] = 99.0
            return array

        source = mutate_npy(self.source, "y_test.npy", mutation)

        self.assert_rejected(
            "extracted/y_test.npy.labels",
            source=source,
        )

    def test_labels_must_be_exact_integers(self):
        def mutation(array):
            array[0] = 0.5
            return array

        source = mutate_npy(self.source, "y_reference.npy", mutation)

        self.assert_rejected(
            "extracted/y_reference.npy.integral",
            source=source,
        )

    def test_reference_class_imbalance_is_rejected(self):
        def mutation(array):
            array[0] = 1.0
            return array

        source = mutate_npy(self.source, "y_reference.npy", mutation)

        self.assert_rejected(
            "extracted/y_reference.npy.class_counts",
            source=source,
        )

    def test_clinical_patient_block_cannot_cross_labels(self):
        def mutation(array):
            array[4], array[10] = array[10], array[4]
            return array

        source = mutate_npy(self.source, "y_2018clinical.npy", mutation)

        self.assert_rejected(
            "extracted/y_2018clinical.npy.patient_blocks.uniform",
            source=source,
        )

    def test_clinical_treatment_must_have_the_expected_number_of_blocks(self):
        def mutation(array):
            array[:5] = 2.0
            return array

        source = mutate_npy(self.source, "y_2019clinical.npy", mutation)

        self.assert_rejected(
            "extracted/y_2019clinical.npy.patient_blocks.counts",
            source=source,
        )


class BacteriaIdRecordMappingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.raw_root = Path(self.temporary_directory.name) / "raw"
        self.source = create_synthetic_bacteria_id_source(self.raw_root)
        self.inspection = _inspect_bacteria_id(
            self.raw_root,
            synthetic_contract(self.source),
        )

    def reference_records(self):
        return list(
            iter_reference_records(self.raw_root, self.inspection),
        )

    def clinical_records(self):
        return list(
            iter_clinical_records(self.raw_root, self.inspection),
        )

    def assert_common_record(
        self,
        record,
        *,
        dataset_id,
        split,
        row,
        role,
        label_space,
        integration_time_s,
    ):
        split_inspection = self.inspection.splits[split]
        self.assertEqual(record.meta.dataset_id, dataset_id)
        self.assertEqual(record.meta.instrument, "Horiba LabRAM HR Evolution")
        self.assertEqual(record.meta.excitation_nm, 633.0)
        self.assertEqual(record.meta.integration_time_s, integration_time_s)
        self.assertIsNone(record.meta.n_accumulations)
        self.assertEqual(record.meta.grating, "300 lines/mm")
        self.assertIsNone(record.meta.detector)
        self.assertIs(
            record.meta.preprocessing_status,
            PreprocessingStatus.KNOWN_CORRECTED,
        )
        self.assertEqual(
            record.meta.preprocessing_steps,
            EXPECTED_PREPROCESSING_STEPS,
        )
        self.assertFalse(record.meta.eligible_for_preprocessing_evaluation)

        metadata = record.meta.source_metadata
        self.assertEqual(metadata["source_dataset_id"], "bacteria-ID")
        self.assertEqual(metadata["source_split"], split)
        self.assertEqual(metadata["source_row"], row)
        self.assertEqual(metadata["split_role"], role)
        self.assertEqual(metadata["label_space"], label_space)
        self.assertEqual(
            metadata["source_spectra_file"],
            split_inspection.spectra_file,
        )
        self.assertEqual(
            metadata["source_target_file"],
            split_inspection.target_file,
        )
        self.assertEqual(metadata["source_axis_file"], "wavenumbers.npy")
        self.assertEqual(
            metadata["source_spectra_sha256"],
            split_inspection.spectra_sha256,
        )
        self.assertEqual(
            metadata["source_target_sha256"],
            split_inspection.target_sha256,
        )
        self.assertEqual(
            metadata["source_axis_sha256"],
            self.inspection.axis_sha256,
        )
        self.assertEqual(metadata["source_intensity_dtype"], "float64")
        self.assertEqual(metadata["stored_intensity_dtype"], "float32")
        self.assertEqual(metadata["source_axis_dtype"], "float64")
        self.assertEqual(metadata["stored_axis_dtype"], "float32")
        self.assertEqual(
            metadata["measurement_mode"],
            "SERS on gold-coated silica",
        )
        self.assertEqual(
            metadata["published_preprocessing"],
            [
                "polynomial_background_correction",
                "min_max_normalization",
            ],
        )

        self.assertEqual(record.intensity.dtype, np.dtype(np.float32))
        self.assertEqual(record.wavenumber.dtype, np.dtype(np.float32))
        self.assertEqual(record.intensity.shape, (8,))
        self.assertEqual(record.wavenumber.shape, (8,))
        self.assertIs(validate_record(record), record)
        self.assertEqual(
            record.provenance.source_url,
            (
                "https://www.dropbox.com/sh/gmgduvzyl5tken6/"
                "AABtSWXWPjoUBkKyC2e7Ag6Da?dl=1"
            ),
        )
        self.assertEqual(
            record.provenance.license,
            (
                "repository MIT; dataset-specific applicability not "
                "separately stated"
            ),
        )
        self.assertIs(
            record.provenance.license_status,
            LicenseStatus.SOURCE_CLAIM,
        )
        self.assertEqual(
            record.provenance.sha256,
            self.inspection.archive_sha256,
        )
        self.assertEqual(record.provenance.retrieved_date, date(2026, 8, 14))
        self.assertEqual(record.provenance.source_artifact, "data.zip")

    def test_literal_class_label_maps_match_pinned_upstream_config(self):
        self.assertEqual(
            dict(REFERENCE_CLASS_LABELS),
            EXPECTED_REFERENCE_CLASS_LABELS,
        )
        self.assertEqual(
            dict(CLINICAL_CLASS_LABELS),
            EXPECTED_CLINICAL_CLASS_LABELS,
        )
        with self.assertRaises(TypeError):
            REFERENCE_CLASS_LABELS[0] = "changed"
        with self.assertRaises(TypeError):
            CLINICAL_CLASS_LABELS[0] = "changed"

    def test_reference_records_preserve_split_order_ids_rows_and_values(self):
        records = self.reference_records()

        self.assertEqual(len(records), 120)
        self.assertEqual(
            [records[index].record_id for index in (0, 59, 60, 89, 90, 119)],
            [
                "reference-000000",
                "reference-000059",
                "finetune-000000",
                "finetune-000029",
                "test-000000",
                "test-000029",
            ],
        )
        expected_edges = (
            (records[0], "reference", 0, "pretraining", 1.0),
            (records[59], "reference", 59, "pretraining", 1.0),
            (
                records[60],
                "finetune",
                0,
                "optical_system_adaptation",
                2.0,
            ),
            (
                records[89],
                "finetune",
                29,
                "optical_system_adaptation",
                2.0,
            ),
            (records[90], "test", 0, "independent_test", 2.0),
            (records[119], "test", 29, "independent_test", 2.0),
        )
        for record, split, row, role, integration_time_s in expected_edges:
            with self.subTest(record_id=record.record_id):
                self.assert_common_record(
                    record,
                    dataset_id="bacteria_id_reference",
                    split=split,
                    row=row,
                    role=role,
                    label_space="isolate",
                    integration_time_s=integration_time_s,
                )
                expected_intensity = np.load(
                    self.raw_root
                    / "extracted"
                    / self.inspection.splits[split].spectra_file,
                    allow_pickle=False,
                )[row].astype(np.float32)
                np.testing.assert_array_equal(
                    record.intensity,
                    expected_intensity,
                )

        expected_axis = np.load(
            self.raw_root / "extracted" / "wavenumbers.npy",
            allow_pickle=False,
        ).astype(np.float32)
        np.testing.assert_array_equal(records[0].wavenumber, expected_axis)
        self.assertEqual(
            [record.targets.class_label for record in records[:8]],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )

    def test_reference_records_do_not_invent_sample_or_row_timepoint(self):
        records = self.reference_records()
        common_expected_keys = COMMON_SOURCE_METADATA_KEYS | {
            "isolate_id",
            "isolate_name",
        }
        reference_expected_keys = common_expected_keys | {
            "reference_collection_measurement_time_points",
            "per_row_measurement_time_point_available",
        }

        for record in records:
            metadata = record.meta.source_metadata
            self.assertIs(validate_record(record), record)
            self.assertIsNone(record.meta.sample_id)
            expected_keys = (
                reference_expected_keys
                if metadata["source_split"] == "reference"
                else common_expected_keys
            )
            self.assertEqual(set(metadata), expected_keys)
            self.assertEqual(
                metadata["isolate_id"],
                record.targets.class_label,
            )
            self.assertEqual(
                metadata["isolate_name"],
                EXPECTED_REFERENCE_CLASS_LABELS[
                    record.targets.class_label
                ],
            )
            if metadata["source_split"] == "reference":
                self.assertEqual(
                    metadata["reference_collection_measurement_time_points"],
                    3,
                )
                self.assertFalse(
                    metadata["per_row_measurement_time_point_available"],
                )
            self.assertNotIn("measurement_time_point", metadata)
            self.assertEqual(
                record.targets,
                Targets(class_label=record.targets.class_label),
            )

    def test_clinical_records_preserve_sparse_labels_ids_and_surrogate_blocks(self):
        records = self.clinical_records()

        self.assertEqual(len(records), 100)
        self.assertEqual(
            [records[index].record_id for index in (0, 49, 50, 99)],
            [
                "clinical2018-000000",
                "clinical2018-000049",
                "clinical2019-000000",
                "clinical2019-000049",
            ],
        )
        self.assertEqual(
            set(record.targets.class_label for record in records),
            {0, 2, 3, 5, 6},
        )
        self.assertEqual(
            len(set(record.meta.sample_id for record in records)),
            20,
        )

        expected_rows = (
            (
                records[0],
                "clinical2018",
                0,
                0,
                0,
                0,
                "clinical2018-treatment-00-patient-block-00",
                None,
            ),
            (
                records[4],
                "clinical2018",
                4,
                0,
                0,
                4,
                "clinical2018-treatment-00-patient-block-00",
                None,
            ),
            (
                records[5],
                "clinical2018",
                5,
                1,
                1,
                0,
                "clinical2018-treatment-00-patient-block-01",
                None,
            ),
            (
                records[10],
                "clinical2018",
                10,
                2,
                0,
                0,
                "clinical2018-treatment-02-patient-block-00",
                None,
            ),
            (
                records[50],
                "clinical2019",
                0,
                0,
                0,
                0,
                "clinical2019-treatment-00-patient-block-00",
                2.0,
            ),
            (
                records[99],
                "clinical2019",
                49,
                9,
                1,
                4,
                "clinical2019-treatment-06-patient-block-01",
                2.0,
            ),
        )
        for (
            record,
            split,
            row,
            patient_block_index,
            patient_index_within_treatment,
            spectrum_index_within_patient,
            sample_id,
            integration_time_s,
        ) in expected_rows:
            with self.subTest(record_id=record.record_id):
                self.assert_common_record(
                    record,
                    dataset_id="bacteria_id_clinical",
                    split=split,
                    row=row,
                    role="clinical_test",
                    label_space="treatment",
                    integration_time_s=integration_time_s,
                )
                metadata = record.meta.source_metadata
                self.assertEqual(record.meta.sample_id, sample_id)
                self.assertEqual(
                    metadata["patient_block_index"],
                    patient_block_index,
                )
                self.assertEqual(
                    metadata["patient_index_within_treatment"],
                    patient_index_within_treatment,
                )
                self.assertEqual(
                    metadata["spectrum_index_within_patient"],
                    spectrum_index_within_patient,
                )

    def test_clinical_records_preserve_target_names_and_no_clean_targets(self):
        records = self.clinical_records()
        common_expected_keys = COMMON_SOURCE_METADATA_KEYS | {
            "treatment_id",
            "treatment_name",
            "patient_block_index",
            "patient_index_within_treatment",
            "spectrum_index_within_patient",
            "sample_id_is_surrogate",
            "sample_id_evidence",
        }
        clinical2018_keys = common_expected_keys | {
            "likely_integration_time_s",
            "canonical_integration_time_s_unknown_reason",
        }

        for record in records:
            metadata = record.meta.source_metadata
            label = record.targets.class_label
            self.assertIs(validate_record(record), record)
            expected_keys = (
                clinical2018_keys
                if metadata["source_split"] == "clinical2018"
                else common_expected_keys
            )
            self.assertEqual(set(metadata), expected_keys)
            self.assertEqual(metadata["treatment_id"], label)
            self.assertEqual(
                metadata["treatment_name"],
                EXPECTED_CLINICAL_CLASS_LABELS[label],
            )
            self.assertIs(metadata["sample_id_is_surrogate"], True)
            self.assertEqual(
                metadata["sample_id_evidence"],
                "pinned public notebook contiguous block indexing",
            )
            if metadata["source_split"] == "clinical2018":
                self.assertIsNone(record.meta.integration_time_s)
                self.assertEqual(
                    metadata["likely_integration_time_s"],
                    1.0,
                )
                self.assertEqual(
                    metadata[
                        "canonical_integration_time_s_unknown_reason"
                    ],
                    (
                        "likely 1 s by contrast, but not uniquely asserted "
                        "for released X_2018clinical.npy"
                    ),
                )
            else:
                self.assertEqual(record.meta.integration_time_s, 2.0)
            self.assertEqual(record.targets, Targets(class_label=label))

    def test_iterators_are_single_pass_and_lazy_until_each_split_is_reached(self):
        real_load = np.load
        calls = []

        def recording_load(path, *args, **kwargs):
            calls.append(Path(path).name)
            return real_load(path, *args, **kwargs)

        with patch("numpy.load", side_effect=recording_load):
            iterator = iter_reference_records(
                self.raw_root,
                self.inspection,
            )
            self.assertIs(iter(iterator), iterator)
            self.assertEqual(calls, [])
            first = next(iterator)

        self.assertEqual(first.record_id, "reference-000000")
        self.assertEqual(
            calls,
            ["wavenumbers.npy", "X_reference.npy", "y_reference.npy"],
        )
        self.assertNotIn("X_finetune.npy", calls)
        self.assertNotIn("X_test.npy", calls)

    def test_omitted_inspection_invokes_preflight_once(self):
        with patch(
            "rpe.io.bacteria_id.inspect_bacteria_id",
            return_value=self.inspection,
        ) as inspect:
            iterator = iter_clinical_records(self.raw_root)
            first = next(iterator)

        self.assertEqual(first.record_id, "clinical2018-000000")
        inspect.assert_called_once_with(self.raw_root)

    def test_explicit_inspection_must_belong_to_the_same_raw_root(self):
        other_raw_root = Path(self.temporary_directory.name) / "other-raw"
        create_synthetic_bacteria_id_source(other_raw_root)

        for iterator_function in (
            iter_reference_records,
            iter_clinical_records,
        ):
            with self.subTest(iterator=iterator_function.__name__):
                iterator = iterator_function(other_raw_root, self.inspection)
                with self.assertRaises(BacteriaIdValidationError) as caught:
                    next(iterator)
                self.assertEqual(caught.exception.path, "inspection.raw_root")

    def test_yielded_arrays_are_independent_from_source_and_other_records(self):
        source_intensity = np.load(
            self.raw_root / "extracted" / "X_reference.npy",
            allow_pickle=False,
        )[0].copy()
        source_axis = np.load(
            self.raw_root / "extracted" / "wavenumbers.npy",
            allow_pickle=False,
        ).copy()
        iterator = iter_reference_records(self.raw_root, self.inspection)
        first = next(iterator)
        first.intensity[0] = np.float32(123.0)
        first.wavenumber[0] = np.float32(321.0)
        second = next(iterator)

        np.testing.assert_array_equal(
            np.load(
                self.raw_root / "extracted" / "X_reference.npy",
                allow_pickle=False,
            )[0],
            source_intensity,
        )
        np.testing.assert_array_equal(
            np.load(
                self.raw_root / "extracted" / "wavenumbers.npy",
                allow_pickle=False,
            ),
            source_axis,
        )
        self.assertNotEqual(second.intensity[0], np.float32(123.0))
        self.assertNotEqual(second.wavenumber[0], np.float32(321.0))


class BacteriaIdStagedBuildTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_bacteria_id_source(self.raw_root)
        self.inspection = _inspect_bacteria_id(
            self.raw_root,
            synthetic_contract(self.source),
        )
        self.staging_parent = self.temporary_root / "staging"
        self.staging_parent.mkdir()

    def build(self, staging_parent: Path | None = None):
        return _build_bacteria_id_staged(
            self.raw_root,
            self.staging_parent if staging_parent is None else staging_parent,
            self.inspection,
        )

    def test_staged_build_creates_two_valid_five_file_datasets(self):
        summary = self.build()

        self.assertIsInstance(summary, BacteriaIdConversionSummary)
        self.assertIsInstance(summary.reference, BacteriaIdDatasetSummary)
        self.assertIsInstance(summary.clinical, BacteriaIdDatasetSummary)
        self.assertEqual(summary.reference.dataset_id, "bacteria_id_reference")
        self.assertEqual(summary.reference.record_count, 120)
        self.assertEqual(summary.reference.axis_id, SYNTHETIC_AXIS_FLOAT32_ID)
        self.assertEqual(summary.clinical.dataset_id, "bacteria_id_clinical")
        self.assertEqual(summary.clinical.record_count, 100)
        self.assertEqual(summary.clinical.axis_id, SYNTHETIC_AXIS_FLOAT32_ID)
        self.assertEqual(summary.eligible_records, 0)

        for dataset_summary in (summary.reference, summary.clinical):
            with self.subTest(dataset_id=dataset_summary.dataset_id):
                dataset_path = self.staging_parent / dataset_summary.dataset_id
                self.assertEqual(
                    {path.name for path in dataset_path.iterdir()},
                    UNIFIED_DATASET_FILES,
                )
                validation = validate_dataset(dataset_path)
                self.assertEqual(
                    validation.record_count,
                    dataset_summary.record_count,
                )
                self.assertEqual(validation.axis_group_count, 1)
                self.assertEqual(
                    validation.preprocessing_status_counts,
                    {
                        "known_raw": 0,
                        "known_corrected": dataset_summary.record_count,
                        "unknown": 0,
                    },
                )
                self.assertEqual(
                    set(dataset_summary.files),
                    UNIFIED_DATASET_FILES,
                )
                self.assertEqual(
                    dataset_summary.output_bytes,
                    sum(
                        details["bytes"]
                        for details in dataset_summary.files.values()
                    ),
                )
                for name, details in dataset_summary.files.items():
                    self.assertEqual(
                        details,
                        {
                            "bytes": (dataset_path / name).stat().st_size,
                            "sha256": file_sha256(dataset_path / name),
                        },
                    )

    def test_staged_manifests_have_exact_class_maps_and_source_rows(self):
        self.build()
        reference_path = self.staging_parent / "bacteria_id_reference"
        clinical_path = self.staging_parent / "bacteria_id_clinical"
        reference_manifest = json.loads(
            (reference_path / "dataset.json").read_text(encoding="utf-8"),
        )
        clinical_manifest = json.loads(
            (clinical_path / "dataset.json").read_text(encoding="utf-8"),
        )

        self.assertEqual(
            reference_manifest["class_labels"],
            {
                str(label): name
                for label, name in EXPECTED_REFERENCE_CLASS_LABELS.items()
            },
        )
        self.assertEqual(
            clinical_manifest["class_labels"],
            {
                str(label): name
                for label, name in EXPECTED_CLINICAL_CLASS_LABELS.items()
            },
        )
        reference_records = read_jsonl(reference_path / "records.jsonl")
        clinical_records = read_jsonl(clinical_path / "records.jsonl")
        self.assertEqual(
            [record["record_id"] for record in reference_records],
            sorted(record["record_id"] for record in reference_records),
        )
        self.assertEqual(
            [record["record_id"] for record in clinical_records],
            sorted(record["record_id"] for record in clinical_records),
        )
        for records in (reference_records, clinical_records):
            rows_by_split = {}
            for record in records:
                metadata = record["meta"]["source_metadata"]
                rows_by_split.setdefault(metadata["source_split"], []).append(
                    metadata["source_row"],
                )
            for split, rows in rows_by_split.items():
                with self.subTest(split=split):
                    self.assertEqual(rows, list(range(len(rows))))

    def test_staged_build_is_byte_deterministic_across_independent_parents(self):
        first = self.build()
        second_parent = self.temporary_root / "staging-second"
        second_parent.mkdir()
        second = self.build(second_parent)

        self.assertEqual(
            first.receipt_path.read_bytes(),
            second.receipt_path.read_bytes(),
        )
        self.assertEqual(first.receipt_sha256, second.receipt_sha256)
        for dataset_id in (
            "bacteria_id_reference",
            "bacteria_id_clinical",
        ):
            for name in UNIFIED_DATASET_FILES:
                with self.subTest(dataset_id=dataset_id, name=name):
                    self.assertEqual(
                        (self.staging_parent / dataset_id / name).read_bytes(),
                        (second_parent / dataset_id / name).read_bytes(),
                    )

    def test_chunk_comparator_reads_slices_and_accepts_exact_outputs(self):
        self.build()
        real_getitem = h5py.Dataset.__getitem__
        intensity_keys = []

        def recording_getitem(dataset, key):
            if dataset.name.endswith("/intensity"):
                intensity_keys.append(key)
            return real_getitem(dataset, key)

        with patch.object(
            h5py.Dataset,
            "__getitem__",
            new=recording_getitem,
        ):
            eligible = _compare_bacteria_id_dataset(
                self.raw_root,
                self.staging_parent / "bacteria_id_reference",
                self.inspection,
                dataset_id="bacteria_id_reference",
            )

        self.assertEqual(eligible, 0)
        self.assertTrue(intensity_keys)
        for key in intensity_keys:
            with self.subTest(key=key):
                self.assertIsInstance(key, tuple)
                row_slice = key[0]
                self.assertIsInstance(row_slice, slice)
                self.assertLessEqual(row_slice.stop - row_slice.start, 512)

    def test_chunk_comparator_rejects_mutated_intensity_with_stable_path(self):
        self.build()
        dataset_path = self.staging_parent / "bacteria_id_reference"
        with h5py.File(dataset_path / "arrays.h5", "r+") as arrays:
            axis_group = next(iter(arrays["axes"].values()))
            axis_group["intensity"][0, 0] += np.float32(0.25)

        with self.assertRaises(BacteriaIdValidationError) as caught:
            _compare_bacteria_id_dataset(
                self.raw_root,
                dataset_path,
                self.inspection,
                dataset_id="bacteria_id_reference",
            )
        self.assertEqual(
            caught.exception.path,
            "bacteria_id_reference.arrays.h5.intensity",
        )

    def test_chunk_comparator_rejects_mutated_label_with_stable_path(self):
        self.build()
        dataset_path = self.staging_parent / "bacteria_id_reference"
        records_path = dataset_path / "records.jsonl"
        records = read_jsonl(records_path)
        records[0]["targets"]["class_label"] = 29
        write_jsonl(records_path, records)

        with self.assertRaises(BacteriaIdValidationError) as caught:
            _compare_bacteria_id_dataset(
                self.raw_root,
                dataset_path,
                self.inspection,
                dataset_id="bacteria_id_reference",
            )
        self.assertEqual(
            caught.exception.path,
            "bacteria_id_reference.records.jsonl[0].targets.class_label",
        )

    def test_chunk_comparator_rejects_mutated_split_metadata(self):
        self.build()
        dataset_path = self.staging_parent / "bacteria_id_reference"
        records_path = dataset_path / "records.jsonl"
        mutations = (
            (
                "split_role",
                "wrong_role",
                (
                    "bacteria_id_reference.records.jsonl[0]."
                    "meta.source_metadata.split_role"
                ),
            ),
            (
                "label_space",
                "wrong_space",
                (
                    "bacteria_id_reference.records.jsonl[0]."
                    "meta.source_metadata.label_space"
                ),
            ),
        )
        original = records_path.read_bytes()
        for key, value, expected_path in mutations:
            with self.subTest(key=key):
                records = [
                    json.loads(line)
                    for line in original.decode("utf-8").splitlines()
                ]
                records[0]["meta"]["source_metadata"][key] = value
                write_jsonl(records_path, records)
                with self.assertRaises(BacteriaIdValidationError) as caught:
                    _compare_bacteria_id_dataset(
                        self.raw_root,
                        dataset_path,
                        self.inspection,
                        dataset_id="bacteria_id_reference",
                    )
                self.assertEqual(caught.exception.path, expected_path)
                records_path.write_bytes(original)

    def test_receipt_has_exact_schema_values_and_file_attribution(self):
        summary = self.build()
        raw = summary.receipt_path.read_bytes()
        receipt = json.loads(raw)

        self.assertEqual(raw, canonical_json_bytes(receipt))
        self.assertEqual(set(receipt), RECEIPT_KEYS)
        self.assertEqual(set(receipt["source"]), RECEIPT_SOURCE_KEYS)
        self.assertEqual(
            set(receipt["members"]),
            set(self.source.member_sha256),
        )
        for member in receipt["members"].values():
            self.assertEqual(set(member), RECEIPT_MEMBER_KEYS)
        self.assertEqual(
            set(receipt["splits"]),
            {
                "reference",
                "finetune",
                "test",
                "clinical2018",
                "clinical2019",
            },
        )
        for split in receipt["splits"].values():
            self.assertEqual(set(split), RECEIPT_SPLIT_KEYS)
        self.assertEqual(
            set(receipt["outputs"]),
            {"bacteria_id_reference", "bacteria_id_clinical"},
        )
        for dataset_id, output in receipt["outputs"].items():
            self.assertEqual(set(output), RECEIPT_OUTPUT_KEYS)
            self.assertEqual(set(output["files"]), UNIFIED_DATASET_FILES)
            for name, details in output["files"].items():
                self.assertEqual(set(details), RECEIPT_OUTPUT_FILE_KEYS)
                path = self.staging_parent / dataset_id / name
                self.assertEqual(details["bytes"], path.stat().st_size)
                self.assertEqual(details["sha256"], file_sha256(path))
        self.assertEqual(
            set(receipt["preprocessing"]),
            RECEIPT_PREPROCESSING_KEYS,
        )
        self.assertEqual(set(receipt["license"]), RECEIPT_LICENSE_KEYS)
        self.assertEqual(len(receipt["known_discrepancies"]), 1)
        self.assertEqual(
            set(receipt["known_discrepancies"][0]),
            RECEIPT_DISCREPANCY_KEYS,
        )
        self.assertEqual(receipt["adapter_version"], "0.1.0")
        self.assertEqual(receipt["schema_version"], "0.1.0")
        self.assertEqual(receipt["source"]["archive_file"], "data.zip")
        self.assertEqual(
            receipt["source"]["archive_bytes"],
            (self.raw_root / "data.zip").stat().st_size,
        )
        self.assertEqual(
            receipt["clinical_patient_blocks"],
            {"clinical2018": 10, "clinical2019": 10},
        )
        self.assertEqual(receipt["preprocessing"]["eligible_records"], 0)
        self.assertEqual(receipt["preprocessing"]["total_records"], 220)
        self.assertEqual(
            summary.receipt_sha256,
            file_sha256(summary.receipt_path),
        )

    def test_receipt_contains_no_staging_path_runtime_or_nonfinite_json(self):
        summary = self.build()
        raw = summary.receipt_path.read_text(encoding="utf-8")

        self.assertNotIn(str(self.temporary_root), raw)
        self.assertNotIn("staging", raw)
        self.assertNotIn("timestamp", raw)
        self.assertNotIn("runtime", raw)
        self.assertNotIn("rss", raw.lower())
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        self.assertNotIn("-Infinity", raw)

    def test_receipt_validator_rejects_unknown_key_with_stable_path(self):
        summary = self.build()
        receipt = json.loads(summary.receipt_path.read_text(encoding="utf-8"))
        receipt["unexpected"] = True
        summary.receipt_path.write_bytes(canonical_json_bytes(receipt))

        with self.assertRaises(BacteriaIdValidationError) as caught:
            _validate_bacteria_id_receipt(summary.receipt_path)
        self.assertEqual(caught.exception.path, "receipt")

    def test_receipt_validator_enforces_exact_collection_key_sets(self):
        summary = self.build()
        original = summary.receipt_path.read_bytes()
        mutations = (
            (
                lambda receipt: receipt["members"].pop("X_reference.npy"),
                "receipt.members",
            ),
            (
                lambda receipt: receipt["splits"].pop("reference"),
                "receipt.splits",
            ),
            (
                lambda receipt: receipt["outputs"].pop(
                    "bacteria_id_reference",
                ),
                "receipt.outputs",
            ),
            (
                lambda receipt: receipt["outputs"][
                    "bacteria_id_reference"
                ]["files"].pop("arrays.h5"),
                "receipt.outputs.bacteria_id_reference.files",
            ),
            (
                lambda receipt: receipt["clinical_patient_blocks"].pop(
                    "clinical2018",
                ),
                "receipt.clinical_patient_blocks",
            ),
        )
        for index, (mutation, expected_path) in enumerate(mutations):
            with self.subTest(index=index):
                receipt = json.loads(original)
                mutation(receipt)
                summary.receipt_path.write_bytes(canonical_json_bytes(receipt))
                with self.assertRaises(BacteriaIdValidationError) as caught:
                    _validate_bacteria_id_receipt(summary.receipt_path)
                self.assertEqual(caught.exception.path, expected_path)
        summary.receipt_path.write_bytes(original)

    def test_staged_build_does_not_mutate_synthetic_raw_source(self):
        before = {
            path.relative_to(self.raw_root).as_posix(): (
                path.stat().st_size,
                file_sha256(path),
            )
            for path in self.raw_root.rglob("*")
            if path.is_file()
        }

        self.build()

        after = {
            path.relative_to(self.raw_root).as_posix(): (
                path.stat().st_size,
                file_sha256(path),
            )
            for path in self.raw_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_staged_build_does_not_touch_production_output_root(self):
        production_output = ROOT / "data" / "unified"
        before = sorted(
            path.relative_to(production_output).as_posix()
            for path in production_output.rglob("*")
        )

        self.build()

        after = sorted(
            path.relative_to(production_output).as_posix()
            for path in production_output.rglob("*")
        )
        self.assertEqual(after, before)


class BacteriaIdTransactionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_bacteria_id_source(self.raw_root)
        self.inspection = _inspect_bacteria_id(
            self.raw_root,
            synthetic_contract(self.source),
        )

    def output_root(self, name: str = "output") -> Path:
        return self.temporary_root / name

    def final_paths(self, output_root: Path) -> tuple[Path, Path, Path]:
        return (
            output_root / "bacteria_id_reference",
            output_root / "bacteria_id_clinical",
            output_root / "bacteria_id_conversion.json",
        )

    def assert_no_staging(self, output_root: Path) -> None:
        if not output_root.exists():
            return
        self.assertFalse(
            any(
                path.name.startswith(".bacteria-id.staging-")
                for path in output_root.iterdir()
            )
        )

    def seed_originals(self, output_root: Path) -> dict[str, object]:
        output_root.mkdir(parents=True)
        reference, clinical, receipt = self.final_paths(output_root)
        reference.mkdir()
        (reference / "original-reference.txt").write_text(
            "original reference\n",
            encoding="utf-8",
        )
        clinical.mkdir()
        (clinical / "original-clinical.txt").write_text(
            "original clinical\n",
            encoding="utf-8",
        )
        receipt.write_text("original receipt\n", encoding="utf-8")
        return {
            path.name: artifact_snapshot(path)
            for path in (reference, clinical, receipt)
        }

    def assert_originals(
        self,
        output_root: Path,
        expected: dict[str, object],
    ) -> None:
        for path in self.final_paths(output_root):
            self.assertTrue(path.exists(), path)
            self.assertEqual(artifact_snapshot(path), expected[path.name])

    def build(
        self,
        output_root: Path,
        *,
        overwrite: bool = False,
    ) -> BacteriaIdConversionSummary:
        with patch(
            "rpe.io.bacteria_id.inspect_bacteria_id",
            return_value=self.inspection,
        ):
            return build_bacteria_id_unified(
                self.raw_root,
                output_root,
                overwrite=overwrite,
            )

    def test_first_publication_is_one_valid_three_artifact_result(self):
        output_root = self.output_root()
        sentinel = output_root / "unowned-sentinel.txt"
        output_root.mkdir()
        sentinel.write_text("keep me\n", encoding="utf-8")

        summary = self.build(output_root)

        reference, clinical, receipt = self.final_paths(output_root)
        self.assertEqual(validate_dataset(reference).record_count, 120)
        self.assertEqual(validate_dataset(clinical).record_count, 100)
        self.assertEqual(summary.reference.record_count, 120)
        self.assertEqual(summary.clinical.record_count, 100)
        self.assertEqual(summary.receipt_path, receipt)
        self.assertEqual(summary.receipt_sha256, file_sha256(receipt))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me\n")
        self.assert_no_staging(output_root)

    def test_overwrite_is_deterministic_and_leaves_no_residue(self):
        output_root = self.output_root()
        first = self.build(output_root)
        before = {
            path.name: artifact_snapshot(path)
            for path in self.final_paths(output_root)
        }

        second = self.build(output_root, overwrite=True)

        self.assertEqual(
            {
                path.name: artifact_snapshot(path)
                for path in self.final_paths(output_root)
            },
            before,
        )
        self.assertEqual(second.receipt_sha256, first.receipt_sha256)
        self.assert_no_staging(output_root)

    def test_existing_output_without_overwrite_fails_before_source_access(self):
        output_root = self.output_root()
        originals = self.seed_originals(output_root)
        missing_raw_root = self.temporary_root / "missing-raw"

        with self.assertRaises(BacteriaIdValidationError) as caught:
            build_bacteria_id_unified(missing_raw_root, output_root)

        self.assertEqual(
            caught.exception.path,
            "output_root/bacteria_id_reference",
        )
        self.assert_originals(output_root, originals)
        self.assert_no_staging(output_root)

    def test_source_preflight_failure_leaves_new_output_root_absent(self):
        output_root = self.output_root()
        missing_raw_root = self.temporary_root / "missing-raw"

        with self.assertRaises(BacteriaIdValidationError) as caught:
            build_bacteria_id_unified(missing_raw_root, output_root)

        self.assertEqual(caught.exception.path, "data.zip")
        self.assertFalse(output_root.exists())

    def test_staged_build_or_validation_failure_leaves_originals_unchanged(self):
        output_root = self.output_root()
        originals = self.seed_originals(output_root)
        failures = (
            OSError("injected staged build failure"),
            BacteriaIdValidationError(
                "staged.validation",
                "injected staged validation failure",
            ),
        )
        for failure in failures:
            with self.subTest(error_type=type(failure).__name__):
                with patch(
                    "rpe.io.bacteria_id._build_bacteria_id_staged",
                    side_effect=failure,
                ):
                    with self.assertRaises(type(failure)):
                        self.build(output_root, overwrite=True)

                self.assert_originals(output_root, originals)
                self.assert_no_staging(output_root)

    def test_staged_build_failure_leaves_new_output_root_absent(self):
        output_root = self.output_root()

        with patch(
            "rpe.io.bacteria_id._build_bacteria_id_staged",
            side_effect=OSError("injected staged build failure"),
        ):
            with self.assertRaisesRegex(
                OSError,
                "injected staged build failure",
            ):
                self.build(output_root)

        self.assertFalse(output_root.exists())

    def test_staging_allocation_failure_leaves_new_output_root_absent(self):
        output_root = self.output_root()

        with patch(
            "rpe.io.bacteria_id.tempfile.mkdtemp",
            side_effect=OSError("injected staging allocation failure"),
        ):
            with self.assertRaisesRegex(
                OSError,
                "injected staging allocation failure",
            ):
                self.build(output_root)

        self.assertFalse(output_root.exists())

    def test_failure_publishing_any_artifact_restores_all_originals(self):
        for target_name in (
            "bacteria_id_reference",
            "bacteria_id_clinical",
            "bacteria_id_conversion.json",
        ):
            with self.subTest(target_name=target_name):
                output_root = self.output_root(f"failure-{target_name}")
                originals = self.seed_originals(output_root)
                sentinel = output_root / "unowned-sentinel.txt"
                sentinel.write_text("keep me\n", encoding="utf-8")
                target = output_root / target_name
                real_replace = os.replace

                def injected_replace(source, destination):
                    source_path = Path(source)
                    destination_path = Path(destination)
                    if (
                        destination_path == target
                        and source_path.parent.name.startswith(
                            ".bacteria-id.staging-"
                        )
                        and not source_path.name.startswith("backup-")
                    ):
                        raise OSError(
                            f"injected publish failure for {target_name}"
                        )
                    return real_replace(source, destination)

                with patch(
                    "rpe.io.bacteria_id.os.replace",
                    side_effect=injected_replace,
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "injected publish failure",
                    ):
                        self.build(output_root, overwrite=True)

                self.assert_originals(output_root, originals)
                self.assertEqual(
                    sentinel.read_text(encoding="utf-8"),
                    "keep me\n",
                )
                self.assert_no_staging(output_root)

    def test_first_publication_failure_removes_only_newly_published_finals(self):
        for target_name in (
            "bacteria_id_reference",
            "bacteria_id_clinical",
            "bacteria_id_conversion.json",
        ):
            with self.subTest(target_name=target_name):
                output_root = self.output_root(f"first-failure-{target_name}")
                output_root.mkdir()
                sentinel = output_root / "unowned-sentinel.txt"
                sentinel.write_text("keep me\n", encoding="utf-8")
                target = output_root / target_name
                real_replace = os.replace

                def injected_replace(source, destination):
                    source_path = Path(source)
                    destination_path = Path(destination)
                    if (
                        destination_path == target
                        and source_path.parent.name.startswith(
                            ".bacteria-id.staging-"
                        )
                    ):
                        raise OSError(
                            f"injected first publish failure for {target_name}"
                        )
                    return real_replace(source, destination)

                with patch(
                    "rpe.io.bacteria_id.os.replace",
                    side_effect=injected_replace,
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "injected first publish failure",
                    ):
                        self.build(output_root)

                for final in self.final_paths(output_root):
                    self.assertFalse(final.exists(), final)
                self.assertEqual(
                    sentinel.read_text(encoding="utf-8"),
                    "keep me\n",
                )
                self.assert_no_staging(output_root)

    def test_failure_backing_up_artifact_restores_prior_backups(self):
        for target_name in (
            "bacteria_id_clinical",
            "bacteria_id_conversion.json",
        ):
            with self.subTest(target_name=target_name):
                output_root = self.output_root(f"backup-failure-{target_name}")
                originals = self.seed_originals(output_root)
                target = output_root / target_name
                real_replace = os.replace

                def injected_replace(source, destination):
                    source_path = Path(source)
                    destination_path = Path(destination)
                    if (
                        source_path == target
                        and destination_path.parent.name.startswith(
                            ".bacteria-id.staging-"
                        )
                        and destination_path.name.startswith("backup-")
                    ):
                        raise OSError(
                            f"injected backup failure for {target_name}"
                        )
                    return real_replace(source, destination)

                with patch(
                    "rpe.io.bacteria_id.os.replace",
                    side_effect=injected_replace,
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "injected backup failure",
                    ):
                        self.build(output_root, overwrite=True)

                self.assert_originals(output_root, originals)
                self.assert_no_staging(output_root)

    def test_restore_failure_preserves_backup_and_manual_recovery_path(self):
        output_root = self.output_root()
        originals = self.seed_originals(output_root)
        reference, clinical, _ = self.final_paths(output_root)
        real_replace = os.replace
        publish_failed = False

        def injected_replace(source, destination):
            nonlocal publish_failed
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                destination_path == clinical
                and source_path.parent.name.startswith(
                    ".bacteria-id.staging-"
                )
                and not source_path.name.startswith("backup-")
            ):
                publish_failed = True
                raise OSError("injected publish failure")
            if (
                publish_failed
                and destination_path == reference
                and source_path.name == "backup-reference"
            ):
                raise OSError("injected restore failure")
            return real_replace(source, destination)

        with patch(
            "rpe.io.bacteria_id.os.replace",
            side_effect=injected_replace,
        ):
            with self.assertRaises(BacteriaIdValidationError) as caught:
                self.build(output_root, overwrite=True)

        self.assertEqual(caught.exception.path, "publication.restore")
        staging = [
            path
            for path in output_root.iterdir()
            if path.name.startswith(".bacteria-id.staging-")
        ]
        self.assertEqual(len(staging), 1)
        self.assertIn(str(staging[0]), str(caught.exception))
        backup = staging[0] / "backup-reference"
        self.assertTrue(backup.is_dir())
        self.assertEqual(
            artifact_snapshot(backup),
            originals["bacteria_id_reference"],
        )
        self.assertFalse(reference.exists())
        self.assertEqual(
            artifact_snapshot(clinical),
            originals["bacteria_id_clinical"],
        )
        receipt = self.final_paths(output_root)[2]
        self.assertEqual(
            artifact_snapshot(receipt),
            originals["bacteria_id_conversion.json"],
        )

    def test_publication_requires_exactly_three_artifacts(self):
        staging = self.temporary_root / "manual-staging"
        output_root = self.output_root("manual-output")
        staging.mkdir()
        output_root.mkdir()
        staged = staging / "artifact"
        staged.write_text("new\n", encoding="utf-8")
        artifact = _PublicationArtifact(
            staged=staged,
            final=output_root / "artifact",
            backup_name="backup-artifact",
        )

        with self.assertRaises(FrozenInstanceError):
            artifact.backup_name = "changed"
        with self.assertRaises(BacteriaIdValidationError) as caught:
            _publish_conversion_artifacts(
                staging,
                (artifact,),
                overwrite=False,
            )
        self.assertEqual(caught.exception.path, "publication.artifacts")
        self.assertTrue(staged.is_file())

    def test_publication_artifact_contract_is_immutable(self):
        staging = self.temporary_root / "manual-three-staging"
        output_root = self.output_root("manual-three-output")
        staging.mkdir()
        output_root.mkdir()
        artifacts = []
        for index in range(3):
            staged = staging / f"artifact-{index}"
            staged.write_text(f"new-{index}\n", encoding="utf-8")
            artifacts.append(
                _PublicationArtifact(
                    staged=staged,
                    final=output_root / f"artifact-{index}",
                    backup_name=f"backup-artifact-{index}",
                )
            )

        _publish_conversion_artifacts(
            staging,
            tuple(artifacts),
            overwrite=False,
        )
        for index in range(3):
            self.assertEqual(
                (output_root / f"artifact-{index}").read_text(
                    encoding="utf-8"
                ),
                f"new-{index}\n",
            )
        self.assertFalse(staging.exists())


class BacteriaIdRuntimeMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.output_root = self.temporary_root / "output"
        self.source = create_synthetic_bacteria_id_source(self.raw_root)
        self.inspection = _inspect_bacteria_id(
            self.raw_root,
            synthetic_contract(self.source),
        )

    def test_linux_ru_maxrss_is_converted_from_kib_to_bytes(self):
        self.assertEqual(_linux_ru_maxrss_to_bytes(0), 0)
        self.assertEqual(_linux_ru_maxrss_to_bytes(1), 1024)
        self.assertEqual(_linux_ru_maxrss_to_bytes(4096), 4_194_304)
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _linux_ru_maxrss_to_bytes(invalid)

    def test_measured_synthetic_conversion_has_exact_metrics_schema(self):
        with patch(
            "rpe.io.bacteria_id.inspect_bacteria_id",
            return_value=self.inspection,
        ):
            summary, metrics = _build_bacteria_id_unified_with_metrics(
                self.raw_root,
                self.output_root,
                overwrite=False,
            )

        self.assertIsInstance(metrics, BacteriaIdRuntimeMetrics)
        self.assertGreaterEqual(metrics.total_seconds, 0.0)
        self.assertGreaterEqual(metrics.inspection_seconds, 0.0)
        self.assertGreaterEqual(metrics.reference_build_seconds, 0.0)
        self.assertGreaterEqual(metrics.clinical_build_seconds, 0.0)
        self.assertGreaterEqual(metrics.validation_seconds, 0.0)
        self.assertGreaterEqual(metrics.peak_rss_raw, 0)
        self.assertEqual(
            metrics.peak_rss_bytes,
            _linux_ru_maxrss_to_bytes(metrics.peak_rss_raw),
        )
        self.assertEqual(
            metrics.combined_output_bytes,
            summary.reference.output_bytes + summary.clinical.output_bytes,
        )
        self.assertEqual(
            metrics.source_bytes,
            (self.raw_root / "data.zip").stat().st_size,
        )
        self.assertEqual(
            metrics.compression_ratio,
            metrics.combined_output_bytes / metrics.source_bytes,
        )
        receipt = json.loads(
            summary.receipt_path.read_text(encoding="utf-8"),
        )
        for forbidden in (
            "total_seconds",
            "inspection_seconds",
            "reference_build_seconds",
            "clinical_build_seconds",
            "validation_seconds",
            "peak_rss_raw",
            "peak_rss_bytes",
            "combined_output_bytes",
            "source_bytes",
            "compression_ratio",
        ):
            self.assertNotIn(forbidden, receipt)

    def test_validation_seconds_includes_receipt_parse_back_validation(self):
        real_validate_receipt = _validate_bacteria_id_receipt

        def delayed_validate_receipt(path):
            time.sleep(0.5)
            return real_validate_receipt(path)

        with patch(
            "rpe.io.bacteria_id.inspect_bacteria_id",
            return_value=self.inspection,
        ):
            with patch(
                "rpe.io.bacteria_id._validate_bacteria_id_receipt",
                side_effect=delayed_validate_receipt,
            ):
                _, metrics = _build_bacteria_id_unified_with_metrics(
                    self.raw_root,
                    self.output_root,
                    overwrite=False,
                )

        self.assertGreaterEqual(metrics.validation_seconds, 0.5)


class BacteriaIdCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.output_root = self.temporary_root / "output"
        self.source = create_synthetic_bacteria_id_source(self.raw_root)
        self.inspection = _inspect_bacteria_id(
            self.raw_root,
            synthetic_contract(self.source),
        )

    def run_main(self, *extra_arguments: str):
        builder = load_bacteria_builder_module()
        stdout = io.StringIO()
        stderr = io.StringIO()
        arguments = [
            "--raw-root",
            str(self.raw_root),
            "--output-root",
            str(self.output_root),
            *extra_arguments,
        ]
        with patch(
            "rpe.io.bacteria_id.inspect_bacteria_id",
            return_value=self.inspection,
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = builder.main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_cli_success_emits_one_exact_json_summary_offline(self):
        builder = load_bacteria_builder_module()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "rpe.io.bacteria_id.inspect_bacteria_id",
            return_value=self.inspection,
        ):
            with patch.object(
                socket.socket,
                "connect",
                side_effect=AssertionError("network connection attempted"),
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = builder.main(
                        [
                            "--raw-root",
                            str(self.raw_root),
                            "--output-root",
                            str(self.output_root),
                        ]
                    )

        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        document = json.loads(stdout.getvalue())
        self.assertEqual(
            stdout.getvalue(),
            canonical_json_bytes(document).decode("utf-8"),
        )
        self.assertEqual(
            set(document),
            {"status", "reference", "clinical", "receipt", "metrics"},
        )
        self.assertEqual(document["status"], "written")
        self.assertEqual(document["reference"]["record_count"], 120)
        self.assertEqual(document["clinical"]["record_count"], 100)
        self.assertEqual(
            set(document["metrics"]),
            {
                "total_seconds",
                "inspection_seconds",
                "reference_build_seconds",
                "clinical_build_seconds",
                "validation_seconds",
                "peak_rss_raw",
                "peak_rss_bytes",
                "combined_output_bytes",
                "source_bytes",
                "compression_ratio",
            },
        )
        self.assertEqual(
            document["receipt"]["path"],
            (self.output_root / "bacteria_id_conversion.json").as_posix(),
        )

    def test_cli_overwrite_flag_and_expected_failure_json(self):
        first_result, _, first_stderr = self.run_main()
        self.assertEqual(first_result, 0, first_stderr)

        failed_result, failed_stdout, failed_stderr = self.run_main()
        self.assertEqual(failed_result, 1)
        self.assertEqual(failed_stdout, "")
        self.assertEqual(len(failed_stderr.splitlines()), 1)
        error = json.loads(failed_stderr)
        self.assertEqual(error["status"], "failed")
        self.assertEqual(error["error_type"], "BacteriaIdValidationError")
        self.assertNotIn("Traceback", failed_stderr)

        overwrite_result, _, overwrite_stderr = self.run_main("--overwrite")
        self.assertEqual(overwrite_result, 0, overwrite_stderr)

    def test_cli_does_not_catch_keyboard_interrupt(self):
        builder = load_bacteria_builder_module()
        with patch.object(
            builder,
            "_build_bacteria_id_unified_with_metrics",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                builder.main(
                    [
                        "--raw-root",
                        str(self.raw_root),
                        "--output-root",
                        str(self.output_root),
                    ]
                )

    def test_cli_subprocess_converts_synthetic_source_with_offline_env(self):
        script = r"""
import os
import sys
from pathlib import Path
assert not any("proxy" in key.lower() for key in os.environ)
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
sys.path.insert(0, str(Path.cwd() / "tests"))
from bacteria_id_helpers import SyntheticBacteriaIdSource, file_sha256
from test_bacteria_id_adapter import synthetic_contract
import rpe.io.bacteria_id as bacteria_id
from tools.build_bacteria_id_unified import main
raw_root = Path(sys.argv[1])
member_sha256 = {
    path.name: file_sha256(path)
    for path in sorted((raw_root / "extracted").glob("*.npy"))
}
source = SyntheticBacteriaIdSource(
    raw_root=raw_root,
    archive_sha256=file_sha256(raw_root / "data.zip"),
    member_sha256=member_sha256,
)
contract = synthetic_contract(source)
bacteria_id.inspect_bacteria_id = lambda root: bacteria_id._inspect_bacteria_id(
    Path(root),
    contract,
)
raise SystemExit(main([
    "--raw-root", str(raw_root),
    "--output-root", sys.argv[2],
]))
"""
        environment = {
            key: value
            for key, value in os.environ.items()
            if "proxy" not in key.lower()
        }
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(self.raw_root),
                str(self.output_root),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout)["status"], "written")
        self.assertEqual(
            validate_dataset(
                self.output_root / "bacteria_id_reference"
            ).record_count,
            120,
        )
        self.assertEqual(
            validate_dataset(
                self.output_root / "bacteria_id_clinical"
            ).record_count,
            100,
        )


@unittest.skipUnless(
    (PRODUCTION_RAW_ROOT / "data.zip").is_file(),
    "retained Bacteria-ID source archive is unavailable",
)
class BacteriaIdProductionInspectionTest(unittest.TestCase):
    def test_public_api_inspects_the_complete_retained_source(self):
        inspection = inspect_bacteria_id(PRODUCTION_RAW_ROOT)

        self.assertEqual(
            inspection.archive_sha256,
            "05f0978e2bcfe96f7f89667734a925f3d2dd261337e00a6d051df30e3006074f",
        )
        self.assertEqual(
            inspection.axis_id,
            "91e468d92cd4215f23c6c785b4611dc6f1ffe39cbb9134d11c60345954f80378",
        )
        self.assertEqual(inspection.axis_shape, (1000,))
        self.assertEqual(inspection.axis_dtype, "float64")
        self.assertEqual(inspection.total_rows, 78500)
        self.assertEqual(
            {
                split: item.row_count
                for split, item in inspection.splits.items()
            },
            {
                "reference": 60000,
                "finetune": 3000,
                "test": 3000,
                "clinical2018": 10000,
                "clinical2019": 2500,
            },
        )
        self.assertEqual(
            dict(inspection.clinical_patient_blocks),
            {"clinical2018": 25, "clinical2019": 25},
        )
        self.assertLessEqual(
            inspection.axis_float32_max_abs_error,
            5.0e-5,
        )
        self.assertLessEqual(
            inspection.intensity_float32_max_abs_error,
            3.0e-8,
        )


if __name__ == "__main__":
    unittest.main()
