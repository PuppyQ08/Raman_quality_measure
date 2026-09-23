import copy
import json
import sys
import unittest
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rpe.io as rpe_io  # noqa: E402
from rpe.io.schema import (  # noqa: E402
    ArrayRef,
    LicenseStatus,
    PreprocessingStatus,
    PreprocessingStep,
    RecordArrays,
    SchemaValidationError,
    axis_id,
    metadata_to_record,
    record_to_metadata,
    validate_record,
)
from unified_helpers import (  # noqa: E402
    AXIS_DECREASING,
    AXIS_DECREASING_ID,
    AXIS_INCREASING,
    AXIS_INCREASING_ID,
    SOURCE_A_SHA256,
    corrected_record,
    raw_record,
    unknown_record,
)


class AxisAndArrayValidationTest(unittest.TestCase):
    def test_axis_id_matches_hand_checked_domain_separated_hashes(self):
        self.assertEqual(axis_id(AXIS_INCREASING), AXIS_INCREASING_ID)
        self.assertEqual(axis_id(AXIS_DECREASING), AXIS_DECREASING_ID)

    def test_validate_record_accepts_increasing_and_decreasing_axes(self):
        increasing = raw_record()
        decreasing = unknown_record()

        self.assertIs(validate_record(increasing), increasing)
        self.assertIs(validate_record(decreasing), decreasing)

    def test_validate_record_rejects_invalid_spectral_arrays(self):
        base = raw_record()
        cases = {
            "intensity dtype": replace(
                base,
                intensity=base.intensity.astype(np.float64),
            ),
            "wavenumber dtype": replace(
                base,
                wavenumber=base.wavenumber.astype(">f4"),
            ),
            "intensity empty": replace(
                base,
                intensity=np.array([], dtype=np.float32),
            ),
            "intensity dimension": replace(
                base,
                intensity=base.intensity.reshape(1, -1),
            ),
            "intensity finite": replace(
                base,
                intensity=np.array(
                    [1.0, np.nan, 3.0, 4.0],
                    dtype=np.float32,
                ),
            ),
            "array length": replace(
                base,
                intensity=np.array([1.0, 2.0], dtype=np.float32),
            ),
            "wavenumber duplicate": replace(
                base,
                wavenumber=np.array(
                    [100.0, 200.0, 200.0, 400.0],
                    dtype=np.float32,
                ),
            ),
            "wavenumber monotonic": replace(
                base,
                wavenumber=np.array(
                    [100.0, 300.0, 200.0, 400.0],
                    dtype=np.float32,
                ),
            ),
        }

        for expected_path, record in cases.items():
            with self.subTest(expected_path=expected_path):
                with self.assertRaisesRegex(
                    SchemaValidationError,
                    expected_path,
                ):
                    validate_record(record)


class MetadataValidationTest(unittest.TestCase):
    def assert_record_error(
        self,
        expected_path: str,
        record,
    ):
        with self.assertRaises(SchemaValidationError) as raised:
            validate_record(record)
        self.assertEqual(raised.exception.path, expected_path)

    def test_validate_record_rejects_invalid_record_and_dataset_ids(self):
        base = raw_record()
        cases = {
            "record_id": replace(base, record_id=""),
            "meta.dataset_id empty": replace(
                base,
                meta=replace(base.meta, dataset_id=""),
            ),
            "meta.dataset_id traversal": replace(
                base,
                meta=replace(base.meta, dataset_id="../escape"),
            ),
            "meta.dataset_id uppercase": replace(
                base,
                meta=replace(base.meta, dataset_id="UpperCase"),
            ),
            "meta.sample_id": replace(
                base,
                meta=replace(base.meta, sample_id=""),
            ),
        }

        for expected_path, record in cases.items():
            with self.subTest(expected_path=expected_path):
                self.assert_record_error(expected_path, record)

    def test_validate_record_rejects_invalid_acquisition_values(self):
        base = raw_record()
        cases = {
            "meta.excitation_nm positive": replace(
                base,
                meta=replace(base.meta, excitation_nm=0.0),
            ),
            "meta.excitation_nm boolean": replace(
                base,
                meta=replace(base.meta, excitation_nm=True),
            ),
            "meta.excitation_nm finite": replace(
                base,
                meta=replace(base.meta, excitation_nm=np.nan),
            ),
            "meta.integration_time_s positive": replace(
                base,
                meta=replace(base.meta, integration_time_s=-1.0),
            ),
            "meta.integration_time_s finite": replace(
                base,
                meta=replace(base.meta, integration_time_s=np.inf),
            ),
            "meta.n_accumulations positive": replace(
                base,
                meta=replace(base.meta, n_accumulations=0),
            ),
            "meta.n_accumulations boolean": replace(
                base,
                meta=replace(base.meta, n_accumulations=False),
            ),
        }

        for expected_path, record in cases.items():
            with self.subTest(expected_path=expected_path):
                self.assert_record_error(expected_path, record)

    def test_preprocessing_status_controls_steps_and_eligibility(self):
        raw = raw_record()
        corrected = corrected_record()
        unknown = unknown_record()

        self.assertTrue(raw.meta.eligible_for_preprocessing_evaluation)
        self.assertFalse(corrected.meta.eligible_for_preprocessing_evaluation)
        self.assertFalse(unknown.meta.eligible_for_preprocessing_evaluation)

        self.assert_record_error(
            "meta.preprocessing_steps",
            replace(
                raw,
                meta=replace(
                    raw.meta,
                    preprocessing_steps=unknown.meta.preprocessing_steps,
                ),
            ),
        )
        self.assert_record_error(
            "meta.preprocessing_steps",
            replace(
                corrected,
                meta=replace(corrected.meta, preprocessing_steps=()),
            ),
        )

    def test_validate_record_rejects_invalid_steps_and_source_metadata(self):
        base = unknown_record()
        valid_step = base.meta.preprocessing_steps[0]
        step_cases = {
            "meta.preprocessing_steps[0].operation": replace(
                valid_step,
                operation="",
            ),
            "meta.preprocessing_steps[0].description": replace(
                valid_step,
                description="",
            ),
            "meta.preprocessing_steps[0].evidence": replace(
                valid_step,
                evidence="",
            ),
        }
        for expected_path, step in step_cases.items():
            with self.subTest(expected_path=expected_path):
                self.assert_record_error(
                    expected_path,
                    replace(
                        base,
                        meta=replace(base.meta, preprocessing_steps=(step,)),
                    ),
                )

        metadata_cases = {
            "meta.source_metadata": {1: "non-string key"},
            "meta.source_metadata.nan": {"nan": np.nan},
            "meta.source_metadata.infinity": {"infinity": np.inf},
            "meta.source_metadata.unsupported": {"unsupported": {"set"}},
        }
        for expected_path, source_metadata in metadata_cases.items():
            with self.subTest(expected_path=expected_path):
                self.assert_record_error(
                    expected_path,
                    replace(
                        base,
                        meta=replace(
                            base.meta,
                            source_metadata=source_metadata,
                        ),
                    ),
                )

    def test_validate_record_rejects_invalid_array_targets(self):
        base = corrected_record()
        cases = {
            "targets.clean dtype": replace(
                base,
                targets=replace(
                    base.targets,
                    clean=base.targets.clean.astype(np.float64),
                ),
            ),
            "targets.clean finite": replace(
                base,
                targets=replace(
                    base.targets,
                    clean=np.array(
                        [3.5, 2.5, np.nan, 0.5],
                        dtype=np.float32,
                    ),
                ),
            ),
            "targets.clean shape": replace(
                base,
                targets=replace(
                    base.targets,
                    clean=np.array([3.5, 2.5], dtype=np.float32),
                ),
            ),
            "targets.baseline dimension": replace(
                base,
                targets=replace(
                    base.targets,
                    baseline=base.targets.baseline.reshape(1, -1),
                ),
            ),
        }

        for expected_path, record in cases.items():
            with self.subTest(expected_path=expected_path):
                self.assert_record_error(expected_path, record)

    def test_validate_record_rejects_invalid_scalar_and_peak_targets(self):
        base = raw_record()
        peak = base.targets.peaks[0]
        cases = {
            "targets.class_label": replace(
                base,
                targets=replace(base.targets, class_label=True),
            ),
            "targets.concentration finite": replace(
                base,
                targets=replace(base.targets, concentration=np.inf),
            ),
            "targets.concentrations": replace(
                base,
                targets=replace(
                    base.targets,
                    concentration=1.0,
                    concentrations={"glucose": 2.0},
                ),
            ),
            "targets.concentrations key": replace(
                base,
                targets=replace(
                    base.targets,
                    class_label=None,
                    concentrations={"": 2.0},
                ),
            ),
            "targets.concentrations.glucose finite": replace(
                base,
                targets=replace(
                    base.targets,
                    class_label=None,
                    concentrations={"glucose": np.nan},
                ),
            ),
            "targets.peaks[0].pos_cm1 finite": replace(
                base,
                targets=replace(
                    base.targets,
                    peaks=(replace(peak, pos_cm1=np.nan),),
                ),
            ),
            "targets.peaks[0].height finite": replace(
                base,
                targets=replace(
                    base.targets,
                    peaks=(replace(peak, height=np.inf),),
                ),
            ),
            "targets.peaks[0].fwhm positive": replace(
                base,
                targets=replace(
                    base.targets,
                    peaks=(replace(peak, fwhm=0.0),),
                ),
            ),
        }

        for expected_path, record in cases.items():
            with self.subTest(expected_path=expected_path):
                self.assert_record_error(expected_path, record)

    def test_validate_record_rejects_invalid_provenance(self):
        base = raw_record()
        cases = {
            "provenance.source_url": replace(
                base,
                provenance=replace(base.provenance, source_url="relative/path"),
            ),
            "provenance.sha256 uppercase": replace(
                base,
                provenance=replace(
                    base.provenance,
                    sha256=SOURCE_A_SHA256.upper(),
                ),
            ),
            "provenance.sha256 short": replace(
                base,
                provenance=replace(base.provenance, sha256="a" * 63),
            ),
            "provenance.sha256 hexadecimal": replace(
                base,
                provenance=replace(base.provenance, sha256="g" * 64),
            ),
            "provenance.license not stated": replace(
                base,
                provenance=replace(
                    base.provenance,
                    license="not stated",
                    license_status=LicenseStatus.STANDARDIZED,
                ),
            ),
            "provenance.license_status": replace(
                base,
                provenance=replace(
                    base.provenance,
                    license="CC0-1.0",
                    license_status=LicenseStatus.NOT_STATED,
                ),
            ),
            "provenance.retrieved_date": replace(
                base,
                provenance=replace(
                    base.provenance,
                    retrieved_date=datetime(2026, 8, 14),
                ),
            ),
            "provenance.source_artifact empty": replace(
                base,
                provenance=replace(base.provenance, source_artifact=""),
            ),
            "provenance.source_artifact POSIX": replace(
                base,
                provenance=replace(
                    base.provenance,
                    source_artifact="/absolute/file.txt",
                ),
            ),
            "provenance.source_artifact Windows": replace(
                base,
                provenance=replace(
                    base.provenance,
                    source_artifact=r"C:\absolute\file.txt",
                ),
            ),
            "provenance.source_artifact Windows drive": replace(
                base,
                provenance=replace(
                    base.provenance,
                    source_artifact=r"C:relative\file.txt",
                ),
            ),
            "provenance.source_artifact traversal": replace(
                base,
                provenance=replace(
                    base.provenance,
                    source_artifact="../outside/file.txt",
                ),
            ),
            "provenance.source_artifact NUL": replace(
                base,
                provenance=replace(
                    base.provenance,
                    source_artifact="archive.zip\x00member.txt",
                ),
            ),
        }

        for expected_path, record in cases.items():
            with self.subTest(expected_path=expected_path):
                self.assert_record_error(expected_path, record)

        archive_member = replace(
            base,
            provenance=replace(
                base.provenance,
                source_artifact="archive.zip/folder/member.txt",
            ),
        )
        self.assertIs(validate_record(archive_member), archive_member)

    def test_validate_record_accepts_empty_peak_assignment(self):
        base = raw_record()
        record = replace(
            base,
            targets=replace(
                base.targets,
                peaks=(replace(base.targets.peaks[0], assignment=""),),
            ),
        )

        self.assertIs(validate_record(record), record)


class MetadataSerializationTest(unittest.TestCase):
    def assert_records_equal(self, expected, actual):
        self.assertEqual(expected.record_id, actual.record_id)
        self.assertEqual(expected.meta, actual.meta)
        self.assertEqual(expected.provenance, actual.provenance)
        self.assertEqual(expected.targets.peaks, actual.targets.peaks)
        self.assertEqual(
            expected.targets.class_label,
            actual.targets.class_label,
        )
        self.assertEqual(
            expected.targets.concentration,
            actual.targets.concentration,
        )
        self.assertEqual(
            expected.targets.concentrations,
            actual.targets.concentrations,
        )
        np.testing.assert_array_equal(expected.intensity, actual.intensity)
        np.testing.assert_array_equal(expected.wavenumber, actual.wavenumber)
        for field in ("clean", "baseline"):
            expected_array = getattr(expected.targets, field)
            actual_array = getattr(actual.targets, field)
            if expected_array is None:
                self.assertIsNone(actual_array)
            else:
                np.testing.assert_array_equal(
                    expected_array,
                    actual_array,
                )

    def test_record_to_metadata_emits_exact_canonical_object(self):
        metadata = record_to_metadata(
            raw_record(),
            ArrayRef(axis_id=AXIS_INCREASING_ID, row=0),
        )

        self.assertEqual(
            metadata,
            {
                "record_id": "record_a_raw",
                "array_ref": {
                    "axis_id": AXIS_INCREASING_ID,
                    "row": 0,
                },
                "meta": {
                    "dataset_id": "fixture_mixed_axes",
                    "sample_id": "sample-a",
                    "instrument": "fixture-scope",
                    "excitation_nm": 785.0,
                    "integration_time_s": 1.0,
                    "n_accumulations": 2,
                    "grating": "fixture-grating",
                    "detector": "fixture-detector",
                    "preprocessing_status": "known_raw",
                    "preprocessing_steps": [],
                    "source_metadata": {
                        "split": "train",
                        "replicate": 1,
                    },
                },
                "targets": {
                    "clean_present": False,
                    "baseline_present": False,
                    "peaks": [
                        {
                            "pos_cm1": 200.0,
                            "height": 2.0,
                            "fwhm": 8.0,
                            "assignment": "fixture_peak",
                        },
                    ],
                    "class_label": 0,
                    "concentration": None,
                    "concentrations": None,
                },
                "provenance": {
                    "source_url": "https://example.org/source",
                    "license": "CC0-1.0",
                    "license_status": "standardized",
                    "sha256": SOURCE_A_SHA256,
                    "retrieved_date": "2026-08-14",
                    "source_artifact": "source_a.txt",
                },
            },
        )

    def test_metadata_round_trip_preserves_all_fixture_record_fields(self):
        fixtures = [
            (raw_record(), AXIS_INCREASING_ID, 0),
            (corrected_record(), AXIS_INCREASING_ID, 1),
            (unknown_record(), AXIS_DECREASING_ID, 0),
        ]
        for expected, expected_axis_id, row in fixtures:
            with self.subTest(record_id=expected.record_id):
                metadata = record_to_metadata(
                    expected,
                    ArrayRef(axis_id=expected_axis_id, row=row),
                )
                arrays = RecordArrays(
                    intensity=expected.intensity.copy(),
                    wavenumber=expected.wavenumber.copy(),
                    clean=(
                        None
                        if expected.targets.clean is None
                        else expected.targets.clean.copy()
                    ),
                    baseline=(
                        None
                        if expected.targets.baseline is None
                        else expected.targets.baseline.copy()
                    ),
                )

                actual = metadata_to_record(metadata, arrays)

                self.assert_records_equal(expected, actual)

    def test_metadata_round_trip_preserves_none_and_empty_peaks(self):
        base = raw_record()
        for peaks in (None, ()):
            with self.subTest(peaks=peaks):
                expected = replace(
                    base,
                    record_id=f"peaks_{'none' if peaks is None else 'empty'}",
                    targets=replace(base.targets, peaks=peaks),
                )
                metadata = record_to_metadata(
                    expected,
                    ArrayRef(axis_id=AXIS_INCREASING_ID, row=0),
                )
                actual = metadata_to_record(
                    metadata,
                    RecordArrays(
                        intensity=expected.intensity.copy(),
                        wavenumber=expected.wavenumber.copy(),
                        clean=None,
                        baseline=None,
                    ),
                )

                self.assertIs(actual.targets.peaks, peaks)

    def test_metadata_to_record_rejects_unknown_and_missing_keys(self):
        base = record_to_metadata(
            raw_record(),
            ArrayRef(axis_id=AXIS_INCREASING_ID, row=0),
        )
        cases = []
        for object_name in (
            "root",
            "array_ref",
            "meta",
            "targets",
            "provenance",
        ):
            container = base if object_name == "root" else base[object_name]
            for mutation in ("unknown", "missing"):
                metadata = copy.deepcopy(base)
                target = (
                    metadata
                    if object_name == "root"
                    else metadata[object_name]
                )
                if mutation == "unknown":
                    target["unexpected"] = "value"
                else:
                    key = next(iter(container))
                    del target[key]
                cases.append((object_name, mutation, metadata))

        arrays = RecordArrays(
            intensity=raw_record().intensity,
            wavenumber=raw_record().wavenumber,
            clean=None,
            baseline=None,
        )
        for object_name, mutation, metadata in cases:
            with self.subTest(object_name=object_name, mutation=mutation):
                with self.assertRaisesRegex(
                    SchemaValidationError,
                    object_name,
                ):
                    metadata_to_record(metadata, arrays)

    def test_metadata_to_record_rejects_array_presence_mismatch(self):
        corrected = corrected_record()
        metadata = record_to_metadata(
            corrected,
            ArrayRef(axis_id=AXIS_INCREASING_ID, row=1),
        )
        cases = {
            "targets.clean_present": RecordArrays(
                intensity=corrected.intensity,
                wavenumber=corrected.wavenumber,
                clean=None,
                baseline=corrected.targets.baseline,
            ),
            "targets.baseline_present": RecordArrays(
                intensity=corrected.intensity,
                wavenumber=corrected.wavenumber,
                clean=corrected.targets.clean,
                baseline=None,
            ),
        }
        for expected_path, arrays in cases.items():
            with self.subTest(expected_path=expected_path):
                with self.assertRaisesRegex(
                    SchemaValidationError,
                    expected_path,
                ):
                    metadata_to_record(metadata, arrays)

    def test_record_to_metadata_rejects_invalid_array_ref(self):
        cases = {
            "array_ref.axis_id": ArrayRef(axis_id="not-a-sha256", row=0),
            "array_ref.row": ArrayRef(
                axis_id=AXIS_INCREASING_ID,
                row=-1,
            ),
            "array_ref.row boolean": ArrayRef(
                axis_id=AXIS_INCREASING_ID,
                row=True,
            ),
        }
        for expected_path, array_ref in cases.items():
            with self.subTest(expected_path=expected_path):
                with self.assertRaisesRegex(
                    SchemaValidationError,
                    expected_path,
                ):
                    record_to_metadata(raw_record(), array_ref)

    def test_record_to_metadata_rejects_axis_ref_for_different_wavenumber(self):
        with self.assertRaises(SchemaValidationError) as raised:
            record_to_metadata(
                raw_record(),
                ArrayRef(axis_id=AXIS_DECREASING_ID, row=0),
            )

        self.assertEqual(raised.exception.path, "array_ref.axis_id")

    def test_public_io_api_exports_task2_serializers(self):
        self.assertIs(rpe_io.record_to_metadata, record_to_metadata)
        self.assertIs(rpe_io.metadata_to_record, metadata_to_record)

    def test_record_to_metadata_normalizes_numpy_scalars_for_json(self):
        raw = raw_record()
        numpy_scalar_record = replace(
            raw,
            meta=replace(
                raw.meta,
                excitation_nm=np.float32(785.0),
                integration_time_s=np.float64(1.0),
                n_accumulations=np.int64(2),
            ),
            targets=replace(
                raw.targets,
                peaks=(
                    replace(
                        raw.targets.peaks[0],
                        pos_cm1=np.float32(200.0),
                        height=np.float64(2.0),
                        fwhm=np.float32(8.0),
                    ),
                ),
                class_label=np.int64(0),
            ),
        )
        unknown = unknown_record()
        numpy_concentrations = replace(
            unknown,
            targets=replace(
                unknown.targets,
                concentrations={
                    "acetate": np.float32(1.0),
                    "glucose": np.float64(2.0),
                },
            ),
        )

        for record, expected_axis_id in (
            (numpy_scalar_record, AXIS_INCREASING_ID),
            (numpy_concentrations, AXIS_DECREASING_ID),
        ):
            with self.subTest(record_id=record.record_id):
                metadata = record_to_metadata(
                    record,
                    ArrayRef(axis_id=expected_axis_id, row=0),
                )
                encoded = json.dumps(metadata, allow_nan=False)
                self.assertIsInstance(encoded, str)


if __name__ == "__main__":
    unittest.main()
