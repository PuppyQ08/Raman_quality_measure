from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rpe.downstream.bacteria_id import (  # noqa: E402
    BacteriaIdBatchLoader,
    BacteriaIdBatchValidationError,
)
from rpe.methods.classical.savitzky_golay import (  # noqa: E402
    SavitzkyGolayPipeline,
    SavitzkyGolayValidationError,
    load_savitzky_golay_pipeline,
)
from rpe.io.schema import PreprocessingStatus, Targets  # noqa: E402
from rpe.io.store import write_dataset  # noqa: E402
from unified_helpers import corrected_record  # noqa: E402


DATASET_ID = "bacteria_id_reference"
CLASS_LABELS = {0: "class_zero", 1: "class_one"}
EXPECTED_SPLITS = ("finetune", "reference", "test")
SG_CONFIG_PATH = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d1_sg11_poly3_interp.json"
)
SG_CONFIG_SHA256 = (
    "775c3dba4c4d6bb13ab0ac44abf3e48011f90ce8bc268f7f79632ed35cb039aa"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bacteria_record(
    source_split: str,
    source_row: int,
    class_label: int,
):
    base = corrected_record()
    split_offset = {
        "finetune": 10.0,
        "reference": 20.0,
        "test": 30.0,
    }.get(source_split, 40.0)
    return replace(
        base,
        record_id=f"{source_split}-{source_row:06d}",
        intensity=np.array(
            [
                split_offset + source_row,
                split_offset + source_row + 0.25,
                split_offset + source_row + 0.5,
                split_offset + source_row + 0.75,
            ],
            dtype=np.float32,
        ),
        meta=replace(
            base.meta,
            dataset_id=DATASET_ID,
            sample_id=None,
            preprocessing_status=PreprocessingStatus.KNOWN_CORRECTED,
            source_metadata={
                "source_split": source_split,
                "source_row": source_row,
                "split_role": {
                    "finetune": "optical_system_adaptation",
                    "reference": "pretraining",
                    "test": "independent_test",
                }.get(source_split, "invalid"),
                "label_space": "isolate",
            },
        ),
        targets=Targets(class_label=class_label),
    )


class BacteriaIdBatchLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.dataset_path = (
            Path(self.temporary_directory.name) / DATASET_ID
        )

    def write_fixture(self, records=None):
        if records is None:
            records = [
                bacteria_record("finetune", 0, 0),
                bacteria_record("finetune", 1, 1),
                bacteria_record("reference", 0, 0),
                bacteria_record("reference", 1, 1),
                bacteria_record("reference", 2, 0),
                bacteria_record("test", 0, 0),
                bacteria_record("test", 1, 1),
            ]
        observed_labels = sorted(
            {record.targets.class_label for record in records}
        )
        write_dataset(
            records,
            self.dataset_path,
            dataset_id=DATASET_ID,
            class_labels={
                class_label: CLASS_LABELS[class_label]
                for class_label in observed_labels
            },
        )

    def test_batches_preserve_split_rows_labels_values_and_read_only_arrays(self):
        self.write_fixture()

        with BacteriaIdBatchLoader(
            self.dataset_path,
            batch_size=2,
        ) as loader:
            batches = list(loader.iter_batches())

        self.assertEqual(
            [
                (batch.source_split, batch.record_ids)
                for batch in batches
            ],
            [
                (
                    "finetune",
                    ("finetune-000000", "finetune-000001"),
                ),
                (
                    "reference",
                    ("reference-000000", "reference-000001"),
                ),
                ("reference", ("reference-000002",)),
                ("test", ("test-000000", "test-000001")),
            ],
        )
        np.testing.assert_array_equal(
            batches[0].intensity,
            np.array(
                [
                    [10.0, 10.25, 10.5, 10.75],
                    [11.0, 11.25, 11.5, 11.75],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            batches[0].class_labels,
            np.array([0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            batches[0].source_rows,
            np.array([0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            batches[0].wavenumber,
            np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
        )
        for batch in batches:
            self.assertEqual(batch.intensity.dtype, np.dtype("<f4"))
            self.assertEqual(batch.class_labels.dtype, np.dtype("<i8"))
            self.assertEqual(batch.source_rows.dtype, np.dtype("<i8"))
            self.assertEqual(batch.wavenumber.dtype, np.dtype("<f4"))
            self.assertFalse(batch.intensity.flags.writeable)
            self.assertFalse(batch.class_labels.flags.writeable)
            self.assertFalse(batch.source_rows.flags.writeable)
            self.assertFalse(batch.wavenumber.flags.writeable)
        with self.assertRaisesRegex(ValueError, "read-only"):
            batches[0].intensity[0, 0] = -1.0

    def test_reader_rejects_invalid_batch_size_and_use_outside_context(self):
        self.write_fixture()

        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(BacteriaIdBatchValidationError):
                    BacteriaIdBatchLoader(
                        self.dataset_path,
                        batch_size=invalid,
                    )

        loader = BacteriaIdBatchLoader(self.dataset_path, batch_size=2)
        with self.assertRaisesRegex(RuntimeError, "not open"):
            list(loader.iter_batches())
        with loader:
            list(loader.iter_batches())
        with self.assertRaisesRegex(RuntimeError, "not open"):
            list(loader.iter_batches())

    def test_reader_rejects_noncanonical_source_split_before_yielding_it(self):
        self.write_fixture(
            [bacteria_record("invalid_split", 0, 0)]
        )

        with BacteriaIdBatchLoader(
            self.dataset_path,
            batch_size=2,
        ) as loader:
            with self.assertRaisesRegex(
                BacteriaIdBatchValidationError,
                "source_split",
            ):
                list(loader.iter_batches())

    def test_reader_rejects_missing_initial_or_terminal_source_split(self):
        cases = (
            (
                "missing initial finetune",
                [
                    bacteria_record("reference", 0, 0),
                    bacteria_record("test", 0, 0),
                ],
            ),
            (
                "missing terminal test",
                [
                    bacteria_record("finetune", 0, 0),
                    bacteria_record("reference", 0, 0),
                ],
            ),
        )
        for label, records in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    dataset_path = Path(temporary_directory) / DATASET_ID
                    write_dataset(
                        records,
                        dataset_path,
                        dataset_id=DATASET_ID,
                        class_labels={0: CLASS_LABELS[0]},
                    )
                    with BacteriaIdBatchLoader(
                        dataset_path,
                        batch_size=2,
                    ) as loader:
                        with self.assertRaisesRegex(
                            BacteriaIdBatchValidationError,
                            "source_split",
                        ):
                            list(loader.iter_batches())

    def test_reader_does_not_modify_unified_dataset_files(self):
        self.write_fixture()
        before = {
            path.name: (path.stat().st_size, file_sha256(path))
            for path in self.dataset_path.iterdir()
            if path.is_file()
        }

        with BacteriaIdBatchLoader(
            self.dataset_path,
            batch_size=3,
        ) as loader:
            list(loader.iter_batches())

        after = {
            path.name: (path.stat().st_size, file_sha256(path))
            for path in self.dataset_path.iterdir()
            if path.is_file()
        }
        self.assertEqual(after, before)


class BacteriaIdRetainedBatchLoaderTest(unittest.TestCase):
    dataset_path = ROOT / "data" / "unified" / DATASET_ID

    @unittest.skipUnless(
        dataset_path.is_dir(),
        "retained Bacteria-ID unified dataset is unavailable",
    )
    def test_retained_batches_match_exact_d1_contract_without_mutation(self):
        files = tuple(sorted(self.dataset_path.iterdir()))
        before = {
            path.name: (path.stat().st_size, file_sha256(path))
            for path in files
            if path.is_file()
        }
        split_counts = Counter()
        class_counts = {
            source_split: Counter()
            for source_split in EXPECTED_SPLITS
        }
        observed_splits = []
        first_ids = {}
        last_ids = {}

        with BacteriaIdBatchLoader(
            self.dataset_path,
            batch_size=4096,
        ) as loader:
            for batch in loader.iter_batches():
                if not observed_splits or observed_splits[-1] != batch.source_split:
                    observed_splits.append(batch.source_split)
                split_counts[batch.source_split] += len(batch.record_ids)
                class_counts[batch.source_split].update(
                    batch.class_labels.tolist()
                )
                first_ids.setdefault(
                    batch.source_split,
                    batch.record_ids[0],
                )
                last_ids[batch.source_split] = batch.record_ids[-1]
                self.assertEqual(batch.intensity.shape[1], 1000)
                self.assertTrue(np.isfinite(batch.intensity).all())
                self.assertTrue(np.isfinite(batch.wavenumber).all())
                self.assertFalse(batch.intensity.flags.writeable)
                self.assertFalse(batch.wavenumber.flags.writeable)

        self.assertEqual(observed_splits, list(EXPECTED_SPLITS))
        self.assertEqual(
            split_counts,
            {
                "finetune": 3000,
                "reference": 60000,
                "test": 3000,
            },
        )
        self.assertEqual(
            first_ids,
            {
                "finetune": "finetune-000000",
                "reference": "reference-000000",
                "test": "test-000000",
            },
        )
        self.assertEqual(
            last_ids,
            {
                "finetune": "finetune-002999",
                "reference": "reference-059999",
                "test": "test-002999",
            },
        )
        for source_split, expected_per_class in (
            ("finetune", 100),
            ("reference", 2000),
            ("test", 100),
        ):
            self.assertEqual(
                class_counts[source_split],
                {class_label: expected_per_class for class_label in range(30)},
            )

        after = {
            path.name: (path.stat().st_size, file_sha256(path))
            for path in files
            if path.is_file()
        }
        self.assertEqual(after, before)

    @unittest.skipUnless(
        dataset_path.is_dir(),
        "retained Bacteria-ID unified dataset is unavailable",
    )
    def test_first_retained_split_batch_loads_within_five_seconds_after_open(self):
        with BacteriaIdBatchLoader(
            self.dataset_path,
            batch_size=4096,
        ) as loader:
            started = perf_counter()
            first_batch = next(loader.iter_batches())
            elapsed_seconds = perf_counter() - started

        self.assertEqual(first_batch.source_split, "finetune")
        self.assertEqual(len(first_batch.record_ids), 3000)
        self.assertLess(
            elapsed_seconds,
            5.0,
            f"first split batch took {elapsed_seconds:.3f}s",
        )


class SavitzkyGolayPipelineTest(unittest.TestCase):
    def test_named_pipeline_constructor_cannot_override_frozen_parameters(self):
        pipeline = SavitzkyGolayPipeline()

        self.assertEqual(pipeline.window_length, 11)
        self.assertEqual(pipeline.polyorder, 3)
        with self.assertRaises(TypeError):
            SavitzkyGolayPipeline(window_length=9)

    def test_frozen_config_loads_exact_approved_pipeline(self):
        pipeline = load_savitzky_golay_pipeline(SG_CONFIG_PATH)

        self.assertIsInstance(pipeline, SavitzkyGolayPipeline)
        self.assertEqual(pipeline.pipeline_id, "sg11_poly3_interp")
        self.assertEqual(
            pipeline.input_condition_id,
            "released_input_control",
        )
        self.assertEqual(
            pipeline.condition_id,
            "released_input_plus_sg",
        )
        self.assertEqual(pipeline.window_length, 11)
        self.assertEqual(pipeline.polyorder, 3)
        self.assertEqual(pipeline.deriv, 0)
        self.assertEqual(pipeline.mode, "interp")
        self.assertEqual(pipeline.array_axis, -1)
        self.assertEqual(pipeline.coordinate_mode, "index")
        self.assertEqual(pipeline.output_dtype, "float32")
        self.assertEqual(
            hashlib.sha256(SG_CONFIG_PATH.read_bytes()).hexdigest(),
            SG_CONFIG_SHA256,
        )

    def test_transform_matches_hand_derived_central_impulse_coefficients(self):
        pipeline = load_savitzky_golay_pipeline(SG_CONFIG_PATH)
        spectrum = np.zeros(21, dtype=np.float32)
        spectrum[10] = 1.0

        transformed = pipeline.transform(spectrum)

        expected_central = (
            np.array(
                [-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36],
                dtype=np.float32,
            )
            / np.float32(429)
        )
        np.testing.assert_allclose(
            transformed[5:16],
            expected_central,
            rtol=0.0,
            atol=1e-7,
        )

    def test_transform_preserves_cubic_edges_with_interp_mode(self):
        pipeline = load_savitzky_golay_pipeline(SG_CONFIG_PATH)
        coordinate = np.arange(15, dtype=np.float32)
        cubic = (
            np.float32(2.0)
            + np.float32(3.0) * coordinate
            - np.float32(0.5) * coordinate**2
            + np.float32(0.125) * coordinate**3
        )

        transformed = pipeline.transform(cubic)

        np.testing.assert_allclose(
            transformed,
            cubic,
            rtol=0.0,
            atol=2e-4,
        )

    def test_transform_returns_independent_read_only_float32_batch(self):
        pipeline = load_savitzky_golay_pipeline(SG_CONFIG_PATH)
        spectra = np.stack(
            (
                np.linspace(0.0, 1.0, 21, dtype=np.float32),
                np.linspace(1.0, 0.0, 21, dtype=np.float32),
            )
        )
        before = spectra.copy()

        transformed = pipeline.transform(spectra)

        np.testing.assert_array_equal(spectra, before)
        self.assertEqual(transformed.shape, spectra.shape)
        self.assertEqual(transformed.dtype, np.dtype("<f4"))
        self.assertFalse(transformed.flags.writeable)
        self.assertFalse(np.shares_memory(transformed, spectra))
        with self.assertRaisesRegex(ValueError, "read-only"):
            transformed[0, 0] = -1.0

    def test_transform_rejects_wrong_dtype_dimension_length_and_nonfinite(self):
        pipeline = load_savitzky_golay_pipeline(SG_CONFIG_PATH)
        cases = (
            ("dtype", np.arange(21, dtype=np.float64)),
            ("dimension", np.zeros((1, 1, 21), dtype=np.float32)),
            ("empty", np.empty((0, 21), dtype=np.float32)),
            ("length", np.zeros(10, dtype=np.float32)),
            (
                "finite",
                np.array([0.0] * 20 + [np.nan], dtype=np.float32),
            ),
        )
        for expected_path, value in cases:
            with self.subTest(expected_path=expected_path):
                with self.assertRaisesRegex(
                    SavitzkyGolayValidationError,
                    expected_path,
                ):
                    pipeline.transform(value)

    def test_loader_rejects_noncanonical_or_mutated_config(self):
        canonical = json.loads(SG_CONFIG_PATH.read_text(encoding="utf-8"))
        cases = (
            (
                "noncanonical",
                json.dumps(canonical, indent=2).encode("utf-8"),
            ),
            (
                "window_length",
                (
                    json.dumps(
                        {
                            **canonical,
                            "steps": [
                                {
                                    **canonical["steps"][0],
                                    "window_length": 9,
                                }
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "window_length",
                (
                    json.dumps(
                        {
                            **canonical,
                            "steps": [
                                {
                                    **canonical["steps"][0],
                                    "window_length": 11.0,
                                }
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "unexpected",
                (
                    json.dumps(
                        {**canonical, "unexpected": True},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8"),
            ),
            (
                "nonfinite",
                (
                    SG_CONFIG_PATH.read_text(encoding="utf-8")
                    .replace('"window_length":11', '"window_length":NaN')
                    .encode("utf-8")
                ),
            ),
        )
        for expected_reason, raw in cases:
            with self.subTest(expected_reason=expected_reason):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = Path(temporary_directory) / "config.json"
                    path.write_bytes(raw)
                    with self.assertRaisesRegex(
                        SavitzkyGolayValidationError,
                        expected_reason,
                    ):
                        load_savitzky_golay_pipeline(path)


if __name__ == "__main__":
    unittest.main()
