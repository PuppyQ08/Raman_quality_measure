from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from phase1_runner_helpers import (  # noqa: E402
    FIXTURE_GLOBAL_SEED,
    decreasing_axis_float32,
    decreasing_intensity_float32,
    increasing_axis_float32,
    increasing_intensity_float32,
    inventory_rows,
    phase1_fixture_config,
    phase1_fixture_records,
    write_custom_phase1_fixture_dataset,
    write_phase1_fixture_dataset,
)
from rpe.io.schema import PreprocessingStatus, PreprocessingStep  # noqa: E402
from rpe.io.store import DatasetValidationError, UnifiedDataset  # noqa: E402
from rpe.runner.phase1_selection import (  # noqa: E402
    Phase1SelectionError,
    SelectedSourceRow,
    hamilton_class_quotas,
    load_phase1_source,
    load_source_inventory,
    select_source_rows,
    selected_records_jsonl_bytes,
    source_subset_jsonl_bytes,
)


def canonical_jsonl_bytes(rows: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


class HamiltonSelectionTest(unittest.TestCase):
    def test_tiny_hamilton_quotas_and_sample_round_robin_are_literal(self):
        rows = inventory_rows(
            {
                0: [("a0", "s0"), ("a1", "s0"), ("a2", "s1")],
                1: [("b0", "s2"), ("b1", "s3")],
                2: [("c0", "s4")],
            }
        )
        self.assertEqual(hamilton_class_quotas(rows, 5), {0: 2, 1: 2, 2: 1})
        selected = select_source_rows(rows, global_seed=20260817, subset_size=5)
        self.assertEqual(
            tuple(row.record_id for row in selected),
            ("a1", "a2", "b0", "b1", "c0"),
        )
        self.assertEqual(
            tuple(row.selection_rank for row in selected),
            (0, 1, 2, 3, 4),
        )

    def test_selection_is_deterministic_across_input_permutations(self):
        ordered = inventory_rows(
            {
                0: [("a0", "s0"), ("a1", "s0"), ("a2", "s1"), ("a3", "s2")],
                1: [("b0", "s3"), ("b1", "s4"), ("b2", "s5")],
                2: [("c0", "s6"), ("c1", "s6"), ("c2", "s7")],
            }
        )
        permuted = tuple(reversed(ordered))
        selected_a = select_source_rows(
            ordered,
            global_seed=FIXTURE_GLOBAL_SEED,
            subset_size=8,
        )
        selected_b = select_source_rows(
            permuted,
            global_seed=FIXTURE_GLOBAL_SEED,
            subset_size=8,
        )
        self.assertEqual(selected_a, selected_b)

    def test_selected_records_jsonl_payload_is_canonical(self):
        rows = (
            SelectedSourceRow(
                selection_rank=0,
                record_id="a1",
                sample_id="s0",
                class_label=0,
                mineral_name="mineral-0",
                axis_id="axis-0",
            ),
            SelectedSourceRow(
                selection_rank=1,
                record_id="b0",
                sample_id="s1",
                class_label=1,
                mineral_name="mineral-1",
                axis_id="axis-1",
            ),
        )
        expected = canonical_jsonl_bytes(
            (
                {
                    "axis_id": "axis-0",
                    "class_label": 0,
                    "mineral_name": "mineral-0",
                    "record_id": "a1",
                    "sample_id": "s0",
                    "selection_rank": 0,
                },
                {
                    "axis_id": "axis-1",
                    "class_label": 1,
                    "mineral_name": "mineral-1",
                    "record_id": "b0",
                    "sample_id": "s1",
                    "selection_rank": 1,
                },
            )
        )
        payload = selected_records_jsonl_bytes(rows)
        self.assertEqual(payload, expected)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            hashlib.sha256(expected).hexdigest(),
        )


class SourceInventoryTest(unittest.TestCase):
    def test_load_source_inventory_returns_sorted_metadata_only_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = write_phase1_fixture_dataset(Path(tmp))
            config = phase1_fixture_config(
                dataset_path,
                subset_size=8,
                shard_source_count=4,
            )

            rows = load_source_inventory(config)

        self.assertEqual(len(rows), 12)
        self.assertEqual(
            tuple(row.record_id for row in rows),
            (
                "a-00",
                "a-01",
                "a-02",
                "b-00",
                "b-01",
                "b-02",
                "b-03",
                "c-00",
                "c-01",
                "c-02",
                "c-03",
                "decreasing-record",
            ),
        )
        self.assertTrue(all(row.sample_id for row in rows))
        self.assertTrue(all(isinstance(row.class_label, int) for row in rows))
        self.assertTrue(all(row.axis_id for row in rows))

    def test_load_source_inventory_rejects_mismatched_bound_file_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = write_phase1_fixture_dataset(Path(tmp))
            config = phase1_fixture_config(
                dataset_path,
                subset_size=8,
                shard_source_count=4,
            )
            bad_files = dict(config.source_files)
            bad_files["records.jsonl"] = replace(
                bad_files["records.jsonl"],
                sha256="0" * 64,
            )
            bad_config = replace(config, source_files=bad_files)

            with self.assertRaisesRegex(
                Phase1SelectionError,
                re.escape("records.jsonl: sha256 mismatch"),
            ):
                load_source_inventory(bad_config)

    def test_load_source_inventory_rejects_non_raw_and_missing_sample_or_class(self):
        base_records = list(phase1_fixture_records())
        bad_status = replace(
            base_records[0],
            meta=replace(
                base_records[0].meta,
                preprocessing_status=PreprocessingStatus.KNOWN_CORRECTED,
                preprocessing_steps=(
                    PreprocessingStep(
                        operation="fixture_step",
                        description="fixture documented preprocessing step",
                        evidence="fixture evidence",
                    ),
                ),
            ),
        )
        bad_sample = replace(
            base_records[1],
            meta=replace(base_records[1].meta, sample_id=None),
        )
        bad_class = replace(
            base_records[2],
            targets=replace(base_records[2].targets, class_label=None),
        )
        cases = (
            ("known_raw", [bad_status, *base_records[1:]], "preprocessing_status"),
            ("sample_id", [base_records[0], bad_sample, *base_records[2:]], "sample_id"),
            ("class_label", [*base_records[:2], bad_class, *base_records[3:]], "class_label"),
        )
        for label, records, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                dataset_path = write_custom_phase1_fixture_dataset(
                    Path(tmp),
                    records=records,
                )
                config = phase1_fixture_config(
                    dataset_path,
                    subset_size=8,
                    shard_source_count=4,
                )
                with self.assertRaisesRegex(Phase1SelectionError, expected):
                    load_source_inventory(config)

    def test_load_source_inventory_rejects_subset_larger_than_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = write_phase1_fixture_dataset(Path(tmp))
            config = phase1_fixture_config(
                dataset_path,
                subset_size=13,
                shard_source_count=4,
            )

            with self.assertRaisesRegex(
                Phase1SelectionError,
                re.escape("subset_size exceeds eligible source count"),
            ):
                load_source_inventory(config)

    def test_inventory_input_config_is_not_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = write_phase1_fixture_dataset(Path(tmp))
            config = phase1_fixture_config(
                dataset_path,
                subset_size=8,
                shard_source_count=4,
            )
            before_files = dict(config.source_files)
            before_related = dict(config.related_provenance)

            load_source_inventory(config)

        self.assertEqual(dict(config.source_files), before_files)
        self.assertEqual(dict(config.related_provenance), before_related)


class SourceConversionTest(unittest.TestCase):
    def test_decreasing_axis_is_reversed_without_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = write_phase1_fixture_dataset(Path(tmp))
            with UnifiedDataset.open(dataset_path, verify_checksums=True) as dataset:
                row = SelectedSourceRow(
                    selection_rank=0,
                    record_id="decreasing-record",
                    sample_id="sample-b",
                    class_label=0,
                    mineral_name="Mineral-A",
                    axis_id=hashlib.sha256(
                        b"unused-in-test"
                    ).hexdigest(),
                )
                source = load_phase1_source(dataset, row)

        self.assertEqual(
            source.spectrum.spectrum_id,
            "rruff_raman_raw::decreasing-record",
        )
        np.testing.assert_array_equal(
            source.spectrum.axis_cm1,
            decreasing_axis_float32().astype("<f8")[::-1],
        )
        np.testing.assert_array_equal(
            source.spectrum.intensity,
            decreasing_intensity_float32().astype("<f8")[::-1],
        )
        self.assertEqual(source.original_axis_orientation, "decreasing")
        self.assertFalse(source.spectrum.axis_cm1.flags.writeable)
        self.assertFalse(source.spectrum.intensity.flags.writeable)

    def test_source_subset_jsonl_commits_conversion_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = write_phase1_fixture_dataset(Path(tmp))
            with UnifiedDataset.open(dataset_path, verify_checksums=True) as dataset:
                rows = (
                    SelectedSourceRow(
                        selection_rank=0,
                        record_id="a-00",
                        sample_id="sample-a",
                        class_label=0,
                        mineral_name="Mineral-A",
                        axis_id=hashlib.sha256(b"axis-a").hexdigest(),
                    ),
                    SelectedSourceRow(
                        selection_rank=1,
                        record_id="decreasing-record",
                        sample_id="sample-b",
                        class_label=0,
                        mineral_name="Mineral-A",
                        axis_id=hashlib.sha256(b"axis-b").hexdigest(),
                    ),
                )
                sources = tuple(load_phase1_source(dataset, row) for row in rows)
                payload = source_subset_jsonl_bytes(sources)

        expected_rows = tuple(
            {
                "axis_id": source.selection.axis_id,
                "class_label": source.selection.class_label,
                "mineral_name": source.selection.mineral_name,
                "normalized_axis_float64_sha256": source.normalized_axis_float64_sha256,
                "normalized_intensity_float64_sha256": source.normalized_intensity_float64_sha256,
                "original_axis_orientation": source.original_axis_orientation,
                "provenance": {
                    key: source.provenance[key]
                    for key in sorted(source.provenance)
                },
                "record_id": source.selection.record_id,
                "sample_id": source.selection.sample_id,
                "selection_rank": source.selection.selection_rank,
                "source_axis_float32_sha256": source.source_axis_float32_sha256,
                "source_intensity_float32_sha256": source.source_intensity_float32_sha256,
            }
            for source in sources
        )
        self.assertEqual(payload, canonical_jsonl_bytes(expected_rows))
        self.assertEqual(len(payload.splitlines()), 2)

    def test_write_dataset_rejects_nonmonotone_axis_before_task2_load(self):
        base = list(phase1_fixture_records())
        axis = increasing_axis_float32()
        axis[50] = axis[49]
        bad = replace(base[0], wavenumber=axis)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                DatasetValidationError,
                re.escape("wavenumber duplicate"),
            ):
                write_custom_phase1_fixture_dataset(
                    Path(tmp),
                    records=[bad, *base[1:]],
                )

    def test_load_phase1_source_does_not_mutate_underlying_dataset_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = write_phase1_fixture_dataset(Path(tmp))
            with UnifiedDataset.open(dataset_path, verify_checksums=True) as dataset:
                record_before = dataset.get("decreasing-record")
                axis_before = record_before.wavenumber.copy()
                intensity_before = record_before.intensity.copy()
                row = SelectedSourceRow(
                    selection_rank=0,
                    record_id="decreasing-record",
                    sample_id="sample-b",
                    class_label=0,
                    mineral_name="Mineral-A",
                    axis_id="axis-0",
                )
                load_phase1_source(dataset, row)
                record_after = dataset.get("decreasing-record")

        np.testing.assert_array_equal(record_after.wavenumber, axis_before)
        np.testing.assert_array_equal(record_after.intensity, intensity_before)


if __name__ == "__main__":
    unittest.main()

