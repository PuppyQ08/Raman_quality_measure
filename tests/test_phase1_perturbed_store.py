from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from phase1_runner_helpers import (  # noqa: E402
    assert_stored_shard_matches_payload,
    write_phase1_fixture_shard_payload,
)
from rpe.io import (  # noqa: E402
    SHARD_FILES,
    SHARD_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    PerturbedStoreError,
    ShardReceipt,
    StoredCell,
    StoredRecord,
    StoredShard,
    StoredSource,
    canonical_json_bytes,
    canonical_json_value,
    canonical_jsonl_bytes,
    float64_le_bytes,
    logical_array_digest,
    read_perturbed_shard,
    write_perturbed_shard,
)
from rpe.perturb.contracts import derive_perturbed_spectrum_id  # noqa: E402
from rpe.runner.phase1_types import CellStatus as RunnerCellStatus  # noqa: E402


RUN_ID = "phase1-task5-fixture"
SCIENTIFIC_CONFIG_SHA256 = "a" * 64
SWEEP_CONFIG_SHA256 = "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"
ALPHA_ORDER = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
COMPLETE_RECORD_ORDER = (
    "p01",
    "p02",
    "p03",
    "p04",
    "p05",
    "p08",
    "p09",
    "p10",
    "p11",
    "p12",
)
CELL_KEY_ORDER = (
    "actual_output_count",
    "dependency",
    "exception_message",
    "exception_path",
    "exception_type",
    "expected_output_count",
    "native_gate",
    "output_spectrum_ids",
    "perturbation_id",
    "reason_code",
    "run_id",
    "scientific_config_sha256",
    "source",
    "state_digest",
    "status",
    "sweep_config_sha256",
)
SOURCE_KEY_ORDER = (
    "axis_id",
    "axis_index",
    "class_label",
    "intensity_length",
    "intensity_offset",
    "mineral_name",
    "normalized_axis_float64_sha256",
    "normalized_intensity_float64_sha256",
    "original_axis_orientation",
    "provenance",
    "sample_id",
    "selection_axis_id",
    "selection_rank",
    "source_axis_float32_sha256",
    "source_dataset_id",
    "source_index",
    "source_intensity_float32_sha256",
    "source_record_id",
    "source_spectrum_id",
)
RECORD_KEY_ORDER = (
    "alpha",
    "alpha_float64_le_hex",
    "axis_behavior",
    "axis_changed",
    "axis_id",
    "axis_length",
    "class_label",
    "diagnostics",
    "intensity_changed",
    "intensity_length",
    "intensity_offset",
    "mineral_name",
    "output_spectrum_id",
    "perturbation_id",
    "run_id",
    "sample_id",
    "source_dataset_id",
    "source_index",
    "source_record_id",
    "source_selection_rank",
    "source_spectrum_id",
    "state_digest",
    "sweep_config_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n") or line == b"\n":
                raise AssertionError(f"noncanonical jsonl line at {index}")
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise AssertionError(f"row {index} is not a json object")
            rows.append(decoded)
    return tuple(rows)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise AssertionError(f"{path} is not a json object")
    return value


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(document))


def _write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.write_bytes(canonical_jsonl_bytes(rows))


def _axis_id(values: np.ndarray) -> str:
    return hashlib.sha256(
        b"rpe-phase1-float64-axis-v1\0"
        + struct.pack("<Q", values.size)
        + np.ascontiguousarray(values, dtype="<f8").tobytes()
    ).hexdigest()


def _source_ids_digest(source_ids: tuple[str, ...]) -> str:
    payload = b"rpe-phase1-shard-source-ids-v1\0" + struct.pack("<Q", len(source_ids))
    for source_id in source_ids:
        encoded = source_id.encode("utf-8")
        payload += struct.pack("<Q", len(encoded)) + encoded
    return hashlib.sha256(payload).hexdigest()


def _logical_content_digest(
    cells_bytes: bytes,
    records_bytes: bytes,
    *,
    axes: tuple[np.ndarray, ...],
    sources: tuple[np.ndarray, ...],
    records: tuple[np.ndarray, ...],
) -> str:
    payload = (
        b"rpe-phase1-shard-logical-content-v1\0"
        + struct.pack("<Q", len(cells_bytes))
        + cells_bytes
        + struct.pack("<Q", len(records_bytes))
        + records_bytes
        + bytes.fromhex(logical_array_digest(axes, sources, records))
    )
    return hashlib.sha256(payload).hexdigest()


def _artifact_json(path: Path) -> dict[str, object]:
    return {
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _extract_logical_arrays(hdf5_path: Path) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    with h5py.File(hdf5_path, "r") as handle:
        axis_offsets = tuple(int(value) for value in handle["axes"]["offsets"][...])
        axis_values = np.asarray(handle["axes"]["values"][...], dtype="<f8")
        axes = tuple(
            np.asarray(axis_values[start:end], dtype="<f8")
            for start, end in zip(axis_offsets[:-1], axis_offsets[1:], strict=True)
        )
        source_offsets = tuple(int(value) for value in handle["sources"]["intensity_offsets"][...])
        source_values = np.asarray(handle["sources"]["intensity_values"][...], dtype="<f8")
        sources = tuple(
            np.asarray(source_values[start:end], dtype="<f8")
            for start, end in zip(source_offsets[:-1], source_offsets[1:], strict=True)
        )
        record_offsets = tuple(int(value) for value in handle["records"]["intensity_offsets"][...])
        record_values = np.asarray(handle["records"]["intensity_values"][...], dtype="<f8")
        records = tuple(
            np.asarray(record_values[start:end], dtype="<f8")
            for start, end in zip(record_offsets[:-1], record_offsets[1:], strict=True)
        )
    return axes, sources, records


def _rebind_receipt(
    shard_dir: Path,
    *,
    rebind_files: bool = True,
    rebind_source_ids: bool = True,
    rebind_logical: bool = True,
) -> None:
    receipt = _read_json(shard_dir / "receipt.json")
    if rebind_files:
        receipt["files"] = {
            name: _artifact_json(shard_dir / name)
            for name in ("arrays.h5", "cells.jsonl", "records.jsonl")
        }
    if rebind_source_ids or rebind_logical:
        cell_rows = _load_jsonl(shard_dir / "cells.jsonl")
    if rebind_source_ids:
        source_ids = tuple(
            row["source"]["source_spectrum_id"]
            for index, row in enumerate(cell_rows)
            if index % 12 == 0
        )
        receipt["source_ids_sha256"] = _source_ids_digest(source_ids)
    if rebind_logical:
        axes, sources, records = _extract_logical_arrays(shard_dir / "arrays.h5")
        receipt["logical_content_sha256"] = _logical_content_digest(
            (shard_dir / "cells.jsonl").read_bytes(),
            (shard_dir / "records.jsonl").read_bytes(),
            axes=axes,
            sources=sources,
            records=records,
        )
    _write_json(shard_dir / "receipt.json", receipt)


def _core_failed_payload(payload):
    failed_cells = []
    for cell in payload.cells:
        if cell.perturbation_id in {"p06", "p07"}:
            failed_cells.append(cell)
            continue
        failed_cells.append(
            replace(
                cell,
                records=(),
                evidence=replace(
                    cell.evidence,
                    status=RunnerCellStatus.FAILED,
                    reason_code=None,
                    exception_type="InjectedFailure",
                    exception_path=f"fixture.{cell.perturbation_id}",
                    exception_message="injected failed cell",
                    native_gate={},
                ),
            )
        )
    return replace(payload, cells=tuple(failed_cells))


def _payload_with_core_not_applicable(payload, perturbation_id: str, *, reason_code: str) -> object:
    updated = []
    for cell in payload.cells:
        if cell.perturbation_id != perturbation_id:
            updated.append(cell)
            continue
        updated.append(
            replace(
                cell,
                records=(),
                evidence=replace(
                    cell.evidence,
                    status=RunnerCellStatus.NOT_APPLICABLE,
                    reason_code=reason_code,
                    exception_type=None,
                    exception_path=None,
                    exception_message=None,
                    native_gate={},
                ),
            )
        )
    return replace(payload, cells=tuple(updated))


def _payload_with_core_failed(payload, perturbation_id: str) -> object:
    updated = []
    for cell in payload.cells:
        if cell.perturbation_id != perturbation_id:
            updated.append(cell)
            continue
        updated.append(
            replace(
                cell,
                records=(),
                evidence=replace(
                    cell.evidence,
                    status=RunnerCellStatus.FAILED,
                    reason_code=None,
                    exception_type="InjectedFailure",
                    exception_path=f"fixture.{perturbation_id}",
                    exception_message="injected failed cell",
                    native_gate={},
                ),
            )
        )
    return replace(payload, cells=tuple(updated))


class PerturbedStoreTest(unittest.TestCase):
    def _build_fixture(self, *, source_count: int = 2):
        tempdir = tempfile.TemporaryDirectory()
        root = Path(tempdir.name)
        payload = write_phase1_fixture_shard_payload(root, source_count=source_count)
        shard_dir = root / "shards" / f"{payload.shard_index:05d}"
        shard_dir.parent.mkdir(parents=True, exist_ok=True)
        return tempdir, payload, shard_dir

    def test_public_exports_and_dataclass_fields_are_exact(self) -> None:
        self.assertEqual(STORE_SCHEMA_VERSION, "phase1-perturbed-store-v1")
        self.assertEqual(SHARD_SCHEMA_VERSION, "phase1-perturbed-shard-v1")
        self.assertEqual(
            SHARD_FILES,
            frozenset({"arrays.h5", "records.jsonl", "cells.jsonl", "receipt.json"}),
        )
        self.assertTrue(issubclass(PerturbedStoreError, ValueError))
        self.assertTrue(is_dataclass(StoredSource))
        self.assertTrue(is_dataclass(StoredRecord))
        self.assertTrue(is_dataclass(StoredCell))
        self.assertTrue(is_dataclass(ShardReceipt))
        self.assertTrue(is_dataclass(StoredShard))
        self.assertEqual(
            tuple(field.name for field in fields(StoredSource)),
            (
                "source_spectrum_id",
                "source_record_id",
                "sample_id",
                "class_label",
                "mineral_name",
                "axis_cm1",
                "intensity",
                "provenance",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(StoredRecord)),
            (
                "source_spectrum_id",
                "perturbation_id",
                "output_spectrum_id",
                "alpha",
                "alpha_float64_le_hex",
                "state_digest",
                "sweep_config_sha256",
                "axis_behavior",
                "axis_changed",
                "intensity_changed",
                "axis_cm1",
                "intensity",
                "diagnostics",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(StoredCell)),
            (
                "source_spectrum_id",
                "perturbation_id",
                "status",
                "reason_code",
                "state_digest",
                "records",
                "native_gate",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(ShardReceipt)),
            (
                "schema_version",
                "run_id",
                "shard_index",
                "source_count",
                "cell_count",
                "record_count",
                "source_ids_sha256",
                "logical_content_sha256",
                "files",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(StoredShard)),
            ("run_id", "shard_index", "sources", "cells", "receipt"),
        )
        self.assertTrue(callable(write_perturbed_shard))
        self.assertTrue(callable(read_perturbed_shard))
        self.assertTrue(callable(canonical_json_value))
        self.assertTrue(callable(canonical_json_bytes))
        self.assertTrue(callable(canonical_jsonl_bytes))
        self.assertTrue(callable(float64_le_bytes))
        self.assertTrue(callable(logical_array_digest))

    def test_canonical_serializers_are_strict_and_do_not_mutate_caller(self) -> None:
        original = {
            "tuple": (1, "x", {"nested": [True, None, 1.5]}),
            "unicode": "矿物",
        }
        encoded = canonical_json_bytes(original)
        self.assertEqual(
            encoded,
            b'{"tuple":[1,"x",{"nested":[true,null,1.5]}],"unicode":"\xe7\x9f\xbf\xe7\x89\xa9"}\n',
        )
        self.assertEqual(
            canonical_json_value(original),
            {"tuple": [1, "x", {"nested": [True, None, 1.5]}], "unicode": "矿物"},
        )
        self.assertEqual(
            original,
            {"tuple": (1, "x", {"nested": [True, None, 1.5]}), "unicode": "矿物"},
        )
        rows = ({"b": 2, "a": 1}, {"z": "末尾"})
        self.assertEqual(
            canonical_jsonl_bytes(rows),
            b'{"a":1,"b":2}\n{"z":"\xe6\x9c\xab\xe5\xb0\xbe"}\n',
        )
        self.assertEqual(canonical_jsonl_bytes(()), b"")
        with self.assertRaisesRegex(PerturbedStoreError, "canonical JSON"):
            canonical_json_bytes({"bad": np.float64(1.0)})
        with self.assertRaisesRegex(PerturbedStoreError, "canonical JSON"):
            canonical_json_bytes({"": 1})
        with self.assertRaisesRegex(PerturbedStoreError, "canonical JSON"):
            canonical_json_bytes({"bad": float("nan")})
        with self.assertRaisesRegex(PerturbedStoreError, "canonical JSONL"):
            canonical_jsonl_bytes((["not-a-mapping"],))

    def test_float64_bytes_and_logical_digest_use_exact_domains(self) -> None:
        first = np.array([1.0, 2.0, 3.5], dtype="<f8")
        second = np.array([8.0, 13.0], dtype="<f8")
        self.assertEqual(float64_le_bytes(first), first.tobytes())
        payload = bytearray()
        payload.extend(b"rpe-phase1-logical-arrays-v1\0")
        for collection in ((first, second), (first,), ()):
            payload.extend(struct.pack("<Q", len(collection)))
            for values in collection:
                payload.extend(struct.pack("<Q", values.size))
                payload.extend(struct.pack("<Q", values.nbytes))
                payload.extend(values.tobytes())
        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(logical_array_digest((first, second), (first,), ()), expected)
        with self.assertRaisesRegex(PerturbedStoreError, "float64"):
            float64_le_bytes(np.array([], dtype="<f8"))
        with self.assertRaisesRegex(PerturbedStoreError, "float64"):
            float64_le_bytes(np.array([1.0], dtype=">f8"))
        with self.assertRaisesRegex(PerturbedStoreError, "float64"):
            float64_le_bytes(np.array([[1.0]], dtype="<f8"))
        with self.assertRaisesRegex(PerturbedStoreError, "float64"):
            float64_le_bytes(np.array([np.inf], dtype="<f8"))

    def test_two_source_round_trip_writes_exact_files_counts_and_public_values(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=2)
        self.addCleanup(tempdir.cleanup)
        receipt = write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        self.assertEqual({path.name for path in shard_dir.iterdir()}, SHARD_FILES)
        self.assertEqual(receipt.schema_version, SHARD_SCHEMA_VERSION)
        self.assertEqual(receipt.run_id, RUN_ID)
        self.assertEqual(receipt.shard_index, 0)
        self.assertEqual(receipt.source_count, 2)
        self.assertEqual(receipt.cell_count, 24)
        self.assertEqual(receipt.record_count, 180)
        self.assertEqual(tuple(receipt.files.keys()), ("arrays.h5", "cells.jsonl", "records.jsonl"))
        stored = read_perturbed_shard(shard_dir)
        self.assertEqual(stored.run_id, RUN_ID)
        self.assertEqual(stored.shard_index, payload.shard_index)
        self.assertEqual(stored.source_count, 2)
        self.assertEqual(stored.cell_count, 24)
        self.assertEqual(stored.record_count, 180)
        assert_stored_shard_matches_payload(stored, payload)

    def test_canonical_rows_have_exact_keys_and_no_absolute_paths(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=2)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        cell_rows = _load_jsonl(shard_dir / "cells.jsonl")
        record_rows = _load_jsonl(shard_dir / "records.jsonl")
        self.assertEqual(len(cell_rows), 24)
        self.assertEqual(len(record_rows), 180)
        self.assertEqual(tuple(cell_rows[0].keys()), CELL_KEY_ORDER)
        self.assertEqual(tuple(cell_rows[0]["source"].keys()), SOURCE_KEY_ORDER)
        self.assertEqual(tuple(record_rows[0].keys()), RECORD_KEY_ORDER)
        for row in cell_rows:
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
            self.assertNotIn(str(ROOT), serialized)
            self.assertNotIn("/data03/", serialized)
        for row in record_rows:
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True)
            self.assertNotIn(str(ROOT), serialized)
            self.assertNotIn("/data03/", serialized)

    def test_hdf5_layout_offsets_and_deduplicated_axis_values_match_rows(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=2)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        cell_rows = _load_jsonl(shard_dir / "cells.jsonl")
        record_rows = _load_jsonl(shard_dir / "records.jsonl")
        with h5py.File(shard_dir / "arrays.h5", "r") as handle:
            self.assertEqual(set(handle.keys()), {"axes", "sources", "records"})
            self.assertEqual(set(handle.attrs.keys()), {"schema_version", "run_id", "shard_index"})
            self.assertEqual(handle.attrs["schema_version"], SHARD_SCHEMA_VERSION)
            self.assertEqual(handle.attrs["run_id"], RUN_ID)
            self.assertEqual(int(handle.attrs["shard_index"]), payload.shard_index)
            axes = handle["axes"]
            sources = handle["sources"]
            records = handle["records"]
            self.assertEqual(set(axes.keys()), {"ids", "offsets", "values"})
            self.assertEqual(set(sources.keys()), {"axis_index", "intensity_offsets", "intensity_values"})
            self.assertEqual(set(records.keys()), {"intensity_offsets", "intensity_values"})
            for group in (axes, sources, records):
                self.assertEqual(len(group.attrs), 0)
            dataset_names = (
                ("axes", "ids"),
                ("axes", "offsets"),
                ("axes", "values"),
                ("sources", "axis_index"),
                ("sources", "intensity_offsets"),
                ("sources", "intensity_values"),
                ("records", "intensity_offsets"),
                ("records", "intensity_values"),
            )
            for group_name, dataset_name in dataset_names:
                dataset = handle[group_name][dataset_name]
                self.assertEqual(dataset.ndim, 1)
                self.assertEqual(len(dataset.attrs), 0)
                self.assertEqual(dataset.compression, "gzip")
                self.assertEqual(dataset.compression_opts, 1)
                self.assertFalse(dataset.shuffle)
                self.assertFalse(dataset.fletcher32)
                self.assertIsNotNone(dataset.chunks)
            self.assertEqual(axes["ids"].dtype, np.dtype("S64"))
            self.assertEqual(axes["offsets"].dtype, np.dtype("<u8"))
            self.assertEqual(axes["values"].dtype, np.dtype("<f8"))
            self.assertEqual(sources["axis_index"].dtype, np.dtype("<u4"))
            self.assertEqual(sources["intensity_offsets"].dtype, np.dtype("<u8"))
            self.assertEqual(sources["intensity_values"].dtype, np.dtype("<f8"))
            self.assertEqual(records["intensity_offsets"].dtype, np.dtype("<u8"))
            self.assertEqual(records["intensity_values"].dtype, np.dtype("<f8"))
            axis_ids = tuple(item.decode("ascii") for item in axes["ids"][...])
            self.assertEqual(axis_ids, tuple(sorted(axis_ids)))
            axis_offsets = tuple(int(value) for value in axes["offsets"][...])
            source_offsets = tuple(int(value) for value in sources["intensity_offsets"][...])
            record_offsets = tuple(int(value) for value in records["intensity_offsets"][...])
            self.assertEqual(axis_offsets[0], 0)
            self.assertEqual(source_offsets[0], 0)
            self.assertEqual(record_offsets[0], 0)
            self.assertEqual(axis_offsets[-1], axes["values"].shape[0])
            self.assertEqual(source_offsets[-1], sources["intensity_values"].shape[0])
            self.assertEqual(record_offsets[-1], records["intensity_values"].shape[0])
            self.assertEqual(len(source_offsets), len(payload.sources) + 1)
            self.assertEqual(len(record_offsets), len(record_rows) + 1)
            self.assertEqual(sources["axis_index"].shape[0], len(payload.sources))
            source_axis_ids = tuple(_axis_id(source.spectrum.axis_cm1) for source in payload.sources)
            stored_axis_ids = []
            for start, end in zip(axis_offsets[:-1], axis_offsets[1:], strict=True):
                stored_axis_ids.append(_axis_id(np.asarray(axes["values"][start:end], dtype="<f8")))
            self.assertEqual(tuple(stored_axis_ids), axis_ids)
            self.assertEqual(
                tuple(axis_ids[index] for index in sources["axis_index"][...]),
                source_axis_ids,
            )
            for source_index, source in enumerate(payload.sources):
                start = source_offsets[source_index]
                end = source_offsets[source_index + 1]
                self.assertTrue(
                    np.array_equal(
                        np.asarray(sources["intensity_values"][start:end], dtype="<f8"),
                        source.spectrum.intensity,
                    )
                )
                self.assertEqual(
                    int(cell_rows[source_index * 12]["source"]["intensity_offset"]),
                    start,
                )
                self.assertEqual(
                    int(cell_rows[source_index * 12]["source"]["intensity_length"]),
                    source.spectrum.intensity.size,
                )
            for row_index, row in enumerate(record_rows):
                start = record_offsets[row_index]
                end = record_offsets[row_index + 1]
                self.assertEqual(end - start, int(row["intensity_length"]))
                self.assertIn(row["axis_id"], axis_ids)

    def test_independent_digests_and_file_hashes_match_literal_formulas(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=2)
        self.addCleanup(tempdir.cleanup)
        receipt = write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        cells_bytes = (shard_dir / "cells.jsonl").read_bytes()
        records_bytes = (shard_dir / "records.jsonl").read_bytes()
        stored = read_perturbed_shard(shard_dir)
        axis_by_id = {
            _axis_id(source.axis_cm1): source.axis_cm1
            for source in stored.sources
        }
        for cell in stored.cells:
            for record in cell.records:
                axis_by_id.setdefault(_axis_id(record.axis_cm1), record.axis_cm1)
        sorted_axes = tuple(axis_by_id[axis_id] for axis_id in sorted(axis_by_id))
        expected_source_ids_digest = _source_ids_digest(
            tuple(source.source_spectrum_id for source in stored.sources)
        )
        self.assertEqual(receipt.source_ids_sha256, expected_source_ids_digest)
        self.assertEqual(
            receipt.logical_content_sha256,
            _logical_content_digest(
                cells_bytes,
                records_bytes,
                axes=sorted_axes,
                sources=tuple(source.intensity for source in stored.sources),
                records=tuple(
                    record.intensity
                    for cell in stored.cells
                    for record in cell.records
                ),
            ),
        )
        self.assertEqual(receipt.files["arrays.h5"].sha256, _sha256_file(shard_dir / "arrays.h5"))
        self.assertEqual(receipt.files["cells.jsonl"].sha256, _sha256_file(shard_dir / "cells.jsonl"))
        self.assertEqual(receipt.files["records.jsonl"].sha256, _sha256_file(shard_dir / "records.jsonl"))

    def test_same_payload_written_under_two_parents_has_same_canonical_text_and_logical_digests(self) -> None:
        with tempfile.TemporaryDirectory() as left_tmp, tempfile.TemporaryDirectory() as right_tmp:
            left_payload = write_phase1_fixture_shard_payload(Path(left_tmp), source_count=2)
            right_payload = write_phase1_fixture_shard_payload(Path(right_tmp), source_count=2)
            left_shard = Path(left_tmp) / "shards" / "00000"
            right_shard = Path(right_tmp) / "shards" / "00000"
            left_shard.parent.mkdir(parents=True, exist_ok=True)
            right_shard.parent.mkdir(parents=True, exist_ok=True)
            left_receipt = write_perturbed_shard(
                left_payload,
                left_shard,
                run_id=RUN_ID,
                scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                sweep_config_sha256=SWEEP_CONFIG_SHA256,
            )
            right_receipt = write_perturbed_shard(
                right_payload,
                right_shard,
                run_id=RUN_ID,
                scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                sweep_config_sha256=SWEEP_CONFIG_SHA256,
            )
            self.assertEqual(
                (left_shard / "cells.jsonl").read_bytes(),
                (right_shard / "cells.jsonl").read_bytes(),
            )
            self.assertEqual(
                (left_shard / "records.jsonl").read_bytes(),
                (right_shard / "records.jsonl").read_bytes(),
            )
            self.assertEqual(left_receipt.source_ids_sha256, right_receipt.source_ids_sha256)
            self.assertEqual(left_receipt.logical_content_sha256, right_receipt.logical_content_sha256)

    def test_writer_rejects_existing_output_and_bad_basename_before_commit(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=2)
        self.addCleanup(tempdir.cleanup)
        bad_dir = shard_dir.parent / "bad"
        with self.assertRaises(PerturbedStoreError):
            write_perturbed_shard(
                payload,
                bad_dir,
                run_id=RUN_ID,
                scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                sweep_config_sha256=SWEEP_CONFIG_SHA256,
            )
        shard_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(PerturbedStoreError):
            write_perturbed_shard(
                payload,
                shard_dir,
                run_id=RUN_ID,
                scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                sweep_config_sha256=SWEEP_CONFIG_SHA256,
            )

    def test_reader_rejects_extra_entry_and_noncanonical_json_drift(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=2)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        extra = shard_dir / "extra.txt"
        extra.write_text("x", encoding="utf-8")
        with self.assertRaises(PerturbedStoreError):
            read_perturbed_shard(shard_dir)
        extra.unlink()
        with (shard_dir / "cells.jsonl").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaises(PerturbedStoreError):
            read_perturbed_shard(shard_dir)

    def test_partial_not_applicable_core_cell_round_trips_without_forcing_record_inventory(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        mutated = _payload_with_core_not_applicable(
            payload,
            "p01",
            reason_code="fixture_not_applicable",
        )
        receipt = write_perturbed_shard(
            mutated,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        self.assertEqual(receipt.record_count, 81)
        stored = read_perturbed_shard(shard_dir)
        self.assertEqual(stored.record_count, 81)
        first = stored.cells[0]
        self.assertEqual(first.perturbation_id, "p01")
        self.assertEqual(first.status, "not_applicable")
        self.assertEqual(first.reason_code, "fixture_not_applicable")
        self.assertEqual(first.records, ())

    def test_mixed_failed_core_cell_round_trips_without_forcing_complete_rows(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        mutated = _payload_with_core_failed(payload, "p05")
        receipt = write_perturbed_shard(
            mutated,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        self.assertEqual(receipt.record_count, 81)
        stored = read_perturbed_shard(shard_dir)
        failed = {cell.perturbation_id: cell for cell in stored.cells}["p05"]
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.records, ())

    def test_zero_record_shard_round_trips_and_reconstructs_sources_from_cells(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        mutated = _core_failed_payload(payload)
        receipt = write_perturbed_shard(
            mutated,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        self.assertEqual(receipt.record_count, 0)
        with h5py.File(shard_dir / "arrays.h5", "r") as handle:
            record_values = handle["records"]["intensity_values"]
            self.assertEqual(record_values.shape, (0,))
            self.assertIsNone(record_values.compression)
            self.assertIsNone(record_values.chunks)
        stored = read_perturbed_shard(shard_dir)
        self.assertEqual(stored.source_count, 1)
        self.assertEqual(stored.record_count, 0)
        self.assertTrue(all(cell.records == () for cell in stored.cells))
        self.assertTrue(all(source.intensity.size > 0 for source in stored.sources))

    def test_stored_record_recomputes_output_id(self) -> None:
        tempdir, payload, _ = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        record = read_perturbed_shard(
            self._write_fixture_and_read(payload)
        ).cells[0].records[0]
        with self.assertRaisesRegex(PerturbedStoreError, "output_spectrum_id"):
            replace(record, output_spectrum_id="wrong")

    def test_stored_cell_rejects_negative_zero_alpha_and_duplicate_output_ids(self) -> None:
        tempdir, payload, _ = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        stored = read_perturbed_shard(self._write_fixture_and_read(payload))
        record = stored.cells[0].records[0]
        neg_zero = replace(
            record,
            alpha=-0.0,
            alpha_float64_le_hex=struct.pack("<d", -0.0).hex(),
            output_spectrum_id=derive_perturbed_spectrum_id(
                record.source_spectrum_id,
                record.perturbation_id,
                -0.0,
                record.state_digest,
                record.sweep_config_sha256,
            ),
        )
        with self.assertRaisesRegex(PerturbedStoreError, "records"):
            StoredCell(
                source_spectrum_id=record.source_spectrum_id,
                perturbation_id=record.perturbation_id,
                status="complete",
                reason_code=None,
                state_digest=record.state_digest,
                records=(neg_zero,) + stored.cells[0].records[1:],
                native_gate={"ok": True},
            )
        duplicate = object.__new__(StoredRecord)
        for field in fields(StoredRecord):
            value = getattr(record, field.name)
            if field.name == "output_spectrum_id":
                value = stored.cells[0].records[1].output_spectrum_id
            object.__setattr__(duplicate, field.name, value)
        with self.assertRaisesRegex(PerturbedStoreError, "output_spectrum_id"):
            StoredCell(
                source_spectrum_id=record.source_spectrum_id,
                perturbation_id=record.perturbation_id,
                status="complete",
                reason_code=None,
                state_digest=record.state_digest,
                records=(duplicate,) + stored.cells[0].records[1:],
                native_gate={"ok": True},
            )

    def test_receipt_and_inventory_errors_map_to_stable_paths_before_receipt_read(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        (shard_dir / "receipt.json").unlink()
        with self.assertRaisesRegex(PerturbedStoreError, r"^path:"):
            read_perturbed_shard(shard_dir)

    def test_receipt_symlink_and_nonregular_entry_are_rejected_lexically(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        target = shard_dir / "receipt-target.json"
        target.write_bytes((shard_dir / "receipt.json").read_bytes())
        (shard_dir / "receipt.json").unlink()
        os.symlink(target, shard_dir / "receipt.json")
        with self.assertRaisesRegex(PerturbedStoreError, r"^receipt\.json:"):
            read_perturbed_shard(shard_dir)

    def test_malformed_receipt_nested_file_entry_maps_to_receipt_path(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        receipt = _read_json(shard_dir / "receipt.json")
        del receipt["files"]["arrays.h5"]["sha256"]
        _write_json(shard_dir / "receipt.json", receipt)
        with self.assertRaisesRegex(PerturbedStoreError, r"^receipt\.json\.files\.arrays\.h5\.sha256:"):
            read_perturbed_shard(shard_dir)

    def test_invalid_hdf_maps_to_arrays_path(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        (shard_dir / "arrays.h5").write_bytes(b"not-hdf")
        _rebind_receipt(shard_dir, rebind_files=True, rebind_source_ids=False, rebind_logical=False)
        with self.assertRaisesRegex(PerturbedStoreError, r"^arrays\.h5:"):
            read_perturbed_shard(shard_dir)

    def test_external_linked_arrays_dataset_is_rejected(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        linked_file = shard_dir.parent / "linked-axes-ids.h5"
        with h5py.File(shard_dir / "arrays.h5", "r") as source_handle:
            source_dataset = source_handle["axes"]["ids"]
            axis_ids = np.asarray(source_dataset[...], dtype="S64")
            source_chunks = source_dataset.chunks
            source_compression = source_dataset.compression
            source_compression_opts = source_dataset.compression_opts
            source_shuffle = bool(source_dataset.shuffle)
            source_fletcher32 = bool(source_dataset.fletcher32)
        with h5py.File(linked_file, "w") as linked_handle:
            linked_handle.create_dataset(
                "ids",
                data=axis_ids,
                chunks=source_chunks,
                compression=source_compression,
                compression_opts=source_compression_opts,
                shuffle=source_shuffle,
                fletcher32=source_fletcher32,
                track_times=False,
            )
        with h5py.File(shard_dir / "arrays.h5", "r+") as handle:
            del handle["axes"]["ids"]
            handle["axes"]["ids"] = h5py.ExternalLink(f"../{linked_file.name}", "/ids")
        _rebind_receipt(shard_dir, rebind_files=True, rebind_source_ids=False, rebind_logical=False)
        with self.assertRaisesRegex(PerturbedStoreError, r"^arrays\.h5(\.|:)"):
            read_perturbed_shard(shard_dir)

    def test_source_offset_axis_index_and_root_attr_mutations_are_rejected(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=2)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        cell_rows = list(_load_jsonl(shard_dir / "cells.jsonl"))
        for index in range(12):
            cell_rows[index]["source"]["intensity_offset"] = 999999
        _write_jsonl(shard_dir / "cells.jsonl", tuple(cell_rows))
        _rebind_receipt(shard_dir)
        with self.assertRaisesRegex(PerturbedStoreError, r"cells\.jsonl\.source\.intensity_offset|cells\.jsonl\[0\]\.source\.intensity_offset"):
            read_perturbed_shard(shard_dir)

        other_parent = shard_dir.parent / "axis-index-mutation"
        other_parent.mkdir(parents=True, exist_ok=True)
        write_perturbed_shard(
            payload,
            other_parent / "00000",
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        other = other_parent / "00000"
        cell_rows = list(_load_jsonl(other / "cells.jsonl"))
        with h5py.File(other / "arrays.h5", "r+") as handle:
            axis_indices = np.asarray(handle["sources"]["axis_index"][...], dtype="<u4")
            axis_indices[0] = np.uint32(1)
            del handle["sources"]["axis_index"]
            handle["sources"].create_dataset("axis_index", data=axis_indices, chunks=(len(axis_indices),), compression="gzip", compression_opts=1, shuffle=False, fletcher32=False, track_times=False)
        for index in range(12):
            cell_rows[index]["source"]["axis_index"] = 1
            cell_rows[index]["source"]["axis_id"] = cell_rows[12]["source"]["axis_id"]
        _write_jsonl(other / "cells.jsonl", tuple(cell_rows))
        _rebind_receipt(other)
        with self.assertRaisesRegex(PerturbedStoreError, r"cells\.jsonl\.source\.axis_id|cells\.jsonl\.source\.axis_index"):
            read_perturbed_shard(other)

        third_parent = shard_dir.parent / "root-attr-mutation"
        third_parent.mkdir(parents=True, exist_ok=True)
        write_perturbed_shard(
            payload,
            third_parent / "00000",
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        third = third_parent / "00000"
        with h5py.File(third / "arrays.h5", "r+") as handle:
            del handle.attrs["shard_index"]
            handle.attrs["shard_index"] = np.int64(0)
        _rebind_receipt(third)
        with self.assertRaisesRegex(PerturbedStoreError, r"arrays\.h5\.shard_index"):
            read_perturbed_shard(third)

    def test_cross_file_sweep_identity_mutation_is_rejected(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        cell_rows = list(_load_jsonl(shard_dir / "cells.jsonl"))
        cell_rows[0]["sweep_config_sha256"] = "c" * 64
        _write_jsonl(shard_dir / "cells.jsonl", tuple(cell_rows))
        _rebind_receipt(shard_dir, rebind_files=True, rebind_source_ids=True, rebind_logical=True)
        with self.assertRaisesRegex(PerturbedStoreError, r"sweep_config_sha256"):
            read_perturbed_shard(shard_dir)

    def test_writer_pre_rename_failures_leave_target_absent_and_cleanup_staging(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        parent = shard_dir.parent
        original_replace = os.replace

        for failure_name, target in (
            ("text_write", "rpe.io.perturbed_store._write_binary_file"),
            ("fsync", "rpe.io.perturbed_store._fsync_path"),
            ("rename", "os.replace"),
            ("readback", "rpe.io.perturbed_store.read_perturbed_shard"),
        ):
            failure_root = parent / failure_name
            if failure_root.exists():
                shutil.rmtree(failure_root)
            failure_root.mkdir(parents=True, exist_ok=True)
            target_dir = failure_root / f"{payload.shard_index:05d}"
            if target == "os.replace":
                with patch(target, side_effect=OSError("rename failed")):
                    with self.assertRaises(OSError):
                        write_perturbed_shard(
                            payload,
                            target_dir,
                            run_id=RUN_ID,
                            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                            sweep_config_sha256=SWEEP_CONFIG_SHA256,
                        )
            elif target == "rpe.io.perturbed_store.read_perturbed_shard":
                with patch(target, side_effect=PerturbedStoreError("readback", "forced")):
                    with self.assertRaises(PerturbedStoreError):
                        write_perturbed_shard(
                            payload,
                            target_dir,
                            run_id=RUN_ID,
                            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                            sweep_config_sha256=SWEEP_CONFIG_SHA256,
                        )
            elif target == "rpe.io.perturbed_store._write_binary_file":
                original = write_perturbed_shard
                call_counter = {"count": 0}
                def failing_write(path, data):
                    call_counter["count"] += 1
                    if call_counter["count"] == 2:
                        raise OSError("forced text write failure")
                    return __import__("rpe.io.perturbed_store", fromlist=["_write_binary_file"])._write_binary_file(path, data)
                with patch(target, side_effect=failing_write):
                    with self.assertRaises(OSError):
                        write_perturbed_shard(
                            payload,
                            target_dir,
                            run_id=RUN_ID,
                            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                            sweep_config_sha256=SWEEP_CONFIG_SHA256,
                        )
            else:
                original_fsync = __import__("rpe.io.perturbed_store", fromlist=["_fsync_path"])._fsync_path
                call_counter = {"count": 0}
                def failing_fsync(path):
                    call_counter["count"] += 1
                    if call_counter["count"] == 1:
                        raise OSError("forced fsync failure")
                    return original_fsync(path)
                with patch(target, side_effect=failing_fsync):
                    with self.assertRaises(OSError):
                        write_perturbed_shard(
                            payload,
                            target_dir,
                            run_id=RUN_ID,
                            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                            sweep_config_sha256=SWEEP_CONFIG_SHA256,
                        )
            self.assertFalse(target_dir.exists())
            leftovers = [child for child in failure_root.iterdir() if child.name.startswith(f".{target_dir.name}.")]
            self.assertEqual(leftovers, [])

    def test_live_source_mutation_during_write_is_detected_before_rename(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        source = payload.sources[0]
        original_write_hdf5 = __import__("rpe.io.perturbed_store", fromlist=["_write_hdf5"])._write_hdf5

        def mutating_write_hdf5(*args, **kwargs):
            source.spectrum.intensity.setflags(write=True)
            source.spectrum.intensity[0] = source.spectrum.intensity[0] + 1.0
            source.spectrum.intensity.setflags(write=False)
            return original_write_hdf5(*args, **kwargs)

        with patch("rpe.io.perturbed_store._write_hdf5", side_effect=mutating_write_hdf5):
            with self.assertRaisesRegex(PerturbedStoreError, r"caller array changed during write"):
                write_perturbed_shard(
                    payload,
                    shard_dir,
                    run_id=RUN_ID,
                    scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
                    sweep_config_sha256=SWEEP_CONFIG_SHA256,
                )
        self.assertFalse(shard_dir.exists())

    def test_reader_rejects_rebound_hash_corruptions_at_intended_layers(self) -> None:
        tempdir, payload, shard_dir = self._build_fixture(source_count=1)
        self.addCleanup(tempdir.cleanup)
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        cell_rows = list(_load_jsonl(shard_dir / "cells.jsonl"))
        cell_rows[0], cell_rows[1] = cell_rows[1], cell_rows[0]
        _write_jsonl(shard_dir / "cells.jsonl", tuple(cell_rows))
        _rebind_receipt(shard_dir)
        with self.assertRaises(PerturbedStoreError):
            read_perturbed_shard(shard_dir)

    def _write_fixture_and_read(self, payload) -> Path:
        inner = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, inner, ignore_errors=True)
        shard_dir = inner / "00000"
        write_perturbed_shard(
            payload,
            shard_dir,
            run_id=RUN_ID,
            scientific_config_sha256=SCIENTIFIC_CONFIG_SHA256,
            sweep_config_sha256=SWEEP_CONFIG_SHA256,
        )
        return shard_dir


if __name__ == "__main__":
    unittest.main()
