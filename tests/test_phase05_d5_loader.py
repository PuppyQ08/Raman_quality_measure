from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.rruff import (  # noqa: E402
    D5LoaderValidationError,
    load_d5_protocol_config,
    load_d5_raw_cohort,
)


CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d5_rruff_protocol.json"
)
DATASET = ROOT / "data" / "unified" / "rruff_raman_raw"
AUDIT = ROOT / "reports" / "phase05" / "d5_step01_protocol_audit.json"
COMPANIONS = (
    ROOT / "data" / "unified" / "rruff_raman_pairs.jsonl",
    ROOT / "data" / "unified" / "rruff_raman_conversion.json",
)
CONFIG_SHA256 = (
    "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009"
)
AUDIT_SHA256 = (
    "a673d0d0c4a0c6536a8a6d3a31514d209607f31599cd0456ba12956dab49ff70"
)
COHORT_RECORD_IDS_SHA256 = (
    "840896eee008cb5df60116ebc727431a649529c124d63a14fb0219f778f0e72d"
)
COHORT_GROUP_IDS_SHA256 = (
    "886613d52869b87c96e101fe1dc2b7bf74008ea9deb5ba9a6cfa237edc9644a3"
)
COHORT_CLASS_LABELS_SHA256 = (
    "289906f3282e0e7f7a82ea61de413bc4c474b6158383bea3aaad21aa956fb94c"
)
SPLIT_SHA256 = (
    "5d292560b28bf41c207c6fbe88d9e5025c4f03764240a41d6abf0da14138e5e5",
    "ee6703717075c7ab4094fcd68886866edfa1237e204f386a5622cb73538dd7c1",
    "91c2bc9c8fd669bb39f7ee29b79eecd2baa3573e37b9941e89b2beebb6593335",
    "01d736bcd75a872fbffbd6fac7cd2cd406bc90ad532227a24ed41fdb19dbb1cb",
    "f3d08a18e88e159b1097a0c2e823bb83d1e2aa0405f4bac3390de57618d97788",
)
QUERY_COUNTS = (1318, 1331, 1327, 1321, 1324)
LIBRARY_COUNTS = (2452, 2439, 2443, 2449, 2446)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ids_digest(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


def split_digest(
    seed: int,
    class_labels: np.ndarray,
    group_ids: tuple[str, ...],
    record_ids: tuple[str, ...],
    query_indices: np.ndarray,
    library_indices: np.ndarray,
) -> str:
    classes = {}
    for class_label in sorted(set(int(value) for value in class_labels)):
        query = [
            int(index)
            for index in query_indices
            if int(class_labels[int(index)]) == class_label
        ]
        library = [
            int(index)
            for index in library_indices
            if int(class_labels[int(index)]) == class_label
        ]
        query_groups = sorted({group_ids[index] for index in query})
        library_groups = sorted({group_ids[index] for index in library})
        if len(query_groups) != 1:
            raise AssertionError(
                f"class {class_label} has {len(query_groups)} query groups"
            )
        classes[str(class_label)] = {
            "query_group": query_groups[0],
            "query_record_count": len(query),
            "library_group_count": len(library_groups),
            "library_record_count": len(library),
        }
    payload = {"seed": seed, "classes": classes}
    return hashlib.sha256(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


class D5ProtocolConfigTest(unittest.TestCase):
    def test_loads_exact_frozen_protocol_and_rejects_modified_copy(self):
        config = load_d5_protocol_config(CONFIG)

        self.assertEqual(config.sha256, CONFIG_SHA256)
        self.assertEqual(config.byte_count, 3855)
        self.assertEqual(config.dataset_id, "rruff_raman_raw")
        self.assertEqual(config.seeds, (0, 1, 2, 3, 4))
        self.assertEqual(config.grid_start_cm1, 200.0)
        self.assertEqual(config.grid_stop_cm1, 1800.0)
        self.assertEqual(config.grid_step_cm1, 2.0)
        self.assertEqual(config.grid_point_count, 801)
        self.assertEqual(config.max_in_range_native_gap_cm1, 3.0)
        self.assertEqual(config.expected_class_count, 681)
        self.assertEqual(config.expected_record_count, 3770)
        self.assertEqual(config.expected_rruff_id_count, 1936)
        self.assertEqual(config.expected_group_count, 1934)
        self.assertEqual(config.audit_sha256, AUDIT_SHA256)
        self.assertEqual(config.split_sha256, SPLIT_SHA256)

        with tempfile.TemporaryDirectory() as temporary_directory:
            changed = json.loads(CONFIG.read_text())
            changed["alignment"]["step_cm1"] = 1.0
            path = Path(temporary_directory) / CONFIG.name
            path.write_text(
                json.dumps(
                    changed,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            with self.assertRaisesRegex(
                D5LoaderValidationError,
                "config identity",
            ):
                load_d5_protocol_config(path)


class D5RetainedLoaderTest(unittest.TestCase):
    def test_retained_loader_peak_rss_stays_below_shared_two_gib_gate(self):
        script = """
import json
import resource
from pathlib import Path
from rpe.downstream.rruff import load_d5_raw_cohort

root = Path.cwd()
cohort = load_d5_raw_cohort(
    root / "experiments/phase05/configs/d5_rruff_protocol.json",
    root / "data/unified/rruff_raman_raw",
)
print(json.dumps({
    "records": len(cohort.record_ids),
    "peak_rss_bytes": (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    ),
}, sort_keys=True, separators=(",", ":")))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["records"], 3770)
        self.assertLess(result["peak_rss_bytes"], 2_147_483_648)

    @classmethod
    def setUpClass(cls):
        cls.before = {
            path.as_posix(): (path.stat().st_size, sha256_file(path))
            for path in (*sorted(DATASET.iterdir()), *COMPANIONS)
            if path.is_file()
        }
        cls.cohort = load_d5_raw_cohort(CONFIG, DATASET)
        cls.after = {
            path.as_posix(): (path.stat().st_size, sha256_file(path))
            for path in (*sorted(DATASET.iterdir()), *COMPANIONS)
            if path.is_file()
        }
        cls.audit = json.loads(AUDIT.read_text())
        cls.source_by_record_id = {}
        cohort_record_ids = set(cls.cohort.record_ids)
        with (DATASET / "records.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["record_id"] in cohort_record_ids:
                    cls.source_by_record_id[row["record_id"]] = row

    def test_reconstructs_exact_read_only_cohort_without_source_mutation(self):
        cohort = self.cohort

        self.assertEqual(self.after, self.before)
        self.assertEqual(cohort.protocol_config_sha256, CONFIG_SHA256)
        self.assertEqual(cohort.dataset_id, "rruff_raman_raw")
        self.assertEqual(cohort.intensity.shape, (3770, 801))
        self.assertEqual(cohort.intensity.dtype, np.dtype("<f4"))
        self.assertEqual(cohort.wavenumber.shape, (801,))
        self.assertEqual(cohort.wavenumber.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(
            cohort.wavenumber,
            np.arange(200.0, 1800.0 + 1.0, 2.0, dtype="<f4"),
        )
        self.assertTrue(np.isfinite(cohort.intensity).all())
        self.assertFalse(cohort.intensity.flags.writeable)
        self.assertFalse(cohort.wavenumber.flags.writeable)
        self.assertFalse(cohort.class_labels.flags.writeable)
        with self.assertRaises(ValueError):
            cohort.intensity[0, 0] = -1.0
        with self.assertRaises(ValueError):
            cohort.class_labels[0] = -1

        self.assertEqual(len(cohort.record_ids), 3770)
        self.assertEqual(len(set(cohort.record_ids)), 3770)
        self.assertEqual(len(set(cohort.rruff_ids)), 1936)
        self.assertEqual(len(set(cohort.group_ids)), 1934)
        self.assertEqual(len(set(int(value) for value in cohort.class_labels)), 681)
        self.assertEqual(ids_digest(cohort.record_ids), COHORT_RECORD_IDS_SHA256)
        self.assertEqual(ids_digest(tuple(set(cohort.group_ids))), COHORT_GROUP_IDS_SHA256)
        self.assertEqual(
            hashlib.sha256(
                (
                    "\n".join(
                        str(value)
                        for value in sorted(
                            set(int(item) for item in cohort.class_labels)
                        )
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest(),
            COHORT_CLASS_LABELS_SHA256,
        )

    def test_preserves_record_mineral_rruff_pin_and_class_identities(self):
        cohort = self.cohort

        self.assertEqual(len(self.source_by_record_id), len(cohort.record_ids))
        for index, record_id in enumerate(cohort.record_ids):
            source = self.source_by_record_id[record_id]
            metadata = source["meta"]["source_metadata"]
            self.assertEqual(source["meta"]["dataset_id"], "rruff_raman_raw")
            self.assertEqual(source["meta"]["preprocessing_status"], "known_raw")
            self.assertEqual(source["meta"]["preprocessing_steps"], [])
            self.assertEqual(source["meta"]["sample_id"], cohort.rruff_ids[index])
            self.assertEqual(metadata["rruff_id"], cohort.rruff_ids[index])
            self.assertEqual(metadata["pin_id"], cohort.pin_ids[index])
            self.assertEqual(metadata["mineral_name"], cohort.mineral_names[index])
            self.assertEqual(
                source["targets"]["class_label"],
                int(cohort.class_labels[index]),
            )

    def test_five_splits_match_frozen_digests_and_have_no_group_leakage(self):
        cohort = self.cohort
        all_indices = set(range(len(cohort.record_ids)))

        self.assertEqual(len(cohort.splits), 5)
        for seed, split in enumerate(cohort.splits):
            self.assertEqual(split.seed, seed)
            self.assertFalse(split.query_indices.flags.writeable)
            self.assertFalse(split.library_indices.flags.writeable)
            query = set(int(index) for index in split.query_indices)
            library = set(int(index) for index in split.library_indices)
            self.assertFalse(query & library)
            self.assertEqual(query | library, all_indices)
            self.assertEqual(len(query), QUERY_COUNTS[seed])
            self.assertEqual(len(library), LIBRARY_COUNTS[seed])
            query_groups = {cohort.group_ids[index] for index in query}
            library_groups = {cohort.group_ids[index] for index in library}
            self.assertFalse(query_groups & library_groups)
            self.assertEqual(len(query_groups), 681)
            self.assertEqual(len(library_groups), 1253)
            self.assertEqual(split.split_sha256, SPLIT_SHA256[seed])
            self.assertEqual(
                split_digest(
                    seed,
                    cohort.class_labels,
                    cohort.group_ids,
                    cohort.record_ids,
                    split.query_indices,
                    split.library_indices,
                ),
                SPLIT_SHA256[seed],
            )
            expected = self.audit["split_replicates"][seed]
            self.assertEqual(
                ids_digest([cohort.record_ids[index] for index in query]),
                expected["query_record_ids_sha256"],
            )
            self.assertEqual(
                ids_digest([cohort.record_ids[index] for index in library]),
                expected["library_record_ids_sha256"],
            )
            self.assertEqual(
                ids_digest(tuple(query_groups)),
                expected["query_group_ids_sha256"],
            )
            self.assertEqual(
                ids_digest(tuple(library_groups)),
                expected["library_group_ids_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
