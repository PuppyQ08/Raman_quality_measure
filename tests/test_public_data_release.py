from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import mlcroissant as mlc

from scripts.data.build_public_metadata import (
    build_croissant,
    build_limitations,
    build_public_checksum_index,
    build_release_matrix,
)
from scripts.data.fetch_sources import (
    DownloadError,
    SourceRegistryError,
    fetch_artifact,
    load_registry,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "metadata" / "sources.json"
CROISSANT = ROOT / "metadata" / "croissant.json"
PUBLIC_MATRIX = ROOT / "metadata" / "release_matrix.csv"
PUBLIC_CHECKSUMS = ROOT / "SHA256SUMS"
PUBLIC_LIMITATIONS = ROOT / "metadata" / "limitations.jsonl"


class SourceRegistryTest(unittest.TestCase):
    def test_registry_is_canonical_and_has_exact_release_groups(self) -> None:
        document = load_registry(REGISTRY)
        self.assertEqual(
            sorted(document["groups"]),
            ["bacteria_id", "rruff", "sugar_mixtures"],
        )
        self.assertEqual(len(document["groups"]["rruff"]), 8)
        self.assertEqual(len(document["groups"]["bacteria_id"]), 1)
        self.assertGreaterEqual(len(document["groups"]["sugar_mixtures"]), 1)
        self.assertEqual(
            REGISTRY.read_bytes(),
            (
                json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        for entries in document["groups"].values():
            for entry in entries:
                self.assertTrue(entry["url"].startswith("https://"))
                self.assertNotIn("cookie", json.dumps(entry).lower())
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(entry["bytes"], 0)
                self.assertFalse(Path(entry["path"]).is_absolute())

    def test_registry_rejects_http_unknown_keys_and_bad_digests(self) -> None:
        base = {
            "schema_version": "rpe-source-registry-v1",
            "groups": {
                "fixture": [
                    {
                        "artifact_id": "fixture",
                        "bytes": 3,
                        "path": "fixture/data.bin",
                        "sha256": hashlib.sha256(b"abc").hexdigest(),
                        "url": "https://example.org/data.bin",
                    }
                ]
            },
        }
        mutations = []
        insecure = json.loads(json.dumps(base))
        insecure["groups"]["fixture"][0]["url"] = "http://example.org/data.bin"
        mutations.append(insecure)
        bad_hash = json.loads(json.dumps(base))
        bad_hash["groups"]["fixture"][0]["sha256"] = "0" * 63
        mutations.append(bad_hash)
        extra = json.loads(json.dumps(base))
        extra["groups"]["fixture"][0]["token"] = "forbidden"
        mutations.append(extra)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.json"
            for document in mutations:
                path.write_text(json.dumps(document) + "\n", encoding="utf-8")
                with self.assertRaises(SourceRegistryError):
                    load_registry(path)


class SourceDownloadTest(unittest.TestCase):
    def _entry(self, source: Path, payload: bytes) -> dict[str, object]:
        return {
            "artifact_id": "fixture",
            "bytes": len(payload),
            "path": "fixture/data.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "url": source.as_uri(),
        }

    def test_fetch_verifies_bytes_and_is_idempotent(self) -> None:
        payload = b"deterministic fixture bytes"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(payload)
            data_root = root / "data"
            target = fetch_artifact(self._entry(source, payload), data_root)
            self.assertEqual(target.read_bytes(), payload)
            self.assertTrue(verify_artifact(self._entry(source, payload), data_root))
            target.chmod(0o444)
            self.assertEqual(fetch_artifact(self._entry(source, payload), data_root), target)

    def test_fetch_rejects_digest_mismatch_without_partial_output(self) -> None:
        payload = b"wrong bytes"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(payload)
            data_root = root / "data"
            entry = self._entry(source, b"expected bytes")
            with self.assertRaises(DownloadError):
                fetch_artifact(entry, data_root)
            self.assertFalse((data_root / "fixture/data.bin").exists())
            self.assertFalse((data_root / "fixture/data.bin.part").exists())

    def test_fetch_refuses_incorrect_existing_file(self) -> None:
        payload = b"correct bytes"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(payload)
            data_root = root / "data"
            target = data_root / "fixture/data.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"stale")
            with self.assertRaises(DownloadError):
                fetch_artifact(self._entry(source, payload), data_root)
            self.assertEqual(target.read_bytes(), b"stale")


class PublicCroissantTest(unittest.TestCase):
    def test_croissant_validates_and_binds_real_distribution_hashes(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="ConjunctiveGraph is deprecated, use Dataset instead.",
                category=DeprecationWarning,
            )
            dataset = mlc.Dataset(CROISSANT)
        self.assertEqual(
            dataset.metadata.name,
            "Raman Preprocessing Benchmark Metadata and Aggregate Results",
        )
        document = json.loads(CROISSANT.read_text(encoding="utf-8"))
        self.assertNotIn("example.invalid", CROISSANT.read_text(encoding="utf-8"))
        self.assertNotIn("0" * 64, CROISSANT.read_text(encoding="utf-8"))
        self.assertNotIn("license", document)
        self.assertEqual(document["datePublished"], "2026-09-01")
        listed_paths = {
            (CROISSANT.parent / item["contentUrl"])
            .resolve()
            .relative_to(ROOT.resolve())
            .as_posix()
            for item in document["distribution"]
        }
        expected_paths = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "benchmark").rglob("*")
            if path.is_file()
        } | {
            "metadata/data_card.md",
            "metadata/metadata_index.parquet",
            "metadata/release_matrix.csv",
            "metadata/limitations.jsonl",
            "metadata/source_environment.json",
            "metadata/sources.json",
        }
        self.assertEqual(listed_paths, expected_paths)
        for item in document["distribution"]:
            path = CROISSANT.parent / item["contentUrl"]
            self.assertTrue(path.is_file(), item["contentUrl"])
            self.assertEqual(item["contentSize"], str(path.stat().st_size))
            self.assertEqual(
                item["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_croissant_is_the_deterministic_generator_output(self) -> None:
        self.assertEqual(
            CROISSANT.read_bytes(),
            build_croissant(ROOT),
        )
        self.assertEqual(
            PUBLIC_MATRIX.read_bytes(),
            build_release_matrix(ROOT),
        )
        self.assertEqual(
            PUBLIC_CHECKSUMS.read_bytes(),
            build_public_checksum_index(ROOT),
        )
        checksum_paths = [
            line.split("  ", 1)[1]
            for line in PUBLIC_CHECKSUMS.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(checksum_paths), len(set(checksum_paths)))
        self.assertIn("scripts/__init__.py", checksum_paths)

    def test_public_limitations_replace_internal_workspace_history(self) -> None:
        self.assertEqual(PUBLIC_LIMITATIONS.read_bytes(), build_limitations())
        rows = [json.loads(line) for line in PUBLIC_LIMITATIONS.read_text().splitlines()]
        by_id = {row["limitation_id"]: row["state"] for row in rows}
        self.assertNotIn("unavailable_no_valid_git_repository", by_id)
        self.assertEqual(by_id["pending_public_repository_license"], "open")
        self.assertEqual(by_id["rruff_redistribution_unconfirmed"], "open")
        self.assertEqual(by_id["bacteria_id_redistribution_unconfirmed"], "open")
        self.assertEqual(by_id["phase5_not_powered_not_run"], "fixed")
        self.assertEqual(by_id["step7_deferred_not_admitted"], "fixed")


if __name__ == "__main__":
    unittest.main()
