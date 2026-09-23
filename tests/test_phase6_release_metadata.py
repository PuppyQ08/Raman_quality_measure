from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/phase6/configs/release_metadata_v1.json"
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_release_metadata import (  # noqa: E402
    DEFAULT_CONFIG,
    ReleaseMetadataError,
    ReleaseMetadataSummary,
    build_phase6_release_metadata,
    load_phase6_release_metadata_config,
)
from rpe.runner.phase6_release_metadata_verifier import (  # noqa: E402
    ReleaseMetadataVerificationError,
    verify_phase6_release_metadata,
)


def canonical(value: object) -> bytes:
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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        row = json.loads(line)
        if line != canonical(row):
            raise AssertionError(f"{path.name} is not canonical JSONL")
        rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def artifact_files(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(canonical(value))


def rewrite_sha256sums(path: Path) -> None:
    names = sorted(
        item.name
        for item in path.iterdir()
        if item.is_file() and item.name != "SHA256SUMS"
    )
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )


def make_step8_receipt(path: Path, run_id: str = "step8-fixture-run") -> None:
    write_canonical_json(
        path / "manifest.json",
        {
            "schema_version": "phase6-publication-core-artifact-v1",
            "run_id": run_id,
            "status": "complete",
        },
    )
    write_canonical_json(
        path / "complete.json",
        {
            "run_id": run_id,
            "status": "complete",
        },
    )
    rewrite_sha256sums(path)


class Phase6ReleaseMetadataConfigTest(unittest.TestCase):
    def test_public_surface_and_canonical_config_exist(self) -> None:
        self.assertTrue(issubclass(ReleaseMetadataError, ValueError))
        self.assertTrue(issubclass(ReleaseMetadataVerificationError, ValueError))
        self.assertEqual(DEFAULT_CONFIG, CONFIG)
        for value in (
            ReleaseMetadataSummary,
            load_phase6_release_metadata_config,
            build_phase6_release_metadata,
            verify_phase6_release_metadata,
        ):
            self.assertTrue(callable(value))
        raw = CONFIG.read_bytes()
        self.assertEqual(raw, canonical(json.loads(raw)))

    def test_config_binds_exact_payload_inventory_dispositions_and_runtime_gate(self) -> None:
        config = load_phase6_release_metadata_config(CONFIG)
        self.assertEqual(config.schema_version, "phase6-release-metadata-v1")
        self.assertEqual(
            tuple(config.document["artifact_contract"]["payload_files"]),
            (
                "config.json",
                "authority_bridge.json",
                "preflight.json",
                "release_matrix.csv",
                "data_card.md",
                "metadata_index.parquet",
                "croissant.json",
                "limitations.jsonl",
                "environment.json",
                "manifest.json",
            ),
        )
        self.assertEqual(
            tuple(config.document["release_rules"]["allowed_dispositions"]),
            (
                "public_release_candidate",
                "metadata_and_rebuild_instructions_only",
                "local_only_pending_redistribution_review",
                "external_reference_only",
                "excluded",
            ),
        )
        self.assertEqual(
            config.document["release_rules"]["source_revision_status"],
            "unavailable_no_valid_git_repository",
        )
        self.assertEqual(
            config.document["release_rules"]["leaderboard_state"],
            "deferred_no_redistributable_hidden_gt",
        )
        self.assertTrue(config.document["formal_build_requires_step8_receipt"])
        self.assertEqual(config.document["runtime_modes"]["default_mode"], "formal")
        self.assertEqual(
            tuple(config.document["paper_copy_targets"]),
            ("paper/croissant.json",),
        )

    def test_config_binds_current_authorities_and_installed_validator_expectation(self) -> None:
        config = load_phase6_release_metadata_config(CONFIG)
        authorities = config.document["authorities"]
        for name in (
            "phase0_step01_report",
            "phase4_step38_report",
            "source_manifest",
            "source_versions",
            "requirements_lock",
            "phase3_requirements_lock",
        ):
            identity = authorities[name]
            path = ROOT / identity["path"]
            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.stat().st_size, identity["byte_count"], name)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                identity["sha256"],
                name,
            )
        self.assertEqual(config.document["validator"]["python_package"], "mlcroissant")
        self.assertEqual(config.document["validator"]["python_package_version"], "1.1.0")
        self.assertEqual(config.document["validator"]["pyarrow_version"], "25.0.1")


class Phase6ReleaseMetadataBuildTest(unittest.TestCase):
    def test_formal_build_requires_explicit_verified_step8_publication_core_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            with self.assertRaisesRegex(
                ReleaseMetadataError,
                "step8.*required|publication-core.*required",
            ):
                build_phase6_release_metadata(
                    output_root,
                    config_path=CONFIG,
                    mode="formal",
                )

    def test_fixture_build_is_deterministic_and_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            first = build_phase6_release_metadata(
                output_root,
                config_path=CONFIG,
                mode="fixture",
            )
            second = build_phase6_release_metadata(
                output_root,
                config_path=CONFIG,
                mode="fixture",
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(artifact_files(first.path), artifact_files(second.path))
            self.assertEqual(
                sorted(item.name for item in first.path.iterdir() if item.is_file()),
                [
                    "SHA256SUMS",
                    "authority_bridge.json",
                    "complete.json",
                    "config.json",
                    "croissant.json",
                    "data_card.md",
                    "environment.json",
                    "limitations.jsonl",
                    "manifest.json",
                    "metadata_index.parquet",
                    "preflight.json",
                    "release_matrix.csv",
                ],
            )
            matrix = read_csv(first.path / "release_matrix.csv")
            self.assertEqual(len(matrix), 6)
            by_id = {row["artifact_id"]: row for row in matrix}
            self.assertEqual(
                by_id["rruff_record_level_derivatives"]["disposition"],
                "local_only_pending_redistribution_review",
            )
            self.assertEqual(
                by_id["bacteria_id_records"]["disposition"],
                "external_reference_only",
            )
            self.assertEqual(
                by_id["sugar_source_evidence"]["disposition"],
                "external_reference_only",
            )
            self.assertEqual(
                by_id["metadata_index_parquet"]["repository_license"],
                "pending_owner_license_selection",
            )
            table = pq.read_table(first.path / "metadata_index.parquet")
            self.assertEqual(
                table.column_names,
                [
                    "artifact_id",
                    "relative_path",
                    "media_type",
                    "disposition",
                    "source_kind",
                    "record_count",
                    "byte_count",
                    "sha256",
                    "restricted_record_bytes_present",
                ],
            )
            self.assertEqual(table.num_rows, 6)
            self.assertTrue(
                all(value is False for value in table["restricted_record_bytes_present"].to_pylist())
            )
            forbidden = {"intensity", "spectra", "axis_cm1", "wavenumber", "target"}
            self.assertTrue(forbidden.isdisjoint(table.column_names))
            self.assertFalse((first.path / "paper").exists())

    def test_fixture_build_croissant_validates_and_records_all_rai_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_phase6_release_metadata(
                Path(tmp) / "out",
                config_path=CONFIG,
                mode="fixture",
            )
            croissant = read_json(summary.path / "croissant.json")
            self.assertEqual(croissant["conformsTo"], "http://mlcommons.org/croissant/1.1")
            expected_fields = {
                "dataCollection",
                "dataCollectionType",
                "dataCollectionMissingData",
                "dataCollectionRawData",
                "dataCollectionTimeFrame",
                "dataImputationProtocol",
                "dataPreprocessingProtocol",
                "dataDataManipulationProtocol",
                "dataAnnotationProtocol",
                "dataAnnotationPlatform",
                "dataAnnotationAnalysis",
                "annotationsPerItem",
                "annotatorDemographics",
                "machineAnnotationTools",
                "dataBiases",
                "dataUseCases",
                "dataLimitations",
                "dataSocialImpact",
                "personalSensitiveInformation",
                "dataReleaseMaintenancePlan",
            }
            self.assertTrue(expected_fields.issubset(set(croissant)))
            self.assertEqual(len(expected_fields), 20)
            self.assertEqual(read_json(summary.path / "preflight.json")["croissant_validation"]["status"], "passed")

    def test_independent_verifier_rebuilds_bytes_and_detects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_phase6_release_metadata(
                Path(tmp) / "out",
                config_path=CONFIG,
                mode="fixture",
            )
            verified = verify_phase6_release_metadata(summary.path, config_path=CONFIG)
            self.assertEqual(verified.run_id, summary.run_id)
            self.assertEqual(artifact_files(verified.path), artifact_files(summary.path))
            (summary.path / "data_card.md").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ReleaseMetadataVerificationError,
                "data_card.md|byte|SHA256SUMS|mismatch",
            ):
                verify_phase6_release_metadata(summary.path, config_path=CONFIG)

    def test_verifier_is_not_allowed_to_import_production_release_metadata_module(self) -> None:
        tree = ast.parse(
            (ROOT / "rpe/runner/phase6_release_metadata_verifier.py").read_text(
                encoding="utf-8"
            )
        )
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase6_release_metadata":
                forbidden.append(node.module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "rpe.runner.phase6_release_metadata":
                        forbidden.append(alias.name)
        self.assertEqual(forbidden, [])

    def test_independent_verifier_detects_rewritten_sha256sums_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_phase6_release_metadata(
                Path(tmp) / "out",
                config_path=CONFIG,
                mode="fixture",
            )
            (summary.path / "data_card.md").write_text("tampered\n", encoding="utf-8")
            rewrite_sha256sums(summary.path)
            with self.assertRaisesRegex(
                ReleaseMetadataVerificationError,
                "data_card.md|byte mismatch",
            ):
                verify_phase6_release_metadata(summary.path, config_path=CONFIG)

    def test_formal_receipt_uses_repo_relative_step8_path_and_no_absolute_repo_path_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp_root = Path(tmp)
            step8 = tmp_root / "step8"
            step8.mkdir()
            make_step8_receipt(step8)
            summary = build_phase6_release_metadata(
                tmp_root / "out",
                config_path=CONFIG,
                mode="formal",
                step8_publication_core_path=step8.resolve(),
            )
            bridge = read_json(summary.path / "authority_bridge.json")
            receipt = bridge["step8_publication_core_receipt"]
            self.assertEqual(receipt["path"], str(step8.resolve().relative_to(ROOT)))
            project_root_bytes = str(ROOT).encode("utf-8")
            for _, payload in artifact_files(summary.path).items():
                self.assertNotIn(project_root_bytes, payload)
            verified = verify_phase6_release_metadata(summary.path, config_path=CONFIG)
            self.assertEqual(verified.run_id, summary.run_id)

    def test_cli_build_verify_and_freeze_subcommands_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            freeze_path = Path(tmp) / "candidate.json"
            build = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_phase6_release_metadata.py"),
                    "build",
                    "--mode",
                    "fixture",
                    "--output-root",
                    str(output_root),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run_path = Path(build.stdout.strip().split("\t", 1)[1])
            verify = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_phase6_release_metadata.py"),
                    "verify",
                    "--run-path",
                    str(run_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)
            freeze = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_phase6_release_metadata.py"),
                    "freeze-config",
                    "--output",
                    str(freeze_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(freeze.returncode, 0, freeze.stderr)
            self.assertTrue(freeze_path.is_file())


if __name__ == "__main__":
    unittest.main()
