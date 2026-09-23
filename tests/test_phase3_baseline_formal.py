from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.methods import TaskLine, load_classical_catalog  # noqa: E402
from rpe.methods.classical import BaselineCanarySource  # noqa: E402
from rpe.runner import (  # noqa: E402
    BaselineCoveragePolicy,
    BaselineFormalError,
    BaselineFormalReceipt,
    BaselineRunStatus,
    build_baseline_receipt_artifact,
    evaluate_baseline_coverage,
    load_phase3_baseline_formal_config,
    load_rruff_baseline_formal_sources,
    verify_phase3_baseline_formal,
)


CONFIG = ROOT / "experiments" / "phase3" / "configs" / "baseline_formal10k_v1.json"
CATALOG = ROOT / "experiments" / "phase3" / "configs" / "classical_system_catalog_v1.json"


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


def oracle_source(record_id: str, rank: int) -> BaselineCanarySource:
    axis = np.linspace(100.0, 400.0, 32, dtype="<f8")
    intensity = (
        0.01 * axis
        + 3.0
        + 20.0 * np.exp(-0.5 * ((axis - 250.0) / 12.0) ** 2)
    ).astype("<f8")
    spectrum = Spectrum1D(
        spectrum_id=f"fixture::{record_id}",
        sample_id=f"sample-{rank}",
        axis_cm1=axis,
        intensity=intensity,
    )
    return BaselineCanarySource(
        band_id="fixture",
        selection_rank=rank,
        record_id=record_id,
        sample_id=f"sample-{rank}",
        class_label=rank + 1,
        mineral_name=f"mineral-{rank}",
        point_count=32,
        axis_sha256=hashlib.sha256(axis.tobytes()).hexdigest(),
        intensity_sha256=hashlib.sha256(intensity.tobytes()).hexdigest(),
        spectrum=spectrum,
    )


def receipt(
    record_id: str,
    rank: int,
    system,
    status: BaselineRunStatus,
) -> BaselineFormalReceipt:
    has_outputs = status in {
        BaselineRunStatus.COMPLETE,
        BaselineRunStatus.COMPLETE_WITH_WARNING,
        BaselineRunStatus.FAILED_CONVERGENCE,
    }
    return BaselineFormalReceipt(
        selection_rank=rank,
        record_id=record_id,
        system_id=system.system_id,
        family_id=system.family_id,
        method_id=system.method_id,
        status=status,
        baseline_sha256=("1" * 64 if has_outputs else None),
        corrected_sha256=("2" * 64 if has_outputs else None),
        warnings=(),
        diagnostics={},
        error_code=("convergence_evidence_failed" if status is BaselineRunStatus.FAILED_CONVERGENCE else None),
        error_message=("did not converge" if status is BaselineRunStatus.FAILED_CONVERGENCE else None),
    )


class Phase3BaselineFormalConfigTest(unittest.TestCase):
    def test_config_is_canonical_and_freezes_receipt_only_gates(self) -> None:
        raw = CONFIG.read_bytes()
        document = json.loads(raw)
        self.assertEqual(raw, canonical(document))
        config = load_phase3_baseline_formal_config(CONFIG)
        self.assertEqual(config.schema_version, "phase3-baseline-formal10k-config-v1")
        self.assertEqual(config.expected_source_count, 10000)
        self.assertEqual(config.expected_system_count, 210)
        self.assertEqual(config.expected_attempt_count, 2100000)
        self.assertEqual(config.worker_processes, 8)
        self.assertEqual(config.shard_source_count, 100)
        self.assertEqual(config.storage_mode, "receipt_only_no_arrays")
        self.assertEqual(config.policy.minimum_successful_fraction, 0.99)
        self.assertEqual(config.policy.minimum_phase5_systems, 200)
        self.assertEqual(config.policy.minimum_phase5_families, 10)
        self.assertEqual(config.policy.minimum_promoted_per_family, 8)
        self.assertEqual(config.policy.minimum_common_record_fraction, 0.95)
        self.assertEqual(
            config.success_statuses,
            (BaselineRunStatus.COMPLETE, BaselineRunStatus.COMPLETE_WITH_WARNING),
        )
        self.assertEqual(
            hashlib.sha256((ROOT / config.catalog.path).read_bytes()).hexdigest(),
            config.catalog.sha256,
        )

    def test_real_formal_source_loader_reproduces_frozen_10k_ledger(self) -> None:
        config = load_phase3_baseline_formal_config(CONFIG)
        sources = load_rruff_baseline_formal_sources(config, project_root=ROOT)
        self.assertEqual(len(sources), 10000)
        self.assertEqual(
            (sources[0].selection_rank, sources[0].record_id, sources[0].point_count),
            (0, "raw-0003d1b556e9bc8434a8a7972d5f", 1331),
        )
        self.assertEqual(sources[-1].selection_rank, 9999)
        self.assertEqual(
            hashlib.sha256(
                b"".join(
                    canonical(
                        {
                            "axis_sha256": source.axis_sha256,
                            "class_label": source.class_label,
                            "intensity_sha256": source.intensity_sha256,
                            "mineral_name": source.mineral_name,
                            "point_count": source.point_count,
                            "record_id": source.record_id,
                            "sample_id": source.sample_id,
                            "selection_rank": source.selection_rank,
                        }
                    )
                    for source in sources
                )
            ).hexdigest(),
            "445ccdc26aaccf404bfcdeffdb0e7389a1e63f612215068dd0915c10c52dfbea",
        )


class Phase3BaselineCoverageGateTest(unittest.TestCase):
    def test_literal_promotion_family_and_common_record_gate(self) -> None:
        catalog = load_classical_catalog(CATALOG)
        asls = [
            system
            for system in catalog.systems
            if system.task_line is TaskLine.BASELINE_CORRECTION
            and system.family_id == "asls"
        ][:2]
        airpls = next(
            system
            for system in catalog.systems
            if system.task_line is TaskLine.BASELINE_CORRECTION
            and system.family_id == "airpls"
        )
        systems = (*asls, airpls)
        receipts = (
            receipt("r0", 0, asls[0], BaselineRunStatus.COMPLETE),
            receipt("r1", 1, asls[0], BaselineRunStatus.COMPLETE_WITH_WARNING),
            receipt("r0", 0, asls[1], BaselineRunStatus.COMPLETE),
            receipt("r1", 1, asls[1], BaselineRunStatus.FAILED_CONVERGENCE),
            receipt("r0", 0, airpls, BaselineRunStatus.COMPLETE),
            receipt("r1", 1, airpls, BaselineRunStatus.COMPLETE),
        )
        policy = BaselineCoveragePolicy(
            minimum_successful_fraction=0.99,
            minimum_phase5_systems=2,
            minimum_phase5_families=2,
            minimum_promoted_per_family=1,
            minimum_common_record_fraction=1.0,
            require_zero_failed_runtime=True,
            require_zero_not_applicable=True,
        )
        result = evaluate_baseline_coverage(
            receipts,
            source_ids=("r0", "r1"),
            systems=systems,
            policy=policy,
        )
        self.assertEqual(result.promoted_system_ids, (asls[0].system_id, airpls.system_id))
        self.assertEqual(result.qualifying_family_ids, ("airpls", "asls"))
        self.assertEqual(result.phase5_eligible_system_count, 2)
        self.assertEqual(result.common_successful_record_count, 2)
        self.assertEqual(result.common_record_fraction, 1.0)
        self.assertTrue(result.gate_passed)
        second_summary = next(
            summary for summary in result.system_summaries if summary.system_id == asls[1].system_id
        )
        self.assertEqual(second_summary.successful_count, 1)
        self.assertEqual(second_summary.status_counts["failed_convergence"], 1)
        self.assertFalse(second_summary.coverage_promoted)

    def test_not_applicable_and_runtime_disqualify_even_at_high_coverage(self) -> None:
        catalog = load_classical_catalog(CATALOG)
        system = next(
            system
            for system in catalog.systems
            if system.task_line is TaskLine.BASELINE_CORRECTION
        )
        policy = BaselineCoveragePolicy(
            minimum_successful_fraction=0.5,
            minimum_phase5_systems=1,
            minimum_phase5_families=1,
            minimum_promoted_per_family=1,
            minimum_common_record_fraction=0.5,
            require_zero_failed_runtime=True,
            require_zero_not_applicable=True,
        )
        for status in (BaselineRunStatus.NOT_APPLICABLE, BaselineRunStatus.FAILED_RUNTIME):
            with self.subTest(status=status):
                rows = (
                    receipt("r0", 0, system, BaselineRunStatus.COMPLETE),
                    receipt("r1", 1, system, status),
                )
                result = evaluate_baseline_coverage(
                    rows, source_ids=("r0", "r1"), systems=(system,), policy=policy
                )
                self.assertEqual(result.promoted_system_ids, ())
                self.assertFalse(result.gate_passed)


class Phase3BaselineReceiptArtifactTest(unittest.TestCase):
    def test_small_artifact_is_receipt_only_deterministic_and_verifiable(self) -> None:
        config = load_phase3_baseline_formal_config(CONFIG)
        catalog = load_classical_catalog(CATALOG)
        systems = tuple(
            system
            for system in catalog.systems
            if system.task_line is TaskLine.BASELINE_CORRECTION
            and system.family_id in {"airpls", "iarpls"}
            and system.hyperparameters.get("lam") == 100000.0
        )
        self.assertEqual(len(systems), 2)
        sources = (oracle_source("fixture-r0", 0), oracle_source("fixture-r1", 1))
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_baseline_receipt_artifact(
                sources,
                systems,
                Path(first_dir),
                catalog=catalog,
                config=config,
                project_root=ROOT,
            )
            second = build_baseline_receipt_artifact(
                sources,
                systems,
                Path(second_dir),
                catalog=catalog,
                config=config,
                project_root=ROOT,
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(first.attempt_count, 4)
            self.assertFalse(first.is_formal_run)
            self.assertFalse(first.gate_passed)
            self.assertTrue((first.path / "failed.json").is_file())
            self.assertFalse((first.path / "complete.json").exists())
            names = {path.relative_to(first.path).as_posix() for path in first.path.rglob("*") if path.is_file()}
            self.assertEqual(
                names,
                {
                    "SHA256SUMS",
                    "failed.json",
                    "family_summary.jsonl",
                    "gate.json",
                    "manifest.json",
                    "promotion.json",
                    "shards/00000.jsonl",
                    "source_subset.jsonl",
                    "system_summary.jsonl",
                },
            )
            for name in names:
                self.assertEqual((first.path / name).read_bytes(), (second.path / name).read_bytes())
            rows = [json.loads(line) for line in (first.path / "shards/00000.jsonl").read_bytes().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [(row["selection_rank"], row["system_id"]) for row in rows],
                sorted((row["selection_rank"], row["system_id"]) for row in rows),
            )
            for row in rows:
                self.assertNotIn("baseline_estimate", row)
                self.assertNotIn("corrected_intensity", row)
                self.assertNotIn("axis_cm1", row)
                self.assertNotIn("intensity", row)
            verified = verify_phase3_baseline_formal(first.path, project_root=ROOT)
            self.assertEqual(verified.run_id, first.run_id)
            self.assertEqual(verified.attempt_count, 4)

    def test_two_worker_fixture_is_byte_identical_to_sequential(self) -> None:
        config = load_phase3_baseline_formal_config(CONFIG)
        catalog = load_classical_catalog(CATALOG)
        systems = tuple(
            system
            for system in catalog.systems
            if system.task_line is TaskLine.BASELINE_CORRECTION
        )[:4]
        sources = tuple(oracle_source(f"fixture-r{index}", index) for index in range(4))
        with tempfile.TemporaryDirectory() as sequential_dir, tempfile.TemporaryDirectory() as parallel_dir:
            sequential = build_baseline_receipt_artifact(
                sources, systems, Path(sequential_dir), catalog=catalog, config=config,
                project_root=ROOT, worker_processes=1, shard_source_count=2,
            )
            parallel = build_baseline_receipt_artifact(
                sources, systems, Path(parallel_dir), catalog=catalog, config=config,
                project_root=ROOT, worker_processes=2, shard_source_count=2,
            )
            self.assertEqual(sequential.run_id, parallel.run_id)
            sequential_files = {
                path.relative_to(sequential.path).as_posix(): path.read_bytes()
                for path in sequential.path.rglob("*")
                if path.is_file()
            }
            parallel_files = {
                path.relative_to(parallel.path).as_posix(): path.read_bytes()
                for path in parallel.path.rglob("*")
                if path.is_file()
            }
            self.assertEqual(sequential_files, parallel_files)

    def test_verifier_rejects_checksum_row_order_and_rebound_gate_drift(self) -> None:
        config = load_phase3_baseline_formal_config(CONFIG)
        catalog = load_classical_catalog(CATALOG)
        systems = tuple(
            system
            for system in catalog.systems
            if system.task_line is TaskLine.BASELINE_CORRECTION
        )[:2]
        sources = (oracle_source("fixture-r0", 0), oracle_source("fixture-r1", 1))
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = build_baseline_receipt_artifact(
                sources, systems, Path(temp_dir), catalog=catalog, config=config, project_root=ROOT
            )
            shard = summary.path / "shards" / "00000.jsonl"
            original = shard.read_bytes()
            shard.write_bytes(b"{}\n")
            with self.assertRaisesRegex(BaselineFormalError, "checksum"):
                verify_phase3_baseline_formal(summary.path, project_root=ROOT)
            shard.write_bytes(original)

            lines = original.splitlines(keepends=True)
            shard.write_bytes(b"".join(reversed(lines)))
            checksum_path = summary.path / "SHA256SUMS"
            checksum_bytes = checksum_path.read_bytes()
            checksum_path.write_text(
                checksum_bytes.decode().replace(
                    hashlib.sha256(original).hexdigest(),
                    hashlib.sha256(shard.read_bytes()).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BaselineFormalError, "row order"):
                verify_phase3_baseline_formal(summary.path, project_root=ROOT)
            shard.write_bytes(original)
            checksum_path.write_bytes(checksum_bytes)

            system_summary_path = summary.path / "system_summary.jsonl"
            system_summary_original = system_summary_path.read_bytes()
            system_summary_rows = [
                json.loads(line) for line in system_summary_original.splitlines()
            ]
            system_summary_rows[0]["successful_count"] += 1
            system_summary_payload = b"".join(
                canonical(row) for row in system_summary_rows
            )
            system_summary_path.write_bytes(system_summary_payload)
            checksum_path.write_text(
                checksum_bytes.decode().replace(
                    hashlib.sha256(system_summary_original).hexdigest(),
                    hashlib.sha256(system_summary_payload).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BaselineFormalError, "system summary"):
                verify_phase3_baseline_formal(summary.path, project_root=ROOT)
            system_summary_path.write_bytes(system_summary_original)
            checksum_path.write_bytes(checksum_bytes)

            gate_path = summary.path / "gate.json"
            gate_original = gate_path.read_bytes()
            gate = json.loads(gate_original)
            gate["gate_passed"] = not gate["gate_passed"]
            gate_payload = canonical(gate)
            gate_path.write_bytes(gate_payload)
            checksum_path.write_text(
                checksum_bytes.decode().replace(
                    hashlib.sha256(gate_original).hexdigest(),
                    hashlib.sha256(gate_payload).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BaselineFormalError, "gate"):
                verify_phase3_baseline_formal(summary.path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
