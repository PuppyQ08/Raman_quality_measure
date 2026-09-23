from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from rpe.evaluation import Spectrum1D
from rpe.methods import TaskLine, load_classical_catalog
from rpe.methods.classical import PeakRunStatus
import rpe.runner.phase3_peak_formal as formal
from rpe.runner import (
    PeakFormalCoveragePolicy, PeakFormalError, PeakFormalReceipt,
    evaluate_peak_formal_coverage, load_phase3_peak_formal_config,
    load_phase3_peak_formal_sources, verify_phase3_peak_formal,
)
import tools.run_phase3_peak_formal as cli

CONFIG = ROOT / "experiments/phase3/configs/peak_detection_v1_formal_coverage.json"
CATALOG = ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def source(record_id: str, order: int, values: tuple[float, ...]) -> formal.PeakFormalSource:
    axis = np.arange(len(values), dtype="<f8") + 100.0
    intensity = np.asarray(values, dtype="<f8")
    spectrum = Spectrum1D(record_id, record_id, axis, intensity)
    return formal.PeakFormalSource(
        "fixture", record_id, order, order, len(values),
        hashlib.sha256(axis.tobytes()).hexdigest(),
        hashlib.sha256(intensity.tobytes()).hexdigest(), spectrum, record_id, f"m{order}",
    )


class PeakFormalConfigTest(unittest.TestCase):
    def test_config_binds_exact_frozen_scope(self) -> None:
        raw = CONFIG.read_bytes()
        self.assertEqual(raw, canonical(json.loads(raw)))
        config = load_phase3_peak_formal_config(CONFIG)
        self.assertEqual(config.schema_version, "phase3-peak-formal-coverage-v1")
        self.assertEqual((config.expected.planned_k, config.expected.runnable_k), (36, 24))
        self.assertEqual((config.expected.find_peaks_system_count, config.expected.find_peaks_cwt_system_count), (12, 12))
        self.assertEqual(config.expected.mspd_unavailable_system_count, 12)
        self.assertEqual((config.expected.source_count, config.expected.source_point_count, config.expected.receipt_count), (10000, 22053403, 240000))
        self.assertEqual(config.system_ids_sha256, "7f2dd112b75a8cb439148855220d8e7877175f7f797cc95e90e05f0e3f72b506")
        self.assertEqual(config.mspd_system_ids_sha256, "d7d98e996605da2a5f5315d4da182ba15673b1a9dc7a7928264549f61aeb9415")
        self.assertEqual(config.phase5_power_status, "not_powered_for_phase5")
        self.assertEqual(config.policy.subset_full_tau_status, "not_evaluable_no_quality_endpoint")
        self.assertEqual((config.operational.worker_processes, config.operational.verifier_worker_processes), (8, 7))

    def test_real_loader_reconstructs_exact_core10k(self) -> None:
        config = load_phase3_peak_formal_config(CONFIG)
        sources = load_phase3_peak_formal_sources(config, project_root=ROOT)
        self.assertEqual(len(sources), 10000)
        self.assertEqual(sum(value.point_count for value in sources), 22053403)
        self.assertEqual(tuple(value.source_order for value in sources), tuple(range(10000)))
        self.assertEqual(sources[0].record_id, "raw-0003d1b556e9bc8434a8a7972d5f")
        self.assertTrue(all(not value.spectrum.intensity.flags.writeable for value in sources[:10]))


class PeakFormalCoverageTest(unittest.TestCase):
    def test_empty_success_promotes_but_one_unsuccessful_blocks(self) -> None:
        catalog = load_classical_catalog(CATALOG)
        systems = tuple(value for value in catalog.systems if value.task_line is TaskLine.PEAK_DETECTION and value.family_id == "find_peaks")[:2]
        receipts = (
            PeakFormalReceipt(systems[0].system_id, systems[0].family_id, systems[0].method_id, "r0", 0, PeakRunStatus.COMPLETE, 0, 0, None, True, (), {}, None, None),
            PeakFormalReceipt(systems[0].system_id, systems[0].family_id, systems[0].method_id, "r1", 1, PeakRunStatus.COMPLETE_WITH_WARNING, 3, 0, "a" * 64, False, (), {}, None, None),
            PeakFormalReceipt(systems[1].system_id, systems[1].family_id, systems[1].method_id, "r0", 0, PeakRunStatus.COMPLETE, 0, 0, None, True, (), {}, None, None),
            PeakFormalReceipt(systems[1].system_id, systems[1].family_id, systems[1].method_id, "r1", 1, PeakRunStatus.NOT_APPLICABLE, 0, 0, None, False, (), {}, "domain", "bad"),
        )
        result = evaluate_peak_formal_coverage(
            receipts=receipts, source_ids=("r0", "r1"), systems=systems,
            policy=PeakFormalCoveragePolicy(True, .99, .95, 1),
        )
        self.assertEqual(result.promoted_system_ids, (systems[0].system_id,))
        self.assertEqual(result.common_successful_record_count, 2)
        self.assertEqual(result.system_summaries[0].empty_receipt_count, 1)
        self.assertFalse(result.system_summaries[1].coverage_promoted)


class PeakFormalArtifactTest(unittest.TestCase):
    def fixture(self):
        catalog = load_classical_catalog(CATALOG)
        systems = (
            next(value for value in catalog.systems if value.task_line is TaskLine.PEAK_DETECTION and value.family_id == "find_peaks"),
            next(value for value in catalog.systems if value.task_line is TaskLine.PEAK_DETECTION and value.family_id == "find_peaks_cwt"),
        )
        sources = (source("r0", 0, (0, 0, 2, 0, 0, 1, 0)), source("r1", 1, (0, 1, 0, 2, 0, 1, 0)))
        return catalog, systems, sources

    def test_streamed_fixture_is_worker_independent_and_verifiable(self) -> None:
        config = load_phase3_peak_formal_config(CONFIG); catalog, systems, sources = self.fixture()
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = formal._build_phase3_peak_formal_fixture_artifact(sources, systems, Path(a), config=config, catalog=catalog, project_root=ROOT, worker_count=1)
            second = formal._build_phase3_peak_formal_fixture_artifact(sources, systems, Path(b), config=config, catalog=catalog, project_root=ROOT, worker_count=2)
            self.assertEqual(first.run_id, second.run_id)
            left = {p.relative_to(first.path).as_posix(): p.read_bytes() for p in first.path.rglob("*") if p.is_file()}
            right = {p.relative_to(second.path).as_posix(): p.read_bytes() for p in second.path.rglob("*") if p.is_file()}
            self.assertEqual(left, right)
            records = [json.loads(line) for line in left["records.jsonl"].splitlines()]
            self.assertEqual([(r["system_id"], r["source_order"]) for r in records], sorted((r["system_id"], r["source_order"]) for r in records))
            self.assertTrue(all("empty_output" in row for row in records))
            self.assertEqual(verify_phase3_peak_formal(first.path, worker_count=3, project_root=ROOT).run_id, first.run_id)

    def test_verifier_rejects_checksum_consistent_peak_tampering(self) -> None:
        config = load_phase3_peak_formal_config(CONFIG); catalog, systems, sources = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            summary = formal._build_phase3_peak_formal_fixture_artifact(sources, systems, Path(root), config=config, catalog=catalog, project_root=ROOT, worker_count=1)
            peaks_path = summary.path / "peaks.jsonl"; checks = summary.path / "SHA256SUMS"
            original = peaks_path.read_bytes(); ledger = checks.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]["height"] += 1.0
            changed = b"".join(canonical(row) for row in rows)
            peaks_path.write_bytes(changed)
            checks.write_text(ledger.replace(hashlib.sha256(original).hexdigest(), hashlib.sha256(changed).hexdigest()))
            with self.assertRaisesRegex(PeakFormalError, "peak ledger"):
                verify_phase3_peak_formal(summary.path, worker_count=2, project_root=ROOT)


class PeakFormalCliTest(unittest.TestCase):
    def test_cli_has_only_build_and_verify(self) -> None:
        built = mock.Mock(path=Path("/tmp/p"), run_id="r")
        with mock.patch.object(cli, "build_phase3_peak_formal", return_value=built) as call:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["build", "--output-root", "/tmp/o", "--worker-count", "8"]), 0)
            self.assertEqual(json.loads(out.getvalue())["status"], "built")
            call.assert_called_once_with(Path("/tmp/o"), worker_count=8)
        with self.assertRaises(SystemExit):
            cli.main(["build", "--output-root", "/tmp/o", "--system-id", "x"])


if __name__ == "__main__":
    unittest.main()
