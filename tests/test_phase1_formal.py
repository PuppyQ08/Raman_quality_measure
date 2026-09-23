from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from phase1_runner_helpers import (  # noqa: E402
    SWEEP_CONFIG,
    phase1_fixture_config,
    write_phase1_fixture_dataset,
)
from rpe.io import canonical_json_bytes, read_perturbed_shard  # noqa: E402
from rpe.io.store import UnifiedDataset  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_config import ArtifactIdentity  # noqa: E402
from rpe.runner.phase1_formal import (  # noqa: E402
    FORMAL_GATE_SCHEMA_VERSION,
    FORMAL_MARKER_SCHEMA_VERSION,
    FORMAL_RUN_SCHEMA_VERSION,
    Phase1FormalError,
    build_phase1_formal_run,
    derive_phase1_formal_run_id,
    verify_phase1_formal_run,
)
from rpe.runner.phase1_perturbations import estimate_p10_peak_bytes  # noqa: E402
from rpe.runner.phase1_selection import (  # noqa: E402
    load_phase1_source,
    load_source_inventory,
    select_source_rows,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Phase1FormalRunTest(unittest.TestCase):
    def _fixture(self, root: Path):
        dataset_path = write_phase1_fixture_dataset(root)
        sweep = load_perturbation_sweep_config(SWEEP_CONFIG)
        config = phase1_fixture_config(
            dataset_path,
            subset_size=4,
            shard_source_count=2,
        )
        config = replace(
            config,
            shared_sweep_path=Path(
                "experiments/shared/raman_perturbation_sweep_v1.json"
            ),
            shared_sweep_identity=ArtifactIdentity(
                byte_count=sweep.byte_count,
                sha256=sweep.sha256,
            ),
        )
        rows = select_source_rows(
            load_source_inventory(config),
            global_seed=sweep.global_seed,
            subset_size=config.subset_size,
        )
        with UnifiedDataset.open(dataset_path, verify_checksums=False) as dataset:
            sources = tuple(load_phase1_source(dataset, row) for row in rows)
        memory_budget = sum(
            estimate_p10_peak_bytes(source.spectrum.intensity.size)
            for source in sources
        )
        return config, sweep, sources, memory_budget

    def test_run_id_uses_only_six_length_prefixed_scientific_inputs(self) -> None:
        values = {
            "schema_version": FORMAL_RUN_SCHEMA_VERSION,
            "scientific_config_sha256": "1" * 64,
            "sweep_config_sha256": "2" * 64,
            "source_snapshot_sha256": "3" * 64,
            "selected_source_subset_sha256": "4" * 64,
            "data_code_snapshot_sha256": "5" * 64,
        }
        payload = bytearray(b"rpe-phase1-perturbation-run-id-v1\0")
        for value in values.values():
            encoded = value.encode("utf-8")
            payload.extend(struct.pack("<Q", len(encoded)))
            payload.extend(encoded)
        self.assertEqual(
            derive_phase1_formal_run_id(**values),
            "phase1-rruff-core10k-" + hashlib.sha256(payload).hexdigest(),
        )

    def test_two_shard_build_and_readback_verification_close_core_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, sweep, sources, budget = self._fixture(root)
            output_root = root / "formal"
            summary = build_phase1_formal_run(
                output_root,
                config=config,
                sweep=sweep,
                sources=sources,
                worker_count=2,
                memory_budget_bytes=budget,
            )

            self.assertEqual(summary.path.parent, output_root)
            self.assertEqual(summary.path.name, summary.run_id)
            self.assertEqual(summary.source_count, 4)
            self.assertEqual(summary.class_count, 3)
            self.assertEqual(summary.shard_count, 2)
            self.assertEqual(summary.cell_count, 48)
            self.assertEqual(summary.record_count, 360)
            self.assertEqual(summary.core_dataset_gate, "pass")
            self.assertEqual(
                summary.full_phase1_gate,
                "deferred_missing_p06_p07",
            )
            self.assertEqual(
                {path.name for path in summary.path.iterdir()},
                {
                    "SHA256SUMS",
                    "complete.json",
                    "gate.json",
                    "manifest.json",
                    "shards",
                    "source_subset.jsonl",
                },
            )
            self.assertEqual(
                sorted(path.name for path in (summary.path / "shards").iterdir()),
                ["00000", "00001"],
            )
            first = read_perturbed_shard(summary.path / "shards" / "00000")
            second = read_perturbed_shard(summary.path / "shards" / "00001")
            self.assertEqual(
                tuple(source.source_record_id for source in first.sources),
                tuple(source.selection.record_id for source in sources[:2]),
            )
            self.assertEqual(
                tuple(source.source_record_id for source in second.sources),
                tuple(source.selection.record_id for source in sources[2:]),
            )

            subset_lines = (summary.path / "source_subset.jsonl").read_bytes().splitlines()
            subset = tuple(json.loads(line) for line in subset_lines)
            self.assertEqual(tuple(row["selection_rank"] for row in subset), (0, 1, 2, 3))
            self.assertEqual(
                tuple(row["record_id"] for row in subset),
                tuple(source.selection.record_id for source in sources),
            )

            manifest_path = summary.path / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            self.assertEqual(manifest_path.read_bytes(), canonical_json_bytes(manifest))
            self.assertEqual(manifest["schema_version"], FORMAL_RUN_SCHEMA_VERSION)
            self.assertEqual(manifest["run_id"], summary.run_id)
            self.assertEqual(
                manifest["operational_config"],
                {
                    "byte_count": config.file_byte_count,
                    "sha256": config.file_sha256,
                },
            )
            self.assertEqual(manifest["counts"]["source_count"], 4)
            self.assertEqual(manifest["counts"]["shard_count"], 2)
            self.assertEqual(
                [entry["path"] for entry in manifest["shards"]],
                ["shards/00000", "shards/00001"],
            )
            self.assertEqual(manifest["vcs_status"], "unavailable")
            self.assertIsNone(manifest["git_commit"])

            gate_path = summary.path / "gate.json"
            gate = json.loads(gate_path.read_bytes())
            self.assertEqual(gate_path.read_bytes(), canonical_json_bytes(gate))
            self.assertEqual(gate["schema_version"], FORMAL_GATE_SCHEMA_VERSION)
            self.assertEqual(gate["core_dataset_gate"], "pass")
            self.assertEqual(gate["full_phase1_gate"], "deferred_missing_p06_p07")
            self.assertEqual(gate["deferred_cell_count"], 8)
            self.assertEqual(gate["failed_cell_count"], 0)
            self.assertEqual(gate["operator_status_counts"]["p06"], {"complete": 0, "failed": 0, "not_applicable": 4})
            self.assertEqual(gate["operator_status_counts"]["p07"], {"complete": 0, "failed": 0, "not_applicable": 4})

            marker_path = summary.path / "complete.json"
            marker = json.loads(marker_path.read_bytes())
            self.assertEqual(marker_path.read_bytes(), canonical_json_bytes(marker))
            self.assertEqual(
                marker,
                {
                    "core_dataset_gate": "pass",
                    "full_phase1_gate": "deferred_missing_p06_p07",
                    "run_id": summary.run_id,
                    "schema_version": FORMAL_MARKER_SCHEMA_VERSION,
                    "state": "complete",
                },
            )

            checksums = {}
            for line in (summary.path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                checksums[relative] = digest
                self.assertEqual(_sha256(summary.path / relative), digest)
            self.assertNotIn("complete.json", checksums)
            self.assertNotIn("SHA256SUMS", checksums)
            self.assertEqual(tuple(checksums), tuple(sorted(checksums)))

            verified = verify_phase1_formal_run(
                summary.path,
                config=config,
                sweep=sweep,
                sources=sources,
            )
            self.assertEqual(verified, summary)
            self.assertEqual(tuple(verified.checked_files), tuple(checksums))

            blocked_root = root / "blocked"
            with self.assertRaisesRegex(
                Phase1FormalError,
                "memory_budget_bytes.*largest P10",
            ):
                build_phase1_formal_run(
                    blocked_root,
                    config=config,
                    sweep=sweep,
                    sources=sources,
                    worker_count=2,
                    memory_budget_bytes=1,
                )
            self.assertFalse(blocked_root.exists())

            with self.assertRaisesRegex(Phase1FormalError, "output.*already exists"):
                build_phase1_formal_run(
                    output_root,
                    config=config,
                    sweep=sweep,
                    sources=sources,
                    worker_count=2,
                    memory_budget_bytes=budget,
                )

            gate_path.write_bytes(gate_path.read_bytes() + b" ")
            with self.assertRaisesRegex(Phase1FormalError, "SHA256SUMS.gate.json"):
                verify_phase1_formal_run(
                    summary.path,
                    config=config,
                    sweep=sweep,
                    sources=sources,
                )

    def test_failed_gate_run_is_checksumming_and_readback_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, sweep, sources, budget = self._fixture(root)
            failing_gate = dict(config.core_gate)
            failing_gate["p01_p04_min_source_fraction"] = 1.1
            failing_config = replace(config, core_gate=failing_gate)
            output_root = root / "failed-formal"

            with self.assertRaisesRegex(Phase1FormalError, "formal core gate failed"):
                build_phase1_formal_run(
                    output_root,
                    config=failing_config,
                    sweep=sweep,
                    sources=sources,
                    worker_count=2,
                    memory_budget_bytes=budget,
                )

            run_paths = tuple(output_root.iterdir())
            self.assertEqual(len(run_paths), 1)
            run_path = run_paths[0]
            self.assertTrue((run_path / "failed.json").is_file())
            self.assertFalse((run_path / "complete.json").exists())
            summary = verify_phase1_formal_run(
                run_path,
                config=failing_config,
                sweep=sweep,
                sources=sources,
            )
            self.assertEqual(summary.core_dataset_gate, "fail")
            self.assertEqual(summary.full_phase1_gate, "fail")
            self.assertEqual(summary.source_count, 4)
            self.assertGreater(len(summary.checked_files), 0)
            with (
                patch(
                    "rpe.runner.phase1_formal.code_snapshot_document",
                    side_effect=AssertionError("must use embedded build snapshot"),
                ),
                patch(
                    "rpe.runner.phase1_formal.code_snapshot_digest",
                    side_effect=AssertionError("must use embedded build snapshot"),
                ),
            ):
                repeated = verify_phase1_formal_run(
                    run_path,
                    config=failing_config,
                    sweep=sweep,
                    sources=sources,
                )
            self.assertEqual(repeated, summary)


if __name__ == "__main__":
    unittest.main()
