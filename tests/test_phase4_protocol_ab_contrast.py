from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_protocol_ab_contrast import (  # noqa: E402
    ARTIFACT_PAYLOAD_FILES,
    Phase4ProtocolABContrastError,
    ProtocolABCellInput,
    adapt_protocol_observation_rows,
    build_phase4_protocol_ab_contrast_from_inputs,
    build_protocol_ab_holm,
    compute_protocol_ab_cell,
    load_phase4_protocol_ab_contrast_config,
    make_synthetic_protocol_ab_contrast_config,
    make_synthetic_protocol_ab_contrast_inputs,
    parse_phase4_protocol_ab_contrast_config,
    reconstruct_phase4_protocol_ab_contrast_inputs,
    validate_parent_artifact,
)


EXPECTED_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "endpoint_status.jsonl",
    "downstream_protocol_effects.jsonl",
    "metric_protocol_effects.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "protocol_downstream_contrast_table.csv",
    "protocol_metric_interaction_table.csv",
    "manifest.json",
)


def _hand_cell() -> ProtocolABCellInput:
    # Shape is metric, cluster, perturbation, alpha. Both clusters are identical.
    metric = np.asarray(
        [
            [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
            [[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]],
        ],
        dtype=np.float64,
    )
    downstream_a = np.asarray(
        [[[0.0, 1.0], [2.0, 3.0]], [[0.0, 1.0], [2.0, 3.0]]],
        dtype=np.float64,
    )
    downstream_b = np.asarray(
        [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    return ProtocolABCellInput(
        cell_id="hand",
        cluster_kind="class_label",
        cluster_ids=("c1", "c2"),
        metric_output_ids=("mse", "candidate"),
        perturbation_ids=("p01", "p02"),
        alpha_grid=(0.1, 0.2),
        metric_harm=metric,
        downstream_a=downstream_a,
        downstream_b=downstream_b,
        metric_reconciliation={"mismatch_count": 0, "max_abs_difference": 0.0},
    )


class ProtocolABPureStatisticsTest(unittest.TestCase):
    def test_exact_parent_alignment_qc_uses_boolean_flags_and_passes(self) -> None:
        cell = replace(
            _hand_cell(),
            parent_alignment_a={
                "mse": {"ag": 0.8, "acc_cross": 0.5, "d_ag": None, "d_acc": None},
                "candidate": {"ag": 0.0, "acc_cross": 1.0, "d_ag": 0.8, "d_acc": 0.5},
            },
            parent_alignment_b={
                "mse": {"ag": 0.0, "acc_cross": 0.75, "d_ag": None, "d_acc": None},
                "candidate": {"ag": 0.5, "acc_cross": 0.5, "d_ag": -0.5, "d_acc": -0.25},
            },
        )
        result = compute_protocol_ab_cell(
            cell, bootstrap_resamples=1, sign_flip_resamples=1, random_seed=20260817
        )
        self.assertTrue(all(type(row["parent_a_exact"]) is bool for row in result["metric_rows"]))
        self.assertTrue(all(row["parent_a_exact"] for row in result["metric_rows"]))
        self.assertTrue(all(row["parent_b_exact"] for row in result["metric_rows"]))

    def test_hand_oracle_freezes_protocol_effect_and_interaction_signs(self) -> None:
        result = compute_protocol_ab_cell(
            _hand_cell(),
            bootstrap_resamples=16,
            sign_flip_resamples=32,
            random_seed=20260817,
        )
        downstream = {
            (row["perturbation_id"], row["summary_type"]): row
            for row in result["downstream_rows"]
        }
        self.assertEqual(downstream[("p01", "integrated")]["gap"], 0.0)
        self.assertEqual(downstream[("p02", "integrated")]["gap"], 2.0)
        metric = {row["metric_output_id"]: row for row in result["metric_rows"]}
        self.assertAlmostEqual(metric["mse"]["ag_a"], 0.8, places=15)
        self.assertAlmostEqual(metric["candidate"]["ag_a"], 0.0, places=15)
        self.assertAlmostEqual(metric["mse"]["ag_b"], 0.0, places=15)
        self.assertAlmostEqual(metric["candidate"]["ag_b"], 0.5, places=15)
        self.assertAlmostEqual(metric["mse"]["delta_ag"], -0.8, places=15)
        self.assertAlmostEqual(metric["candidate"]["delta_ag"], 0.5, places=15)
        self.assertAlmostEqual(metric["candidate"]["i_ag"], -1.3, places=15)
        self.assertAlmostEqual(metric["mse"]["acc_a"], 0.5, places=15)
        self.assertAlmostEqual(metric["candidate"]["acc_a"], 1.0, places=15)
        self.assertAlmostEqual(metric["mse"]["acc_b"], 0.75, places=15)
        self.assertAlmostEqual(metric["candidate"]["acc_b"], 0.5, places=15)
        self.assertAlmostEqual(metric["mse"]["delta_acc"], 0.25, places=15)
        self.assertAlmostEqual(metric["candidate"]["delta_acc"], -0.5, places=15)
        self.assertAlmostEqual(metric["candidate"]["i_acc"], -0.75, places=15)
        intervals = {
            (row["statistic"], row.get("metric_output_id"), row.get("perturbation_id")): row
            for row in result["bootstrap_rows"]
        }
        self.assertEqual(intervals[("g", None, "p02")]["interval"], [2.0, 2.0])
        self.assertEqual(intervals[("delta_ag", "mse", None)]["interval"], [-0.8, -0.8])
        self.assertEqual(intervals[("i_ag", "candidate", None)]["interval"], [-1.3, -1.3])
        self.assertEqual(intervals[("delta_acc", "mse", None)]["interval"], [0.25, 0.25])
        self.assertEqual(intervals[("i_acc", "candidate", None)]["interval"], [-0.75, -0.75])

    def test_holm_uses_full_slot_count_and_preserves_frozen_order(self) -> None:
        slots = [(f"slot-{index:02d}", 1.0) for index in range(29)]
        slots[0] = ("slot-00", 0.001)
        slots[1] = ("slot-01", 0.05)
        rows = build_protocol_ab_holm("cell", slots)
        self.assertEqual([row["slot_id"] for row in rows], [item[0] for item in slots])
        self.assertTrue(all(row["family_size"] == 29 for row in rows))
        self.assertEqual(rows[0]["adjusted_p_value"], 0.029)
        self.assertTrue(rows[0]["rejected"])
        self.assertEqual(rows[1]["adjusted_p_value"], 1.0)
        self.assertFalse(rows[1]["rejected"])

    def test_inference_rows_follow_downstream_then_ag_then_acc_order(self) -> None:
        base = _hand_cell()
        cell = replace(
            base,
            metric_output_ids=("mse", "candidate", "candidate_2"),
            metric_harm=np.concatenate((base.metric_harm, base.metric_harm[[1]]), axis=0),
        )
        result = compute_protocol_ab_cell(
            cell, bootstrap_resamples=1, sign_flip_resamples=1, random_seed=20260817
        )
        bootstrap_order = [
            (row["statistic"], row.get("metric_output_id"))
            for row in result["bootstrap_rows"]
            if row["statistic"] != "gap" and row["statistic"] != "g"
        ]
        self.assertEqual(bootstrap_order, [
            ("delta_ag", "mse"), ("delta_ag", "candidate"),
            ("delta_ag", "candidate_2"), ("delta_acc", "mse"),
            ("delta_acc", "candidate"), ("delta_acc", "candidate_2"),
            ("i_ag", "candidate"), ("i_ag", "candidate_2"),
            ("i_acc", "candidate"), ("i_acc", "candidate_2"),
        ])
        self.assertEqual([row["slot_id"] for row in result["sign_flip_rows"]], [
            "downstream:p01", "downstream:p02",
            "i_ag:candidate", "i_ag:candidate_2",
            "i_acc:candidate", "i_acc:candidate_2",
        ])


class ProtocolABAdapterTest(unittest.TestCase):
    def _rows(self, *, state: object = "missing", offset: float = 0.0):
        rows = []
        for metric_index, metric_id in enumerate(("mse", "candidate")):
            for perturbation_index, perturbation in enumerate(("p01", "p02")):
                for alpha in (0.1, 0.2):
                    for cluster in (0, 1):
                        row = {
                            "metric_output_id": metric_id,
                            "perturbation_id": perturbation,
                            "alpha": alpha,
                            "class_label": cluster,
                            "metric_harm": float(metric_index + perturbation_index + alpha) + offset,
                            "downstream_harm": float(perturbation_index + alpha),
                            "shot_count": 5,
                        }
                        if state != "missing":
                            row["state"] = state
                        rows.append(row)
        return rows

    def test_legacy_d2_state_requires_complete_parent_and_exact_grid(self) -> None:
        adapted = adapt_protocol_observation_rows(
            cell_id="d2_5shot",
            rows_a=self._rows(),
            rows_b=self._rows(),
            cluster_field="class_label",
            cluster_kind="class_label",
            metric_output_ids=("mse", "candidate"),
            perturbation_ids=("p01", "p02"),
            alpha_grid=(0.1, 0.2),
            parents_complete=True,
            legacy_missing_state=True,
            use_protocol_a_metric=True,
        )
        self.assertEqual(adapted.cluster_ids, ("0", "1"))
        self.assertEqual(adapted.metric_reconciliation["mismatch_count"], 0)
        with self.assertRaisesRegex(Phase4ProtocolABContrastError, "legacy state"):
            adapt_protocol_observation_rows(
                cell_id="d2_5shot",
                rows_a=self._rows(), rows_b=self._rows(),
                cluster_field="class_label", cluster_kind="class_label",
                metric_output_ids=("mse", "candidate"),
                perturbation_ids=("p01", "p02"), alpha_grid=(0.1, 0.2),
                parents_complete=False, legacy_missing_state=True,
                use_protocol_a_metric=True,
            )

    def test_d4_uses_protocol_a_metric_and_records_exact_reconciliation(self) -> None:
        rows_a = self._rows(state="complete")
        rows_b = self._rows(state="complete", offset=math.ulp(1.0))
        adapted = adapt_protocol_observation_rows(
            cell_id="d4_full_domain_core",
            rows_a=rows_a,
            rows_b=rows_b,
            cluster_field="class_label",
            cluster_kind="physical_well",
            metric_output_ids=("mse", "candidate"),
            perturbation_ids=("p01", "p02"), alpha_grid=(0.1, 0.2),
            parents_complete=True, legacy_missing_state=False,
            use_protocol_a_metric=True,
        )
        self.assertTrue(np.array_equal(adapted.metric_harm[:, 0, 0, 0], np.asarray([0.1, 1.1])))
        self.assertGreater(adapted.metric_reconciliation["mismatch_count"], 0)
        self.assertGreater(adapted.metric_reconciliation["max_abs_difference"], 0.0)
        rows_a[0]["acquisition_count"] = 32
        rows_b[0]["acquisition_count"] = 31
        with self.assertRaisesRegex(Phase4ProtocolABContrastError, "within-cluster count"):
            adapt_protocol_observation_rows(
                cell_id="d4_full_domain_core", rows_a=rows_a, rows_b=rows_b,
                cluster_field="class_label", cluster_kind="physical_well",
                count_field="acquisition_count",
                metric_output_ids=("mse", "candidate"),
                perturbation_ids=("p01", "p02"), alpha_grid=(0.1, 0.2),
                parents_complete=True, legacy_missing_state=False,
                use_protocol_a_metric=True,
            )

    def test_parent_admission_pins_directory_manifest_terminal_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "historical-directory-id"
            parent.mkdir()
            manifest = {"run_id": "different-manifest-id", "status": "complete"}
            terminal = {"run_id": "different-manifest-id", "status": "complete"}
            payloads = {
                "manifest.json": (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
                "complete.json": (json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n").encode(),
                "observations.jsonl": b"{\"value\":1}\n",
            }
            for name, raw in payloads.items():
                (parent / name).write_bytes(raw)
            ledger = "".join(f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in payloads)
            (parent / "SHA256SUMS").write_text(ledger)
            expected = {
                "directory_name": "historical-directory-id",
                "manifest_run_id": "different-manifest-id",
                "terminal_name": "complete.json",
                "terminal_run_id": "different-manifest-id",
                "status": "complete",
                "sha256sums_sha256": hashlib.sha256(ledger.encode()).hexdigest(),
                "required_inventory": sorted([*payloads, "SHA256SUMS"]),
            }
            receipt = validate_parent_artifact(parent, expected)
            self.assertEqual(receipt["directory_name"], "historical-directory-id")
            self.assertEqual(receipt["manifest_run_id"], "different-manifest-id")
            (parent / "observations.jsonl").write_text('{"value":2}\n')
            tampered = dict(payloads)
            tampered["observations.jsonl"] = b'{"value":2}\n'
            new_ledger = "".join(f"{hashlib.sha256(tampered[name]).hexdigest()}  {name}\n" for name in tampered)
            (parent / "SHA256SUMS").write_text(new_ledger)
            with self.assertRaisesRegex(Phase4ProtocolABContrastError, "ledger identity"):
                validate_parent_artifact(parent, expected)


class ProtocolABArtifactTest(unittest.TestCase):
    def test_statistical_failure_emits_fixed_closed_rows_and_failed_marker(self) -> None:
        config = make_synthetic_protocol_ab_contrast_config()
        inputs = make_synthetic_protocol_ab_contrast_inputs()
        broken = replace(inputs.cells[0], downstream_b=np.zeros_like(inputs.cells[0].downstream_b))
        failed_inputs = replace(inputs, cells=(broken, *inputs.cells[1:]))
        with tempfile.TemporaryDirectory() as directory:
            summary = build_phase4_protocol_ab_contrast_from_inputs(
                Path(directory), inputs=failed_inputs, config=config, worker_count=1,
                bootstrap_resamples=2, sign_flip_resamples=4,
            )
            self.assertEqual(summary.status, "failed")
            self.assertTrue((summary.path / "failed.json").is_file())
            self.assertFalse((summary.path / "complete.json").exists())
            manifest = json.loads((summary.path / "manifest.json").read_bytes())
            self.assertEqual(manifest["counts"], {
                "artifact_files": 14, "bootstrap_results": 60,
                "configured_payloads": 12, "downstream_protocol_effects": 30,
                "endpoint_status": 6, "holm_family": 20,
                "metric_protocol_effects": 10, "sign_flip_results": 20,
            })
            self.assertIn("failed_alignment_recomputation", (summary.path / "failed.json").read_text())
            self.assertIn("not_tested_endpoint_closed", (summary.path / "holm_family.jsonl").read_text())

    def test_real_config_and_parent_reconstruction_match_frozen_contract(self) -> None:
        config = load_phase4_protocol_ab_contrast_config()
        self.assertEqual(config.cell_ids, (
            "d5_full_domain_core", "d2_5shot", "d2_10shot",
            "d2_20shot", "d1_full_domain_core", "d4_full_domain_core",
        ))
        self.assertEqual(config.metric_output_ids[0], "mse")
        self.assertEqual(config.perturbation_ids, ("p08", "p09", "p10", "p11", "p12"))
        self.assertEqual(config.document["expected"]["paired_observation_row_count"], 525720)
        inputs = reconstruct_phase4_protocol_ab_contrast_inputs(config)
        self.assertEqual(tuple(cell.cell_id for cell in inputs.cells), (
            "d5_full_domain_core", "d2_5shot", "d2_10shot",
            "d2_20shot", "d4_full_domain_core",
        ))
        self.assertEqual([cell.metric_harm.shape for cell in inputs.cells], [
            (13, 681, 5, 8), (13, 30, 5, 8), (13, 30, 5, 8),
            (13, 30, 5, 8), (13, 240, 5, 8),
        ])
        self.assertEqual(inputs.preflight["paired_observation_row_count"], 525720)
        self.assertEqual(inputs.endpoint_status_rows[4]["state"], "not_evaluable_failed_alpha0_equivalence")
        document = dict(config.document)
        document["alpha_grid"] = list(config.alpha_grid[:-1])
        raw = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self.assertRaisesRegex(Phase4ProtocolABContrastError, "frozen identity"):
            parse_phase4_protocol_ab_contrast_config(config.path, raw)

    def test_synthetic_artifact_has_fixed_inventory_d1_closure_and_worker_independence(self) -> None:
        config = make_synthetic_protocol_ab_contrast_config()
        inputs = make_synthetic_protocol_ab_contrast_inputs()
        snapshots = []
        for workers in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                summary = build_phase4_protocol_ab_contrast_from_inputs(
                    Path(directory), inputs=inputs, config=config,
                    worker_count=workers, bootstrap_resamples=8,
                    sign_flip_resamples=32,
                )
                files = {p.name: p.read_bytes() for p in summary.path.iterdir() if p.is_file()}
                self.assertEqual(set(files), set(EXPECTED_PAYLOAD_FILES) | {"complete.json", "SHA256SUMS"})
                statuses = [json.loads(line) for line in files["endpoint_status.jsonl"].splitlines()]
                d1 = [row for row in statuses if row["cell_id"] == "d1_full_domain_core"]
                self.assertEqual(len(d1), 1)
                self.assertEqual(d1[0]["state"], "not_evaluable_failed_alpha0_equivalence")
                self.assertNotIn("d1_full_domain_core", files["sign_flip_results.jsonl"].decode())
                ledger = files["SHA256SUMS"].decode().splitlines()
                self.assertEqual(len(ledger), 13)
                for line in ledger:
                    digest, name = line.split("  ", 1)
                    self.assertEqual(hashlib.sha256(files[name]).hexdigest(), digest)
                snapshots.append((summary.run_id, files))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(ARTIFACT_PAYLOAD_FILES, EXPECTED_PAYLOAD_FILES)


class ProtocolABVerifierSourceTest(unittest.TestCase):
    def test_verifier_matches_non_degenerate_production_rows_exactly(self) -> None:
        from rpe.runner.phase4_protocol_ab_contrast_verifier import _recompute_cell

        generator = np.random.default_rng(1)
        cell = ProtocolABCellInput(
            cell_id="non_degenerate",
            cluster_kind="class_label",
            cluster_ids=("c1", "c2", "c3", "c4"),
            metric_output_ids=("mse", "candidate", "candidate_2"),
            perturbation_ids=("p01", "p02"),
            alpha_grid=(0.1, 0.2, 0.3),
            metric_harm=generator.normal(size=(3, 4, 2, 3)),
            downstream_a=generator.normal(size=(4, 2, 3)),
            downstream_b=generator.normal(size=(4, 2, 3)),
            metric_reconciliation={
                "mismatch_count": 0, "max_abs_difference": 0.0,
            },
        )
        production = compute_protocol_ab_cell(
            cell, bootstrap_resamples=8, sign_flip_resamples=32,
            random_seed=20260817,
        )
        rebuilt = _recompute_cell(
            cell, bootstrap_resamples=8, sign_flip_resamples=32,
            seed=20260817,
        )
        self.assertEqual(dict(rebuilt), dict(production))

    def test_verifier_interval_matches_production_tail_arithmetic_exactly(self) -> None:
        from rpe.runner import phase4_protocol_ab_contrast as production
        from rpe.runner.phase4_protocol_ab_contrast_verifier import (
            _percentile_interval,
        )

        values = np.arange(8, dtype=np.float64)
        self.assertEqual(_percentile_interval(values), production._interval(values))

    def test_verifier_does_not_import_production_module(self) -> None:
        source = (ROOT / "rpe/runner/phase4_protocol_ab_contrast_verifier.py").read_text()
        tree = ast.parse(source)
        forbidden = "rpe.runner.phase4_protocol_ab_contrast"
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertNotIn(forbidden, imported)

    def test_verifier_rebuilds_and_rejects_checksum_consistent_tampering(self) -> None:
        from rpe.runner.phase4_protocol_ab_contrast_verifier import (
            Phase4ProtocolABContrastVerifierError,
            verify_phase4_protocol_ab_contrast_from_inputs,
        )

        config = make_synthetic_protocol_ab_contrast_config()
        inputs = make_synthetic_protocol_ab_contrast_inputs()
        with tempfile.TemporaryDirectory() as directory:
            summary = build_phase4_protocol_ab_contrast_from_inputs(
                Path(directory), inputs=inputs, config=config, worker_count=1,
                bootstrap_resamples=8, sign_flip_resamples=32,
            )
            verified = verify_phase4_protocol_ab_contrast_from_inputs(
                summary.path, inputs=inputs, config_path=config.path,
                config_bytes=config.raw_bytes, worker_count=2,
                bootstrap_resamples=8, sign_flip_resamples=32,
            )
            self.assertEqual(verified.run_id, summary.run_id)
            self.assertEqual(verified.holm_slot_count, 20)
            target = summary.path / "endpoint_status.jsonl"
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[0]["cluster_count"] = 999
            target.write_bytes(b"".join(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for row in rows
            ))
            ledger_rows = []
            for name in EXPECTED_PAYLOAD_FILES:
                ledger_rows.append(f"{hashlib.sha256((summary.path / name).read_bytes()).hexdigest()}  {name}")
            ledger_rows.append(f"{hashlib.sha256((summary.path / 'complete.json').read_bytes()).hexdigest()}  complete.json")
            (summary.path / "SHA256SUMS").write_text("\n".join(ledger_rows) + "\n")
            with self.assertRaisesRegex(Phase4ProtocolABContrastVerifierError, "byte mismatch"):
                verify_phase4_protocol_ab_contrast_from_inputs(
                    summary.path, inputs=inputs, config_path=config.path,
                    config_bytes=config.raw_bytes, worker_count=2,
                    bootstrap_resamples=8, sign_flip_resamples=32,
                )

    def test_cli_build_surface_is_compact_and_does_not_eagerly_import_verifier(self) -> None:
        cli_path = ROOT / "tools/run_phase4_protocol_ab_contrast.py"
        spec = importlib.util.spec_from_file_location("step35_cli_test", cli_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        summary = type(
            "Summary", (),
            {"path": Path("out/run"), "run_id": "run-id",
             "status": "complete", "endpoint_status_count": 6,
             "bootstrap_result_count": 475, "holm_slot_count": 145},
        )()
        with mock.patch.object(module, "build_phase4_protocol_ab_contrast", return_value=summary) as build:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = module.main(["build", "--output-root", "out", "--worker-count", "2"])
        self.assertEqual(code, 0)
        build.assert_called_once_with(Path("out"), worker_count=2)
        self.assertEqual(json.loads(output.getvalue()), {
            "bootstrap_result_count": 475, "endpoint_status_count": 6,
            "holm_slot_count": 145, "path": "out/run",
            "run_id": "run-id", "status": "complete",
        })
        source = cli_path.read_text()
        tree = ast.parse(source)
        top_level_imports = [
            node.module for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        self.assertNotIn("rpe.runner.phase4_protocol_ab_contrast_verifier", top_level_imports)


if __name__ == "__main__":
    unittest.main()
