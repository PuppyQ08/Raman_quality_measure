from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.w1_axis_robustness import (  # noqa: E402
    NativePanelTable,
    W1AxisRobustnessConfig,
    W1AxisRobustnessInputs,
    W1AxisRobustnessError,
    build_inventory_only_inputs,
    build_coverage_status_rows,
    build_endpoint_seed_rows,
    build_w1_axis_robustness,
    build_w1_axis_robustness_from_inputs,
    compute_baseline_reproduction_rows,
    compute_native_point_estimates,
    condition_observation_count_per_cluster,
    condition_perturbation_ids,
    count_oc_family_pairs,
    derive_endpoint_seed,
    load_w1_axis_robustness_config,
    summarize_oc_pair_decomposition,
    summarize_task_harm,
)
import rpe.runner.w1_axis_robustness as w1_axis  # noqa: E402


def _synthetic_config() -> W1AxisRobustnessConfig:
    return W1AxisRobustnessConfig(
        path=Path("synthetic-w1-axis-config.json"),
        document={
            "schema_version": "w1-axis-robustness-config-v1",
            "experiment_id": "synthetic-w1-axis",
        },
        metric_output_ids=("mse", "candidate"),
        perturbation_ids=("p08", "p09", "p10", "p11", "p12"),
        no_axis_perturbation_ids=("p08", "p09", "p10"),
        alpha_grid=(0.05, 0.10),
        paper_alignment_csv=Path("synthetic-paper.csv"),
        baseline_atol=1e-10,
        baseline_rtol=1e-8,
        bootstrap_resamples=2000,
        sign_flip_resamples=100000,
        artifact_root=ROOT,
        default_output_root=ROOT / "results" / "synthetic_w1_axis",
        parent_phase4_protocol_ab_contrast_config=ROOT / "experiments/phase4/configs/protocol_ab_contrast_v1.json",
        panels=(
            {
                "panel_id": "panel_a",
                "endpoint_id": "endpoint",
                "protocol_id": "a",
                "parent_key": "panel_a",
                "parent_config_path": "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json",
                "cluster_field": "cluster_id",
                "cluster_kind": "synthetic_cluster",
                "observation_file": "class_observations.jsonl",
                "count_field": None,
                "shot_count": None,
                "legacy_missing_state": False,
            },
        ),
    )


def _synthetic_table() -> NativePanelTable:
    metric_harm = np.asarray(
        [
            [
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            ],
            [
                [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]],
                [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]],
            ],
        ],
        dtype=np.float64,
    )
    downstream_harm = np.asarray(
        [
            [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]],
            [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]],
        ],
        dtype=np.float64,
    )
    return NativePanelTable(
        panel_id="panel_a",
        endpoint_id="endpoint",
        protocol_id="a",
        parent_key="panel_a",
        cluster_kind="synthetic_cluster",
        cluster_ids=("c0", "c1"),
        metric_output_ids=("mse", "candidate"),
        perturbation_ids=("p08", "p09", "p10", "p11", "p12"),
        alpha_grid=(0.05, 0.10),
        metric_harm=metric_harm,
        downstream_harm=downstream_harm,
        within_cluster_counts=None,
        parent_status="complete",
        common_grid_ready=True,
    )


class W1AxisRobustnessConfigTest(unittest.TestCase):
    def test_load_real_config_freezes_11_panels_and_13_metrics(self) -> None:
        config = load_w1_axis_robustness_config()

        self.assertEqual(config.metric_output_ids[0], "mse")
        self.assertEqual(len(config.metric_output_ids), 13)
        self.assertEqual(config.perturbation_ids, ("p08", "p09", "p10", "p11", "p12"))
        self.assertEqual(config.no_axis_perturbation_ids, ("p08", "p09", "p10"))
        self.assertEqual(len(config.alpha_grid), 8)
        self.assertEqual(len(config.panels), 11)
        self.assertTrue(config.paper_alignment_csv.is_file())
        self.assertEqual(
            config.default_output_root,
            ROOT / "results/robustness/w1_axis_v1",
        )
        self.assertEqual(
            tuple(panel["panel_id"] for panel in config.panels),
            (
                "d1_a",
                "d2_5_a",
                "d2_5_b",
                "d2_10_a",
                "d2_10_b",
                "d2_20_a",
                "d2_20_b",
                "d4_a",
                "d4_b",
                "d5_a",
                "d5_b",
            ),
        )
        self.assertNotIn("d1_b", {panel["panel_id"] for panel in config.panels})

    def test_endpoint_seed_matches_frozen_literals(self) -> None:
        self.assertEqual(derive_endpoint_seed("d5", "bootstrap"), 935402821026910069)
        self.assertEqual(derive_endpoint_seed("d5", "sign_flip"), 10523723992761282615)
        self.assertEqual(derive_endpoint_seed("d2_10", "bootstrap"), 10586182501079144328)
        self.assertEqual(derive_endpoint_seed("d4", "bootstrap"), 182137638545738873)

    def test_endpoint_seed_rejects_unknown_operation_and_protocol_suffixed_endpoint(self) -> None:
        with self.assertRaisesRegex(W1AxisRobustnessError, "operation_id"):
            derive_endpoint_seed("d5", "other")
        with self.assertRaisesRegex(W1AxisRobustnessError, "endpoint_id"):
            derive_endpoint_seed("d5_a", "bootstrap")
        with self.assertRaisesRegex(W1AxisRobustnessError, "endpoint_id"):
            derive_endpoint_seed("d2_10_b", "sign_flip")


class W1AxisRobustnessConditionTest(unittest.TestCase):
    def test_native_no_axis_filters_p11_and_p12_and_counts_are_exact(self) -> None:
        config = load_w1_axis_robustness_config()
        table = _synthetic_table()

        self.assertEqual(condition_perturbation_ids(config, "native_all5"), ("p08", "p09", "p10", "p11", "p12"))
        self.assertEqual(condition_perturbation_ids(config, "native_no_axis"), ("p08", "p09", "p10"))
        self.assertEqual(condition_observation_count_per_cluster(table, config, "native_all5"), 10)
        self.assertEqual(condition_observation_count_per_cluster(table, config, "native_no_axis"), 6)
        self.assertEqual(count_oc_family_pairs(("p08", "p09", "p10", "p11", "p12"), tuple(range(8))), 640)
        self.assertEqual(count_oc_family_pairs(("p08", "p09", "p10"), tuple(range(8))), 192)


class W1AxisRobustnessDecompositionTest(unittest.TestCase):
    def test_oc_pair_decomposition_uses_mutually_exclusive_counts(self) -> None:
        summary = summarize_oc_pair_decomposition(
            strict_agreement_count=3,
            strict_disagreement_count=1,
            metric_tie_count=2,
            downstream_tie_count=3,
            double_tie_count=1,
        )

        self.assertEqual(summary["pair_count"], 10)
        self.assertAlmostEqual(summary["agreement_fraction"], 0.30, places=12)
        self.assertAlmostEqual(summary["tie_fraction"], 0.60, places=12)
        self.assertAlmostEqual(summary["disagreement_fraction"], 0.10, places=12)
        self.assertAlmostEqual(summary["accuracy"], 0.60, places=12)

    def test_task_harm_summary_preserves_positive_negative_cancellation(self) -> None:
        summary = summarize_task_harm([1.0, -1.0, 2.0, -2.0, 0.0])

        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["positive_count"], 2)
        self.assertEqual(summary["negative_count"], 2)
        self.assertEqual(summary["zero_count"], 1)
        self.assertAlmostEqual(summary["signed_sum"], 0.0, places=12)
        self.assertAlmostEqual(summary["positive_sum"], 3.0, places=12)
        self.assertAlmostEqual(summary["negative_sum"], -3.0, places=12)
        self.assertAlmostEqual(summary["absolute_sum"], 6.0, places=12)


class W1AxisRobustnessNativePointEstimateTest(unittest.TestCase):
    def test_constant_metric_harm_has_zero_alignment_gap_and_family_sum_matches_total(self) -> None:
        rows = compute_native_point_estimates(_synthetic_table(), _synthetic_config(), "native_all5")
        by_metric = {row["metric_output_id"]: row for row in rows["point_rows"]}
        details = {
            (row["metric_output_id"], row["cluster_id"]): row
            for row in rows["ag_detail_rows"]
        }

        self.assertAlmostEqual(by_metric["mse"]["ag"], 0.0, places=12)
        self.assertAlmostEqual(by_metric["mse"]["acc_cross"], 0.5, places=12)
        candidate_sum = sum(
            row["ag_cluster_contribution"]
            for row in rows["ag_detail_rows"]
            if row["metric_output_id"] == "candidate"
        )
        self.assertAlmostEqual(candidate_sum, by_metric["candidate"]["ag"], places=12)
        self.assertGreaterEqual(details[("candidate", "c0")]["ag_cluster_contribution"], 0.0)
        self.assertGreaterEqual(details[("candidate", "c1")]["ag_cluster_contribution"], 0.0)


class W1AxisRobustnessBaselineReproductionTest(unittest.TestCase):
    def test_compact_fixture_preserves_missing_rows_and_status(self) -> None:
        config = _synthetic_config()
        rows = compute_native_point_estimates(_synthetic_table(), config, "native_all5")
        paper_rows = (
            {
                "panel_id": "panel_a",
                "metric_output_id": "mse",
                "ag": str(rows["point_rows"][0]["ag"]),
                "acc_cross": str(rows["point_rows"][0]["acc_cross"]),
                "d_ag": "",
                "d_acc": "",
                "panel_state": "complete",
            },
        )

        reproduced = compute_baseline_reproduction_rows(
            point_rows=rows["point_rows"],
            paper_rows=paper_rows,
            atol=1e-10,
            rtol=1e-8,
        )
        by_metric = {row["metric_output_id"]: row for row in reproduced}

        self.assertEqual(len(reproduced), 2)
        self.assertEqual(by_metric["mse"]["status"], "matched")
        self.assertEqual(by_metric["mse"]["paper_panel_state"], "complete")
        self.assertEqual(by_metric["candidate"]["status"], "missing_reference_row")
        self.assertEqual(by_metric["candidate"]["paper_panel_state"], "missing_reference_row")


class W1AxisRobustnessCoverageTest(unittest.TestCase):
    def test_coverage_rows_expand_to_44_and_track_native_vs_common_grid(self) -> None:
        config = load_w1_axis_robustness_config()
        inventory = {
            panel["panel_id"]: {
                "native_ready": True,
                "native_status": "ready",
                "common_grid_ready": panel["panel_id"] != "d1_a",
                "common_grid_status": "ready_not_built" if panel["panel_id"] != "d1_a" else "missing_support_grid",
                "parent_status": "complete",
            }
            for panel in config.panels
        }

        rows = build_coverage_status_rows(config, inventory)

        self.assertEqual(len(rows), 44)
        d1_common = [
            row for row in rows
            if row["panel_id"] == "d1_a" and row["condition_id"].startswith("common_grid")
        ]
        self.assertEqual(len(d1_common), 2)
        self.assertTrue(all(row["ready"] is False for row in d1_common))
        self.assertTrue(all(row["status"] == "missing_support_grid" for row in d1_common))
        native = [
            row for row in rows
            if row["condition_id"].startswith("native_")
        ]
        self.assertTrue(all(row["ready"] is True for row in native))

    def test_common_grid_requires_support_grid_and_complete_parent(self) -> None:
        config = load_w1_axis_robustness_config()
        inventory = {
            panel["panel_id"]: {
                "native_ready": panel["panel_id"] != "d4_a",
                "native_status": "ready" if panel["panel_id"] != "d4_a" else "parent_failed",
                "common_grid_ready": False if panel["panel_id"] == "d4_a" else panel["panel_id"] != "d1_a",
                "common_grid_status": (
                    "parent_failed"
                    if panel["panel_id"] == "d4_a"
                    else "ready_not_built"
                    if panel["panel_id"] != "d1_a"
                    else "missing_support_grid"
                ),
                "parent_status": "failed" if panel["panel_id"] == "d4_a" else "complete",
            }
            for panel in config.panels
        }

        rows = build_coverage_status_rows(config, inventory)
        d4_common = [
            row for row in rows
            if row["panel_id"] == "d4_a" and row["condition_id"].startswith("common_grid")
        ]

        self.assertEqual(len(d4_common), 2)
        self.assertTrue(all(row["ready"] is False for row in d4_common))
        self.assertTrue(all(row["status"] == "parent_failed" for row in d4_common))

    def test_inventory_marks_d1_a_common_grid_ready_via_shared_bacteria_support(self) -> None:
        config = load_w1_axis_robustness_config()

        inputs = build_inventory_only_inputs(config)

        d1 = inputs.inventory_document["panels"]["d1_a"]
        d1_common = [
            row for row in inputs.coverage_rows
            if row["panel_id"] == "d1_a" and row["condition_id"].startswith("common_grid")
        ]

        self.assertTrue(d1["common_grid_ready"])
        self.assertEqual(d1["common_grid_status"], "ready_not_built")
        self.assertEqual(len(d1_common), 2)
        self.assertTrue(all(row["ready"] is True for row in d1_common))
        self.assertTrue(all(row["status"] == "ready_not_built" for row in d1_common))


class W1AxisRobustnessBuildTest(unittest.TestCase):
    def _inputs(self) -> W1AxisRobustnessInputs:
        config = _synthetic_config()
        table = _synthetic_table()
        point_rows = compute_native_point_estimates(table, config, "native_all5")["point_rows"]
        paper_rows = tuple(
            {
                "panel_id": row["panel_id"],
                "metric_output_id": row["metric_output_id"],
                "ag": repr(row["ag"]),
                "acc_cross": repr(row["acc_cross"]),
                "d_ag": "" if row["d_ag"] is None else repr(row["d_ag"]),
                "d_acc": "" if row["d_acc"] is None else repr(row["d_acc"]),
                "panel_state": "complete",
            }
            for row in point_rows
        )
        return W1AxisRobustnessInputs(
            inventory_document={
                "schema_version": "w1-axis-robustness-inventory-v1",
                "parents": {"panel_a": {"status": "complete"}},
            },
            coverage_rows=(
                {
                    "panel_id": "panel_a",
                    "endpoint_id": "endpoint",
                    "protocol_id": "a",
                    "condition_id": "native_all5",
                    "family": "native",
                    "ready": True,
                    "status": "ready",
                    "parent_status": "complete",
                },
            ),
            tables=(table,),
            paper_rows=paper_rows,
        )

    def test_build_from_inputs_writes_marker_only_after_successful_publish(self) -> None:
        config = _synthetic_config()
        inputs = self._inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            summary = build_w1_axis_robustness_from_inputs(
                output_root,
                config=config,
                inputs=inputs,
                stage="inventory",
                resume=False,
            )

            self.assertTrue(summary.path.is_dir())
            self.assertTrue((summary.path / "inventory.complete.json").is_file())
            self.assertTrue((summary.path / "input_inventory.json").is_file())
            self.assertTrue((summary.path / "coverage_status.csv").is_file())
            self.assertFalse(any(".staging-" in item.name for item in output_root.iterdir()))

    def test_failed_publish_leaves_no_run_path_or_stage_marker(self) -> None:
        config = _synthetic_config()
        inputs = self._inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with mock.patch(
                "rpe.runner.w1_axis_robustness._write_stage_payloads",
                side_effect=OSError("injected stage failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected stage failure"):
                    build_w1_axis_robustness_from_inputs(
                        output_root,
                        config=config,
                        inputs=inputs,
                        stage="inventory",
                        resume=False,
                    )
            self.assertEqual(list(output_root.iterdir()), [])


class W1AxisRobustnessInventoryOnlyTest(unittest.TestCase):
    def test_inventory_only_input_build_does_not_require_native_tables_or_paper_csv(self) -> None:
        config = load_w1_axis_robustness_config()

        with mock.patch(
            "rpe.runner.w1_axis_robustness._load_native_panel_table",
            side_effect=AssertionError("inventory stage must not load native tables"),
        ), mock.patch(
            "rpe.runner.w1_axis_robustness.load_paper_alignment_rows",
            side_effect=AssertionError("inventory stage must not load paper csv"),
        ):
            inputs = build_inventory_only_inputs(config)

        self.assertEqual(inputs.tables, ())
        self.assertEqual(inputs.paper_rows, ())
        self.assertEqual(len(inputs.coverage_rows), 44)

    def test_inventory_build_path_does_not_call_full_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with mock.patch(
                "rpe.runner.w1_axis_robustness.reconstruct_w1_axis_robustness_inputs",
                side_effect=AssertionError("inventory stage must not call full reconstruction"),
            ):
                summary = build_w1_axis_robustness(
                    stage="inventory",
                    output_root=output_root,
                )
            self.assertEqual(summary.status, "complete")
            self.assertTrue((summary.path / "inventory.complete.json").is_file())

    def test_inventory_manifest_records_unique_endpoint_seed_pairs(self) -> None:
        rows = build_endpoint_seed_rows(
            (
                {"endpoint_id": "d5"},
                {"endpoint_id": "d2_10"},
                {"endpoint_id": "d5"},
            )
        )
        by_key = {(row["endpoint_id"], row["operation_id"]): row["seed"] for row in rows}

        self.assertEqual(len(rows), 4)
        self.assertEqual(by_key[("d5", "bootstrap")], 935402821026910069)
        self.assertEqual(by_key[("d5", "sign_flip")], 10523723992761282615)
        self.assertEqual(by_key[("d2_10", "bootstrap")], 10586182501079144328)


class W1AxisRobustnessAllStageTest(unittest.TestCase):
    def test_partial_metric_panel_is_persisted_with_metric_completeness(self) -> None:
        """Stage payloads must preserve a usable complete metric subset for Task 3."""
        base_config = _synthetic_config()
        config = W1AxisRobustnessConfig(**{
            **base_config.__dict__,
            "panels": ({**dict(base_config.panels[0]), "panel_id": "d2_5_a"},),
        })
        base_inputs = W1AxisRobustnessBuildTest()._inputs()
        table = NativePanelTable(**{**base_inputs.tables[0].__dict__, "panel_id": "d2_5_a"})
        inputs = W1AxisRobustnessInputs(
            inventory_document=base_inputs.inventory_document,
            coverage_rows=base_inputs.coverage_rows + ({
                "panel_id": "d2_5_a", "endpoint_id": "endpoint", "protocol_id": "a",
                "condition_id": "common_grid_all5", "family": "common_grid",
                "ready": True, "status": "ready", "parent_status": "complete",
            },),
            tables=(table,), paper_rows=base_inputs.paper_rows,
        )
        module = __import__("rpe.runner.w1_axis_common_grid", fromlist=["CommonGridDatasetResult"])
        rows = tuple(
            {
                "cluster_id": cluster_id, "perturbation_id": perturbation_id, "alpha": alpha,
                "metric_output_id": "mse", "metric_harm": 0.25,
            }
            for cluster_id in ("c0", "c1")
            for perturbation_id in ("p08", "p09", "p10", "p11", "p12")
            for alpha in (0.05, 0.10)
        )
        result = module.CommonGridDatasetResult(
            dataset_id="bacteria", aggregated_rows=rows,
            failure_rows=({
                "record_id": "r0", "cluster_id": "c0", "perturbation_id": "p08", "alpha": 0.05,
                "metric_output_id": "candidate", "reason_code": "metric_failure:candidate:RuntimeError:injected",
            },),
            processed_record_count=2, planned_record_count=2, status="incomplete_grid", elapsed_seconds=0.5, cache_hit_count=0,
        )
        with mock.patch(
            "rpe.runner.w1_axis_common_grid.materialize_real_common_grid_dataset", return_value=object(),
        ), mock.patch(
            "rpe.runner.w1_axis_common_grid.evaluate_or_load_real_common_grid_dataset", return_value=result,
        ), mock.patch(
            "rpe.runner.w1_axis_common_grid.common_grid_numerical_code_identity", return_value=("c" * 64, {"kernel": "d" * 64}),
        ):
            _, payload = w1_axis.build_common_grid_resample_stage_payloads(
                config=config, inputs=inputs, resume=False, max_records_per_dataset=None, worker_count=1,
            )

        panel = payload["panel_tables"][0]
        self.assertEqual(panel["status"], "partial_metric_grid")
        self.assertEqual(panel["complete_metric_output_ids"], ["mse"])
        self.assertEqual(panel["incomplete_metric_output_ids"], ["candidate"])
        self.assertEqual(panel["table"]["metric_output_ids"], ["mse"])
        self.assertEqual(payload["panels"][0]["status"], "partial_metric_grid")

    def test_common_grid_stage_payload_persists_process_and_cache_receipts(self) -> None:
        """Stage serialization must retain process evidence, not only worker counts."""
        config = _synthetic_config()
        inputs = W1AxisRobustnessBuildTest()._inputs()
        module = __import__("rpe.runner.w1_axis_common_grid", fromlist=["CommonGridDatasetResult"])
        result = module.CommonGridDatasetResult(
            dataset_id="bacteria", aggregated_rows=(), failure_rows=(), processed_record_count=2,
            planned_record_count=2, status="complete", elapsed_seconds=0.5, cache_hit_count=1,
            requested_worker_count=16, effective_worker_count=2, p10_worker_capacity=4,
            worker_process_ids=(11, 22), worker_blas_thread_limits=(1,), cache_hit=True,
        )
        with mock.patch(
            "rpe.runner.w1_axis_common_grid.materialize_real_common_grid_dataset", return_value=object(),
        ), mock.patch(
            "rpe.runner.w1_axis_common_grid.evaluate_or_load_real_common_grid_dataset", return_value=result,
        ), mock.patch(
            "rpe.runner.w1_axis_common_grid.common_grid_numerical_code_identity", return_value=("c" * 64, {"kernel": "d" * 64}),
        ):
            _, payload = w1_axis.build_common_grid_resample_stage_payloads(
                config=config, inputs=inputs, resume=True, max_records_per_dataset=None, worker_count=16,
            )

        row = payload["dataset_results"]["bacteria"]
        self.assertTrue(row["cache_hit"])
        self.assertEqual(row["requested_worker_count"], 16)
        self.assertEqual(row["effective_worker_count"], 2)
        self.assertEqual(row["worker_process_ids"], [11, 22])
        self.assertEqual(row["worker_process_count"], 2)
        self.assertEqual(row["worker_blas_thread_limits"], [1])
        self.assertEqual(payload["numerical_code_sha256"], "c" * 64)

    def test_resample_stage_streams_each_dataset_and_persists_end_to_end_elapsed(self) -> None:
        """A later dataset must not load before the preceding one has been evaluated."""
        config = _synthetic_config()
        inputs = W1AxisRobustnessBuildTest()._inputs()
        module = __import__("rpe.runner.w1_axis_common_grid", fromlist=["CommonGridDatasetResult"])
        events = []

        def materialize(dataset_id, **_kwargs):
            events.append(("load", dataset_id))
            return dataset_id

        def evaluate(dataset, **_kwargs):
            events.append(("evaluate", dataset))
            return module.CommonGridDatasetResult(
                dataset_id=dataset, aggregated_rows=(), failure_rows=(), processed_record_count=2,
                planned_record_count=2, status="complete", elapsed_seconds=0.5, cache_hit_count=0,
            )

        with mock.patch(
            "rpe.runner.w1_axis_common_grid.materialize_real_common_grid_dataset", side_effect=materialize,
        ), mock.patch(
            "rpe.runner.w1_axis_common_grid.evaluate_or_load_real_common_grid_dataset", side_effect=evaluate,
        ), mock.patch(
            "rpe.runner.w1_axis_common_grid.common_grid_numerical_code_identity", return_value=("c" * 64, {"kernel": "d" * 64}),
        ), mock.patch.object(w1_axis.time, "monotonic", side_effect=(10.0, 20.0, 23.0, 30.0, 37.0, 40.0, 51.0, 60.0)):
            _, payload = w1_axis.build_common_grid_resample_stage_payloads(
                config=config, inputs=inputs, resume=False, max_records_per_dataset=None, worker_count=1,
            )

        self.assertEqual(events, [
            ("load", "bacteria"), ("evaluate", "bacteria"),
            ("load", "sugar"), ("evaluate", "sugar"),
            ("load", "d5"), ("evaluate", "d5"),
        ])
        self.assertEqual(payload["dataset_results"]["bacteria"]["end_to_end_elapsed_seconds"], 3.0)
        self.assertEqual(payload["dataset_results"]["sugar"]["end_to_end_elapsed_seconds"], 7.0)
        self.assertEqual(payload["dataset_results"]["d5"]["end_to_end_elapsed_seconds"], 11.0)
        self.assertEqual(payload["application_elapsed_seconds"], 50.0)
    def test_canary_and_full_run_ids_cannot_collide_or_false_resume(self) -> None:
        config = _synthetic_config()
        self.assertNotEqual(
            w1_axis._run_id(config, "resample", max_records_per_dataset=1),
            w1_axis._run_id(config, "resample", max_records_per_dataset=None),
        )
    def test_real_resample_stage_marks_canary_incomplete_from_materialized_records(self) -> None:
        config = _synthetic_config()
        inputs = W1AxisRobustnessBuildTest()._inputs()
        module = __import__("rpe.runner.w1_axis_common_grid", fromlist=["CommonGridDatasetResult"])
        canary = module.CommonGridDatasetResult(
            dataset_id="synthetic", aggregated_rows=(), failure_rows=(), processed_record_count=1,
            planned_record_count=2, status="canary_partial", elapsed_seconds=0.0, cache_hit_count=0,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with mock.patch(
                "rpe.runner.w1_axis_common_grid.materialize_real_common_grid_dataset",
                return_value=object(),
            ) as materializer:
                with mock.patch(
                    "rpe.runner.w1_axis_common_grid.evaluate_or_load_real_common_grid_dataset", return_value=canary,
                ):
                    summary = build_w1_axis_robustness_from_inputs(
                        output_root, config=config, inputs=inputs, stage="resample", resume=False,
                        max_records_per_dataset=1,
                    )
            self.assertEqual(summary.status, "incomplete")
            self.assertEqual(materializer.call_count, 3)
    def test_resample_stage_writes_complete_marker_from_common_grid_stage(self) -> None:
        config = _synthetic_config()
        inputs = W1AxisRobustnessBuildTest()._inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with mock.patch(
                "rpe.runner.w1_axis_robustness.build_common_grid_resample_stage_payloads",
                return_value=(
                    {
                        "common_grid_tables.json": json.dumps(
                            {"panel_count": 1, "complete_panel_count": 1},
                            sort_keys=True,
                        ).encode("utf-8")
                    },
                    {
                        "schema_version": "w1-axis-common-grid-stage-v1",
                        "panel_count": 1,
                        "complete_panel_count": 1,
                        "incomplete_panel_count": 0,
                    },
                ),
            ) as builder:
                summary = build_w1_axis_robustness_from_inputs(
                    output_root,
                    config=config,
                    inputs=inputs,
                    stage="resample",
                    resume=False,
                )

            marker = json.loads((summary.path / "resample.complete.json").read_text(encoding="utf-8"))
            manifest = json.loads((summary.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary.status, "complete")
            self.assertEqual(marker["status"], "complete")
            self.assertEqual(manifest["status"], "complete")
            self.assertIn("common_grid_tables.json", manifest["payload_files"])
            builder.assert_called_once_with(
                config=config,
                inputs=inputs,
                resume=False,
                max_records_per_dataset=None,
                worker_count=1,
            )

    def test_build_passes_canary_record_limit_to_resample_stage(self) -> None:
        config = _synthetic_config()
        inputs = W1AxisRobustnessBuildTest()._inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with mock.patch(
                "rpe.runner.w1_axis_robustness.build_common_grid_resample_stage_payloads",
                return_value=(
                    {
                        "common_grid_tables.json": json.dumps(
                            {"panel_count": 1, "complete_panel_count": 1},
                            sort_keys=True,
                        ).encode("utf-8")
                    },
                    {
                        "schema_version": "w1-axis-common-grid-stage-v1",
                        "panel_count": 1,
                        "complete_panel_count": 1,
                        "incomplete_panel_count": 0,
                    },
                ),
            ) as builder:
                summary = build_w1_axis_robustness_from_inputs(
                    output_root,
                    config=config,
                    inputs=inputs,
                    stage="resample",
                    resume=False,
                    max_records_per_dataset=1,
                )

            self.assertEqual(summary.status, "complete")
            builder.assert_called_once_with(
                config=config,
                inputs=inputs,
                resume=False,
                max_records_per_dataset=1,
                worker_count=1,
            )

    def test_all_stage_stops_with_explicit_incomplete_status(self) -> None:
        config = _synthetic_config()
        inputs = W1AxisRobustnessBuildTest()._inputs()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            with mock.patch(
                "rpe.runner.w1_axis_robustness.build_common_grid_resample_stage_payloads",
                return_value=(
                    {
                        "common_grid_tables.json": json.dumps(
                            {"panel_count": 1, "complete_panel_count": 1},
                            sort_keys=True,
                        ).encode("utf-8")
                    },
                    {
                        "schema_version": "w1-axis-common-grid-stage-v1",
                        "panel_count": 1,
                        "complete_panel_count": 1,
                        "incomplete_panel_count": 0,
                    },
                ),
            ):
                summary = build_w1_axis_robustness_from_inputs(
                    output_root,
                    config=config,
                    inputs=inputs,
                    stage="all",
                    resume=False,
                )

            marker = json.loads((summary.path / "all.incomplete.json").read_text(encoding="utf-8"))
            self.assertEqual(summary.status, "incomplete")
            self.assertEqual(marker["status"], "incomplete")
            self.assertEqual(marker["reason"], "common_grid_resample_and_summary_deferred")


if __name__ == "__main__":
    unittest.main()
