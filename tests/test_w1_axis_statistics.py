from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from rpe.runner.w1_axis_robustness import NativePanelTable
import tools.run_w1_axis_robustness as w1_cli


def _table() -> NativePanelTable:
    # Two clusters, three perturbation families, and one alpha make every
    # expected quantity hand-checkable while retaining a real paired layout.
    metric = np.asarray(
        [
            [[[0.0], [1.0], [3.0]], [[0.0], [2.0], [6.0]]],
            [[[0.0], [3.0], [1.0]], [[0.0], [6.0], [2.0]]],
        ],
        dtype=np.float64,
    )
    downstream = np.asarray(
        [[[0.0], [1.0], [3.0]], [[0.0], [2.0], [6.0]]], dtype=np.float64
    )
    return NativePanelTable(
        panel_id="panel_a", endpoint_id="endpoint", protocol_id="a",
        parent_key="panel_a", cluster_kind="fixture", cluster_ids=("c0", "c1"),
        metric_output_ids=("mse", "candidate"),
        perturbation_ids=("p08", "p09", "p10"), alpha_grid=(0.1,),
        metric_harm=metric, downstream_harm=downstream, within_cluster_counts=None,
        parent_status="complete", common_grid_ready=True,
    )


def _multi_cluster_table() -> NativePanelTable:
    """Five-family, two-alpha fixture with a value per coordinate."""
    clusters = 3
    families = ("p08", "p09", "p10", "p11", "p12")
    alpha_grid = (0.1, 0.2)
    metric = np.empty((2, clusters, len(families), len(alpha_grid)), dtype=np.float64)
    downstream = np.empty((clusters, len(families), len(alpha_grid)), dtype=np.float64)
    for metric_index in range(2):
        for cluster in range(clusters):
            for family in range(len(families)):
                for alpha in range(len(alpha_grid)):
                    metric[metric_index, cluster, family, alpha] = 1000 * metric_index + 100 * cluster + 10 * family + alpha
                    downstream[cluster, family, alpha] = 7 * cluster + 3 * family + alpha * (family + 1)
    return NativePanelTable(
        panel_id="multi", endpoint_id="multi_endpoint", protocol_id="a",
        parent_key="multi", cluster_kind="fixture", cluster_ids=("c0", "c1", "c2"),
        metric_output_ids=("mse", "candidate"), perturbation_ids=families, alpha_grid=alpha_grid,
        metric_harm=metric, downstream_harm=downstream, within_cluster_counts=None,
        parent_status="complete", common_grid_ready=True,
    )


def _sign_changing_alpha_table() -> NativePanelTable:
    table = _multi_cluster_table()
    downstream = np.zeros_like(table.downstream_harm)
    # Per-cluster alpha means cancel, but the transformed condition values do not.
    downstream[0, 0] = (4.0, -4.0)
    downstream[1, 0] = (-2.0, 2.0)
    return NativePanelTable(**{**table.__dict__, "downstream_harm": downstream})


def _harm_share_table() -> NativePanelTable:
    table = _multi_cluster_table()
    downstream = np.zeros_like(table.downstream_harm)
    # Family means by cluster are respectively
    # c0: [1, 3, 0, 2, -1], c1: [3, 1, 2, 0, -1].
    downstream[0, :, :] = np.asarray([1.0, 3.0, 0.0, 2.0, -1.0])[:, None]
    downstream[1, :, :] = np.asarray([3.0, 1.0, 2.0, 0.0, -1.0])[:, None]
    return NativePanelTable(**{**table.__dict__, "cluster_ids": ("2", "10", "1"), "downstream_harm": downstream})


class W1AxisStatisticsTest(unittest.TestCase):
    def test_cli_accepts_statistics_stages(self) -> None:
        self.assertEqual(w1_cli._parser().parse_args(["--stage", "summarize"]).stage, "summarize")
        self.assertEqual(w1_cli._parser().parse_args(["--stage", "infer", "--resume"]).stage, "infer")

    def test_analysis_grid_retains_an_unavailable_metric_slot(self) -> None:
        from rpe.runner.w1_axis_statistics import build_alignment_summary_rows

        rows = build_alignment_summary_rows(
            tables={"native": _table(), "common_grid": _table()},
            scopes={"all5": ("p08", "p09", "p11")},
            unavailable={("common_grid", "candidate")},
            bootstrap_draws={"endpoint": np.asarray([[0, 1], [1, 1]], dtype=np.int64)},
            bootstrap_resamples=2, sign_flip_resamples=10,
        )
        self.assertEqual(len(rows), 4)
        unavailable = next(row for row in rows if row["analysis_id"] == "common_grid_all5" and row["metric_output_id"] == "candidate")
        self.assertEqual(unavailable["ag_state"], "unavailable")
        self.assertEqual(unavailable["oc_state"], "unavailable")
        self.assertIsNone(unavailable["raw_p_ag"])
        self.assertEqual(unavailable["adjusted_p_ag"], 1.0)

    def test_oc_counts_are_exclusive_and_ties_score_one_half(self) -> None:
        from rpe.runner.w1_axis_statistics import oc_pair_counts

        result = oc_pair_counts(
            metric=np.asarray([0.0, 0.0, 1.0, 2.0]),
            downstream=np.asarray([0.0, 1.0, 1.0, 0.0]),
            families=("p08", "p09", "p10", "p11"),
            left_family="p08", right_family="p09",
        )
        self.assertEqual(result["pair_count"], 1)
        self.assertEqual(result["metric_tie_count"], 1)
        self.assertEqual(result["strict_agreement_count"], 0)
        self.assertEqual(result["accuracy"], 0.5)

    def test_task_harm_splits_before_averaging_and_cancels(self) -> None:
        from rpe.runner.w1_axis_statistics import task_harm_parts

        values = np.asarray([-2.0, 0.0, 4.0])
        result = task_harm_parts(values)
        self.assertEqual(result["signed_mean"], 2.0 / 3.0)
        self.assertEqual(result["positive_mean"], 4.0 / 3.0)
        self.assertEqual(result["negative_mean"], 2.0 / 3.0)
        self.assertEqual(result["absolute_mean"], 2.0)

    def test_task_harm_bootstrap_quantiles_are_computed_per_named_part(self) -> None:
        from rpe.runner.w1_axis_statistics import _task_rows

        rows = _task_rows(_table(), np.asarray([[0, 1], [1, 1]], dtype=np.int64))
        self.assertEqual(len(rows), 6)  # 3 families times alpha and integrated rows
        self.assertTrue(all("positive_mean_lower" in row for row in rows))

    def test_as_matrix_preserves_cluster_family_alpha_order(self) -> None:
        """Regression: advanced family indexing must not move the cluster axis."""
        from rpe.runner.w1_axis_statistics import _as_matrix

        table = _multi_cluster_table()
        for scope, families in (("all5", table.perturbation_ids), ("no_axis", table.perturbation_ids[:3])):
            metric, downstream, labels = _as_matrix(table, "candidate", scope)
            self.assertEqual(labels, tuple(family for family in families for _ in table.alpha_grid))
            expected_metric = np.asarray([
                [table.metric_harm[1, cluster, family, alpha] for family in range(len(families)) for alpha in range(len(table.alpha_grid))]
                for cluster in range(len(table.cluster_ids))
            ])
            expected_downstream = np.asarray([
                [table.downstream_harm[cluster, family, alpha] for family in range(len(families)) for alpha in range(len(table.alpha_grid))]
                for cluster in range(len(table.cluster_ids))
            ])
            np.testing.assert_array_equal(metric, expected_metric)
            np.testing.assert_array_equal(downstream, expected_downstream)

    def test_unit_cluster_weights_match_point_alignment_and_oc_for_both_scopes(self) -> None:
        """Regression: a full-weight bootstrap draw is exactly the point statistic."""
        from rpe.runner.w1_axis_statistics import _as_matrix, _bootstrap_pair, _point_alignment, _scope_positions, _weighted_gap

        table = _multi_cluster_table()
        draw = np.arange(len(table.cluster_ids), dtype=np.int64)[None, :]
        for scope in ("all5", "no_axis"):
            point = _point_alignment(table, "candidate", scope)
            metric, downstream, _ = _as_matrix(table, "candidate", scope)
            weighted = _weighted_gap(metric, downstream, len(_scope_positions(table, scope)), len(table.alpha_grid), np.ones(len(table.cluster_ids), dtype=np.int64))
            self.assertAlmostEqual(weighted[0], point["ag"], delta=1e-10)
            self.assertAlmostEqual(weighted[1], point["ag_sst"], delta=1e-10)
            self.assertAlmostEqual(weighted[2], point["ag_sse_pooled"], delta=1e-10)
            self.assertAlmostEqual(weighted[3], point["ag_sse_separate"], delta=1e-10)
            boot = _bootstrap_pair(table, table, "mse", "candidate", scope, draw)["values"][0]
            reference = _point_alignment(table, "mse", scope)
            np.testing.assert_allclose(boot, (reference["ag"], point["ag"], reference["ag"] - point["ag"], reference["oc"], point["oc"], point["oc"] - reference["oc"]), atol=1e-10, rtol=1e-8)

    def test_integrated_task_harm_transforms_each_alpha_before_averaging(self) -> None:
        """Regression: alpha sign cancellation must not erase positive/negative harm."""
        from rpe.runner.w1_axis_statistics import _task_rows

        rows = _task_rows(_sign_changing_alpha_table(), np.asarray([[0, 1], [0, 0]], dtype=np.int64))
        integrated = next(row for row in rows if row["family"] == "p08" and row["scope"] == "integrated")
        self.assertEqual(integrated["signed_mean"], 0.0)
        # The third fixture cluster is zero: 6 positive and 6 negative
        # units over six cluster-alpha observations.
        self.assertEqual(integrated["positive_mean"], 1.0)
        self.assertEqual(integrated["negative_mean"], 1.0)
        self.assertEqual(integrated["absolute_mean"], 2.0)
        self.assertEqual(integrated["positive_mean_lower"], 1.5125)
        self.assertEqual(integrated["positive_mean_upper"], 1.9875)
        self.assertEqual(integrated["absolute_mean_lower"], 3.025)
        self.assertEqual(integrated["absolute_mean_upper"], 3.975)

    def test_ag_family_decomposition_matches_canonical_observation_order(self) -> None:
        """A non-lexical table order must not mispair canonical predictions and y."""
        from rpe.runner.w1_axis_statistics import _decomposition_rows, _point_alignment

        table = NativePanelTable(**{**_multi_cluster_table().__dict__, "cluster_ids": ("2", "10", "1")})
        point = _point_alignment(table, "candidate", "all5")
        rows = _decomposition_rows(table, "native", "candidate", "all5", False, point)

        self.assertAlmostEqual(sum(row["d_p"] for row in rows), point["ag"], delta=1e-10)
        self.assertAlmostEqual(sum(row["pooled_sse"] for row in rows), point["ag_sse_pooled"], delta=1e-10)
        self.assertAlmostEqual(sum(row["separate_sse"] for row in rows), point["ag_sse_separate"], delta=1e-10)
        self.assertGreaterEqual(min(row["d_p"] for row in rows), -1e-10)

    def test_point_rows_preserve_exact_representation_and_scope_metadata(self) -> None:
        from rpe.runner.w1_axis_statistics import _point_alignment, _point_row

        table = _multi_cluster_table()
        expected = {
            "native_all5": ("native", "all5"),
            "native_no_axis": ("native", "no_axis"),
            "common_grid_all5": ("common_grid", "all5"),
            "common_grid_no_axis": ("common_grid", "no_axis"),
        }
        for analysis_id, (representation, scope) in expected.items():
            values = _point_alignment(table, "candidate", scope)
            row = _point_row(table, analysis_id, "candidate", "complete", values, values)
            self.assertEqual((row["representation"], row["perturbation_scope"]), (representation, scope))

    def test_zero_sst_draw_keeps_oc_values_and_intervals_finite(self) -> None:
        from rpe.runner.w1_axis_statistics import _bootstrap_pair

        table = _multi_cluster_table()
        downstream = table.downstream_harm.copy()
        downstream[0] = 0.0
        table = NativePanelTable(**{**table.__dict__, "downstream_harm": downstream})
        result = _bootstrap_pair(
            table, table, "mse", "candidate", "all5",
            np.asarray([[0, 0, 0], [0, 1, 2]], dtype=np.int64),
        )

        self.assertEqual(result["failed_sst_draw_count"], 1)
        self.assertTrue(np.isfinite(result["values"][0, 3:]).all())
        self.assertTrue(np.isfinite(result["candidate_oc_interval"]).all())
        self.assertIsNone(result["candidate_ag_interval"])

    def test_summarize_only_does_not_require_bootstrap_results(self) -> None:
        from rpe.runner.w1_axis_statistics import StatisticInputs, summarize_statistics

        original = _multi_cluster_table()
        table = NativePanelTable(**{
            **original.__dict__,
            "metric_output_ids": ("mse", "wasserstein_1_cm1"),
        })
        inputs = StatisticInputs(
            SimpleNamespace(metric_output_ids=table.metric_output_ids),
            {table.panel_id: table}, {table.panel_id: table}, frozenset(), "fixture", (),
        )
        result = summarize_statistics(inputs, infer=False)
        self.assertEqual(len(result["alignment_summary"]), 8)
        self.assertEqual(len(result["protocol_interactions"]), 40)

    def test_cross_family_harm_shares_use_paired_draw_ratios(self) -> None:
        from rpe.runner.w1_axis_statistics import _task_share_rows

        draws = np.asarray([[0, 1, 2], [0, 0, 0], [1, 1, 1]], dtype=np.int64)
        rows = _task_share_rows(_harm_share_table(), draws)
        by_component = {row["component"]: row for row in rows}

        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(by_component["p08"]["positive_harm_share"], 1.0 / 3.0)
        self.assertAlmostEqual(by_component["p08"]["positive_harm_share_lower"], 0.175)
        self.assertAlmostEqual(by_component["p08"]["positive_harm_share_upper"], 0.49166666666666664)
        self.assertAlmostEqual(by_component["p08"]["absolute_harm_share"], 2.0 / 7.0)
        self.assertAlmostEqual(by_component["axis_p11_p12"]["positive_harm_share"], 1.0 / 6.0)
        self.assertAlmostEqual(by_component["axis_p11_p12"]["absolute_harm_share"], 2.0 / 7.0)
        self.assertEqual(by_component["axis_p11_p12"]["positive_denominator_state"], "complete")

    def test_endpoint_draw_seed_and_duplicate_clusters_are_preserved(self) -> None:
        from rpe.runner.w1_axis_statistics import endpoint_bootstrap_draws

        first = endpoint_bootstrap_draws("endpoint", 3, 4)
        second = endpoint_bootstrap_draws("endpoint", 3, 4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (4, 3))
        self.assertTrue(any(np.any(np.bincount(draw, minlength=3) > 1) for draw in first))

    def test_holm_family_sizes_include_unavailable_placeholders(self) -> None:
        from rpe.runner.w1_axis_statistics import apply_holm_families

        rows = [
            {"hypothesis_id": "a", "family_id": "primary", "raw_p": 0.01, "state": "complete"},
            {"hypothesis_id": "b", "family_id": "primary", "raw_p": None, "state": "unavailable"},
        ]
        apply_holm_families(rows, family_sizes={"primary": 2})
        self.assertEqual(rows[0]["family_size"], 2)
        self.assertEqual(rows[1]["family_size"], 2)
        self.assertIsNone(rows[1]["adjusted_p"])
        self.assertEqual(rows[1]["p_for_adjustment"], 1.0)
        self.assertIsNone(rows[1]["raw_p"])

    def test_holm_unavailable_placeholder_uses_null_display_p_and_one_adjustment_input(self) -> None:
        from rpe.runner.w1_axis_statistics import apply_holm_families

        rows = [
            {"hypothesis_id": "a", "family_id": "primary", "raw_p": 0.01, "state": "complete"},
            {"hypothesis_id": "b", "family_id": "primary", "raw_p": None, "state": "unavailable"},
        ]
        apply_holm_families(rows, family_sizes={"primary": 2})
        self.assertIsNone(rows[1]["adjusted_p"])
        self.assertEqual(rows[1]["p_for_adjustment"], 1.0)
        self.assertIsNone(rows[1]["holm_rank"])
        self.assertFalse(rows[1]["rejected"])

    def test_new_analysis_family_grid_includes_unavailable_w1_slots(self) -> None:
        from rpe.runner.w1_axis_statistics import _inference_family

        metrics = ("mse", "wasserstein_1_cm1", *tuple(f"other_{index}" for index in range(11)))
        assignments = [
            _inference_family(analysis_id, metric)
            for panel_id in range(11)
            for analysis_id in ("native_no_axis", "common_grid_all5", "common_grid_no_axis")
            for metric in metrics
        ]
        self.assertEqual(sum(item == "w1_axis_primary_66" for item in assignments) * 2, 66)
        self.assertEqual(sum(item == "other_metrics_secondary_726" for item in assignments) * 2, 726)
        self.assertEqual(_inference_family("common_grid_all5", "wasserstein_1_cm1"), "w1_axis_primary_66")

    def test_d5_common_grid_w1_interaction_is_retained_as_unavailable(self) -> None:
        from rpe.runner.w1_axis_statistics import _interaction_rows

        rows = _interaction_rows(None, {}, infer=False)
        d5 = [row for row in rows if row["endpoint_id"] == "d5" and row["analysis_id"] == "common_grid_all5"]
        self.assertEqual(len(d5), 2)
        self.assertTrue(all(row["state"] == "unavailable" for row in d5))

    def test_parallel_bootstrap_matches_reference_with_the_same_draw_table(self) -> None:
        from rpe.runner.w1_axis_statistics import _bootstrap_pair, parallel_bootstrap_pairs

        table = _table()
        draws = np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int64)
        reference = _bootstrap_pair(table, table, "mse", "candidate", "no_axis", draws)
        actual = parallel_bootstrap_pairs(
            [("fixture", table, table, "mse", "candidate", "no_axis", draws)],
            worker_count=2,
        )["fixture"]
        self.assertEqual(actual["failed_sst_draw_count"], reference["failed_sst_draw_count"])
        np.testing.assert_allclose(actual["values"], reference["values"], atol=1e-10, rtol=1e-8)

    def test_interaction_intervals_use_paired_endpoint_draws(self) -> None:
        """Changing one protocol's bootstrap row must move the interaction interval."""
        from rpe.runner.w1_axis_statistics import _interaction_rows

        table_a = _table()
        table_b = NativePanelTable(
            **{**table_a.__dict__, "panel_id": "d2_5_b", "endpoint_id": "d2_5", "protocol_id": "b"}
        )
        table_a = NativePanelTable(
            **{**table_a.__dict__, "panel_id": "d2_5_a", "endpoint_id": "d2_5", "protocol_id": "a"}
        )
        draws = np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int64)
        values_a = {"ag": 0.1, "oc": 0.2}
        values_b = {"ag": 0.3, "oc": 0.4}
        comparisons = {}
        for panel, values in ((table_a, values_a), (table_b, values_b)):
            for analysis in ("native_all5", "native_no_axis", "common_grid_all5", "common_grid_no_axis"):
                comparisons[(panel.panel_id, analysis, "mse")] = (panel, values, draws)
                comparisons[(panel.panel_id, analysis, "wasserstein_1_cm1")] = (panel, values, draws)
        bootstrap = {}
        for panel, contrast in ((table_a, np.asarray([0.2, 0.4, 0.6])), (table_b, np.asarray([0.7, 0.4, 0.1]))):
            for analysis in ("native_all5", "native_no_axis", "common_grid_all5", "common_grid_no_axis"):
                rows = np.zeros((3, 6), dtype=float)
                rows[:, 2] = contrast
                rows[:, 5] = contrast / 10.0
                bootstrap[f"{panel.panel_id}|{analysis}|wasserstein_1_cm1"] = {"values": rows}
        rows = _interaction_rows(None, comparisons, infer=True, bootstrap_results=bootstrap)
        ag = next(row for row in rows if row["endpoint_id"] == "d2_5" and row["analysis_id"] == "native_all5" and row["outcome"] == "ag")
        # Same draw rows give B-A = [0.5, 0.0, -0.5], not a subtraction of two marginal intervals.
        self.assertAlmostEqual(ag["bootstrap_lower"], -0.475)
        self.assertAlmostEqual(ag["bootstrap_upper"], 0.475)
        self.assertTrue(ag["shared_endpoint_draws"])

    def test_bridge_rejects_mismatched_cluster_order(self) -> None:
        from rpe.runner.w1_axis_statistics import _bridge_rows, W1AxisStatisticsError

        native = _table()
        common = NativePanelTable(
            **{**native.__dict__, "cluster_ids": ("c1", "c0")}
        )
        inputs = SimpleNamespace(native_tables={"panel_a": native})
        comparisons = {
            ("panel_a", "native_all5", "wasserstein_1_cm1"): (native, {"ag": 0.1, "oc": 0.2}, np.asarray([[0, 1]])),
            ("panel_a", "common_grid_all5", "mse"): (common, {"ag": 0.2, "oc": 0.1}, np.asarray([[0, 1]])),
            ("panel_a", "native_no_axis", "wasserstein_1_cm1"): (native, {"ag": 0.1, "oc": 0.2}, np.asarray([[0, 1]])),
            ("panel_a", "common_grid_no_axis", "mse"): (common, {"ag": 0.2, "oc": 0.1}, np.asarray([[0, 1]])),
        }
        with self.assertRaisesRegex(W1AxisStatisticsError, "cluster mismatch"):
            _bridge_rows(inputs, comparisons, infer=False)

    def test_bridge_d5_common_grid_mse_requires_a_real_bootstrap_result(self) -> None:
        from rpe.runner.w1_axis_statistics import _bridge_rows, W1AxisStatisticsError

        native = _table()
        native = NativePanelTable(**{**native.__dict__, "panel_id": "d5_a", "endpoint_id": "d5"})
        common = NativePanelTable(**{**native.__dict__, "metric_output_ids": ("mse",), "metric_harm": native.metric_harm[:1]})
        inputs = SimpleNamespace(native_tables={"d5_a": native})
        draws = np.asarray([[0, 1]], dtype=np.int64)
        comparisons = {
            ("d5_a", "native_all5", "wasserstein_1_cm1"): (native, {"ag": 0.1, "oc": 0.2}, draws),
            ("d5_a", "common_grid_all5", "mse"): (common, {"ag": 0.2, "oc": 0.1}, draws),
            ("d5_a", "native_no_axis", "wasserstein_1_cm1"): (native, {"ag": 0.1, "oc": 0.2}, draws),
            ("d5_a", "common_grid_no_axis", "mse"): (common, {"ag": 0.2, "oc": 0.1}, draws),
        }
        with self.assertRaisesRegex(W1AxisStatisticsError, "missing bridge bootstrap"):
            _bridge_rows(inputs, comparisons, infer=True, bootstrap_results={})

    def test_bootstrap_cache_reuses_completed_shared_draw_result(self) -> None:
        from rpe.runner.w1_axis_statistics import parallel_bootstrap_pairs

        table = _table()
        draws = np.asarray([[0, 1], [1, 1], [0, 0]], dtype=np.int64)
        task = ("fixture", table, table, "mse", "candidate", "no_axis", draws)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = parallel_bootstrap_pairs([task], worker_count=1, cache_dir=cache)["fixture"]
            self.assertEqual(len(list(cache.glob("*.npz"))), 1)
            # A changed table would yield a different fresh answer; cache use is
            # the resume contract for an already-completed immutable task key.
            changed = NativePanelTable(**{**table.__dict__, "metric_harm": table.metric_harm + 3.0})
            resumed = parallel_bootstrap_pairs([("fixture", table, changed, "mse", "candidate", "no_axis", draws)], worker_count=1, cache_dir=cache)["fixture"]
            np.testing.assert_allclose(resumed["values"], first["values"], atol=1e-10, rtol=1e-8)

    def test_batched_sign_flip_matches_individual_two_sided_counts(self) -> None:
        from rpe.runner.w1_axis_statistics import _sign_flip, batched_sign_flip

        signs = np.asarray([[1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]], dtype=np.int8)
        contributions = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.4]], dtype=float)
        actual = batched_sign_flip(contributions, signs, aggregations=("sum", "mean"))
        expected = [_sign_flip(row, signs, kind)["raw_p"] for row, kind in zip(contributions, ("sum", "mean"), strict=True)]
        self.assertEqual(actual, expected)

    def test_endpoint_batched_sign_flip_parallel_chunks_match_fixed_sign_table_and_resume(self) -> None:
        from rpe.runner.w1_axis_statistics import (
            _endpoint_signs, _sign_flip, batched_sign_flip, endpoint_batched_sign_flip,
        )

        endpoint = "fixture_endpoint"
        contributions = np.asarray(
            [[1.0, -2.0, 0.5], [0.3, 0.1, -0.4], [0.0, 1.0, -1.0]],
            dtype=float,
        )
        aggregations = ("sum", "mean", "sum")
        signs = _endpoint_signs(endpoint, 3)
        expected = [_sign_flip(row, signs, aggregation) for row, aggregation in zip(contributions, aggregations, strict=True)]
        expected_batched = batched_sign_flip(contributions, signs, aggregations=aggregations)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "sign-flip-progress.json"
            actual = endpoint_batched_sign_flip(
                endpoint, contributions, aggregations=aggregations, worker_count=2,
                cache_dir=root / "cache", progress_path=progress, chunk_size=4096,
            )
            self.assertEqual(actual["extreme"].tolist(), [item["extreme"] for item in expected])
            self.assertEqual(actual["raw_p"], expected_batched)
            self.assertEqual(json.loads(progress.read_text())["status"], "complete")
            self.assertEqual(len(list((root / "cache").glob("*.npz"))), 1)
            resumed = endpoint_batched_sign_flip(
                endpoint, contributions, aggregations=aggregations, worker_count=1,
                cache_dir=root / "cache", progress_path=progress, chunk_size=4096,
            )
            self.assertEqual(resumed["extreme"].tolist(), actual["extreme"].tolist())
            changed = endpoint_batched_sign_flip(
                endpoint, contributions + 0.25, aggregations=aggregations, worker_count=1,
                cache_dir=root / "cache", progress_path=progress, chunk_size=4096,
            )
            self.assertNotEqual(changed["cache_identity"], actual["cache_identity"])
            self.assertEqual(len(list((root / "cache").glob("*.npz"))), 2)


if __name__ == "__main__":
    unittest.main()
