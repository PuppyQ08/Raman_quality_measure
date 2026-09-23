from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.environ.get("RPE_W1_AXIS_ARTIFACT_ROOT", ROOT)).resolve()
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.runner.w1_axis_robustness import NativePanelTable  # noqa: E402
import tools.run_w1_axis_robustness as w1_cli  # noqa: E402


def _module():
    return importlib.import_module("rpe.runner.w1_axis_common_grid")


def _spectrum(spectrum_id: str, axis: np.ndarray, intensity: np.ndarray) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id=spectrum_id,
        axis_cm1=np.asarray(axis, dtype=np.float64),
        intensity=np.asarray(intensity, dtype=np.float64),
    )


def _flat_detector(_spectrum: Spectrum1D):
    return ()


def _reconstructible_worker_detector(module):
    from rpe.methods.catalog import load_classical_catalog

    catalog_path = ARTIFACT_ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
    system_id = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
    system = next(item for item in load_classical_catalog(catalog_path).systems if item.system_id == system_id)
    return (
        module.CommonGridWorkerConfig(
            str(catalog_path), system_id,
            str(ARTIFACT_ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"),
            str(ARTIFACT_ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"),
        ),
        module._peak_detector(system),
    )


def _preoptimization_record_oracle(module, record, support, detector):
    """Hand-preserved Task-2 pre-memoization record algorithm."""
    success_rows = []
    failure_rows = []
    try:
        baseline_source, baseline_self = module.resample_pair_to_common_grid(
            record.source_spectrum, record.source_spectrum, support.axis_cm1,
            interpolator="linear", max_in_range_native_gap_cm1=support.max_in_range_native_gap_cm1,
        )
        baseline_values = module.compute_common_grid_metric_vector(
            baseline_source, baseline_self, detect_peaks=detector,
        )
    except Exception as error:
        reason = f"baseline_failure:{type(error).__name__}:{error}"
        return [], [
            {
                "record_id": record.record_id, "cluster_id": record.cluster_id,
                "perturbation_id": condition.perturbation_id, "alpha": float(condition.alpha),
                "reason_code": reason,
            }
            for condition in record.condition_spectra
        ]
    for condition in record.condition_spectra:
        try:
            resampled_source, resampled_candidate = module.resample_pair_to_common_grid(
                record.source_spectrum, condition.spectrum, support.axis_cm1,
                interpolator="linear", max_in_range_native_gap_cm1=support.max_in_range_native_gap_cm1,
            )
            current_values = module.compute_common_grid_metric_vector(
                resampled_source, resampled_candidate, detect_peaks=detector,
            )
            harms = module.orient_metric_harms_from_alpha_zero(current_values, baseline_values)
        except Exception as error:
            failure_rows.append({
                "record_id": record.record_id, "cluster_id": record.cluster_id,
                "perturbation_id": condition.perturbation_id, "alpha": float(condition.alpha),
                "reason_code": str(error),
            })
            continue
        for metric_output_id, metric_harm in harms.items():
            success_rows.append({
                "record_id": record.record_id, "cluster_id": record.cluster_id,
                "perturbation_id": condition.perturbation_id, "alpha": float(condition.alpha),
                "metric_output_id": metric_output_id, "metric_harm": float(metric_harm),
                "aggregation_weight": int(record.aggregation_weight),
            })
    return success_rows, failure_rows


def _tiny_parent_table() -> NativePanelTable:
    metric_harm = np.asarray(
        [
            [[[0.0], [0.1]]],
            [[[0.0], [0.2]]],
        ],
        dtype=np.float64,
    )
    downstream = np.asarray([[[0.0], [0.3]]], dtype=np.float64)
    counts = np.asarray([[[2], [2]]], dtype=np.int64)
    return NativePanelTable(
        panel_id="panel_a",
        endpoint_id="endpoint",
        protocol_id="a",
        parent_key="panel_a",
        cluster_kind="synthetic_cluster",
        cluster_ids=("c0",),
        metric_output_ids=("mse", "wasserstein_1_cm1"),
        perturbation_ids=("p08", "p11"),
        alpha_grid=(0.05,),
        metric_harm=metric_harm,
        downstream_harm=downstream,
        within_cluster_counts=counts,
        parent_status="complete",
        common_grid_ready=True,
    )


class W1AxisCommonGridCliTest(unittest.TestCase):
    def test_cli_accepts_resample_stage_and_worker_count(self) -> None:
        arguments = w1_cli._parser().parse_args(["--stage", "resample", "--worker-count", "3"])

        self.assertEqual(arguments.stage, "resample")
        self.assertEqual(arguments.worker_count, 3)

    def test_cli_accepts_max_records_per_dataset_for_canary(self) -> None:
        arguments = w1_cli._parser().parse_args(
            ["--stage", "resample", "--max-records-per-dataset", "1"]
        )

        self.assertEqual(arguments.stage, "resample")
        self.assertEqual(arguments.max_records_per_dataset, 1)


class W1AxisCommonGridSupportSpecTest(unittest.TestCase):
    def test_support_specs_match_frozen_dataset_grids(self) -> None:
        module = _module()

        specs = module.load_common_grid_support_specs()

        self.assertEqual(tuple(sorted(specs)), ("bacteria", "d5", "sugar"))
        self.assertEqual(specs["bacteria"].point_count, 997)
        self.assertEqual(
            specs["bacteria"].axis_sha256_f64,
            "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6",
        )
        self.assertEqual(specs["sugar"].point_count, 1999)
        self.assertEqual(
            specs["sugar"].axis_sha256_f64,
            "db910b11f92151db481391e96b8a06e246140596cfd4abad64764b87d2d84ee5",
        )
        self.assertEqual(specs["d5"].point_count, 799)
        self.assertEqual(
            specs["d5"].axis_sha256_f64,
            "23a29616dbac14efd7de072b9b84ee82c6740cf46ddd89e644819967f4648422",
        )


class W1AxisCommonGridMetricTest(unittest.TestCase):
    def test_common_grid_preserves_shifted_peak_displacement_for_mse(self) -> None:
        module = _module()
        source = _spectrum(
            "s0",
            np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        )
        shifted = _spectrum(
            "sa",
            np.asarray([0.25, 1.25, 2.25, 3.25, 4.25], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),
        )
        support_axis = np.asarray([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75], dtype=np.float64)

        native_mse = float(np.mean((source.intensity - shifted.intensity) ** 2))
        resampled_source, resampled_shifted = module.resample_pair_to_common_grid(
            source,
            shifted,
            support_axis,
            interpolator="linear",
            max_in_range_native_gap_cm1=None,
        )
        metrics = module.compute_common_grid_metric_vector(
            resampled_source,
            resampled_shifted,
            detect_peaks=_flat_detector,
        )

        self.assertEqual(native_mse, 0.0)
        self.assertGreater(metrics["mse"], 0.0)

    def test_common_grid_rejects_extrapolation(self) -> None:
        module = _module()
        source = _spectrum("s0", np.asarray([0.0, 1.0, 2.0], dtype=np.float64), np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        candidate = _spectrum("sa", np.asarray([0.25, 1.25, 2.25], dtype=np.float64), np.asarray([0.0, 1.0, 0.0], dtype=np.float64))

        with self.assertRaisesRegex(module.W1AxisCommonGridError, "extrapolation"):
            module.resample_pair_to_common_grid(
                source,
                candidate,
                np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
                interpolator="linear",
                max_in_range_native_gap_cm1=None,
            )

    def test_alpha_zero_common_grid_harms_are_zero_for_all_13_metrics(self) -> None:
        module = _module()
        source = _spectrum(
            "s0",
            np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.asarray([0.2, 0.1, 0.6, 0.1, 0.2], dtype=np.float64),
        )
        values = module.compute_common_grid_metric_vector(
            source,
            source,
            detect_peaks=_flat_detector,
        )
        harms = module.orient_metric_harms_from_alpha_zero(values, values)

        self.assertEqual(tuple(harms), tuple(module.METRIC_OUTPUT_IDS))
        for metric_id, value in harms.items():
            self.assertAlmostEqual(value, 0.0, places=12, msg=metric_id)


class W1AxisCommonGridRecordExecutionTest(unittest.TestCase):
    def test_zero_positive_mass_fails_only_w1_and_retains_other_metric_rows(self) -> None:
        """A W1 domain rejection must not discard the other twelve metrics."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([1.0, 2.0, 1.0]))
        zero_positive_mass = _spectrum("zero", axis, np.asarray([-1.0, -2.0, -0.5]))
        result = module.evaluate_common_grid_records(
            (module.CommonGridRecord(
                "r0", "c0", source,
                (module.CommonGridCondition("p08", 0.05, zero_positive_mass),),
            ),),
            support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
        )

        self.assertEqual(len(result.aggregated_rows), 12)
        self.assertEqual(
            {row["metric_output_id"] for row in result.aggregated_rows},
            set(module.METRIC_OUTPUT_IDS) - {"wasserstein_1_cm1"},
        )
        self.assertEqual(len(result.failure_rows), 1)
        failure = result.failure_rows[0]
        self.assertEqual(failure["metric_output_id"], "wasserstein_1_cm1")
        self.assertTrue(str(failure["reason_code"]).startswith("metric_failure:wasserstein_1_cm1:"))

    def test_baseline_w1_failure_marks_only_w1_unavailable_for_each_condition(self) -> None:
        """Alpha-zero failures propagate by metric, not by complete condition."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([1.0, 2.0, 1.0]))
        candidate = _spectrum("candidate", axis, np.asarray([1.0, 1.5, 1.0]))
        conditions = tuple(
            module.CommonGridCondition("p08", 0.05 + index / 1000.0, candidate)
            for index in range(40)
        )
        real_evaluate_metric = module.evaluate_metric

        def baseline_w1_failure(metric, request):
            if (
                isinstance(metric, module.Wasserstein1Metric)
                and request.candidate.spectrum_id == "source:common_grid"
            ):
                raise RuntimeError("injected baseline W1 failure")
            return real_evaluate_metric(metric, request)

        with mock.patch.object(module, "evaluate_metric", side_effect=baseline_w1_failure):
            result = module.evaluate_common_grid_records(
                (module.CommonGridRecord("r0", "c0", source, conditions),),
                support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
            )

        self.assertEqual(len(result.aggregated_rows), 40 * 12)
        self.assertEqual(len(result.failure_rows), 40)
        self.assertTrue(all(row["metric_output_id"] == "wasserstein_1_cm1" for row in result.failure_rows))
        self.assertTrue(all(str(row["reason_code"]).startswith("baseline_metric_failure:wasserstein_1_cm1:") for row in result.failure_rows))

    def test_source_resampling_failure_explicitly_covers_all_metrics_and_conditions(self) -> None:
        """An unavailable source representation remains a full 40x13 failure grid."""
        module = _module()
        support_axis = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=support_axis, point_count=support_axis.size,
            axis_sha256_f64=module._array_sha(support_axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source_outside_support = _spectrum("bad-source", np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 1.0]))
        candidate = _spectrum("candidate", support_axis, np.asarray([1.0, 2.0, 1.0]))
        conditions = tuple(
            module.CommonGridCondition("p08", 0.05 + index / 1000.0, candidate)
            for index in range(40)
        )
        result = module.evaluate_common_grid_records(
            (module.CommonGridRecord("r0", "c0", source_outside_support, conditions),),
            support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
        )

        self.assertEqual(result.aggregated_rows, ())
        self.assertEqual(len(result.failure_rows), 40 * len(module.METRIC_OUTPUT_IDS))
        self.assertEqual({row["metric_output_id"] for row in result.failure_rows}, set(module.METRIC_OUTPUT_IDS))
        self.assertTrue(all(str(row["reason_code"]).startswith("source_resample_failure:") for row in result.failure_rows))

    def test_candidate_peak_detection_failure_affects_only_peak_metrics(self) -> None:
        """Shared candidate peak detection may close only the five peak outputs."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([1.0, 2.0, 1.0]))
        candidate = _spectrum("candidate", axis, np.asarray([1.0, 1.5, 1.0]))

        def candidate_peak_failure(spectrum):
            if spectrum.spectrum_id.startswith("candidate:"):
                raise RuntimeError("injected candidate detector failure")
            return ()

        result = module.evaluate_common_grid_records(
            (module.CommonGridRecord(
                "r0", "c0", source,
                (module.CommonGridCondition("p08", 0.05, candidate),),
            ),),
            support_spec=support, detect_peaks=candidate_peak_failure, interpolator="linear",
        )

        self.assertEqual(len(result.aggregated_rows), 8)
        self.assertEqual({row["metric_output_id"] for row in result.failure_rows}, set(module.METRIC_OUTPUT_IDS[8:]))

    def test_record_evaluation_aggregates_successes_and_retains_failures(self) -> None:
        module = _module()
        support_axis = np.asarray([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75], dtype=np.float64)
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic",
            axis_cm1=support_axis,
            point_count=int(support_axis.size),
            axis_sha256_f64=module._array_sha(support_axis, dtype="<f8"),
            max_in_range_native_gap_cm1=None,
        )
        source = _spectrum(
            "s0",
            np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.asarray([0.2, 0.1, 0.6, 0.1, 0.2], dtype=np.float64),
        )
        shifted = _spectrum(
            "s1",
            np.asarray([0.25, 1.25, 2.25, 3.25, 4.25], dtype=np.float64),
            np.asarray([0.2, 0.1, 0.6, 0.1, 0.2], dtype=np.float64),
        )
        truncated = _spectrum(
            "s2",
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.asarray([0.2, 0.1, 0.6, 0.1], dtype=np.float64),
        )
        records = (
            module.CommonGridRecord(
                record_id="r0",
                cluster_id="c0",
                source_spectrum=source,
                condition_spectra=(
                    module.CommonGridCondition(
                        perturbation_id="p08",
                        alpha=0.05,
                        spectrum=shifted,
                    ),
                ),
                aggregation_weight=1,
            ),
            module.CommonGridRecord(
                record_id="r1",
                cluster_id="c0",
                source_spectrum=source,
                condition_spectra=(
                    module.CommonGridCondition(
                        perturbation_id="p08",
                        alpha=0.05,
                        spectrum=truncated,
                    ),
                ),
                aggregation_weight=1,
            ),
        )

        observed = module.evaluate_common_grid_records(
            records,
            support_spec=support,
            detect_peaks=_flat_detector,
            interpolator="linear",
        )

        self.assertEqual(observed.processed_record_count, 2)
        self.assertEqual(len(observed.aggregated_rows), len(module.METRIC_OUTPUT_IDS))
        self.assertEqual(len(observed.failure_rows), len(module.METRIC_OUTPUT_IDS))
        self.assertTrue(all(row["record_id"] == "r1" for row in observed.failure_rows))
        self.assertTrue(all(row["cluster_id"] == "c0" for row in observed.failure_rows))
        self.assertTrue(all(row["perturbation_id"] == "p08" for row in observed.failure_rows))
        self.assertTrue(all(row["alpha"] == 0.05 for row in observed.failure_rows))
        self.assertEqual({row["metric_output_id"] for row in observed.failure_rows}, set(module.METRIC_OUTPUT_IDS))
        mse_rows = [row for row in observed.aggregated_rows if row["metric_output_id"] == "mse"]
        self.assertEqual(len(mse_rows), 1)
        self.assertGreater(mse_rows[0]["metric_harm"], 0.0)
        self.assertEqual(mse_rows[0]["within_cluster_count"], 1)


class W1AxisCommonGridAggregationTest(unittest.TestCase):
    def test_partial_metric_grid_keeps_complete_metric_table_and_downstream(self) -> None:
        """One unavailable metric must not close the independent complete subset."""
        module = _module()
        parent = _tiny_parent_table()
        rows = tuple(
            {
                "cluster_id": "c0", "perturbation_id": perturbation_id, "alpha": 0.05,
                "metric_output_id": "mse", "metric_harm": float(index + 1),
            }
            for index, perturbation_id in enumerate(parent.perturbation_ids)
        )
        failure_rows = ({
            "record_id": "r0", "cluster_id": "c0", "perturbation_id": "p08", "alpha": 0.05,
            "metric_output_id": "wasserstein_1_cm1",
            "reason_code": "metric_failure:wasserstein_1_cm1:TransportMetricError:positive mass",
        },)

        result = module.finalize_common_grid_panel_result(
            panel_id=parent.panel_id, parent_table=parent, aggregated_rows=rows, failure_rows=failure_rows,
        )

        self.assertEqual(result.status, "partial_metric_grid")
        self.assertEqual(result.complete_metric_output_ids, ("mse",))
        self.assertEqual(result.incomplete_metric_output_ids, ("wasserstein_1_cm1",))
        self.assertIsNotNone(result.table)
        self.assertEqual(result.table.metric_output_ids, ("mse",))
        self.assertEqual(result.table.downstream_harm.tobytes(), parent.downstream_harm.tobytes())
    def test_weighted_and_unweighted_aggregation_follow_dataset_contracts(self) -> None:
        module = _module()
        unweighted = module.aggregate_record_metric_rows(
            (
                {"cluster_id": "c0", "perturbation_id": "p08", "alpha": 0.05, "metric_output_id": "mse", "metric_harm": 1.0},
                {"cluster_id": "c0", "perturbation_id": "p08", "alpha": 0.05, "metric_output_id": "mse", "metric_harm": 3.0},
            ),
            cluster_field="cluster_id",
            weight_field=None,
        )
        weighted = module.aggregate_record_metric_rows(
            (
                {"cluster_id": "c0", "perturbation_id": "p08", "alpha": 0.05, "metric_output_id": "mse", "metric_harm": 1.0, "occurrence_count": 2},
                {"cluster_id": "c0", "perturbation_id": "p08", "alpha": 0.05, "metric_output_id": "mse", "metric_harm": 4.0, "occurrence_count": 1},
            ),
            cluster_field="cluster_id",
            weight_field="occurrence_count",
        )

        self.assertEqual(len(unweighted), 1)
        self.assertAlmostEqual(unweighted[0]["metric_harm"], 2.0, places=12)
        self.assertEqual(unweighted[0]["within_cluster_count"], 2)
        self.assertEqual(len(weighted), 1)
        self.assertAlmostEqual(weighted[0]["metric_harm"], 2.0, places=12)
        self.assertEqual(weighted[0]["within_cluster_count"], 3)

    def test_metric_harm_replacement_preserves_parent_downstream_bitwise(self) -> None:
        module = _module()
        parent = _tiny_parent_table()
        replacement = np.asarray(
            [
                [[[9.0], [8.0]]],
                [[[7.0], [6.0]]],
            ],
            dtype=np.float64,
        )

        observed = module.replace_metric_harm(parent, replacement)

        self.assertTrue(np.array_equal(observed.metric_harm, replacement))
        self.assertTrue(np.array_equal(observed.downstream_harm, parent.downstream_harm))
        self.assertEqual(observed.downstream_harm.tobytes(), parent.downstream_harm.tobytes())
        self.assertTrue(np.array_equal(observed.within_cluster_counts, parent.within_cluster_counts))

    def test_incomplete_grid_returns_failures_instead_of_silent_drop(self) -> None:
        module = _module()
        parent = _tiny_parent_table()
        aggregated_rows = (
            {"cluster_id": "c0", "perturbation_id": "p08", "alpha": 0.05, "metric_output_id": "mse", "metric_harm": 1.0, "within_cluster_count": 2},
        )
        failure_rows = (
            {"cluster_id": "c0", "perturbation_id": "p11", "alpha": 0.05, "reason_code": "out_of_support"},
        )

        result = module.finalize_common_grid_panel_result(
            panel_id=parent.panel_id,
            parent_table=parent,
            aggregated_rows=aggregated_rows,
            failure_rows=failure_rows,
        )

        self.assertEqual(result.status, "incomplete_grid")
        self.assertIsNone(result.table)
        self.assertEqual(tuple(result.failure_rows), failure_rows)

    def test_full_dataset_rows_materialize_a_task3_compatible_panel_table(self) -> None:
        module = _module()
        parent = _tiny_parent_table()
        rows = tuple(
            {
                "cluster_id": "c0", "perturbation_id": perturbation_id,
                "alpha": 0.05, "metric_output_id": metric_output_id,
                "metric_harm": float(metric_index * 10 + perturbation_index),
            }
            for metric_index, metric_output_id in enumerate(parent.metric_output_ids)
            for perturbation_index, perturbation_id in enumerate(parent.perturbation_ids)
        )
        result = module.finalize_common_grid_panel_result(
            panel_id=parent.panel_id, parent_table=parent, aggregated_rows=rows, failure_rows=(),
        )

        self.assertEqual(result.status, "complete")
        self.assertIsNotNone(result.table)
        self.assertEqual(result.table.metric_harm[1, 0, 1, 0], 11.0)
        self.assertEqual(result.table.downstream_harm.tobytes(), parent.downstream_harm.tobytes())


class W1AxisCommonGridCacheTest(unittest.TestCase):
    def test_cache_identity_must_match_before_resume_reuse(self) -> None:
        module = _module()
        identity = module.build_common_grid_cache_identity(
            panel_id="d4_a",
            source_spectrum_sha256="a" * 64,
            sweep_sha256="b" * 64,
            representation_sha256="c" * 64,
            code_sha256="d" * 64,
        )
        payload = {"panel_id": "d4_a", "status": "complete"}

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory) / "cache"
            module.write_complete_common_grid_cache(
                cache_root,
                identity=identity,
                payload=payload,
            )
            loaded = module.load_complete_common_grid_cache(
                cache_root,
                expected_identity=identity,
            )

            self.assertEqual(loaded, payload)
            with self.assertRaisesRegex(module.W1AxisCommonGridError, "cache identity mismatch"):
                module.load_complete_common_grid_cache(
                    cache_root,
                    expected_identity={**identity, "code_sha256": "e" * 64},
                )

    def test_resume_reuses_matching_complete_cache_without_rebuild(self) -> None:
        module = _module()
        identity = module.build_common_grid_cache_identity(
            panel_id="d5",
            source_spectrum_sha256="1" * 64,
            sweep_sha256="2" * 64,
            representation_sha256="3" * 64,
            code_sha256="4" * 64,
        )
        payload = {"dataset_id": "d5", "status": "complete", "row_count": 40}

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory) / "cache"
            module.write_complete_common_grid_cache(
                cache_root,
                identity=identity,
                payload=payload,
            )
            builder = mock.Mock(side_effect=AssertionError("cache hit must not rebuild"))

            observed, cache_hit = module.load_or_build_common_grid_cache(
                cache_root,
                identity=identity,
                resume=True,
                build_payload=builder,
            )

            self.assertEqual(observed, payload)
            self.assertTrue(cache_hit)
            builder.assert_not_called()

    def test_missing_cache_builds_payload_and_publishes_completion_marker(self) -> None:
        module = _module()
        identity = module.build_common_grid_cache_identity(
            panel_id="bacteria",
            source_spectrum_sha256="5" * 64,
            sweep_sha256="6" * 64,
            representation_sha256="7" * 64,
            code_sha256="8" * 64,
        )
        payload = {"dataset_id": "bacteria", "status": "complete", "row_count": 120}

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory) / "cache"

            observed, cache_hit = module.load_or_build_common_grid_cache(
                cache_root,
                identity=identity,
                resume=True,
                build_payload=lambda: payload,
            )

            self.assertEqual(observed, payload)
            self.assertFalse(cache_hit)
            self.assertTrue((cache_root / "complete.json").is_file())
            reloaded = module.load_complete_common_grid_cache(
                cache_root,
                expected_identity=identity,
            )
            self.assertEqual(reloaded, payload)


class W1AxisCommonGridRealAdapterContractTest(unittest.TestCase):
    def test_real_dataset_materializer_exposes_all_three_source_loaders(self) -> None:
        module = _module()

        self.assertTrue(hasattr(module, "materialize_real_common_grid_records"))
        materializer = module.materialize_real_common_grid_records
        self.assertEqual(
            tuple(materializer.__annotations__.get("dataset_ids", ())),
            (),
        )

    def test_canary_dataset_result_is_explicitly_partial_never_complete(self) -> None:
        module = _module()
        self.assertTrue(hasattr(module, "CommonGridDatasetResult"))
        result = module.CommonGridDatasetResult(
            dataset_id="bacteria",
            aggregated_rows=(),
            failure_rows=(),
            processed_record_count=1,
            planned_record_count=3000,
            status="canary_partial",
            elapsed_seconds=0.1,
            cache_hit_count=0,
        )
        self.assertEqual(result.status, "canary_partial")
        self.assertLess(result.processed_record_count, result.planned_record_count)


class W1AxisCommonGridReviewRoundTest(unittest.TestCase):
    def test_actual_dataset_loaders_emit_raw_adapter_records_without_parent_materialization(self) -> None:
        """The production D2/D4/D5 loader routes must defer all P08--P12 work to workers."""
        module = _module()
        import rpe.downstream.rruff as rruff
        import rpe.downstream.sugar_quantitative as sugar
        import rpe.runner.phase4_d2_protocol_a as d2
        import rpe.runner.phase4_d4_eligibility as d4
        import rpe.runner.phase4_d5_protocol_a as d5

        axis = np.asarray([400.0, 401.0, 402.0], dtype=np.float64)
        d2_inputs = SimpleNamespace(
            record_ids=("d2-record",), native_test_labels=np.asarray([7]),
            native_test_spectra=(_spectrum("d2-source", axis, np.asarray([1.0, 2.0, 1.0])),),
        )
        d4_cohort = SimpleNamespace(
            record_ids=("d4-record",), well_ids=("well-9",), wavenumber=axis,
            intensity=np.asarray([[1.0, 2.0, 1.0]], dtype=np.float64),
        )
        d5_cohort = SimpleNamespace(
            record_ids=("d5-zero", "d5-one"), class_labels=np.asarray([3, 4]),
            mineral_names=("zero", "one"),
            splits=(SimpleNamespace(query_indices=(1, 1)),),
        )
        system_id = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
        catalog = SimpleNamespace(systems=(SimpleNamespace(system_id=system_id),))
        eligibility = SimpleNamespace(cwt_system_id=system_id, p10_memory_budget_bytes=4096)

        with mock.patch.object(module, "_conditions_for_source", side_effect=AssertionError("parent materialized conditions")), mock.patch.object(
            module, "_peak_detector", return_value=_flat_detector,
        ), mock.patch.object(module, "load_classical_catalog", create=True, return_value=catalog):
            with mock.patch.object(d2, "reconstruct_d2_protocol_a_inputs", return_value=d2_inputs), mock.patch.object(
                d2, "load_phase4_d2_protocol_a_config", return_value=object(),
            ), mock.patch("rpe.methods.catalog.load_classical_catalog", return_value=catalog):
                bacteria = module.materialize_real_common_grid_dataset("bacteria", root=ARTIFACT_ROOT, max_records_per_dataset=1)
            with mock.patch.object(sugar, "load_d4_sugar_cohort", return_value=d4_cohort), mock.patch.object(
                d4, "load_phase4_d4_eligibility_config", return_value=eligibility,
            ), mock.patch("rpe.methods.catalog.load_classical_catalog", return_value=catalog):
                sugar_dataset = module.materialize_real_common_grid_dataset("sugar", root=ARTIFACT_ROOT, max_records_per_dataset=1)
            with mock.patch.object(rruff, "load_d5_raw_cohort", return_value=d5_cohort), mock.patch.object(
                rruff, "load_d5_native_spectra", return_value=(_spectrum("d5-source", axis, np.asarray([1.0, 2.0, 1.0])),),
            ), mock.patch.object(d5, "_resolve_cwt", return_value=SimpleNamespace(system_id=system_id)), mock.patch(
                "rpe.methods.catalog.load_classical_catalog", return_value=catalog,
            ):
                d5_dataset = module.materialize_real_common_grid_dataset("d5", root=ARTIFACT_ROOT, max_records_per_dataset=1)

        self.assertEqual((bacteria.records[0].source_adapter_id, bacteria.records[0].cluster_id), ("d2", "7"))
        self.assertEqual((sugar_dataset.records[0].source_adapter_id, sugar_dataset.records[0].cluster_id), ("d4", "well-9"))
        self.assertEqual(
            (d5_dataset.records[0].source_adapter_id, d5_dataset.records[0].source_adapter_metadata, d5_dataset.records[0].aggregation_weight),
            ("d5", ("one",), 2),
        )
        for dataset in (bacteria, sugar_dataset, d5_dataset):
            self.assertEqual(dataset.records[0].condition_spectra, ())
            self.assertTrue(module.is_fused_raw_common_grid_record(dataset.records[0]))

    def test_real_frozen_d2_adapter_full_p08_p12_oracle_matches_worker_materialization(self) -> None:
        """All 40 real frozen P08--P12 cells must agree before and after worker rematerialization."""
        module = _module()
        from rpe.perturb import load_perturbation_sweep_config
        from rpe.runner.phase1_config import load_phase1_core_config
        from rpe.runner.phase4_d2_protocol_a import _phase1_source

        support = module.load_common_grid_support_specs()["bacteria"]
        axis = support.axis_cm1
        rng = np.random.default_rng(11)
        source = _spectrum(
            "d2-real-frozen-source", axis,
            np.sin(axis / 8.0) + np.exp(-((axis - 900.0) / 36.0) ** 2) + 0.25 * rng.normal(size=axis.size),
        )
        raw = module.CommonGridRecord(
            "d2-real-frozen-source", "7", source, (), source_order=4, source_adapter_id="d2",
        )
        worker_config = module.CommonGridWorkerConfig(
            str(ARTIFACT_ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"),
            "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511",
            str(ARTIFACT_ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"),
            str(ARTIFACT_ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"),
            source_adapter_id="d2", p10_memory_budget_bytes=64 * 1024**3,
        )
        phase1 = load_phase1_core_config(Path(worker_config.phase1_config_path))
        sweep = load_perturbation_sweep_config(Path(worker_config.sweep_config_path))
        old_conditions = module._conditions_for_source(
            _phase1_source(raw.source_order, raw.record_id, int(raw.cluster_id), raw.source_spectrum),
            phase1=phase1, sweep=sweep, p10_memory_budget_bytes=worker_config.p10_memory_budget_bytes,
        )
        self.assertEqual(len(old_conditions), 40)
        module._worker_initializer(worker_config)
        fused_conditions = module._process_materialize_conditions(raw, worker_config)
        self.assertEqual(
            tuple((condition.perturbation_id, condition.alpha) for condition in old_conditions),
            tuple((condition.perturbation_id, condition.alpha) for condition in fused_conditions),
        )
        for expected, observed in zip(old_conditions, fused_conditions, strict=True):
            np.testing.assert_allclose(expected.spectrum.axis_cm1, observed.spectrum.axis_cm1, atol=0.0, rtol=0.0)
            np.testing.assert_allclose(expected.spectrum.intensity, observed.spectrum.intensity, atol=1e-10, rtol=1e-8)

        old_rows, old_failures = module._evaluate_common_grid_record(
            module.replace(raw, condition_spectra=old_conditions), support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
        )
        fused_rows, fused_failures = module._evaluate_fused_common_grid_record(
            raw, support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
            materialize_conditions=lambda record: module._process_materialize_conditions(record, worker_config),
        )
        self.assertEqual(old_failures, fused_failures)
        self.assertEqual(
            len(old_rows) + len(old_failures),
            40 * len(module.METRIC_OUTPUT_IDS),
        )
        self.assertEqual(len(old_rows), len(fused_rows))
        for expected, observed in zip(old_rows, fused_rows, strict=True):
            self.assertEqual(
                {key: value for key, value in expected.items() if key != "metric_harm"},
                {key: value for key, value in observed.items() if key != "metric_harm"},
            )
            self.assertTrue(np.isclose(expected["metric_harm"], observed["metric_harm"], atol=1e-10, rtol=1e-8))

    def test_fused_raw_records_match_serial_and_process_with_nonempty_peaks(self) -> None:
        """Fused adapter records retain nonempty CWT semantics across serial and process execution."""
        module = _module()
        worker_config, detector = _reconstructible_worker_detector(module)
        worker_config = module.replace(worker_config, source_adapter_id="d2", p10_memory_budget_bytes=64 * 1024**3)
        support = module.load_common_grid_support_specs()["bacteria"]
        axis = support.axis_cm1
        rng = np.random.default_rng(7)
        records = tuple(
            module.CommonGridRecord(
                f"d2-fused-{index}", "7",
                _spectrum(
                    f"d2-fused-{index}", axis,
                    np.sin(axis / 8.0) + np.exp(-((axis - (850.0 + 20.0 * index)) / 30.0) ** 2)
                    + 0.25 * rng.normal(size=axis.size),
                ),
                (), source_order=index, source_adapter_id="d2",
            )
            for index in range(2)
        )
        serial = module.evaluate_common_grid_records(
            records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=1, worker_config=worker_config,
        )
        parallel = module.evaluate_common_grid_records(
            records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=2, worker_config=worker_config,
        )

        self.assertEqual(serial.failure_rows, parallel.failure_rows)
        self.assertEqual(serial.aggregated_rows, parallel.aggregated_rows)
        self.assertGreater(len(detector(records[0].source_spectrum)), 0)
        self.assertGreater(len(serial.aggregated_rows), 0)
        self.assertGreaterEqual(len(parallel.worker_process_ids), 2)
        self.assertNotIn(os.getpid(), parallel.worker_process_ids)
        self.assertEqual(parallel.worker_blas_thread_limits, (1,))

    def test_fused_record_evaluator_matches_prematerialized_p08_p12_rows(self) -> None:
        """Moving rematerialization into a worker must preserve every old row/failure."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0, 3.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([0.1, 0.9, 0.2, 0.1]))
        conditions = (
            module.CommonGridCondition("p08", 0.05, _spectrum("p08", axis, np.asarray([0.1, 0.8, 0.3, 0.1]))),
            module.CommonGridCondition("p12", 0.10, _spectrum("p12", axis, np.asarray([0.2, 0.7, 0.3, 0.1]))),
        )
        old_record = module.CommonGridRecord("r0", "c0", source, conditions)
        raw_record = module.CommonGridRecord("r0", "c0", source, (), source_adapter_id="d2")
        expected = module._evaluate_common_grid_record(
            old_record, support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
        )
        observed = module._evaluate_fused_common_grid_record(
            raw_record, support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
            materialize_conditions=lambda _: conditions,
        )

        self.assertEqual(expected[1], observed[1])
        self.assertEqual(len(expected[0]), len(observed[0]))
        for old_row, fused_row in zip(expected[0], observed[0], strict=True):
            self.assertEqual(
                {key: value for key, value in old_row.items() if key != "metric_harm"},
                {key: value for key, value in fused_row.items() if key != "metric_harm"},
            )
            self.assertTrue(np.isclose(old_row["metric_harm"], fused_row["metric_harm"], atol=1e-10, rtol=1e-8))

    def test_real_loader_records_are_raw_without_parent_condition_materialization(self) -> None:
        """A real source loader must leave 40 P08-P12 spectra for worker execution."""
        module = _module()
        source = _spectrum("source", np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]))
        record = module.CommonGridRecord("r0", "c0", source, (), source_adapter_id="d2")

        self.assertTrue(module.is_fused_raw_common_grid_record(record))
        self.assertEqual(record.condition_spectra, ())
    def test_dataset_cache_roundtrip_preserves_process_execution_receipts(self) -> None:
        """Dropping process receipts from cache payloads must be externally visible."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        dataset = module.CommonGridDatasetInput(
            dataset_id="synthetic", records=(), planned_record_count=0, support_spec=support,
            detect_peaks=_flat_detector, sweep_sha256="a" * 64,
        )
        fresh = module.CommonGridDatasetResult(
            dataset_id="synthetic", aggregated_rows=(), failure_rows=(), processed_record_count=0,
            planned_record_count=0, status="complete", elapsed_seconds=1.25, cache_hit_count=0,
            requested_worker_count=16, effective_worker_count=4, p10_worker_capacity=4,
            worker_process_ids=(101, 202), worker_blas_thread_limits=(1,), cache_hit=False,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(module, "evaluate_real_common_grid_dataset", return_value=fresh) as evaluator:
                first = module.evaluate_or_load_real_common_grid_dataset(
                    dataset, cache_root=Path(temporary_directory), resume=True, code_sha256="b" * 64, worker_count=16,
                )
                second = module.evaluate_or_load_real_common_grid_dataset(
                    dataset, cache_root=Path(temporary_directory), resume=True, code_sha256="b" * 64, worker_count=16,
                )

        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(first.worker_process_ids, (101, 202))
        self.assertEqual(second.worker_process_ids, (101, 202))
        self.assertEqual(second.requested_worker_count, 16)
        self.assertEqual(second.effective_worker_count, 4)
        self.assertEqual(second.worker_blas_thread_limits, (1,))
        self.assertTrue(second.cache_hit)

    def test_multiworker_rejects_unreconstructible_detector_instead_of_using_empty_peaks(self) -> None:
        """A nonempty caller detector must never be silently replaced by empty peaks."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([1.0, 2.0, 1.0]))
        records = tuple(
            module.CommonGridRecord(
                f"r{index}", "c0", source,
                (module.CommonGridCondition("p08", 0.05, source),),
            )
            for index in range(2)
        )
        nonempty_detector = lambda _spectrum: (
            module.Peak1D(position_cm1=1.0, intensity=2.0),
        )

        with self.assertRaisesRegex(module.W1AxisCommonGridError, "worker_config"):
            module.evaluate_common_grid_records(
                records, support_spec=support, detect_peaks=nonempty_detector, interpolator="linear", worker_count=2,
            )

    def test_reconstructible_nonempty_peak_detector_matches_serial_and_process_metrics(self) -> None:
        """Process workers must preserve real nonempty CWT peak semantics."""
        module = _module()
        worker_config, detector = _reconstructible_worker_detector(module)
        axis = np.linspace(400.0, 600.0, 65, dtype=np.float64)
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        records = tuple(
            module.CommonGridRecord(
                f"r{index}", "c0",
                _spectrum(f"s{index}", axis, np.exp(-((axis - (500.0 + index)) / 8.0) ** 2)),
                (module.CommonGridCondition("p08", 0.05, _spectrum(f"c{index}", axis, np.exp(-((axis - (501.0 + index)) / 8.0) ** 2))),),
            )
            for index in range(2)
        )
        serial = module.evaluate_common_grid_records(
            records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=1,
        )
        parallel = module.evaluate_common_grid_records(
            records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=2, worker_config=worker_config,
        )

        self.assertEqual(serial.failure_rows, parallel.failure_rows)
        self.assertEqual(serial.aggregated_rows, parallel.aggregated_rows)

    def test_indexed_completion_reports_fast_record_before_slow_record_and_reassembles_order(self) -> None:
        """Replacing completion collection with input-order waiting hides live progress."""
        module = _module()
        checkpoints = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(lambda: (time.sleep(0.05), "slow")[1])
            second = executor.submit(lambda: "fast")
            observed = module.collect_indexed_future_results(
                {first: 0, second: 1}, on_completed=lambda count: checkpoints.append(count),
            )

        self.assertEqual(checkpoints[0], 1)
        self.assertEqual(observed, ("slow", "fast"))
    def test_record_memoizes_source_resample_and_peak_detection(self) -> None:
        """Removing source memoization must reintroduce this record-level cost."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([0.2, 0.1, 0.8, 0.1, 0.2]))
        shifted = _spectrum("shifted", axis, np.asarray([0.2, 0.1, 0.7, 0.1, 0.2]))
        record = module.CommonGridRecord(
            "r0", "c0", source,
            tuple(module.CommonGridCondition("p08", alpha, shifted) for alpha in (0.05, 0.1, 0.2)),
        )
        resample_calls = []
        detector_calls = []
        original_resample = module.resample_spectrum

        def tracking_resample(native_axis, native_intensity, target_axis, **kwargs):
            resample_calls.append((np.asarray(native_axis).tobytes(), np.asarray(native_intensity).tobytes()))
            return original_resample(native_axis, native_intensity, target_axis, **kwargs)

        def tracking_detector(spectrum):
            detector_calls.append(spectrum.spectrum_id)
            return ()

        with mock.patch.object(module, "resample_spectrum", side_effect=tracking_resample):
            module.evaluate_common_grid_records(
                (record,), support_spec=support, detect_peaks=tracking_detector, interpolator="linear",
            )

        self.assertEqual(
            sum(
                axis_bytes == source.axis_cm1.tobytes() and intensity_bytes == source.intensity.tobytes()
                for axis_bytes, intensity_bytes in resample_calls
            ),
            1,
        )
        self.assertEqual(sum(item == "source:common_grid" for item in detector_calls), 1)
        self.assertEqual(len(detector_calls), 4)

    def test_optimized_record_matches_preserved_preoptimization_oracle(self) -> None:
        """A semantic change in the optimized source/candidate split must be observable."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([0.2, 0.1, 0.8, 0.1, 0.2]))
        shifted = _spectrum("shifted", axis + 0.01, np.asarray([0.25, 0.15, 0.65, 0.15, 0.25]))
        record = module.CommonGridRecord(
            "r0", "c0", source,
            (module.CommonGridCondition("p08", 0.05, shifted), module.CommonGridCondition("p08", 0.1, source)),
        )
        expected_successes, expected_failures = _preoptimization_record_oracle(module, record, support, _flat_detector)
        observed_successes, observed_failures = module._evaluate_common_grid_record(
            record, support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
        )
        self.assertEqual(len(expected_failures), 1)
        self.assertEqual(len(observed_failures), len(module.METRIC_OUTPUT_IDS))
        self.assertEqual({row["metric_output_id"] for row in observed_failures}, set(module.METRIC_OUTPUT_IDS))
        self.assertEqual(len(expected_successes), len(observed_successes))
        for expected, observed in zip(expected_successes, observed_successes, strict=True):
            self.assertEqual({key: value for key, value in expected.items() if key != "metric_harm"}, {key: value for key, value in observed.items() if key != "metric_harm"})
            self.assertTrue(np.isclose(expected["metric_harm"], observed["metric_harm"], atol=1e-10, rtol=1e-8))

    def test_progress_log_records_dataset_boundaries_and_completed_counts(self) -> None:
        """Dropping full-run observability must remove these durable execution receipts."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([1.0, 2.0, 1.0]))
        record = module.CommonGridRecord(
            "r0", "c0", source, (module.CommonGridCondition("p08", 0.05, source),),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            progress_path = Path(temporary_directory) / "outside-git-progress.jsonl"
            module.evaluate_common_grid_records(
                (record,), support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
                progress_log_path=progress_path,
            )
            rows = [__import__("json").loads(line) for line in progress_path.read_text().splitlines()]

        self.assertEqual([row["event"] for row in rows], ["dataset_start", "checkpoint", "dataset_end"])
        self.assertEqual([row["completed_record_count"] for row in rows], [0, 1, 1])
        self.assertTrue(all(row["planned_record_count"] == 1 for row in rows))
        self.assertTrue(all(row["elapsed_seconds"] >= 0.0 for row in rows))

    def test_progress_elapsed_can_start_before_record_evaluation(self) -> None:
        """The parent loader's materialization window must be visible in progress elapsed."""
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        source = _spectrum("source", axis, np.asarray([1.0, 2.0, 1.0]))
        record = module.CommonGridRecord(
            "r0", "c0", source, (module.CommonGridCondition("p08", 0.05, source),),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            progress_path = Path(temporary_directory) / "materialization-inclusive-progress.jsonl"
            with mock.patch.object(module.time, "monotonic", side_effect=(15.0, 16.0, 17.0)):
                module.evaluate_common_grid_records(
                    (record,), support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
                    progress_log_path=progress_path, progress_started_at=10.0,
                )
            rows = [__import__("json").loads(line) for line in progress_path.read_text().splitlines()]

        self.assertEqual([row["elapsed_seconds"] for row in rows], [5.0, 6.0, 7.0])

    def test_multiworker_runs_in_independent_processes_with_blas_one(self) -> None:
        """Replacing process execution with threads must collapse observed worker PIDs."""
        module = _module()
        worker_config, detector = _reconstructible_worker_detector(module)
        axis = np.asarray([0.0, 1.0, 2.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=axis.size,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        records = tuple(
            module.CommonGridRecord(
                f"r{index}", f"c{index}", _spectrum(f"r{index}", axis, np.asarray([1.0, 2.0 + index, 1.0])),
                tuple(
                    module.CommonGridCondition("p08", 0.05 + condition * 0.001, _spectrum(f"c{index}-{condition}", axis, np.asarray([1.0, 1.5, 1.0])))
                    for condition in range(40)
                ),
            )
            for index in range(4)
        )
        result = module.evaluate_common_grid_records(
            records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=2, worker_config=worker_config,
        )
        self.assertEqual(result.effective_worker_count, 2)
        self.assertGreaterEqual(len(result.worker_process_ids), 2)
        self.assertNotIn(os.getpid(), result.worker_process_ids)
        self.assertEqual(result.worker_blas_thread_limits, (1,))

    def test_cache_source_identity_binds_cluster_weight_and_frozen_authorities(self) -> None:
        module = _module()
        source = _spectrum("r0", np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 2.0, 1.0]))
        support_axis = np.asarray([0.0, 1.0, 2.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=support_axis, point_count=3,
            axis_sha256_f64=module._array_sha(support_axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        base = module.CommonGridDatasetInput(
            dataset_id="synthetic",
            records=(module.CommonGridRecord("r0", "c0", source, (), 1),),
            planned_record_count=1, support_spec=support, detect_peaks=_flat_detector,
            sweep_sha256="a" * 64, frozen_authority_sha256s={"phase1": "b" * 64},
            p10_memory_budget_bytes=1024,
        )
        changed_cluster = module.CommonGridDatasetInput(
            **{**base.__dict__, "records": (module.CommonGridRecord("r0", "changed", source, (), 1),)}
        )
        changed_weight = module.CommonGridDatasetInput(
            **{**base.__dict__, "records": (module.CommonGridRecord("r0", "c0", source, (), 2),)}
        )
        changed_authority = module.CommonGridDatasetInput(
            **{**base.__dict__, "frozen_authority_sha256s": {"phase1": "c" * 64}}
        )

        self.assertNotEqual(module.common_grid_dataset_source_sha256(base), module.common_grid_dataset_source_sha256(changed_cluster))
        self.assertNotEqual(module.common_grid_dataset_source_sha256(base), module.common_grid_dataset_source_sha256(changed_weight))
        self.assertNotEqual(module.common_grid_dataset_cache_identity(base, "d" * 64), module.common_grid_dataset_cache_identity(changed_authority, "d" * 64))
        identity = module.common_grid_dataset_cache_identity(base, "d" * 64)
        self.assertEqual(identity["frozen_authority_sha256s"], {"phase1": "b" * 64})
        self.assertEqual(identity["p10_memory_budget_bytes"], 1024)
        self.assertEqual(identity["record_provenance_sha256"], module.common_grid_dataset_source_sha256(base))

    def test_every_declared_numerical_dependency_digest_changes_aggregate_identity(self) -> None:
        """Omitting any declared numerical dependency must be detectable in the cache identity."""
        module = _module()
        declared = tuple(module._NUMERICAL_CODE_DEPENDENCY_PATHS)
        baseline_digests = {path: f"digest-{index}" for index, path in enumerate(declared)}
        baseline = module.aggregate_numerical_dependency_identity(baseline_digests)

        for path in declared:
            changed = dict(baseline_digests)
            changed[path] = changed[path] + "-changed"
            self.assertNotEqual(
                baseline, module.aggregate_numerical_dependency_identity(changed), msg=path,
            )

    def test_numerical_dependency_receipt_includes_all_direct_fused_source_adapters(self) -> None:
        """Changing an adapter called by worker rematerialization must invalidate its cache."""
        module = _module()
        declared = set(module._NUMERICAL_CODE_DEPENDENCY_PATHS)

        self.assertTrue({
            "rpe/runner/phase4_d2_protocol_a.py",
            "rpe/runner/phase4_d4_eligibility.py",
            "rpe/runner/phase4_d5_protocol_a.py",
        }.issubset(declared))

    def test_baseline_failure_retains_all_40_condition_rows_and_continues(self) -> None:
        module = _module()
        axis = np.asarray([0.0, 1.0, 2.0])
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=3,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        bad = _spectrum("bad", np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 1.0]))
        good = _spectrum("good", axis, np.asarray([1.0, 2.0, 1.0]))
        conditions = tuple(
            module.CommonGridCondition("p08", float(alpha), good)
            for alpha in (0.05, 0.1)
        )
        result = module.evaluate_common_grid_records(
            (
                module.CommonGridRecord("bad", "c0", bad, conditions),
                module.CommonGridRecord("good", "c1", good, conditions),
            ),
            support_spec=support, detect_peaks=_flat_detector, interpolator="linear",
        )
        self.assertEqual(len(result.failure_rows), 2 * len(module.METRIC_OUTPUT_IDS))
        self.assertTrue(all(row["reason_code"].startswith("source_resample_failure:") for row in result.failure_rows))
        self.assertTrue(all(row["record_id"] == "bad" for row in result.failure_rows))
        self.assertEqual(len(result.aggregated_rows), 2 * len(module.METRIC_OUTPUT_IDS))

    def test_worker_count_one_and_two_are_deterministically_equivalent(self) -> None:
        module = _module()
        worker_config, detector = _reconstructible_worker_detector(module)
        axis = np.asarray([0.0, 1.0, 2.0])
        source = _spectrum("r0", axis, np.asarray([1.0, 2.0, 1.0]))
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=3,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        records = tuple(
            module.CommonGridRecord(
                f"r{index}", "c0", source,
                (module.CommonGridCondition("p08", 0.05, source),),
            )
            for index in range(2)
        )
        one = module.evaluate_common_grid_records(records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=1)
        two = module.evaluate_common_grid_records(records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=2, worker_config=worker_config)
        self.assertEqual(one.aggregated_rows, two.aggregated_rows)
        self.assertEqual(one.failure_rows, two.failure_rows)

    def test_multiworker_evaluation_holds_one_process_blas_limit_for_executor_lifetime(self) -> None:
        """Every process worker must retain the BLAS=1 numerical boundary."""
        module = _module()
        worker_config, detector = _reconstructible_worker_detector(module)
        axis = np.asarray([0.0, 1.0, 2.0])
        source_a = _spectrum("a", axis, np.asarray([1.0, 2.0, 1.0]))
        source_b = _spectrum("b", axis, np.asarray([1.0, 1.0, 2.0]))
        support = module.CommonGridSupportSpec(
            dataset_id="synthetic", axis_cm1=axis, point_count=3,
            axis_sha256_f64=module._array_sha(axis, dtype="<f8"), max_in_range_native_gap_cm1=None,
        )
        records = (
            module.CommonGridRecord(
                "a", "c0", source_a,
                tuple(module.CommonGridCondition("p08", 0.05 + 0.001 * index, source_a) for index in range(40)),
            ),
            module.CommonGridRecord(
                "b", "c1", source_b,
                tuple(module.CommonGridCondition("p08", 0.05 + 0.001 * index, source_b) for index in range(40)),
            ),
        )
        with multiprocessing.Manager() as manager:
            gated_worker_config = module.replace(
                worker_config, process_start_gate=manager.Barrier(2),
            )
            result = module.evaluate_common_grid_records(
                records, support_spec=support, detect_peaks=detector, interpolator="linear", worker_count=2, worker_config=gated_worker_config,
            )

        self.assertEqual(result.effective_worker_count, 2)
        self.assertGreaterEqual(len(result.worker_process_ids), 2)
        self.assertEqual(result.worker_blas_thread_limits, (1,))


if __name__ == "__main__":
    unittest.main()
