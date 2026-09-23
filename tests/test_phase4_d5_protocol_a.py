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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.alignment import AlignmentObservation, paired_cluster_bootstrap  # noqa: E402
from rpe.alignment.bulk import bulk_paired_cluster_bootstrap  # noqa: E402
from rpe.downstream.rruff import (  # noqa: E402
    CONFIG_SHA256 as D5_CONFIG_SHA256,
    D5LibraryQuerySplit,
    D5RawCohort,
)
from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.methods.catalog import load_classical_catalog  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_config import load_phase1_core_config  # noqa: E402
from rpe.runner.phase4_d5_protocol_a import (  # noqa: E402
    ARTIFACT_PAYLOAD_FILES,
    METRIC_OUTPUT_IDS,
    Phase4D5ProtocolAError,
    aggregate_class_observations,
    build_phase4_d5_protocol_a_from_inputs,
    fixed_holm_family,
    load_phase4_d5_protocol_a_config,
    match_d5_protocol_a_799,
    metric_preflight_states,
    parse_phase4_d5_protocol_a_config,
    render_figure_payloads,
)
from rpe.runner.phase4_d5_protocol_a_verifier import (  # noqa: E402
    verify_phase4_d5_protocol_a_from_inputs,
)
from tools.run_phase4_d5_protocol_a import main as protocol_a_cli_main  # noqa: E402


SWEEP_PATH = ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG = ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
CATALOG_PATH = ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
CONFIG_PATH = ROOT / "experiments/phase4/configs/d5_protocol_a_full_domain_v1.json"
CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)


def _read_only(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _canonical(value: object) -> bytes:
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


def _digest(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _synthetic_cohort() -> D5RawCohort:
    axis = _read_only(np.arange(200.0, 1802.0, 2.0, dtype="<f4"))
    values = _read_only(np.ones((6, 801), dtype="<f4"))
    labels = _read_only(np.asarray([0, 0, 0, 1, 1, 1], dtype="<i8"))
    split0 = D5LibraryQuerySplit(
        seed=0,
        query_indices=_read_only(np.asarray([0, 1, 3, 4], dtype="<i8")),
        library_indices=_read_only(np.asarray([2, 5], dtype="<i8")),
        split_sha256="a" * 64,
    )
    split1 = D5LibraryQuerySplit(
        seed=1,
        query_indices=_read_only(np.asarray([0, 1, 5], dtype="<i8")),
        library_indices=_read_only(np.asarray([2, 3, 4], dtype="<i8")),
        split_sha256="b" * 64,
    )
    return D5RawCohort(
        protocol_config_sha256=D5_CONFIG_SHA256,
        dataset_id="rruff_raman_raw",
        intensity=values,
        wavenumber=axis,
        class_labels=labels,
        record_ids=("r0", "r1", "r2", "r3", "r4", "r5"),
        mineral_names=("m0", "m0", "m0", "m1", "m1", "m1"),
        rruff_ids=("s0", "s1", "s2", "s3", "s4", "s5"),
        pin_ids=(None,) * 6,
        group_ids=("g0", "g0", "g1", "h0", "h0", "h1"),
        splits=(split0, split1),
    )


def _synthetic_native(cohort: D5RawCohort) -> tuple[Spectrum1D, ...]:
    axis = np.arange(180.0, 1822.0, 2.0, dtype="<f8")
    output = []
    for index, record_id in enumerate(cohort.record_ids):
        intensity = 1.2 + 0.0002 * (axis - 180.0)
        for peak_index, center in enumerate((360 + 5 * index, 710 + 9 * index, 1260 - 4 * index)):
            intensity = intensity + (1.5 - 0.2 * peak_index) * np.exp(
                -0.5 * ((axis - center) / (10 + 3 * peak_index)) ** 2
            )
        intensity = intensity + 0.01 * np.sin(axis / (9.0 + index))
        output.append(
            Spectrum1D(
                spectrum_id=f"rruff_raman_raw::{record_id}",
                sample_id=cohort.rruff_ids[index],
                axis_cm1=axis,
                intensity=np.asarray(intensity, dtype="<f8"),
            )
        )
    return tuple(output)


def _metric_manifest() -> list[dict[str, object]]:
    directions = {
        "mse": "lower_is_better",
        "rmse": "lower_is_better",
        "mae": "lower_is_better",
        "sam": "lower_is_better",
        "pearson_r": "higher_is_better",
        "nmse": "lower_is_better",
        "wasserstein_1_cm1": "lower_is_better",
        "is_like_structure_to_noise": "higher_is_better",
        "precision": "higher_is_better",
        "recall": "higher_is_better",
        "f1": "higher_is_better",
        "artifact_peak_ratio": "lower_is_better",
        "missing_peak_ratio": "lower_is_better",
    }
    return [
        {"output_id": output_id, "preferred_direction": directions[output_id]}
        for output_id in METRIC_OUTPUT_IDS
    ]


def _synthetic_config(cohort: D5RawCohort):
    query_indices = sorted(
        {int(value) for split in cohort.splits for value in split.query_indices},
        key=lambda index: cohort.record_ids[index],
    )
    occurrence_count = sum(len(split.query_indices) for split in cohort.splits)
    query_ids = [cohort.record_ids[index] for index in query_indices]
    group_ids = sorted({cohort.group_ids[index] for index in query_indices})
    class_ids = sorted({str(int(cohort.class_labels[index])) for index in query_indices})
    query_count = len(query_ids)
    class_count = len(class_ids)
    split_count = len(cohort.splits)
    document = {
        "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS),
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "authorities": {
            "d5_config_sha256": D5_CONFIG_SHA256,
            "phase1_core_config_sha256": hashlib.sha256(PHASE1_CONFIG.read_bytes()).hexdigest(),
            "sweep_sha256": hashlib.sha256(SWEEP_PATH.read_bytes()).hexdigest(),
        },
        "claim_boundary": "local_execution_artifact_redistribution_not_cleared",
        "code_authority": {},
        "cwt_system_id": CWT_SYSTEM_ID,
        "denominators": {
            "class_count": class_count,
            "cohort_record_count": len(cohort.record_ids),
            "query_occurrence_count": occurrence_count,
            "query_record_count": query_count,
            "split_count": split_count,
        },
        "environment_authority": {},
        "expected": {
            "apply_check_count": query_count * 5 * 9,
            "canonical_query_condition_count": query_count * 41,
            "class_observation_count_per_metric": class_count * 40,
            "figure1_row_count": 13 * 40,
            "figure2_row_count": 13,
            "holm_slot_count": 24,
            "matcher_call_count": split_count * 41,
            "operator_cell_count": query_count * 5,
            "prediction_row_count": occurrence_count * 41,
            "projected_unique_row_count": len(cohort.record_ids) + query_count * 40,
            "record_metric_row_count": query_count * 41 * 13,
            "secondary_table_row_count": 13,
        },
        "experiment_id": "phase4-d5-protocol-a-full-domain-v1",
        "figure_contract": {
            "dpi": 300,
            "figure1_inches": [12.0, 12.0],
            "figure2_inches": [14.0, 8.0],
            "font_family": "DejaVu Sans",
            "svg_hashsalt": "rpe-phase4-d5-v1",
        },
        "frozen_identities": {
            "query_class_labels_sha256": _digest(class_ids),
            "query_group_ids_sha256": _digest(group_ids),
            "query_record_ids_sha256": _digest(query_ids),
            "split_sha256": [split.split_sha256 for split in cohort.splits],
        },
        "inference": {
            "bootstrap_resamples": 8,
            "confidence_level": 0.95,
            "holm_alpha": 0.05,
            "random_seed": 20260817,
            "sign_flip_resamples": 64,
        },
        "metric_manifest": _metric_manifest(),
        "schema_version": "phase4-d5-protocol-a-full-domain-config-v1",
        "support_grid": {
            "max_in_range_native_gap_cm1": 3.0,
            "point_count": 799,
            "start_cm1": 204.0,
            "step_cm1": 2.0,
            "stop_cm1": 1800.0,
        },
        "synthetic_fixture": True,
    }
    raw = _canonical(document)
    return parse_phase4_d5_protocol_a_config(
        Path("synthetic.json"), raw, require_frozen_identity=False
    )


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }


def _two_cluster_tables():
    reference = []
    candidate = []
    for cluster_id in ("c1", "c2"):
        for perturbation_id, alpha, metric, downstream, perfect in (
            ("p08", 0.1, 0.0, 0.0, 0.0),
            ("p08", 0.2, 1.0, 1.0, 1.0),
            ("p09", 0.1, 0.0, 2.0, 2.0),
            ("p09", 0.2, 1.0, 3.0, 3.0),
        ):
            reference.append(AlignmentObservation(cluster_id, perturbation_id, alpha, metric, downstream))
            candidate.append(AlignmentObservation(cluster_id, perturbation_id, alpha, perfect, downstream))
    return tuple(reference), tuple(candidate)


class FrozenProtocolAConfigTest(unittest.TestCase):
    def test_loads_exact_real_config_and_rejects_drift(self) -> None:
        config = load_phase4_d5_protocol_a_config(CONFIG_PATH)
        self.assertEqual(config.perturbation_ids, PERTURBATIONS)
        self.assertEqual(config.metric_output_ids, METRIC_OUTPUT_IDS)
        self.assertEqual(config.query_record_count, 3012)
        self.assertEqual(config.query_occurrence_count, 6621)
        self.assertEqual(config.class_count, 681)
        self.assertEqual(config.expected_matcher_call_count, 205)
        self.assertEqual(config.expected_prediction_row_count, 271461)
        self.assertEqual(config.expected_holm_slot_count, 24)
        self.assertNotIn("p01", config.document["active_perturbation_ids"])
        self.assertEqual(config.document["protocol"], "A")

        changed = json.loads(CONFIG_PATH.read_text())
        changed["protocol"] = "B"
        with self.assertRaisesRegex(Phase4D5ProtocolAError, "frozen config identity"):
            parse_phase4_d5_protocol_a_config(
                Path("drift.json"), _canonical(changed), require_frozen_identity=True
            )


class ScientificAdapterTest(unittest.TestCase):
    def test_799_matcher_uses_max_record_class_score_and_label_ties(self) -> None:
        cohort = _synthetic_cohort()
        split = cohort.splits[0]
        query_ids = tuple(cohort.record_ids[int(index)] for index in split.query_indices)
        library_ids = tuple(cohort.record_ids[int(index)] for index in split.library_indices)
        library = np.zeros((2, 799), dtype="<f8")
        library[0, 0] = 1.0
        library[1, 1] = 1.0
        query = np.zeros((4, 799), dtype="<f8")
        query[0, 0] = 1.0
        query[1, 1] = 1.0
        query[2, :2] = 1.0
        query[3, :2] = 1.0
        result = match_d5_protocol_a_799(
            cohort,
            split,
            condition_id="fixture",
            query_record_ids=query_ids,
            library_record_ids=library_ids,
            query_values=query,
            library_values=library,
        )
        np.testing.assert_array_equal(result.ranked_class_labels[0, :2], [0, 1])
        np.testing.assert_array_equal(result.ranked_class_labels[1, :2], [1, 0])
        np.testing.assert_array_equal(result.ranked_class_labels[2, :2], [0, 1])

    def test_occurrence_weighted_class_aggregation_and_metric_harms(self) -> None:
        condition = f"p08:{struct.pack('<d', 0.1).hex()}"
        occurrence_rows = [
            {"class_label": 0, "condition_id": "alpha0", "record_id": "r0", "top1_correct": True},
            {"class_label": 0, "condition_id": condition, "record_id": "r0", "top1_correct": False},
            {"class_label": 0, "condition_id": "alpha0", "record_id": "r0", "top1_correct": True},
            {"class_label": 0, "condition_id": condition, "record_id": "r0", "top1_correct": True},
            {"class_label": 0, "condition_id": "alpha0", "record_id": "r1", "top1_correct": False},
            {"class_label": 0, "condition_id": condition, "record_id": "r1", "top1_correct": False},
        ]
        metric_rows = {
            ("r0", "alpha0", "mse"): 0.0,
            ("r0", condition, "mse"): 2.0,
            ("r1", "alpha0", "mse"): 0.0,
            ("r1", condition, "mse"): 8.0,
        }
        rows = aggregate_class_observations(
            occurrence_rows,
            metric_rows,
            metric_output_id="mse",
            preferred_direction="lower_is_better",
            positive_conditions=(condition,),
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["downstream_harm"], 1.0 / 3.0)
        self.assertAlmostEqual(rows[0]["metric_harm"], 4.0)

    def test_metric_failure_closes_mse_or_only_candidate_and_keeps_24_slots(self) -> None:
        complete = {output_id: {"complete": 10, "planned": 10} for output_id in METRIC_OUTPUT_IDS}
        states = metric_preflight_states(complete)
        self.assertTrue(all(value["state"] == "complete" for value in states.values()))
        candidate_failed = {key: dict(value) for key, value in complete.items()}
        candidate_failed["sam"] = {"complete": 9, "planned": 10}
        states = metric_preflight_states(candidate_failed)
        self.assertEqual(states["sam"]["state"], "not_evaluable_incomplete_grid")
        family = fixed_holm_family(
            {"rmse:d_ag": 0.01, "rmse:d_acc": 0.02},
            states,
            {"rmse:d_ag": 0.1, "rmse:d_acc": 0.2},
        )
        self.assertEqual(len(family), 24)
        sam = [row for row in family if row["metric_output_id"] == "sam"]
        self.assertEqual([row["multiplicity_p_value"] for row in sam], [1.0, 1.0])
        self.assertTrue(all(row["state"] == "not_tested_metric_incomplete" for row in sam))
        mse_failed = {key: dict(value) for key, value in complete.items()}
        mse_failed["mse"] = {"complete": 9, "planned": 10}
        with self.assertRaisesRegex(Phase4D5ProtocolAError, "MSE comparator"):
            metric_preflight_states(mse_failed)

    def test_holm_rejection_requires_favorable_observed_direction(self) -> None:
        complete = {
            output_id: {"complete": 10, "planned": 10, "state": "complete"}
            for output_id in METRIC_OUTPUT_IDS
        }
        family = fixed_holm_family(
            {"rmse:d_ag": 0.0, "rmse:d_acc": 0.0},
            complete,
            {"rmse:d_ag": -0.25, "rmse:d_acc": 0.25},
        )
        by_statistic = {
            row["statistic"]: row
            for row in family
            if row["metric_output_id"] == "rmse"
        }
        self.assertEqual(by_statistic["d_ag"]["adjusted_p_value"], 0.0)
        self.assertFalse(by_statistic["d_ag"]["favorable"])
        self.assertFalse(by_statistic["d_ag"]["rejected"])
        self.assertEqual(by_statistic["d_acc"]["adjusted_p_value"], 0.0)
        self.assertTrue(by_statistic["d_acc"]["favorable"])
        self.assertTrue(by_statistic["d_acc"]["rejected"])


class BulkInferenceTest(unittest.TestCase):
    def test_bulk_bootstrap_matches_existing_point_oracle(self) -> None:
        reference, candidate = _two_cluster_tables()
        expected = paired_cluster_bootstrap(reference, candidate, resamples=32)
        observed = bulk_paired_cluster_bootstrap(reference, candidate, resamples=32)
        self.assertEqual(observed, expected)

    def test_bulk_bootstrap_matches_existing_random_complete_table(self) -> None:
        generator = np.random.Generator(np.random.PCG64(17))
        reference = []
        candidate = []
        for cluster in range(4):
            for perturbation in ("p08", "p09", "p10"):
                for alpha in (0.1, 0.2, 0.4):
                    downstream = float(generator.normal())
                    left = float(generator.normal())
                    right = float(generator.normal())
                    reference.append(AlignmentObservation(str(cluster), perturbation, alpha, left, downstream))
                    candidate.append(AlignmentObservation(str(cluster), perturbation, alpha, right, downstream))
        expected = paired_cluster_bootstrap(tuple(reference), tuple(candidate), resamples=25)
        observed = bulk_paired_cluster_bootstrap(tuple(reference), tuple(candidate), resamples=25)
        for name in (
            "reference_ag_interval",
            "candidate_ag_interval",
            "d_ag_interval",
            "reference_acc_interval",
            "candidate_acc_interval",
            "d_acc_interval",
        ):
            np.testing.assert_allclose(getattr(observed, name), getattr(expected, name), rtol=0, atol=1e-14)
        self.assertEqual(observed.resamples, expected.resamples)
        self.assertEqual(observed.duplicate_cluster_resamples, expected.duplicate_cluster_resamples)


class FigureAndArtifactTest(unittest.TestCase):
    def test_figure_projection_bytes_are_deterministic(self) -> None:
        figure1 = [
            {
                "metric_output_id": metric,
                "perturbation_id": perturbation,
                "alpha": alpha,
                "mean_metric_harm": float(index + alpha),
                "mean_downstream_harm": float(index / 10 + alpha / 2),
                "metric_state": "complete",
            }
            for index, metric in enumerate(METRIC_OUTPUT_IDS)
            for perturbation in PERTURBATIONS
            for alpha in ALPHAS[1:]
        ]
        figure2 = [
            {
                "metric_output_id": metric,
                "metric_state": "complete",
                "ag": 0.1 + index / 100,
                "ag_lower": 0.05,
                "ag_upper": 0.3,
                "acc": 0.5,
                "acc_lower": 0.4,
                "acc_upper": 0.6,
                "d_ag": None if metric == "mse" else 0.01,
                "d_ag_lower": None if metric == "mse" else -0.01,
                "d_ag_upper": None if metric == "mse" else 0.03,
                "d_acc": None if metric == "mse" else 0.02,
                "d_acc_lower": None if metric == "mse" else 0.0,
                "d_acc_upper": None if metric == "mse" else 0.04,
                "d_ag_favorable": None if metric == "mse" else index % 2 == 0,
                "d_acc_favorable": None if metric == "mse" else index % 2 != 0,
                "d_ag_raw_p": None if metric == "mse" else 0.01,
                "d_ag_adjusted_p": None if metric == "mse" else 0.5,
                "d_acc_raw_p": None if metric == "mse" else 0.02,
                "d_acc_adjusted_p": None if metric == "mse" else 0.5,
            }
            for index, metric in enumerate(METRIC_OUTPUT_IDS)
        ]
        first = render_figure_payloads(figure1, figure2)
        second = render_figure_payloads(figure1, figure2)
        self.assertEqual(first, second)
        self.assertEqual(len(figure1), 520)
        self.assertEqual(len(figure2), 13)
        self.assertTrue(first["figure1_d5_protocol_a_full_domain.png"].startswith(b"\x89PNG"))
        figure2_svg = first["figure2_d5_protocol_a_full_domain.svg"]
        self.assertIn(b"<svg", figure2_svg)
        self.assertIn(b"raw=0.01; adj=0.5; favorable", figure2_svg)
        self.assertIn(b"raw=0.01; adj=0.5; unfavorable", figure2_svg)
        self.assertIn(b"raw=0.02; adj=0.5; favorable", figure2_svg)

        ineligible1 = [dict(row) for row in figure1]
        for row in ineligible1:
            if row["metric_output_id"] == "sam":
                row["metric_state"] = "not_evaluable_incomplete_grid"
                row["mean_metric_harm"] = None
                row["mean_downstream_harm"] = None
        ineligible2 = [dict(row) for row in figure2]
        for row in ineligible2:
            if row["metric_output_id"] == "sam":
                row.update({key: None for key in row if key not in {"metric_output_id", "metric_state"}})
                row["metric_state"] = "not_evaluable_incomplete_grid"
        payloads = render_figure_payloads(ineligible1, ineligible2)
        self.assertTrue(payloads["figure1_d5_protocol_a_full_domain.png"].startswith(b"\x89PNG"))
        self.assertIn(
            b"not_evaluable_incomplete_grid",
            payloads["figure2_d5_protocol_a_full_domain.svg"],
        )

    def test_synthetic_build_and_independent_verifier_are_byte_identical(self) -> None:
        cohort = _synthetic_cohort()
        native = _synthetic_native(cohort)
        sweep = load_perturbation_sweep_config(SWEEP_PATH)
        phase1 = load_phase1_core_config(PHASE1_CONFIG)
        catalog = load_classical_catalog(CATALOG_PATH, project_root=ROOT)
        config = _synthetic_config(cohort)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            summary = build_phase4_d5_protocol_a_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                classical_catalog=catalog,
                config=config,
                worker_count=2,
                inference_resamples=8,
            )
            verified = verify_phase4_d5_protocol_a_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                classical_catalog=catalog,
                worker_count=1,
                inference_resamples=8,
            )
            self.assertEqual(summary.run_id, verified.run_id)
            self.assertEqual(summary.status, verified.status)
            observed = {item.name for item in output.iterdir() if item.is_file()}
            self.assertEqual(len(observed & {"complete.json", "failed.json"}), 1)
            self.assertEqual(observed, set(ARTIFACT_PAYLOAD_FILES) | {"SHA256SUMS"})
            manifest = json.loads((output / "manifest.json").read_bytes())
            self.assertEqual(manifest["protocol"], "A")
            self.assertEqual(manifest["perturbation_ids"], list(PERTURBATIONS))
            self.assertNotIn("p01", json.dumps(manifest))
            self.assertEqual(manifest["claim_boundary"], "local_execution_artifact_redistribution_not_cleared")
            metric_rows = [json.loads(line) for line in (output / "metric_values.jsonl").read_text().splitlines()]
            condition_order = {condition: index for index, condition in enumerate(("alpha0",) + tuple(
                f"{perturbation}:{np.float64(alpha).tobytes().hex()}"
                for perturbation in PERTURBATIONS for alpha in ALPHAS[1:]
            ))}
            metric_order = {metric: index for index, metric in enumerate(METRIC_OUTPUT_IDS)}
            observed_keys = [
                (int(row["record_order"]), condition_order[row["condition_id"]], metric_order[row["metric_output_id"]])
                for row in metric_rows
            ]
            self.assertEqual(observed_keys, sorted(observed_keys))

    def test_cli_does_not_expose_skip_reexecution_or_protocol_b(self) -> None:
        with self.assertRaises(SystemExit):
            protocol_a_cli_main(["verify", "--run-path", "x", "--no-reexecute"])
        with self.assertRaises(SystemExit):
            protocol_a_cli_main(["build", "--output-root", "x", "--protocol", "B"])


if __name__ == "__main__":
    unittest.main()
