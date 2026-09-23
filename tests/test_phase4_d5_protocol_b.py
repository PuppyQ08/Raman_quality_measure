from __future__ import annotations

import hashlib
import json
import math
import importlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from threadpoolctl import threadpool_info

import numpy as np
import rpe.runner.phase4_d5_protocol_b as protocol_b_module

from rpe.downstream.rruff import CONFIG_SHA256 as D5_CONFIG_SHA256
from rpe.downstream.rruff import D5LibraryQuerySplit, D5RawCohort
from rpe.runner.phase4_d5_protocol_b import (
    ARTIFACT_PAYLOAD_FILES,
    CONDITION_BRIDGE_SHA256,
    METRIC_OUTPUT_IDS,
    OPERATOR_BRIDGE_SHA256,
    Phase4D5ProtocolBError,
    aggregate_protocol_b_class_observations,
    build_phase4_d5_protocol_b_from_inputs,
    build_protocol_b_authority_bridge,
    fixed_protocol_b_holm_family,
    load_phase4_d5_protocol_b_config,
    match_d5_protocol_b_799,
    parse_phase4_d5_protocol_b_config,
    render_protocol_b_figure_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.1)
POSITIVE_CONDITIONS = tuple(
    f"{perturbation}:{np.float64(0.1).tobytes().hex()}"
    for perturbation in PERTURBATIONS
)
CONDITIONS = ("alpha0",) + POSITIVE_CONDITIONS
DIRECTIONS = {
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


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ids_digest(values: list[str]) -> str:
    return _sha(("\n".join(sorted(values)) + "\n").encode())


def _array_sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _cohort() -> D5RawCohort:
    labels = _read_only(np.asarray([0, 0, 0, 1, 1, 1], dtype="<i8"))
    split0 = D5LibraryQuerySplit(
        seed=0,
        query_indices=_read_only(np.asarray([0, 1, 3], dtype="<i8")),
        library_indices=_read_only(np.asarray([2, 4, 5], dtype="<i8")),
        split_sha256="a" * 64,
    )
    split1 = D5LibraryQuerySplit(
        seed=1,
        query_indices=_read_only(np.asarray([0, 3, 4], dtype="<i8")),
        library_indices=_read_only(np.asarray([1, 2, 5], dtype="<i8")),
        split_sha256="b" * 64,
    )
    return D5RawCohort(
        protocol_config_sha256=D5_CONFIG_SHA256,
        dataset_id="rruff_raman_raw",
        intensity=_read_only(np.ones((6, 801), dtype="<f4")),
        wavenumber=_read_only(np.arange(200.0, 1802.0, 2.0, dtype="<f4")),
        class_labels=labels,
        record_ids=("r0", "r1", "r2", "r3", "r4", "r5"),
        mineral_names=("m0", "m0", "m0", "m1", "m1", "m1"),
        rruff_ids=("s0", "s1", "s2", "s3", "s4", "s5"),
        pin_ids=(None,) * 6,
        group_ids=("g0", "g0", "g1", "h0", "h0", "h1"),
        splits=(split0, split1),
    )


def _query_indices(cohort: D5RawCohort) -> list[int]:
    return sorted(
        {int(index) for split in cohort.splits for index in split.query_indices},
        key=lambda index: cohort.record_ids[index],
    )


def _condition_rows(cohort: D5RawCohort, *, query_only: bool) -> list[dict[str, object]]:
    indices = _query_indices(cohort) if query_only else list(range(len(cohort.record_ids)))
    rows: list[dict[str, object]] = []
    for record_order, cohort_index in enumerate(indices):
        record_id = cohort.record_ids[cohort_index]
        for condition_id in CONDITIONS:
            rows.append(
                {
                    "record_order": record_order if query_only else cohort_index,
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "axis_sha256": _array_sha(f"axis:{record_id}:{condition_id}"),
                    "intensity_sha256": _array_sha(f"intensity:{record_id}:{condition_id}"),
                    ("projection_sha256" if query_only else "support_projection_sha256"): _array_sha(
                        f"projection:{record_id}:{condition_id}"
                    ),
                    **(
                        {}
                        if query_only
                        else {
                            "alpha": 0.0 if condition_id == "alpha0" else 0.1,
                            "alpha_float64_le_hex": (
                                np.float64(0.0).tobytes().hex()
                                if condition_id == "alpha0"
                                else np.float64(0.1).tobytes().hex()
                            ),
                            "class_label": int(cohort.class_labels[cohort_index]),
                            "condition_kind": "alpha0" if condition_id == "alpha0" else "positive",
                            "group_id": cohort.group_ids[cohort_index],
                            "perturbation_id": None if condition_id == "alpha0" else condition_id.split(":", 1)[0],
                            "state": "complete",
                            "support_max_in_range_gap_cm1": 2.0,
                            "support_point_count": 799,
                        }
                    ),
                }
            )
    return rows


def _operator_rows(cohort: D5RawCohort, *, query_only: bool) -> list[dict[str, object]]:
    indices = _query_indices(cohort) if query_only else list(range(len(cohort.record_ids)))
    rows: list[dict[str, object]] = []
    for record_order, cohort_index in enumerate(indices):
        record_id = cohort.record_ids[cohort_index]
        for perturbation in PERTURBATIONS:
            outputs = []
            for alpha in ALPHAS:
                alpha_hex = np.float64(alpha).tobytes().hex()
                base = {
                    "alpha": alpha,
                    "alpha_float64_le_hex": alpha_hex,
                    "diagnostics": {"synthetic": True, "alpha": alpha},
                    "output_spectrum_id": f"{record_id}:{perturbation}:{alpha_hex}",
                    ("axis_sha256" if query_only else "output_axis_sha256"): _array_sha(
                        f"axis:{record_id}:{perturbation}:{alpha_hex}"
                    ),
                    ("intensity_sha256" if query_only else "output_intensity_sha256"): _array_sha(
                        f"intensity:{record_id}:{perturbation}:{alpha_hex}"
                    ),
                }
                if not query_only:
                    base.update(
                        {
                            "axis_changed": alpha > 0 and perturbation in {"p11", "p12"},
                            "intensity_changed": alpha > 0 and perturbation not in {"p11", "p12"},
                            "support_max_in_range_gap_cm1": 2.0,
                            "support_point_count": 799,
                            "support_projection_sha256": _array_sha(
                                f"projection:{record_id}:{perturbation}:{alpha_hex}"
                            ),
                        }
                    )
                outputs.append(base)
            common = {
                "record_id": record_id,
                "perturbation_id": perturbation,
                "state_digest": _array_sha(f"state:{record_id}:{perturbation}"),
                "native_gate": {"synthetic": True},
                "outputs": outputs,
            }
            if query_only:
                common["record_order"] = record_order
            else:
                common.update(
                    {
                        "class_label": int(cohort.class_labels[cohort_index]),
                        "exception": None,
                        "group_id": cohort.group_ids[cohort_index],
                        "output_count": len(outputs),
                        "p10_estimated_peak_bytes": None,
                        "reason_code": None,
                        "state": "complete",
                    }
                )
            rows.append(common)
    return rows


def _metric_rows(cohort: D5RawCohort) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record_order, cohort_index in enumerate(_query_indices(cohort)):
        record_id = cohort.record_ids[cohort_index]
        for condition_index, condition_id in enumerate(CONDITIONS):
            for metric_index, metric in enumerate(METRIC_OUTPUT_IDS):
                rows.append(
                    {
                        "record_order": record_order,
                        "record_id": record_id,
                        "condition_id": condition_id,
                        "metric_output_id": metric,
                        "state": "complete",
                        "value": float(record_order + condition_index + metric_index / 100.0),
                        "diagnostics": {},
                        "exception": None,
                    }
                )
    return rows


def _peak_rows(cohort: D5RawCohort) -> list[dict[str, object]]:
    rows = []
    for record_order, cohort_index in enumerate(_query_indices(cohort)):
        record_id = cohort.record_ids[cohort_index]
        for condition_id in CONDITIONS:
            rows.append(
                {
                    "record_order": record_order,
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "status": "complete",
                    "peaks_sha256": _array_sha(f"peaks:{record_id}:{condition_id}"),
                    "peak_count": 1,
                    "warnings": [],
                    "diagnostics": {},
                    "error_code": None,
                    "error_message": None,
                }
            )
    return rows


def _unique_records(cohort: D5RawCohort) -> list[dict[str, object]]:
    query_counts = {index: 0 for index in range(len(cohort.record_ids))}
    library_counts = {index: 0 for index in range(len(cohort.record_ids))}
    for split in cohort.splits:
        for index in split.query_indices:
            query_counts[int(index)] += 1
        for index in split.library_indices:
            library_counts[int(index)] += 1
    return [
        {
            "record_order": index,
            "cohort_index": index,
            "record_id": cohort.record_ids[index],
            "group_id": cohort.group_ids[index],
            "class_label": int(cohort.class_labels[index]),
            "query_occurrence_count": query_counts[index],
            "library_occurrence_count": library_counts[index],
            "role_occurrence_count": query_counts[index] + library_counts[index],
        }
        for index in range(len(cohort.record_ids))
    ]


def _role_rows(cohort: D5RawCohort) -> list[dict[str, object]]:
    rows = []
    for split in cohort.splits:
        for role, indices in (("query", split.query_indices), ("library", split.library_indices)):
            for role_order, raw_index in enumerate(indices):
                index = int(raw_index)
                rows.append(
                    {
                        "split_seed": int(split.seed),
                        "split_sha256": split.split_sha256,
                        "role": role,
                        "role_order": role_order,
                        "cohort_index": index,
                        "record_id": cohort.record_ids[index],
                        "group_id": cohort.group_ids[index],
                        "class_label": int(cohort.class_labels[index]),
                        "unique_record_order": index,
                    }
                )
    return rows


def _write_payloads(path: Path, payloads: dict[str, bytes]) -> None:
    path.mkdir(parents=True)
    for name, value in payloads.items():
        path.joinpath(name).write_bytes(value)
    checksummed = [name for name in payloads if name != "SHA256SUMS"]
    path.joinpath("SHA256SUMS").write_text(
        "".join(f"{_sha(payloads[name])}  {name}\n" for name in checksummed),
        encoding="utf-8",
    )


def _condition_bridge_digest(rows: list[dict[str, object]]) -> str:
    normalized = [
        {
            "record_id": row["record_id"],
            "condition_id": row["condition_id"],
            "axis_sha256": row["axis_sha256"],
            "intensity_sha256": row["intensity_sha256"],
            "projection_sha256": row["projection_sha256"],
        }
        for row in rows
    ]
    normalized.sort(key=lambda row: f"{row['record_id']}|{row['condition_id']}")
    return _sha(_jsonl(normalized))


def _operator_bridge_digest(rows: list[dict[str, object]]) -> str:
    normalized = [
        {
            "record_id": row["record_id"],
            "perturbation_id": row["perturbation_id"],
            "state_digest": row["state_digest"],
            "native_gate": row["native_gate"],
            "outputs": [
                {
                    "alpha": output["alpha"],
                    "alpha_float64_le_hex": output["alpha_float64_le_hex"],
                    "diagnostics": output["diagnostics"],
                    "output_spectrum_id": output["output_spectrum_id"],
                    "axis_sha256": output["axis_sha256"],
                    "intensity_sha256": output["intensity_sha256"],
                }
                for output in row["outputs"]
            ],
        }
        for row in rows
    ]
    normalized.sort(key=lambda row: f"{row['record_id']}|{row['perturbation_id']}")
    return _sha(_jsonl(normalized))


def _parents(root: Path, cohort: D5RawCohort) -> tuple[Path, Path, dict[str, object]]:
    step7 = root / "step7"
    step9 = root / "step9"
    step7_conditions = _condition_rows(cohort, query_only=True)
    step9_conditions = _condition_rows(cohort, query_only=False)
    step7_operators = _operator_rows(cohort, query_only=True)
    step9_operators = _operator_rows(cohort, query_only=False)
    metric_rows = _metric_rows(cohort)
    peak_rows = _peak_rows(cohort)
    step7_payloads = {
        "config.json": _canonical({"protocol": "A", "synthetic_fixture": True}),
        "manifest.json": _canonical(
            {
                "protocol": "A",
                "status": "complete",
                "counts": {
                    "operator_cells": len(step7_operators),
                    "record_conditions": len(step7_conditions),
                    "metric_values": len(metric_rows),
                    "peak_receipts": len(peak_rows),
                },
            }
        ),
        "complete.json": _canonical({"status": "complete"}),
        "operator_cells.jsonl": _jsonl(step7_operators),
        "record_conditions.jsonl": _jsonl(step7_conditions),
        "metric_values.jsonl": _jsonl(metric_rows),
        "peak_receipts.jsonl": _jsonl(peak_rows),
        # These files must be ignored, not rejected merely for existing.
        "matcher_predictions.jsonl": b"forbidden sentinel\n",
        "class_observations.jsonl": b"forbidden sentinel\n",
        "alignment_results.jsonl": b"forbidden sentinel\n",
    }
    _write_payloads(step7, step7_payloads)
    step9_payloads = {
        "config.json": _canonical({"protocol": "B", "synthetic_fixture": True}),
        "manifest.json": _canonical(
            {
                "protocol": "B",
                "status": "complete",
                "counts": {
                    "unique_records": len(cohort.record_ids),
                    "role_occurrences": sum(len(s.query_indices) + len(s.library_indices) for s in cohort.splits),
                    "operator_cells": len(step9_operators),
                    "record_conditions": len(step9_conditions),
                },
            }
        ),
        "complete.json": _canonical({"status": "pass"}),
        "unique_records.jsonl": _jsonl(_unique_records(cohort)),
        "role_occurrences.jsonl": _jsonl(_role_rows(cohort)),
        "operator_cells.jsonl": _jsonl(step9_operators),
        "record_conditions.jsonl": _jsonl(step9_conditions),
    }
    _write_payloads(step9, step9_payloads)
    step7_sha = {name: _sha(path.read_bytes()) for name, path in ((name, step7 / name) for name in step7_payloads)}
    step7_sha["SHA256SUMS"] = _sha((step7 / "SHA256SUMS").read_bytes())
    step9_sha = {name: _sha(path.read_bytes()) for name, path in ((name, step9 / name) for name in step9_payloads)}
    step9_sha["SHA256SUMS"] = _sha((step9 / "SHA256SUMS").read_bytes())
    query_indices = _query_indices(cohort)
    query_record_ids = [cohort.record_ids[index] for index in query_indices]
    query_occurrences = sum(len(split.query_indices) for split in cohort.splits)
    config_document: dict[str, object] = {
        "schema_version": "phase4-d5-protocol-b-full-domain-config-v1",
        "experiment_id": "phase4-d5-protocol-b-full-domain-v1",
        "protocol": "B",
        "tier": "full_domain_core",
        "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS),
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "authorities": {
            "d5_config_sha256": D5_CONFIG_SHA256,
            "sweep_sha256": "1" * 64,
            "phase1_core_config_sha256": "2" * 64,
            "parent_plan_sha256": "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5",
        },
        "parent_artifacts": {
            "protocol_a": {
                "relative_path": "synthetic-step7",
                "payload_sha256": step7_sha,
            },
            "eligibility": {
                "relative_path": "synthetic-step9",
                "payload_sha256": step9_sha,
            },
        },
        "authority_bridge": {
            "condition_bridge_sha256": _condition_bridge_digest(step7_conditions),
            "operator_bridge_sha256": _operator_bridge_digest(step7_operators),
        },
        "metric_manifest": [
            {"output_id": metric, "preferred_direction": DIRECTIONS[metric]}
            for metric in METRIC_OUTPUT_IDS
        ],
        "denominators": {
            "full_record_count": len(cohort.record_ids),
            "group_count": len(set(cohort.group_ids)),
            "class_count": len(set(int(value) for value in cohort.class_labels)),
            "query_record_count": len(query_record_ids),
            "query_occurrence_count": query_occurrences,
            "library_occurrence_count": sum(len(split.library_indices) for split in cohort.splits),
            "split_count": len(cohort.splits),
        },
        "expected": {
            "operator_cell_count": len(cohort.record_ids) * len(PERTURBATIONS),
            "apply_check_count": len(cohort.record_ids) * len(PERTURBATIONS) * len(ALPHAS),
            "full_condition_count": len(cohort.record_ids) * len(CONDITIONS),
            "query_condition_count": len(query_record_ids) * len(CONDITIONS),
            "metric_row_count": len(query_record_ids) * len(CONDITIONS) * len(METRIC_OUTPUT_IDS),
            "peak_receipt_count": len(query_record_ids) * len(CONDITIONS),
            "matcher_call_count": len(cohort.splits) * len(CONDITIONS),
            "prediction_row_count": query_occurrences * len(CONDITIONS),
            "class_observation_count_per_metric": len(set(int(value) for value in cohort.class_labels)) * len(PERTURBATIONS),
            "class_observation_count": len(set(int(value) for value in cohort.class_labels)) * len(PERTURBATIONS) * len(METRIC_OUTPUT_IDS),
            "holm_slot_count": 24,
            "figure1_row_count": 13 * 5,
            "figure2_row_count": 13,
            "secondary_table_row_count": 13,
        },
        "frozen_identities": {
            "record_ids": list(cohort.record_ids),
            "group_ids": list(cohort.group_ids),
            "class_labels": [int(value) for value in cohort.class_labels],
            "query_record_ids": query_record_ids,
            "split_sha256": [split.split_sha256 for split in cohort.splits],
            "record_ids_sha256": _ids_digest(list(cohort.record_ids)),
            "group_ids_sha256": _ids_digest(sorted(set(cohort.group_ids))),
            "class_labels_sha256": _ids_digest(
                [str(value) for value in sorted(set(int(v) for v in cohort.class_labels))]
            ),
            "query_record_ids_sha256": _ids_digest(query_record_ids),
        },
        "inference": {
            "bootstrap_resamples": 8,
            "sign_flip_resamples": 64,
            "random_seed": 20260817,
            "confidence_level": 0.95,
            "holm_alpha": 0.05,
        },
        "support_grid": {
            "start_cm1": 204.0,
            "stop_cm1": 1800.0,
            "step_cm1": 2.0,
            "point_count": 799,
            "max_in_range_native_gap_cm1": 3.0,
        },
        "figure_contract": {
            "svg_hashsalt": "rpe-phase4-d5-protocol-b-v1",
            "dpi": 300,
        },
        "inherited_rulings": {
            "p01_p05": "not_evaluable_coverage",
            "p06_p07": "structurally_ineligible_missing_explicit_baseline",
        },
        "claim_boundary": "local_execution_artifact_redistribution_not_cleared",
        "code_authority": {},
        "environment_authority": {},
        "trust_anchor": {
            "config_authority_relative_path": "rpe/runner/phase4_d5_protocol_b_authority.py"
        },
        "synthetic_fixture": True,
    }
    return step7, step9, config_document


class ProtocolBConfigTest(unittest.TestCase):
    def test_numeric_class_digest_does_not_reapply_lexicographic_sort(self) -> None:
        expected = _sha(b"1\n2\n10\n")
        self.assertEqual(protocol_b_module._class_digest([10, 1, 2]), expected)

    def test_loads_exact_real_config_and_rejects_byte_drift(self) -> None:
        path = ROOT / "experiments/phase4/configs/d5_protocol_b_full_domain_v1.json"
        config = load_phase4_d5_protocol_b_config(path)
        self.assertEqual(config.protocol, "B")
        self.assertEqual(config.tier, "full_domain_core")
        self.assertEqual(config.full_record_count, 3770)
        self.assertEqual(config.query_record_count, 3012)
        self.assertEqual(config.query_occurrence_count, 6621)
        self.assertEqual(config.library_occurrence_count, 12229)
        self.assertEqual(config.expected_matcher_call_count, 205)
        self.assertEqual(config.expected_prediction_row_count, 271461)
        self.assertEqual(config.expected_class_observation_count, 354120)
        self.assertEqual(config.expected_holm_slot_count, 24)
        self.assertEqual(config.condition_bridge_sha256, CONDITION_BRIDGE_SHA256)
        self.assertEqual(config.operator_bridge_sha256, OPERATOR_BRIDGE_SHA256)
        changed = json.loads(path.read_bytes())
        changed["protocol"] = "A"
        with self.assertRaisesRegex(Phase4D5ProtocolBError, "frozen config identity"):
            parse_phase4_d5_protocol_b_config(
                Path("drift.json"), _canonical(changed), require_frozen_identity=True
            )

    def test_synthetic_config_derives_exact_counts_and_rejects_scope_drift(self) -> None:
        cohort = _cohort()
        with tempfile.TemporaryDirectory() as temporary:
            _, _, document = _parents(Path(temporary), cohort)
            config = parse_phase4_d5_protocol_b_config(
                Path("synthetic.json"), _canonical(document), require_frozen_identity=False
            )
            self.assertEqual(config.protocol, "B")
            self.assertEqual(config.perturbation_ids, PERTURBATIONS)
            self.assertEqual(config.expected_full_condition_count, 36)
            self.assertEqual(config.expected_metric_row_count, 312)
            self.assertEqual(config.expected_holm_slot_count, 24)
            changed = json.loads(_canonical(document))
            changed["protocol"] = "A"
            with self.assertRaisesRegex(Phase4D5ProtocolBError, "protocol"):
                parse_phase4_d5_protocol_b_config(
                    Path("drift.json"), _canonical(changed), require_frozen_identity=False
                )


class ProtocolBAuthorityBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cohort = _cohort()
        self.step7, self.step9, document = _parents(self.root, self.cohort)
        self.config = parse_phase4_d5_protocol_b_config(
            self.root / "config.json", _canonical(document), require_frozen_identity=False
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self):
        return build_protocol_b_authority_bridge(
            cohort=self.cohort,
            protocol_a_path=self.step7,
            eligibility_path=self.step9,
            config=self.config,
        )

    def test_bridge_accepts_exact_parents_and_excludes_library_only_metric_rows(self) -> None:
        result = self._build()
        self.assertEqual(result.document["condition_bridge"]["mismatch_count"], 0)
        self.assertEqual(result.document["operator_bridge"]["mismatch_count"], 0)
        self.assertEqual(len(result.metric_values), 312)
        self.assertNotIn("r2", result.query_record_ids)
        self.assertNotIn("r5", result.query_record_ids)
        self.assertEqual(result.document["condition_bridge"]["sha256"], self.config.condition_bridge_sha256)
        self.assertEqual(result.document["operator_bridge"]["sha256"], self.config.operator_bridge_sha256)

    def test_bridge_rejects_one_bit_condition_mismatch_before_metric_use(self) -> None:
        rows = [json.loads(line) for line in self.step9.joinpath("record_conditions.jsonl").read_text().splitlines()]
        rows[0]["support_projection_sha256"] = "f" * 64
        self.step9.joinpath("record_conditions.jsonl").write_bytes(_jsonl(rows))
        with self.assertRaisesRegex(Phase4D5ProtocolBError, "condition bridge"):
            self._build()

    def test_bridge_rejects_missing_or_duplicate_condition_key(self) -> None:
        original = self.step9.joinpath("record_conditions.jsonl").read_bytes()
        rows = [json.loads(line) for line in original.decode().splitlines()]
        for changed in (rows[:-1], rows + [dict(rows[0])]):
            self.step9.joinpath("record_conditions.jsonl").write_bytes(_jsonl(changed))
            with self.subTest(row_count=len(changed)):
                with self.assertRaisesRegex(Phase4D5ProtocolBError, "condition bridge"):
                    self._build()
        self.step9.joinpath("record_conditions.jsonl").write_bytes(original)

    def test_bridge_rejects_operator_output_drift(self) -> None:
        rows = [json.loads(line) for line in self.step9.joinpath("operator_cells.jsonl").read_text().splitlines()]
        rows[0]["outputs"][1]["diagnostics"]["alpha"] = 0.2
        self.step9.joinpath("operator_cells.jsonl").write_bytes(_jsonl(rows))
        with self.assertRaisesRegex(Phase4D5ProtocolBError, "operator bridge"):
            self._build()

    def test_bridge_uses_integer_class_vectors_not_parent_digest_encoding(self) -> None:
        result = self._build()
        self.assertEqual(result.document["cohort_bridge"]["class_labels"], [0, 0, 0, 1, 1, 1])

    def test_bridge_rejects_frozen_query_digest_drift(self) -> None:
        document = dict(self.config.document)
        frozen = dict(document["frozen_identities"])
        frozen["query_record_ids_sha256"] = "f" * 64
        document["frozen_identities"] = frozen
        changed = parse_phase4_d5_protocol_b_config(
            self.root / "drift.json", _canonical(document), require_frozen_identity=False
        )
        with self.assertRaisesRegex(Phase4D5ProtocolBError, "cohort bridge"):
            build_protocol_b_authority_bridge(
                cohort=self.cohort,
                protocol_a_path=self.step7,
                eligibility_path=self.step9,
                config=changed,
            )

    def test_bridge_does_not_open_forbidden_protocol_a_outcome_files(self) -> None:
        original_open = Path.open
        forbidden = {
            "matcher_predictions.jsonl",
            "class_observations.jsonl",
            "alignment_results.jsonl",
        }

        def guarded_open(path: Path, *args, **kwargs):
            if path.parent == self.step7 and path.name in forbidden:
                raise AssertionError(f"forbidden Protocol-A outcome opened: {path.name}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            result = self._build()
        self.assertEqual(len(result.metric_values), 312)

    def test_canonical_receipts_are_stable_literals_for_fixture(self) -> None:
        result = self._build()
        self.assertEqual(
            result.document["condition_bridge"]["sha256"],
            _condition_bridge_digest(_condition_rows(self.cohort, query_only=True)),
        )
        self.assertEqual(
            result.document["operator_bridge"]["sha256"],
            _operator_bridge_digest(_operator_rows(self.cohort, query_only=True)),
        )
        self.assertNotEqual(CONDITION_BRIDGE_SHA256, OPERATOR_BRIDGE_SHA256)


class ProtocolBOutcomePureFunctionsTest(unittest.TestCase):
    def test_matcher_requires_same_condition_and_preserves_class_tie_order(self) -> None:
        cohort = replace(
            _cohort(),
            group_ids=("g0", "g1", "g2", "h0", "h1", "h2"),
        )
        split = cohort.splits[0]
        query_ids = tuple(cohort.record_ids[int(index)] for index in split.query_indices)
        library_ids = tuple(cohort.record_ids[int(index)] for index in split.library_indices)
        query = np.zeros((3, 799), dtype="<f4")
        library = np.zeros((3, 799), dtype="<f4")
        query[:, :2] = 1.0
        library[0, 0] = 1.0
        library[1:, 1] = 1.0
        condition = POSITIVE_CONDITIONS[0]
        result = match_d5_protocol_b_799(
            cohort,
            split,
            query_condition_id=condition,
            library_condition_id=condition,
            query_record_ids=query_ids,
            library_record_ids=library_ids,
            query_values=query,
            library_values=library,
        )
        np.testing.assert_array_equal(result.ranked_class_labels[:, :2], [[0, 1]] * 3)
        with self.assertRaisesRegex(Phase4D5ProtocolBError, "same condition"):
            match_d5_protocol_b_799(
                cohort,
                split,
                query_condition_id=condition,
                library_condition_id="alpha0",
                query_record_ids=query_ids,
                library_record_ids=library_ids,
                query_values=query,
                library_values=library,
            )

    def test_class_aggregation_uses_query_occurrences_only(self) -> None:
        condition = POSITIVE_CONDITIONS[0]
        predictions = [
            {"class_label": 0, "record_id": "r0", "condition_id": "alpha0", "top1_correct": True},
            {"class_label": 0, "record_id": "r0", "condition_id": condition, "top1_correct": False},
            {"class_label": 0, "record_id": "r0", "condition_id": "alpha0", "top1_correct": True},
            {"class_label": 0, "record_id": "r0", "condition_id": condition, "top1_correct": True},
            {"class_label": 0, "record_id": "r1", "condition_id": "alpha0", "top1_correct": False},
            {"class_label": 0, "record_id": "r1", "condition_id": condition, "top1_correct": False},
        ]
        metric_values = {
            ("r0", "alpha0", "mse"): 0.0,
            ("r0", condition, "mse"): 2.0,
            ("r1", "alpha0", "mse"): 0.0,
            ("r1", condition, "mse"): 8.0,
            # Library-only r2 must never contribute, even when present in authority lookup.
            ("r2", "alpha0", "mse"): 0.0,
            ("r2", condition, "mse"): 1000.0,
        }
        rows = aggregate_protocol_b_class_observations(
            predictions,
            metric_values,
            metric_output_id="mse",
            preferred_direction="lower_is_better",
            positive_conditions=(condition,),
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["downstream_harm"], 1.0 / 3.0)
        self.assertAlmostEqual(rows[0]["metric_harm"], 4.0)
        self.assertEqual(rows[0]["occurrence_count"], 3)

    def test_holm_family_is_separate_24_slots_and_direction_gated(self) -> None:
        raw = {
            f"{metric}:{statistic}": 0.0
            for metric in METRIC_OUTPUT_IDS[1:]
            for statistic in ("d_ag", "d_acc")
        }
        contrasts = dict.fromkeys(raw, 0.1)
        contrasts["rmse:d_ag"] = -0.1
        family = fixed_protocol_b_holm_family(raw, contrasts)
        self.assertEqual(len(family), 24)
        self.assertTrue(all(row["family_id"] == "d5_protocol_b_full_domain_secondary_24" for row in family))
        adverse = next(row for row in family if row["hypothesis_id"] == "rmse:d_ag")
        self.assertEqual(adverse["adjusted_p_value"], 0.0)
        self.assertFalse(adverse["favorable"])
        self.assertFalse(adverse["rejected"])

    def test_protocol_b_figure_payloads_are_deterministic_and_labelled(self) -> None:
        alphas = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
        figure1 = [
            {
                "metric_output_id": metric,
                "perturbation_id": perturbation,
                "alpha": alpha,
                "mean_metric_harm": float(metric_index + alpha),
                "mean_downstream_harm": float(alpha / 2 + metric_index / 100),
                "metric_state": "complete",
            }
            for metric_index, metric in enumerate(METRIC_OUTPUT_IDS)
            for perturbation in PERTURBATIONS
            for alpha in alphas
        ]
        figure2 = [
            {
                "metric_output_id": metric, "metric_state": "complete",
                "ag": 0.1, "ag_lower": 0.05, "ag_upper": 0.15,
                "acc": 0.5, "acc_lower": 0.4, "acc_upper": 0.6,
                "d_ag": None if metric == "mse" else 0.01,
                "d_ag_lower": None if metric == "mse" else 0.0,
                "d_ag_upper": None if metric == "mse" else 0.02,
                "d_ag_favorable": None if metric == "mse" else True,
                "d_ag_raw_p": None if metric == "mse" else 0.01,
                "d_ag_adjusted_p": None if metric == "mse" else 0.24,
                "d_acc": None if metric == "mse" else 0.02,
                "d_acc_lower": None if metric == "mse" else 0.01,
                "d_acc_upper": None if metric == "mse" else 0.03,
                "d_acc_favorable": None if metric == "mse" else True,
                "d_acc_raw_p": None if metric == "mse" else 0.02,
                "d_acc_adjusted_p": None if metric == "mse" else 0.24,
            }
            for metric in METRIC_OUTPUT_IDS
        ]
        first = render_protocol_b_figure_payloads(figure1, figure2)
        second = render_protocol_b_figure_payloads(figure1, figure2)
        self.assertEqual(first, second)
        self.assertEqual(len(figure1), 520)
        self.assertIn("figure1_d5_protocol_b_full_domain.png", first)
        self.assertIn(b"matched-reference Protocol B", first["figure1_d5_protocol_b_full_domain.svg"])
        self.assertNotIn(b"Protocol A", first["figure2_d5_protocol_b_full_domain.svg"])


def _outcome_config_for_parents(
    root: Path,
    cohort: D5RawCohort,
    step7: Path,
    step9: Path,
    alpha_grid: tuple[float, ...],
) -> object:
    _, _, document = _parents(root / "schema-template", cohort)
    condition_ids = ("alpha0",) + tuple(
        f"{perturbation}:{np.float64(alpha).tobytes().hex()}"
        for perturbation in PERTURBATIONS
        for alpha in alpha_grid[1:]
    )
    query_indices = _query_indices(cohort)
    query_ids = [cohort.record_ids[index] for index in query_indices]
    query_occurrences = sum(len(split.query_indices) for split in cohort.splits)
    library_occurrences = sum(len(split.library_indices) for split in cohort.splits)

    def hashes(path: Path, names: set[str]) -> dict[str, str]:
        return {name: _sha(path.joinpath(name).read_bytes()) for name in sorted(names)}

    step7_names = {
        "config.json", "manifest.json", "complete.json", "operator_cells.jsonl",
        "record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl", "SHA256SUMS",
    }
    step9_names = {
        "config.json", "manifest.json", "complete.json", "unique_records.jsonl",
        "role_occurrences.jsonl", "operator_cells.jsonl", "record_conditions.jsonl", "SHA256SUMS",
    }
    step7_conditions = [
        json.loads(line)
        for line in step7.joinpath("record_conditions.jsonl").read_text().splitlines()
    ]
    step7_operators = [
        json.loads(line)
        for line in step7.joinpath("operator_cells.jsonl").read_text().splitlines()
    ]
    document["alpha_grid"] = list(alpha_grid)
    document["authorities"]["sweep_sha256"] = _sha(
        ROOT.joinpath("experiments/shared/raman_perturbation_sweep_v1.json").read_bytes()
    )
    document["authorities"]["phase1_core_config_sha256"] = _sha(
        ROOT.joinpath("experiments/phase1/configs/rruff_raw_core10k_v1.json").read_bytes()
    )
    document["parent_artifacts"] = {
        "protocol_a": {"relative_path": step7.name, "payload_sha256": hashes(step7, step7_names)},
        "eligibility": {"relative_path": step9.name, "payload_sha256": hashes(step9, step9_names)},
    }
    document["authority_bridge"] = {
        "condition_bridge_sha256": _condition_bridge_digest(step7_conditions),
        "operator_bridge_sha256": _operator_bridge_digest(step7_operators),
    }
    class_count = len(set(int(value) for value in cohort.class_labels))
    document["denominators"] = {
        "full_record_count": len(cohort.record_ids),
        "group_count": len(set(cohort.group_ids)),
        "class_count": class_count,
        "query_record_count": len(query_ids),
        "query_occurrence_count": query_occurrences,
        "library_occurrence_count": library_occurrences,
        "split_count": len(cohort.splits),
    }
    positive_count = len(condition_ids) - 1
    document["expected"] = {
        "operator_cell_count": len(cohort.record_ids) * len(PERTURBATIONS),
        "apply_check_count": len(cohort.record_ids) * len(PERTURBATIONS) * len(alpha_grid),
        "full_condition_count": len(cohort.record_ids) * len(condition_ids),
        "query_condition_count": len(query_ids) * len(condition_ids),
        "metric_row_count": len(query_ids) * len(condition_ids) * len(METRIC_OUTPUT_IDS),
        "peak_receipt_count": len(query_ids) * len(condition_ids),
        "matcher_call_count": len(cohort.splits) * len(condition_ids),
        "prediction_row_count": query_occurrences * len(condition_ids),
        "class_observation_count_per_metric": class_count * positive_count,
        "class_observation_count": class_count * positive_count * len(METRIC_OUTPUT_IDS),
        "holm_slot_count": 24,
        "figure1_row_count": len(METRIC_OUTPUT_IDS) * positive_count,
        "figure2_row_count": len(METRIC_OUTPUT_IDS),
        "secondary_table_row_count": len(METRIC_OUTPUT_IDS),
    }
    document["frozen_identities"] = {
        "record_ids": list(cohort.record_ids),
        "group_ids": list(cohort.group_ids),
        "class_labels": [int(value) for value in cohort.class_labels],
        "query_record_ids": query_ids,
        "split_sha256": [split.split_sha256 for split in cohort.splits],
    }
    document["inference"] = {
        "bootstrap_resamples": 8,
        "sign_flip_resamples": 64,
        "random_seed": 20260817,
        "confidence_level": 0.95,
        "holm_alpha": 0.05,
    }
    return parse_phase4_d5_protocol_b_config(
        root / "synthetic-outcome.json",
        _canonical(document),
        require_frozen_identity=False,
    )


class ProtocolBOutcomeArtifactTest(unittest.TestCase):
    def test_independent_matcher_uses_single_blas_thread_for_byte_stability(self) -> None:
        import rpe.runner.phase4_d5_protocol_b_verifier as verifier

        cohort = replace(
            _cohort(),
            group_ids=("g0", "g1", "g2", "h0", "h1", "h2"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, document = _parents(root, cohort)
            config = parse_phase4_d5_protocol_b_config(
                root / "config.json", _canonical(document), require_frozen_identity=False
            )
            projected = {}
            for record_index, record_id in enumerate(cohort.record_ids):
                for condition_index, condition_id in enumerate(config.condition_ids):
                    values = np.zeros(799, dtype="<f4")
                    values[(record_index + condition_index) % 799] = 1.0
                    values[(record_index * 7 + condition_index + 1) % 799] = 0.25
                    projected[(record_id, condition_id)] = values
            observed_threads: list[int] = []
            real_match = verifier.match_d5_protocol_a_values

            def checking_match(*args, **kwargs):
                observed_threads.extend(
                    int(item["num_threads"])
                    for item in threadpool_info()
                    if item.get("user_api") == "blas"
                )
                return real_match(*args, **kwargs)

            with patch.object(verifier, "match_d5_protocol_a_values", side_effect=checking_match):
                rows = verifier._predict(cohort, projected, config)
            self.assertEqual(len(rows), config.expected_prediction_row_count)
            self.assertTrue(observed_threads)
            self.assertEqual(max(observed_threads), 1)

    def test_synthetic_builder_rematerializes_step9_and_emits_complete_b_artifact(self) -> None:
        from tests.test_phase4_d5_protocol_a import (
            _synthetic_cohort as parent_cohort,
            _synthetic_config as parent_a_config,
            _synthetic_native as parent_native,
        )
        from tests.test_phase4_d5_protocol_b_eligibility import (
            _config_document as eligibility_document,
        )
        from rpe.methods.catalog import load_classical_catalog
        from rpe.perturb import load_perturbation_sweep_config
        from rpe.runner.phase1_config import load_phase1_core_config
        from rpe.runner.phase4_d5_protocol_a import build_phase4_d5_protocol_a_from_inputs
        from rpe.runner.phase4_d5_protocol_b_eligibility import (
            build_phase4_d5_protocol_b_eligibility_from_inputs,
            parse_phase4_d5_protocol_b_eligibility_config,
        )

        cohort = parent_cohort()
        native = parent_native(cohort)
        sweep = load_perturbation_sweep_config(
            ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
        )
        phase1 = load_phase1_core_config(
            ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
        )
        catalog = load_classical_catalog(
            ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json",
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step7 = root / "step7"
            step9 = root / "step9"
            build_phase4_d5_protocol_a_from_inputs(
                step7,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                classical_catalog=catalog,
                config=parent_a_config(cohort),
                worker_count=2,
                inference_resamples=8,
            )
            eligibility_config = parse_phase4_d5_protocol_b_eligibility_config(
                root / "eligibility.json",
                _canonical(eligibility_document(cohort)),
                require_frozen_identity=False,
            )
            build_phase4_d5_protocol_b_eligibility_from_inputs(
                step9,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=eligibility_config,
                worker_count=2,
            )
            config = _outcome_config_for_parents(
                root, cohort, step7, step9, tuple(sweep.alpha_grid)
            )
            output = root / "protocol-b"
            summary = build_phase4_d5_protocol_b_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=config,
                protocol_a_path=step7,
                eligibility_path=step9,
                worker_count=2,
                inference_resamples=8,
            )
            self.assertEqual(summary.status, "complete")
            self.assertEqual(summary.prediction_row_count, config.expected_prediction_row_count)
            self.assertEqual(
                summary.class_observation_count,
                config.expected_class_observation_count,
            )
            observed = {path.name for path in output.iterdir() if path.is_file()}
            self.assertEqual(
                observed,
                set(ARTIFACT_PAYLOAD_FILES) | {"complete.json", "SHA256SUMS"},
            )
            self.assertEqual(
                output.joinpath("operator_cells.jsonl").read_bytes(),
                step9.joinpath("operator_cells.jsonl").read_bytes(),
            )
            self.assertEqual(
                output.joinpath("record_conditions.jsonl").read_bytes(),
                step9.joinpath("record_conditions.jsonl").read_bytes(),
            )
            self.assertFalse(output.joinpath("metric_values.jsonl").exists())
            predictions = [
                json.loads(line)
                for line in output.joinpath("matcher_predictions.jsonl").read_text().splitlines()
            ]
            self.assertTrue(
                all(
                    row["query_condition_id"] == row["library_condition_id"]
                    for row in predictions
                )
            )
            manifest = json.loads(output.joinpath("manifest.json").read_bytes())
            self.assertEqual(manifest["protocol"], "B")
            self.assertEqual(
                manifest["counts"]["matcher_predictions"],
                config.expected_prediction_row_count,
            )

    def test_independent_verifier_rebuilds_bytes_without_production_builder(self) -> None:
        from tests.test_phase4_d5_protocol_a import (
            _synthetic_cohort as parent_cohort,
            _synthetic_config as parent_a_config,
            _synthetic_native as parent_native,
        )
        from tests.test_phase4_d5_protocol_b_eligibility import (
            _config_document as eligibility_document,
        )
        from rpe.methods.catalog import load_classical_catalog
        from rpe.perturb import load_perturbation_sweep_config
        from rpe.runner.phase1_config import load_phase1_core_config
        from rpe.runner.phase4_d5_protocol_a import build_phase4_d5_protocol_a_from_inputs
        from rpe.runner.phase4_d5_protocol_b_eligibility import (
            build_phase4_d5_protocol_b_eligibility_from_inputs,
            parse_phase4_d5_protocol_b_eligibility_config,
        )
        import rpe.runner.phase4_d5_protocol_b_verifier as verifier_module

        cohort = parent_cohort()
        native = parent_native(cohort)
        sweep = load_perturbation_sweep_config(
            ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
        )
        phase1 = load_phase1_core_config(
            ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
        )
        catalog = load_classical_catalog(
            ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json",
            project_root=ROOT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step7 = root / "step7"
            step9 = root / "step9"
            output = root / "protocol-b"
            build_phase4_d5_protocol_a_from_inputs(
                step7,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                classical_catalog=catalog,
                config=parent_a_config(cohort),
                worker_count=2,
                inference_resamples=8,
            )
            eligibility_config = parse_phase4_d5_protocol_b_eligibility_config(
                root / "eligibility.json",
                _canonical(eligibility_document(cohort)),
                require_frozen_identity=False,
            )
            build_phase4_d5_protocol_b_eligibility_from_inputs(
                step9,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=eligibility_config,
                worker_count=2,
            )
            config = _outcome_config_for_parents(
                root, cohort, step7, step9, tuple(sweep.alpha_grid)
            )
            built = build_phase4_d5_protocol_b_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=config,
                protocol_a_path=step7,
                eligibility_path=step9,
                worker_count=2,
                inference_resamples=8,
            )
            with patch(
                "rpe.runner.phase4_d5_protocol_b.build_phase4_d5_protocol_b_from_inputs",
                side_effect=AssertionError("production builder must not be called"),
            ), patch(
                "rpe.runner.phase4_d5_protocol_b.build_protocol_b_authority_bridge",
                side_effect=AssertionError("production bridge must not be called"),
            ), patch(
                "rpe.runner.phase4_d5_protocol_b.aggregate_protocol_b_class_observations",
                side_effect=AssertionError("production aggregation must not be called"),
            ), patch(
                "rpe.runner.phase4_d5_protocol_b.parse_phase4_d5_protocol_b_config",
                side_effect=AssertionError("production config parser must not be imported or called"),
            ):
                try:
                    reloaded = importlib.reload(verifier_module)
                    verified = reloaded.verify_phase4_d5_protocol_b_from_inputs(
                        output,
                        cohort=cohort,
                        native_spectra=native,
                        sweep=sweep,
                        phase1_config=phase1,
                        protocol_a_path=step7,
                        eligibility_path=step9,
                        worker_count=1,
                        inference_resamples=8,
                    )
                finally:
                    importlib.reload(verifier_module)
            self.assertEqual(verified.run_id, built.run_id)
            self.assertEqual(verified.status, "complete")


class ProtocolBCliTest(unittest.TestCase):
    def test_cli_exposes_only_build_and_mandatory_reexecuting_verify(self) -> None:
        from tools.run_phase4_d5_protocol_b import main

        for arguments in (
            ["verify", "--run-path", "x", "--no-reexecute"],
            ["build", "--output-root", "x", "--protocol", "A"],
            ["build", "--output-root", "x", "--compare-a-b"],
            ["build", "--output-root", "x", "--inference-resamples", "8"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    main(arguments)


if __name__ == "__main__":
    unittest.main()
