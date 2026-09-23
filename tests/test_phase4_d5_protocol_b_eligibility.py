from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.rruff import (  # noqa: E402
    CONFIG_SHA256 as D5_CONFIG_SHA256,
    D5LibraryQuerySplit,
    D5RawCohort,
)
from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_config import load_phase1_core_config  # noqa: E402
from rpe.runner.phase4_d5_protocol_b_eligibility import (  # noqa: E402
    ACTIVE_PERTURBATIONS,
    ARTIFACT_PAYLOAD_FILES,
    INACTIVE_PERTURBATIONS,
    Phase4D5ProtocolBEligibilityError,
    build_phase4_d5_protocol_b_eligibility_from_inputs,
    evaluate_d5_protocol_b_all_role_gates,
    load_phase4_d5_protocol_b_eligibility_config,
    parse_phase4_d5_protocol_b_eligibility_config,
    reconstruct_d5_protocol_b_role_ledgers,
    validate_protocol_b_outcome_blind_payload,
)
from rpe.runner.phase4_d5_protocol_b_eligibility_verifier import (  # noqa: E402
    verify_phase4_d5_protocol_b_eligibility_from_inputs,
)
from tools.run_phase4_d5_protocol_b_eligibility import (  # noqa: E402
    main as protocol_b_cli_main,
)


SWEEP_PATH = ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG = ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
CONFIG_PATH = ROOT / "experiments/phase4/configs/d5_protocol_b_all_role_eligibility_v1.json"
ALL_PERTURBATIONS = ACTIVE_PERTURBATIONS + INACTIVE_PERTURBATIONS


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


def _digest_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode("utf-8")).hexdigest()


def _numeric_class_digest(values: list[int]) -> str:
    return hashlib.sha256(
        ("\n".join(str(value) for value in sorted(set(values))) + "\n").encode("utf-8")
    ).hexdigest()


def _synthetic_cohort() -> D5RawCohort:
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


def _synthetic_native(cohort: D5RawCohort) -> tuple[Spectrum1D, ...]:
    axis = np.arange(180.0, 1822.0, 2.0, dtype="<f8")
    spectra = []
    for index, record_id in enumerate(cohort.record_ids):
        centers = (340.0 + 9.0 * index, 720.0 + 7.0 * index, 1260.0 - 6.0 * index)
        intensity = 0.2 + 0.0002 * (axis - axis[0])
        for peak_index, center in enumerate(centers):
            intensity = intensity + (1.4 - 0.15 * peak_index) * np.exp(
                -0.5 * ((axis - center) / (11.0 + 2.5 * peak_index)) ** 2
            )
        spectra.append(
            Spectrum1D(
                spectrum_id=f"rruff_raman_raw::{record_id}",
                sample_id=cohort.rruff_ids[index],
                axis_cm1=axis,
                intensity=np.asarray(intensity, dtype="<f8"),
            )
        )
    return tuple(spectra)


def _three_class_digest_cohort() -> D5RawCohort:
    labels = _read_only(np.asarray([1, 10, 1, 2, 2, 10], dtype="<i8"))
    split0 = D5LibraryQuerySplit(
        seed=0,
        query_indices=_read_only(np.asarray([0, 1, 3], dtype="<i8")),
        library_indices=_read_only(np.asarray([2, 4, 5], dtype="<i8")),
        split_sha256="c" * 64,
    )
    split1 = D5LibraryQuerySplit(
        seed=1,
        query_indices=_read_only(np.asarray([1, 2, 4], dtype="<i8")),
        library_indices=_read_only(np.asarray([0, 3, 5], dtype="<i8")),
        split_sha256="d" * 64,
    )
    return D5RawCohort(
        protocol_config_sha256=D5_CONFIG_SHA256,
        dataset_id="rruff_raman_raw",
        intensity=_read_only(np.ones((6, 801), dtype="<f4")),
        wavenumber=_read_only(np.arange(200.0, 1802.0, 2.0, dtype="<f4")),
        class_labels=labels,
        record_ids=("r0", "r1", "r2", "r3", "r4", "r5"),
        mineral_names=("m0", "m10", "m1", "m2", "m2", "m10"),
        rruff_ids=("s0", "s1", "s2", "s3", "s4", "s5"),
        pin_ids=(None,) * 6,
        group_ids=("g0", "g1", "g0", "h0", "h1", "h1"),
        splits=(split0, split1),
    )


def _config_document(cohort: D5RawCohort) -> dict[str, object]:
    sweep = load_perturbation_sweep_config(SWEEP_PATH)
    record_ids = list(cohort.record_ids)
    group_ids = sorted(set(cohort.group_ids))
    class_label_values = [int(value) for value in cohort.class_labels.tolist()]
    class_labels = [str(value) for value in sorted(set(class_label_values))]
    query_occurrences = sum(len(split.query_indices) for split in cohort.splits)
    library_occurrences = sum(len(split.library_indices) for split in cohort.splits)
    positive_conditions = len(record_ids) * len(ACTIVE_PERTURBATIONS) * (len(sweep.alpha_grid) - 1)
    canonical_conditions = len(record_ids) * (
        1 + len(ACTIVE_PERTURBATIONS) * (len(sweep.alpha_grid) - 1)
    )
    return {
        "active_perturbation_ids": list(ACTIVE_PERTURBATIONS),
        "alpha_grid": list(sweep.alpha_grid),
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "authorities": {
            "d5_config_sha256": D5_CONFIG_SHA256,
            "dataset_sha256sums_sha256": "0" * 64,
            "parent_plan_sha256": "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5",
            "phase1_core_config_sha256": hashlib.sha256(PHASE1_CONFIG.read_bytes()).hexdigest(),
            "phase4_step1_sha256": "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f",
            "phase4_step4_sha256": "2ae40b991d79935f173b16cf59ad8d7f531ad6a0f5717373712da96a802279e5",
            "phase4_step5_sha256": "2f1d30966e10dcce33caca86651b80747ddcc485109205ad7c1c7785ad8f4341",
            "phase4_step6_sha256": "571e22d1104c37536d8ce033a22756c2e2868ae82d021139ae7e75756f8a0100",
            "phase4_step7_sha256": "93ae3bb15c3f4b3d5e27f6a2c55e087e9e5541a0352783db98e11c01ca368375",
            "protocol_sha256": "1" * 64,
            "sweep_sha256": sweep.sha256,
        },
        "claim_boundary": "outcome_blind_protocol_b_all_role_eligibility_only",
        "code_authority": {},
        "denominators": {
            "class_count": len(class_labels),
            "group_count": len(group_ids),
            "library_role_occurrence_count": library_occurrences,
            "query_role_occurrence_count": query_occurrences,
            "role_occurrence_count": query_occurrences + library_occurrences,
            "unique_record_count": len(record_ids),
        },
        "derived_counts": {
            "apply_check_count": len(record_ids) * len(ACTIVE_PERTURBATIONS) * len(sweep.alpha_grid),
            "canonical_record_condition_count": canonical_conditions,
            "class_summary_count": len(class_labels) * len(ACTIVE_PERTURBATIONS),
            "group_summary_count": len(group_ids),
            "operator_cell_count": len(record_ids) * len(ACTIVE_PERTURBATIONS),
            "positive_record_condition_count": positive_conditions,
            "role_condition_reference_count": (query_occurrences + library_occurrences) * (1 + len(ACTIVE_PERTURBATIONS) * (len(sweep.alpha_grid) - 1)),
        },
        "environment_authority": {},
        "experiment_id": "phase4-d5-protocol-b-all-role-eligibility-v1",
        "frozen_identities": {
            "class_labels_sha256": _numeric_class_digest(class_label_values),
            "group_ids_sha256": _digest_lines(group_ids),
            "record_ids_sha256": _digest_lines(record_ids),
            "split_sha256": [split.split_sha256 for split in cohort.splits],
        },
        "inherited_rulings": {
            "p01_p04_full_domain_core": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 0,
                "state": "not_evaluable_coverage",
            },
            "p05_full_domain_core": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 0,
                "state": "not_evaluable_coverage",
            },
            "p06": {"reason": "structurally_ineligible_missing_explicit_baseline", "state": "structurally_ineligible_missing_explicit_baseline"},
            "p07": {"reason": "structurally_ineligible_missing_explicit_baseline", "state": "structurally_ineligible_missing_explicit_baseline"},
            "peak_common_support": {"state": "not_evaluable_coverage"},
        },
        "inactive_perturbation_ids": list(INACTIVE_PERTURBATIONS),
        "p10": {
            "correlation_length_cm1": 20.0,
            "memory_budget_bytes": 64 * 2**30,
            "peak_estimate_formula": "32*N^2+64*N+2^30",
        },
        "phase1_native_gate_relative_tolerance": 1e-12,
        "protocol": "B",
        "role_condition_policy": "roles_are_references_not_independent_units",
        "schema_version": "phase4-d5-protocol-b-all-role-eligibility-config-v1",
        "support_grid": {
            "max_in_range_native_gap_cm1": 3.0,
            "point_count": 799,
            "start_cm1": 204.0,
            "step_cm1": 2.0,
            "stop_cm1": 1800.0,
        },
        "synthetic_fixture": True,
        "trust_anchor": {"config_authority_relative_path": "rpe/runner/phase4_d5_protocol_b_eligibility_authority.py"},
    }


def _synthetic_config(cohort: D5RawCohort):
    raw = _canonical(_config_document(cohort))
    return parse_phase4_d5_protocol_b_eligibility_config(
        Path("synthetic.json"), raw, require_frozen_identity=False
    )


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }


class FrozenProtocolBConfigTest(unittest.TestCase):
    def test_loads_exact_real_config_and_rejects_drift(self) -> None:
        config = load_phase4_d5_protocol_b_eligibility_config(CONFIG_PATH)
        self.assertEqual(config.protocol, "B")
        self.assertEqual(config.unique_record_count, 3770)
        self.assertEqual(config.group_count, 1934)
        self.assertEqual(config.class_count, 681)
        self.assertEqual(config.query_role_occurrence_count, 6621)
        self.assertEqual(config.library_role_occurrence_count, 12229)
        self.assertEqual(config.role_occurrence_count, 18850)
        self.assertEqual(config.expected_operator_cell_count, 18850)
        self.assertEqual(config.expected_apply_check_count, 169650)
        self.assertEqual(config.expected_positive_record_condition_count, 150800)
        self.assertEqual(config.expected_canonical_record_condition_count, 154570)
        self.assertEqual(config.expected_class_summary_count, 3405)
        self.assertEqual(config.support_point_count, 799)
        self.assertEqual(config.p10_memory_budget_bytes, 64 * 2**30)
        self.assertEqual(config.artifact_payload_files, ARTIFACT_PAYLOAD_FILES)
        self.assertEqual(
            config.inherited_rulings["peak_common_support"]["state"],
            "not_evaluable_coverage",
        )
        self.assertEqual(
            config.inherited_rulings["p06"]["state"],
            "structurally_ineligible_missing_explicit_baseline",
        )
        self.assertEqual(
            config.trust_anchor["config_authority_relative_path"],
            "rpe/runner/phase4_d5_protocol_b_eligibility_authority.py",
        )

        changed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        changed["protocol"] = "A"
        with self.assertRaisesRegex(
            Phase4D5ProtocolBEligibilityError, "frozen config identity"
        ):
            parse_phase4_d5_protocol_b_eligibility_config(
                Path("drift.json"),
                _canonical(changed),
                require_frozen_identity=True,
            )


class ProtocolBRoleLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = _synthetic_cohort()
        self.native = _synthetic_native(self.cohort)
        self.config = _synthetic_config(self.cohort)

    def test_reconstructs_full_cohort_roles_and_keeps_library_only_record(self) -> None:
        ledgers = reconstruct_d5_protocol_b_role_ledgers(
            self.cohort, self.native, self.config
        )
        self.assertEqual(len(ledgers.unique_records), 6)
        self.assertEqual(len(ledgers.role_occurrences), 12)
        self.assertEqual(len(ledgers.groups), 4)
        self.assertEqual(ledgers.unique_group_count, 4)
        self.assertEqual(ledgers.unique_class_count, 2)
        self.assertEqual(
            [row["record_id"] for row in ledgers.unique_records],
            ["r0", "r1", "r2", "r3", "r4", "r5"],
        )
        never_query = next(
            row for row in ledgers.unique_records if row["record_id"] == "r2"
        )
        self.assertEqual(never_query["query_occurrence_count"], 0)
        self.assertEqual(never_query["library_occurrence_count"], 2)
        self.assertEqual(never_query["role_occurrence_count"], 2)
        self.assertEqual(never_query["query_split_seeds"], [])
        self.assertEqual(never_query["library_split_seeds"], [0, 1])
        self.assertEqual(
            [
                (row["split_seed"], row["role"], row["role_order"], row["record_id"])
                for row in ledgers.role_occurrences
            ],
            [
                (0, "query", 0, "r0"),
                (0, "query", 1, "r1"),
                (0, "query", 2, "r3"),
                (0, "library", 0, "r2"),
                (0, "library", 1, "r4"),
                (0, "library", 2, "r5"),
                (1, "query", 0, "r0"),
                (1, "query", 1, "r3"),
                (1, "query", 2, "r4"),
                (1, "library", 0, "r1"),
                (1, "library", 1, "r2"),
                (1, "library", 2, "r5"),
            ],
        )

    def test_accepts_numeric_class_label_digest_order_for_non_lexicographic_labels(self) -> None:
        cohort = _three_class_digest_cohort()
        native = _synthetic_native(cohort)
        document = _config_document(cohort)
        document["frozen_identities"]["class_labels_sha256"] = _numeric_class_digest(
            [1, 2, 10]
        )
        config = parse_phase4_d5_protocol_b_eligibility_config(
            Path("synthetic-three-class.json"),
            _canonical(document),
            require_frozen_identity=False,
        )
        ledgers = reconstruct_d5_protocol_b_role_ledgers(cohort, native, config)
        self.assertEqual(len(ledgers.unique_records), 6)
        self.assertEqual(ledgers.unique_class_count, 3)
        self.assertEqual(
            sorted({int(row["class_label"]) for row in ledgers.unique_records}),
            [1, 2, 10],
        )

    def test_all_role_gate_is_record_and_class_fail_closed_with_group_role_audits_only(self) -> None:
        ledgers = reconstruct_d5_protocol_b_role_ledgers(
            self.cohort, self.native, self.config
        )
        operator_cells = []
        for record in ledgers.unique_records:
            record_id = record["record_id"]
            for perturbation_id in ACTIVE_PERTURBATIONS:
                operator_cells.append(
                    {
                        "perturbation_id": perturbation_id,
                        "record_id": record_id,
                        "state": "complete",
                    }
                )
        target = next(
            row
            for row in operator_cells
            if row["record_id"] == "r0" and row["perturbation_id"] == "p08"
        )
        target["state"] = "failed_runtime"
        class_rows, gate = evaluate_d5_protocol_b_all_role_gates(
            ledgers.unique_records,
            operator_cells,
            ledgers.groups,
            ledgers.role_occurrences,
            self.config,
        )
        p08_rows = [row for row in class_rows if row["perturbation_id"] == "p08"]
        self.assertEqual(len(p08_rows), 2)
        failed_class = next(row for row in p08_rows if row["class_label"] == 0)
        self.assertFalse(failed_class["complete"])
        self.assertEqual(failed_class["complete_record_count"], 2)
        self.assertEqual(failed_class["required_record_count"], 3)
        self.assertEqual(gate["operators"]["p08"]["complete_record_count"], 5)
        self.assertEqual(gate["operators"]["p08"]["required_record_count"], 6)
        self.assertEqual(gate["operators"]["p08"]["complete_class_count"], 1)
        self.assertEqual(gate["operators"]["p08"]["required_class_count"], 2)
        self.assertEqual(gate["operators"]["p08"]["state"], "not_evaluable_coverage")
        self.assertEqual(gate["group_audits"]["p08"]["failed_group_count"], 1)
        self.assertEqual(gate["role_audits"]["p08"]["query_failed_count"], 2)
        self.assertEqual(gate["role_audits"]["p08"]["library_failed_count"], 0)

    def test_rejects_any_outcome_metric_or_matcher_payload(self) -> None:
        for field in (
            "prediction",
            "top1_correct",
            "metric_value",
            "downstream_harm",
            "alignment_gap",
            "p_value",
            "table",
            "figure",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    Phase4D5ProtocolBEligibilityError, "outcome-blind"
                ):
                    validate_protocol_b_outcome_blind_payload(
                        {"nested": [{field: 1.0}]}
                    )


class ProtocolBArtifactTest(unittest.TestCase):
    def test_synthetic_build_and_independent_verifier_are_byte_identical(self) -> None:
        cohort = _synthetic_cohort()
        native = _synthetic_native(cohort)
        sweep = load_perturbation_sweep_config(SWEEP_PATH)
        phase1 = load_phase1_core_config(PHASE1_CONFIG)
        config = _synthetic_config(cohort)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            summary = build_phase4_d5_protocol_b_eligibility_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=config,
                worker_count=2,
            )
            verified = verify_phase4_d5_protocol_b_eligibility_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                worker_count=1,
            )
            self.assertEqual(summary.run_id, verified.run_id)
            self.assertEqual(summary.status, verified.status)
            observed = {item.name for item in output.iterdir() if item.is_file()}
            marker_name = next(iter(observed & {"complete.json", "failed.json"}))
            self.assertEqual(len(observed & {"complete.json", "failed.json"}), 1)
            self.assertEqual(
                observed,
                set(ARTIFACT_PAYLOAD_FILES) | {"SHA256SUMS", marker_name},
            )
            manifest = json.loads((output / "manifest.json").read_bytes())
            self.assertEqual(manifest["protocol"], "B")
            self.assertEqual(manifest["claim_boundary"], "outcome_blind_protocol_b_all_role_eligibility_only")
            self.assertEqual(
                manifest["artifact_order"],
                list(ARTIFACT_PAYLOAD_FILES),
            )
            self.assertEqual(summary.unique_record_count, 6)
            self.assertEqual(summary.role_occurrence_count, 12)
            self.assertEqual(summary.operator_cell_count, 30)
            self.assertEqual(summary.record_condition_count, 246)
            self.assertEqual(summary.class_summary_count, 10)
            self.assertEqual(_tree(output), _tree(output))
            checksums = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [line.split("  ", 1)[1] for line in checksums],
                list(ARTIFACT_PAYLOAD_FILES) + [marker_name],
            )
            condition_rows = [
                json.loads(line)
                for line in (output / "record_conditions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(condition_rows), 246)
            alpha0_rows = [row for row in condition_rows if row["condition_kind"] == "alpha0"]
            self.assertEqual(len(alpha0_rows), 6)
            self.assertTrue(all(row["perturbation_id"] is None for row in alpha0_rows))
            p10_rows = [
                row for row in condition_rows if row["perturbation_id"] == "p10" and row["alpha"] > 0.0
            ]
            self.assertEqual(len(p10_rows), 48)
            self.assertTrue(all(row["support_point_count"] == 799 for row in condition_rows))
            self.assertTrue(all(row["support_max_in_range_gap_cm1"] <= 3.0 for row in condition_rows))

    def test_p11_positive_alpha_preserves_intensity_but_shifts_axis_with_real_native_gate(self) -> None:
        cohort = _synthetic_cohort()
        native = _synthetic_native(cohort)
        sweep = load_perturbation_sweep_config(SWEEP_PATH)
        phase1 = load_phase1_core_config(PHASE1_CONFIG)
        config = _synthetic_config(cohort)
        source = native[0]
        source_axis_sha256 = hashlib.sha256(
            np.ascontiguousarray(source.axis_cm1, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        source_intensity_sha256 = hashlib.sha256(
            np.ascontiguousarray(source.intensity, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            build_phase4_d5_protocol_b_eligibility_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=config,
                worker_count=1,
            )
            operator_cells = [
                json.loads(line)
                for line in (output / "operator_cells.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            p11_cell = next(
                row
                for row in operator_cells
                if row["record_id"] == "r0" and row["perturbation_id"] == "p11"
            )
            alpha0_output = next(row for row in p11_cell["outputs"] if row["alpha"] == 0.0)
            positive_output = next(row for row in p11_cell["outputs"] if row["alpha"] == 0.05)
            self.assertEqual(alpha0_output["output_axis_sha256"], source_axis_sha256)
            self.assertEqual(alpha0_output["output_intensity_sha256"], source_intensity_sha256)
            self.assertTrue(positive_output["axis_changed"])
            self.assertFalse(positive_output["intensity_changed"])
            self.assertEqual(positive_output["output_intensity_sha256"], source_intensity_sha256)
            self.assertNotEqual(positive_output["output_axis_sha256"], source_axis_sha256)
            self.assertEqual(
                positive_output["diagnostics"].get("transform"),
                "global_additive_shift",
            )
            self.assertEqual(
                positive_output["diagnostics"].get("interpolation"),
                "none",
            )
            self.assertTrue(
                positive_output["diagnostics"].get("intensity_preserved"),
            )
            self.assertGreater(
                float(positive_output["diagnostics"]["realized_max_abs_offset_cm1"]),
                0.0,
            )
            self.assertIn("diagnostic_max_abs_offset_cm1", p11_cell["native_gate"])
            self.assertIn("observed_max_abs_offset_cm1", p11_cell["native_gate"])
            self.assertEqual(
                p11_cell["native_gate"]["diagnostic_max_abs_offset_cm1"][0],
                0.0,
            )
            self.assertGreater(
                float(p11_cell["native_gate"]["diagnostic_max_abs_offset_cm1"][1]),
                0.0,
            )
            condition_rows = [
                json.loads(line)
                for line in (output / "record_conditions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            alpha0_row = next(
                row
                for row in condition_rows
                if row["record_id"] == "r0" and row["condition_kind"] == "alpha0"
            )
            self.assertEqual(alpha0_row["axis_sha256"], source_axis_sha256)
            self.assertEqual(alpha0_row["intensity_sha256"], source_intensity_sha256)

    def test_verifier_remains_independent_when_production_builder_is_patched_to_raise(self) -> None:
        cohort = _synthetic_cohort()
        native = _synthetic_native(cohort)
        sweep = load_perturbation_sweep_config(SWEEP_PATH)
        phase1 = load_phase1_core_config(PHASE1_CONFIG)
        config = _synthetic_config(cohort)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            summary = build_phase4_d5_protocol_b_eligibility_from_inputs(
                output,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=config,
                worker_count=1,
            )
            verifier_module_name = (
                "rpe.runner.phase4_d5_protocol_b_eligibility_verifier"
            )
            verifier_module = sys.modules[verifier_module_name]
            with patch(
                "rpe.runner.phase4_d5_protocol_b_eligibility.build_phase4_d5_protocol_b_eligibility_from_inputs",
                side_effect=AssertionError("production builder must not be called by verifier"),
            ):
                try:
                    reloaded = importlib.reload(verifier_module)
                    verified = reloaded.verify_phase4_d5_protocol_b_eligibility_from_inputs(
                        output,
                        cohort=cohort,
                        native_spectra=native,
                        sweep=sweep,
                        phase1_config=phase1,
                        worker_count=1,
                    )
                finally:
                    importlib.reload(verifier_module)
            self.assertEqual(verified.run_id, summary.run_id)
            self.assertEqual(verified.status, summary.status)

    def test_p10_admission_rejects_before_output_creation(self) -> None:
        cohort = _synthetic_cohort()
        native = list(_synthetic_native(cohort))
        huge_axis = np.arange(0.0, 250000.0, 2.0, dtype="<f8")
        native[0] = Spectrum1D(
            spectrum_id=native[0].spectrum_id,
            sample_id=native[0].sample_id,
            axis_cm1=huge_axis,
            intensity=np.ones_like(huge_axis, dtype="<f8"),
        )
        sweep = load_perturbation_sweep_config(SWEEP_PATH)
        phase1 = load_phase1_core_config(PHASE1_CONFIG)
        config = _synthetic_config(cohort)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            with self.assertRaisesRegex(
                Phase4D5ProtocolBEligibilityError, "64 GiB"
            ):
                build_phase4_d5_protocol_b_eligibility_from_inputs(
                    output,
                    cohort=cohort,
                    native_spectra=tuple(native),
                    sweep=sweep,
                    phase1_config=phase1,
                    config=config,
                    worker_count=1,
                )
            self.assertFalse(output.exists())

    def test_cli_does_not_expose_skip_reexecution_outcome_or_protocol_flags(self) -> None:
        with self.assertRaises(SystemExit):
            protocol_b_cli_main(["verify", "--run-path", "x", "--no-reexecute"])
        with self.assertRaises(SystemExit):
            protocol_b_cli_main(["build", "--output-root", "x", "--protocol", "A"])
        with self.assertRaises(SystemExit):
            protocol_b_cli_main(["build", "--output-root", "x", "--outcome"])


if __name__ == "__main__":
    unittest.main()
