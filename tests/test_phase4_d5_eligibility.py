from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

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
from rpe.runner.phase4_d5_eligibility import (  # noqa: E402
    ALL_PERTURBATIONS,
    ARTIFACT_STATIC_FILES,
    Phase4D5EligibilityError,
    build_phase4_d5_eligibility_from_inputs,
    evaluate_d5_eligibility_gates,
    load_phase4_d5_eligibility_config,
    parse_phase4_d5_eligibility_config,
    reconstruct_d5_eligibility_ledgers,
    validate_outcome_blind_payload,
)
from rpe.runner.phase4_d5_eligibility_verifier import (  # noqa: E402
    verify_phase4_d5_eligibility_from_inputs,
)
from tools.run_phase4_d5_eligibility import main as eligibility_cli_main  # noqa: E402


SWEEP_PATH = ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG = ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
ELIGIBILITY_CONFIG = ROOT / "experiments/phase4/configs/d5_eligibility_v1.json"


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
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _synthetic_cohort() -> D5RawCohort:
    record_ids = ("r0", "r1", "r2", "r3", "r4", "r5")
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
        intensity=_read_only(np.ones((6, 801), dtype="<f4")),
        wavenumber=_read_only(np.arange(200.0, 1802.0, 2.0, dtype="<f4")),
        class_labels=labels,
        record_ids=record_ids,
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
        centers = (360.0 + 7.0 * index, 720.0 + 11.0 * index, 1240.0 - 5.0 * index)
        intensity = 0.1 + 0.0001 * (axis - axis[0])
        for peak_index, center in enumerate(centers):
            intensity = intensity + (1.0 - 0.12 * peak_index) * np.exp(
                -0.5 * ((axis - center) / (10.0 + 2.0 * peak_index)) ** 2
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


def _config_document(
    cohort: D5RawCohort,
    native: tuple[Spectrum1D, ...],
) -> dict[str, object]:
    sweep = load_perturbation_sweep_config(SWEEP_PATH)
    query_indices = [int(index) for split in cohort.splits for index in split.query_indices]
    unique_indices = sorted(set(query_indices), key=lambda index: cohort.record_ids[index])
    record_ids = [cohort.record_ids[index] for index in unique_indices]
    group_ids = sorted({cohort.group_ids[index] for index in unique_indices})
    class_labels = sorted({int(cohort.class_labels[index]) for index in unique_indices})
    return {
        "active_perturbation_ids": ["p01", "p02", "p03", "p04", "p05", "p08", "p09", "p10", "p11", "p12"],
        "alpha_grid": list(sweep.alpha_grid),
        "authorities": {
            "d5_config_sha256": D5_CONFIG_SHA256,
            "phase1_core_config_sha256": hashlib.sha256(PHASE1_CONFIG.read_bytes()).hexdigest(),
            "protocol_sha256": "d" * 64,
            "sweep_sha256": sweep.sha256,
        },
        "claim_boundary": "outcome_blind_eligibility_only",
        "code_authority": {},
        "denominators": {
            "class_count": len(class_labels),
            "group_count": len(group_ids),
            "split_occurrence_count": len(query_indices),
            "unique_record_count": len(record_ids),
        },
        "expected": {
            "active_cell_count": len(record_ids) * 10,
            "apply_call_count": len(record_ids) * 10 * len(sweep.alpha_grid),
            "cell_count": len(record_ids) * 12,
            "class_summary_count": len(class_labels) * 12,
            "inactive_cell_count": len(record_ids) * 2,
        },
        "experiment_id": "phase4-d5-eligibility-v1",
        "frozen_identities": {
            "query_class_labels_sha256": _digest_lines([str(value) for value in class_labels]),
            "query_group_ids_sha256": _digest_lines(group_ids),
            "query_record_ids_sha256": _digest_lines(record_ids),
            "split_sha256": [split.split_sha256 for split in cohort.splits],
        },
        "gates": {
            "full_domain_class_fraction": 1.0,
            "full_domain_record_fraction": 1.0,
            "p01_p04_class_fraction": 0.95,
            "p01_p04_record_fraction": 0.95,
            "p05_class_fraction": 0.9,
            "p05_record_fraction": 0.9,
            "peak_common_class_fraction": 0.9,
            "peak_common_record_fraction": 0.9,
        },
        "inactive_perturbation_ids": ["p06", "p07"],
        "not_applicable_reason_codes": [
            "false_peak_placement_impossible",
            "insufficient_points_for_peak_model",
            "invalid_peak_component",
            "no_detected_peak",
            "nonpositive_intensity_range",
            "zero_false_peak_insertion_capacity",
        ],
        "p10": {
            "correlation_length_cm1": 20.0,
            "memory_budget_bytes": 64 * 2**30,
            "peak_estimate_formula": "32*N^2+64*N+2^30",
        },
        "phase1_native_gate_relative_tolerance": 1e-12,
        "schema_version": "phase4-d5-eligibility-config-v1",
        "support_grid": {
            "max_in_range_native_gap_cm1": 3.0,
            "point_count": 799,
            "start_cm1": 204.0,
            "step_cm1": 2.0,
            "stop_cm1": 1800.0,
        },
        "synthetic_fixture": True,
    }


def _synthetic_config(cohort: D5RawCohort, native: tuple[Spectrum1D, ...]):
    document = _config_document(cohort, native)
    raw = _canonical(document)
    return parse_phase4_d5_eligibility_config(
        Path("synthetic.json"), raw, require_frozen_identity=False
    )


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }


class Phase4D5EligibilityConfigTest(unittest.TestCase):
    def test_loads_only_the_exact_frozen_config(self) -> None:
        config = load_phase4_d5_eligibility_config(ELIGIBILITY_CONFIG)
        self.assertEqual(config.unique_record_count, 3012)
        self.assertEqual(config.class_count, 681)
        self.assertEqual(config.split_occurrence_count, 6621)
        self.assertEqual(config.expected_cell_count, 36144)
        self.assertEqual(config.expected_apply_call_count, 271080)
        self.assertEqual(config.p10_memory_budget_bytes, 64 * 2**30)

        changed = json.loads(ELIGIBILITY_CONFIG.read_text())
        changed["gates"]["p05_record_fraction"] = 0.89
        raw = _canonical(changed)
        with self.assertRaisesRegex(Phase4D5EligibilityError, "frozen config identity"):
            parse_phase4_d5_eligibility_config(
                Path("changed.json"), raw, require_frozen_identity=True
            )


class Phase4D5EligibilityLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = _synthetic_cohort()
        self.native = _synthetic_native(self.cohort)
        self.config = _synthetic_config(self.cohort, self.native)

    def test_reconstructs_literal_occurrences_and_unique_denominators(self) -> None:
        ledgers = reconstruct_d5_eligibility_ledgers(
            self.cohort, self.native, self.config
        )
        self.assertEqual(len(ledgers.unique_records), 5)
        self.assertEqual(len(ledgers.split_occurrences), 7)
        self.assertEqual(ledgers.unique_group_count, 3)
        self.assertEqual(ledgers.unique_class_count, 2)
        self.assertEqual(
            [row["record_id"] for row in ledgers.unique_records],
            ["r0", "r1", "r3", "r4", "r5"],
        )
        repeated = next(row for row in ledgers.unique_records if row["record_id"] == "r0")
        self.assertEqual(repeated["occurrence_count"], 2)
        self.assertEqual(repeated["split_seeds"], [0, 1])
        self.assertEqual(
            [(row["split_seed"], row["query_order"]) for row in ledgers.split_occurrences],
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
        )

    def test_gate_uses_original_records_and_fail_closed_classes(self) -> None:
        cohort = self.cohort
        native = self.native
        document = _config_document(cohort, native)
        document["denominators"] = {
            "class_count": 10,
            "group_count": 20,
            "split_occurrence_count": 20,
            "unique_record_count": 20,
        }
        document["expected"] = {
            "active_cell_count": 200,
            "apply_call_count": 1800,
            "cell_count": 240,
            "class_summary_count": 120,
            "inactive_cell_count": 40,
        }
        raw = _canonical(document)
        config = parse_phase4_d5_eligibility_config(
            Path("gate.json"), raw, require_frozen_identity=False
        )
        records = [
            {"class_label": index // 2, "group_id": f"g{index}", "record_id": f"r{index}"}
            for index in range(20)
        ]
        cells = [
            {
                "perturbation_id": perturbation_id,
                "record_id": record["record_id"],
                "state": (
                    "structurally_ineligible"
                    if perturbation_id in {"p06", "p07"}
                    else "complete"
                ),
            }
            for record in records
            for perturbation_id in ALL_PERTURBATIONS
        ]
        class_rows, common_rows, gate = evaluate_d5_eligibility_gates(
            records, cells, config
        )
        self.assertEqual(len(class_rows), 120)
        self.assertEqual(len(common_rows), 30)
        self.assertEqual(gate["full_domain_core"]["state"], "evaluable")
        self.assertEqual(gate["peak_common_support"]["state"], "evaluable")

        drifted = [dict(row) for row in cells]
        target = next(
            row
            for row in drifted
            if row["record_id"] == "r0" and row["perturbation_id"] == "p01"
        )
        target["state"] = "not_applicable"
        _, _, failed_gate = evaluate_d5_eligibility_gates(records, drifted, config)
        p01 = failed_gate["operators"]["p01"]
        self.assertEqual(p01["complete_record_count"], 19)
        self.assertEqual(p01["complete_class_count"], 9)
        self.assertEqual(p01["state"], "not_evaluable_coverage")

    def test_rejects_any_outcome_metric_or_alignment_field(self) -> None:
        for field in (
            "top1_correct",
            "mse",
            "metric_value",
            "downstream_harm",
            "alignment_gap",
            "p_value",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(Phase4D5EligibilityError, "outcome-blind"):
                    validate_outcome_blind_payload({"nested": [{field: 1.0}]})


class Phase4D5EligibilityArtifactTest(unittest.TestCase):
    def test_builds_exact_ledgers_and_rebuilds_byte_identically(self) -> None:
        cohort = _synthetic_cohort()
        native = _synthetic_native(cohort)
        config = _synthetic_config(cohort, native)
        sweep = load_perturbation_sweep_config(SWEEP_PATH)
        phase1 = load_phase1_core_config(PHASE1_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            summary = build_phase4_d5_eligibility_from_inputs(
                first,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=config,
                worker_count=2,
            )
            build_phase4_d5_eligibility_from_inputs(
                second,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                config=config,
                worker_count=1,
            )
            self.assertEqual(_tree(first), _tree(second))
            manifest = json.loads((first / "manifest.json").read_bytes())
            authority = manifest["run_identity"]["config_authority"]
            authority_path = ROOT / "rpe/runner/phase4_d5_eligibility_authority.py"
            self.assertEqual(authority["bytes"], authority_path.stat().st_size)
            self.assertEqual(
                authority["sha256"],
                hashlib.sha256(authority_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(summary.unique_record_count, 5)
            self.assertEqual(summary.split_occurrence_count, 7)
            self.assertEqual(summary.cell_count, 60)
            self.assertEqual(summary.class_summary_count, 24)
            observed = {path.name for path in first.iterdir() if path.is_file()}
            self.assertTrue(set(ARTIFACT_STATIC_FILES).issubset(observed))
            self.assertEqual(len(observed & {"complete.json", "failed.json"}), 1)
            cells = [json.loads(line) for line in (first / "cells.jsonl").read_text().splitlines()]
            self.assertEqual(len(cells), 60)
            self.assertEqual(
                [(row["record_id"], row["perturbation_id"]) for row in cells],
                [
                    (record_id, perturbation_id)
                    for record_id in ("r0", "r1", "r3", "r4", "r5")
                    for perturbation_id in ALL_PERTURBATIONS
                ],
            )
            for row in cells:
                if row["perturbation_id"] in {"p06", "p07"}:
                    self.assertEqual(row["state"], "structurally_ineligible")
                    self.assertEqual(
                        row["reason_code"],
                        "structurally_ineligible_missing_explicit_baseline",
                    )
            verified = verify_phase4_d5_eligibility_from_inputs(
                first,
                cohort=cohort,
                native_spectra=native,
                sweep=sweep,
                phase1_config=phase1,
                worker_count=2,
            )
            self.assertEqual(verified.run_id, summary.run_id)

    def test_cli_does_not_offer_a_skip_reexecution_verification_path(self) -> None:
        with self.assertRaises(SystemExit):
            eligibility_cli_main(
                [
                    "verify",
                    "--run-path",
                    "unread-path",
                    "--no-reexecute",
                ]
            )


if __name__ == "__main__":
    unittest.main()
