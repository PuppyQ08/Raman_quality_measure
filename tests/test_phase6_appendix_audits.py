from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from types import MappingProxyType, SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import interpolate, stats

import rpe.runner.phase6_appendix_audits as audits_module
from rpe.runner.phase6_appendix_audits import (
    AppendixAuditsConfig,
    AppendixAuditsError,
    AppendixAuditsInputs,
    AppendixAuditsSummary,
    build_phase6_appendix_audits,
    build_phase6_appendix_audits_from_inputs,
    exact_spectrum_sha256,
    load_phase6_appendix_audits_config,
    normalize_spectrum,
    project_phase6_peak_tolerance_rows,
    rank_stability_rows,
    resample_spectrum,
    scale_invariance_passes,
    scan_near_duplicates,
    select_calibration_pairs,
    assemble_phase4_sensitivity_rows,
)
from rpe.runner.phase6_appendix_metric_core import RecordSensitivityValues
from rpe.runner.phase6_appendix_verifier_science import (
    VerifierDatasetSensitivityResult,
    v_peak_metrics,
    v_resample,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/phase6/configs/appendix_audits_v1.json"
FIXTURE_ID = "phase6-appendix-audits-tiny-verifier-fixture-v1"


def process_payload_roundtrip(job: tuple[int, str]) -> tuple[int, str]:
    """Top-level probe used to prove the focused process-pool payload contract."""
    return job


def process_result_roundtrip(payload: object) -> object:
    """Return a packed record result from a real child process unchanged."""
    return payload


def canonical_json_bytes(value: object) -> bytes:
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


def exact_hash_reference(axis: np.ndarray, intensity: np.ndarray) -> str:
    import hashlib

    axis_le = np.asarray(axis, dtype="<f8")
    intensity_le = np.asarray(intensity, dtype="<f4")
    payload = bytearray()
    payload.extend(b"rpe-step7-exact-spectrum-v1")
    payload.extend(np.uint64(axis_le.size).astype("<u8").tobytes())
    payload.extend(axis_le.tobytes())
    payload.extend(np.uint64(intensity_le.size).astype("<u8").tobytes())
    payload.extend(intensity_le.tobytes())
    return hashlib.sha256(payload).hexdigest()


def files(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }


def rewrite_sha256sums(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_bytes())
    ordered = tuple(manifest["payload_files"]) + ("complete.json",)
    (path / "SHA256SUMS").write_bytes(
        b"".join(
            f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}\n".encode("utf-8")
            for name in ordered
        )
    )


def write_terminal_ledger_fixture(
    path: Path,
    *,
    terminals: tuple[str, ...],
    payload_name: str = "payload.json",
) -> None:
    payload_bytes = canonical_json_bytes({"payload": "fixture"})
    (path / payload_name).write_bytes(payload_bytes)
    entries: list[tuple[str, bytes]] = [(payload_name, payload_bytes)]
    for terminal_name in terminals:
        terminal_bytes = canonical_json_bytes({"status": terminal_name.removesuffix(".json")})
        (path / terminal_name).write_bytes(terminal_bytes)
        entries.append((terminal_name, terminal_bytes))
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n"
            for name, data in entries
        ),
        encoding="utf-8",
    )


def _panel_meta(config: AppendixAuditsConfig) -> dict[str, dict[str, object]]:
    return {
        str(row["panel_id"]): dict(row)
        for row in config.raw["panel_manifest"]
    }


def _numeric_panels(config: AppendixAuditsConfig) -> tuple[str, ...]:
    return tuple(
        str(panel_id)
        for panel_id in config.raw["fixed_panel_order"]
        if str(panel_id) != "d1_b_closed"
    )


def _resampling_conditions(config: AppendixAuditsConfig) -> tuple[str, ...]:
    return tuple(
        f"{interpolator}:{float(spacing):g}"
        for interpolator in ("linear", "cubic", "pchip")
        for spacing in config.raw["resampling_conditions"]["spacing_cm1"]
    )


def _normalization_conditions(config: AppendixAuditsConfig) -> tuple[str, ...]:
    return tuple(str(value) for value in config.raw["normalization_conditions"])


def _phase4_tolerance_conditions() -> tuple[str, ...]:
    return ("tolerance:1", "tolerance:2", "tolerance:4", "tolerance:8")


def _fixture_phase4_tables(
    config: AppendixAuditsConfig,
    *,
    component: str,
    conditions: tuple[str, ...],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    lookup = _panel_meta(config)
    alignments: list[dict[str, object]] = []
    ranks: list[dict[str, object]] = []
    reference = {
        "resampling": "native",
        "normalization": "none",
        "phase4_peak_tolerance": "tolerance:2",
    }[component]
    for panel_index, panel_id in enumerate(_numeric_panels(config)):
        meta = lookup[panel_id]
        for condition_index, condition_id in enumerate(conditions):
            basis = (panel_index + 1) * 1000 + (condition_index + 1) * 100
            for metric_index, metric_id in enumerate(config.raw["metric_ids"]):
                alignments.append(
                    {
                        "panel_id": panel_id,
                        "endpoint_id": meta["endpoint_id"],
                        "protocol_id": meta["protocol_id"],
                        "condition_id": condition_id,
                        "metric_output_id": metric_id,
                        "ag": float(basis + metric_index) / 100.0,
                        "acc_cross": float(500000 - basis - metric_index) / 1000000.0,
                        "state": "complete_numeric",
                        "reason_code": "",
                    }
                )
            for statistic in ("ag", "acc_cross"):
                ranks.append(
                    {
                        "panel_id": panel_id,
                        "endpoint_id": meta["endpoint_id"],
                        "protocol_id": meta["protocol_id"],
                        "condition_id": condition_id,
                        "statistic": statistic,
                        "reference_condition_id": reference,
                        "tau_b": 1.0,
                        "reference_tie_count": 0,
                        "candidate_tie_count": 0,
                        "max_abs_rank_displacement": 0.0,
                        "stable": True,
                        "state": "complete_numeric",
                        "reason_code": "",
                    }
                )
    return tuple(alignments), tuple(ranks)


def _fixture_system_ids() -> tuple[str, ...]:
    return tuple(f"fixture-system-{index:02d}" for index in range(21))


def _fixture_phase6_peak_rows(
    config: AppendixAuditsConfig,
    system_ids: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for system_index, system_id in enumerate(system_ids):
        for endpoint_index, endpoint in enumerate(config.raw["phase6_endpoint_manifest"]):
            base = float((system_index + 1) * 100 + endpoint_index) / 1000.0
            if endpoint["preferred_direction"] == "higher_is_better":
                curve = {"1": base - 0.01, "2": base, "4": base - 0.001, "8": base - 0.002}
            elif endpoint["preferred_direction"] == "lower_is_better":
                curve = {"1": base + 0.01, "2": base, "4": base + 0.001, "8": base + 0.002}
            else:
                curve = {"1": base, "2": base, "4": base, "8": base}
            rows.append(
                {
                    "system_id": system_id,
                    "family_id": f"fixture-family-{system_index % 3}",
                    "method_id": f"fixture-method-{system_index:02d}",
                    "endpoint_id": endpoint["endpoint_id"],
                    "curve_by_tolerance_cm1": curve,
                    "interval_by_tolerance_cm1": {
                        key: [value - 0.05, value + 0.05] for key, value in curve.items()
                    },
                    "design_class_count": 4,
                    "alpha_estimates_json": {"0.05": curve["2"]},
                    "contributing_class_counts_json": {"fixture-class": 4},
                    "state": "complete_numeric",
                    "reason_code": "",
                }
            )
    return tuple(rows)


def _fixture_leakage_rows(config: AppendixAuditsConfig) -> tuple[dict[str, object], ...]:
    inherited = {
        "phase3_denoising_fit_vs_transform",
        "phase6_baseline_d4_fit_selection_test_lifecycle",
        "phase6_denoising_identity_equivalence",
        "phase6_peak_claim_boundary",
    }
    rows: list[dict[str, object]] = []
    for index, definition in enumerate(config.raw["leakage_boundary_definitions"]):
        boundary_id = str(definition["boundary_id"])
        if boundary_id == "cross_dataset_provenance_d1_d2_d4_d5":
            status = "not_evaluable"
            reason = "not_evaluable_insufficient_cross_dataset_identity"
        elif boundary_id in inherited:
            status = "pass_via_parent_hash"
            reason = ""
        else:
            status = "pass"
            reason = ""
        rows.append(
            {
                "boundary_id": boundary_id,
                "producer_authorities": [],
                "consumer_authorities": [],
                "fit_roles": list(definition.get("fit_roles", ())),
                "selection_roles": list(definition.get("selection_roles", ())),
                "evaluation_roles": list(definition.get("evaluation_roles", ())),
                "leakage_entity": definition.get("leakage_entity", ""),
                "expected_overlap_semantics": definition.get("expected_overlap_semantics", ""),
                "recomputed_digests": {"fixture_digest": f"fixture-{index:02d}"},
                "recomputed_counts": {"fixture_count": index},
                "exact_overlap_count": 0,
                "near_candidate_count": 0,
                "status": status,
                "reason_code": reason,
                "evidence_paths": [f"fixture://{boundary_id}"],
            }
        )
    return tuple(rows)


def _fixture_audit_status_rows(
    config: AppendixAuditsConfig,
    *,
    system_ids: tuple[str, ...],
    leakage_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    statuses: list[dict[str, object]] = []
    order = 0
    lookup = _panel_meta(config)
    for component, conditions in (
        ("resampling", _resampling_conditions(config)),
        ("normalization", _normalization_conditions(config)),
        ("phase4_peak_tolerance", _phase4_tolerance_conditions()),
    ):
        for panel_id in config.raw["fixed_panel_order"]:
            meta = lookup[str(panel_id)]
            for condition_id in conditions:
                if str(panel_id) == "d1_b_closed":
                    statuses.append(
                        {
                            "component": component,
                            "status_order": order,
                            "panel_id": str(panel_id),
                            "endpoint_id": meta["endpoint_id"],
                            "protocol_id": meta["protocol_id"],
                            "condition_id": condition_id,
                            "state": "not_evaluable",
                            "reason_code": "not_evaluable_failed_alpha0_equivalence",
                            "numerical_row_count": 0,
                            "rank_row_count": 0,
                            "evidence_key": "fixture_d1_b_closure",
                        }
                    )
                else:
                    statuses.append(
                        {
                            "component": component,
                            "status_order": order,
                            "panel_id": str(panel_id),
                            "endpoint_id": meta["endpoint_id"],
                            "protocol_id": meta["protocol_id"],
                            "condition_id": condition_id,
                            "state": "complete_numeric",
                            "reason_code": "",
                            "numerical_row_count": 13,
                            "rank_row_count": 2,
                            "evidence_key": f"fixture_{component}",
                        }
                    )
                order += 1
    for endpoint in config.raw["phase6_endpoint_manifest"]:
        for tolerance in (1, 2, 4, 8):
            statuses.append(
                {
                    "component": "phase6_peak_tolerance",
                    "status_order": order,
                    "panel_id": "",
                    "endpoint_id": endpoint["endpoint_id"],
                    "protocol_id": "",
                    "condition_id": f"tolerance:{tolerance}",
                    "state": "complete_numeric",
                    "reason_code": "",
                    "numerical_row_count": len(system_ids),
                    "rank_row_count": 0 if endpoint["preferred_direction"] == "non_monotonic" or tolerance == 2 else 1,
                    "evidence_key": "fixture_step5_projection",
                }
            )
            order += 1
    for row in leakage_rows:
        statuses.append(
            {
                "component": "leakage_boundaries",
                "status_order": order,
                "panel_id": "",
                "endpoint_id": "",
                "protocol_id": "",
                "condition_id": row["boundary_id"],
                "state": row["status"],
                "reason_code": row["reason_code"],
                "numerical_row_count": 0,
                "rank_row_count": 0,
                "evidence_key": "fixture_leakage",
            }
        )
        order += 1
    return tuple(statuses)


def _fixture_inputs(config: AppendixAuditsConfig) -> AppendixAuditsInputs:
    system_ids = _fixture_system_ids()
    leakage_rows = _fixture_leakage_rows(config)
    resampling_alignment, resampling_rank = _fixture_phase4_tables(
        config,
        component="resampling",
        conditions=_resampling_conditions(config),
    )
    normalization_alignment, normalization_rank = _fixture_phase4_tables(
        config,
        component="normalization",
        conditions=_normalization_conditions(config),
    )
    peak_alignment, peak_rank = _fixture_phase4_tables(
        config,
        component="phase4_peak_tolerance",
        conditions=_phase4_tolerance_conditions(),
    )
    identity = {
        "run_identity": {
            "fixture_kind": FIXTURE_ID,
            "fixture_version": 1,
            "worker_count_identity_excluded": True,
        },
        "phase6_system_ids": system_ids,
        "audit_status": _fixture_audit_status_rows(
            config,
            system_ids=system_ids,
            leakage_rows=leakage_rows,
        ),
        "resampling_alignment": resampling_alignment,
        "resampling_rank_stability": resampling_rank,
        "normalization_alignment": normalization_alignment,
        "normalization_rank_stability": normalization_rank,
        "peak_tolerance_phase4_alignment": peak_alignment,
        "peak_tolerance_phase4_rank_stability": peak_rank,
        "leakage_duplicate_candidates": (),
    }
    authority_bridge = {
        "fixture_kind": FIXTURE_ID,
        "fixture_version": 1,
        "schema_version": config.raw["schema_version"],
        "expected_rows": dict(config.raw["expected_rows"]),
    }
    return AppendixAuditsInputs(
        panel_inputs=(),
        phase6_peak_rows=_fixture_phase6_peak_rows(config, system_ids),
        leakage_inputs=leakage_rows,
        authority_bridge=authority_bridge,
        identity=identity,
    )


def _fixture_sugar_synthetic_inputs(config: AppendixAuditsConfig) -> dict[str, object]:
    metric_ids = tuple(str(value) for value in config.raw["metric_ids"])
    parent_rows: list[dict[str, object]] = []
    for well_index, well_id in enumerate(("fixture-well-a", "fixture-well-b")):
        for perturbation_index, perturbation_id in enumerate(("p08", "p09")):
            for alpha_index, alpha in enumerate((0.05, 0.1)):
                downstream = float((well_index + 1) * 100 + (perturbation_index + 1) * 10 + alpha_index + 1)
                for metric_index, metric_id in enumerate(metric_ids):
                    parent_rows.append(
                        {
                            "well_id": well_id,
                            "perturbation_id": perturbation_id,
                            "alpha": alpha,
                            "metric_output_id": metric_id,
                            "metric_harm": downstream + float(metric_index + 1) / 1000.0,
                            "downstream_harm": downstream,
                            "state": "complete",
                        }
                    )
    reference_seed = assemble_phase4_sensitivity_rows(
        parent_rows,
        parent_rows,
        panel_id="d4_a",
        endpoint_id="sugar_quantification",
        protocol_id="a",
        condition_id="native",
        metric_ids=metric_ids,
        reference_by_metric={metric_id: (0.0, 0.0) for metric_id in metric_ids},
    )
    reference = MappingProxyType(
        {
            str(row["metric_output_id"]): (float(row["ag"]), float(row["acc_cross"]))
            for row in reference_seed.alignment_rows
        }
    )
    return {
        "records": (),
        "parent_rows": MappingProxyType(
            {"d4_a": tuple(parent_rows), "d4_b": tuple(parent_rows)}
        ),
        "parent_alignment": MappingProxyType({"d4_a": reference, "d4_b": reference}),
    }


class Phase6AppendixAuditsPublicSurfaceTest(unittest.TestCase):
    def test_public_surface_exists(self) -> None:
        self.assertTrue(issubclass(AppendixAuditsError, ValueError))
        for value in (
            AppendixAuditsConfig,
            AppendixAuditsInputs,
            AppendixAuditsSummary,
            load_phase6_appendix_audits_config,
            build_phase6_appendix_audits_from_inputs,
            build_phase6_appendix_audits,
            resample_spectrum,
            normalize_spectrum,
            scale_invariance_passes,
            rank_stability_rows,
            exact_spectrum_sha256,
            select_calibration_pairs,
            scan_near_duplicates,
            project_phase6_peak_tolerance_rows,
        ):
            self.assertTrue(callable(value))
        self.assertTrue(hasattr(audits_module, "__file__"))


class Phase6AppendixAuditsConfigContractTest(unittest.TestCase):
    def load_document(self) -> dict[str, object]:
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_binds_frozen_literals_counts_and_output_order(self) -> None:
        raw = CONFIG.read_bytes()
        document = self.load_document()
        self.assertEqual(raw, canonical_json_bytes(document))
        config = load_phase6_appendix_audits_config(CONFIG)
        self.assertEqual(config.raw["schema_version"], "phase6-appendix-audits-v1")
        self.assertEqual(config.raw["source_revision_status"], "unavailable_no_valid_git_repository")
        self.assertEqual(
            config.raw["authorities"]["appendix_protocol"],
            {
                "byte_count": 29229,
                "path": "reports/phase6/step06_appendix_sensitivity_leakage_protocol.md",
                "sha256": "b480fdc0433a2c1af76c433343575530fdf195a2ac9684e1e913c01987fa5ad4",
            },
        )
        self.assertEqual(tuple(config.raw["phase4_condition_ids"]), ("P8", "P9", "P10", "P11", "P12"))
        self.assertEqual(tuple(config.raw["alpha_grid"]), (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8))
        self.assertEqual(tuple(config.raw["positive_alpha_grid"]), (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8))
        self.assertEqual(
            tuple(config.raw["artifact_contract"]["payload_files"]),
            (
                "config.json",
                "authority_bridge.json",
                "preflight.json",
                "audit_status.csv",
                "resampling_alignment.csv",
                "resampling_rank_stability.csv",
                "normalization_alignment.csv",
                "normalization_rank_stability.csv",
                "peak_tolerance_phase4_alignment.csv",
                "peak_tolerance_phase4_rank_stability.csv",
                "peak_tolerance_phase6_system.csv",
                "peak_tolerance_phase6_rank_stability.csv",
                "leakage_boundaries.jsonl",
                "leakage_duplicate_candidates.jsonl",
                "summary.md",
                "manifest.json",
            ),
        )
        self.assertEqual(
            dict(config.raw["status_row_counts_by_component"]),
            {
                "resampling": 108,
                "normalization": 48,
                "phase4_peak_tolerance": 48,
                "phase6_peak_tolerance": 40,
                "leakage_boundaries": 15,
            },
        )


class Phase6AppendixAuditsPrimitiveBehaviorTest(unittest.TestCase):
    def test_phase4_aggregation_imports_one_binary64_downstream_and_enforces_identity(self) -> None:
        parent = []
        fresh = []
        for metric_id in ("mse", "rmse"):
            for perturbation_id, alpha, harm in (("P8", 0.05, 0.1), ("P8", 0.1, 0.15), ("P9", 0.05, 0.2), ("P9", 0.1, 0.25)):
                parent.append({
                    "class_label": "c1", "perturbation_id": perturbation_id,
                    "alpha": alpha, "metric_output_id": metric_id,
                    "metric_harm": harm, "downstream_harm": 0.25 if perturbation_id == "P8" else 0.5, "state": "complete",
                })
                fresh.append({
                    "class_label": "c1", "perturbation_id": perturbation_id,
                    "alpha": alpha, "metric_output_id": metric_id,
                    "metric_harm": harm, "state": "complete",
                })
        result = assemble_phase4_sensitivity_rows(
            parent, fresh, panel_id="fixture", endpoint_id="dX", protocol_id="a",
            condition_id="tolerance:2", metric_ids=("mse", "rmse"),
            reference_by_metric={"mse": (0.0, 1.0), "rmse": (0.0, 1.0)},
            reference_condition_id="tolerance:2",
            require_identity=True,
        )
        self.assertEqual(len(result.alignment_rows), 2)
        self.assertEqual(len(result.rank_rows), 2)
        self.assertEqual(
            {row["reference_condition_id"] for row in result.rank_rows},
            {"tolerance:2"},
        )
        self.assertTrue(all(row["state"] == "complete_numeric" for row in result.alignment_rows))
        inconsistent = list(parent)
        inconsistent[-1] = {**inconsistent[-1], "downstream_harm": 0.5000000000000001}
        with self.assertRaisesRegex(AppendixAuditsError, "downstream harm"):
            assemble_phase4_sensitivity_rows(
                inconsistent, fresh, panel_id="fixture", endpoint_id="dX", protocol_id="a",
                condition_id="tolerance:2", metric_ids=("mse", "rmse"),
                reference_by_metric={"mse": (0.0, 1.0), "rmse": (0.0, 1.0)},
                require_identity=True,
            )

    def test_resampling_matches_three_definitions_and_rejects_extrapolation(self) -> None:
        axis = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        intensity = np.array([0.0, 1.0, 0.25, 2.0], dtype=np.float64)
        target = np.array([0.5, 1.5, 2.5], dtype=np.float64)

        expected_linear = np.interp(target, axis, intensity)
        expected_cubic = interpolate.CubicSpline(
            axis, intensity, bc_type="not-a-knot", extrapolate=False
        )(target)
        expected_pchip = interpolate.PchipInterpolator(axis, intensity, extrapolate=False)(target)

        np.testing.assert_allclose(
            resample_spectrum(axis, intensity, target, interpolator="linear"),
            expected_linear,
        )
        np.testing.assert_allclose(
            resample_spectrum(axis, intensity, target, interpolator="cubic"),
            expected_cubic,
        )
        np.testing.assert_allclose(
            resample_spectrum(axis, intensity, target, interpolator="pchip"),
            expected_pchip,
        )

        with self.assertRaisesRegex(AppendixAuditsError, "extrapolation|support|bracket"):
            resample_spectrum(axis, intensity, np.array([-0.1, 0.5], dtype=np.float64), interpolator="linear")

    def test_resampling_enforces_inherited_native_gap_before_interpolation(self) -> None:
        axis = np.array([0.0, 1.0, 3.0, 4.0], dtype=np.float64)
        intensity = np.array([0.0, 1.0, 0.25, 2.0], dtype=np.float64)
        target = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        with self.assertRaisesRegex(AppendixAuditsError, "native gap"):
            resample_spectrum(
                axis, intensity, target, interpolator="linear",
                max_in_range_native_gap_cm1=1.5,
            )
        with self.assertRaisesRegex(ValueError, "native gap"):
            v_resample(axis, intensity, target, "linear", max_in_range_native_gap_cm1=1.5)

    def test_normalization_domains_and_registered_invariance_controls(self) -> None:
        axis = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        positive = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        expected_maximum = positive / 4.0
        expected_area = positive / np.trapezoid(np.abs(positive), axis)
        expected_snv = (positive - positive.mean()) / positive.std(ddof=0)

        np.testing.assert_allclose(normalize_spectrum(axis, positive, method="none"), positive)
        np.testing.assert_allclose(normalize_spectrum(axis, positive, method="maximum"), expected_maximum)
        np.testing.assert_allclose(normalize_spectrum(axis, positive, method="area"), expected_area)
        np.testing.assert_allclose(normalize_spectrum(axis, positive, method="snv"), expected_snv)

        with self.assertRaisesRegex(AppendixAuditsError, "max|domain|positive"):
            normalize_spectrum(axis, np.array([0.0, -1.0, 0.0], dtype=np.float64), method="maximum")
        with self.assertRaisesRegex(AppendixAuditsError, "std|domain|constant"):
            normalize_spectrum(axis, np.array([2.0, 2.0, 2.0], dtype=np.float64), method="snv")

        self.assertTrue(scale_invariance_passes(10.0, 10.0 + 1e-12))
        self.assertFalse(scale_invariance_passes(10.0, 10.0 + 1e-6))
        edge = 1e-12 + math.ulp(1e-12)
        self.assertFalse(scale_invariance_passes(0.0, edge))

    def test_verifier_peak_metrics_match_empty_denominator_contract(self) -> None:
        both_empty = v_peak_metrics((), (), tolerance_cm1=2.0)
        candidate_empty = v_peak_metrics((object(),), (), tolerance_cm1=2.0)
        reference_empty = v_peak_metrics((), (object(),), tolerance_cm1=2.0)
        self.assertEqual((both_empty["precision"], both_empty["recall"], both_empty["f1"]), (1.0, 1.0, 1.0))
        self.assertEqual((candidate_empty["precision"], candidate_empty["recall"], candidate_empty["f1"]), (1.0, 0.0, 0.0))
        self.assertEqual((reference_empty["precision"], reference_empty["recall"], reference_empty["f1"]), (0.0, 1.0, 0.0))

    def test_rank_stability_uses_orientation_average_ties_and_strict_tau_b(self) -> None:
        metric_ids = (
            "mse",
            "rmse",
            "mae",
            "sam",
            "pearson_r",
            "nmse",
            "wasserstein_1_cm1",
            "is_like_structure_to_noise",
            "precision",
            "recall",
            "f1",
            "artifact_peak_ratio",
            "missing_peak_ratio",
        )
        reference = {
            "mse": 0.1,
            "rmse": 0.2,
            "mae": 0.3,
            "sam": 0.4,
            "pearson_r": 0.5,
            "nmse": 0.6,
            "wasserstein_1_cm1": 0.7,
            "is_like_structure_to_noise": 0.8,
            "precision": 0.9,
            "recall": 1.0,
            "f1": 1.1,
            "artifact_peak_ratio": 1.2,
            "missing_peak_ratio": 1.3,
        }
        candidate = dict(reference)
        candidate["rmse"] = candidate["mse"]
        rows = rank_stability_rows(reference, candidate, statistic="ag", metric_ids=metric_ids)
        self.assertEqual(rows["statistic"], "ag")
        self.assertEqual(rows["state"], "complete_numeric")
        self.assertEqual(rows["candidate_tie_count"], 1)
        self.assertGreaterEqual(rows["tau_b"], 0.9)

        reversed_candidate = {metric_id: -reference[metric_id] for metric_id in metric_ids}
        reversed_rows = rank_stability_rows(reference, reversed_candidate, statistic="ag", metric_ids=metric_ids)
        self.assertEqual(reversed_rows["reason_code"], "complete_ranking_instability_or_reversal")
        self.assertFalse(reversed_rows["stable"])

        tied = {metric_id: 1.0 for metric_id in metric_ids}
        all_tied = rank_stability_rows(reference, tied, statistic="acc_cross", metric_ids=metric_ids)
        self.assertEqual(all_tied["reason_code"], "not_evaluable_all_tied_ranking")

        incomplete = dict(reference)
        incomplete.pop("f1")
        missing = rank_stability_rows(reference, incomplete, statistic="ag", metric_ids=metric_ids)
        self.assertEqual(missing["reason_code"], "not_evaluable_incomplete_metric_set")

    def test_phase4_peak_projection_reuses_peaks_and_tolerance_two_is_identity(self) -> None:
        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(document["phase4_peak_tolerance"]["reference_tolerance_cm1"], 2)
        self.assertEqual(
            tuple(document["phase4_peak_tolerance"]["varying_metric_ids"]),
            ("precision", "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio"),
        )
        self.assertEqual(
            tuple(document["phase4_peak_tolerance"]["bit_identical_metric_ids"]),
            (
                "mse",
                "rmse",
                "mae",
                "sam",
                "pearson_r",
                "nmse",
                "wasserstein_1_cm1",
                "is_like_structure_to_noise",
            ),
        )

    def test_phase6_peak_projection_uses_step5_curves_and_requires_all_systems(self) -> None:
        endpoint_manifest = (
            {"endpoint_id": "P1-retention", "preferred_direction": "higher_is_better"},
            {"endpoint_id": "P3-position-mae", "preferred_direction": "lower_is_better"},
            {"endpoint_id": "P3-signed-log2-fwhm-ratio", "preferred_direction": "non_monotonic"},
        )
        system_ids = ("sys-a", "sys-b")
        rows = (
            {
                "system_id": "sys-a",
                "family_id": "fam",
                "method_id": "m1",
                "endpoint_id": "P1-retention",
                "curve_by_tolerance_cm1": {"1": 0.8, "2": 0.8, "4": 0.79, "8": 0.78},
                "interval_by_tolerance_cm1": {"1": [0.7, 0.9], "2": [0.7, 0.9], "4": [0.69, 0.89], "8": [0.68, 0.88]},
                "design_class_count": 10,
                "alpha_estimates_json": {"0.05": 0.8},
                "contributing_class_counts_json": {"class-a": 2},
            },
            {
                "system_id": "sys-b",
                "family_id": "fam",
                "method_id": "m2",
                "endpoint_id": "P1-retention",
                "curve_by_tolerance_cm1": {"1": 0.6, "2": 0.6, "4": 0.61, "8": 0.62},
                "interval_by_tolerance_cm1": {"1": [0.5, 0.7], "2": [0.5, 0.7], "4": [0.51, 0.71], "8": [0.52, 0.72]},
                "design_class_count": 10,
                "alpha_estimates_json": {"0.05": 0.6},
                "contributing_class_counts_json": {"class-a": 2},
            },
            {
                "system_id": "sys-a",
                "family_id": "fam",
                "method_id": "m1",
                "endpoint_id": "P3-position-mae",
                "curve_by_tolerance_cm1": {"1": 0.3, "2": 0.2, "4": 0.2, "8": 0.2},
                "interval_by_tolerance_cm1": {"1": [0.2, 0.4], "2": [0.1, 0.3], "4": [0.1, 0.3], "8": [0.1, 0.3]},
                "design_class_count": 10,
                "alpha_estimates_json": {"0.05": 0.2},
                "contributing_class_counts_json": {"class-a": 2},
            },
            {
                "system_id": "sys-b",
                "family_id": "fam",
                "method_id": "m2",
                "endpoint_id": "P3-position-mae",
                "curve_by_tolerance_cm1": {"1": 0.4, "2": 0.5, "4": 0.5, "8": 0.5},
                "interval_by_tolerance_cm1": {"1": [0.3, 0.5], "2": [0.4, 0.6], "4": [0.4, 0.6], "8": [0.4, 0.6]},
                "design_class_count": 10,
                "alpha_estimates_json": {"0.05": 0.5},
                "contributing_class_counts_json": {"class-a": 2},
            },
            {
                "system_id": "sys-a",
                "family_id": "fam",
                "method_id": "m1",
                "endpoint_id": "P3-signed-log2-fwhm-ratio",
                "curve_by_tolerance_cm1": {"1": 0.0, "2": 0.0, "4": 0.0, "8": 0.0},
                "interval_by_tolerance_cm1": {"1": [-0.1, 0.1], "2": [-0.1, 0.1], "4": [-0.1, 0.1], "8": [-0.1, 0.1]},
                "design_class_count": 10,
                "alpha_estimates_json": {"0.05": 0.0},
                "contributing_class_counts_json": {"class-a": 2},
            },
            {
                "system_id": "sys-b",
                "family_id": "fam",
                "method_id": "m2",
                "endpoint_id": "P3-signed-log2-fwhm-ratio",
                "curve_by_tolerance_cm1": {"1": 0.1, "2": 0.1, "4": 0.1, "8": 0.1},
                "interval_by_tolerance_cm1": {"1": [0.0, 0.2], "2": [0.0, 0.2], "4": [0.0, 0.2], "8": [0.0, 0.2]},
                "design_class_count": 10,
                "alpha_estimates_json": {"0.05": 0.1},
                "contributing_class_counts_json": {"class-a": 2},
            },
        )
        system_rows, rank_rows = project_phase6_peak_tolerance_rows(
            rows,
            system_ids=system_ids,
            endpoint_manifest=endpoint_manifest,
            tolerances=(1, 2, 4, 8),
        )
        self.assertEqual(len(system_rows), 24)
        self.assertEqual(len(rank_rows), 6)
        self.assertTrue(all(row["endpoint_id"] != "P3-signed-log2-fwhm-ratio" for row in rank_rows))
        self.assertTrue(all(row["reference_tolerance_cm1"] == 2 for row in rank_rows))

    def test_exact_hash_and_near_duplicate_calibration_follow_literal_protocol(self) -> None:
        axis = np.array([387.0, 388.0, 389.0], dtype=np.float64)
        intensity = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        self.assertEqual(exact_spectrum_sha256(axis, intensity), exact_hash_reference(axis, intensity))

        pairs = select_calibration_pairs(
            tuple(range(6)),
            ("a", "a", "b", "c", "d", "e"),
            seed=20260817,
            target_count=100,
            batch_size=262144,
            max_batches=64,
        )
        admissible = {(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)}
        self.assertEqual(pairs, tuple(sorted(admissible)))

        left_records = (
            {"role_id": "fit", "record_id": "fit-a", "entity_id": "a", "source_sha256": "s1", "axis": np.array([387.0, 388.0, 389.0]), "intensity": np.array([0.0, 1.0, 2.0])},
            {"role_id": "fit", "record_id": "fit-b", "entity_id": "b", "source_sha256": "s2", "axis": np.array([387.0, 388.0, 389.0]), "intensity": np.array([5.0, 5.0, 5.0])},
        )
        right_records = (
            {"role_id": "test", "record_id": "test-a", "entity_id": "z", "source_sha256": "s3", "axis": np.array([387.0, 388.0, 389.0]), "intensity": np.array([0.0, 1.0, 2.0])},
            {"role_id": "test", "record_id": "test-b", "entity_id": "y", "source_sha256": "s4", "axis": np.array([387.0, 388.0, 389.0]), "intensity": np.array([0.0, 1.0005, 2.0005])},
        )
        candidates = scan_near_duplicates(
            left_records,
            right_records,
            threshold=1e-3,
            query_block_size=2,
            fit_block_size=2,
        )
        self.assertTrue(any(row["left_record_id"] == "fit-a" and row["right_record_id"] == "test-a" for row in candidates))
        self.assertTrue(any(row["left_record_id"] == "fit-a" and row["right_record_id"] == "test-b" for row in candidates))
        self.assertTrue(any("typed_closure" in row["reason_code"] for row in candidates))

    def test_scalable_leakage_uses_count_arithmetic_and_vector_blocks(self) -> None:
        for relative in (
            "rpe/runner/phase6_appendix_leakage.py",
            "rpe/runner/phase6_appendix_verify_leakage.py",
            "rpe/runner/phase6_appendix_verifier_science.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("admissible = [", source)
            self.assertNotIn("pair_cells", source)
        for relative in (
            "rpe/runner/phase6_appendix_bacteria.py",
            "rpe/runner/phase6_appendix_rruff.py",
            "rpe/runner/phase6_appendix_verify_bacteria.py",
            "rpe/runner/phase6_appendix_verify_rruff.py",
        ):
            self.assertIn("ProcessPoolExecutor", (ROOT / relative).read_text(encoding="utf-8"))

    def test_process_workers_receive_only_pickle_safe_jobs(self) -> None:
        modules = (
            ("rpe.runner.phase6_appendix_bacteria", "rpe/runner/phase6_appendix_bacteria.py"),
            ("rpe.runner.phase6_appendix_rruff", "rpe/runner/phase6_appendix_rruff.py"),
            ("rpe.runner.phase6_appendix_verify_bacteria", "rpe/runner/phase6_appendix_verify_bacteria.py"),
            ("rpe.runner.phase6_appendix_verify_rruff", "rpe/runner/phase6_appendix_verify_rruff.py"),
        )
        for module_name, relative in modules:
            module = __import__(module_name, fromlist=["_process_job"])
            self.assertTrue(hasattr(module, "_process_job"), module_name)
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("initializer=_init_process_worker", source)
            self.assertIn("executor.map(_process_job, jobs)", source)
        jobs = ((0, "record-a"), (1, "record-b"))
        for job in jobs:
            pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL)
        with ProcessPoolExecutor(max_workers=2) as executor:
            self.assertEqual(tuple(executor.map(process_payload_roundtrip, jobs)), jobs)

    def test_process_workers_roundtrip_packed_record_results(self) -> None:
        production = RecordSensitivityValues(
            "record", ("p08:01",), ("mse",), ("linear:1",), ("none",), ("2",),
            np.zeros((1, 1, 1), dtype="<f8"), np.ones((1, 1, 1), dtype="<f8"),
            np.full((1, 1, 1), 2.0, dtype="<f8"),
            MappingProxyType({"normalization:none": "ValueError"}),
            MappingProxyType({"detector_call_count": 41}),
        )
        verifier = VerifierDatasetSensitivityResult(
            "record", ("p08:01",), ("mse",), ("linear:1",), ("none",), ("2",),
            MappingProxyType({"p08:01": MappingProxyType({"mse": 0.0})}),
            MappingProxyType({"normalization:none": "ValueError"}),
            np.zeros((1, 1, 1), dtype="<f8"), np.ones((1, 1, 1), dtype="<f8"),
            np.full((1, 1, 1), 2.0, dtype="<f8"),
            MappingProxyType({"detector_call_count": 41}),
        )
        modules = (
            "rpe.runner.phase6_appendix_bacteria", "rpe.runner.phase6_appendix_rruff",
            "rpe.runner.phase6_appendix_sugar", "rpe.runner.phase6_appendix_verify_bacteria",
            "rpe.runner.phase6_appendix_verify_rruff", "rpe.runner.phase6_appendix_verify_sugar",
        )
        for module_name in modules:
            module = __import__(module_name, fromlist=["_pack_record_result", "_unpack_record_result"])
            source = production if "verify" not in module_name else verifier
            packed = module._pack_record_result(source)
            pickle.dumps(packed, protocol=pickle.HIGHEST_PROTOCOL)
            with ProcessPoolExecutor(max_workers=2) as executor:
                returned, = executor.map(process_result_roundtrip, (packed,))
            restored = module._unpack_record_result(returned)
            self.assertIsInstance(restored.closure_reasons, MappingProxyType)
            self.assertIsInstance(restored.identity, MappingProxyType)
            self.assertFalse(restored.resampling_harms.flags.writeable)
            np.testing.assert_array_equal(restored.resampling_harms, source.resampling_harms)


class Phase6AppendixAuditsIntegrationTest(unittest.TestCase):
    def fixture(self) -> tuple[AppendixAuditsConfig, AppendixAuditsInputs]:
        config = load_phase6_appendix_audits_config(CONFIG)
        return config, _fixture_inputs(config)

    def test_validate_ledger_accepts_declared_failed_parent_terminal(self) -> None:
        from rpe.runner.phase6_appendix_audits_verifier import _validate_ledger

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_terminal_ledger_fixture(run, terminals=("failed.json",))
            _validate_ledger(run, expected_names=None, terminal_name="failed.json")

    def test_validate_ledger_rejects_dual_terminal_files(self) -> None:
        from rpe.runner.phase6_appendix_audits_verifier import (
            AppendixAuditsVerificationError,
            _validate_ledger,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_terminal_ledger_fixture(run, terminals=("failed.json", "complete.json"))
            with self.assertRaisesRegex(
                AppendixAuditsVerificationError,
                "exactly one|terminal",
            ):
                _validate_ledger(run, expected_names=None, terminal_name="failed.json")

    def test_validate_ledger_rejects_undeclared_terminal_name(self) -> None:
        from rpe.runner.phase6_appendix_audits_verifier import (
            AppendixAuditsVerificationError,
            _validate_ledger,
        )

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_terminal_ledger_fixture(run, terminals=("failed.json",))
            with self.assertRaisesRegex(
                AppendixAuditsVerificationError,
                "terminal",
            ):
                _validate_ledger(run, expected_names=None, terminal_name="unexpected.json")

    def test_sugar_success_statuses_use_component_names_and_canonicalize_without_keyerror(self) -> None:
        import rpe.runner.phase6_appendix_sugar as sugar_module

        config = load_phase6_appendix_audits_config(CONFIG)
        synthetic = _fixture_sugar_synthetic_inputs(config)
        fresh_rows = tuple(synthetic["parent_rows"]["d4_a"])
        with mock.patch.object(
            sugar_module,
            "_rows_from_records",
            return_value=(fresh_rows, ""),
        ):
            sugar = sugar_module.build_sugar_phase4_sensitivity(
                config.raw,
                ROOT,
                1,
                _synthetic=synthetic,
            )
        ordered = audits_module._canonicalize_inputs(
            config,
            {
                "audit_status": sugar.audit_status,
                "resampling_alignment": sugar.resampling_alignment,
                "resampling_rank_stability": sugar.resampling_rank_stability,
                "normalization_alignment": sugar.normalization_alignment,
                "normalization_rank_stability": sugar.normalization_rank_stability,
                "peak_tolerance_phase4_alignment": sugar.peak_tolerance_phase4_alignment,
                "peak_tolerance_phase4_rank_stability": sugar.peak_tolerance_phase4_rank_stability,
            },
            SimpleNamespace(boundary_rows=(), candidate_rows=(), status_rows=()),
            _fixture_phase6_peak_rows(config, _fixture_system_ids()),
            _fixture_system_ids(),
        )
        self.assertEqual(
            {
                row["component"]
                for row in ordered["audit_status"]
                if row.get("panel_id") in {"d4_a", "d4_b"}
                and row["state"] == "complete_numeric"
            },
            {"resampling", "normalization", "phase4_peak_tolerance"},
        )

    def test_fixture_artifact_has_fixed_slots_typed_closures_and_exact_inventory(self) -> None:
        config, inputs = self.fixture()
        inputs = AppendixAuditsInputs(
            panel_inputs=inputs.panel_inputs,
            phase6_peak_rows=inputs.phase6_peak_rows,
            leakage_inputs=inputs.leakage_inputs,
            authority_bridge=MappingProxyType({"fixture": np.int64(1), "nested": MappingProxyType({"array": np.array([1, 2], dtype=np.int64)})}),
            identity=MappingProxyType({**inputs.identity, "run_identity": MappingProxyType({"fixture_kind": FIXTURE_ID, "tuple": (np.float64(1.5),)})}),
        )
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_appendix_audits_from_inputs(
                inputs,
                Path(output_root),
                config=config,
                worker_count=1,
            )
            present = files(summary.path)
            self.assertEqual(
                set(present),
                set(config.raw["artifact_contract"]["payload_files"]) | {"complete.json", "SHA256SUMS"},
            )
            self.assertEqual(
                len((summary.path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()),
                17,
            )
            manifest = json.loads((summary.path / "manifest.json").read_bytes())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["counts"]["audit_status"], 259)
            self.assertEqual(manifest["counts"]["resampling_alignment"], 1287)
            self.assertEqual(manifest["counts"]["leakage_duplicate_candidates"], 0)
            self.assertFalse((summary.path / "failed.json").exists())
            rows = tuple(
                csv.DictReader(
                    (summary.path / "audit_status.csv").read_text(encoding="utf-8").splitlines()
                )
            )
            d1_b = tuple(row for row in rows if row["panel_id"] == "d1_b_closed")
            self.assertEqual(len(d1_b), 17)
            self.assertTrue(all(row["state"] == "not_evaluable" for row in d1_b))
            self.assertEqual(
                {row["reason_code"] for row in d1_b},
                {"not_evaluable_failed_alpha0_equivalence"},
            )

    def test_worker_count_does_not_change_run_id_or_bytes(self) -> None:
        config, inputs = self.fixture()
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = build_phase6_appendix_audits_from_inputs(
                inputs,
                Path(first_root),
                config=config,
                worker_count=1,
            )
            second = build_phase6_appendix_audits_from_inputs(
                inputs,
                Path(second_root),
                config=config,
                worker_count=7,
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(files(first.path), files(second.path))

    def test_adapter_merge_uses_declared_panel_and_condition_order(self) -> None:
        config = load_phase6_appendix_audits_config(CONFIG)
        empty = {name: () for name in (
            "audit_status", "resampling_alignment", "resampling_rank_stability",
            "normalization_alignment", "normalization_rank_stability",
            "peak_tolerance_phase4_alignment", "peak_tolerance_phase4_rank_stability",
        )}
        empty["resampling_alignment"] = (
            {"panel_id": "d4_a", "condition_id": "pchip:2", "metric_output_id": "rmse"},
            {"panel_id": "d5_a", "condition_id": "linear:0.5", "metric_output_id": "mse"},
        )
        leakage = SimpleNamespace(boundary_rows=(), candidate_rows=(), status_rows=())
        system_ids = tuple(str(value) for value in json.loads((ROOT / config.raw["authorities"]["phase6_peak_checksum_ledger"]["path"]).parent.joinpath("config.json").read_text())["promoted_system_ids"])
        peak_rows = tuple(json.loads(line) for line in (ROOT / config.raw["authorities"]["phase6_peak_checksum_ledger"]["path"]).parent.joinpath("bootstrap_results.jsonl").read_text().splitlines())
        ordered = audits_module._canonicalize_inputs(config, empty, leakage, peak_rows, system_ids)
        self.assertEqual([row["panel_id"] for row in ordered["resampling_alignment"]], ["d5_a", "d4_a"])

    def test_verifier_rebuilds_bytes_and_rejects_checksum_rewritten_tamper(self) -> None:
        from rpe.runner.phase6_appendix_audits_verifier import (
            AppendixAuditsVerificationError,
            verify_phase6_appendix_audits,
        )

        config, inputs = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_appendix_audits_from_inputs(
                inputs,
                Path(output_root),
                config=config,
                worker_count=1,
            )
            verified = verify_phase6_appendix_audits(
                summary.path,
                worker_count=2,
                project_root=ROOT,
            )
            self.assertEqual(verified.run_id, summary.run_id)
            self.assertEqual(verified.status, "complete")
            self.assertEqual(verified.counts["audit_status"], 259)

            target = summary.path / "resampling_alignment.csv"
            rows = list(csv.DictReader(target.read_text(encoding="utf-8").splitlines()))
            rows[0]["ag"] = "999.125"
            with target.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=tuple(rows[0]),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            rewrite_sha256sums(summary.path)
            with self.assertRaisesRegex(
                AppendixAuditsVerificationError,
                "differ|mismatch|byte|payload",
            ):
                verify_phase6_appendix_audits(
                    summary.path,
                    worker_count=3,
                    project_root=ROOT,
                )

    def test_verifier_does_not_import_production_runner(self) -> None:
        verifier_path = ROOT / "rpe/runner/phase6_appendix_audits_verifier.py"
        tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
        forbidden_modules = {
            "rpe.runner.phase6_appendix_audits",
            "rpe.runner.phase6_appendix_metric_core",
            "rpe.runner.phase6_appendix_bacteria",
            "rpe.runner.phase6_appendix_sugar",
            "rpe.runner.phase6_appendix_rruff",
            "rpe.runner.phase6_appendix_leakage",
        }
        dynamic_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name, forbidden_modules)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, forbidden_modules)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in forbidden_modules:
                    dynamic_names.append(node.value)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec"}:
                    self.fail("verifier must not dynamically import or exec production code")
                if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    self.fail("verifier must not dynamically import production modules")
        self.assertEqual(dynamic_names, [])

    def test_cli_exposes_only_build_verify_and_worker_controls(self) -> None:
        completed = subprocess.run([sys.executable, "tools/run_phase6_appendix_audits.py", "--help"], cwd=ROOT, check=True, text=True, capture_output=True)
        self.assertIn("{build,verify}", completed.stdout)
        self.assertNotIn("--config", completed.stdout)

