from __future__ import annotations

import ast
import hashlib
import importlib
import json
import struct
import sys
import tempfile
import unittest
from collections import Counter
from collections.abc import Mapping as MappingABC
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.sugar_quantitative import (  # noqa: E402
    D4SugarCohort,
    D4WellSplit,
    load_d4_sugar_cohort,
)


SUBJECT_MODULE = "rpe.runner.phase4_d4_protocol_b_eligibility"
VERIFIER_MODULE = "rpe.runner.phase4_d4_protocol_b_eligibility_verifier"
CLI_MODULE = "tools.run_phase4_d4_protocol_b_eligibility"
REAL_CONFIG_PATH = (
    ROOT / "experiments/phase4/configs/d4_protocol_b_all_role_eligibility_v1.json"
)
STEP30_REPORT_PATH = (
    ROOT / "reports/phase4/step30_d4_protocol_b_all_role_eligibility_design.md"
)
PLAN_PATH = (
    ROOT / "docs/superpowers/plans/2026-08-25-phase4-d4-protocol-b-all-role-eligibility.md"
)
CANONICAL_ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
ACTIVE_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
EXPECTED_ARTIFACT_NAMES = {
    "config.json",
    "parent_bridge.json",
    "role_condition_summaries.jsonl",
    "model_condition_readiness.jsonl",
    "blank_condition_summaries.jsonl",
    "gate.json",
    "manifest.json",
    "complete.json",
    "SHA256SUMS",
}
FORBIDDEN_OUTCOME_KEYS = (
    "selected_n_components",
    "validation_scores",
    "prediction",
    "predicted_targets",
    "metric_value",
    "lod",
    "loq",
    "alignment",
    "bootstrap",
    "sign_flip",
    "holm",
    "table",
    "figure",
)
ALLOWED_PARENT_RECEIPT_KEYS = (
    "metrics",
    "result_sha256",
    "diagnostics_sha256",
    "peak_list_sha256",
)


def _load_subject():
    return importlib.import_module(SUBJECT_MODULE)


def _load_verifier():
    return importlib.import_module(VERIFIER_MODULE)


def _load_cli_module():
    return importlib.import_module(CLI_MODULE)


def _has_spec(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


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


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ids_digest(values: list[str] | tuple[str, ...]) -> str:
    return _sha_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _hex_alpha(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


CANONICAL_CONDITION_IDS = ("alpha0",) + tuple(
    f"{perturbation_id}:{_hex_alpha(alpha)}"
    for perturbation_id in ACTIVE_PERTURBATION_IDS
    for alpha in CANONICAL_ALPHA_GRID[1:]
)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _array_sha(value: np.ndarray, *, dtype: str) -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _native_axis_f32() -> np.ndarray:
    axis_f8 = np.empty(2000, dtype=np.float64)
    axis_f8[0] = 142.17398071289062
    axis_f8[1] = 145.83834838867188
    step = (3684.83544921875 - axis_f8[1]) / 1998.0
    for index in range(2, 2000):
        axis_f8[index] = axis_f8[1] + step * (index - 1)
    return _read_only(np.asarray(axis_f8, dtype="<f4"))


def _native_intensity_f32(axis: np.ndarray, *, scale: float) -> np.ndarray:
    axis_f8 = np.asarray(axis, dtype="<f8")
    intensity = (
        0.4
        + 0.002 * scale * np.sin(axis_f8 / 37.0)
        + 0.01 * np.exp(-0.5 * ((axis_f8 - (710.0 + 3.0 * scale)) / 18.0) ** 2)
        + 0.005 * np.exp(-0.5 * ((axis_f8 - (1220.0 - 2.0 * scale)) / 27.0) ** 2)
    )
    return _read_only(np.asarray(intensity, dtype="<f4"))


def _support_axis(axis: np.ndarray) -> np.ndarray:
    axis_f8 = np.asarray(axis, dtype="<f8")
    return np.asarray(axis_f8[axis_f8 >= (145.83834838867188 - 1e-12)], dtype="<f8")


def _synthetic_cohort() -> D4SugarCohort:
    axis = _native_axis_f32()
    wells = tuple(f"A{index}_1" for index in range(1, 6))
    record_ids = tuple(f"mix-{index:04d}" for index in range(5))
    source_members = tuple(f"synthetic/{record_id}.csv" for record_id in record_ids)
    intensity = _read_only(
        np.stack(
            [_native_intensity_f32(axis, scale=float(index)) for index in range(5)],
            axis=0,
        )
    )
    targets = _read_only(
        np.asarray(
            [
                [0.0, 0.08, 0.20, 0.32],
                [0.32, 0.20, 0.08, 0.0],
                [0.08, 0.0, 0.32, 0.20],
                [0.20, 0.32, 0.0, 0.08],
                [0.0, 0.20, 0.08, 0.32],
            ],
            dtype="<f8",
        )
    )
    folds = tuple(_read_only(np.asarray([index], dtype="<i8")) for index in range(5))
    splits = []
    for seed in range(5):
        test_fold = seed
        validation_fold = (seed + 1) % 5
        train_folds = tuple(
            fold for fold in range(5) if fold not in (test_fold, validation_fold)
        )
        splits.append(
            D4WellSplit(
                seed=seed,
                train_indices=_read_only(np.asarray(train_folds, dtype="<i8")),
                validation_indices=_read_only(np.asarray([validation_fold], dtype="<i8")),
                test_indices=_read_only(np.asarray([test_fold], dtype="<i8")),
                train_folds=train_folds,
                validation_fold=validation_fold,
                test_fold=test_fold,
            )
        )
    return D4SugarCohort(
        protocol_config_sha256="0" * 64,
        intensity=intensity,
        targets=targets,
        wavenumber=axis,
        target_names=("sucrose", "fructose", "maltose", "glucose"),
        record_ids=record_ids,
        well_ids=wells,
        source_members=source_members,
        rounds=_read_only(np.ones(5, dtype="<i8")),
        repetitions=_read_only(np.ones(5, dtype="<i8")),
        blank_intensity=_read_only(
            np.stack(
                [
                    _native_intensity_f32(axis, scale=99.0),
                    _native_intensity_f32(axis, scale=100.0),
                ],
                axis=0,
            )
        ),
        blank_targets=_read_only(np.zeros((2, 4), dtype="<f8")),
        blank_record_ids=("blank-0000", "blank-0001"),
        blank_well_ids=("E1_3", "E1_3"),
        blank_source_members=("synthetic/blank-0000.csv", "synthetic/blank-0001.csv"),
        folds=folds,
        splits=tuple(splits),
    )


def _role_ids_for_seed(cohort: D4SugarCohort, seed: int) -> dict[str, tuple[str, ...]]:
    split = cohort.splits[seed]
    return {
        "train": tuple(cohort.record_ids[index] for index in split.train_indices.tolist()),
        "validation": tuple(
            cohort.record_ids[index] for index in split.validation_indices.tolist()
        ),
        "test": tuple(cohort.record_ids[index] for index in split.test_indices.tolist()),
    }


def _well_ids_for_seed(cohort: D4SugarCohort, seed: int) -> dict[str, tuple[str, ...]]:
    split = cohort.splits[seed]
    return {
        "train": tuple(cohort.well_ids[index] for index in split.train_indices.tolist()),
        "validation": tuple(
            cohort.well_ids[index] for index in split.validation_indices.tolist()
        ),
        "test": tuple(cohort.well_ids[index] for index in split.test_indices.tolist()),
    }


def _synthetic_config_raw(*, condition_ids: tuple[str, ...]) -> bytes:
    cohort = _synthetic_cohort()
    axis = _native_axis_f32()
    support = _support_axis(axis)
    return _canonical(
        {
            "schema_version": "phase4-d4-protocol-b-all-role-eligibility-config-v1",
            "experiment_id": "phase4-d4-protocol-b-all-role-eligibility-v1",
            "artifact_schema": "phase4-d4-protocol-b-all-role-eligibility-artifact-v1",
            "marker_schema": "phase4-d4-protocol-b-all-role-eligibility-marker-v1",
            "run_prefix": "phase4-d4-protocol-b-all-role-eligibility-",
            "protocol": "B",
            "claim_boundary": "outcome_blind_protocol_b_all_role_eligibility_only",
            "synthetic_fixture": True,
            "active_perturbation_ids": list(ACTIVE_PERTURBATION_IDS),
            "alpha_grid": list(CANONICAL_ALPHA_GRID),
            "canonical_condition_ids": list(condition_ids),
            "denominators": {
                "mixture_record_count": 5,
                "mixture_well_count": 5,
                "blank_record_count": 2,
                "blank_well_count": 1,
                "model_cell_count": 5,
                "train_record_count": 3,
                "validation_record_count": 1,
                "test_record_count": 1,
                "train_well_count": 3,
                "validation_well_count": 1,
                "test_well_count": 1,
                "role_condition_summary_count": 615,
                "model_condition_readiness_count": 205,
                "blank_condition_summary_count": 41,
            },
            "frozen_identities": {
                "native_axis_f32_sha256": _array_sha(axis, dtype="<f4"),
                "support_axis_f64_sha256": _array_sha(support, dtype="<f8"),
                "mixture_record_ids_sha256": _ids_digest(list(cohort.record_ids)),
                "mixture_well_ids_sha256": _ids_digest(list(cohort.well_ids)),
                "blank_record_ids_sha256": _ids_digest(list(cohort.blank_record_ids)),
            },
            "step30_design_receipt": {
                "relative_path": STEP30_REPORT_PATH.relative_to(ROOT).as_posix(),
                "sha256": "4" * 64,
                "byte_count": 1,
            },
            "parent_step27": {
                "run_relative_path": (
                    "results/phase4/d4_protocol_a_full_domain_eligibility_v1/"
                    "phase4-d4-protocol-a-full-domain-eligibility-synthetic"
                ),
                "sha256sums_sha256": "8" * 64,
                "required_inventory": [
                    "config.json",
                    "source_records.jsonl",
                    "well_folds.jsonl",
                    "model_cells.jsonl",
                    "model_role_occurrences.jsonl",
                    "operator_cells.jsonl",
                    "blank_cells.jsonl",
                    "record_conditions.jsonl",
                    "blank_conditions.jsonl",
                    "well_summaries.jsonl",
                    "common_support.jsonl",
                    "gate.json",
                    "manifest.json",
                    "failed.json",
                    "SHA256SUMS",
                ],
            },
            "artifact_payload_files": [
                "config.json",
                "parent_bridge.json",
                "role_condition_summaries.jsonl",
                "model_condition_readiness.jsonl",
                "blank_condition_summaries.jsonl",
                "gate.json",
                "manifest.json",
            ],
            "forbidden_outcome_keys": list(FORBIDDEN_OUTCOME_KEYS),
            "authorities": {},
            "code_authority": {},
            "environment_authority": {},
            "trust_anchor": {
                "direction": "authority_to_config_only",
                "config_authority_relative_path": (
                    "rpe/runner/phase4_d4_protocol_b_eligibility_authority.py"
                ),
            },
        }
    )


def _synthetic_record_condition_rows(
    *,
    condition_ids: tuple[str, ...],
    incomplete_record_id: str | None = None,
    incomplete_condition_id: str | None = None,
) -> list[dict[str, object]]:
    rows = []
    for record_id in _synthetic_cohort().record_ids:
        for condition_id in condition_ids:
            state = (
                "failed"
                if (
                    record_id == incomplete_record_id
                    and condition_id == incomplete_condition_id
                )
                else "complete"
            )
            rows.append(
                {
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "state": state,
                    "metrics": {
                        f"metric_{index:02d}": {"state": state} for index in range(13)
                    },
                    "result_sha256": _sha_bytes(
                        f"result:{record_id}:{condition_id}".encode("utf-8")
                    ),
                    "diagnostics_sha256": _sha_bytes(
                        f"diagnostics:{record_id}:{condition_id}".encode("utf-8")
                    ),
                    "peak_list_sha256": _sha_bytes(
                        f"peak:{record_id}:{condition_id}".encode("utf-8")
                    ),
                }
            )
    return rows


def _synthetic_blank_condition_rows(
    *,
    condition_ids: tuple[str, ...],
    failed_condition_id: str | None = None
) -> list[dict[str, object]]:
    rows = []
    for record_id in _synthetic_cohort().blank_record_ids:
        for condition_id in condition_ids:
            state = (
                "failed"
                if failed_condition_id == condition_id and record_id == "blank-0000"
                else "complete"
            )
            rows.append(
                {
                    "record_id": record_id,
                    "well_id": "E1_3",
                    "condition_id": condition_id,
                    "state": state,
                    "result_sha256": _sha_bytes(
                        f"blank-result:{record_id}:{condition_id}".encode("utf-8")
                    ),
                    "diagnostics_sha256": _sha_bytes(
                        f"blank-diagnostics:{record_id}:{condition_id}".encode("utf-8")
                    ),
                    "warning_sha256": _sha_bytes(
                        f"blank-warning:{record_id}:{condition_id}".encode("utf-8")
                    ),
                }
            )
    return rows


def _parent_tree(
    *,
    condition_ids: tuple[str, ...],
    incomplete_record_id: str | None = None,
    incomplete_condition_id: str | None = None,
    blank_failed_condition_id: str | None = None,
    marker_filename: str = "failed.json",
    condition_drift: bool = False,
) -> dict[str, bytes]:
    cohort = _synthetic_cohort()
    source_records = [
        {
            "record_id": record_id,
            "well_id": well_id,
            "source_member": source_member,
            "state": "complete",
        }
        for record_id, well_id, source_member in zip(
            cohort.record_ids,
            cohort.well_ids,
            cohort.source_members,
            strict=True,
        )
    ]
    well_folds = []
    model_cells = []
    model_role_occurrences = []
    for seed in range(5):
        role_ids = _role_ids_for_seed(cohort, seed)
        role_wells = _well_ids_for_seed(cohort, seed)
        well_folds.append(
            {
                "seed": seed,
                "train_record_ids_sha256": _ids_digest(list(role_ids["train"])),
                "validation_record_ids_sha256": _ids_digest(list(role_ids["validation"])),
                "test_record_ids_sha256": _ids_digest(list(role_ids["test"])),
                "train_well_ids_sha256": _ids_digest(list(role_wells["train"])),
                "validation_well_ids_sha256": _ids_digest(list(role_wells["validation"])),
                "test_well_ids_sha256": _ids_digest(list(role_wells["test"])),
            }
        )
        model_cells.append(
            {
                "seed": seed,
                "train_record_count": 3,
                "validation_record_count": 1,
                "test_record_count": 1,
            }
        )
        for role in ("train", "validation", "test"):
            for record_id in role_ids[role]:
                model_role_occurrences.append(
                    {"seed": seed, "role": role, "record_id": record_id}
                )
    operator_cells = [
        {
            "record_id": record_id,
            "perturbation_id": perturbation_id,
            "state": "complete",
            "result_sha256": _sha_bytes(f"{record_id}:{perturbation_id}".encode("utf-8")),
        }
        for record_id in cohort.record_ids
        for perturbation_id in ACTIVE_PERTURBATION_IDS
    ]
    record_conditions = _synthetic_record_condition_rows(
        condition_ids=condition_ids,
        incomplete_record_id=incomplete_record_id,
        incomplete_condition_id=incomplete_condition_id,
    )
    if condition_drift:
        record_conditions[-1]["condition_id"] = "drifted-condition"
    blank_conditions = _synthetic_blank_condition_rows(
        condition_ids=condition_ids,
        failed_condition_id=blank_failed_condition_id
    )
    gate = {
        "full_domain_core": {
            "state": "evaluable",
            "p08_p12_complete_model_condition_count": 200,
        },
        "marker_filename": marker_filename,
    }
    manifest = {
        "claim_boundary": "outcome_blind_protocol_b_all_role_eligibility_only",
        "row_counts": {
            "source_records": 5,
            "well_folds": 5,
            "model_cells": 5,
            "model_role_occurrences": 25,
            "record_conditions": 205,
            "blank_conditions": 82,
        },
    }
    return {
        "config.json": _canonical({"synthetic_parent": True}),
        "source_records.jsonl": _jsonl_bytes(source_records),
        "well_folds.jsonl": _jsonl_bytes(well_folds),
        "model_cells.jsonl": _jsonl_bytes(model_cells),
        "model_role_occurrences.jsonl": _jsonl_bytes(model_role_occurrences),
        "operator_cells.jsonl": _jsonl_bytes(operator_cells),
        "blank_cells.jsonl": _jsonl_bytes(
            [{"record_id": record_id} for record_id in cohort.blank_record_ids]
        ),
        "record_conditions.jsonl": _jsonl_bytes(record_conditions),
        "blank_conditions.jsonl": _jsonl_bytes(blank_conditions),
        "well_summaries.jsonl": _jsonl_bytes(
            [{"well_id": well_id, "state": "complete"} for well_id in cohort.well_ids]
        ),
        "common_support.jsonl": _jsonl_bytes(
            [{"condition_id": condition_id, "state": "complete"} for condition_id in condition_ids]
        ),
        "gate.json": _canonical(gate),
        "manifest.json": _canonical(manifest),
        "failed.json": _canonical({"status": "failed-parent-marker"}),
    }


def _write_parent_run(path: Path, tree: dict[str, bytes]) -> None:
    for name, payload in tree.items():
        (path / name).write_bytes(payload)
    ordered = [
        "config.json",
        "source_records.jsonl",
        "well_folds.jsonl",
        "model_cells.jsonl",
        "model_role_occurrences.jsonl",
        "operator_cells.jsonl",
        "blank_cells.jsonl",
        "record_conditions.jsonl",
        "blank_conditions.jsonl",
        "well_summaries.jsonl",
        "common_support.jsonl",
        "gate.json",
        "manifest.json",
        "failed.json",
    ]
    (path / "SHA256SUMS").write_text(
        "".join(f"{_sha_bytes(tree[name])}  {name}\n" for name in ordered),
        encoding="utf-8",
    )


def _load_synthetic_config(subject, *, condition_ids: tuple[str, ...] | None = None):
    return subject.parse_phase4_d4_protocol_b_eligibility_config(
        Path("synthetic.json"),
        _synthetic_config_raw(
            condition_ids=tuple(subject.CONDITION_IDS) if condition_ids is None else condition_ids
        ),
        require_frozen_identity=False,
    )


def _build_synthetic_parent(path: Path, **kwargs):
    _write_parent_run(path, _parent_tree(**kwargs))


def _artifact_tree(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(_jsonl_bytes(rows))


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_bytes(_canonical(document))


def _refresh_sha256sums(path: Path) -> None:
    ordered = [
        "config.json",
        "source_records.jsonl",
        "well_folds.jsonl",
        "model_cells.jsonl",
        "model_role_occurrences.jsonl",
        "operator_cells.jsonl",
        "blank_cells.jsonl",
        "record_conditions.jsonl",
        "blank_conditions.jsonl",
        "well_summaries.jsonl",
        "common_support.jsonl",
        "gate.json",
        "manifest.json",
        "failed.json",
    ]
    if (path / "complete.json").exists():
        ordered = [
            "config.json",
            "parent_bridge.json",
            "role_condition_summaries.jsonl",
            "model_condition_readiness.jsonl",
            "blank_condition_summaries.jsonl",
            "gate.json",
            "manifest.json",
            "complete.json",
        ]
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{_sha_bytes((path / name).read_bytes())}  {name}\n"
            for name in ordered
        ),
        encoding="utf-8",
    )


class Phase4D4ProtocolBEligibilityRedCheckpointTest(unittest.TestCase):
    def test_red_checkpoint_missing_protocol_b_eligibility_module(self) -> None:
        _load_subject()


@unittest.skipUnless(
    _has_spec(SUBJECT_MODULE),
    "awaiting Step 31 D4 Protocol-B all-role eligibility implementation",
)
class Phase4D4ProtocolBEligibilityApiConfigTest(unittest.TestCase):
    def test_public_api_and_frozen_identifiers_match_step30(self) -> None:
        subject = _load_subject()
        plan = PLAN_PATH.read_text(encoding="utf-8")
        required_names = (
            "Phase4D4ProtocolBEligibilityError",
            "Phase4D4ProtocolBEligibilityConfig",
            "D4ProtocolBRoleInputs",
            "Phase4D4ProtocolBEligibilitySummary",
            "parse_phase4_d4_protocol_b_eligibility_config",
            "load_phase4_d4_protocol_b_eligibility_config",
            "reconstruct_d4_protocol_b_role_inputs",
            "validate_protocol_b_outcome_blind_payload",
            "validate_d4_protocol_b_parent_bridge",
            "build_d4_protocol_b_role_lift",
            "build_phase4_d4_protocol_b_eligibility_from_inputs",
            "build_phase4_d4_protocol_b_eligibility",
            "verify_phase4_d4_protocol_b_eligibility_from_inputs",
            "verify_phase4_d4_protocol_b_eligibility",
        )
        missing = [name for name in required_names if not hasattr(subject, name)]
        self.assertEqual(missing, [])
        self.assertEqual(tuple(subject.CONDITION_IDS), CANONICAL_CONDITION_IDS)
        for identifier in (
            "phase4-d4-protocol-b-all-role-eligibility-config-v1",
            "phase4-d4-protocol-b-all-role-eligibility-artifact-v1",
            "phase4-d4-protocol-b-all-role-eligibility-marker-v1",
            "phase4-d4-protocol-b-all-role-eligibility-",
        ):
            self.assertIn(identifier, plan)

    def test_synthetic_config_parses_and_preserves_615_205_41_contract(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        self.assertEqual(config.protocol, "B")
        self.assertEqual(tuple(config.active_perturbation_ids), ACTIVE_PERTURBATION_IDS)
        self.assertEqual(tuple(config.alpha_grid), CANONICAL_ALPHA_GRID)
        self.assertEqual(config.role_condition_summary_count, 615)
        self.assertEqual(config.model_condition_readiness_count, 205)
        self.assertEqual(config.blank_condition_summary_count, 41)

    def test_real_config_is_canonical_and_requires_structured_authority_receipts(self) -> None:
        subject = _load_subject()
        raw = REAL_CONFIG_PATH.read_bytes()
        drift = json.loads(raw)
        self.assertEqual(_canonical(drift), raw)
        self.assertEqual(tuple(drift["canonical_condition_ids"]), CANONICAL_CONDITION_IDS)
        for name, receipt in drift["authorities"].items():
            self.assertIsInstance(receipt, dict, name)
            self.assertIn("sha256", receipt, name)
            self.assertIn("bytes", receipt, name)
            self.assertTrue("path" in receipt or "member_path" in receipt, name)
        structured = dict(drift)
        structured["authorities"] = {
            name: {
                "path": f"synthetic/{name}.json",
                "bytes": 1,
                "sha256": "0" * 64,
            }
            for name in drift["authorities"]
        }
        first_key = next(iter(structured["authorities"]))
        structured["authorities"][first_key]["sha256"] = "1" * 64
        with self.assertRaisesRegex(
            subject.Phase4D4ProtocolBEligibilityError,
            "authorit|receipt|sha",
        ):
            subject.parse_phase4_d4_protocol_b_eligibility_config(
                REAL_CONFIG_PATH,
                _canonical(structured),
                require_frozen_identity=False,
            )


@unittest.skipUnless(
    _has_spec(SUBJECT_MODULE),
    "awaiting Step 31 D4 Protocol-B all-role eligibility implementation",
)
class Phase4D4ProtocolBRoleLiftTest(unittest.TestCase):
    def test_real_parent_static_ledgers_match_independent_reconstruction(self) -> None:
        subject = _load_subject()
        config = subject.load_phase4_d4_protocol_b_eligibility_config(
            REAL_CONFIG_PATH
        )
        cohort = load_d4_sugar_cohort(
            ROOT / "experiments/phase05/configs/d4_sugar_protocol.json",
            ROOT / "data/raw/ramanbench/cache/10779223/Raw data.zip",
        )
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(cohort, config)
        bridge = subject.validate_d4_protocol_b_parent_bridge(
            parent_run_path=ROOT / config.parent_run_relative_path,
            inputs=inputs,
            config=config,
            validate_condition_order=False,
        )
        self.assertEqual(bridge["parent_marker_filename"], "failed.json")
        self.assertEqual(bridge["source_record_count"], 7680)
        self.assertEqual(bridge["model_cell_count"], 5)
        self.assertEqual(bridge["model_role_occurrence_count"], 38400)
        self.assertEqual(bridge["blank_role_occurrence_count"], 160)

    def test_reconstructs_disjoint_five_fold_roles_and_exact_multiplicity(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        self.assertEqual(len(inputs.source_records), 5)
        self.assertEqual(len(inputs.model_cells), 5)
        self.assertEqual(len(inputs.model_role_occurrences), 25)
        counts = Counter((row["record_id"], row["role"]) for row in inputs.model_role_occurrences)
        for record_id in _synthetic_cohort().record_ids:
            self.assertEqual(counts[(record_id, "train")], 3)
            self.assertEqual(counts[(record_id, "validation")], 1)
            self.assertEqual(counts[(record_id, "test")], 1)

    def test_role_lift_emits_615_205_41_and_preserves_canonical_order(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            _build_synthetic_parent(parent, condition_ids=tuple(subject.CONDITION_IDS))
            lift = subject.build_d4_protocol_b_role_lift(
                parent_run_path=parent,
                inputs=inputs,
                config=config,
            )
        role_summaries = lift["role_condition_summaries"]
        model_readiness = lift["model_condition_readiness"]
        blank_summaries = lift["blank_condition_summaries"]
        self.assertEqual(len(role_summaries), 5 * 3 * 41)
        self.assertEqual(len(model_readiness), 5 * 41)
        self.assertEqual(len(blank_summaries), 41)
        self.assertEqual(
            [row["condition_id"] for row in role_summaries[:41]],
            list(CANONICAL_CONDITION_IDS),
        )

    def test_one_incomplete_role_condition_fails_only_its_model_and_primary_gate(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        target_condition = tuple(subject.CONDITION_IDS)[1]
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            _build_synthetic_parent(
                parent,
                condition_ids=tuple(subject.CONDITION_IDS),
                incomplete_record_id="mix-0000",
                incomplete_condition_id=target_condition,
            )
            lift = subject.build_d4_protocol_b_role_lift(
                parent_run_path=parent,
                inputs=inputs,
                config=config,
            )
        impacted_seeds = {
            row["seed"]
            for row in lift["model_condition_readiness"]
            if row["condition_id"] == target_condition and row["state"] != "ready"
        }
        self.assertEqual(impacted_seeds, {0, 1, 2, 3, 4})
        self.assertEqual(lift["gate"]["full_domain_core"]["state"], "not_evaluable_coverage")
        self.assertEqual(lift["gate"]["marker_filename"], "failed.json")
        self.assertEqual(len(lift["model_condition_readiness"]), 205)

    def test_blank_failure_is_auxiliary_and_does_not_close_primary_gate(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            _build_synthetic_parent(
                parent,
                condition_ids=tuple(subject.CONDITION_IDS),
                blank_failed_condition_id="alpha0",
            )
            lift = subject.build_d4_protocol_b_role_lift(
                parent_run_path=parent,
                inputs=inputs,
                config=config,
            )
        failed_blank = [
            row for row in lift["blank_condition_summaries"] if row["state"] != "complete"
        ]
        self.assertEqual(len(failed_blank), 1)
        self.assertEqual(failed_blank[0]["condition_id"], "alpha0")
        self.assertEqual(lift["gate"]["full_domain_core"]["state"], "evaluable")
        self.assertEqual(len(lift["model_condition_readiness"]), 205)

    def test_real_schema_blank_condition_receipt_completeness(self) -> None:
        subject = _load_subject()
        row = {
            "record_id": "blank-0000",
            "well_id": "E1_3",
            "condition_id": "alpha0",
            "state": "complete",
            "result_sha256": "1" * 64,
            "diagnostics_sha256": "2" * 64,
            "warning_sha256": "3" * 64,
        }
        self.assertTrue(subject._blank_condition_row_complete(row))
        self.assertFalse(
            subject._blank_condition_row_complete(
                {key: value for key, value in row.items() if key != "warning_sha256"}
            )
        )
        self.assertFalse(
            subject._blank_condition_row_complete({**row, "warning_sha256": "bad"})
        )

    def test_parent_bridge_rejects_inventory_checksum_marker_and_condition_drift(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            extra = root / "extra"
            extra.mkdir()
            _build_synthetic_parent(extra, condition_ids=tuple(subject.CONDITION_IDS))
            (extra / "unexpected.txt").write_text("forbidden\n", encoding="utf-8")
            with self.assertRaisesRegex(
                subject.Phase4D4ProtocolBEligibilityError,
                "inventory|unexpected",
            ):
                subject.validate_d4_protocol_b_parent_bridge(
                    parent_run_path=extra,
                    inputs=inputs,
                    config=config,
                )

            checksum = root / "checksum"
            checksum.mkdir()
            _build_synthetic_parent(checksum, condition_ids=tuple(subject.CONDITION_IDS))
            (checksum / "record_conditions.jsonl").write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(
                subject.Phase4D4ProtocolBEligibilityError,
                "SHA256|checksum",
            ):
                subject.validate_d4_protocol_b_parent_bridge(
                    parent_run_path=checksum,
                    inputs=inputs,
                    config=config,
                )

            marker = root / "marker"
            marker.mkdir()
            _build_synthetic_parent(
                marker,
                condition_ids=tuple(subject.CONDITION_IDS),
                marker_filename="complete.json",
            )
            with self.assertRaisesRegex(
                subject.Phase4D4ProtocolBEligibilityError,
                "failed.json|marker|evaluable",
            ):
                subject.validate_d4_protocol_b_parent_bridge(
                    parent_run_path=marker,
                    inputs=inputs,
                    config=config,
                )

            drift = root / "condition"
            drift.mkdir()
            _build_synthetic_parent(
                drift,
                condition_ids=tuple(subject.CONDITION_IDS),
                condition_drift=True,
            )
            with self.assertRaisesRegex(
                subject.Phase4D4ProtocolBEligibilityError,
                "condition|canonical|drift",
            ):
                subject.validate_d4_protocol_b_parent_bridge(
                    parent_run_path=drift,
                    inputs=inputs,
                    config=config,
                )

    def test_parent_bridge_rejects_source_model_and_role_semantic_drift(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("source_records.jsonl", "source semantics", {"well_id": "Z9_9"}),
                ("model_cells.jsonl", "model cell semantics", {"train_record_count": 2}),
                ("model_role_occurrences.jsonl", "role semantics", {"role": "train", "seed": 99}),
            )
            for filename, reason, updates in cases:
                with self.subTest(filename=filename):
                    parent = root / filename.replace(".", "_")
                    parent.mkdir()
                    _build_synthetic_parent(parent, condition_ids=tuple(subject.CONDITION_IDS))
                    rows = _read_jsonl_rows(parent / filename)
                    rows[0] = {**rows[0], **updates}
                    _write_jsonl_rows(parent / filename, rows)
                    _refresh_sha256sums(parent)
                    with self.assertRaisesRegex(
                        subject.Phase4D4ProtocolBEligibilityError,
                        "source|model|role|semantic|mismatch",
                    ):
                        subject.validate_d4_protocol_b_parent_bridge(
                            parent_run_path=parent,
                            inputs=inputs,
                            config=config,
                        )

    def test_build_role_lift_reads_record_conditions_once_without_parent_prescan(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        counts: Counter[str] = Counter()
        original = subject._read_jsonl

        def counting_read_jsonl(path: Path):
            counts[Path(path).name] += 1
            yield from original(path)

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            _build_synthetic_parent(parent, condition_ids=tuple(subject.CONDITION_IDS))
            subject._read_jsonl = counting_read_jsonl
            try:
                subject.build_d4_protocol_b_role_lift(
                    parent_run_path=parent,
                    inputs=inputs,
                    config=config,
                )
            finally:
                subject._read_jsonl = original
        self.assertEqual(counts["record_conditions.jsonl"], 1)
        self.assertEqual(counts["blank_conditions.jsonl"], 1)

    def test_parent_rows_with_forbidden_outcome_fields_are_rejected(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, key in (
                ("record_conditions.jsonl", "predicted_targets"),
                ("blank_conditions.jsonl", "metric_value"),
            ):
                with self.subTest(filename=filename):
                    parent = root / filename.replace(".", "_")
                    parent.mkdir()
                    _build_synthetic_parent(parent, condition_ids=tuple(subject.CONDITION_IDS))
                    rows = _read_jsonl_rows(parent / filename)
                    rows[0][key] = [0.1] if key == "predicted_targets" else 1.0
                    _write_jsonl_rows(parent / filename, rows)
                    _refresh_sha256sums(parent)
                    with self.assertRaisesRegex(
                        subject.Phase4D4ProtocolBEligibilityError,
                        "forbidden|outcome|prediction|metric_value",
                    ):
                        subject.build_d4_protocol_b_role_lift(
                            parent_run_path=parent,
                            inputs=inputs,
                            config=config,
                        )

    def test_recursive_firewall_rejects_step29_and_outcome_fields(self) -> None:
        subject = _load_subject()
        valid = {
            "parent_bridge": {
                "metrics": {"metric_00": {"state": "complete"}},
                "result_sha256": "1" * 64,
                "diagnostics_sha256": "2" * 64,
                "peak_list_sha256": "3" * 64,
            }
        }
        subject.validate_protocol_b_outcome_blind_payload(valid)
        for key in ALLOWED_PARENT_RECEIPT_KEYS:
            self.assertIn(key, valid["parent_bridge"])
        subject.validate_protocol_b_outcome_blind_payload(
            {
                "authority": {
                    "path": "reports/phase4/step29_d4_protocol_a_full_domain.md",
                    "bytes": 16560,
                    "sha256": "a" * 64,
                }
            }
        )
        with self.assertRaisesRegex(
            subject.Phase4D4ProtocolBEligibilityError,
            "step29|protocol_a_full_domain",
        ):
            subject.validate_protocol_b_outcome_blind_payload(
                {
                    "path": (
                        "results/phase4/d4_protocol_a_full_domain_v1/"
                        "phase4-d4-protocol-a-full-domain-deadbeef"
                    )
                }
            )
        with self.assertRaisesRegex(
            subject.Phase4D4ProtocolBEligibilityError,
            "selected_n_components|prediction|table",
        ):
            subject.validate_protocol_b_outcome_blind_payload(
                {
                    "selected_n_components": 3,
                    "nested": {"prediction": [0.1], "table": "forbidden"},
                }
            )


@unittest.skipUnless(
    _has_spec(SUBJECT_MODULE) and _has_spec(VERIFIER_MODULE) and _has_spec(CLI_MODULE),
    "awaiting Step 31 D4 Protocol-B build, verifier, and CLI implementation",
)
class Phase4D4ProtocolBArtifactVerifierCliTest(unittest.TestCase):
    def test_build_is_nine_file_content_addressed_append_only_and_manifest_is_identity_rich(self) -> None:
        subject = _load_subject()
        config = _load_synthetic_config(subject)
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(_synthetic_cohort(), config)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            output = Path(temporary) / "artifact"
            parent.mkdir()
            _build_synthetic_parent(parent, condition_ids=tuple(subject.CONDITION_IDS))
            summary = subject.build_phase4_d4_protocol_b_eligibility_from_inputs(
                output,
                parent_run_path=parent,
                inputs=inputs,
                config=config,
            )
            artifact_names = {item.name for item in summary.path.iterdir() if item.is_file()}
            self.assertEqual(
                artifact_names,
                {
                    "config.json",
                    "parent_bridge.json",
                    "role_condition_summaries.jsonl",
                    "model_condition_readiness.jsonl",
                    "blank_condition_summaries.jsonl",
                    "gate.json",
                    "manifest.json",
                    "complete.json",
                    "SHA256SUMS",
                },
            )
            self.assertEqual(summary.status, "complete")
            self.assertEqual(summary.role_condition_summary_count, 615)
            self.assertEqual(summary.model_condition_readiness_count, 205)
            manifest = json.loads((summary.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["numerical_execution_count"], 0)
            self.assertIn("authority_identities", manifest)
            self.assertIn("code_authority", manifest)
            self.assertIn("environment_authority", manifest)
            self.assertIn("inherited_rulings", manifest)
            self.assertIn("reconstructed_digests", manifest)
            self.assertIn("source_record_ids_sha256", manifest["reconstructed_digests"])
            self.assertIn("source_well_ids_sha256", manifest["reconstructed_digests"])
            self.assertIn("model_cells_sha256", manifest["reconstructed_digests"])
            self.assertIn("model_role_occurrences_sha256", manifest["reconstructed_digests"])
            with self.assertRaisesRegex(
                subject.Phase4D4ProtocolBEligibilityError,
                "exists|append",
            ):
                subject.build_phase4_d4_protocol_b_eligibility_from_inputs(
                    output,
                    parent_run_path=parent,
                    inputs=inputs,
                    config=config,
                )

    def test_independent_verifier_rejects_semantic_artifact_and_run_id_drift_and_has_no_production_import(self) -> None:
        subject = _load_subject()
        verifier = _load_verifier()
        tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
        forbidden = SUBJECT_MODULE
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                self.assertFalse(
                    node.module == forbidden or node.module.startswith(f"{forbidden}.")
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name == forbidden or alias.name.startswith(f"{forbidden}.")
                    )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "exec")

        config = _load_synthetic_config(subject)
        cohort = _synthetic_cohort()
        inputs = subject.reconstruct_d4_protocol_b_role_inputs(cohort, config)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary) / "parent"
            output = Path(temporary) / "artifact"
            parent.mkdir()
            _build_synthetic_parent(parent, condition_ids=tuple(subject.CONDITION_IDS))
            built = subject.build_phase4_d4_protocol_b_eligibility_from_inputs(
                output,
                parent_run_path=parent,
                inputs=inputs,
                config=config,
            )
            verified = verifier.verify_phase4_d4_protocol_b_eligibility_from_inputs(
                built.path,
                parent_run_path=parent,
                cohort=cohort,
                config_path=built.path / "config.json",
            )
            self.assertEqual(verified.path, built.path)
            self.assertEqual(verified.run_id, built.run_id)
            self.assertEqual(verified.status, built.status)
            self.assertEqual(_artifact_tree(verified.path), _artifact_tree(built.path))
            mutated = Path(temporary) / "artifact-mutated"
            mutated.mkdir()
            for name, payload in _artifact_tree(built.path).items():
                (mutated / name).write_bytes(payload)
            role_rows = _read_jsonl_rows(mutated / "role_condition_summaries.jsonl")
            role_rows[0]["state"] = "not_evaluable_coverage"
            _write_jsonl_rows(mutated / "role_condition_summaries.jsonl", role_rows)
            _refresh_sha256sums(mutated)
            with self.assertRaisesRegex(
                verifier.Phase4D4ProtocolBEligibilityVerifierError,
                "byte|semantic|mismatch|role_condition",
            ):
                verifier.verify_phase4_d4_protocol_b_eligibility_from_inputs(
                    mutated,
                    parent_run_path=parent,
                    cohort=cohort,
                    config_path=mutated / "config.json",
                )

            manifest_drift = Path(temporary) / "artifact-manifest"
            manifest_drift.mkdir()
            for name, payload in _artifact_tree(built.path).items():
                (manifest_drift / name).write_bytes(payload)
            manifest = json.loads((manifest_drift / "manifest.json").read_text(encoding="utf-8"))
            manifest["run_id"] = subject.RUN_PREFIX + ("f" * 64)
            _write_json(manifest_drift / "manifest.json", manifest)
            _refresh_sha256sums(manifest_drift)
            with self.assertRaisesRegex(
                verifier.Phase4D4ProtocolBEligibilityVerifierError,
                "run_id|rebuild|mismatch",
            ):
                verifier.verify_phase4_d4_protocol_b_eligibility_from_inputs(
                    manifest_drift,
                    parent_run_path=parent,
                    cohort=cohort,
                    config_path=manifest_drift / "config.json",
                )

    def test_cli_rejects_worker_skip_protocol_model_and_outcome_overrides(self) -> None:
        cli = _load_cli_module()
        with self.assertRaises(SystemExit):
            cli.main(["verify", "--run-path", "x", "--no-reexecute"])
        with self.assertRaises(SystemExit):
            cli.main(["build", "--output-root", "x", "--worker-count", "2"])
        with self.assertRaises(SystemExit):
            cli.main(["build", "--output-root", "x", "--skip-parent"])
        with self.assertRaises(SystemExit):
            cli.main(["build", "--output-root", "x", "--protocol", "A"])
        with self.assertRaises(SystemExit):
            cli.main(["build", "--output-root", "x", "--model", "pls2"])
        with self.assertRaises(SystemExit):
            cli.main(["build", "--output-root", "x", "--outcome"])


if __name__ == "__main__":
    unittest.main()
