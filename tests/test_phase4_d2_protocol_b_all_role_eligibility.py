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
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_config import load_phase1_core_config  # noqa: E402
from rpe.runner.phase4_d2_protocol_b_eligibility import (  # noqa: E402
    ACTIVE_PERTURBATION_IDS,
    ARTIFACT_PAYLOAD_FILES,
    DATASET_RELATIVE_PATH,
    Phase4D2ProtocolBEligibilityError,
    SELECTION_RELATIVE_PATH,
    build_phase4_d2_protocol_b_eligibility_from_inputs,
    evaluate_d2_protocol_b_all_role_gates,
    load_phase4_d2_protocol_b_eligibility_config,
    parse_phase4_d2_protocol_b_eligibility_config,
    reconstruct_d2_protocol_b_inputs,
)
from rpe.runner.phase4_d2_protocol_b_eligibility_verifier import (  # noqa: E402
    verify_phase4_d2_protocol_b_eligibility_from_inputs,
)
from tools.run_phase4_d2_protocol_b_eligibility import main as protocol_b_cli_main  # noqa: E402


SWEEP_PATH = ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_PATH = ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
REAL_CONFIG_PATH = (
    ROOT / "experiments/phase4/configs/d2_protocol_b_all_role_eligibility_v1.json"
)
SHOT_COUNTS = (5, 10, 20)
MODEL_SEEDS = (0, 1)
CLASS_LABELS = (0, 1)
TEST_RECORDS_PER_CLASS = 1
VALIDATION_RECORDS_PER_CLASS = 1
TRAIN_RECORDS_PER_CLASS = 20
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
INHERITED_STRUCTURAL_REASON = "structurally_ineligible_missing_explicit_baseline"
REAL_SHOT_UNION_DIGESTS = {
    "5": "f2d536c112410fca390dcfe82aebe92f36354688e73f023ef624b20b20206a50",
    "10": "e8f0908ad5e20a5136f5b69af420ac7c13191c3b21ca391c1ba6d8871a89b382",
    "20": "b2b886e4d83a459e235a8b96a73e311bbacdb7d333c82dca6aeed8feb65ff6a1",
}
REAL_P01_P04_BOUNDS = {
    "5": {
        "class_upper_bound_complete": 0,
        "record_upper_bound_complete": 1898,
        "required_class_count": 30,
        "required_record_count": 4456,
    },
    "10": {
        "class_upper_bound_complete": 0,
        "record_upper_bound_complete": 2237,
        "required_class_count": 30,
        "required_record_count": 4778,
    },
    "20": {
        "class_upper_bound_complete": 0,
        "record_upper_bound_complete": 2721,
        "required_class_count": 30,
        "required_record_count": 5238,
    },
}
REAL_P05_COMMON_BOUNDS = {
    "5": {
        "class_upper_bound_complete": 0,
        "record_upper_bound_complete": 1692,
        "required_class_count": 30,
        "required_record_count": 4221,
    },
    "10": {
        "class_upper_bound_complete": 0,
        "record_upper_bound_complete": 2031,
        "required_class_count": 30,
        "required_record_count": 4527,
    },
    "20": {
        "class_upper_bound_complete": 0,
        "record_upper_bound_complete": 2515,
        "required_class_count": 30,
        "required_record_count": 4962,
    },
}


def _expected_inherited_rulings() -> dict[str, object]:
    return {
        "p01_p04_full_domain_core": {
            "by_shot": REAL_P01_P04_BOUNDS,
            "state": "not_evaluable_coverage",
        },
        "p05_full_domain_core": {
            "by_shot": REAL_P05_COMMON_BOUNDS,
            "state": "not_evaluable_coverage",
        },
        "p06": {
            "reason": INHERITED_STRUCTURAL_REASON,
            "state": INHERITED_STRUCTURAL_REASON,
        },
        "p07": {
            "reason": INHERITED_STRUCTURAL_REASON,
            "state": INHERITED_STRUCTURAL_REASON,
        },
    }


def _synthetic_inherited_rulings() -> dict[str, object]:
    return {
        "p01_p04_full_domain_core": {
            "by_shot": {
                "5": {
                    "class_upper_bound_complete": 0,
                    "record_upper_bound_complete": 10,
                    "required_class_count": len(CLASS_LABELS),
                    "required_record_count": 25,
                },
                "10": {
                    "class_upper_bound_complete": 0,
                    "record_upper_bound_complete": 20,
                    "required_class_count": len(CLASS_LABELS),
                    "required_record_count": 44,
                },
                "20": {
                    "class_upper_bound_complete": 0,
                    "record_upper_bound_complete": 30,
                    "required_class_count": len(CLASS_LABELS),
                    "required_record_count": 82,
                },
            },
            "state": "not_evaluable_coverage",
        },
        "p05_full_domain_core": {
            "by_shot": {
                "5": {
                    "class_upper_bound_complete": 0,
                    "record_upper_bound_complete": 8,
                    "required_class_count": len(CLASS_LABELS),
                    "required_record_count": 24,
                },
                "10": {
                    "class_upper_bound_complete": 0,
                    "record_upper_bound_complete": 18,
                    "required_class_count": len(CLASS_LABELS),
                    "required_record_count": 42,
                },
                "20": {
                    "class_upper_bound_complete": 0,
                    "record_upper_bound_complete": 28,
                    "required_class_count": len(CLASS_LABELS),
                    "required_record_count": 78,
                },
            },
            "state": "not_evaluable_coverage",
        },
        "p06": {
            "reason": INHERITED_STRUCTURAL_REASON,
            "state": INHERITED_STRUCTURAL_REASON,
        },
        "p07": {
            "reason": INHERITED_STRUCTURAL_REASON,
            "state": INHERITED_STRUCTURAL_REASON,
        },
    }


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


def _array_sha(value: np.ndarray, *, dtype: str) -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _hex_alpha(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def _decreasing_axis() -> np.ndarray:
    axis = np.linspace(1800.0, 300.0, 1000, dtype="<f4")
    axis.setflags(write=False)
    return axis


def _support_grid() -> np.ndarray:
    return _decreasing_axis()[::-1].astype("<f8")[3:]


def _test_record_id(class_label: int, replicate: int) -> str:
    return f"test-c{class_label}-r{replicate}"


def _train_record_id(class_label: int, seed: int, index: int) -> str:
    return f"finetune-c{class_label}-seed{seed}-train-{index:02d}"


def _validation_record_id(class_label: int, seed: int, index: int) -> str:
    return f"finetune-c{class_label}-seed{seed}-val-{index:02d}"


def _ordered_train_ids(class_label: int, seed: int) -> list[str]:
    return [
        _train_record_id(class_label, seed, index)
        for index in range(TRAIN_RECORDS_PER_CLASS)
    ]


def _validation_ids(class_label: int, seed: int) -> list[str]:
    return [
        _validation_record_id(class_label, seed, index)
        for index in range(VALIDATION_RECORDS_PER_CLASS)
    ]


def _synthetic_selection_document() -> dict[str, object]:
    selections = []
    test_ids = [
        _test_record_id(class_label, replicate)
        for class_label in CLASS_LABELS
        for replicate in range(TEST_RECORDS_PER_CLASS)
    ]
    for seed in MODEL_SEEDS:
        class_documents = []
        train_ids_by_shot = {str(shot): [] for shot in SHOT_COUNTS}
        validation_ids = []
        for class_label in CLASS_LABELS:
            ordered = _ordered_train_ids(class_label, seed)
            validation = _validation_ids(class_label, seed)
            class_documents.append(
                {
                    "candidate_count": TRAIN_RECORDS_PER_CLASS + VALIDATION_RECORDS_PER_CLASS,
                    "class_label": class_label,
                    "ordered_train_record_ids": ordered,
                    "source_split": "finetune",
                    "train_record_ids": {
                        str(shot): ordered[:shot] for shot in SHOT_COUNTS
                    },
                    "train_record_ids_sha256": {
                        str(shot): _ids_digest(ordered[:shot]) for shot in SHOT_COUNTS
                    },
                    "validation_record_ids": validation,
                    "validation_record_ids_sha256": _ids_digest(validation),
                }
            )
            for shot in SHOT_COUNTS:
                train_ids_by_shot[str(shot)].extend(ordered[:shot])
            validation_ids.extend(validation)
        selections.append(
            {
                "classes": class_documents,
                "seed": seed,
                "train_record_ids_sha256": {
                    str(shot): _ids_digest(sorted(train_ids_by_shot[str(shot)]))
                    for shot in SHOT_COUNTS
                },
                "validation_record_ids_sha256": _ids_digest(sorted(validation_ids)),
            }
        )
    return {
        "code": {},
        "conditions": ["released_input_control"],
        "config": {
            "bytes": 858,
            "path": "d2_few_shot_selection.json",
            "sha256": "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138",
        },
        "dataset": {
            "dataset_id": "bacteria_id_reference",
            "files": {},
            "record_count": 86,
            "split_counts": {"finetune": 84, "reference": 0, "test": 2},
        },
        "experiment_id": "d2_bacteria_id_few_shot",
        "schema_version": "phase05-d2-selection-artifact-v1",
        "seeds": list(MODEL_SEEDS),
        "selections": selections,
        "shot_counts": list(SHOT_COUNTS),
        "status": "frozen",
        "test": {
            "count": len(test_ids),
            "record_ids_sha256": _ids_digest(test_ids),
            "source_split": "test",
        },
        "validation_per_class": VALIDATION_RECORDS_PER_CLASS,
    }


def _record_intensity(class_label: int, variant: int, axis_descending: np.ndarray) -> list[float]:
    coordinate = axis_descending.astype("<f8")
    left_peak = 675.0 + 11.0 * class_label + 0.5 * (variant % 9)
    right_peak = 1230.0 + 7.0 * class_label - 0.4 * (variant % 11)
    baseline = (
        0.18
        + 0.03 * class_label
        + 0.0005 * variant
        + 0.05 * np.sin(coordinate / (37.0 + 3.0 * class_label))
        + 0.62 * np.exp(-((coordinate - left_peak) / 22.0) ** 2)
        + 0.41 * np.exp(-((coordinate - right_peak) / 35.0) ** 2)
    )
    return np.asarray(baseline, dtype="<f4").tolist()


def _synthetic_dataset_document() -> dict[str, object]:
    axis = _decreasing_axis()
    records = []
    source_row = 0
    for class_label in CLASS_LABELS:
        for replicate in range(TEST_RECORDS_PER_CLASS):
            records.append(
                {
                    "class_label": class_label,
                    "record_id": _test_record_id(class_label, replicate),
                    "source_row": source_row,
                    "source_split": "test",
                    "stored_intensity": _record_intensity(class_label, source_row, axis),
                }
            )
            source_row += 1
    for seed in MODEL_SEEDS:
        for class_label in CLASS_LABELS:
            for index in range(VALIDATION_RECORDS_PER_CLASS):
                records.append(
                    {
                        "class_label": class_label,
                        "record_id": _validation_record_id(class_label, seed, index),
                        "source_row": source_row,
                        "source_split": "finetune",
                        "stored_intensity": _record_intensity(class_label, source_row, axis),
                    }
                )
                source_row += 1
            for index in range(TRAIN_RECORDS_PER_CLASS):
                records.append(
                    {
                        "class_label": class_label,
                        "record_id": _train_record_id(class_label, seed, index),
                        "source_row": source_row,
                        "source_split": "finetune",
                        "stored_intensity": _record_intensity(class_label, source_row, axis),
                    }
                )
                source_row += 1
    return {
        "dataset_id": "bacteria_id_reference",
        "native_axis_cm1_decreasing_f32": axis.tolist(),
        "records": records,
    }


def _source_union_ids(
    selection_document: dict[str, object],
    *,
    shot_count: int,
) -> list[str]:
    identifiers = [
        _test_record_id(class_label, replicate)
        for class_label in CLASS_LABELS
        for replicate in range(TEST_RECORDS_PER_CLASS)
    ]
    for seed_document in selection_document["selections"]:
        for class_document in seed_document["classes"]:
            identifiers.extend(class_document["validation_record_ids"])
            identifiers.extend(class_document["train_record_ids"][str(shot_count)])
    return sorted(set(str(value) for value in identifiers))


def _source_membership_map(selection_document: dict[str, object]) -> dict[str, list[int]]:
    membership = {}
    for shot_count in SHOT_COUNTS:
        for record_id in _source_union_ids(selection_document, shot_count=shot_count):
            membership.setdefault(record_id, []).append(shot_count)
    return membership


def _synthetic_config_document() -> dict[str, object]:
    selection = _synthetic_selection_document()
    dataset = _synthetic_dataset_document()
    membership = _source_membership_map(selection)
    support_axis = _support_grid()
    axis_decreasing = _decreasing_axis()
    axis_increasing = axis_decreasing[::-1].astype("<f8")
    shot_union_counts = {
        str(shot): len(_source_union_ids(selection, shot_count=shot))
        for shot in SHOT_COUNTS
    }
    role_occurrence_counts = {
        "5": 28,
        "10": 48,
        "20": 88,
    }
    all_record_ids = sorted(str(row["record_id"]) for row in dataset["records"])
    return {
        "active_perturbation_ids": list(ACTIVE_PERTURBATION_IDS),
        "alpha_grid": list(ALPHA_GRID),
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "authorities": {},
        "claim_boundary": "outcome_blind_protocol_b_all_role_eligibility_only",
        "code_authority": {},
        "denominators": {
            "class_count": len(CLASS_LABELS),
            "model_cell_count": len(MODEL_SEEDS) * len(SHOT_COUNTS),
            "model_role_occurrence_count": 164,
            "role_condition_audit_count": 164 * 41,
            "shot_role_occurrence_counts": role_occurrence_counts,
            "shot_source_union_counts": shot_union_counts,
            "source_record_count": len(dataset["records"]),
        },
        "environment_authority": {},
        "expected": {
            "apply_check_count": len(dataset["records"])
            * len(ACTIVE_PERTURBATION_IDS)
            * len(ALPHA_GRID),
            "canonical_record_condition_count": len(dataset["records"])
            * (1 + len(ACTIVE_PERTURBATION_IDS) * len(POSITIVE_ALPHAS)),
            "class_summary_count": len(SHOT_COUNTS)
            * len(ACTIVE_PERTURBATION_IDS)
            * len(CLASS_LABELS),
            "operator_cell_count": len(dataset["records"]) * len(ACTIVE_PERTURBATION_IDS),
            "positive_record_condition_count": len(dataset["records"])
            * len(ACTIVE_PERTURBATION_IDS)
            * len(POSITIVE_ALPHAS),
        },
        "experiment_id": "phase4-d2-protocol-b-all-role-eligibility-v1",
        "frozen_identities": {
            "model_seed_ids": list(MODEL_SEEDS),
            "native_axis_decreasing_f32_sha256": _array_sha(axis_decreasing, dtype="<f4"),
            "native_axis_increasing_f64_sha256": _array_sha(axis_increasing, dtype="<f8"),
            "shot_source_union_record_ids_sha256": {
                str(shot): _ids_digest(_source_union_ids(selection, shot_count=shot))
                for shot in SHOT_COUNTS
            },
            "source_record_ids_sha256": _ids_digest(all_record_ids),
            "support_axis_f32_sha256": _array_sha(
                support_axis.astype("<f4"), dtype="<f4"
            ),
            "support_axis_f64_sha256": _array_sha(support_axis, dtype="<f8"),
        },
        "inherited_rulings": {
            **_synthetic_inherited_rulings(),
        },
        "p10": {
            "correlation_length_cm1": 20.0,
            "memory_budget_bytes": 64 * 2**30,
            "peak_estimate_formula": "32*N^2+64*N+2^30",
        },
        "phase1_native_gate_relative_tolerance": 1e-12,
        "protocol": "B",
        "schema_version": "phase4-d2-protocol-b-all-role-eligibility-config-v1",
        "shot_counts": list(SHOT_COUNTS),
        "support_grid": {
            "coordinates_cm1": support_axis.tolist(),
            "max_in_range_native_gap_cm1": 1.561,
            "point_count": int(support_axis.size),
        },
        "synthetic_fixture": True,
        "trust_anchor": {},
    }


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _prepare_fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset_path = root / "dataset.json"
    selection_path = root / "selection.json"
    config_path = root / "config.json"
    _write_json(dataset_path, _synthetic_dataset_document())
    _write_json(selection_path, _synthetic_selection_document())
    _write_json(config_path, _synthetic_config_document())
    return dataset_path, selection_path, config_path


def _fixture_inputs(root: Path):
    dataset_path, selection_path, config_path = _prepare_fixture(root)
    config = parse_phase4_d2_protocol_b_eligibility_config(
        config_path,
        config_path.read_bytes(),
        require_frozen_identity=False,
    )
    inputs = reconstruct_d2_protocol_b_inputs(dataset_path, selection_path, config)
    sweep = load_perturbation_sweep_config(SWEEP_PATH)
    phase1_config = load_phase1_core_config(PHASE1_CONFIG_PATH)
    return dataset_path, selection_path, config_path, config, inputs, sweep, phase1_config


def _dataset_record_by_id(dataset_document: dict[str, object], record_id: str) -> dict[str, object]:
    for record in dataset_document["records"]:
        if str(record["record_id"]) == record_id:
            return record
    raise AssertionError(f"missing record {record_id}")


def _native_spectrum(record: dict[str, object], axis_descending: np.ndarray) -> Spectrum1D:
    axis = axis_descending[::-1].astype("<f8")
    intensity = np.asarray(record["stored_intensity"], dtype="<f4")[::-1].astype("<f8")
    return Spectrum1D(
        spectrum_id=f"bacteria_id_reference::{record['record_id']}",
        sample_id=None,
        axis_cm1=axis,
        intensity=intensity,
    )


def _artifact_tree(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }


class Phase4D2ProtocolBConfigAndLedgerTest(unittest.TestCase):
    def test_real_config_is_exact_and_drift_is_rejected(self) -> None:
        config = load_phase4_d2_protocol_b_eligibility_config(REAL_CONFIG_PATH)

        self.assertEqual(config.source_record_count, 5513)
        self.assertEqual(config.model_cell_count, 15)
        self.assertEqual(config.model_role_occurrence_count, 54750)
        self.assertEqual(config.support_point_count, 997)
        self.assertEqual(tuple(config.active_perturbation_ids), ACTIVE_PERTURBATION_IDS)
        self.assertEqual(tuple(config.artifact_payload_files), ARTIFACT_PAYLOAD_FILES)

        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            document["denominators"]["shot_source_union_counts"],
            {"5": 4690, "10": 5029, "20": 5513},
        )
        self.assertEqual(
            document["frozen_identities"]["shot_source_union_record_ids_sha256"],
            REAL_SHOT_UNION_DIGESTS,
        )
        self.assertEqual(document["expected"]["operator_cell_count"], 27565)
        self.assertEqual(document["expected"]["canonical_record_condition_count"], 226033)
        self.assertEqual(document["expected"]["class_summary_count"], 450)
        self.assertEqual(document["inherited_rulings"], _expected_inherited_rulings())
        self.assertEqual(
            document["authorities"]["phase4_step12_design_sha256"],
            "eeb2350ce6c359d84d9f1372962b8fcfb7fef7f623a77af18ae2e0abd03a4b38",
        )
        self.assertEqual(
            document["authorities"]["d1_model_recipe_sha256"],
            "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            drift = dict(document)
            drift["claim_boundary"] = "drifted"
            drift_path = temporary / REAL_CONFIG_PATH.name
            drift_path.write_text(
                json.dumps(
                    drift,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                Phase4D2ProtocolBEligibilityError,
                "frozen config identity",
            ):
                load_phase4_d2_protocol_b_eligibility_config(drift_path)

    def test_real_config_binds_exact_step16_shot_union_digests(self) -> None:
        config = load_phase4_d2_protocol_b_eligibility_config(REAL_CONFIG_PATH)
        self.assertEqual(
            dict(config.frozen_identities["shot_source_union_record_ids_sha256"]),
            REAL_SHOT_UNION_DIGESTS,
        )

    def test_real_config_inherited_rulings_match_step16_bounds(self) -> None:
        config = load_phase4_d2_protocol_b_eligibility_config(REAL_CONFIG_PATH)
        self.assertEqual(dict(config.inherited_rulings), _expected_inherited_rulings())

    def test_real_config_includes_step16_authority_bindings(self) -> None:
        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            document["authorities"]["phase4_step12_design_sha256"],
            "eeb2350ce6c359d84d9f1372962b8fcfb7fef7f623a77af18ae2e0abd03a4b38",
        )
        self.assertEqual(
            document["authorities"]["d1_model_recipe_sha256"],
            "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046",
        )

    def test_real_reconstruction_fails_closed_on_shot_union_digest_mismatch(self) -> None:
        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        drift = json.loads(json.dumps(document))
        drift["frozen_identities"]["shot_source_union_record_ids_sha256"]["5"] = "0" * 64
        raw = _canonical(drift)
        config = parse_phase4_d2_protocol_b_eligibility_config(
            REAL_CONFIG_PATH,
            raw,
            require_frozen_identity=False,
        )
        with self.assertRaisesRegex(
            Phase4D2ProtocolBEligibilityError,
            "shot_source_union_record_ids_sha256",
        ):
            reconstruct_d2_protocol_b_inputs(
                ROOT / DATASET_RELATIVE_PATH,
                ROOT / SELECTION_RELATIVE_PATH,
                config,
            )

    def test_reconstructs_nested_shot_unions_and_exact_role_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, _, config, inputs, _, _ = _fixture_inputs(root)
            self.assertEqual(len(inputs.source_records), 86)
            self.assertEqual(len(inputs.model_cells), 6)
            self.assertEqual(len(inputs.model_role_occurrences), 164)
            self.assertEqual(
                [(row["seed"], row["shot_count"]) for row in inputs.model_cells],
                [(0, 5), (0, 10), (0, 20), (1, 5), (1, 10), (1, 20)],
            )

            shot_union_counts = {
                str(shot): sum(shot in row["shot_memberships"] for row in inputs.source_records)
                for shot in SHOT_COUNTS
            }
            self.assertEqual(shot_union_counts, {"5": 26, "10": 46, "20": 86})
            self.assertEqual(
                Counter(row["shot_count"] for row in inputs.model_role_occurrences),
                Counter({5: 28, 10: 48, 20: 88}),
            )
            self.assertEqual(
                Counter(row["role"] for row in inputs.model_role_occurrences if row["shot_count"] == 5),
                Counter({"train": 20, "validation": 4, "test": 4}),
            )
            self.assertEqual(
                next(
                    row["shot_memberships"]
                    for row in inputs.source_records
                    if row["record_id"] == "finetune-c0-seed0-train-15"
                ),
                [20],
            )
            self.assertEqual(
                next(
                    row["shot_memberships"]
                    for row in inputs.source_records
                    if row["record_id"] == "finetune-c0-seed0-train-04"
                ),
                [5, 10, 20],
            )
            self.assertEqual(
                config.frozen_identities["shot_source_union_record_ids_sha256"]["20"],
                _ids_digest(sorted(str(row["record_id"]) for row in inputs.source_records)),
            )

    def test_record_used_only_by_20_shot_closes_20_not_5_or_10(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, _, config, inputs, _, _ = _fixture_inputs(root)
            operator_cells = []
            for source_record in inputs.source_records:
                for perturbation_id in ACTIVE_PERTURBATION_IDS:
                    operator_cells.append(
                        {
                            "class_label": source_record["class_label"],
                            "perturbation_id": perturbation_id,
                            "record_id": source_record["record_id"],
                            "state": "complete",
                        }
                    )
            target = next(
                row
                for row in operator_cells
                if row["record_id"] == "finetune-c0-seed0-train-15"
                and row["perturbation_id"] == "p08"
            )
            target["state"] = "failed_runtime"
            class_rows, gate = evaluate_d2_protocol_b_all_role_gates(
                inputs.source_records,
                inputs.model_cells,
                inputs.model_role_occurrences,
                operator_cells,
                config,
            )
            self.assertEqual(len(class_rows), 30)
            self.assertEqual(gate["shots"]["5"]["operators"]["p08"]["state"], "evaluable")
            self.assertEqual(gate["shots"]["10"]["operators"]["p08"]["state"], "evaluable")
            self.assertEqual(
                gate["shots"]["20"]["operators"]["p08"]["state"],
                "not_evaluable_coverage",
            )
            self.assertEqual(gate["shots"]["20"]["operators"]["p08"]["complete_record_count"], 85)
            self.assertEqual(gate["shots"]["20"]["operators"]["p08"]["required_record_count"], 86)
            self.assertEqual(gate["shots"]["20"]["operators"]["p08"]["complete_class_count"], 1)
            self.assertEqual(gate["shots"]["20"]["operators"]["p08"]["required_class_count"], 2)


class Phase4D2ProtocolBArtifactTest(unittest.TestCase):
    def test_inherited_p1_p7_rulings_are_not_executed_or_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, _, _, inputs, sweep, phase1_config = _fixture_inputs(root)
            output = root / "artifact"
            summary = build_phase4_d2_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                sweep=sweep,
                phase1_config=phase1_config,
                config=parse_phase4_d2_protocol_b_eligibility_config(
                    root / "config.json",
                    (root / "config.json").read_bytes(),
                    require_frozen_identity=False,
                ),
                worker_count=2,
            )
            self.assertEqual(summary.operator_cell_count, 430)
            operator_cells = [
                json.loads(line)
                for line in (output / "operator_cells.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                sorted({row["perturbation_id"] for row in operator_cells}),
                list(ACTIVE_PERTURBATION_IDS),
            )
            gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
            self.assertEqual(
                gate["inherited_rulings"],
                _synthetic_inherited_rulings(),
            )
            class_summaries = [
                json.loads(line)
                for line in (output / "class_summaries.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(class_summaries), 30)
            self.assertTrue(
                all(row["perturbation_id"] in ACTIVE_PERTURBATION_IDS for row in class_summaries)
            )

    def test_synthetic_build_and_independent_verifier_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, config_path, config, inputs, sweep, phase1_config = _fixture_inputs(root)
            output = root / "artifact"
            summary = build_phase4_d2_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                sweep=sweep,
                phase1_config=phase1_config,
                config=config,
                worker_count=2,
            )
            verified = verify_phase4_d2_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                sweep=sweep,
                phase1_config=phase1_config,
                config_path=config_path,
                worker_count=1,
            )
            self.assertEqual(summary.run_id, verified.run_id)
            self.assertEqual(summary.status, verified.status)
            self.assertEqual(summary.source_record_count, 86)
            self.assertEqual(summary.model_cell_count, 6)
            self.assertEqual(summary.role_occurrence_count, 164)
            self.assertEqual(summary.operator_cell_count, 430)
            self.assertEqual(summary.record_condition_count, 3526)
            self.assertEqual(summary.class_summary_count, 30)

            observed = {item.name for item in output.iterdir() if item.is_file()}
            marker_name = next(iter(observed & {"complete.json", "failed.json"}))
            self.assertEqual(
                observed,
                set(ARTIFACT_PAYLOAD_FILES) | {"SHA256SUMS", marker_name},
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["protocol"], "B")
            self.assertEqual(manifest["artifact_order"], list(ARTIFACT_PAYLOAD_FILES))
            self.assertEqual(
                manifest["claim_boundary"],
                "outcome_blind_protocol_b_all_role_eligibility_only",
            )
            self.assertEqual(
                manifest["run_identity"]["inherited_rulings"],
                _synthetic_inherited_rulings(),
            )

            checksums = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [line.split("  ", 1)[1] for line in checksums],
                list(ARTIFACT_PAYLOAD_FILES) + [marker_name],
            )
            condition_rows = [
                json.loads(line)
                for line in (output / "record_conditions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(condition_rows), 3526)
            alpha0_rows = [row for row in condition_rows if row["condition_kind"] == "alpha0"]
            self.assertEqual(len(alpha0_rows), 86)
            self.assertTrue(all(row["perturbation_id"] is None for row in alpha0_rows))
            self.assertTrue(all(row["condition_id"] == "alpha0" for row in alpha0_rows))
            self.assertTrue(all(row["alpha"] == 0.0 for row in alpha0_rows))
            self.assertEqual(_artifact_tree(output), _artifact_tree(output))

    def test_real_native_gate_keeps_p11_intensity_and_shifts_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, _, config, inputs, sweep, phase1_config = _fixture_inputs(root)
            output = root / "artifact"
            build_phase4_d2_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                sweep=sweep,
                phase1_config=phase1_config,
                config=config,
                worker_count=1,
            )

            dataset_document = _synthetic_dataset_document()
            source_record = _dataset_record_by_id(dataset_document, "test-c0-r0")
            source_spectrum = _native_spectrum(source_record, _decreasing_axis())
            source_axis_sha256 = _array_sha(source_spectrum.axis_cm1, dtype="<f8")
            source_intensity_sha256 = _array_sha(source_spectrum.intensity, dtype="<f8")

            operator_cells = [
                json.loads(line)
                for line in (output / "operator_cells.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            p11_cell = next(
                row
                for row in operator_cells
                if row["record_id"] == "test-c0-r0" and row["perturbation_id"] == "p11"
            )
            alpha0_output = next(row for row in p11_cell["outputs"] if row["alpha"] == 0.0)
            positive_output = next(row for row in p11_cell["outputs"] if row["alpha"] == 0.05)

            self.assertEqual(alpha0_output["output_axis_sha256"], source_axis_sha256)
            self.assertEqual(alpha0_output["output_intensity_sha256"], source_intensity_sha256)
            self.assertTrue(positive_output["axis_changed"])
            self.assertFalse(positive_output["intensity_changed"])
            self.assertEqual(positive_output["output_intensity_sha256"], source_intensity_sha256)
            self.assertNotEqual(positive_output["output_axis_sha256"], source_axis_sha256)
            self.assertIn("diagnostic_max_abs_offset_cm1", p11_cell["native_gate"])
            self.assertIn("observed_max_abs_offset_cm1", p11_cell["native_gate"])
            observed_offsets = p11_cell["native_gate"]["observed_max_abs_offset_cm1"]
            self.assertEqual(observed_offsets[0], 0.0)
            self.assertGreater(observed_offsets[1], 0.0)

            condition_rows = [
                json.loads(line)
                for line in (output / "record_conditions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            alpha0_row = next(
                row
                for row in condition_rows
                if row["record_id"] == "test-c0-r0" and row["condition_kind"] == "alpha0"
            )
            self.assertEqual(alpha0_row["axis_sha256"], source_axis_sha256)
            self.assertEqual(alpha0_row["intensity_sha256"], source_intensity_sha256)

    def test_p10_admission_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path, selection_path, config_path = _prepare_fixture(root)
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            huge_axis = np.linspace(1800.0, 300.0, 100000, dtype="<f4")
            dataset["native_axis_cm1_decreasing_f32"] = huge_axis.tolist()
            for index, record in enumerate(dataset["records"]):
                record["stored_intensity"] = _record_intensity(
                    int(record["class_label"]),
                    index,
                    huge_axis,
                )
            _write_json(dataset_path, dataset)
            config = parse_phase4_d2_protocol_b_eligibility_config(
                config_path,
                config_path.read_bytes(),
                require_frozen_identity=False,
            )
            inputs = reconstruct_d2_protocol_b_inputs(dataset_path, selection_path, config)
            sweep = load_perturbation_sweep_config(SWEEP_PATH)
            phase1_config = load_phase1_core_config(PHASE1_CONFIG_PATH)
            output = root / "artifact"
            with self.assertRaisesRegex(
                Phase4D2ProtocolBEligibilityError,
                "64 GiB",
            ):
                build_phase4_d2_protocol_b_eligibility_from_inputs(
                    output,
                    inputs=inputs,
                    sweep=sweep,
                    phase1_config=phase1_config,
                    config=config,
                    worker_count=1,
                )
            self.assertFalse(output.exists())

    def test_verifier_does_not_import_or_call_production_builder(self) -> None:
        verifier_path = (
            ROOT / "rpe/runner/phase4_d2_protocol_b_eligibility_verifier.py"
        )
        verifier_source = verifier_path.read_text(encoding="utf-8")
        self.assertNotIn(
            "_PRODUCTION_PATH.read_text",
            verifier_source,
            "verifier must not read production source text",
        )
        tree = ast.parse(verifier_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase4_d2_protocol_b_eligibility":
                self.fail("verifier must not import any symbol from phase4_d2_protocol_b_eligibility.py")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "rpe.runner.phase4_d2_protocol_b_eligibility":
                        self.fail("verifier must not import phase4_d2_protocol_b_eligibility.py")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "exec":
                self.fail("verifier must not exec production source")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, _, config_path, config, inputs, sweep, phase1_config = _fixture_inputs(root)
            output = root / "artifact"
            built = build_phase4_d2_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                sweep=sweep,
                phase1_config=phase1_config,
                config=config,
                worker_count=1,
            )
            verifier_module_name = "rpe.runner.phase4_d2_protocol_b_eligibility_verifier"
            verifier_module = sys.modules[verifier_module_name]
            with patch(
                "rpe.runner.phase4_d2_protocol_b_eligibility.build_phase4_d2_protocol_b_eligibility_from_inputs",
                side_effect=AssertionError("production builder must not be called by verifier"),
            ):
                try:
                    reloaded = importlib.reload(verifier_module)
                    verified = reloaded.verify_phase4_d2_protocol_b_eligibility_from_inputs(
                        output,
                        inputs=inputs,
                        sweep=sweep,
                        phase1_config=phase1_config,
                        config_path=config_path,
                        worker_count=1,
                    )
                finally:
                    importlib.reload(verifier_module)
            self.assertEqual(verified.run_id, built.run_id)
            self.assertEqual(verified.status, built.status)

    def test_verifier_record_jobs_use_spawn_process_pool(self) -> None:
        verifier_path = (
            ROOT / "rpe/runner/phase4_d2_protocol_b_eligibility_verifier.py"
        )
        verifier_source = verifier_path.read_text(encoding="utf-8")
        tree = ast.parse(verifier_source)
        run_record_jobs = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_record_jobs"
        )

        for node in ast.walk(run_record_jobs):
            if (
                isinstance(node, ast.With)
                and len(node.items) == 1
                and isinstance(node.items[0].context_expr, ast.Call)
                and isinstance(node.items[0].context_expr.func, ast.Name)
                and node.items[0].context_expr.func.id == "ThreadPoolExecutor"
            ):
                self.fail(
                    "verifier record jobs must not use ThreadPoolExecutor; "
                    "the multi-worker contract requires a spawn process pool"
                )

        process_pool_calls = [
            node
            for node in ast.walk(run_record_jobs)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ProcessPoolExecutor"
        ]
        self.assertEqual(
            len(process_pool_calls),
            1,
            "_run_record_jobs must construct exactly one ProcessPoolExecutor",
        )
        process_pool_call = process_pool_calls[0]
        keywords = {keyword.arg: keyword.value for keyword in process_pool_call.keywords}
        self.assertIn("mp_context", keywords)
        self.assertIn("initializer", keywords)
        self.assertIn("initargs", keywords)
        self.assertIsInstance(keywords["initializer"], ast.Name)
        self.assertEqual(keywords["initializer"].id, "_initialize_process")
        self.assertIsInstance(keywords["mp_context"], ast.Call)
        self.assertIsInstance(keywords["mp_context"].func, ast.Attribute)
        self.assertIsInstance(keywords["mp_context"].func.value, ast.Name)
        self.assertEqual(keywords["mp_context"].func.value.id, "multiprocessing")
        self.assertEqual(keywords["mp_context"].func.attr, "get_context")
        self.assertEqual(
            [arg.value for arg in keywords["mp_context"].args if isinstance(arg, ast.Constant)],
            ["spawn"],
        )

    def test_cli_exposes_no_bypass_or_outcome_flags(self) -> None:
        with self.assertRaises(SystemExit):
            protocol_b_cli_main(["verify", "--run-path", "x", "--no-reexecute"])
        with self.assertRaises(SystemExit):
            protocol_b_cli_main(["build", "--output-root", "x", "--protocol", "A"])
        with self.assertRaises(SystemExit):
            protocol_b_cli_main(["build", "--output-root", "x", "--outcome"])


if __name__ == "__main__":
    unittest.main()
