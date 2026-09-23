from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import (  # noqa: E402
    PeakPairInput,
    SingleSpectrumInput,
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.metrics import (  # noqa: E402
    ISLikeStructureToNoiseMetric,
    MAEMetric,
    MSEMetric,
    NMSEMetric,
    PeakDetectionCurvesMetric,
    PearsonRMetric,
    RMSEMetric,
    SAMMetric,
    Wasserstein1Metric,
)
from rpe.methods import load_classical_catalog  # noqa: E402
from rpe.methods.catalog import Phase3System, TaskLine  # noqa: E402
from rpe.methods.classical.peaks import run_peak_detection_system  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_config import load_phase1_core_config  # noqa: E402
from rpe.runner.phase1_perturbations import (  # noqa: E402
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow  # noqa: E402
from rpe.runner.d2_selection import (  # noqa: E402
    D2SelectionValidationError,
    validate_d2_few_shot_selection,
)
import rpe.runner.phase4_d2_eligibility as production_d2  # noqa: E402
import rpe.runner.phase4_d2_eligibility_verifier as verifier_d2  # noqa: E402
from rpe.runner.phase4_d2_eligibility import (  # noqa: E402
    ACTIVE_PERTURBATION_IDS,
    ALL_PERTURBATION_IDS,
    ARTIFACT_STATIC_FILES,
    CWT_SYSTEM_ID,
    FULL_DOMAIN_PERTURBATION_IDS,
    METRIC_OUTPUT_IDS,
    PEAK_PERTURBATION_IDS,
    Phase4D2EligibilityError,
    build_phase4_d2_eligibility_from_inputs,
    evaluate_d2_eligibility_gates,
    load_phase4_d2_eligibility_config,
    parse_phase4_d2_eligibility_config,
    project_d2_support,
    reconstruct_d2_eligibility_inputs,
    validate_outcome_blind_d2_payload,
    _execute_d2_record,
)
from rpe.runner.phase4_d2_eligibility_verifier import (  # noqa: E402
    verify_phase4_d2_eligibility_from_inputs,
)
from tools.run_phase4_d2_eligibility import main as d2_cli_main  # noqa: E402


SWEEP_PATH = ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_PATH = ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
REAL_CONFIG_PATH = (
    ROOT
    / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
)
SELECTION_CONFIG_PATH = ROOT / "experiments/phase05/configs/d2_few_shot_selection.json"
SELECTION_ARTIFACT_PATH = (
    ROOT / "results/phase05/d2_selection_artifact_frozen.json"
)
REAL_SELECTION_ARTIFACT_PATH = (
    ROOT
    / "results/phase05/d2"
    / "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
    / "selection.json"
)
CATALOG_PATH = (
    ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
)


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


def _canonical_science(value: object) -> bytes:
    def ready(item: object) -> object:
        if is_dataclass(item):
            return {field.name: ready(getattr(item, field.name)) for field in fields(item)}
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, dict) or hasattr(item, "items"):
            return {str(key): ready(current) for key, current in item.items()}
        if isinstance(item, (tuple, list)):
            return [ready(current) for current in item]
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        return item
    return _canonical(ready(value))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _ids_digest(values: list[str] | tuple[str, ...]) -> str:
    return _sha_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _array_sha(value: np.ndarray, *, dtype: str) -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _hex_alpha(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def _build_cwt_system() -> Phase3System:
    from rpe.methods import load_classical_catalog

    catalog = load_classical_catalog(CATALOG_PATH)
    matches = [
        system
        for system in catalog.systems
        if system.task_line is TaskLine.PEAK_DETECTION
        and system.system_id == CWT_SYSTEM_ID
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one CWT system for {CWT_SYSTEM_ID}")
    return matches[0]


def _synthetic_selection_document() -> dict[str, object]:
    seeds = [0, 1]
    shot_counts = [5, 10, 20]
    test_ids = [f"test-{index:04d}" for index in range(4)]
    selections = []
    for seed in seeds:
        class_docs = []
        train_ids_by_shot = {str(shot): [] for shot in shot_counts}
        validation_ids = []
        for class_label in range(2):
            ordered = [
                f"finetune-c{class_label}-seed{seed}-train-{index:02d}"
                for index in range(20)
            ]
            validation = [
                f"finetune-c{class_label}-seed{seed}-val-{index:02d}"
                for index in range(10)
            ]
            class_docs.append(
                {
                    "candidate_count": 100,
                    "class_label": class_label,
                    "ordered_train_record_ids": ordered,
                    "source_split": "finetune",
                    "train_record_ids": {
                        str(shot): ordered[:shot] for shot in shot_counts
                    },
                    "train_record_ids_sha256": {
                        str(shot): _ids_digest(ordered[:shot]) for shot in shot_counts
                    },
                    "validation_record_ids": validation,
                    "validation_record_ids_sha256": _ids_digest(validation),
                }
            )
            for shot in shot_counts:
                train_ids_by_shot[str(shot)].extend(ordered[:shot])
            validation_ids.extend(validation)
        selections.append(
            {
                "classes": class_docs,
                "seed": seed,
                "train_record_ids_sha256": {
                    str(shot): _ids_digest(train_ids_by_shot[str(shot)])
                    for shot in shot_counts
                },
                "validation_record_ids_sha256": _ids_digest(validation_ids),
            }
        )
    return {
        "code": {},
        "conditions": [
            "released_input_control",
            "released_input_plus_sg",
        ],
        "config": {
            "bytes": 858,
            "path": "d2_few_shot_selection.json",
            "sha256": "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138",
        },
        "dataset": {
            "dataset_id": "bacteria_id_reference",
            "files": {},
            "record_count": 184,
            "split_counts": {
                "finetune": 180,
                "reference": 0,
                "test": 4,
            },
        },
        "experiment_id": "d2_bacteria_id_few_shot",
        "schema_version": "phase05-d2-selection-artifact-v1",
        "seeds": seeds,
        "selections": selections,
        "shot_counts": shot_counts,
        "status": "frozen",
        "test": {
            "count": len(test_ids),
            "record_ids_sha256": _ids_digest(test_ids),
            "source_split": "test",
        },
        "validation_per_class": 10,
    }


def _decreasing_axis() -> np.ndarray:
    return _read_only(np.linspace(1800.0, 300.0, 1000, dtype="<f4"))


def _synthetic_test_spectra() -> tuple[dict[str, object], ...]:
    axis = _decreasing_axis()
    records = []
    for class_label in range(2):
        for replicate in range(2):
            record_id = f"test-{class_label * 2 + replicate:04d}"
            coordinate = axis.astype("<f8")
            baseline = (
                1.0
                + 0.08 * class_label
                + 0.015 * replicate
                + 0.08 * np.sin(coordinate / 43.0)
                + 0.65 * np.exp(-((coordinate - (710.0 + 35.0 * class_label)) / 24.0) ** 2)
                + 0.42 * np.exp(-((coordinate - (1220.0 + 18.0 * replicate)) / 37.0) ** 2)
            ).astype("<f4")
            records.append(
                {
                    "class_label": class_label,
                    "record_id": record_id,
                    "source_split": "test",
                    "stored_axis_cm1": axis.tolist(),
                    "stored_intensity": baseline.tolist(),
                }
            )
    return tuple(records)


def _native_spectrum(record: dict[str, object]) -> Spectrum1D:
    axis = np.asarray(record["stored_axis_cm1"], dtype="<f4")[::-1].astype("<f8")
    intensity = np.asarray(record["stored_intensity"], dtype="<f4")[::-1].astype("<f8")
    return Spectrum1D(
        spectrum_id=f"bacteria_id_reference::{record['record_id']}",
        sample_id=None,
        axis_cm1=axis,
        intensity=intensity,
    )


def _phase1_source_for_oracle(spectrum: Spectrum1D, *, record_id: str, class_label: int) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=0,
            record_id=record_id,
            sample_id=record_id,
            class_label=class_label,
            mineral_name=f"bacteria-{class_label}",
            axis_id=f"native::{_array_sha(spectrum.axis_cm1, dtype='<f8')}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1, dtype="<f8"),
        normalized_intensity_float64_sha256=_array_sha(spectrum.intensity, dtype="<f8"),
        provenance={
            "license": None,
            "license_status": "not_stated",
            "retrieved_date": "2026-08-20",
            "sha256": "0" * 64,
            "source_artifact": "bacteria_id_reference",
            "source_url": "local://synthetic",
        },
    )


def _synthetic_dataset_document() -> dict[str, object]:
    return {
        "dataset_id": "bacteria_id_reference",
        "native_axis_cm1_decreasing_f32": _decreasing_axis().tolist(),
        "test_records": list(_synthetic_test_spectra()),
    }


def _support_grid() -> list[float]:
    return _decreasing_axis()[::-1].astype("<f8")[3:].tolist()


def _config_document() -> dict[str, object]:
    selection = _synthetic_selection_document()
    dataset = _synthetic_dataset_document()
    test_ids = [row["record_id"] for row in dataset["test_records"]]
    native_axis_f32 = _decreasing_axis()
    native_axis_f64 = native_axis_f32[::-1].astype("<f8")
    support_axis_f64 = np.asarray(_support_grid(), dtype="<f8")
    return {
        "active_perturbation_ids": list(ACTIVE_PERTURBATION_IDS),
        "alpha_grid": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8],
        "authorities": {
            "bacteria_id_retained_snapshot_sha256": (
                "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02"
            ),
            "bacteria_id_sha256sums_sha256": (
                "6d6d1399ac0e5197a9e51a1a0cf32edf51c924ce8d7c12aeb9d58f9af93c7f3e"
            ),
            "d2_phase05_config_sha256": (
                "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7"
            ),
            "d2_selection_artifact_sha256": _sha_bytes(_canonical(selection)),
            "d2_selection_config_sha256": (
                "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
            ),
            "phase1_core_config_sha256": _sha_file(PHASE1_CONFIG_PATH),
            "phase4_step01_sha256": (
                "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f"
            ),
            "phase4_step11_sha256": (
                "b2da0f26c60f930902c8b899091b4e1553bdbd160404b252608816785146ab18"
            ),
            "phase4_step12_sha256": (
                "d6d039a6381323aed5b14cac4a8a3a9a5dffc2b793141cf9f5646de2ea1447ae"
            ),
            "sweep_sha256": (
                "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"
            ),
        },
        "claim_boundary": (
            "outcome_blind_eligibility_only_no_predictions_metrics_peak_lists_alignment_or_inference"
        ),
        "code_authority": {},
        "denominators": {
            "class_count": 2,
            "finetune_union_record_count": 120,
            "model_cell_count": 6,
            "model_role_occurrence_count": 284,
            "source_record_count": 124,
            "test_record_count": 4,
        },
        "environment_authority": {},
        "expected": {
            "active_cell_count": 40,
            "apply_call_count": 360,
            "cell_count": 48,
            "class_summary_count": 24,
            "condition_count": 164,
            "cwt_receipt_count": 164,
            "inactive_cell_count": 8,
            "metric_status_count": 2132,
        },
        "experiment_id": "phase4-d2-protocol-a-full-domain-eligibility-v1",
        "frozen_identities": {
            "model_seed_ids": [0, 1],
            "native_axis_decreasing_f32_sha256": _array_sha(
                native_axis_f32, dtype="<f4"
            ),
            "native_axis_id": (
                "91e468d92cd4215f23c6c785b4611dc6f1ffe39cbb9134d11c60345954f80378"
            ),
            "native_axis_increasing_f64_sha256": _array_sha(
                native_axis_f64, dtype="<f8"
            ),
            "selection_test_record_ids_sha256": _ids_digest(test_ids),
            "support_axis_f32_sha256": _array_sha(
                support_axis_f64.astype("<f4"), dtype="<f4"
            ),
            "support_axis_f64_sha256": _array_sha(support_axis_f64, dtype="<f8"),
            "support_point_count": len(support_axis_f64),
            "test_record_ids_sha256": _ids_digest(test_ids),
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
        "metric_output_ids": list(METRIC_OUTPUT_IDS),
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
        "schema_version": "phase4-d2-protocol-a-full-domain-eligibility-config-v1",
        "support_grid": {
            "coordinates_cm1": _support_grid(),
            "max_in_range_native_gap_cm1": 1.561,
            "point_count": 997,
        },
        "synthetic_fixture": True,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _rewrite_test_record_ids(
    dataset: dict[str, object],
    selection: dict[str, object],
    *,
    prefix: str,
) -> tuple[dict[str, object], dict[str, object]]:
    rewritten_dataset = json.loads(_canonical(dataset))
    rewritten_selection = json.loads(_canonical(selection))
    record_ids = [
        str(row["record_id"]) for row in rewritten_dataset["test_records"]
    ]
    mapping = {
        record_id: f"{prefix}-{index:04d}" for index, record_id in enumerate(record_ids)
    }
    for row in rewritten_dataset["test_records"]:
        row["record_id"] = mapping[str(row["record_id"])]
    rewritten_selection["test"]["record_ids_sha256"] = _ids_digest(
        [mapping[record_id] for record_id in record_ids]
    )
    return rewritten_dataset, rewritten_selection


def _function_node(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


class Phase4D2EligibilityConfigTest(unittest.TestCase):
    def test_real_config_loads_exact_frozen_identity(self) -> None:
        config = load_phase4_d2_eligibility_config(REAL_CONFIG_PATH)

        self.assertEqual(config.schema_version, "phase4-d2-protocol-a-full-domain-eligibility-config-v1")
        self.assertEqual(config.test_record_count, 3000)
        self.assertEqual(config.class_count, 30)
        self.assertEqual(config.model_cell_count, 15)
        self.assertEqual(config.model_role_occurrence_count, 54750)
        self.assertEqual(config.support_point_count, 997)
        self.assertEqual(config.support_max_gap_cm1, 1.561)
        self.assertEqual(config.p10_memory_budget_bytes, 64 * 2**30)

        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            document["artifact_payload_files"],
            [
                "config.json",
                "source_records.jsonl",
                "model_cells.jsonl",
                "model_role_occurrences.jsonl",
                "cells.jsonl",
                "record_conditions.jsonl",
                "metric_statuses.jsonl",
                "cwt_receipts.jsonl",
                "class_summaries.jsonl",
                "common_support.jsonl",
                "gate.json",
                "manifest.json",
            ],
        )
        expected_code_paths = [
            "rpe/downstream/bacteria_id.py",
            "rpe/evaluation/contracts.py",
            "rpe/methods/classical/peaks.py",
            "rpe/metrics/fidelity.py",
            "rpe/metrics/peak.py",
            "rpe/metrics/reference_free.py",
            "rpe/metrics/transport.py",
            "rpe/perturb/contracts.py",
            "rpe/perturb/sweep.py",
            "rpe/runner/d2_selection.py",
            "rpe/runner/phase1_config.py",
            "rpe/runner/phase1_perturbations.py",
            "rpe/runner/phase1_selection.py",
            "rpe/runner/phase1_types.py",
            "rpe/runner/phase4_d2_eligibility.py",
            "rpe/runner/phase4_d2_eligibility_verifier.py",
            "tools/run_phase4_d2_eligibility.py",
        ]
        self.assertEqual(list(document["code_authority"]), expected_code_paths)
        for relative_path in expected_code_paths:
            target = ROOT / relative_path
            self.assertEqual(
                document["code_authority"][relative_path],
                {"bytes": target.stat().st_size, "sha256": _sha_file(target)},
            )
        self.assertEqual(
            document["environment_authority"],
            {
                "h5py": "3.16.0",
                "machine": "x86_64",
                "numpy": "2.5.2",
                "python": "3.13.11",
                "scikit_learn": "1.9.0",
                "scipy": "1.18.0",
                "system": "Linux",
                "threadpoolctl": "3.6.0",
            },
        )
        self.assertEqual(
            document["trust_anchor"],
            {
                "config_authority_relative_path":
                    "rpe/runner/phase4_d2_eligibility_authority.py"
            },
        )
        self.assertEqual(
            set(document["authorities"]),
            {
                "parent_plan_sha256",
                "phase4_step01_sha256",
                "phase4_step11_sha256",
                "phase4_step12_design_sha256",
                "step13_plan_sha256",
                "sweep_sha256",
                "phase1_core_config_sha256",
                "d2_selection_config_sha256",
                "d2_selection_artifact_path",
                "d2_selection_artifact_sha256",
                "d2_phase05_runner_config_sha256",
                "d1_model_recipe_sha256",
                "classical_catalog_sha256",
                "bacteria_id_sha256sums_sha256",
                "bacteria_id_retained_snapshot_sha256",
            },
        )

        changed = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        changed["gates"]["peak_common_record_fraction"] = 0.89
        with self.assertRaisesRegex(
            Phase4D2EligibilityError,
            "frozen config identity",
        ):
            parse_phase4_d2_eligibility_config(
                Path("changed.json"),
                _canonical(changed),
                require_frozen_identity=True,
            )

    def test_synthetic_config_rejects_scope_and_count_drift(self) -> None:
        document = _config_document()
        config = parse_phase4_d2_eligibility_config(
            Path("synthetic.json"),
            _canonical(document),
            require_frozen_identity=False,
        )
        self.assertEqual(config.test_record_count, 4)
        self.assertEqual(config.model_cell_count, 6)
        self.assertEqual(config.active_perturbation_ids, ACTIVE_PERTURBATION_IDS)

        drift = json.loads(_canonical(document))
        drift["denominators"]["model_role_occurrence_count"] = 451
        with self.assertRaisesRegex(Phase4D2EligibilityError, "model_role_occurrence_count"):
            parse_phase4_d2_eligibility_config(
                Path("synthetic.json"),
                _canonical(drift),
                require_frozen_identity=False,
            )

    def test_real_config_binds_literal_support_coordinates_not_uniform_range_metadata(self) -> None:
        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        support_grid = document["support_grid"]

        self.assertIn("coordinates_cm1", support_grid)
        self.assertNotIn("step_cm1", support_grid)
        self.assertNotIn("start_cm1", support_grid)
        self.assertNotIn("stop_cm1", support_grid)

        coordinates = np.asarray(support_grid["coordinates_cm1"], dtype="<f8")
        self.assertEqual(int(coordinates.size), 997)
        self.assertEqual(_array_sha(coordinates, dtype="<f8"), "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6")
        self.assertEqual(_array_sha(coordinates.astype("<f4"), dtype="<f4"), "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59")


class Phase4D2EligibilityInputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.selection_path = self.root / "selection.json"
        self.dataset_path = self.root / "dataset.json"
        _write_json(self.selection_path, _synthetic_selection_document())
        _write_json(self.dataset_path, _synthetic_dataset_document())
        self.config = parse_phase4_d2_eligibility_config(
            Path("synthetic.json"),
            _canonical(_config_document()),
            require_frozen_identity=False,
        )

    def test_reconstructs_axis_adapter_and_source_vs_role_ledgers(self) -> None:
        inputs = reconstruct_d2_eligibility_inputs(
            self.dataset_path,
            self.selection_path,
            self.config,
        )

        self.assertEqual(len(inputs.source_records), 124)
        self.assertEqual(len(inputs.model_cells), 6)
        self.assertEqual(len(inputs.model_role_occurrences), 284)
        self.assertEqual(len(inputs.test_spectra), 4)
        self.assertEqual(inputs.test_record_ids_sha256, _ids_digest([f"test-{index:04d}" for index in range(4)]))
        first = inputs.test_spectra[0]
        self.assertEqual(first.axis_cm1.size, 1000)
        self.assertTrue(np.all(np.diff(first.axis_cm1) > 0.0))
        self.assertEqual(inputs.native_axis_point_count, 1000)
        self.assertEqual(inputs.support_axis_point_count, 997)

        for cell in inputs.model_cells:
            train_ids = tuple(cell["train_record_ids"])
            validation_ids = tuple(cell["validation_record_ids"])
            self.assertFalse(set(train_ids) & set(validation_ids))
            self.assertEqual(train_ids, tuple(sorted(train_ids)))
            self.assertEqual(validation_ids, tuple(sorted(validation_ids)))

    def test_rejects_forbidden_phase05_outcome_artifact(self) -> None:
        forbidden = self.root / "complete_cells.json"
        forbidden.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(Phase4D2EligibilityError, "forbidden Phase-0.5 outcome"):
            reconstruct_d2_eligibility_inputs(
                self.dataset_path,
                forbidden,
                self.config,
            )

    def test_reconstructs_frozen_real_bacteria_dataset_directory(self) -> None:
        config = load_phase4_d2_eligibility_config(REAL_CONFIG_PATH)
        inputs = reconstruct_d2_eligibility_inputs(
            ROOT / "data/unified/bacteria_id_reference",
            REAL_SELECTION_ARTIFACT_PATH,
            config,
        )

        self.assertEqual(len(inputs.source_records), 5513)
        self.assertEqual(len(inputs.model_cells), 15)
        self.assertEqual(len(inputs.model_role_occurrences), 54750)
        self.assertEqual(len(inputs.test_spectra), 3000)
        self.assertEqual(set(inputs.test_class_labels), set(range(30)))
        self.assertEqual(inputs.native_axis_point_count, 1000)
        self.assertEqual(inputs.support_axis_point_count, 997)
        self.assertEqual(
            inputs.test_record_ids_sha256,
            "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd",
        )
        self.assertEqual(
            _array_sha(inputs.support_axis_cm1, dtype="<f8"),
            "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6",
        )
        for row in inputs.source_records:
            self.assertIsInstance(row["class_label"], int)
            self.assertIsInstance(row["source_row"], int)
            self.assertGreaterEqual(row["source_row"], 0)
            self.assertEqual(len(row["native_axis_sha256"]), 64)
            self.assertEqual(len(row["native_intensity_sha256"]), 64)
            self.assertEqual(len(row["support_axis_sha256"]), 64)
            self.assertEqual(len(row["support_intensity_sha256"]), 64)
        role_counts: dict[str, int] = {}
        for role in inputs.model_role_occurrences:
            record_id = str(role["record_id"])
            role_counts[record_id] = role_counts.get(record_id, 0) + 1
        self.assertEqual(
            {str(row["record_id"]): int(row["role_count"]) for row in inputs.source_records},
            role_counts,
        )

    def test_project_support_rejects_gap_drift_and_preserves_alpha_zero_identity(self) -> None:
        spectrum = _native_spectrum(_synthetic_dataset_document()["test_records"][0])
        projected = project_d2_support(spectrum, self.config)
        direct = np.asarray(spectrum.intensity[3:], dtype="<f4")

        self.assertEqual(projected.dtype, np.dtype("<f4"))
        self.assertTrue(np.array_equal(projected, direct))

        sparse = Spectrum1D(
            spectrum_id="gap",
            sample_id=None,
            axis_cm1=np.asarray(
                [300.0, 304.5045166015625, 307.0, 1800.0],
                dtype="<f8",
            ),
            intensity=np.asarray([1.0, 2.0, 3.0, 4.0], dtype="<f8"),
        )
        with self.assertRaisesRegex(Phase4D2EligibilityError, "max_in_range_native_gap_cm1"):
            project_d2_support(sparse, self.config)

        zero = Spectrum1D(
            spectrum_id="zero-norm",
            sample_id=None,
            axis_cm1=spectrum.axis_cm1,
            intensity=np.zeros_like(spectrum.intensity),
        )
        for module in (production_d2, verifier_d2):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(module.Phase4D2EligibilityError, "zero norm"):
                    module.project_d2_support(zero, self.config)

        expected_bytes = estimate_p10_peak_bytes(1000)
        self.assertEqual(expected_bytes, 1105805824)


class Phase4D2EligibilityGateAndPayloadTest(unittest.TestCase):
    def test_gate_counts_shared_alpha_zero_in_metric_and_cwt_grids(self) -> None:
        """Fails if the shared alpha-zero rows are dropped from 123k grids."""
        config = parse_phase4_d2_eligibility_config(
            Path("synthetic.json"),
            _canonical(_config_document()),
            require_frozen_identity=False,
        )
        labels = [0, 0, 1, 1]
        cells = [
            {
                "class_label": label,
                "record_id": f"test-{index:04d}",
                "perturbation_id": perturbation_id,
                "state": (
                    "structurally_ineligible"
                    if perturbation_id in {"p06", "p07"}
                    else "complete"
                ),
            }
            for index, label in enumerate(labels)
            for perturbation_id in ALL_PERTURBATION_IDS
        ]
        conditions = [
            {
                "record_id": f"test-{index:04d}",
                "condition_id": condition_id,
                "state": "complete",
            }
            for index in range(len(labels))
            for condition_id in (
                "alpha0",
                *(
                    f"{perturbation_id}:{_hex_alpha(alpha)}"
                    for perturbation_id in FULL_DOMAIN_PERTURBATION_IDS
                    for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
                ),
            )
        ]
        metrics = [
            {
                "record_id": row["record_id"],
                "condition_id": row["condition_id"],
                "metric_output_id": output_id,
                "state": "complete",
            }
            for row in conditions
            for output_id in METRIC_OUTPUT_IDS
        ]
        cwt = [
            {
                "record_id": row["record_id"],
                "condition_id": row["condition_id"],
                "state": "complete",
            }
            for row in conditions
        ]

        for module in (production_d2, verifier_d2):
            with self.subTest(module=module.__name__):
                gate = module.evaluate_d2_eligibility_gates(
                    cells,
                    labels,
                    config,
                    record_conditions=conditions,
                    metric_statuses=metrics,
                    cwt_receipts=cwt,
                )
                self.assertEqual(gate["support_grid"]["planned"], 164)
                self.assertEqual(gate["cwt"]["planned"], 164)
                self.assertEqual(gate["cwt"]["complete"], 164)
                self.assertEqual(gate["metric_outputs"]["mse"]["planned"], 164)
                self.assertEqual(gate["metric_outputs"]["mse"]["failed"], 0)
                self.assertEqual(gate["full_domain_core"]["state"], "evaluable")
                self.assertEqual(gate["overall_status"], "pass")

    def test_gate_keeps_non_mse_and_cwt_candidate_failures_independent(self) -> None:
        config = parse_phase4_d2_eligibility_config(Path("synthetic.json"), _canonical(_config_document()), require_frozen_identity=False)
        labels = [0, 0, 0, 1, 1, 1]
        cells = [
            {"class_label": label, "record_id": f"test-{index:04d}", "perturbation_id": perturbation_id, "state": "structurally_ineligible" if perturbation_id in {"p06", "p07"} else "complete"}
            for index, label in enumerate(labels) for perturbation_id in ALL_PERTURBATION_IDS
        ]
        conditions = [
            {"record_id": f"test-{index:04d}", "condition_id": f"{perturbation_id}:{_hex_alpha(0.05)}", "state": "complete"}
            for index in range(len(labels)) for perturbation_id in FULL_DOMAIN_PERTURBATION_IDS
        ]
        metrics = [
            {"record_id": row["record_id"], "condition_id": row["condition_id"], "metric_output_id": output_id, "state": "complete"}
            for row in conditions for output_id in METRIC_OUTPUT_IDS
        ]
        cwt = [{"record_id": row["record_id"], "condition_id": row["condition_id"], "state": "complete"} for row in conditions]
        def gate_for(*, metric_id: str | None = None, cwt_failure: bool = False):
            changed_metrics = [dict(row) for row in metrics]
            changed_cwt = [dict(row) for row in cwt]
            if metric_id is not None:
                next(row for row in changed_metrics if row["metric_output_id"] == metric_id)["state"] = "failed_runtime"
            if cwt_failure:
                changed_cwt[0]["state"] = "failed_runtime"
            return evaluate_d2_eligibility_gates(cells, labels, config, record_conditions=conditions, metric_statuses=changed_metrics, cwt_receipts=changed_cwt)
        self.assertEqual(gate_for(metric_id="mse")["full_domain_core"]["state"], "not_evaluable_coverage")
        rmse_gate = gate_for(metric_id="rmse")
        self.assertEqual(rmse_gate["full_domain_core"]["state"], "evaluable")
        self.assertEqual(rmse_gate["metric_outputs"]["rmse"]["state"], "not_evaluable_incomplete_grid")
        cwt_gate = gate_for(cwt_failure=True)
        self.assertEqual(cwt_gate["full_domain_core"]["state"], "evaluable")
        self.assertEqual(cwt_gate["metric_outputs"]["precision"]["state"], "not_evaluable_incomplete_grid")

    def test_gate_uses_original_denominators_and_failed_marker_semantics(self) -> None:
        config = parse_phase4_d2_eligibility_config(
            Path("synthetic.json"),
            _canonical(_config_document()),
            require_frozen_identity=False,
        )
        cells = []
        class_labels = []
        for class_label in range(2):
            for replicate in range(3):
                record_id = f"test-{class_label * 3 + replicate:04d}"
                class_labels.append(class_label)
                for perturbation_id in ALL_PERTURBATION_IDS:
                    state = "complete"
                    if perturbation_id in {"p06", "p07"}:
                        state = "structurally_ineligible"
                    if perturbation_id == "p05" and record_id == "test-0000":
                        state = "not_applicable"
                    if perturbation_id == "p11" and record_id == "test-0005":
                        state = "failed_runtime"
                    cells.append(
                        {
                            "class_label": class_label,
                            "perturbation_id": perturbation_id,
                            "record_id": record_id,
                            "state": state,
                        }
                    )
        gate = evaluate_d2_eligibility_gates(cells, class_labels, config)

        self.assertEqual(gate["full_domain_core"]["state"], "not_evaluable_coverage")
        self.assertEqual(gate["peak_common_support"]["state"], "not_evaluable_coverage")
        self.assertEqual(gate["overall_status"], "fail")
        self.assertEqual(gate["marker_filename"], "failed.json")
        self.assertEqual(
            gate["full_domain_core"]["by_perturbation"]["p11"]["complete_record_count"],
            5,
        )
        self.assertEqual(
            gate["peak_common_support"]["p05"]["complete_record_count"],
            5,
        )

    def test_outcome_blind_payload_rejects_metric_values_predictions_and_peak_lists(self) -> None:
        valid = {
            "class_label": 0,
            "condition_id": f"p08:{_hex_alpha(0.05)}",
            "diagnostics_sha256": "a" * 64,
            "metric_output_id": "mse",
            "record_id": "test-0000",
            "result_sha256": "b" * 64,
            "state": "complete",
            "warning_sha256": None,
        }
        validate_outcome_blind_d2_payload(valid)

        for key, value in (
            ("metric_value", 1.0),
            ("prediction", 0),
            ("peak_list", []),
            ("peak_count", 4),
            ("selected_c", 0.1),
            ("alignment_gap", 0.2),
        ):
            invalid = dict(valid)
            invalid[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(Phase4D2EligibilityError, key):
                    validate_outcome_blind_d2_payload(invalid)

    def test_metric_and_cwt_statuses_are_state_only(self) -> None:
        source = _native_spectrum(_synthetic_dataset_document()["test_records"][0])
        perturbed = Spectrum1D(
            spectrum_id="perturbed",
            sample_id=None,
            axis_cm1=source.axis_cm1,
            intensity=source.intensity + 0.02 * np.sin(source.axis_cm1 / 31.0),
        )
        spectrum_request = SpectrumPairInput(source, perturbed)
        pair_metrics = {
            "mse": MSEMetric(),
            "rmse": RMSEMetric(),
            "mae": MAEMetric(),
            "sam": SAMMetric(),
            "pearson_r": PearsonRMetric(),
            "nmse": NMSEMetric(),
            "wasserstein_1_cm1": Wasserstein1Metric(),
        }
        observed = {}
        for output_id, metric in pair_metrics.items():
            result = evaluate_metric(metric, spectrum_request)
            observed[output_id] = result
            self.assertTrue(result.outputs)
            self.assertIn(output_id, {output.output_id for output in result.outputs})
        single_result = evaluate_metric(
            ISLikeStructureToNoiseMetric(),
            SingleSpectrumInput(source),
        )
        observed["is_like_structure_to_noise"] = single_result
        self.assertIn(
            "is_like_structure_to_noise",
            {output.output_id for output in single_result.outputs},
        )

        cwt_system = _build_cwt_system()
        receipt = run_peak_detection_system(cwt_system, perturbed)
        peak_metric = PeakDetectionCurvesMetric()
        peak_request = PeakPairInput(
            reference_peaks=tuple(peak.to_peak1d() for peak in receipt.peaks),
            candidate_peaks=tuple(peak.to_peak1d() for peak in receipt.peaks),
            position_tolerance_cm1=2.0,
            prominence_thresholds=(0.0,),
        )
        observed["peak_curves"] = evaluate_metric(peak_metric, peak_request)
        self.assertIn(receipt.status.value, {"complete", "complete_with_warning", "not_applicable"})
        self.assertTrue(receipt.diagnostics)
        if receipt.peaks:
            self.assertIsNotNone(receipt.peaks_sha256)
        self.assertNotIn("peak_count", observed["peak_curves"].diagnostics)


class Phase4D2EligibilityBuildVerifyAndCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.selection_path = self.root / "selection.json"
        self.dataset_path = self.root / "dataset.json"
        self.fixture_config_path = self.root / "config.json"
        self.output_path = self.root / "output"
        _write_json(self.selection_path, _synthetic_selection_document())
        _write_json(self.dataset_path, _synthetic_dataset_document())
        _write_json(self.fixture_config_path, _config_document())
        self.config = parse_phase4_d2_eligibility_config(
            Path("synthetic.json"),
            _canonical(_config_document()),
            require_frozen_identity=False,
        )
        self.inputs = reconstruct_d2_eligibility_inputs(
            self.dataset_path,
            self.selection_path,
            self.config,
        )
        self.sweep = load_perturbation_sweep_config(SWEEP_PATH)
        self.phase1_config = load_phase1_core_config(PHASE1_CONFIG_PATH)

    def test_bounded_worker_returns_only_json_ready_payloads(self) -> None:
        catalog = load_classical_catalog(CATALOG_PATH)
        cwt_system = next(system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID)
        payload = _execute_d2_record(
            record_order=0, spectrum=self.inputs.test_spectra[0],
            class_label=int(self.inputs.test_class_labels[0]), sweep=self.sweep,
            phase1_config=self.phase1_config, config=self.config,
            admission=P10MemoryAdmission(self.config.p10_memory_budget_bytes), cwt_system=cwt_system,
        )
        json.dumps(payload, sort_keys=True, allow_nan=False)

        def assert_json_ready(value: object) -> None:
            self.assertNotIsInstance(value, (Spectrum1D, np.ndarray))
            self.assertNotIn(type(value).__name__, {"Phase1Cell", "Peak1D", "PeakReceipt", "MetricResult"})
            if isinstance(value, dict) or hasattr(value, "items"):
                for item in value.values():
                    assert_json_ready(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    assert_json_ready(item)

        assert_json_ready(payload)
        self.assertEqual(len(payload["record_conditions"]), 41)
        self.assertEqual(len(payload["metric_statuses"]), 41 * 13)
        self.assertEqual(len(payload["cwt_receipts"]), 41)
        for row in payload["cells"]:
            if row["state"] == "complete":
                self.assertEqual(len(row["outputs"]), 9)
                self.assertTrue(
                    all(
                        isinstance(output.get("support_sha256"), str)
                        and len(output["support_sha256"]) == 64
                        for output in row["outputs"]
                    )
                )

    def test_record_jobs_use_processes_and_static_p10_budget(self) -> None:
        """Fails if CPU-heavy D2 work stays in the parent or can exceed 64 GiB."""
        jobs = tuple(
            enumerate(
                zip(
                    self.inputs.test_spectra,
                    self.inputs.test_class_labels,
                    strict=True,
                )
            )
        )
        per_record_estimate = estimate_p10_peak_bytes(1000)
        for module in (production_d2, verifier_d2):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    module._bounded_process_count(
                        requested_workers=16,
                        point_counts=(1000, 1000, 1000),
                        memory_budget_bytes=2 * per_record_estimate,
                    ),
                    2,
                )
                records = module._run_d2_record_jobs(
                    jobs,
                    sweep=self.sweep,
                    phase1_config=self.phase1_config,
                    config=self.config,
                    worker_count=2,
                )
                self.assertEqual(
                    [record["record_order"] for record in records],
                    [0, 1, 2, 3],
                )
                self.assertTrue(
                    all(record["_worker_pid"] != os.getpid() for record in records)
                )

    def test_build_writes_exact_inventory_and_verifier_is_independent(self) -> None:
        summary = build_phase4_d2_eligibility_from_inputs(
            self.output_path,
            inputs=self.inputs,
            sweep=self.sweep,
            phase1_config=self.phase1_config,
            config=self.config,
            worker_count=2,
        )
        self.assertEqual(summary.status, "fail")
        self.assertEqual(summary.test_record_count, 4)
        self.assertEqual(summary.model_cell_count, 6)
        self.assertEqual(summary.path, self.output_path)

        names = sorted(path.name for path in self.output_path.iterdir() if path.is_file())
        self.assertEqual(
            names,
            sorted(ARTIFACT_STATIC_FILES + ("failed.json",)),
        )

        sha256sums = (self.output_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(sha256sums), len(ARTIFACT_STATIC_FILES))

        for row_name in ("metric_statuses.jsonl", "cwt_receipts.jsonl"):
            first_line = (self.output_path / row_name).read_text(encoding="utf-8").splitlines()[0]
            payload = json.loads(first_line)
            self.assertNotIn("metric_value", payload)
            self.assertNotIn("peak_count", payload)
            self.assertNotIn("peaks", payload)

        verifier_summary = verify_phase4_d2_eligibility_from_inputs(
            self.output_path,
            inputs=self.inputs,
            sweep=self.sweep,
            phase1_config=self.phase1_config,
            config_path=self.output_path / "config.json",
            worker_count=1,
        )
        self.assertEqual(verifier_summary.status, "fail")

        verifier_path = ROOT / "rpe/runner/phase4_d2_eligibility_verifier.py"
        tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase4_d2_eligibility":
                self.fail("verifier must not import any symbol from phase4_d2_eligibility.py")

    def test_build_serializes_real_operator_receipts_and_metric_results(self) -> None:
        """Fails if D2 substitutes record-id templates for real Phase-1 science."""
        summary = build_phase4_d2_eligibility_from_inputs(
            self.output_path,
            inputs=self.inputs,
            sweep=self.sweep,
            phase1_config=self.phase1_config,
            config=self.config,
            worker_count=2,
        )
        self.assertEqual(summary.path, self.output_path)
        spectrum = self.inputs.test_spectra[0]
        record_id = spectrum.spectrum_id.split("::", 1)[1]
        source = _phase1_source_for_oracle(spectrum, record_id=record_id, class_label=0)
        p10_admission = P10MemoryAdmission(64 * 2**30)
        oracle_cells = {
            perturbation_id: run_perturbation_cell(
                source,
                perturbation_id,
                self.phase1_config,
                self.sweep,
                p10_admission=p10_admission if perturbation_id == "p10" else None,
            )
            for perturbation_id in ("p08", "p11")
        }
        serialized_cells = [
            json.loads(line)
            for line in (self.output_path / "cells.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for perturbation_id, oracle in oracle_cells.items():
            row = next(
                item for item in serialized_cells
                if item["record_id"] == record_id and item["perturbation_id"] == perturbation_id
            )
            self.assertEqual(row["state"], oracle.status.value)
            self.assertEqual(row["state_digest"], oracle.state.state_digest)
            self.assertEqual(row["native_gate"], json.loads(_canonical(dict(oracle.evidence.native_gate))))
            self.assertEqual(
                [item["output_spectrum_id"] for item in row["outputs"]],
                [item.result.output.spectrum_id for item in oracle.records],
            )

        oracle_p08 = oracle_cells["p08"]
        alpha_record = next(item for item in oracle_p08.records if item.alpha == 0.05)
        condition_id = f"p08:{_hex_alpha(0.05)}"
        condition = next(
            json.loads(line)
            for line in (self.output_path / "record_conditions.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["record_id"] == record_id and json.loads(line)["condition_id"] == condition_id
        )
        self.assertEqual(condition["output_axis_sha256"], _array_sha(alpha_record.result.output.axis_cm1, dtype="<f8"))
        self.assertEqual(condition["output_intensity_sha256"], _array_sha(alpha_record.result.output.intensity, dtype="<f8"))
        self.assertEqual(condition["support_sha256"], _sha_bytes(project_d2_support(alpha_record.result.output, self.config).tobytes(order="C")))

        mse = evaluate_metric(MSEMetric(), SpectrumPairInput(spectrum, alpha_record.result.output))
        mse_row = next(
            json.loads(line)
            for line in (self.output_path / "metric_statuses.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["record_id"] == record_id
            and json.loads(line)["condition_id"] == condition_id
            and json.loads(line)["metric_output_id"] == "mse"
        )
        self.assertEqual(mse_row["state"], "complete")
        self.assertEqual(mse_row["result_sha256"], _sha_bytes(_canonical_science(mse)))

        cwt_system = _build_cwt_system()
        reference_receipt = run_peak_detection_system(cwt_system, spectrum)
        candidate_receipt = run_peak_detection_system(cwt_system, alpha_record.result.output)
        peak_result = evaluate_metric(
            PeakDetectionCurvesMetric(),
            PeakPairInput(
                reference_peaks=tuple(peak.to_peak1d() for peak in reference_receipt.peaks),
                candidate_peaks=tuple(peak.to_peak1d() for peak in candidate_receipt.peaks),
                position_tolerance_cm1=2.0,
                prominence_thresholds=(0.0,),
            ),
        )
        structure_row = next(
            json.loads(line)
            for line in (self.output_path / "metric_statuses.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["record_id"] == record_id
            and json.loads(line)["condition_id"] == condition_id
            and json.loads(line)["metric_output_id"] == "precision"
        )
        self.assertEqual(structure_row["result_sha256"], _sha_bytes(_canonical_science(peak_result)))

        old_threshold_result = evaluate_metric(
            PeakDetectionCurvesMetric(),
            PeakPairInput(
                reference_peaks=tuple(peak.to_peak1d() for peak in reference_receipt.peaks),
                candidate_peaks=tuple(peak.to_peak1d() for peak in candidate_receipt.peaks),
                position_tolerance_cm1=2.0,
                prominence_thresholds=(0.05, 0.1, 0.2),
            ),
        )
        self.assertNotEqual(
            structure_row["result_sha256"],
            _sha_bytes(_canonical_science(old_threshold_result)),
        )

    def test_selection_replay_rejects_rehashed_wrong_deterministic_ids(self) -> None:
        artifact = json.loads(REAL_SELECTION_ARTIFACT_PATH.read_text(encoding="utf-8"))
        mutated = json.loads(_canonical(artifact))
        first = mutated["selections"][0]["classes"][0]
        first["ordered_train_record_ids"][0], first["ordered_train_record_ids"][1] = (
            first["ordered_train_record_ids"][1],
            first["ordered_train_record_ids"][0],
        )
        for shot in (5, 10, 20):
            ids = first["ordered_train_record_ids"][:shot]
            first["train_record_ids"][str(shot)] = ids
            first["train_record_ids_sha256"][str(shot)] = _ids_digest(ids)
        seed = mutated["selections"][0]
        for shot in (5, 10, 20):
            all_ids = [
                item
                for class_doc in seed["classes"]
                for item in class_doc["train_record_ids"][str(shot)]
            ]
            seed["train_record_ids_sha256"][str(shot)] = _ids_digest(all_ids)
        with self.assertRaisesRegex(D2SelectionValidationError, "deterministic selection"):
            validate_d2_few_shot_selection(
                mutated, SELECTION_CONFIG_PATH, ROOT / "data/unified/bacteria_id_reference"
            )

    def test_build_is_content_derived_across_distinct_output_directories(self) -> None:
        left = self.root / "left-output"
        right = self.root / "right-output"

        left_summary = build_phase4_d2_eligibility_from_inputs(
            left,
            inputs=self.inputs,
            sweep=self.sweep,
            phase1_config=self.phase1_config,
            config=self.config,
            worker_count=2,
        )
        right_summary = build_phase4_d2_eligibility_from_inputs(
            right,
            inputs=self.inputs,
            sweep=self.sweep,
            phase1_config=self.phase1_config,
            config=self.config,
            worker_count=2,
        )

        self.assertEqual(left_summary.run_id, right_summary.run_id)
        self.assertEqual(
            (left / "SHA256SUMS").read_bytes(),
            (right / "SHA256SUMS").read_bytes(),
        )

    def test_gate_is_invariant_to_test_record_id_labels(self) -> None:
        renamed_dataset, renamed_selection = _rewrite_test_record_ids(
            _synthetic_dataset_document(),
            _synthetic_selection_document(),
            prefix="probe",
        )
        renamed_dataset_path = self.root / "renamed-dataset.json"
        renamed_selection_path = self.root / "renamed-selection.json"
        renamed_output_path = self.root / "renamed-output"
        _write_json(renamed_dataset_path, renamed_dataset)
        _write_json(renamed_selection_path, renamed_selection)
        renamed_inputs = reconstruct_d2_eligibility_inputs(
            renamed_dataset_path,
            renamed_selection_path,
            self.config,
        )

        build_phase4_d2_eligibility_from_inputs(
            renamed_output_path,
            inputs=renamed_inputs,
            sweep=self.sweep,
            phase1_config=self.phase1_config,
            config=self.config,
            worker_count=2,
        )
        build_phase4_d2_eligibility_from_inputs(
            self.output_path,
            inputs=self.inputs,
            sweep=self.sweep,
            phase1_config=self.phase1_config,
            config=self.config,
            worker_count=2,
        )

        self.assertEqual(
            json.loads((self.output_path / "gate.json").read_text(encoding="utf-8")),
            json.loads((renamed_output_path / "gate.json").read_text(encoding="utf-8")),
        )

    def test_top_level_entrypoints_do_not_depend_on_synthetic_fixture_env_or_runtime_wiring(self) -> None:
        production_tree = ast.parse(
            (ROOT / "rpe/runner/phase4_d2_eligibility.py").read_text(encoding="utf-8")
        )
        verifier_tree = ast.parse(
            (ROOT / "rpe/runner/phase4_d2_eligibility_verifier.py").read_text(encoding="utf-8")
        )
        production_build = _function_node(production_tree, "build_phase4_d2_eligibility")
        verifier_build = _function_node(verifier_tree, "verify_phase4_d2_eligibility")

        for function_node in (production_build, verifier_build):
            constants = {
                node.value
                for node in ast.walk(function_node)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            self.assertNotIn(
                "RPE_D2_ELIGIBILITY_SYNTHETIC_FIXTURE_DIR",
                constants,
            )

    def test_production_and_verifier_require_real_science_entrypoints_not_fake_cell_state(self) -> None:
        production_tree = ast.parse(
            (ROOT / "rpe/runner/phase4_d2_eligibility.py").read_text(encoding="utf-8")
        )
        verifier_tree = ast.parse(
            (ROOT / "rpe/runner/phase4_d2_eligibility_verifier.py").read_text(
                encoding="utf-8"
            )
        )
        for tree in (production_tree, verifier_tree):
            function_names = {
                node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            }
            self.assertNotIn("_cell_state", function_names)
            called_names = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            self.assertIn("run_perturbation_cell", called_names)
            self.assertIn("evaluate_metric", called_names)
            self.assertIn("run_peak_detection_system", called_names)

    def test_cli_surface_is_exact_and_rejects_protocol_skip_and_resample_flags(self) -> None:
        for arguments in (
            ["build", "--output-root", str(self.output_path), "--protocol", "A"],
            ["build", "--output-root", str(self.output_path), "--skip-verify"],
            ["build", "--output-root", str(self.output_path), "--outcome-mode", "full"],
            ["verify", "--run-path", str(self.output_path), "--resample-count", "10"],
        ):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/run_phase4_d2_eligibility.py"), *arguments],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("unrecognized arguments", completed.stderr)


if __name__ == "__main__":
    unittest.main()
