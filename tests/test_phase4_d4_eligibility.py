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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from rpe.downstream.sugar_quantitative import (  # noqa: E402
    D4SugarCohort,
    D4WellSplit,
)
from rpe.evaluation import Spectrum1D  # noqa: E402

# RED checkpoint: production does not exist yet.
import rpe.runner.phase4_d4_eligibility as d4  # noqa: E402,F401
import rpe.runner.phase4_d4_eligibility_verifier as verifier_d4  # noqa: E402,F401
from tools.run_phase4_d4_eligibility import main as d4_cli_main  # noqa: E402,F401


SWEEP_PATH = ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_PATH = ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
REAL_CONFIG_PATH = (
    ROOT / "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
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


def _synthetic_native_spectrum(*, record_id: str, axis: np.ndarray, scale: float) -> Spectrum1D:
    axis_f8 = np.asarray(axis, dtype="<f8")
    intensity_f8 = np.asarray(_native_intensity_f32(axis, scale=scale), dtype="<f8")
    return Spectrum1D(
        spectrum_id=f"d4_sugar_low_snr::synthetic::{record_id}",
        sample_id=None,
        axis_cm1=axis_f8,
        intensity=intensity_f8,
    )


def _synthetic_cohort() -> D4SugarCohort:
    axis = _native_axis_f32()
    wells = tuple(f"A{index}_1" for index in range(1, 6))
    record_ids = tuple(f"mix-{index:04d}" for index in range(len(wells)))
    source_members = tuple(f"synthetic/{record_id}.csv" for record_id in record_ids)
    intensity = _read_only(
        np.stack(
            [_native_intensity_f32(axis, scale=float(index)) for index in range(len(wells))],
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
    rounds = _read_only(np.ones(len(wells), dtype="<i8"))
    repetitions = _read_only(np.ones(len(wells), dtype="<i8"))

    blank_record_ids = ("blank-0000",)
    blank_well_ids = ("E1_3",)
    blank_members = ("synthetic/blank.csv",)
    blank_intensity = _read_only(
        np.stack([_native_intensity_f32(axis, scale=99.0)], axis=0)
    )
    blank_targets = _read_only(np.zeros((1, 4), dtype="<f8"))

    folds = tuple(_read_only(np.asarray([index], dtype="<i8")) for index in range(5))
    splits: list[D4WellSplit] = []
    for seed in range(5):
        test_fold = seed
        validation_fold = (seed + 1) % 5
        train_folds = tuple(fold for fold in range(5) if fold not in (test_fold, validation_fold))
        train = np.asarray(train_folds, dtype="<i8")
        splits.append(
            D4WellSplit(
                seed=seed,
                train_indices=_read_only(train),
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
        rounds=rounds,
        repetitions=repetitions,
        blank_intensity=blank_intensity,
        blank_targets=blank_targets,
        blank_record_ids=blank_record_ids,
        blank_well_ids=blank_well_ids,
        blank_source_members=blank_members,
        folds=folds,
        splits=tuple(splits),
    )


def _support_axis(axis: np.ndarray) -> np.ndarray:
    axis_f8 = np.asarray(axis, dtype="<f8")
    return np.asarray(axis_f8[axis_f8 >= (145.83834838867188 - 1e-12)], dtype="<f8")


def _synthetic_config_document() -> dict[str, object]:
    axis = _native_axis_f32()
    support = _support_axis(axis)
    mixture_record_ids = [f"mix-{index:04d}" for index in range(5)]
    blank_record_ids = ["blank-0000"]
    return {
        "schema_version": "phase4-d4-protocol-a-full-domain-eligibility-config-v1",
        "experiment_id": "phase4-d4-protocol-a-full-domain-eligibility-v1",
        "synthetic_fixture": True,
        "alpha_grid": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8],
        "active_perturbation_ids": [
            "p01",
            "p02",
            "p03",
            "p04",
            "p05",
            "p08",
            "p09",
            "p10",
            "p11",
            "p12",
        ],
        "inactive_perturbation_ids": ["p06", "p07"],
        "full_domain_perturbation_ids": ["p08", "p09", "p10", "p11", "p12"],
        "peak_perturbation_ids": ["p01", "p02", "p03", "p04", "p05"],
        "metric_output_ids": [
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
        ],
        "cwt_system_id": "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511",
        "denominators": {
            "mixture_record_count": 5,
            "blank_record_count": 1,
            "mixture_well_count": 5,
            "blank_well_count": 1,
            "model_cell_count": 5,
            "model_role_occurrence_count": 30,
            "records_per_well": 1,
        },
        "frozen_identities": {
            "native_axis_f32_sha256": _array_sha(axis, dtype="<f4"),
            "native_axis_f64_sha256": _array_sha(axis.astype("<f8"), dtype="<f8"),
            "support_axis_f64_sha256": _array_sha(support, dtype="<f8"),
            "support_axis_f32_sha256": _array_sha(support.astype("<f4"), dtype="<f4"),
            "mixture_record_ids_sha256": _ids_digest(mixture_record_ids),
            "blank_record_ids_sha256": _ids_digest(blank_record_ids),
        },
        "support_grid": {
            "point_count": int(support.size),
            "first_cm1": float(support[0]),
            "last_cm1": float(support[-1]),
            "max_in_range_native_gap_cm1": 3.665,
            "coordinates_cm1": [float(value) for value in support],
        },
        "gates": {
            "full_domain_record_fraction": 1.0,
            "full_domain_well_fraction": 1.0,
            "p01_p04_record_fraction": 0.95,
            "p01_p04_well_fraction": 0.95,
            "p05_record_fraction": 0.9,
            "p05_well_fraction": 0.9,
            "peak_common_record_fraction": 0.9,
            "peak_common_well_fraction": 0.9,
        },
        "phase1_native_gate_relative_tolerance": 1e-12,
        "p10": {
            "correlation_length_cm1": 20.0,
            "memory_budget_bytes": 64 * 2**30,
            "peak_estimate_formula": "32*N^2+64*N+2^30",
        },
        "authorities": {},
        "claim_boundary": (
            "outcome_blind_eligibility_only_no_predictions_metrics_peak_lists_alignment_or_inference"
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _function_node(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


class Phase4D4EligibilityImportTest(unittest.TestCase):
    def test_imported(self) -> None:
        self.assertIsNotNone(d4)


class Phase4D4EligibilityApiSurfaceTest(unittest.TestCase):
    def test_public_api_symbols_exist(self) -> None:
        required = (
            "Phase4D4EligibilityConfig",
            "D4EligibilityInputs",
            "Phase4D4EligibilitySummary",
            "Phase4D4EligibilityError",
            "ARTIFACT_STATIC_FILES",
            "ACTIVE_PERTURBATION_IDS",
            "ALL_PERTURBATION_IDS",
            "FULL_DOMAIN_PERTURBATION_IDS",
            "PEAK_PERTURBATION_IDS",
            "METRIC_OUTPUT_IDS",
            "CWT_SYSTEM_ID",
            "parse_phase4_d4_eligibility_config",
            "load_phase4_d4_eligibility_config",
            "reconstruct_d4_eligibility_inputs",
            "project_d4_support",
            "evaluate_d4_eligibility_gates",
            "validate_outcome_blind_d4_payload",
            "build_phase4_d4_eligibility_from_inputs",
            "verify_phase4_d4_eligibility_from_inputs",
            "build_phase4_d4_eligibility",
            "verify_phase4_d4_eligibility",
            "_execute_d4_record",
        )
        missing = [name for name in required if not hasattr(d4, name)]
        self.assertEqual(missing, [])


class Phase4D4EligibilityConfigTest(unittest.TestCase):
    def test_synthetic_config_parses_with_flexible_identity(self) -> None:
        config = d4.parse_phase4_d4_eligibility_config(
            Path("synthetic.json"),
            _canonical(_synthetic_config_document()),
            require_frozen_identity=False,
        )
        self.assertEqual(config.schema_version, "phase4-d4-protocol-a-full-domain-eligibility-config-v1")
        self.assertEqual(config.support_point_count, 1999)
        self.assertEqual(config.mixture_record_count, 5)
        self.assertEqual(config.mixture_well_count, 5)
        self.assertEqual(config.blank_record_count, 1)
        self.assertEqual(config.p10_memory_budget_bytes, 64 * 2**30)

    def test_real_config_has_frozen_d4_identities_and_counts(self) -> None:
        if not REAL_CONFIG_PATH.exists():
            self.skipTest("real D4 eligibility config does not exist yet")
        config = d4.load_phase4_d4_eligibility_config(REAL_CONFIG_PATH)
        self.assertEqual(config.schema_version, "phase4-d4-protocol-a-full-domain-eligibility-config-v1")
        self.assertEqual(config.mixture_record_count, 7680)
        self.assertEqual(config.blank_record_count, 32)
        self.assertEqual(config.mixture_well_count, 240)
        self.assertEqual(config.support_point_count, 1999)
        self.assertEqual(config.support_max_gap_cm1, 3.665)
        self.assertEqual(config.p10_memory_budget_bytes, 64 * 2**30)
        self.assertEqual(config.model_role_occurrence_count, 38560)
        self.assertEqual(
            config.native_axis_float32_sha256,
            "9b0b88641a767abc74439e21539f7c41366adcf41c9bc3ab60d6de91021431d7",
        )
        self.assertEqual(
            config.support_axis_float64_sha256,
            "db910b11f92151db481391e96b8a06e246140596cfd4abad64764b87d2d84ee5",
        )

    def test_real_config_requires_complete_authority_code_and_environment_receipts(self) -> None:
        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIn("code_authority", document)
        self.assertIn("environment_authority", document)
        self.assertIn("trust_anchor", document)
        self.assertTrue(document["authorities"])
        self.assertTrue(document["code_authority"])
        self.assertTrue(document["environment_authority"])
        expected_code_paths = {
            "rpe/downstream/sugar_quantitative.py",
            "rpe/evaluation/contracts.py",
            "rpe/methods/catalog.py",
            "rpe/methods/classical/peaks.py",
            "rpe/metrics/fidelity.py",
            "rpe/metrics/peak.py",
            "rpe/metrics/reference_free.py",
            "rpe/metrics/transport.py",
            "rpe/perturb/axis_transform.py",
            "rpe/perturb/baseline_distortion.py",
            "rpe/perturb/contracts.py",
            "rpe/perturb/correlated_noise.py",
            "rpe/perturb/gaussian_noise.py",
            "rpe/perturb/peak_family.py",
            "rpe/perturb/sweep.py",
            "rpe/runner/phase1_config.py",
            "rpe/runner/phase1_gates.py",
            "rpe/runner/phase1_perturbations.py",
            "rpe/runner/phase1_selection.py",
            "rpe/runner/phase1_types.py",
            "rpe/runner/phase4_d4_eligibility.py",
            "rpe/runner/phase4_d4_eligibility_verifier.py",
            "tests/test_phase4_d4_eligibility.py",
            "tools/run_phase4_d4_eligibility.py",
        }
        self.assertEqual(set(document["code_authority"]), expected_code_paths)
        for relative_path in expected_code_paths:
            target = ROOT / relative_path
            self.assertEqual(
                document["code_authority"][relative_path],
                {"bytes": target.stat().st_size, "sha256": _sha_file(target)},
            )
        self.assertEqual(
            document["environment_authority"],
            {
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
                    "rpe/runner/phase4_d4_eligibility_authority.py",
                "config_binds_authority": False,
                "direction": "authority_to_config_only",
            },
        )

        mutations = (
            ("authorities", {}),
            ("code_authority", {}),
            ("environment_authority", {}),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                mutated = json.loads(json.dumps(document))
                mutated[field] = replacement
                raw = _canonical(mutated)
                with self.assertRaisesRegex(d4.Phase4D4EligibilityError, field):
                    d4.parse_phase4_d4_eligibility_config(
                        REAL_CONFIG_PATH, raw, require_frozen_identity=False
                    )
                with self.assertRaisesRegex(
                    verifier_d4.Phase4D4EligibilityVerifierError, field
                ):
                    verifier_d4.parse_phase4_d4_eligibility_config(
                        REAL_CONFIG_PATH, raw, require_frozen_identity=False
                    )


class Phase4D4EligibilityInputsAndSupportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "config.json"
        _write_json(self.config_path, _synthetic_config_document())
        self.config = d4.parse_phase4_d4_eligibility_config(
            Path("synthetic.json"),
            _canonical(_synthetic_config_document()),
            require_frozen_identity=False,
        )
        self.cohort = _synthetic_cohort()

    def test_reconstructs_record_and_whole_well_ledgers(self) -> None:
        inputs = d4.reconstruct_d4_eligibility_inputs(self.cohort, self.config)
        self.assertEqual(len(inputs.mixture_spectra), 5)
        self.assertEqual(len(inputs.blank_spectra), 1)
        self.assertEqual(len(inputs.well_ids), 5)
        self.assertEqual(inputs.support_axis_cm1.size, 1999)
        self.assertEqual(inputs.mixture_record_ids_sha256, _ids_digest(list(self.cohort.record_ids)))

        # Each of the 5 cells: test fold s, validation fold (s+1)%5, 3 folds train.
        self.assertEqual(len(inputs.model_cells), 5)
        self.assertEqual(len(inputs.model_role_occurrences), 30)

    def test_gate_output_is_json_serializable(self) -> None:
        config = d4.parse_phase4_d4_eligibility_config(
            Path("synthetic.json"),
            _canonical(_synthetic_config_document()),
            require_frozen_identity=False,
        )
        cells = []
        for record_id, well_id in zip((f"mix-{index:04d}" for index in range(5)), (f"A{index}_1" for index in range(1, 6)), strict=True):
            for perturbation_id in d4.ALL_PERTURBATION_IDS:
                cells.append(
                    {
                        "record_id": record_id,
                        "well_id": well_id,
                        "perturbation_id": perturbation_id,
                        "state": "structurally_ineligible" if perturbation_id in {"p06", "p07"} else "complete",
                    }
                )
        conditions = [
            {
                "record_id": record_id,
                "well_id": well_id,
                "condition_id": condition_id,
                "state": "complete",
                "metrics": {
                    metric_id: {
                        "state": "complete",
                        "diagnostics_sha256": "0" * 64,
                        "result_sha256": "1" * 64,
                    }
                    for metric_id in d4.METRIC_OUTPUT_IDS
                },
                "cwt": {
                    "state": "complete",
                    "diagnostics_sha256": "2" * 64,
                    "peak_list_sha256": "3" * 64,
                    "warning_sha256": "4" * 64,
                },
            }
            for record_id, well_id in zip((f"mix-{index:04d}" for index in range(5)), (f"A{index}_1" for index in range(1, 6)), strict=True)
            for condition_id in (
                "alpha0",
                *(
                    f"{perturbation_id}:{_hex_alpha(alpha)}"
                    for perturbation_id in d4.FULL_DOMAIN_PERTURBATION_IDS
                    for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
                ),
            )
        ]
        gate = d4.evaluate_d4_eligibility_gates(
            cells=cells,
            mixture_record_ids=[f"mix-{index:04d}" for index in range(5)],
            mixture_well_ids=[f"A{index}_1" for index in range(1, 6)],
            config=config,
            record_conditions=conditions,
            blank_conditions=[],
        )
        encoded = _canonical(gate)
        self.assertIn(b"overall_status", encoded)

    def test_project_support_is_1999_point_direct_selection_for_alpha_zero(self) -> None:
        spectrum = _synthetic_native_spectrum(record_id="mix-0000", axis=_native_axis_f32(), scale=0.0)
        projected = d4.project_d4_support(spectrum, self.config)
        self.assertEqual(projected.dtype, np.dtype("<f4"))
        self.assertTrue(
            np.array_equal(
                projected,
                np.asarray(spectrum.intensity[1:], dtype="<f4"),
            )
        )


class Phase4D4EligibilityGatesAndPayloadTest(unittest.TestCase):
    def test_partial_acquisition_well_is_not_counted_as_complete(self) -> None:
        document = _synthetic_config_document()
        document["denominators"]["mixture_record_count"] = 2
        document["denominators"]["mixture_well_count"] = 1
        document["denominators"]["records_per_well"] = 2
        config = d4.parse_phase4_d4_eligibility_config(
            Path("synthetic.json"),
            _canonical(document),
            require_frozen_identity=False,
        )
        record_ids = ["mix-a", "mix-b"]
        cells = []
        for record_id in record_ids:
            for perturbation_id in d4.ALL_PERTURBATION_IDS:
                state = (
                    "structurally_ineligible"
                    if perturbation_id in {"p06", "p07"}
                    else "complete"
                )
                if record_id == "mix-b" and perturbation_id == "p01":
                    state = "not_applicable"
                cells.append(
                    {
                        "record_id": record_id,
                        "well_id": "well-a",
                        "perturbation_id": perturbation_id,
                        "state": state,
                    }
                )
        conditions = [
            {
                "record_id": record_id,
                "well_id": "well-a",
                "condition_id": condition_id,
                "state": "complete",
                "metrics": {
                    metric_id: {"state": "complete"}
                    for metric_id in d4.METRIC_OUTPUT_IDS
                },
                "cwt": {"state": "complete"},
            }
            for record_id in record_ids
            for condition_id in (
                "alpha0",
                *(
                    f"{perturbation_id}:{_hex_alpha(alpha)}"
                    for perturbation_id in d4.FULL_DOMAIN_PERTURBATION_IDS
                    for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
                ),
            )
        ]

        gate = d4.evaluate_d4_eligibility_gates(
            cells=cells,
            mixture_record_ids=record_ids,
            mixture_well_ids=["well-a"],
            config=config,
            record_conditions=conditions,
            blank_conditions=[],
        )

        self.assertEqual(
            gate["peak_common_support"]["p01"]["complete_well_count"], 0
        )
        self.assertEqual(gate["peak_common_support"]["common_well_count"], 0)

    def test_gate_preserves_full_domain_when_peak_common_support_fails(self) -> None:
        config = d4.parse_phase4_d4_eligibility_config(
            Path("synthetic.json"),
            _canonical(_synthetic_config_document()),
            require_frozen_identity=False,
        )
        mixture_record_ids = [f"mix-{index:04d}" for index in range(5)]
        well_ids = [f"A{index}_1" for index in range(1, 6)]
        cells = []
        for record_id, well_id in zip(mixture_record_ids, well_ids, strict=True):
            for perturbation_id in d4.ALL_PERTURBATION_IDS:
                state = "complete"
                if perturbation_id in {"p06", "p07"}:
                    state = "structurally_ineligible"
                if perturbation_id == "p05" and record_id == "mix-0000":
                    state = "not_applicable"
                cells.append(
                    {
                        "record_id": record_id,
                        "well_id": well_id,
                        "perturbation_id": perturbation_id,
                        "state": state,
                    }
                )
        conditions = [
            {
                "record_id": record_id,
                "well_id": well_id,
                "condition_id": condition_id,
                "state": "complete",
                "metrics": {metric_id: {"state": "complete"} for metric_id in d4.METRIC_OUTPUT_IDS},
                "cwt": {"state": "complete"},
            }
            for record_id, well_id in zip(mixture_record_ids, well_ids, strict=True)
            for condition_id in (
                "alpha0",
                *(
                    f"{perturbation_id}:{_hex_alpha(alpha)}"
                    for perturbation_id in d4.FULL_DOMAIN_PERTURBATION_IDS
                    for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
                ),
            )
        ]
        gate = d4.evaluate_d4_eligibility_gates(
            cells=cells,
            mixture_record_ids=mixture_record_ids,
            mixture_well_ids=well_ids,
            config=config,
            record_conditions=conditions,
            blank_conditions=[],
        )
        self.assertEqual(gate["full_domain_core"]["state"], "evaluable")
        self.assertEqual(gate["peak_common_support"]["state"], "not_evaluable_coverage")
        self.assertEqual(gate["overall_status"], "fail")
        self.assertEqual(gate["marker_filename"], "failed.json")

    def test_outcome_blind_payload_rejects_metric_magnitudes_and_peak_lists(self) -> None:
        valid = {
            "record_id": "mix-0000",
            "well_id": "A1_1",
            "condition_id": f"p08:{_hex_alpha(0.05)}",
            "state": "complete",
            "metrics": {metric_id: {"state": "complete"} for metric_id in d4.METRIC_OUTPUT_IDS},
            "cwt": {"state": "complete"},
        }
        d4.validate_outcome_blind_d4_payload(valid)

        invalid = json.loads(_canonical(valid))
        invalid["metrics"]["mse"]["metric_value"] = 1.0
        with self.assertRaisesRegex(d4.Phase4D4EligibilityError, "metric_value"):
            d4.validate_outcome_blind_d4_payload(invalid)

        invalid = json.loads(_canonical(valid))
        invalid["cwt"]["peaks"] = []
        with self.assertRaisesRegex(d4.Phase4D4EligibilityError, "peaks"):
            d4.validate_outcome_blind_d4_payload(invalid)


class Phase4D4EligibilityBuildVerifyAndCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.left_output = self.root / "left"
        self.right_output = self.root / "right"
        self.config_path = self.root / "config.json"
        _write_json(self.config_path, _synthetic_config_document())
        self.config = d4.parse_phase4_d4_eligibility_config(
            Path("synthetic.json"),
            _canonical(_synthetic_config_document()),
            require_frozen_identity=False,
        )
        self.inputs = d4.reconstruct_d4_eligibility_inputs(_synthetic_cohort(), self.config)

    def test_build_rejects_existing_output_path(self) -> None:
        self.left_output.mkdir(parents=True)
        with self.assertRaisesRegex(d4.Phase4D4EligibilityError, "append-only"):
            d4.build_phase4_d4_eligibility_from_inputs(
                self.left_output,
                inputs=self.inputs,
                sweep_path=SWEEP_PATH,
                phase1_config_path=PHASE1_CONFIG_PATH,
                config=self.config,
                worker_count=1,
            )

    def test_build_writes_15_file_inventory_and_is_deterministic_across_dirs(self) -> None:
        left = d4.build_phase4_d4_eligibility_from_inputs(
            self.left_output,
            inputs=self.inputs,
            sweep_path=SWEEP_PATH,
            phase1_config_path=PHASE1_CONFIG_PATH,
            config=self.config,
            worker_count=1,
        )
        right = d4.build_phase4_d4_eligibility_from_inputs(
            self.right_output,
            inputs=self.inputs,
            sweep_path=SWEEP_PATH,
            phase1_config_path=PHASE1_CONFIG_PATH,
            config=self.config,
            worker_count=2,
        )
        self.assertEqual(left.run_id, right.run_id)
        self.assertEqual(
            (self.left_output / "SHA256SUMS").read_bytes(),
            (self.right_output / "SHA256SUMS").read_bytes(),
        )
        names = sorted(path.name for path in self.left_output.iterdir() if path.is_file())
        self.assertEqual(
            names,
            sorted(tuple(d4.ARTIFACT_STATIC_FILES) + (left.marker_filename,)),
        )
        sha256sums = (self.left_output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(sha256sums), len(d4.ARTIFACT_STATIC_FILES))

    def test_blank_alpha_zero_condition_is_emitted_once_per_record(self) -> None:
        d4.build_phase4_d4_eligibility_from_inputs(
            self.left_output,
            inputs=self.inputs,
            sweep_path=SWEEP_PATH,
            phase1_config_path=PHASE1_CONFIG_PATH,
            config=self.config,
            worker_count=1,
        )
        rows = [
            json.loads(line)
            for line in (self.left_output / "blank_conditions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        alpha_zero_rows = [row for row in rows if row["condition_id"] == "alpha0"]
        self.assertEqual(len(alpha_zero_rows), len(self.inputs.blank_record_ids))
        self.assertEqual(
            len(rows),
            len(self.inputs.blank_record_ids)
            * (1 + len(d4.FULL_DOMAIN_PERTURBATION_IDS) * 8),
        )

    def test_verifier_is_independent_and_has_no_bypass_imports(self) -> None:
        verifier_path = ROOT / "rpe/runner/phase4_d4_eligibility_verifier.py"
        tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "rpe.runner.phase4_d4_eligibility"
            ):
                self.fail("verifier must not import any symbol from phase4_d4_eligibility.py")

        verifier_build = _function_node(tree, "verify_phase4_d4_eligibility")
        called_names = {
            node.func.id
            for node in ast.walk(verifier_build)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("_skip_verify", called_names)

    def test_cli_surface_is_exact_and_rejects_extra_flags(self) -> None:
        cli_path = ROOT / "tools/run_phase4_d4_eligibility.py"
        if not cli_path.exists():
            self.skipTest("D4 eligibility CLI does not exist yet")
        for arguments in (
            ["build", "--output-root", str(self.left_output), "--protocol", "A"],
            ["build", "--output-root", str(self.left_output), "--skip-verify"],
            ["verify", "--run-path", str(self.left_output), "--resample-count", "10"],
        ):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [str(ROOT / ".venv/bin/python"), str(cli_path), *arguments],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("unrecognized arguments", completed.stderr)


if __name__ == "__main__":
    unittest.main()

