from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import inspect
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SUBJECT_MODULE = "rpe.runner.phase4_d4_protocol_b"
VERIFIER_MODULE = "rpe.runner.phase4_d4_protocol_b_verifier"
CLI_MODULE = "tools.run_phase4_d4_protocol_b"
AUTHORITY_MODULE = "rpe.runner.phase4_d4_protocol_b_authority"

SUBJECT_FILE = ROOT / "rpe/runner/phase4_d4_protocol_b.py"
VERIFIER_FILE = ROOT / "rpe/runner/phase4_d4_protocol_b_verifier.py"
AUTHORITY_FILE = ROOT / "rpe/runner/phase4_d4_protocol_b_authority.py"
CLI_FILE = ROOT / "tools/run_phase4_d4_protocol_b.py"
REAL_CONFIG_PATH = ROOT / "experiments/phase4/configs/d4_protocol_b_full_domain_v1.json"
DESIGN_REPORT_PATH = (
    ROOT / "reports/phase4/step32_d4_protocol_b_full_domain_outcome_design.md"
)
PLAN_PATH = (
    ROOT / "docs/superpowers/plans/2026-08-26-phase4-d4-protocol-b-full-domain.md"
)

EXPECTED_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "alpha0_equivalence.json",
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "predictions.jsonl",
    "blank_predictions.jsonl",
    "well_conditions.jsonl",
    "technical_lod_loq.jsonl",
    "condition_summary.csv",
    "well_observations.jsonl",
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "figure1_d4_protocol_b_full_domain.png",
    "figure1_d4_protocol_b_full_domain.svg",
    "figure1_d4_protocol_b_full_domain_data.csv",
    "figure2_d4_protocol_b_full_domain.png",
    "figure2_d4_protocol_b_full_domain.svg",
    "figure2_d4_protocol_b_full_domain_data.csv",
    "d4_protocol_b_full_domain_secondary_table.csv",
    "manifest.json",
)

EXPECTED_MODEL_CELL_COUNT = 205
EXPECTED_VALIDATION_ROW_COUNT = 1025
EXPECTED_PREDICTION_ROW_COUNT = 314_880
EXPECTED_BLANK_PREDICTION_ROW_COUNT = 6_560
EXPECTED_TECHNICAL_LOD_LOQ_ROW_COUNT = 820
EXPECTED_WELL_CONDITION_ROW_COUNT = 9_840
EXPECTED_WELL_OBSERVATION_ROW_COUNT = 124_800
EXPECTED_ALIGNMENT_ROW_COUNT = 13
EXPECTED_BOOTSTRAP_ROW_COUNT = 12
EXPECTED_SIGN_FLIP_ROW_COUNT = 24
EXPECTED_HOLM_ROW_COUNT = 24
EXPECTED_CONDITION_SUMMARY_ROW_COUNT = 41
EXPECTED_FIGURE1_ROW_COUNT = 520
EXPECTED_FIGURE2_ROW_COUNT = 13
EXPECTED_TABLE_ROW_COUNT = 13

EXPECTED_MEASUREMENT_BRIDGE_SHA256 = (
    "1e0dac3d3ab54ca1e0faab1b55f66be85c3c51a181661ed0570366ac13a744c9"
)
EXPECTED_ALPHA0_MODEL_SHA256 = (
    "dcebe39171c6855570b0b3c52e5039e76de5cd12ccabba4bac123cd217e9afca"
)
EXPECTED_ALPHA0_VALIDATION_SHA256 = (
    "c94b1a28464e5ff36e14d40ac2443544a4d89f7095405146e710cb9a5b6eb215"
)
EXPECTED_ALPHA0_PREDICTION_SHA256 = (
    "2479d66588c660f14eb18b47987bbda74238c2efeb7c6d34151cfb60b4524ce4"
)
EXPECTED_ALPHA0_BLANK_SHA256 = (
    "ac81670f74a8daac9e46ab2bd541c97b269e4413e70f261cb77792f146f68a03"
)
EXPECTED_ALPHA0_LOD_SHA256 = (
    "d7f14517e8acd944577423c48294b93f0c828bccf60b6b6097882ddb826dd83a"
)

STEP27_RUN_ID = (
    "phase4-d4-protocol-a-full-domain-eligibility-"
    "0f45ae5a0815ecb852b410549ad4582c4916dbe16fa0e3e20e2079f7d088097f"
)
STEP29_RUN_ID = (
    "phase4-d4-protocol-a-full-domain-"
    "1ea534006fcaaecda614d7fbd0f4b4931d532a3f8979f0b7d7ac249025cbb2f7"
)
STEP31_RUN_ID = (
    "phase4-d4-protocol-b-all-role-eligibility-"
    "031be2fb81d1eef7bb88b23ae099320c8b22070d577738bdbc04e42430eb814d"
)

ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
METRIC_OUTPUT_IDS = (
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


def _hex_alpha(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


CONDITION_IDS = ("alpha0",) + tuple(
    f"{perturbation}:{_hex_alpha(alpha)}"
    for perturbation in PERTURBATIONS
    for alpha in ALPHAS[1:]
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _rewrite_sha256sums(path: Path) -> None:
    terminal_name = "complete.json" if (path / "complete.json").exists() else "failed.json"
    ordered = [*EXPECTED_PAYLOAD_FILES, terminal_name]
    rows = [
        f"{_sha256_file(path / name)}  {name}"
        for name in ordered
    ]
    (path / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


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


def _subject_surface_available() -> bool:
    return _has_spec(SUBJECT_MODULE) and REAL_CONFIG_PATH.is_file()


def _full_surface_available() -> bool:
    return (
        _subject_surface_available()
        and _has_spec(VERIFIER_MODULE)
        and _has_spec(CLI_MODULE)
        and _has_spec(AUTHORITY_MODULE)
        and VERIFIER_FILE.is_file()
        and CLI_FILE.is_file()
        and AUTHORITY_FILE.is_file()
    )


def _call_supported(function, /, *args, **kwargs):
    signature = inspect.signature(function)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return function(*args, **kwargs)
    filtered = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return function(*args, **filtered)


def _make_synthetic_inputs(subject, **overrides):
    return _call_supported(
        subject.make_synthetic_d4_protocol_b_inputs,
        well_count=5,
        acquisitions_per_well=2,
        model_folds=5,
        feature_count=8,
        blank_record_count=2,
        condition_ids=CONDITION_IDS,
        **overrides,
    )


def _make_synthetic_config(subject, inputs, **overrides):
    return _call_supported(
        subject.make_synthetic_d4_protocol_b_config,
        inputs,
        condition_ids=CONDITION_IDS,
        artifact_payload_files=EXPECTED_PAYLOAD_FILES,
        measurement_bridge_sha256=EXPECTED_MEASUREMENT_BRIDGE_SHA256,
        alpha0_expected_digests={
            "model_digest": EXPECTED_ALPHA0_MODEL_SHA256,
            "validation_digest": EXPECTED_ALPHA0_VALIDATION_SHA256,
            "prediction_digest": EXPECTED_ALPHA0_PREDICTION_SHA256,
            "blank_prediction_digest": EXPECTED_ALPHA0_BLANK_SHA256,
            "technical_lod_loq_digest": EXPECTED_ALPHA0_LOD_SHA256,
        },
        **overrides,
    )


def _build_synthetic(subject, *, inputs=None, config=None, worker_count=1):
    inputs = _make_synthetic_inputs(subject) if inputs is None else inputs
    config = _make_synthetic_config(subject, inputs) if config is None else config
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "artifact"
        summary = subject.build_phase4_d4_protocol_b_from_inputs(
            output,
            inputs=inputs,
            config=config,
            rematerialization_worker_count=worker_count,
            model_worker_count=min(worker_count, 2),
            bootstrap_resamples=8,
            sign_flip_resamples=16,
        )
        files = {
            item.name: item.read_bytes()
            for item in summary.path.iterdir()
            if item.is_file()
        }
        return summary, files


def _read_jsonl_bytes(raw: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def _read_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(raw.decode("utf-8").splitlines()))


def _find_authority_receipt(document: dict[str, object], relative_path: str) -> dict[str, object]:
    for receipt in document.get("authorities", {}).values():
        if receipt.get("path") == relative_path:
            return receipt
    raise AssertionError(f"missing authority receipt for {relative_path}")


def _assert_receipt_matches_file(test_case: unittest.TestCase, receipt: dict[str, object]) -> None:
    if "path" in receipt:
        path = ROOT / str(receipt["path"])
        test_case.assertTrue(path.is_file(), path)
        test_case.assertEqual(int(receipt["bytes"]), path.stat().st_size)
        test_case.assertEqual(str(receipt["sha256"]), _sha256_file(path))
        return
    archive_path = ROOT / str(receipt["archive_path"])
    test_case.assertTrue(archive_path.is_file(), archive_path)
    if "member_path" not in receipt:
        return
    import zipfile

    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(str(receipt["member_path"]))
    test_case.assertEqual(int(receipt["bytes"]), len(payload))
    test_case.assertEqual(str(receipt["sha256"]), _sha256_bytes(payload))


class Phase4D4ProtocolBRedCheckpointTest(unittest.TestCase):
    def test_missing_step33_module_config_and_cli(self) -> None:
        missing = []
        if not _has_spec(SUBJECT_MODULE):
            missing.append(SUBJECT_MODULE)
        if not REAL_CONFIG_PATH.is_file():
            missing.append(REAL_CONFIG_PATH.relative_to(ROOT).as_posix())
        if not AUTHORITY_FILE.is_file():
            missing.append(AUTHORITY_FILE.relative_to(ROOT).as_posix())
        if not CLI_FILE.is_file():
            missing.append(CLI_FILE.relative_to(ROOT).as_posix())
        self.assertEqual(
            missing,
            [],
            "awaiting Step 33 D4 Protocol-B module/config/authority/CLI: "
            + ", ".join(missing),
        )

    def test_first_positive_model_lifecycle_failure_emits_fixed_failed_artifact(self) -> None:
        self.assertTrue(
            _subject_surface_available(),
            "awaiting Step 33 D4 Protocol-B build implementation for "
            "failed_model_lifecycle regression",
        )

        subject = _load_subject()
        failure_condition = CONDITION_IDS[1]
        failure_fold = 0
        failure_components = 2
        failure_receipt = {
            "state": "failed_model_lifecycle",
            "condition_id": failure_condition,
            "fold": failure_fold,
            "n_components": failure_components,
            "warning_category": "RuntimeWarning",
            "warning_message": "synthetic first-positive failure",
        }
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        original_fit = subject.fit_d4_protocol_b_condition

        def _side_effect(*args, **kwargs):
            condition_id = kwargs["condition_id"]
            if condition_id == "alpha0":
                return original_fit(*args, **kwargs)
            if condition_id == failure_condition:
                error = subject.Phase4D4ProtocolBError("synthetic first-positive failure")
                error.receipt = dict(failure_receipt)
                raise error
            return original_fit(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            with mock.patch.object(
                subject,
                "fit_d4_protocol_b_condition",
                side_effect=_side_effect,
            ):
                summary = subject.build_phase4_d4_protocol_b_from_inputs(
                    output,
                    inputs=inputs,
                    config=config,
                    rematerialization_worker_count=1,
                    model_worker_count=1,
                    bootstrap_resamples=8,
                    sign_flip_resamples=16,
                )

            files = {
                item.name: item.read_bytes()
                for item in summary.path.iterdir()
                if item.is_file()
            }
            self.assertEqual(summary.status, "failed")
            self.assertEqual(len(files), 26)
            self.assertIn("failed.json", files)
            self.assertNotIn("complete.json", files)

            ledger = (summary.path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(ledger), 25)
            for line in ledger:
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, _sha256_file(summary.path / name))

            failed_doc = json.loads(files["failed.json"].decode("utf-8"))
            manifest = json.loads(files["manifest.json"].decode("utf-8"))
            authority_bridge = json.loads(files["authority_bridge.json"].decode("utf-8"))
            self.assertEqual(failed_doc["status"], "failed")
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["endpoint_state"], "failed_model_lifecycle")
            self.assertEqual(
                authority_bridge["measurement_bridge"]["state"],
                "not_tested_endpoint_closed",
            )
            self.assertEqual(failed_doc["failure"]["condition_id"], failure_condition)
            self.assertEqual(failed_doc["failure"]["fold"], failure_fold)
            self.assertEqual(failed_doc["failure"]["n_components"], failure_components)

            model_rows = _read_jsonl_bytes(files["model_cells.jsonl"])
            validation_rows = _read_jsonl_bytes(files["validation_scores.jsonl"])
            prediction_rows = _read_jsonl_bytes(files["predictions.jsonl"])
            blank_rows = _read_jsonl_bytes(files["blank_predictions.jsonl"])
            well_rows = _read_jsonl_bytes(files["well_conditions.jsonl"])
            lod_rows = _read_jsonl_bytes(files["technical_lod_loq.jsonl"])
            holm_rows = _read_jsonl_bytes(files["holm_family.jsonl"])

            alpha0_models = [row for row in model_rows if row["condition_id"] == "alpha0"]
            alpha0_validation = [
                row for row in validation_rows if row["condition_id"] == "alpha0"
            ]
            alpha0_predictions = [
                row for row in prediction_rows if row["condition_id"] == "alpha0"
            ]
            self.assertTrue(all(row["state"] == "complete" for row in alpha0_models))
            self.assertTrue(all(row["state"] == "complete" for row in alpha0_validation))
            self.assertTrue(all(row["state"] == "complete" for row in alpha0_predictions))

            failed_models = [
                row
                for row in model_rows
                if row["condition_id"] == failure_condition
                and int(row["fold"]) == failure_fold
                and row["state"] == "failed_model_lifecycle"
            ]
            self.assertEqual(len(failed_models), 1)
            self.assertEqual(
                failed_models[0].get("selected_n_components", failed_models[0].get("n_components")),
                failure_components,
            )

            failed_validation = [
                row
                for row in validation_rows
                if row["condition_id"] == failure_condition
                and int(row["fold"]) == failure_fold
                and int(row["n_components"]) == failure_components
            ]
            self.assertEqual(len(failed_validation), 1)
            self.assertEqual(failed_validation[0]["state"], "failed_model_lifecycle")

            remaining_model_rows = [
                row
                for row in model_rows
                if row["condition_id"] != "alpha0"
                and not (
                    row["condition_id"] == failure_condition
                    and int(row["fold"]) == failure_fold
                )
            ]
            remaining_validation_rows = [
                row
                for row in validation_rows
                if row["condition_id"] != "alpha0"
                and not (
                    row["condition_id"] == failure_condition
                    and int(row["fold"]) == failure_fold
                    and int(row["n_components"]) == failure_components
                )
            ]
            self.assertTrue(
                all(row["state"] == "not_tested_endpoint_closed" for row in remaining_model_rows)
            )
            self.assertTrue(
                all(row["state"] == "not_tested_endpoint_closed" for row in remaining_validation_rows)
            )
            self.assertTrue(
                all(
                    row["state"] == "not_tested_endpoint_closed"
                    for row in prediction_rows
                    if row["condition_id"] != "alpha0"
                )
            )
            self.assertTrue(
                all(
                    row["state"] == "not_tested_endpoint_closed"
                    for row in blank_rows
                    if row["condition_id"] != "alpha0"
                )
            )
            self.assertTrue(
                all(
                    row["state"] == "not_tested_endpoint_closed"
                    for row in well_rows
                    if row["condition_id"] != "alpha0"
                )
            )
            self.assertTrue(
                all(
                    row["state"] == "not_tested_endpoint_closed"
                    for row in lod_rows
                    if row["condition_id"] != "alpha0"
                )
            )
            self.assertEqual(len(holm_rows), EXPECTED_HOLM_ROW_COUNT)
            self.assertTrue(
                all(row["state"] == "not_tested_endpoint_closed" for row in holm_rows)
            )
            self.assertTrue(all(row["raw_p_value"] == 1.0 for row in holm_rows))
            self.assertTrue(all(row["adjusted_p_value"] == 1.0 for row in holm_rows))

            if _has_spec(VERIFIER_MODULE):
                verifier = _load_verifier()
                verify_signature = inspect.signature(
                    verifier.verify_phase4_d4_protocol_b_from_inputs
                )
                injection_name = next(
                    (
                        name
                        for name in (
                            "failure_injector",
                            "model_failure_injector",
                            "model_lifecycle_injector",
                        )
                        if name in verify_signature.parameters
                    ),
                    None,
                )
                if injection_name is not None:
                    def _verifier_injector(**kwargs):
                        if (
                            kwargs.get("condition_id") == failure_condition
                            and kwargs.get("fold") == failure_fold
                            and kwargs.get("n_components") == failure_components
                        ):
                            error = subject.Phase4D4ProtocolBError(
                                "synthetic first-positive failure"
                            )
                            error.receipt = dict(failure_receipt)
                            raise error

                    verified = verifier.verify_phase4_d4_protocol_b_from_inputs(
                        summary.path,
                        inputs=inputs,
                        config_path=summary.path / "config.json",
                        rematerialization_worker_count=1,
                        model_worker_count=1,
                        bootstrap_resamples=8,
                        sign_flip_resamples=16,
                        **{injection_name: _verifier_injector},
                    )
                    self.assertEqual(verified.run_id, summary.run_id)
                    self.assertEqual(verified.status, summary.status)


@unittest.skipUnless(
    _subject_surface_available(),
    "awaiting Step 33 D4 Protocol-B module and config implementation",
)
class Phase4D4ProtocolBConfigParentTest(unittest.TestCase):
    def test_public_api_and_frozen_config_bind_step32(self) -> None:
        subject = _load_subject()
        required = (
            "Phase4D4ProtocolBError",
            "Phase4D4ProtocolBConfig",
            "D4ProtocolBInputs",
            "D4ProtocolBModelCell",
            "Phase4D4ProtocolBSummary",
            "parse_phase4_d4_protocol_b_config",
            "load_phase4_d4_protocol_b_config",
            "make_synthetic_d4_protocol_b_inputs",
            "make_synthetic_d4_protocol_b_config",
            "validate_d4_protocol_b_parent_authorities",
            "reconstruct_d4_protocol_b_inputs",
            "fit_d4_protocol_b_condition",
            "validate_d4_protocol_b_alpha0_equivalence",
            "build_d4_protocol_b_measurement_bridge",
            "aggregate_d4_protocol_b",
            "render_d4_protocol_b_figures",
            "build_phase4_d4_protocol_b_from_inputs",
            "build_phase4_d4_protocol_b",
        )
        self.assertEqual([name for name in required if not hasattr(subject, name)], [])

        config = subject.load_phase4_d4_protocol_b_config(REAL_CONFIG_PATH)
        self.assertEqual(tuple(config.artifact_payload_files), EXPECTED_PAYLOAD_FILES)
        self.assertEqual(config.model_cell_count, EXPECTED_MODEL_CELL_COUNT)
        self.assertEqual(config.validation_row_count, EXPECTED_VALIDATION_ROW_COUNT)
        self.assertEqual(config.prediction_row_count, EXPECTED_PREDICTION_ROW_COUNT)
        self.assertEqual(
            config.blank_prediction_row_count,
            EXPECTED_BLANK_PREDICTION_ROW_COUNT,
        )
        self.assertEqual(
            config.technical_lod_loq_row_count,
            EXPECTED_TECHNICAL_LOD_LOQ_ROW_COUNT,
        )
        self.assertEqual(
            config.well_condition_row_count,
            EXPECTED_WELL_CONDITION_ROW_COUNT,
        )
        self.assertEqual(
            config.well_observation_row_count,
            EXPECTED_WELL_OBSERVATION_ROW_COUNT,
        )
        self.assertEqual(
            config.measurement_bridge_sha256,
            EXPECTED_MEASUREMENT_BRIDGE_SHA256,
        )

        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(tuple(document["artifact_payload_files"]), EXPECTED_PAYLOAD_FILES)
        self.assertEqual(tuple(document["condition_ids"]), CONDITION_IDS)
        self.assertEqual(document["expected"]["configured_payload_count"], 24)
        self.assertEqual(document["expected"]["artifact_file_count"], 26)
        self.assertEqual(tuple(document["active_perturbation_ids"]), PERTURBATIONS)
        self.assertEqual(tuple(document["alpha_grid"]), ALPHAS)
        self.assertEqual(tuple(document["metric_output_ids"]), METRIC_OUTPUT_IDS)
        self.assertEqual(
            document["model_recipe"],
            {
                "copy": True,
                "max_iter": 500,
                "n_components_grid": [2, 4, 8, 16, 32],
                "refit_with_validation": False,
                "scale": True,
                "selection": "minimum_validation_macro_normalized_rmse_lowest_components_on_tie",
                "tol": 1e-6,
                "type": "PLSRegression",
            },
        )
        self.assertEqual(document["inference"]["bootstrap_resamples"], 2000)
        self.assertEqual(document["inference"]["sign_flip_resamples"], 100000)
        self.assertEqual(document["inference"]["random_seed"], 20260817)
        self.assertEqual(document["inference"]["holm_slot_count"], 24)
        self.assertEqual(document["resource_policy"]["default_condition_workers"], 16)
        self.assertEqual(document["resource_policy"]["default_model_workers"], 5)
        self.assertEqual(document["resource_policy"]["default_verifier_condition_workers"], 12)
        self.assertEqual(document["resource_policy"]["default_verifier_model_workers"], 4)
        step32_receipt = _find_authority_receipt(
            document,
            DESIGN_REPORT_PATH.relative_to(ROOT).as_posix(),
        )
        _assert_receipt_matches_file(self, step32_receipt)
        for receipt in document["authorities"].values():
            _assert_receipt_matches_file(self, receipt)
        expected_code_paths = {
            SUBJECT_FILE.relative_to(ROOT).as_posix(),
            VERIFIER_FILE.relative_to(ROOT).as_posix(),
            CLI_FILE.relative_to(ROOT).as_posix(),
            Path(__file__).resolve().relative_to(ROOT).as_posix(),
        }
        self.assertEqual(set(document["code_authority"]), expected_code_paths)
        for relative_path, receipt in document["code_authority"].items():
            _assert_receipt_matches_file(
                self,
                {"path": relative_path, **receipt},
            )
        self.assertEqual(
            document["environment_authority"],
            subject._environment_authority(),
        )

    def test_three_parent_validation_and_payload_firewall(self) -> None:
        subject = _load_subject()
        config = subject.load_phase4_d4_protocol_b_config(REAL_CONFIG_PATH)
        bridge = subject.validate_d4_protocol_b_parent_authorities(
            config,
            parse_step29_measurements=False,
        )
        bridge_text = json.dumps(bridge, ensure_ascii=False, sort_keys=True)
        self.assertIn(STEP27_RUN_ID, bridge_text)
        self.assertIn(STEP29_RUN_ID, bridge_text)
        self.assertIn(STEP31_RUN_ID, bridge_text)
        self.assertIn("record_conditions.jsonl", bridge_text)
        self.assertIn("record_measurements.jsonl", bridge_text)
        self.assertNotIn("predictions.jsonl", tuple(bridge.get("allowed_step29_payloads", ())))
        self.assertNotIn(
            "blank_predictions.jsonl",
            tuple(bridge.get("allowed_step29_payloads", ())),
        )
        self.assertNotIn(
            "well_conditions.jsonl",
            tuple(bridge.get("allowed_step29_payloads", ())),
        )
        self.assertNotIn(
            "alignment_results.jsonl",
            tuple(bridge.get("allowed_step29_payloads", ())),
        )
        self.assertEqual(
            tuple(bridge.get("allowed_step29_payloads", ())),
            ("record_measurements.jsonl",),
        )

    def test_real_measurement_bridge_reproduces_314880_zero_mismatch_receipt(self) -> None:
        if os.environ.get("RPE_ENABLE_D4_PROTOCOL_B_REAL_BRIDGE") != "1":
            self.skipTest(
                "set RPE_ENABLE_D4_PROTOCOL_B_REAL_BRIDGE=1 for the explicit 314,880-row bridge check"
            )
        subject = _load_subject()
        config = subject.load_phase4_d4_protocol_b_config(REAL_CONFIG_PATH)
        bridge = subject.validate_d4_protocol_b_parent_authorities(
            config,
            parse_step29_measurements=True,
        )
        self.assertEqual(bridge["bridge_row_count"], EXPECTED_PREDICTION_ROW_COUNT)
        self.assertEqual(bridge["bridge_sha256"], EXPECTED_MEASUREMENT_BRIDGE_SHA256)
        mismatch = bridge["mismatch_counts"]
        self.assertEqual(mismatch["missing_key_count"], 0)
        self.assertEqual(mismatch["extra_key_count"], 0)
        self.assertEqual(mismatch["duplicate_key_count"], 0)
        self.assertEqual(mismatch["state_mismatch_count"], 0)
        self.assertEqual(mismatch["metric_receipt_mismatch_count"], 0)
        self.assertEqual(mismatch["cwt_receipt_mismatch_count"], 0)

    def test_reconstructs_exact_five_fold_roles_support_and_resource_admission(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        self.assertEqual(tuple(getattr(inputs, "condition_ids", CONDITION_IDS)), CONDITION_IDS)
        self.assertEqual(len(tuple(getattr(inputs, "fold_ids", range(5)))), 5)
        support_axis = getattr(inputs, "support_axis_cm1", None)
        if support_axis is not None:
            self.assertEqual(tuple(support_axis.shape), (1999,))
        builder_signature = inspect.signature(subject.build_phase4_d4_protocol_b)
        self.assertEqual(
            builder_signature.parameters["rematerialization_worker_count"].default,
            16,
        )
        self.assertEqual(
            builder_signature.parameters["model_worker_count"].default,
            5,
        )
        self.assertIn("measurement_bridge_sha256", repr(config))

    def test_real_rematerialization_uses_step27_spectrum_id_domains(self) -> None:
        production_source = SUBJECT_FILE.read_text(encoding="utf-8")
        verifier_source = VERIFIER_FILE.read_text(encoding="utf-8")
        for source in (production_source, verifier_source):
            self.assertIn('spectrum_id=f"d4_sugar_low_snr::{record_id}"', source)
            self.assertIn('spectrum_id=f"d4_blank::{record_id}"', source)
            self.assertNotIn('spectrum_id=f"d4b::{record_id}"', source)
            self.assertNotIn('spectrum_id=f"d4b-blank::{record_id}"', source)

    def test_rematerialization_pins_perturbation_blas_to_one_thread(self) -> None:
        sentinel = object()
        for module in (_load_subject(), _load_verifier()):
            observed: list[int] = []

            def fake_run(*args, **kwargs):
                observed.extend(
                    int(item["num_threads"])
                    for item in module.threadpoolctl.threadpool_info()
                    if item.get("user_api") == "blas"
                )
                return sentinel

            with self.subTest(module=module.__name__), mock.patch.object(
                module, "run_perturbation_cell", side_effect=fake_run
            ):
                result = module._run_perturbation_cell_single_blas(
                    object(),
                    "p10",
                    object(),
                    object(),
                    p10_admission=None,
                )
                self.assertIs(result, sentinel)
                self.assertTrue(observed)
                self.assertEqual(set(observed), {1})

    def test_verifier_projection_interpolates_axis_changing_perturbations(self) -> None:
        verifier = _load_verifier()
        axis = np.asarray([0.0, 1.5, 2.5, 4.0], dtype="<f8")
        intensity = np.asarray([0.0, 15.0, 25.0, 40.0], dtype="<f8")
        support = np.asarray([1.0, 3.0], dtype="<f8")
        projected = verifier._project_to_frozen_support(
            axis=axis,
            intensity=intensity,
            support=support,
            max_gap_cm1=4.0,
        )
        np.testing.assert_array_equal(
            projected, np.asarray([10.0, 30.0], dtype="<f4")
        )


@unittest.skipUnless(
    _subject_surface_available(),
    "awaiting Step 33 D4 Protocol-B build implementation",
)
class Phase4D4ProtocolBModelGateTest(unittest.TestCase):
    def test_same_condition_pls2_uses_candidate_order_tie_break_and_no_refit(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(
            subject,
            validation_macro_nrmse_by_component={
                2: 0.125,
                4: 0.125,
                8: 0.25,
                16: 0.5,
                32: 0.75,
            },
        )
        config = _make_synthetic_config(subject, inputs)
        _, files = _build_synthetic(subject, inputs=inputs, config=config)
        model_rows = _read_jsonl_bytes(files["model_cells.jsonl"])
        validation_rows = _read_jsonl_bytes(files["validation_scores.jsonl"])
        alpha0_models = [row for row in model_rows if row["condition_id"] == "alpha0"]
        alpha0_validation = [
            row for row in validation_rows if row["condition_id"] == "alpha0"
        ]
        self.assertEqual(len(alpha0_models), 5)
        self.assertEqual(len(alpha0_validation), 25)
        self.assertTrue(
            all(
                row["train_condition_id"]
                == row["validation_condition_id"]
                == row["test_condition_id"]
                == row["blank_condition_id"]
                == row["condition_id"]
                for row in alpha0_models
            )
        )
        self.assertTrue(all(row["selected_n_components"] == 2 for row in alpha0_models))
        self.assertTrue(all(row["refit_with_validation"] is False for row in alpha0_models))
        for fold in range(5):
            candidate_order = [
                row["n_components"]
                for row in alpha0_validation
                if int(row["fold"]) == fold
            ]
            self.assertEqual(candidate_order, [2, 4, 8, 16, 32])

    def test_alpha0_requires_all_five_exact_projections(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        _, files = _build_synthetic(subject, inputs=inputs, config=config)
        alpha0 = json.loads(files["alpha0_equivalence.json"].decode("utf-8"))
        self.assertEqual(alpha0["status"], "passed")
        self.assertEqual(alpha0["mismatch_count"], 0)
        self.assertEqual(
            set(alpha0["observed_digests"]),
            {
                "model_digest",
                "validation_digest",
                "prediction_digest",
                "blank_prediction_digest",
                "technical_lod_loq_digest",
            },
        )
        self.assertEqual(alpha0["expected_digests"]["model_digest"], EXPECTED_ALPHA0_MODEL_SHA256)
        self.assertEqual(
            alpha0["expected_digests"]["validation_digest"],
            EXPECTED_ALPHA0_VALIDATION_SHA256,
        )
        self.assertEqual(
            alpha0["expected_digests"]["prediction_digest"],
            EXPECTED_ALPHA0_PREDICTION_SHA256,
        )
        self.assertEqual(
            alpha0["expected_digests"]["blank_prediction_digest"],
            EXPECTED_ALPHA0_BLANK_SHA256,
        )
        self.assertEqual(
            alpha0["expected_digests"]["technical_lod_loq_digest"],
            EXPECTED_ALPHA0_LOD_SHA256,
        )

    def test_alpha0_mismatch_emits_fixed_failed_shape_and_does_not_open_measurements(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(
            subject,
            inputs,
            tamper_alpha0_projection="prediction_digest",
        )
        original_open = Path.open
        original_read_bytes = Path.read_bytes
        original_read_text = Path.read_text

        def _guard_open(path_obj: Path, *args, **kwargs):
            if path_obj.name == "record_measurements.jsonl":
                raise AssertionError("record_measurements.jsonl must not be opened before alpha0 passes")
            return original_open(path_obj, *args, **kwargs)

        def _guard_read_bytes(path_obj: Path, *args, **kwargs):
            if path_obj.name == "record_measurements.jsonl":
                raise AssertionError("record_measurements.jsonl must not be read before alpha0 passes")
            return original_read_bytes(path_obj, *args, **kwargs)

        def _guard_read_text(path_obj: Path, *args, **kwargs):
            if path_obj.name == "record_measurements.jsonl":
                raise AssertionError("record_measurements.jsonl must not be read before alpha0 passes")
            return original_read_text(path_obj, *args, **kwargs)

        with (
            mock.patch.object(Path, "open", _guard_open),
            mock.patch.object(Path, "read_bytes", _guard_read_bytes),
            mock.patch.object(Path, "read_text", _guard_read_text),
        ):
            summary, files = _build_synthetic(subject, inputs=inputs, config=config)
        self.assertEqual(summary.status, "failed")
        self.assertIn("failed.json", files)
        self.assertNotIn("complete.json", files)

        manifest = json.loads(files["manifest.json"].decode("utf-8"))
        authority_bridge = json.loads(files["authority_bridge.json"].decode("utf-8"))
        self.assertEqual(manifest["endpoint_state"], "failed_alpha0_equivalence")
        self.assertEqual(
            authority_bridge["measurement_bridge"]["state"],
            "not_tested_endpoint_closed",
        )

        model_rows = _read_jsonl_bytes(files["model_cells.jsonl"])
        validation_rows = _read_jsonl_bytes(files["validation_scores.jsonl"])
        prediction_rows = _read_jsonl_bytes(files["predictions.jsonl"])
        blank_rows = _read_jsonl_bytes(files["blank_predictions.jsonl"])
        well_rows = _read_jsonl_bytes(files["well_conditions.jsonl"])
        lod_rows = _read_jsonl_bytes(files["technical_lod_loq.jsonl"])
        holm_rows = _read_jsonl_bytes(files["holm_family.jsonl"])
        self.assertEqual(len(model_rows), 5 * len(CONDITION_IDS))
        self.assertEqual(len(validation_rows), 25 * len(CONDITION_IDS))
        self.assertEqual(len(holm_rows), EXPECTED_HOLM_ROW_COUNT)
        self.assertTrue(
            all(
                row["state"] == "not_tested_endpoint_closed"
                for row in holm_rows
            )
        )
        self.assertTrue(all(row["raw_p_value"] == 1.0 for row in holm_rows))
        self.assertTrue(all(row["adjusted_p_value"] == 1.0 for row in holm_rows))

        positive_model_rows = [row for row in model_rows if row["condition_id"] != "alpha0"]
        positive_validation_rows = [
            row for row in validation_rows if row["condition_id"] != "alpha0"
        ]
        positive_prediction_rows = [
            row for row in prediction_rows if row["condition_id"] != "alpha0"
        ]
        positive_blank_rows = [row for row in blank_rows if row["condition_id"] != "alpha0"]
        positive_well_rows = [row for row in well_rows if row["condition_id"] != "alpha0"]
        positive_lod_rows = [row for row in lod_rows if row["condition_id"] != "alpha0"]
        self.assertTrue(
            all(row["state"] == "not_tested_endpoint_closed" for row in positive_model_rows)
        )
        self.assertTrue(
            all(row["state"] == "not_tested_endpoint_closed" for row in positive_validation_rows)
        )
        self.assertTrue(
            all(row["state"] == "not_tested_endpoint_closed" for row in positive_prediction_rows)
        )
        self.assertTrue(
            all(row["state"] == "not_tested_endpoint_closed" for row in positive_blank_rows)
        )
        self.assertTrue(
            all(row["state"] == "not_tested_endpoint_closed" for row in positive_well_rows)
        )
        self.assertTrue(
            all(row["state"] == "not_tested_endpoint_closed" for row in positive_lod_rows)
        )

        figure1_rows = _read_csv_bytes(files["figure1_d4_protocol_b_full_domain_data.csv"])
        figure2_rows = _read_csv_bytes(files["figure2_d4_protocol_b_full_domain_data.csv"])
        table_rows = _read_csv_bytes(files["d4_protocol_b_full_domain_secondary_table.csv"])
        self.assertEqual(len(figure1_rows), EXPECTED_FIGURE1_ROW_COUNT)
        self.assertEqual(len(figure2_rows), EXPECTED_FIGURE2_ROW_COUNT)
        self.assertEqual(len(table_rows), EXPECTED_TABLE_ROW_COUNT)
        self.assertTrue(
            all(row.get("state", "") == "not_tested_endpoint_closed" for row in figure2_rows)
        )

    def test_positive_conditions_execute_only_after_alpha0_and_in_frozen_batches(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        observed = []
        original = subject.fit_d4_protocol_b_condition

        def _wrapped_fit(*args, **kwargs):
            observed.append(kwargs["condition_id"])
            return original(*args, **kwargs)

        with mock.patch.object(subject, "fit_d4_protocol_b_condition", side_effect=_wrapped_fit):
            _build_synthetic(subject, inputs=inputs, config=config)
        self.assertEqual(observed[0], "alpha0")
        self.assertEqual(observed, list(CONDITION_IDS))


@unittest.skipUnless(
    _subject_surface_available(),
    "awaiting Step 33 D4 Protocol-B outcome implementation",
)
class Phase4D4ProtocolBOutcomeTest(unittest.TestCase):
    def test_hand_derived_stitched_predictions_well_loss_and_lod_loq(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(
            subject,
            prediction_fixture="hand_derived_regression",
            blank_fixture="isolated_auxiliary_failure",
        )
        config = _make_synthetic_config(subject, inputs)
        _, files = _build_synthetic(subject, inputs=inputs, config=config)

        predictions = _read_jsonl_bytes(files["predictions.jsonl"])
        self.assertEqual(
            len({(row["record_id"], row["condition_id"]) for row in predictions}),
            len(predictions),
        )
        alpha0_predictions = [
            row for row in predictions if row["condition_id"] == "alpha0"
        ]
        self.assertEqual(len(alpha0_predictions), len(predictions) // len(CONDITION_IDS))

        summaries = _read_csv_bytes(files["condition_summary.csv"])
        alpha0_summary = next(row for row in summaries if row["condition_id"] == "alpha0")
        p08_summary = next(
            row for row in summaries if row["condition_id"] == CONDITION_IDS[1]
        )
        self.assertLess(
            float(alpha0_summary["macro_normalized_rmse"]),
            float(p08_summary["macro_normalized_rmse"]),
        )

        well_rows = _read_jsonl_bytes(files["well_conditions.jsonl"])
        alpha0_well = next(
            row for row in well_rows if row["condition_id"] == "alpha0"
        )
        p08_well = next(
            row for row in well_rows if row["condition_id"] == CONDITION_IDS[1]
        )
        self.assertEqual(alpha0_well["downstream_harm"], 0.0)
        self.assertAlmostEqual(
            p08_well["downstream_harm"],
            p08_well["loss"] - alpha0_well["loss"],
        )

        lod_rows = _read_jsonl_bytes(files["technical_lod_loq.jsonl"])
        complete_lod = next(row for row in lod_rows if row["condition_id"] == "alpha0")
        self.assertGreater(complete_lod["slope"], 0.0)
        self.assertGreaterEqual(complete_lod["ich_loq"], complete_lod["ich_lod"])
        self.assertGreaterEqual(complete_lod["ich_lod"], complete_lod["iupac_lod"])

    def test_measurement_bridge_orients_metric_harm_and_emits_124800_contract(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject, metric_fixture="orientation_contract")
        config = _make_synthetic_config(subject, inputs)
        _, files = _build_synthetic(subject, inputs=inputs, config=config)

        authority_bridge = json.loads(files["authority_bridge.json"].decode("utf-8"))
        self.assertEqual(authority_bridge["measurement_bridge"]["state"], "complete")

        observations = _read_jsonl_bytes(files["well_observations.jsonl"])
        metric_ids = {row["metric_output_id"] for row in observations}
        self.assertEqual(metric_ids, set(METRIC_OUTPUT_IDS))
        expected_rows = 13 * len({row["well_id"] for row in observations}) * 40
        self.assertEqual(len(observations), expected_rows)

        mse_row = next(row for row in observations if row["metric_output_id"] == "mse")
        pearson_row = next(
            row for row in observations if row["metric_output_id"] == "pearson_r"
        )
        self.assertEqual(mse_row["preferred_direction"], "lower_is_better")
        self.assertEqual(pearson_row["preferred_direction"], "higher_is_better")

    def test_fixed_24_slot_inference_and_auxiliary_failure_isolation(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(
            subject,
            blank_fixture="isolated_auxiliary_failure",
        )
        config = _make_synthetic_config(subject, inputs)
        summary, files = _build_synthetic(subject, inputs=inputs, config=config)
        self.assertEqual(summary.status, "complete")

        lod_rows = _read_jsonl_bytes(files["technical_lod_loq.jsonl"])
        auxiliary_failure = next(
            row
            for row in lod_rows
            if row["state"] == "not_evaluable_nonpositive_slope"
        )
        self.assertIsNone(auxiliary_failure["ich_lod"])
        self.assertIsNone(auxiliary_failure["ich_loq"])

        self.assertEqual(
            len(_read_jsonl_bytes(files["alignment_results.jsonl"])),
            EXPECTED_ALIGNMENT_ROW_COUNT,
        )
        self.assertEqual(
            len(_read_jsonl_bytes(files["bootstrap_results.jsonl"])),
            EXPECTED_BOOTSTRAP_ROW_COUNT,
        )
        self.assertEqual(
            len(_read_jsonl_bytes(files["sign_flip_results.jsonl"])),
            EXPECTED_SIGN_FLIP_ROW_COUNT,
        )
        self.assertEqual(
            len(_read_jsonl_bytes(files["holm_family.jsonl"])),
            EXPECTED_HOLM_ROW_COUNT,
        )

    def test_success_artifact_uses_preregistered_statistics_not_placeholder_slots(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject, metric_fixture="orientation_contract")
        config = _make_synthetic_config(subject, inputs)
        _, files = _build_synthetic(subject, inputs=inputs, config=config)

        alignment = _read_jsonl_bytes(files["alignment_results.jsonl"])
        bootstrap = _read_jsonl_bytes(files["bootstrap_results.jsonl"])
        sign_flip = _read_jsonl_bytes(files["sign_flip_results.jsonl"])
        holm = _read_jsonl_bytes(files["holm_family.jsonl"])
        self.assertEqual(
            set(alignment[0]),
            {
                "acc_cross", "acc_interval", "ag", "ag_interval",
                "ag_raw", "clusters", "cross_pair_count",
                "metric_output_id", "observation_count", "state",
            },
        )
        self.assertEqual(bootstrap[0]["metric_output_id"], "rmse")
        self.assertEqual(sign_flip[0]["hypothesis_id"], "rmse:d_ag")
        self.assertEqual(holm[0]["hypothesis_id"], "rmse:d_ag")
        self.assertEqual({row["statistic"] for row in sign_flip}, {"d_ag", "d_acc"})
        self.assertGreater(len(files["figure1_d4_protocol_b_full_domain.png"]), 10_000)
        self.assertGreater(len(files["figure2_d4_protocol_b_full_domain.png"]), 10_000)


@unittest.skipUnless(
    _full_surface_available(),
    "awaiting Step 33 D4 Protocol-B verifier, authority, and CLI implementation",
)
class Phase4D4ProtocolBArtifactVerifierCliTest(unittest.TestCase):
    def test_audit_payloads_record_frozen_authorities_preflight_and_counts(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        _, files = _build_synthetic(subject, inputs=inputs, config=config)

        bridge = json.loads(files["authority_bridge.json"])
        self.assertEqual(set(bridge["parents"]), {"step27", "step29", "step31"})
        self.assertEqual(
            bridge["step29_payload_policy"]["allowed_computational_inputs"],
            ["record_measurements.jsonl"],
        )
        self.assertIn(
            "predictions.jsonl",
            bridge["step29_payload_policy"]["allowed_postcomputation_qc"],
        )
        self.assertIn(
            "alignment_results.jsonl",
            bridge["step29_payload_policy"]["forbidden_inputs"],
        )

        preflight = json.loads(files["preflight.json"])
        self.assertTrue(
            {
                "source_identity", "support_identity", "condition_identity",
                "role_identity", "matrix_receipts", "resource_admission",
                "readiness",
            }.issubset(preflight)
        )
        self.assertEqual(preflight["condition_identity"]["condition_count"], 41)
        self.assertEqual(
            preflight["matrix_receipts"]["mixture_condition_matrix_count"],
            41,
        )

        manifest = json.loads(files["manifest.json"])
        self.assertEqual(manifest["claim_boundary"], config.document["claim_boundary"])
        self.assertTrue(
            {"identities", "states", "inherited_rulings", "fixed_capacities", "payload_order"}.issubset(manifest)
        )
        self.assertEqual(manifest["counts"]["model_cells"], 205)
        self.assertEqual(manifest["counts"]["predictions"], 410)
        self.assertEqual(manifest["states"]["model_cells"], {"complete": 205})
        self.assertEqual(manifest["fixed_capacities"]["holm_slots"], 24)

    def test_success_artifact_has_24_payloads_26_files_and_exact_rows(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        summary, files = _build_synthetic(subject, inputs=inputs, config=config)
        self.assertEqual(summary.status, "complete")
        self.assertEqual(len(EXPECTED_PAYLOAD_FILES), 24)
        self.assertEqual(len(files), 26)
        self.assertIn("complete.json", files)
        self.assertIn("SHA256SUMS", files)

        manifest = json.loads(files["manifest.json"].decode("utf-8"))
        self.assertEqual(tuple(manifest["payload_files"]), EXPECTED_PAYLOAD_FILES)
        self.assertEqual(manifest["counts"]["artifact_files"], 26)
        self.assertEqual(
            len(_read_jsonl_bytes(files["model_cells.jsonl"])),
            5 * len(CONDITION_IDS),
        )
        self.assertEqual(
            len(_read_jsonl_bytes(files["validation_scores.jsonl"])),
            25 * len(CONDITION_IDS),
        )
        self.assertEqual(
            len(_read_jsonl_bytes(files["blank_predictions.jsonl"])),
            5 * 2 * len(CONDITION_IDS),
        )
        self.assertEqual(
            len(_read_csv_bytes(files["condition_summary.csv"])),
            EXPECTED_CONDITION_SUMMARY_ROW_COUNT,
        )

    def test_worker_counts_do_not_change_run_id_or_bytes(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        first_summary, first_files = _build_synthetic(
            subject,
            inputs=inputs,
            config=config,
            worker_count=1,
        )
        second_summary, second_files = _build_synthetic(
            subject,
            inputs=inputs,
            config=config,
            worker_count=2,
        )
        self.assertEqual(first_summary.run_id, second_summary.run_id)
        self.assertEqual(first_files["SHA256SUMS"], second_files["SHA256SUMS"])
        self.assertEqual(first_files["manifest.json"], second_files["manifest.json"])
        manifest_text = first_files["manifest.json"].decode("utf-8")
        self.assertNotIn("rematerialization_worker_count", manifest_text)
        self.assertNotIn("model_worker_count", manifest_text)

    def test_figure_sources_and_left_axis_labels_are_deterministic(self) -> None:
        subject = _load_subject()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        _, first_files = _build_synthetic(subject, inputs=inputs, config=config)
        _, second_files = _build_synthetic(subject, inputs=inputs, config=config)
        self.assertEqual(
            first_files["figure1_d4_protocol_b_full_domain_data.csv"],
            second_files["figure1_d4_protocol_b_full_domain_data.csv"],
        )
        self.assertEqual(
            first_files["figure2_d4_protocol_b_full_domain_data.csv"],
            second_files["figure2_d4_protocol_b_full_domain_data.csv"],
        )
        figure2_svg = first_files["figure2_d4_protocol_b_full_domain.svg"].decode(
            "utf-8",
            errors="replace",
        )
        for label in METRIC_OUTPUT_IDS:
            self.assertIn(label, figure2_svg)

    def test_independent_verifier_rejects_checksum_consistent_semantic_tampering(self) -> None:
        subject = _load_subject()
        verifier = _load_verifier()
        inputs = _make_synthetic_inputs(subject)
        config = _make_synthetic_config(subject, inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            built = subject.build_phase4_d4_protocol_b_from_inputs(
                output,
                inputs=inputs,
                config=config,
                rematerialization_worker_count=1,
                model_worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            rows = [
                json.loads(line)
                for line in (built.path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["predicted_targets"][0] = float(rows[0]["predicted_targets"][0]) + 0.01
            (built.path / "predictions.jsonl").write_text(
                "".join(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            _rewrite_sha256sums(built.path)
            with self.assertRaisesRegex(
                verifier.Phase4D4ProtocolBVerifierError,
                "semantic|payload|mismatch",
            ):
                verifier.verify_phase4_d4_protocol_b_from_inputs(
                    built.path,
                    inputs=inputs,
                    config_path=built.path / "config.json",
                    rematerialization_worker_count=1,
                    model_worker_count=1,
                    bootstrap_resamples=8,
                    sign_flip_resamples=16,
                )

    def test_verifier_ast_has_no_production_import(self) -> None:
        tree = ast.parse(VERIFIER_FILE.read_text(encoding="utf-8"))
        forbidden = SUBJECT_MODULE
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(forbidden, {alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, forbidden)

    def test_cli_rejects_skip_qc_protocol_component_inference_and_outcome_overrides(self) -> None:
        cli = _load_cli_module()
        for arguments in (
            ["build", "--output-root", "x", "--skip-qc"],
            ["build", "--output-root", "x", "--protocol", "A"],
            ["build", "--output-root", "x", "--condition", "alpha0"],
            ["build", "--output-root", "x", "--n-components", "2"],
            ["build", "--output-root", "x", "--bootstrap-resamples", "8"],
            ["verify", "--run-path", "x", "--outcome", "override"],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    cli.main(arguments)


if __name__ == "__main__":
    unittest.main()
