from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEP15_RUN_ID = "67fedde2a92c0ee57c504830ebefc23c2ca0ba6dbebc33a026e6cee2279628c6"
STEP15_SHA256SUMS_SHA256 = "389e56c3eb7c261b3b67d43ab47a18a3d613ce703fdb6cc8d22f1868841f7073"
STEP21_RUN_ID = "phase4-d1-protocol-a-full-domain-4425f6ea885491aa0a6382a136a839125b2574b54170c2c2d5a740ade84a253e"
STEP21_SHA256SUMS_SHA256 = "1ef563fe162c374647d0beff261e8fe532dcb74fb5245cd0a634bb4d16f24888"
STEP23_RUN_ID = "phase4-d1-protocol-b-all-role-eligibility-3dde99eeda881d193e33ee5d3d4d2ac0b6e8e7bcc939e754d93e710a4112f3a1"
STEP23_SHA256SUMS_SHA256 = "496097f4dc08ab62a12a8b648a7aad4454ed7671e0541bc0c1daeeab12fdf22f"
TEST_RECORD_ID_DIGEST = "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
BRIDGE_SHA256 = "031b4f4f30235a6237ad1a6f89b03fb9f2d1b724f55f7c8ae9d52f402eabaa7c"
WORKER_ADMISSION_BYTES = 3_179_405_824
ADMISSION_BUDGET_BYTES = 64 * 1024**3


class D1ProtocolBContractTest(unittest.TestCase):
    @staticmethod
    def _rewrite_sha256sums(path: Path, terminal_name: str = "complete.json") -> None:
        from rpe.runner.phase4_d1_protocol_b_authority import ARTIFACT_PAYLOAD_FILES

        rows = []
        for name in ARTIFACT_PAYLOAD_FILES + (terminal_name,):
            digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
            rows.append(f"{digest}  {name}")
        (path / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_phase4_d1_protocol_b_module_and_config_exist(self) -> None:
        subject = importlib.import_module("rpe.runner.phase4_d1_protocol_b")
        verifier = importlib.import_module("rpe.runner.phase4_d1_protocol_b_verifier")
        config_path = ROOT / "experiments/phase4/configs/d1_protocol_b_full_domain_v1.json"
        self.assertTrue(config_path.is_file())
        self.assertTrue(hasattr(subject, "load_phase4_d1_protocol_b_config"))
        self.assertTrue(hasattr(subject, "build_phase4_d1_protocol_b"))
        self.assertTrue(hasattr(subject, "build_phase4_d1_protocol_b_from_inputs"))
        self.assertEqual(subject.WORKER_ADMISSION_BYTES, WORKER_ADMISSION_BYTES)
        self.assertEqual(
            inspect.signature(subject.build_phase4_d1_protocol_b).parameters["worker_count"].default,
            5,
        )
        self.assertEqual(
            inspect.signature(verifier.verify_phase4_d1_protocol_b).parameters["worker_count"].default,
            4,
        )

    def test_production_source_contains_no_placeholder_real_path_raise_or_literal_figure_bytes(self) -> None:
        source = (ROOT / "rpe/runner/phase4_d1_protocol_b.py").read_text(encoding="utf-8")
        self.assertNotIn("real Step-25 reconstruction is authority-bound and not available in focused tests", source)
        self.assertNotIn("synthetic-d1b-figure1-png", source)
        self.assertNotIn("synthetic-d1b-figure2-png", source)

    def test_frozen_config_binds_real_parents_counts_and_alpha0_contract(self) -> None:
        import rpe.runner.phase4_d1_protocol_b as subject
        from rpe.runner.phase4_d1_protocol_b_authority import (
            ARTIFACT_PAYLOAD_FILES,
            CODE_RELATIVE_PATHS,
            REAL_ALPHA0_MODEL_DIGEST,
            REAL_ALPHA0_PREDICTION_DIGEST,
            REAL_ALPHA0_VALIDATION_DIGEST,
        )

        document = json.loads(
            (ROOT / "experiments/phase4/configs/d1_protocol_b_full_domain_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(tuple(document["artifact_payload_files"]), ARTIFACT_PAYLOAD_FILES)
        self.assertEqual(set(document["code_authority"]), set(CODE_RELATIVE_PATHS))
        self.assertEqual(document["code_authority"], subject._code_authority())
        self.assertEqual(document["environment_authority"], subject._environment_authority())
        for receipt in document["authorities"].values():
            target = ROOT / receipt["path"]
            self.assertTrue(target.is_file())
            self.assertEqual(receipt["bytes"], target.stat().st_size)
            self.assertEqual(receipt["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(document["parent_artifacts"]["step15"]["run_id"], STEP15_RUN_ID)
        self.assertEqual(document["parent_artifacts"]["step15"]["sha256sums_sha256"], STEP15_SHA256SUMS_SHA256)
        self.assertEqual(document["parent_artifacts"]["step21"]["run_id"], STEP21_RUN_ID)
        self.assertEqual(document["parent_artifacts"]["step21"]["sha256sums_sha256"], STEP21_SHA256SUMS_SHA256)
        self.assertEqual(document["parent_artifacts"]["step23"]["run_id"], STEP23_RUN_ID)
        self.assertEqual(document["parent_artifacts"]["step23"]["sha256sums_sha256"], STEP23_SHA256SUMS_SHA256)
        self.assertEqual(document["alpha0_equivalence"]["model_digest"], REAL_ALPHA0_MODEL_DIGEST)
        self.assertEqual(document["alpha0_equivalence"]["validation_digest"], REAL_ALPHA0_VALIDATION_DIGEST)
        self.assertEqual(document["alpha0_equivalence"]["prediction_digest"], REAL_ALPHA0_PREDICTION_DIGEST)
        self.assertEqual(document["model_recipe"]["c_grid"], [0.01, 0.1, 1.0, 10.0])
        self.assertEqual(document["inference"]["bootstrap_resamples"], 2000)
        self.assertEqual(document["inference"]["sign_flip_resamples"], 100000)
        expected = document["expected"]
        self.assertEqual(expected["configured_payload_count"], 22)
        self.assertEqual(expected["artifact_file_count"], 24)
        self.assertEqual(expected["model_cell_count"], 205)
        self.assertEqual(expected["validation_row_count"], 820)
        self.assertEqual(expected["prediction_row_count"], 615000)
        self.assertEqual(expected["seed_class_condition_count"], 6150)
        self.assertEqual(expected["class_observation_count"], 15600)
        self.assertEqual(expected["alignment_row_count"], 13)
        self.assertEqual(expected["bootstrap_row_count"], 12)
        self.assertEqual(expected["sign_flip_row_count"], 24)
        self.assertEqual(expected["holm_row_count"], 24)

    def test_real_parent_authority_validation_rebuilds_exact_bridge_and_authorized_counts(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import (
            load_phase4_d1_protocol_b_config,
            validate_d1_protocol_b_parent_authorities,
        )
        from rpe.runner.phase4_d1_protocol_b_authority import condition_ids, PERTURBATIONS, ALPHAS

        bridge = validate_d1_protocol_b_parent_authorities(load_phase4_d1_protocol_b_config())
        self.assertEqual(bridge["bridge_row_count"], 123000)
        self.assertEqual(bridge["metric_row_count"], 1_599_000)
        self.assertEqual(bridge["cwt_row_count"], 123000)
        self.assertEqual(bridge["bridge_sha256"], BRIDGE_SHA256)
        self.assertEqual(bridge["test_record_ids_sha256"], TEST_RECORD_ID_DIGEST)
        self.assertEqual(tuple(bridge["condition_ids"]), condition_ids(PERTURBATIONS, ALPHAS))
        self.assertEqual(tuple(bridge["authorized_step15_payloads"]), ("record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"))
        self.assertEqual(bridge["step15"]["run_id"], STEP15_RUN_ID)
        self.assertEqual(bridge["step15"]["sha256sums_sha256"], STEP15_SHA256SUMS_SHA256)
        self.assertEqual(bridge["step21"]["run_id"], STEP21_RUN_ID)
        self.assertEqual(bridge["step21"]["sha256sums_sha256"], STEP21_SHA256SUMS_SHA256)
        self.assertEqual(bridge["step23"]["run_id"], STEP23_RUN_ID)
        self.assertEqual(bridge["step23"]["sha256sums_sha256"], STEP23_SHA256SUMS_SHA256)
        mismatch = bridge["mismatch_counts"]
        self.assertEqual(mismatch["missing_key_count"], 0)
        self.assertEqual(mismatch["extra_key_count"], 0)
        self.assertEqual(mismatch["duplicate_key_count"], 0)
        self.assertEqual(mismatch["field_mismatch_count"], 0)
        self.assertEqual(mismatch["test_order_mismatch_count"], 0)
        self.assertEqual(mismatch["state_mismatch_count"], 0)

    def test_real_matrix_gather_uses_canonical_role_order_and_local_test_order(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import (
            gather_protocol_b_role_matrices,
            load_phase4_d1_protocol_b_config,
            validate_d1_protocol_b_parent_authorities,
        )

        config = load_phase4_d1_protocol_b_config()
        bridge = validate_d1_protocol_b_parent_authorities(config)
        matrices = gather_protocol_b_role_matrices(config, bridge, condition_id="alpha0", model_seed=0)
        self.assertEqual(matrices["train"].shape, (62700, 997))
        self.assertEqual(matrices["validation"].shape, (300, 997))
        self.assertEqual(matrices["test"].shape, (3000, 997))
        self.assertEqual(matrices["train"].dtype, np.float32)
        self.assertEqual(matrices["validation"].dtype, np.float32)
        self.assertEqual(matrices["test"].dtype, np.float32)
        self.assertEqual(tuple(matrices["test_orders"][:5]), (0, 1, 2, 3, 4))
        self.assertEqual(matrices["test_record_ids_sha256"], TEST_RECORD_ID_DIGEST)
        self.assertEqual(matrices["role_counts"], {"train": 62700, "validation": 300, "test": 3000})

    def test_model_lifecycle_uses_same_condition_train_only_pca_and_lowest_c_on_tie(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import (
            fit_protocol_b_condition_models,
            make_synthetic_d1_protocol_b_config,
            make_synthetic_d1_protocol_b_inputs,
        )

        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=3,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
            feature_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        models = fit_protocol_b_condition_models(inputs, config)
        self.assertEqual(len(models), 205)
        alpha0 = [cell for cell in models if cell.condition_id == "alpha0"]
        self.assertEqual(len(alpha0), 5)
        self.assertTrue(all(cell.selected_c == 0.01 for cell in alpha0))
        self.assertTrue(
            all(tuple(row["c"] for row in cell.validation_scores) == (0.01, 0.1, 1.0, 10.0) for cell in alpha0)
        )
        self.assertTrue(all(cell.refit_with_validation is False for cell in alpha0))
        self.assertTrue(all(cell.train_condition_id == cell.condition_id for cell in alpha0))
        self.assertTrue(all(cell.validation_condition_id == cell.condition_id for cell in alpha0))
        self.assertTrue(all(cell.test_condition_id == cell.condition_id for cell in alpha0))

    def test_lr_numeric_policy_ignores_only_benign_underflow(self) -> None:
        import pickle

        import rpe.runner.phase4_d1_protocol_b as subject
        import rpe.runner.phase4_d1_protocol_b_verifier as verifier

        observed_policies: list[dict[str, str]] = []

        class DeterministicPCA:
            def __init__(self, *, n_components: int, **_kwargs: object) -> None:
                self.n_components = n_components

            def fit_transform(self, values: np.ndarray) -> np.ndarray:
                array = np.asarray(values, dtype=np.float64)
                self.components_ = np.eye(self.n_components, array.shape[1], dtype=np.float64)
                self.mean_ = np.mean(array, axis=0)
                self.explained_variance_ = np.ones(self.n_components, dtype=np.float64)
                return array[:, : self.n_components]

            def transform(self, values: np.ndarray) -> np.ndarray:
                return np.asarray(values, dtype=np.float64)[:, : self.n_components]

        class UnderflowingLogisticRegression:
            def __init__(self, *, C: float, **_kwargs: object) -> None:
                self.C = C

            def fit(self, features: np.ndarray, labels: np.ndarray) -> "UnderflowingLogisticRegression":
                policy = dict(np.geterr())
                observed_policies.append(policy)
                if policy["under"] == "ignore":
                    np.exp(np.asarray([-1000.0], dtype=np.float64))
                self.classes_ = np.unique(labels)
                self.coef_ = np.zeros((len(self.classes_), features.shape[1]), dtype=np.float64)
                self.intercept_ = np.zeros(len(self.classes_), dtype=np.float64)
                self.n_iter_ = np.asarray([1], dtype=np.int64)
                return self

            def score(self, _features: np.ndarray, _labels: np.ndarray) -> float:
                return 1.0

            def predict(self, features: np.ndarray) -> np.ndarray:
                return np.resize(self.classes_, len(features))

        labels = np.arange(30, dtype=np.int64)
        matrix = np.arange(30 * 25, dtype=np.float32).reshape(30, 25)
        ids = tuple(f"record-{index}" for index in range(30))
        with patch.object(subject, "PCA", DeterministicPCA), patch.object(
            subject, "LogisticRegression", UnderflowingLogisticRegression
        ):
            subject._fit_predict_single_condition(
                0, "alpha0", matrix, labels, matrix, labels, matrix, labels, ids
            )

        data = {
            "train": matrix,
            "validation": matrix,
            "test": matrix,
            "labels": {"train": labels, "validation": labels, "test": labels},
            "ids": {"train": ids, "validation": ids, "test": ids},
        }
        with patch.object(verifier, "PCA", DeterministicPCA), patch.object(
            verifier, "LogisticRegression", UnderflowingLogisticRegression
        ):
            verifier_result = verifier._fit_cell((0, "alpha0", data))
        pickle.loads(pickle.dumps(verifier_result))

        expected = {"divide": "raise", "over": "raise", "under": "ignore", "invalid": "raise"}
        self.assertEqual(len(observed_policies), 8)
        self.assertTrue(all(policy == expected for policy in observed_policies))

    def test_lr_warning_failure_identifies_condition_seed_and_warning(self) -> None:
        import pickle

        import rpe.runner.phase4_d1_protocol_b as subject

        class WarningLogisticRegression:
            def __init__(self, *, C: float, **_kwargs: object) -> None:
                self.C = C

            def fit(self, features: np.ndarray, labels: np.ndarray) -> "WarningLogisticRegression":
                import warnings

                warnings.warn("did not converge", RuntimeWarning)
                self.classes_ = np.unique(labels)
                self.coef_ = np.zeros((len(self.classes_), features.shape[1]), dtype=np.float64)
                self.intercept_ = np.zeros(len(self.classes_), dtype=np.float64)
                self.n_iter_ = np.asarray([1000], dtype=np.int64)
                return self

            def score(self, _features: np.ndarray, _labels: np.ndarray) -> float:
                return 1.0

        labels = np.repeat(np.arange(2, dtype=np.int64), 2)
        matrix = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        with patch.object(subject, "LogisticRegression", WarningLogisticRegression):
            with self.assertRaises(subject.Phase4D1ProtocolBModelLifecycleWarning) as raised:
                subject._fit_predict_single_condition(
                    3,
                    "p12:deadbeef",
                    matrix,
                    labels,
                    matrix,
                    labels,
                    matrix,
                    labels,
                    tuple(f"record-{index}" for index in range(4)),
                )
        self.assertEqual(
            raised.exception.receipt,
            {
                "state": "failed_model_lifecycle",
                "condition_id": "p12:deadbeef",
                "model_seed": 3,
                "c": 0.01,
                "warning_category": "RuntimeWarning",
                "warning_message": "did not converge",
            },
        )
        restored = pickle.loads(pickle.dumps(raised.exception))
        self.assertEqual(restored.receipt, raised.exception.receipt)

    def test_real_metadata_alpha0_and_figure_bytes_match_independent_verifier(self) -> None:
        import rpe.runner.phase4_d1_protocol_b as subject
        import rpe.runner.phase4_d1_protocol_b_verifier as verifier

        config = json.loads(
            (ROOT / "experiments/phase4/configs/d1_protocol_b_full_domain_v1.json").read_text(
                encoding="utf-8"
            )
        )
        bridge = {
            "bridge_sha256": BRIDGE_SHA256,
            "test_record_ids_sha256": TEST_RECORD_ID_DIGEST,
            "condition_ids": tuple(config["condition_ids"]),
            "bridge_row_count": 123000,
            "metric_row_count": 1599000,
            "cwt_row_count": 123000,
            "mismatch_counts": {
                "missing_key_count": 0,
                "extra_key_count": 0,
                "duplicate_key_count": 0,
                "field_mismatch_count": 0,
                "test_order_mismatch_count": 0,
                "state_mismatch_count": 0,
            },
            "parent_checksum_trees": {
                "step15": {
                    "complete.json": "1" * 64,
                    "metric_values.jsonl": "2" * 64,
                    "model_cells.jsonl": "3" * 64,
                    "peak_receipts.jsonl": "4" * 64,
                    "predictions.jsonl": "5" * 64,
                    "record_conditions.jsonl": "6" * 64,
                },
                "step21": {"complete.json": "7" * 64},
                "step23": {
                    "complete.json": "8" * 64,
                    "condition_matrices_000.f32le": "9" * 64,
                },
            },
        }
        current = {
            "model_digest": "a" * 64,
            "validation_digest": "b" * 64,
            "prediction_digest": "c" * 64,
        }
        production_alpha = subject._real_alpha0_receipt(current, current, current)
        verifier_alpha = verifier._real_alpha0_receipt(current, current, current)
        self.assertEqual(production_alpha, verifier_alpha)
        self.assertEqual(
            set(production_alpha),
            {
                "model_digest", "validation_digest", "prediction_digest",
                "parent_digests", "expected_digests", "mismatch_counts",
                "mismatch_count", "status",
            },
        )
        drifted = {**current, "prediction_digest": "d" * 64}
        failed_production_alpha = subject._real_alpha0_receipt(drifted, current, current)
        failed_verifier_alpha = verifier._real_alpha0_receipt(drifted, current, current)
        self.assertEqual(failed_production_alpha, failed_verifier_alpha)
        self.assertEqual(failed_production_alpha["status"], "failed")
        self.assertEqual(failed_production_alpha["mismatch_count"], 1)
        self.assertEqual(
            failed_production_alpha["mismatch_counts"],
            {"model_digest": 0, "validation_digest": 0, "prediction_digest": 1},
        )

        projection = {
            "class_observations": tuple(
                {
                    "metric_output_id": metric,
                    "class_label": 0,
                    "condition_id": f"{perturbation}:{alpha}",
                    "perturbation_id": perturbation,
                    "alpha": alpha,
                    "metric_harm": float(metric_index + perturbation_index) / 100 + alpha,
                    "downstream_harm": float(perturbation_index) / 100 + alpha / 10,
                    "state": "complete",
                }
                for metric_index, metric in enumerate(config["metric_output_ids"])
                for perturbation_index, perturbation in enumerate(config["active_perturbation_ids"])
                for alpha in config["alpha_grid"][1:]
            ),
            "alignment_results": tuple(
                {
                    "metric_output_id": metric, "state": "complete",
                    "ag": 0.1 + index / 100, "ag_raw": 0.1 + index / 100,
                    "ag_interval": [0.0, 0.2], "acc_cross": 0.6,
                    "acc_interval": [0.5, 0.7],
                    "d_ag": None if metric == "mse" else 0.01,
                    "d_ag_interval": None if metric == "mse" else [0.0, 0.02],
                    "d_acc": None if metric == "mse" else 0.02,
                    "d_acc_interval": None if metric == "mse" else [0.01, 0.03],
                }
                for index, metric in enumerate(config["metric_output_ids"])
            ),
            "holm_family": tuple(
                {
                    "metric_output_id": metric, "statistic": statistic,
                    "raw_p_value": 0.01, "adjusted_p_value": 0.02,
                    "rank": index * 2 + offset + 1, "family_size": 24,
                    "family_state": "complete", "favorable": True,
                    "rejected": True,
                }
                for index, metric in enumerate(config["metric_output_ids"][1:])
                for offset, statistic in enumerate(("d_ag", "d_acc"))
            ),
        }
        production_metadata = subject._real_artifact_metadata(
            config, bridge, production_alpha, "d" * 64, projection
        )
        verifier_metadata = verifier._real_artifact_metadata(
            config, bridge, verifier_alpha, "d" * 64, projection
        )
        self.assertEqual(production_metadata, verifier_metadata)
        run_id, bridge_doc, preflight, manifest = production_metadata
        expected_run_document = {
            "bridge_sha256": BRIDGE_SHA256,
            "condition_ids": config["condition_ids"],
            "config_sha256": hashlib.sha256(
                (ROOT / "experiments/phase4/configs/d1_protocol_b_full_domain_v1.json").read_bytes()
            ).hexdigest(),
            "model_seeds": config["model_seeds"],
        }
        expected_run_id = "phase4-d1-protocol-b-full-domain-" + hashlib.sha256(
            (json.dumps(expected_run_document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        self.assertEqual(run_id, expected_run_id)
        self.assertIn("parent_checksum_trees", bridge_doc)
        self.assertIn("forbidden_step15_payloads", bridge_doc)
        self.assertEqual(preflight["production_worker_count"], 5)
        self.assertEqual(preflight["verifier_worker_count"], 4)
        self.assertNotIn("worker_count", preflight)
        self.assertEqual(manifest["counts"]["artifact_files"], 24)
        self.assertEqual(manifest["claim_boundary"], config["claim_boundary"])

        production_figures = subject._render_real_protocol_b_figures(projection, None)
        verifier_figures = verifier._render_real(projection)
        self.assertEqual(production_figures, verifier_figures)
        self.assertEqual(
            len(production_figures["figure1_d1_protocol_b_full_domain_data.csv"].splitlines()) - 1,
            520,
        )
        self.assertEqual(
            len(production_figures["figure2_d1_protocol_b_full_domain_data.csv"].splitlines()) - 1,
            13,
        )

    def test_real_failure_artifact_keeps_denominators_and_byte_matches_verifier(self) -> None:
        from types import MappingProxyType

        import rpe.runner.phase4_d1_protocol_b as subject
        import rpe.runner.phase4_d1_protocol_b_verifier as verifier

        config_path = ROOT / "experiments/phase4/configs/d1_protocol_b_full_domain_v1.json"
        raw = config_path.read_bytes()
        config = subject.parse_phase4_d1_protocol_b_config(
            config_path, raw, require_frozen_identity=False
        )
        config_document = json.loads(raw)
        record_ids = tuple(f"test-{index:06d}" for index in range(3000))
        labels = np.repeat(np.arange(30, dtype=np.int64), 100)
        digest = "a" * 64
        code_identity = hashlib.sha256(
            (ROOT / "rpe/runner/phase4_d1_protocol_b.py").read_bytes()
        ).hexdigest()
        alpha0_model_rows = tuple(
            {
                "model_seed": seed, "condition_id": "alpha0",
                "train_condition_id": "alpha0", "validation_condition_id": "alpha0",
                "test_condition_id": "alpha0", "selected_c": 0.01,
                "validation_scores": (), "train_matrix_sha256": digest,
                "validation_matrix_sha256": digest, "pca_train_feature_sha256": digest,
                "pca_validation_feature_sha256": digest, "model_state_sha256": digest,
                "train_record_ids_sha256": digest, "validation_record_ids_sha256": digest,
                "test_record_ids_sha256": digest, "warning_state": "none",
                "refit_with_validation": False,
            }
            for seed in range(5)
        )
        alpha0_validation_rows = tuple(
            {"condition_id": "alpha0", "c": c, "model_seed": seed, "validation_top1_accuracy": 0.5}
            for seed in range(5) for c in (0.01, 0.1, 1.0, 10.0)
        )
        alpha0_prediction_rows = tuple(
            {
                "code_identity": code_identity, "condition_id": "alpha0",
                "config_sha256": config.sha256, "correct": True,
                "model_identity": f"{seed}:alpha0", "model_seed": seed,
                "predicted_class": int(label), "projected_row_sha256": digest,
                "record_id": record_id, "record_order": order, "true_class": int(label),
            }
            for seed in range(5)
            for order, (record_id, label) in enumerate(zip(record_ids, labels, strict=True))
        )
        alpha0_seed_rows = tuple(
            {"model_seed": seed, "class_label": label, "condition_id": "alpha0", "accuracy_seed_class": 1.0}
            for seed in range(5) for label in range(30)
        )
        checksum_trees = {
            "step15": {
                "complete.json": digest, "record_conditions.jsonl": digest,
                "metric_values.jsonl": digest, "peak_receipts.jsonl": digest,
                "model_cells.jsonl": digest,
            },
            "step21": {"complete.json": digest},
            "step23": {"complete.json": digest},
        }
        bridge = {
            "bridge_sha256": BRIDGE_SHA256, "test_record_ids_sha256": TEST_RECORD_ID_DIGEST,
            "condition_ids": tuple(config_document["condition_ids"]),
            "bridge_row_count": 123000, "metric_row_count": 1599000,
            "cwt_row_count": 123000,
            "mismatch_counts": {
                "missing_key_count": 0, "extra_key_count": 0, "duplicate_key_count": 0,
                "field_mismatch_count": 0, "test_order_mismatch_count": 0,
                "state_mismatch_count": 0,
            },
            "parent_checksum_trees": checksum_trees,
        }
        alpha0_receipt = {
            "model_digest": "b" * 64,
            "validation_digest": "c" * 64,
            "prediction_digest": "d" * 64,
            "parent_digests": dict(config_document["alpha0_equivalence"]),
            "expected_digests": dict(config_document["alpha0_equivalence"]),
            "mismatch_counts": {"model_digest": 1, "validation_digest": 1, "prediction_digest": 1},
            "mismatch_count": 3, "status": "failed",
        }
        failure = dict(subject._alpha0_failure_receipt(alpha0_receipt))
        self.assertEqual(failure, dict(verifier._alpha0_failure_receipt(alpha0_receipt)))
        base_test_rows = tuple(
            {"record_id": record_id, "source_row": order, "class_label": int(label)}
            for order, (record_id, label) in enumerate(zip(record_ids, labels, strict=True))
        )
        run_id, production_values = subject._build_real_failure_payloads(
            config, bridge, alpha0_receipt=alpha0_receipt,
            alpha0_model_rows=alpha0_model_rows, alpha0_validation_rows=alpha0_validation_rows,
            alpha0_prediction_rows=alpha0_prediction_rows, alpha0_seed_class_rows=alpha0_seed_rows,
            base_test_rows=base_test_rows, code_identity=code_identity, failure_receipt=failure,
        )
        production_payloads = {
            name: production_values[name] for name in subject.ARTIFACT_PAYLOAD_FILES
        }
        production = {
            **production_payloads,
            "failed.json": production_values["failed.json"],
            "SHA256SUMS": subject.write_sha256sums(
                production_payloads, "failed.json", production_values["failed.json"]
            ),
        }
        verifier_inputs = verifier._RealInputs(
            record_ids, labels, MappingProxyType({}), (), MappingProxyType(bridge)
        )
        fitted = {
            "status": "failed", "failure_receipt": failure,
            "model_cells": alpha0_model_rows, "validation_rows": alpha0_validation_rows,
            "predictions": alpha0_prediction_rows, "seed_class_rows": alpha0_seed_rows,
            "matrix_receipt_sha256": digest,
        }
        rebuilt = dict(
            verifier._serialize_real_failed(
                config_document, verifier_inputs, fitted, alpha0_receipt, failure
            )
        )
        self.assertEqual(production, rebuilt)
        self.assertEqual(
            set(production),
            set(subject.ARTIFACT_PAYLOAD_FILES) | {"failed.json", "SHA256SUMS"},
        )
        self.assertEqual(json.loads(production["failed.json"])["failure"]["mismatch_count"], 3)
        manifest = json.loads(production["manifest.json"])
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["endpoint_state"], "failed_alpha0_equivalence")
        self.assertEqual(manifest["counts"]["predictions"], 615000)
        self.assertEqual(run_id, manifest["run_id"])

    def test_alpha0_equivalence_requires_exact_step21_projection_digests(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import (
            make_synthetic_d1_protocol_b_config,
            make_synthetic_d1_protocol_b_inputs,
            validate_alpha0_equivalence,
        )

        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
            feature_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        receipt = validate_alpha0_equivalence(
            inputs.synthetic_alpha0_model_rows,
            inputs.synthetic_alpha0_validation_rows,
            inputs.synthetic_alpha0_prediction_rows,
            config,
        )
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["mismatch_count"], 0)

        drifted_predictions = list(inputs.synthetic_alpha0_prediction_rows)
        drifted_predictions[0] = {
            **drifted_predictions[0],
            "predicted_class": int(drifted_predictions[0]["predicted_class"]) + 1,
        }
        with self.assertRaisesRegex(Exception, "alpha0|digest|mismatch"):
            validate_alpha0_equivalence(
                inputs.synthetic_alpha0_model_rows,
                inputs.synthetic_alpha0_validation_rows,
                tuple(drifted_predictions),
                config,
            )

    def test_synthetic_build_keeps_exact_inventory_and_worker_invariant_run_identity(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import (
            ARTIFACT_PAYLOAD_FILES,
            build_phase4_d1_protocol_b_from_inputs,
            make_synthetic_d1_protocol_b_config,
            make_synthetic_d1_protocol_b_inputs,
        )

        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
            feature_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        self.assertEqual(len(ARTIFACT_PAYLOAD_FILES), 22)
        with tempfile.TemporaryDirectory() as directory:
            first = build_phase4_d1_protocol_b_from_inputs(
                Path(directory) / "artifact-a",
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            second = build_phase4_d1_protocol_b_from_inputs(
                Path(directory) / "artifact-b",
                inputs=inputs,
                config=config,
                worker_count=2,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            self.assertEqual(first.run_id, second.run_id)
            inventory = {item.name for item in first.path.iterdir()}
            self.assertEqual(len(inventory), 24)
            self.assertEqual(inventory & {"complete.json", "failed.json"}, {"complete.json"})
            self.assertIn("SHA256SUMS", inventory)
            manifest = json.loads((first.path / "manifest.json").read_text(encoding="utf-8"))
            preflight = json.loads((first.path / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(tuple(manifest["payload_files"]), ARTIFACT_PAYLOAD_FILES)
            self.assertEqual(manifest["condition_count"], 41)
            self.assertEqual(manifest["model_seed_count"], 5)
            self.assertEqual(manifest["prediction_row_count"], 820)
            self.assertEqual(manifest["class_observation_count"], 208)
            self.assertEqual(manifest["alpha0_equivalence"]["status"], "passed")
            self.assertEqual(preflight["worker_count"], 1)
            self.assertEqual(preflight["process_start_method"], "spawn")
            self.assertEqual(preflight["blas_thread_limit"], 1)
            self.assertEqual(preflight["admission_bytes_per_job"], WORKER_ADMISSION_BYTES)
            self.assertEqual(preflight["admission_budget_bytes"], ADMISSION_BUDGET_BYTES)
            self.assertEqual(preflight["configured_payload_count"], 22)

    def test_synthetic_build_and_independent_verifier_byte_compare(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import (
            build_phase4_d1_protocol_b_from_inputs,
            make_synthetic_d1_protocol_b_config,
            make_synthetic_d1_protocol_b_inputs,
        )
        from rpe.runner.phase4_d1_protocol_b_verifier import verify_phase4_d1_protocol_b_from_inputs

        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
            feature_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            built = build_phase4_d1_protocol_b_from_inputs(
                path,
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            verified = verify_phase4_d1_protocol_b_from_inputs(
                path,
                inputs=inputs,
                config_path=built.path / "config.json",
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            self.assertEqual(verified.run_id, built.run_id)

    def test_verifier_rebuild_detects_checksum_consistent_semantic_tampering(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import (
            build_phase4_d1_protocol_b_from_inputs,
            make_synthetic_d1_protocol_b_config,
            make_synthetic_d1_protocol_b_inputs,
        )
        from rpe.runner.phase4_d1_protocol_b_authority import canonical_json_bytes
        from rpe.runner.phase4_d1_protocol_b_verifier import (
            Phase4D1ProtocolBVerifierError,
            verify_phase4_d1_protocol_b_from_inputs,
        )

        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
            feature_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            built = build_phase4_d1_protocol_b_from_inputs(
                path,
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            rows = [
                json.loads(line)
                for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["predicted_class"] = int(rows[0]["predicted_class"]) + 1
            (path / "predictions.jsonl").write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
            self._rewrite_sha256sums(path)
            with self.assertRaisesRegex(Phase4D1ProtocolBVerifierError, "semantic|payload|mismatch"):
                verify_phase4_d1_protocol_b_from_inputs(
                    path,
                    inputs=inputs,
                    config_path=built.path / "config.json",
                    worker_count=1,
                    bootstrap_resamples=8,
                    sign_flip_resamples=16,
                )

    def test_public_build_and_cli_expose_no_outcome_affecting_overrides(self) -> None:
        from rpe.runner.phase4_d1_protocol_b import build_phase4_d1_protocol_b
        from tools.run_phase4_d1_protocol_b import main

        with self.assertRaises(SystemExit):
            main(["build", "--output-root", "x", "--bootstrap-resamples", "8"])
        with self.assertRaises(SystemExit):
            main(["verify", "--run-path", "x", "--seed", "0"])
        self.assertEqual(
            inspect.signature(build_phase4_d1_protocol_b).parameters["worker_count"].default,
            5,
        )

    def test_real_production_uses_spawn_pool_and_pre_output_admission(self) -> None:
        import rpe.runner.phase4_d1_protocol_b as subject

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            with patch.object(subject, "load_phase4_d1_protocol_b_config", side_effect=RuntimeError("after admission")):
                with self.assertRaisesRegex(RuntimeError, "after admission"):
                    subject.build_phase4_d1_protocol_b(output, worker_count=5)
            self.assertFalse(output.exists())
        with self.assertRaisesRegex(subject.Phase4D1ProtocolBError, "worker_count"):
            subject.build_phase4_d1_protocol_b(Path("unused"), worker_count=6)

    def test_verifier_is_independent_and_public_surface_is_minimal(self) -> None:
        import rpe.runner.phase4_d1_protocol_b_verifier as verifier

        tree = ast.parse(inspect.getsource(verifier))
        forbidden = "rpe.runner.phase4_d1_protocol_b"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(forbidden, {alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, forbidden)
        self.assertEqual(
            tuple(inspect.signature(verifier.verify_phase4_d1_protocol_b).parameters),
            ("path", "worker_count"),
        )

    def test_verifier_spawn_job_payload_is_pickleable(self) -> None:
        import pickle
        from types import MappingProxyType

        import rpe.runner.phase4_d1_protocol_b_verifier as verifier

        nested = MappingProxyType(
            {
                "train": np.ones((2, 3), dtype=np.float32),
                "validation": np.ones((1, 3), dtype=np.float32),
                "test": np.ones((1, 3), dtype=np.float32),
                "labels": MappingProxyType(
                    {
                        "train": np.asarray([0, 1]),
                        "validation": np.asarray([0]),
                        "test": np.asarray([0]),
                    }
                ),
                "ids": MappingProxyType(
                    {"train": ("a", "b"), "validation": ("c",), "test": ("d",)}
                ),
            }
        )
        payload = verifier._spawn_job_data(nested)
        restored = pickle.loads(pickle.dumps(payload))
        self.assertIsInstance(restored["labels"], dict)
        self.assertIsInstance(restored["ids"], dict)
        np.testing.assert_array_equal(restored["train"], nested["train"])

    def test_package_exports_step25_public_surface(self) -> None:
        import rpe.runner as runner

        self.assertTrue(hasattr(runner, "Phase4D1ProtocolBConfig"))
        self.assertTrue(hasattr(runner, "Phase4D1ProtocolBError"))
        self.assertTrue(hasattr(runner, "Phase4D1ProtocolBSummary"))
        self.assertTrue(hasattr(runner, "build_phase4_d1_protocol_b"))
        self.assertTrue(hasattr(runner, "load_phase4_d1_protocol_b_config"))
        self.assertTrue(hasattr(runner, "verify_phase4_d1_protocol_b"))

