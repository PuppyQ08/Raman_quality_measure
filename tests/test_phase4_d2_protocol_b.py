from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class D2ProtocolBContractRedTest(unittest.TestCase):
    """Focused RED module for Phase 4 Step 19 D2 Protocol-B full-domain work."""

    def test_phase4_d2_protocol_b_module_and_config_exist(self) -> None:
        subject = importlib.import_module("rpe.runner.phase4_d2_protocol_b")
        config_path = ROOT / "experiments/phase4/configs/d2_protocol_b_full_domain_v1.json"
        self.assertTrue(config_path.is_file())
        self.assertTrue(hasattr(subject, "parse_phase4_d2_protocol_b_config"))
        self.assertTrue(hasattr(subject, "build_phase4_d2_protocol_b_from_inputs"))

    def test_production_module_has_no_known_stub_or_forbidden_outcome_assembly_imports(self) -> None:
        """The real path must be independently assembled, not delegated to A/Step-17."""
        source = (ROOT / "rpe/runner/phase4_d2_protocol_b.py").read_text(encoding="utf-8")
        self.assertNotIn("aggregate_d2_protocol_a", source)
        self.assertNotIn("reconstruct_step17_inputs", source)
        self.assertNotIn("tiny_png_bytes", source)
        self.assertNotIn("svg_bytes", source)
        self.assertNotIn("real 5513-source build intentionally", source)

    def test_authority_has_no_placeholder_image_helpers_and_config_has_complete_contract_sections(self) -> None:
        authority_path = ROOT / "rpe/runner/phase4_d2_protocol_b_authority.py"
        authority = authority_path.read_text(encoding="utf-8")
        self.assertNotIn("tiny_png", authority)
        self.assertNotIn("def svg_bytes", authority)
        config = json.loads((ROOT / "experiments/phase4/configs/d2_protocol_b_full_domain_v1.json").read_text(encoding="utf-8"))
        for field in ("authorities", "parent_artifacts", "environment_authority", "model_recipe", "inference", "figure_contract", "inherited_rulings", "code_authority"):
            self.assertIn(field, config)
        from rpe.runner.phase4_d2_protocol_b_authority import CODE_RELATIVE_PATHS

        self.assertEqual(set(config["code_authority"]), set(CODE_RELATIVE_PATHS))
        for relative_path, receipt in config["code_authority"].items():
            raw = (ROOT / relative_path).read_bytes()
            self.assertEqual(receipt, {
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            })
        self.assertEqual(config["expected"]["operator_cell_count"], 27565)
        self.assertEqual(config["expected"]["apply_check_count"], 248085)
        self.assertEqual(config["expected"]["configured_payload_count"], 34)
        self.assertEqual(config["expected"]["artifact_file_count"], 36)
        self.assertEqual(
            config["trust_anchor"]["config_authority_relative_path"],
            "rpe/runner/phase4_d2_protocol_b_authority.py",
        )

    def test_real_parsers_reject_empty_code_authority(self) -> None:
        import rpe.runner.phase4_d2_protocol_b as production
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier
        from rpe.runner.phase4_d2_protocol_b_authority import canonical_json_bytes

        config_path = ROOT / "experiments/phase4/configs/d2_protocol_b_full_domain_v1.json"
        document = json.loads(config_path.read_text(encoding="utf-8"))
        document["code_authority"] = {}
        raw = canonical_json_bytes(document)
        with self.assertRaisesRegex(production.Phase4D2ProtocolBError, "code_authority"):
            production.parse_phase4_d2_protocol_b_config(
                config_path, raw, require_frozen_identity=False
            )
        with self.assertRaisesRegex(verifier.Phase4D2ProtocolBVerifierError, "code_authority"):
            verifier._parse_config(config_path, raw, frozen=False)

    def test_real_rematerialization_uses_bounded_spawn_pool_and_canonical_collection(self) -> None:
        source = (ROOT / "rpe/runner/phase4_d2_protocol_b.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        pool_calls = [
            node for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "ProcessPoolExecutor"
        ]
        self.assertEqual(len(pool_calls), 1)
        keywords = {item.arg: item.value for item in pool_calls[0].keywords}
        self.assertIn("max_workers", keywords)
        self.assertIn("mp_context", keywords)
        self.assertIn("initializer", keywords)
        self.assertIn("initargs", keywords)
        self.assertIn('multiprocessing.get_context("spawn")', source)
        self.assertIn("capacity = budget // 1105805824", source)
        self.assertIn("threadpool_limits(limits=1, user_api=\"blas\")", source)
        self.assertIn("executor.map(_d2_protocol_b_source_job, jobs)", source)
        self.assertIn("order != expected_order", source)

    def test_real_build_resolves_frozen_parent_artifact_paths(self) -> None:
        """The real build must consume the parent paths frozen in parent_artifacts."""
        import rpe.runner.phase4_d2_protocol_b as subject

        config = subject.load_phase4_d2_protocol_b_config(
            ROOT / "experiments/phase4/configs/d2_protocol_b_full_domain_v1.json"
        )

        class StopAfterBridge(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            subject, "build_d2_protocol_b_authority_bridge", side_effect=StopAfterBridge
        ) as bridge:
            with self.assertRaises(StopAfterBridge):
                subject.build_phase4_d2_protocol_b_from_inputs(
                    Path(directory) / "artifact",
                    inputs=object(),
                    config=config,
                    worker_count=16,
                )

        call = bridge.call_args.kwargs
        self.assertEqual(
            call["protocol_a_path"],
            ROOT / config.document["parent_artifacts"]["protocol_a"]["relative_path"],
        )
        self.assertEqual(
            call["eligibility_path"],
            ROOT / config.document["parent_artifacts"]["eligibility"]["relative_path"],
        )

    def test_l2_recipe_is_warning_free_in_production_and_verifier(self) -> None:
        """The frozen L2 recipe must not trip sklearn's deprecated-API warning."""
        import rpe.runner.phase4_d2_protocol_b as production
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        rng = np.random.default_rng(20260817)
        class_count = 3
        feature_count = 8
        centers = np.eye(class_count, feature_count, dtype=np.float64) * 12.0
        train_labels = np.repeat(np.arange(class_count), 5)
        validation_labels = np.repeat(np.arange(class_count), 2)
        test_labels = np.repeat(np.arange(class_count), 2)
        train = np.asarray(np.vstack(
            [centers[label] + rng.normal(0.0, 0.1, feature_count) for label in train_labels]
        ), dtype="<f4")
        validation = np.asarray(np.vstack(
            [centers[label] + rng.normal(0.0, 0.1, feature_count) for label in validation_labels]
        ), dtype="<f4")
        test = np.asarray(np.vstack(
            [centers[label] + rng.normal(0.0, 0.1, feature_count) for label in test_labels]
        ), dtype="<f4")
        inputs = production.D2ProtocolBSyntheticInputs(
            class_count=class_count,
            records_per_class=2,
            model_seeds=(0,),
            record_ids=tuple(f"test-{index}" for index in range(test.shape[0])),
            test_labels=test_labels,
            train_values={(5, 0, "alpha0"): train},
            train_labels={(5, 0): train_labels},
            validation_values={(5, 0, "alpha0"): validation},
            validation_labels={(5, 0): validation_labels},
            test_values={"alpha0": test},
            parent_step15_payloads={},
            parent_step17_payloads={},
            source_records=({"record_id": "enable-real-warning-gate"},),
        )

        produced = production._fit_predict_for_cell(5, 0, "alpha0", inputs)
        independently_rebuilt = verifier._fit_cell(
            5, 0, "alpha0", inputs, SimpleNamespace(class_count=class_count, synthetic=False)
        )
        from sklearn.decomposition import PCA

        expected_features = PCA(
            n_components=8, svd_solver="randomized", whiten=False, random_state=0
        ).fit_transform(train)
        self.assertEqual(
            produced[0]["train_matrix_sha256"], hashlib.sha256(train.tobytes()).hexdigest()
        )
        self.assertEqual(
            produced[0]["pca_train_feature_sha256"],
            hashlib.sha256(
                np.ascontiguousarray(expected_features, dtype="<f8").tobytes()
            ).hexdigest(),
        )
        self.assertEqual(produced, independently_rebuilt)

    def test_real_reconstruction_preserves_frozen_model_role_order(self) -> None:
        """Model matrices use selection order while Step-17 ledgers stay canonical."""
        import rpe.runner.phase4_d2_protocol_b as production
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        config_path = ROOT / "experiments/phase4/configs/d2_protocol_b_full_domain_v1.json"
        selection_path = (
            ROOT
            / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json"
        )
        dataset_path = ROOT / "data/unified/bacteria_id_reference"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        seed_zero = selection["selections"][0]
        expected_train = tuple(
            str(record_id)
            for class_doc in seed_zero["classes"]
            for record_id in class_doc["train_record_ids"]["5"]
        )
        expected_validation = tuple(
            str(record_id)
            for class_doc in seed_zero["classes"]
            for record_id in class_doc["validation_record_ids"]
        )

        production_config = production.load_phase4_d2_protocol_b_config(config_path)
        production_inputs = production.reconstruct_d2_protocol_b_outcome_inputs(
            dataset_path, selection_path, production_config
        )
        production_roles = production_inputs.parent_step17_payloads["role_lookup"]
        self.assertEqual(production_roles[(5, 0, "train")], expected_train)
        self.assertEqual(production_roles[(5, 0, "validation")], expected_validation)
        self.assertEqual(
            production_inputs.model_cells[0]["train_record_ids"],
            tuple(sorted(expected_train)),
        )

        verifier_config = verifier._parse_config(
            config_path, config_path.read_bytes(), frozen=True
        )
        verifier_inputs = verifier._reconstruct_real_inputs(verifier_config)
        self.assertEqual(verifier_inputs.role_lookup[(5, 0, "train")], expected_train)
        self.assertEqual(
            verifier_inputs.role_lookup[(5, 0, "validation")], expected_validation
        )
        self.assertEqual(
            verifier_inputs.model_cells[0]["train_record_ids"],
            tuple(sorted(expected_train)),
        )

    def test_verifier_parent_rows_round_trip_through_canonical_serializer(self) -> None:
        """Validated parent JSONL rows must remain canonically serializable."""
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier
        from rpe.runner.phase4_d2_protocol_b_authority import jsonl_bytes

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.jsonl"
            expected = b'{"condition_id":"alpha0","record_id":"r0"}\n'
            path.write_bytes(expected)
            self.assertEqual(jsonl_bytes(verifier._read_jsonl(path)), expected)

    def test_verifier_prediction_aggregation_is_single_pass(self) -> None:
        """Independent aggregation must not repeatedly scan 1.845M predictions."""
        import rpe.runner.phase4_d2_protocol_b as production
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        class CountingSequence:
            def __init__(self, rows):
                self.rows = tuple(rows)
                self.iterations = 0

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

            def __iter__(self):
                self.iterations += 1
                return iter(self.rows)

        inputs = production.make_synthetic_d2_protocol_b_inputs(
            class_count=2, records_per_class=2, model_seeds=(0,)
        )
        production_config = production.make_synthetic_d2_protocol_b_config(inputs)
        config = verifier._parse_config(
            Path("<synthetic>"), production_config.raw_bytes, frozen=False
        )
        bridge = verifier._bridge(inputs, config)
        fitted = verifier._fit(inputs, config)

        aggregate_rows = CountingSequence(fitted["predictions"])
        projection = verifier._aggregate(
            aggregate_rows, bridge["metrics"], config, 2, 4
        )
        self.assertEqual(aggregate_rows.iterations, 1)
        self.assertEqual(
            len(projection["class_observations"]),
            config.expected["class_observation_count"],
        )

        summary_rows = CountingSequence(fitted["predictions"])
        summary = verifier._condition_summary(summary_rows, config)
        self.assertEqual(summary_rows.iterations, 1)
        self.assertEqual(len(summary), 3 * len(config.conditions))

    def test_verifier_downstream_harm_preserves_seed_first_reduction_order(self) -> None:
        """Aggregate seed accuracies separately before taking their contrast."""
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        condition = verifier.condition_ids()[1]
        baseline_correct = (64, 58, 44, 33, 60)
        current_correct = (56, 34, 44, 33, 61)
        predictions = []
        for shot in verifier.SHOTS:
            for seed, (baseline_count, current_count) in enumerate(
                zip(baseline_correct, current_correct, strict=True)
            ):
                for condition_id, correct_count in (
                    ("alpha0", baseline_count),
                    (condition, current_count),
                ):
                    predictions.extend(
                        {
                            "shot_count": shot,
                            "model_seed": seed,
                            "true_class": 0,
                            "condition_id": condition_id,
                            "record_order": order,
                            "correct": order < correct_count,
                        }
                        for order in range(100)
                    )
        metrics = tuple(
            {
                "record_order": order,
                "condition_id": condition_id,
                "metric_output_id": metric,
                "value": 0.0 if condition_id == "alpha0" else 1.0,
            }
            for order in range(100)
            for condition_id in ("alpha0", condition)
            for metric in verifier.METRIC_OUTPUT_IDS
        )
        config = SimpleNamespace(
            seeds=tuple(range(5)),
            conditions=("alpha0", condition),
            class_count=1,
            records_per_class=100,
        )
        projection = verifier._aggregate(predictions, metrics, config, 2, 4)
        expected = float(
            np.mean(np.asarray(baseline_correct, dtype=np.float64) / 100.0)
            - np.mean(np.asarray(current_correct, dtype=np.float64) / 100.0)
        )
        self.assertEqual(projection["class_observations"][0]["downstream_harm"], expected)

    def test_verifier_orients_metric_values_before_mean_without_negative_zero(self) -> None:
        """Higher-is-better zero harms serialize as canonical positive zero."""
        import math
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        condition = verifier.condition_ids()[1]
        predictions = tuple(
            {
                "shot_count": shot,
                "model_seed": seed,
                "true_class": 0,
                "condition_id": condition_id,
                "record_order": order,
                "correct": True,
            }
            for shot in verifier.SHOTS
            for seed in (0,)
            for condition_id in ("alpha0", condition)
            for order in (0,)
        )
        metrics = tuple(
            {
                "record_order": 0,
                "condition_id": condition_id,
                "metric_output_id": metric,
                "value": 0.0,
            }
            for condition_id in ("alpha0", condition)
            for metric in verifier.METRIC_OUTPUT_IDS
        )
        config = SimpleNamespace(
            seeds=(0,), conditions=("alpha0", condition),
            class_count=1, records_per_class=1,
        )
        projection = verifier._aggregate(predictions, metrics, config, 2, 4)
        pearson = next(
            row for row in projection["class_observations"]
            if row["shot_count"] == 5 and row["metric_output_id"] == "pearson_r"
        )
        self.assertEqual(pearson["metric_harm"], 0.0)
        self.assertEqual(math.copysign(1.0, pearson["metric_harm"]), 1.0)

    def test_cli_lazily_imports_verifier_only_for_verify_branch(self) -> None:
        source = (ROOT / "tools/run_phase4_d2_protocol_b.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_imports = [
            node for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase4_d2_protocol_b_verifier"
        ]
        self.assertEqual(top_level_imports, [])
        subject = importlib.import_module("tools.run_phase4_d2_protocol_b")
        self.assertTrue(callable(subject.main))
        summary = SimpleNamespace(
            path=Path("artifact/run"), run_id="run-id", status="complete",
            prediction_row_count=1845000, class_observation_count=46800,
        )
        with mock.patch.object(subject, "build_phase4_d2_protocol_b", return_value=summary) as build:
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(subject.main(["build", "--output-root", "artifact"]), 0)
        build.assert_called_once_with(Path("artifact"), worker_count=16)
        self.assertEqual(json.loads(output.getvalue())["run_id"], "run-id")

    def test_synthetic_contract_covers_bridge_condition_major_alpha0_and_byte_identity(self) -> None:
        from rpe.runner.phase4_d2_protocol_b import (
            Phase4D2ProtocolBError,
            build_phase4_d2_protocol_b_from_inputs,
            build_d2_protocol_b_authority_bridge,
            make_synthetic_d2_protocol_b_config,
            make_synthetic_d2_protocol_b_inputs,
            validate_alpha0_equivalence,
        )
        from rpe.runner.phase4_d2_protocol_b_verifier import (
            Phase4D2ProtocolBVerifierError,
            verify_phase4_d2_protocol_b_from_inputs,
        )

        inputs = make_synthetic_d2_protocol_b_inputs(class_count=2, records_per_class=2, model_seeds=(0, 1))
        config = make_synthetic_d2_protocol_b_config(inputs)

        bridge = build_d2_protocol_b_authority_bridge(
            inputs=inputs,
            protocol_a_path=Path("synthetic_protocol_a"),
            eligibility_path=Path("synthetic_eligibility"),
            config=config,
        )
        self.assertEqual(bridge.condition_bridge_sha256, config.condition_bridge_sha256)
        self.assertEqual(len(bridge.metric_rows), config.expected_metric_row_count)
        self.assertEqual(len(bridge.cwt_rows), config.expected_cwt_row_count)
        self.assertEqual(len(bridge.bridge_rows), config.expected_bridge_row_count)
        self.assertEqual(
            tuple(row["record_order"] for row in bridge.bridge_rows[: len(config.condition_ids)]),
            (0,) * len(config.condition_ids),
        )
        self.assertEqual(
            tuple(row["condition_id"] for row in bridge.bridge_rows[: len(config.condition_ids)]),
            config.condition_ids,
        )

        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "artifact"
            artifact_root.mkdir()
            (artifact_root / "superseded-run").mkdir()
            built = build_phase4_d2_protocol_b_from_inputs(
                artifact_root,
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            manifest = json.loads((built.path / "manifest.json").read_text(encoding="utf-8"))
            preflight = json.loads((built.path / "preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["prediction_row_count"], config.expected_prediction_row_count)
            self.assertEqual(
                manifest["alpha0_equivalence"]["prediction_digest"],
                validate_alpha0_equivalence(built.predictions, bridge, config)["prediction_digest"],
            )
            for field in (
                "artifact_schema_version", "authorities", "claim_boundary",
                "code", "config", "config_authority", "counts",
                "environment", "inherited_rulings", "metric_states",
                "parent_artifacts", "run_identity",
            ):
                self.assertIn(field, manifest)
            self.assertEqual(manifest["counts"]["configured_payloads"], 34)
            self.assertEqual(manifest["counts"]["artifact_files"], 36)
            for field in (
                "authority_bridge_state", "parent_artifacts",
                "shot_gate_states", "rematerialization",
            ):
                self.assertIn(field, preflight)
            self.assertEqual(preflight["rematerialization"]["receipt_mismatch_count"], 0)
            self.assertEqual(preflight["rematerialization"]["matrix_input_state"], "ready")

            verified = verify_phase4_d2_protocol_b_from_inputs(
                built.path,
                inputs=inputs,
                config_path=built.path / "config.json",
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            self.assertEqual(verified.run_id, built.run_id)

            rows = (built.path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            changed = json.loads(rows[0])
            changed["predicted_class"] = "tampered"
            rows[0] = json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            (built.path / "predictions.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            sums = []
            for line in (built.path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                _digest, name = line.split("  ", 1)
                sums.append(f"{hashlib.sha256((built.path / name).read_bytes()).hexdigest()}  {name}")
            (built.path / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Phase4D2ProtocolBVerifierError, "semantic|rebuild|payload"):
                verify_phase4_d2_protocol_b_from_inputs(
                    built.path,
                    inputs=inputs,
                    config_path=built.path / "config.json",
                    worker_count=1,
                    bootstrap_resamples=8,
                    sign_flip_resamples=16,
                )

        with self.assertRaisesRegex(Phase4D2ProtocolBError, "forbidden Step-15"):
            make_synthetic_d2_protocol_b_inputs(
                class_count=2,
                records_per_class=2,
                model_seeds=(0,),
                include_forbidden_parent_payload=True,
            )

    def test_public_surfaces_pin_config_cli_and_worker_contract(self) -> None:
        from rpe.runner.phase4_d2_protocol_b import (
            Phase4D2ProtocolBError,
            make_synthetic_d2_protocol_b_config,
            make_synthetic_d2_protocol_b_inputs,
            reconstruct_d2_protocol_b_outcome_inputs,
        )
        from tools.run_phase4_d2_protocol_b import main

        inputs = make_synthetic_d2_protocol_b_inputs(class_count=2, records_per_class=2, model_seeds=(0,))
        config = make_synthetic_d2_protocol_b_config(inputs)
        with self.assertRaisesRegex(Phase4D2ProtocolBError, "forbidden Phase-0.5"):
            reconstruct_d2_protocol_b_outcome_inputs(Path("complete_cells.json"), Path("selection.json"), config)
        with self.assertRaises(SystemExit):
            main(["build", "--output-root", "x", "--bootstrap-resamples", "8"])

    def test_verifier_has_firewall_against_production_outcome_module(self) -> None:
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        tree = ast.parse(inspect.getsource(verifier))
        forbidden = "rpe.runner.phase4_d2_protocol_b"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(forbidden, {alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, forbidden)

    def test_verifier_public_path_owns_real_reconstruction_chain(self) -> None:
        """The public verifier must rebuild science; a fail-close stub is insufficient."""
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        source = inspect.getsource(verifier)
        self.assertNotIn("real verification requires retained-data parent reconstruction", source)
        for symbol in (
            "_reconstruct_real_inputs",
            "_validate_real_parents_and_bridge",
            "_rematerialize_real_science",
            "_fit_real_cells",
            "_validate_real_alpha0_equivalence",
        ):
            self.assertTrue(callable(getattr(verifier, symbol, None)), symbol)

    def test_verifier_real_alpha0_compares_all_three_parent_projections(self) -> None:
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier
        from rpe.runner.phase4_d2_protocol_b_authority import jsonl_bytes, sha256_hex

        model = {
            "shot_count": 5, "model_seed": 0, "condition_id": "alpha0",
            "selected_c": 0.1, "train_matrix_sha256": "train",
            "validation_matrix_sha256": "validation",
            "pca_train_feature_sha256": "pca-train",
            "pca_validation_feature_sha256": "pca-validation",
            "train_record_ids_sha256": "train-ids",
            "validation_record_ids_sha256": "validation-ids",
            "model_state_sha256": "model", "warning_state": "none",
        }
        validation = {
            "shot_count": 5, "model_seed": 0, "condition_id": "alpha0",
            "c": 0.1, "validation_top1_accuracy": 0.75,
        }
        prediction = {
            "shot_count": 5, "model_seed": 0, "condition_id": "alpha0",
            "record_order": 0, "record_id": "test-0", "true_class": 0,
            "predicted_class": 0, "correct": True,
            "projected_row_sha256": "projection",
        }
        fitted = {
            "model_cells": (model,), "validation_rows": (validation,),
            "predictions": (prediction,),
        }
        projections = verifier._alpha0_projections(fitted)
        expected = {name: sha256_hex(jsonl_bytes(rows)) for name, rows in projections.items()}
        config = SimpleNamespace(document={"alpha0_equivalence": expected})
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent_model = {key: value for key, value in model.items() if key != "condition_id"}
            parent_validation = {key: value for key, value in validation.items() if key != "condition_id"}
            (parent / "model_cells.jsonl").write_bytes(jsonl_bytes((parent_model,)))
            (parent / "validation_scores.jsonl").write_bytes(jsonl_bytes((parent_validation,)))
            (parent / "predictions.jsonl").write_bytes(jsonl_bytes((prediction,)))
            receipt = verifier._validate_real_alpha0_equivalence(fitted, parent, config)
            self.assertEqual(receipt["mismatch_count"], 0)
            changed = dict(prediction, predicted_class=1, correct=False)
            (parent / "predictions.jsonl").write_bytes(jsonl_bytes((changed,)))
            with self.assertRaisesRegex(verifier.Phase4D2ProtocolBVerifierError, "alpha-zero"):
                verifier._validate_real_alpha0_equivalence(fitted, parent, config)

    def test_public_verifier_rebuilds_and_byte_compares_complete_artifact(self) -> None:
        import rpe.runner.phase4_d2_protocol_b_verifier as verifier

        config = SimpleNamespace(
            document={"inference": {"bootstrap_resamples": 2, "sign_flip_resamples": 4}},
        )
        manifest = {
            "run_id": "rebuilt-run", "status": "complete",
            "prediction_row_count": 1845000, "class_observation_count": 46800,
        }
        rebuilt = {"manifest.json": b"rebuilt"}
        with mock.patch.object(verifier, "_parse_config", return_value=config), \
             mock.patch.object(verifier, "_validate_inventory"), \
             mock.patch.object(verifier, "_reconstruct_real_inputs", return_value=object()), \
             mock.patch.object(verifier, "_validate_real_parents_and_bridge", return_value={"protocol_a": Path("parent"), "metrics": (), "document": {}, "digest": "bridge"}), \
             mock.patch.object(verifier, "_rematerialize_real_science", return_value={}) as rematerialize, \
             mock.patch.object(verifier, "_fit_real_cells", return_value={"model_cells": (), "validation_rows": (), "predictions": (), "seed_class_rows": ()}), \
             mock.patch.object(verifier, "_validate_real_alpha0_equivalence", return_value={}), \
             mock.patch.object(verifier, "_aggregate", return_value={}), \
             mock.patch.object(verifier, "_serialize_rebuild", return_value=(rebuilt, manifest)) as serialize, \
             mock.patch.object(verifier, "_compare") as compare:
            summary = verifier.verify_phase4_d2_protocol_b(Path("artifact"), worker_count=12)
        serialize.assert_called_once()
        rematerialize.assert_called_once()
        self.assertEqual(rematerialize.call_args.kwargs["worker_count"], 12)
        self.assertEqual(serialize.call_args.kwargs["worker_count"], 16)
        compare.assert_called_once_with(Path("artifact"), rebuilt)
        self.assertEqual(summary.run_id, "rebuilt-run")

    def test_rendering_is_byte_deterministic_and_verifier_has_no_svg_or_checksum_waiver(self) -> None:
        """A different build process must reproduce SVG and checksum bytes exactly."""
        from rpe.runner.phase4_d2_protocol_b import (
            aggregate_d2_protocol_b,
            fit_predict_d2_protocol_b,
            make_synthetic_d2_protocol_b_config,
            make_synthetic_d2_protocol_b_inputs,
            rematerialize_d2_protocol_b_science,
            render_d2_protocol_b_figures,
        )

        inputs = make_synthetic_d2_protocol_b_inputs(
            class_count=2, records_per_class=2, model_seeds=(0, 1)
        )
        config = make_synthetic_d2_protocol_b_config(inputs)
        science = rematerialize_d2_protocol_b_science(inputs, config, worker_count=1)
        fitted = fit_predict_d2_protocol_b(inputs, science, config)
        from rpe.runner.phase4_d2_protocol_b import build_d2_protocol_b_authority_bridge
        bridge = build_d2_protocol_b_authority_bridge(
            inputs=inputs, protocol_a_path=Path("synthetic_protocol_a"),
            eligibility_path=Path("synthetic_eligibility"), config=config,
        )
        projection = aggregate_d2_protocol_b(fitted["predictions"], bridge.metric_rows, config)
        first = render_d2_protocol_b_figures(projection, config)
        second = render_d2_protocol_b_figures(projection, config)
        self.assertEqual(first, second)

        verifier_source = (
            ROOT / "rpe/runner/phase4_d2_protocol_b_verifier.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("_canonical_svg", verifier_source)
        self.assertNotIn('name.endswith(".svg")', verifier_source)
        self.assertNotIn('name == "SHA256SUMS"', verifier_source)


if __name__ == "__main__":
    unittest.main()
