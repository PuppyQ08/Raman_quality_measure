from __future__ import annotations

import json
import ast
import hashlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class D2ProtocolAContractTest(unittest.TestCase):
    """Protects the D2 outcome contracts that would otherwise drift silently."""

    def test_inputs_retain_native_test_spectra_and_legacy_proxy_science_is_gone(self) -> None:
        """P8--P12 must start from native spectra, never a projected cosine proxy."""
        import rpe.runner.phase4_d2_protocol_a as subject
        from rpe.evaluation import Spectrum1D

        inputs = subject.make_synthetic_d2_protocol_a_inputs(
            class_count=2, records_per_class=2, model_seeds=(0,)
        )
        self.assertFalse(hasattr(subject, "_science"))
        self.assertFalse(hasattr(subject, "_projection"))
        self.assertEqual(len(inputs.native_test_spectra), len(inputs.record_ids))
        self.assertEqual(inputs.native_test_labels.tolist(), inputs.test_labels.tolist())
        self.assertTrue(all(isinstance(item, Spectrum1D) for item in inputs.native_test_spectra))
        self.assertTrue(all((__import__("numpy").diff(item.axis_cm1) > 0.0).all() for item in inputs.native_test_spectra))

    def test_synthetic_full_chain_keeps_shared_science_and_separate_shots(self) -> None:
        # A regression that multiplied science by shot/seed would change these
        # literal denominators; a pooled endpoint would lose the three keys.
        from rpe.runner.phase4_d2_protocol_a import (
            build_phase4_d2_protocol_a_from_inputs,
            make_synthetic_d2_protocol_a_inputs,
            make_synthetic_d2_protocol_a_config,
        )

        inputs = make_synthetic_d2_protocol_a_inputs(
            class_count=2, records_per_class=2, model_seeds=(0, 1)
        )
        config = make_synthetic_d2_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            summary = build_phase4_d2_protocol_a_from_inputs(
                Path(directory) / "artifact", inputs=inputs, config=config, worker_count=1, inference_resamples=8
            )
            manifest = json.loads((summary.path / "manifest.json").read_text())
            self.assertEqual(manifest["shared_condition_count"], 164)
            self.assertEqual(manifest["shared_metric_value_count"], 2132)
            self.assertEqual(manifest["shared_cwt_receipt_count"], 164)
            self.assertEqual(manifest["model_fit_count"], 6)
            self.assertEqual(manifest["lr_candidate_fit_count"], 24)
            self.assertEqual(manifest["prediction_row_count"], 984)
            self.assertEqual(manifest["seed_class_condition_count"], 492)
            self.assertEqual(tuple(sorted(manifest["shot_endpoint_states"], key=int)), ("5", "10", "20"))
            inventory = {path.name for path in summary.path.iterdir()}
            self.assertEqual(len(inventory), 38)
            self.assertEqual(len(inventory & {"complete.json", "failed.json"}), 1)

    def test_config_forbidden_phase05_and_cli_override_contracts(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            Phase4D2ProtocolAError,
            make_synthetic_d2_protocol_a_config,
            make_synthetic_d2_protocol_a_inputs,
            reconstruct_d2_protocol_a_inputs,
        )
        from tools.run_phase4_d2_protocol_a import main

        inputs = make_synthetic_d2_protocol_a_inputs(class_count=2, records_per_class=2, model_seeds=(0,))
        config = make_synthetic_d2_protocol_a_config(inputs)
        with self.assertRaisesRegex(Phase4D2ProtocolAError, "forbidden Phase-0.5"):
            reconstruct_d2_protocol_a_inputs(Path("complete_cells.json"), Path("selection.json"), config)
        with self.assertRaises(SystemExit):
            main(["build", "--output-root", "x", "--inference-resamples", "8"])

    def test_verifier_is_independent_and_byte_compares_synthetic_artifact(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            build_phase4_d2_protocol_a_from_inputs,
            make_synthetic_d2_protocol_a_config,
            make_synthetic_d2_protocol_a_inputs,
        )
        from rpe.runner.phase4_d2_protocol_a_verifier import (
            verify_phase4_d2_protocol_a_from_inputs,
        )

        inputs = make_synthetic_d2_protocol_a_inputs(class_count=2, records_per_class=2, model_seeds=(0, 1))
        config = make_synthetic_d2_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            built = build_phase4_d2_protocol_a_from_inputs(
                path, inputs=inputs, config=config, worker_count=1, inference_resamples=8
            )
            verified = verify_phase4_d2_protocol_a_from_inputs(
                path, inputs=inputs, config_path=built.path / "config.json", worker_count=1, inference_resamples=8
            )
            self.assertEqual(verified.run_id, built.run_id)

    def test_verifier_forbids_the_production_outcome_module(self) -> None:
        """The independent verifier may share primitives, never D2 outcome assembly."""
        import rpe.runner.phase4_d2_protocol_a_verifier as verifier

        tree = ast.parse(inspect.getsource(verifier))
        forbidden = "rpe.runner.phase4_d2_protocol_a"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(forbidden, {alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, forbidden)

    def test_verifier_aggregate_separates_bootstrap_and_sign_flip_resamples(self) -> None:
        """Real inference authorities must never share the 2,000 bootstrap count."""
        import numpy as np
        import rpe.runner.phase4_d2_protocol_a_verifier as verifier

        predictions, metrics = [], []
        conditions = verifier._conditions()
        for condition_id, perturbation, _alpha in conditions:
            for order, label in enumerate([label for label in range(30) for _ in range(2)]):
                for shot in verifier.SHOTS:
                    for seed in (0, 1):
                        predictions.append({
                            "shot_count": shot, "model_seed": seed, "true_class": label,
                            "condition_id": condition_id, "record_order": order,
                            "correct": bool(perturbation is None or (order + seed + shot + int(_alpha * 100)) % 5 != 0),
                        })
                for metric in verifier.METRICS:
                    metrics.append({
                        "record_order": order, "condition_id": condition_id,
                        "metric_output_id": metric, "value": 0.0 if perturbation is None else float((order + 1) * _alpha),
                    })
        _obs, _alignment, bootstrap, signs, _holm = verifier._aggregate(
            predictions, metrics, SimpleNamespace(class_count=30, seeds=(0, 1)),
            bootstrap_resamples=3, sign_flip_resamples=7,
        )
        self.assertTrue(all(row["resamples"] in {None, 3} for row in bootstrap))
        self.assertTrue(all(row["resamples"] in {None, 7} for row in signs))

    def test_non_synthetic_science_uses_spawn_capacity_limited_workers(self) -> None:
        """The retained-data route must spawn only the admitted worker count."""
        import numpy as np
        import rpe.runner.phase4_d2_protocol_a_verifier as verifier
        from rpe.evaluation import Spectrum1D

        axis = np.linspace(350.0, 1830.0, 4, dtype=np.float64)
        spectra = tuple(
            Spectrum1D(
                spectrum_id=f"test::{index}", sample_id=str(index), axis_cm1=axis,
                intensity=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            )
            for index in range(3)
        )
        inputs = SimpleNamespace(
            support_axis=axis, native_labels=np.asarray([0, 1, 0]),
            native_spectra=spectra, record_ids=("a", "b", "c"),
            test_labels=np.asarray([0, 1, 0]),
        )
        calls = {}

        class FakeExecutor:
            def __init__(self, *, max_workers, mp_context, initializer, initargs):
                calls.update(max_workers=max_workers, mp_context=mp_context, initializer=initializer, initargs=initargs)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def map(self, function, jobs):
                calls["function"] = function
                return tuple(function(job) for job in reversed(tuple(jobs)))

        def worker(job):
            order, _record_id, _label, _spectrum = job
            return order, {condition: np.asarray([order], dtype=np.float32) for condition, _, _ in verifier._conditions()}, [], [], []

        budget = 64 * 1024**3
        with patch.object(verifier, "estimate_p10_peak_bytes", return_value=budget // 2), \
             patch.object(verifier, "ProcessPoolExecutor", FakeExecutor), \
             patch.object(verifier.multiprocessing, "get_context", return_value="spawn-context") as get_context, \
             patch.object(verifier, "_real_science_worker", side_effect=worker), \
             patch.object(verifier, "load_perturbation_sweep_config"), \
             patch.object(verifier, "load_phase1_core_config"), \
             patch.object(verifier, "load_classical_catalog", return_value=SimpleNamespace(systems=())):
            matrices, conditions, metrics, peaks = verifier._real_science(
                inputs, SimpleNamespace(synthetic=False), worker_count=9
            )
        self.assertEqual(calls["max_workers"], 2)
        self.assertEqual(calls["mp_context"], "spawn-context")
        self.assertIs(calls["initializer"], verifier._initialize_real_science_worker)
        self.assertTrue(calls["function"].called)
        get_context.assert_called_once_with("spawn")
        self.assertEqual([float(row[0]) for row in matrices["alpha0"]], [0.0, 1.0, 2.0])
        self.assertEqual((conditions, metrics, peaks), ((), (), ()))

    def test_verifier_rebuild_detects_checksum_consistent_semantic_tampering(self) -> None:
        """A new checksum cannot bless a changed prediction/summary payload."""
        from rpe.runner.phase4_d2_protocol_a import (
            build_phase4_d2_protocol_a_from_inputs,
            make_synthetic_d2_protocol_a_config,
            make_synthetic_d2_protocol_a_inputs,
        )
        from rpe.runner.phase4_d2_protocol_a_verifier import (
            Phase4D2ProtocolAVerifierError, verify_phase4_d2_protocol_a_from_inputs,
        )

        inputs = make_synthetic_d2_protocol_a_inputs(class_count=2, records_per_class=2, model_seeds=(0, 1))
        config = make_synthetic_d2_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            built = build_phase4_d2_protocol_a_from_inputs(
                path, inputs=inputs, config=config, worker_count=1, inference_resamples=8
            )
            rows = (path / "predictions.jsonl").read_text().splitlines()
            changed = json.loads(rows[0]); changed["correct"] = not changed["correct"]
            rows[0] = json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            (path / "predictions.jsonl").write_text("\n".join(rows) + "\n")
            sums = []
            for line in (path / "SHA256SUMS").read_text().splitlines():
                _digest, name = line.split("  ", 1)
                sums.append(f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}")
            (path / "SHA256SUMS").write_text("\n".join(sums) + "\n")
            with self.assertRaisesRegex(Phase4D2ProtocolAVerifierError, "rebuild|semantic|payload"):
                verify_phase4_d2_protocol_a_from_inputs(
                    built.path, inputs=inputs, config_path=built.path / "config.json",
                    worker_count=1, inference_resamples=8,
                )

    def test_verifier_supports_failed_terminal_marker_and_public_surface_has_no_bypass(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            build_phase4_d2_protocol_a_from_inputs,
            make_synthetic_d2_protocol_a_config,
            make_synthetic_d2_protocol_a_inputs,
        )
        import rpe.runner.phase4_d2_protocol_a_verifier as verifier

        self.assertEqual(tuple(inspect.signature(verifier.verify_phase4_d2_protocol_a).parameters), ("path", "worker_count"))
        self.assertNotIn("skip", inspect.getsource(verifier.verify_phase4_d2_protocol_a))
        self.assertNotIn("bypass", inspect.getsource(verifier.verify_phase4_d2_protocol_a))
        inputs = make_synthetic_d2_protocol_a_inputs(class_count=2, records_per_class=2, model_seeds=(0, 1))
        config = make_synthetic_d2_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            built = build_phase4_d2_protocol_a_from_inputs(
                Path(directory) / "artifact", inputs=inputs, config=config, worker_count=1, inference_resamples=8
            )
            verified = verifier.verify_phase4_d2_protocol_a_from_inputs(
                built.path, inputs=inputs, config_path=built.path / "config.json", worker_count=1, inference_resamples=8
            )
            self.assertIn(verified.status, {"complete", "failed"})

    def test_real_config_freeze_requires_step14_authority_contracts_in_both_parsers(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            Phase4D2ProtocolAError,
            load_phase4_d2_protocol_a_config,
        )
        import rpe.runner.phase4_d2_protocol_a_verifier as verifier

        path = ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json"
        minimal = (
            b'{"active_perturbation_ids":["p08","p09","p10","p11","p12"],'
            b'"alpha_grid":[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.65,0.8],'
            b'"artifact_payload_files":["config.json","eligibility_bridge.json","model_cells.jsonl","validation_scores.jsonl","record_conditions.jsonl","metric_values.jsonl","peak_receipts.jsonl","downstream_rows.jsonl","predictions.jsonl","seed_class_conditions.jsonl","condition_summary.csv","class_observations.jsonl","alignment_results.jsonl","bootstrap_results.jsonl","sign_flip_results.jsonl","holm_family.jsonl","figure1_d2_5shot_protocol_a_full_domain.png","figure1_d2_5shot_protocol_a_full_domain.svg","figure1_d2_5shot_protocol_a_full_domain_data.csv","figure2_d2_5shot_protocol_a_full_domain.png","figure2_d2_5shot_protocol_a_full_domain.svg","figure2_d2_5shot_protocol_a_full_domain_data.csv","figure1_d2_10shot_protocol_a_full_domain.png","figure1_d2_10shot_protocol_a_full_domain.svg","figure1_d2_10shot_protocol_a_full_domain_data.csv","figure2_d2_10shot_protocol_a_full_domain.png","figure2_d2_10shot_protocol_a_full_domain.svg","figure2_d2_10shot_protocol_a_full_domain_data.csv","figure1_d2_20shot_protocol_a_full_domain.png","figure1_d2_20shot_protocol_a_full_domain.svg","figure1_d2_20shot_protocol_a_full_domain_data.csv","figure2_d2_20shot_protocol_a_full_domain.png","figure2_d2_20shot_protocol_a_full_domain.svg","figure2_d2_20shot_protocol_a_full_domain_data.csv","d2_protocol_a_full_domain_secondary_table.csv","manifest.json"],'
            b'"denominators":{"class_count":30,"test_record_count":3000},'
            b'"expected":{"condition_count":123000,"cwt_receipt_count":123000,"lr_candidate_fit_count":60,"metric_value_count":1599000,"model_cell_count":15,"prediction_row_count":1845000,"seed_class_condition_count":18450},'
            b'"experiment_id":"phase4-d2-protocol-a-full-domain-v1",'
            b'"inference":{"bootstrap_resamples":2000,"random_seed":20260817,"sign_flip_resamples":100000},'
            b'"metric_output_ids":["mse","rmse","mae","sam","pearson_r","nmse","wasserstein_1_cm1","is_like_structure_to_noise","precision","recall","f1","artifact_peak_ratio","missing_peak_ratio"],'
            b'"model_seeds":[0,1,2,3,4],"schema_version":"phase4-d2-protocol-a-full-domain-config-v1","shot_counts":[5,10,20],"support_point_count":997}\n'
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            stale = temporary / path.name
            stale.write_bytes(minimal)
            with self.assertRaisesRegex(
                Phase4D2ProtocolAError,
                "protocol|tier|claim_boundary|authorit|environment|figure|trust|frozen",
            ):
                load_phase4_d2_protocol_a_config(stale)
            with self.assertRaisesRegex(
                verifier.Phase4D2ProtocolAVerifierError,
                "protocol|tier|claim|authorit|environment|figure|trust|frozen",
            ):
                verifier._load_config(stale, require_frozen_identity=True)

            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["claim_boundary"] = "drift"
            drift = temporary / "drift.json"
            drift.write_text(
                json.dumps(changed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Phase4D2ProtocolAError, "frozen config identity"):
                load_phase4_d2_protocol_a_config(drift)
            with self.assertRaisesRegex(verifier.Phase4D2ProtocolAVerifierError, "frozen config identity"):
                verifier._load_config(drift, require_frozen_identity=True)

    def test_final_step13_parent_is_a_full_domain_authority(self) -> None:
        """A peak-common failed marker must not close the full-domain endpoint."""
        from rpe.runner.phase4_d2_protocol_a import (
            load_phase4_d2_protocol_a_config,
            validate_d2_eligibility_parent,
        )

        config = load_phase4_d2_protocol_a_config(
            ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json"
        )
        parent = ROOT / (
            "results/phase4/d2_protocol_a_full_domain_eligibility_v1/"
            "phase4-d2-protocol-a-full-domain-eligibility-"
            "0f43f329bfc6a7a1ff7232a336d115849b06dbca884de28d71f6b28502598ef8"
        )
        bridge = validate_d2_eligibility_parent(parent, config)
        self.assertEqual(bridge["parent_run_id"], parent.name)
        self.assertEqual(bridge["full_domain_state"], "evaluable")
        self.assertEqual(bridge["peak_common_state"], "not_evaluable_coverage")
        self.assertEqual(bridge["record_condition_count"], 123000)
        self.assertEqual(bridge["metric_status_count"], 1599000)

    def test_final_step13_bridge_is_canonical_json_serializable(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            _canonical, load_phase4_d2_protocol_a_config, validate_d2_eligibility_parent,
        )

        config = load_phase4_d2_protocol_a_config(
            ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json"
        )
        parent = ROOT / (
            "results/phase4/d2_protocol_a_full_domain_eligibility_v1/"
            "phase4-d2-protocol-a-full-domain-eligibility-"
            "0f43f329bfc6a7a1ff7232a336d115849b06dbca884de28d71f6b28502598ef8"
        )
        bridge = validate_d2_eligibility_parent(parent, config)
        self.assertEqual(json.loads(_canonical(dict(bridge))), dict(bridge))

    def test_real_retained_loader_reconstructs_exact_model_cells(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            load_phase4_d2_protocol_a_config,
            reconstruct_d2_protocol_a_inputs,
        )

        config = load_phase4_d2_protocol_a_config(
            ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json"
        )
        inputs = reconstruct_d2_protocol_a_inputs(
            ROOT / "data/unified/bacteria_id_reference",
            ROOT / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json",
            config,
        )
        self.assertEqual(inputs.test_values.shape, (3000, 997))
        self.assertEqual(inputs.validation_values[(5, 0)].shape, (300, 997))
        self.assertEqual(inputs.train_values[(5, 0)].shape, (150, 997))
        self.assertEqual(inputs.train_values[(20, 4)].shape, (600, 997))
        self.assertEqual(tuple(inputs.record_ids[:2]), ("test-000000", "test-000001"))

    def test_real_projection_and_rendering_are_not_placeholder_payloads(self) -> None:
        """Regression target: alpha-as-harm and signature-only fake figures."""
        from rpe.runner.phase4_d2_protocol_a import (
            aggregate_d2_protocol_a,
            make_synthetic_d2_protocol_a_config,
            make_synthetic_d2_protocol_a_inputs,
            render_d2_protocol_a_figures,
        )

        inputs = make_synthetic_d2_protocol_a_inputs(
            class_count=2, records_per_class=4, model_seeds=(0, 1)
        )
        config = make_synthetic_d2_protocol_a_config(inputs)
        predictions = []
        metrics = []
        conditions = [("alpha0", None, 0.0)] + [
            (f"{perturbation}:{__import__("numpy").float64(alpha).tobytes().hex()}", perturbation, alpha)
            for perturbation in ("p08", "p09", "p10", "p11", "p12")
            for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
        ]
        for condition_id, perturbation, alpha in conditions:
            for order, label in enumerate(inputs.test_labels):
                for shot in (5, 10, 20):
                    for seed in inputs.model_seeds:
                        predictions.append({"shot_count": shot, "model_seed": seed, "class_label": int(label), "condition_id": condition_id, "record_order": order, "correct": bool(perturbation is None or (order + seed) % 3 != 0)})
                for metric in ("mse", "rmse", "mae", "sam", "pearson_r", "nmse", "wasserstein_1_cm1", "is_like_structure_to_noise", "precision", "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio"):
                    metrics.append({"record_order": order, "condition_id": condition_id, "metric_output_id": metric, "value": 0.0 if perturbation is None else float(order + 1 + (metric != "mse"))})
        projection = aggregate_d2_protocol_a(predictions, metrics, config, inference_resamples=8)
        self.assertEqual(len(projection.bootstrap_results), 36)
        self.assertEqual(len(projection.sign_flip_results), 72)
        self.assertEqual(len(projection.holm_family), 72)
        mse = next(row for row in projection.alignment_results if row["shot_count"] == 5 and row["metric_output_id"] == "mse")
        self.assertIn("ag_interval", mse)
        self.assertIn("acc_interval", mse)
        self.assertIn(mse["state"], {"complete", "not_evaluable_constant_downstream"})
        observed = [row for row in projection.class_observations if row["metric_output_id"] == "mse"]
        self.assertNotEqual(observed[0]["metric_harm"], observed[0]["alpha"])
        payloads = render_d2_protocol_a_figures(projection, config)
        png = payloads["figure1_d2_5shot_protocol_a_full_domain.png"]
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(png[12:16], b"IHDR")
        self.assertIn(b"<svg", payloads["figure2_d2_20shot_protocol_a_full_domain.svg"])
        csv_rows = payloads["figure2_d2_5shot_protocol_a_full_domain_data.csv"].decode().splitlines()
        self.assertEqual(len(csv_rows), 14)
        self.assertIn("d_ag", csv_rows[0])
        self.assertIn("d_acc", csv_rows[0])

    def test_serialization_has_shot_condition_summaries_and_is_append_only(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            Phase4D2ProtocolAError, build_phase4_d2_protocol_a_from_inputs,
            make_synthetic_d2_protocol_a_config, make_synthetic_d2_protocol_a_inputs,
        )

        inputs = make_synthetic_d2_protocol_a_inputs(class_count=2, records_per_class=2, model_seeds=(0, 1))
        config = make_synthetic_d2_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            built = build_phase4_d2_protocol_a_from_inputs(output, inputs=inputs, config=config, worker_count=1, inference_resamples=8)
            rows = (built.path / "condition_summary.csv").read_text().splitlines()
            self.assertEqual(len(rows), 1 + 3 * 41)
            self.assertIn("shot_count", rows[0])
            self.assertIn("macro_f1", rows[0])
            with (built.path / "downstream_rows.jsonl").open() as stream:
                self.assertEqual(sum(1 for _ in stream), 164)
            bridge = json.loads((built.path / "eligibility_bridge.json").read_text())
            self.assertIn("parent_run_id", bridge)
            receipt = json.loads((built.path / "model_cells.jsonl").read_text().splitlines()[0])
            self.assertIn("pca_train_feature_sha256", receipt)
            self.assertIn("pca_validation_feature_sha256", receipt)
            self.assertNotIn("pca_feature_sha256", receipt)
            with self.assertRaisesRegex(Phase4D2ProtocolAError, "append-only"):
                build_phase4_d2_protocol_a_from_inputs(output, inputs=inputs, config=config, worker_count=1, inference_resamples=8)

    def test_aggregate_preserves_canonical_grid_and_holm_row_contract(self) -> None:
        from rpe.runner.phase4_d2_protocol_a import (
            Phase4D2ProtocolAError, aggregate_d2_protocol_a,
            make_synthetic_d2_protocol_a_config, make_synthetic_d2_protocol_a_inputs,
        )

        inputs = make_synthetic_d2_protocol_a_inputs(class_count=2, records_per_class=2, model_seeds=(0, 1))
        config = make_synthetic_d2_protocol_a_config(inputs)
        prediction = {"shot_count": 5, "model_seed": 0, "class_label": 0, "condition_id": "alpha0", "record_order": 0, "correct": True}
        with self.assertRaisesRegex(Phase4D2ProtocolAError, "exact canonical condition grid"):
            aggregate_d2_protocol_a([prediction], [], config, inference_resamples=2)

        # This exact fixture needs a complete grid; use the existing synthetic
        # artifact to assert Figure-2 stores full Holm provenance.
        with tempfile.TemporaryDirectory() as directory:
            from rpe.runner.phase4_d2_protocol_a import build_phase4_d2_protocol_a_from_inputs
            built = build_phase4_d2_protocol_a_from_inputs(Path(directory) / "artifact", inputs=inputs, config=config, worker_count=1, inference_resamples=2)
            row = json.loads((built.path / "holm_family.jsonl").read_text().splitlines()[0])
            self.assertTrue({"raw_p_value", "adjusted_p_value", "rank", "family_size", "favorable", "rejected"}.issubset(row))

    def test_public_build_orchestrates_the_real_production_stages(self) -> None:
        """Regression target: the former intentional public-build raise."""
        from rpe.runner.phase4_d2_protocol_a import (
            Phase4D2ProtocolASummary,
            build_phase4_d2_protocol_a,
            make_synthetic_d2_protocol_a_config,
            make_synthetic_d2_protocol_a_inputs,
        )

        inputs = make_synthetic_d2_protocol_a_inputs(
            class_count=2, records_per_class=4, model_seeds=(0, 1)
        )
        config = make_synthetic_d2_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory, \
             patch("rpe.runner.phase4_d2_protocol_a.load_phase4_d2_protocol_a_config", return_value=config), \
             patch("rpe.runner.phase4_d2_protocol_a.validate_d2_eligibility_parent", return_value={"full_domain_state": "evaluable"}), \
             patch("rpe.runner.phase4_d2_protocol_a.reconstruct_d2_protocol_a_inputs", return_value=inputs), \
             patch("rpe.runner.phase4_d2_protocol_a.rematerialize_d2_protocol_a_science", return_value=None), \
             patch("rpe.runner.phase4_d2_protocol_a.build_phase4_d2_protocol_a_from_inputs", return_value=Phase4D2ProtocolASummary(Path(directory), "run", "complete", 4, 6)) as build:
            summary = build_phase4_d2_protocol_a(Path(directory), worker_count=1)
        self.assertEqual(summary.run_id, "run")
        build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
