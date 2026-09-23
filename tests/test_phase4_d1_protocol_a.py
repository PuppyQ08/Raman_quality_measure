from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


class D1ProtocolAContractTest(unittest.TestCase):
    def test_synthetic_end_to_end_inventory_and_terminal_contract(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            ARTIFACT_PAYLOAD_FILES,
            build_phase4_d1_protocol_a_from_inputs,
            make_synthetic_d1_protocol_a_config,
            make_synthetic_d1_protocol_a_inputs,
        )

        inputs = make_synthetic_d1_protocol_a_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
        )
        config = make_synthetic_d1_protocol_a_config(inputs)
        self.assertEqual(len(ARTIFACT_PAYLOAD_FILES), 21)
        with tempfile.TemporaryDirectory() as directory:
            summary = build_phase4_d1_protocol_a_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            inventory = {item.name for item in summary.path.iterdir()}
            self.assertEqual(len(inventory), 23)
            self.assertEqual(
                inventory & {"complete.json", "failed.json"},
                {"complete.json"},
            )
            self.assertIn("SHA256SUMS", inventory)
            manifest = json.loads((summary.path / "manifest.json").read_text())
            self.assertEqual(manifest["endpoint_count"], 1)
            self.assertEqual(manifest["active_perturbation_count"], 5)
            self.assertEqual(manifest["condition_count"], 41)
            self.assertEqual(manifest["model_seed_count"], 5)
            self.assertEqual(tuple(manifest["payload_files"]), ARTIFACT_PAYLOAD_FILES)

    def test_model_lifecycle_uses_train_only_pca_and_first_strict_validation_maximum(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            fit_d1_protocol_a_models,
            make_synthetic_d1_protocol_a_config,
            make_synthetic_d1_protocol_a_inputs,
        )

        inputs = make_synthetic_d1_protocol_a_inputs(
            class_count=3,
            records_per_class=3,
            model_seeds=(0, 1, 2, 3, 4),
        )
        config = make_synthetic_d1_protocol_a_config(inputs)
        models = fit_d1_protocol_a_models(inputs, config)
        self.assertEqual(len(models), 5)
        self.assertTrue(all(model.selected_c in (0.01, 0.1, 1.0, 10.0) for model in models))
        self.assertTrue(all(len(model.validation_scores) == 4 for model in models))
        self.assertTrue(all(tuple(row["c"] for row in model.validation_scores) == (0.01, 0.1, 1.0, 10.0) for model in models))
        self.assertTrue(all(model.refit_with_validation is False for model in models))
        self.assertTrue(all(model.condition_call_count == 41 for model in models))
        self.assertTrue(all(model.pca_train_feature_sha256 != model.pca_validation_feature_sha256 for model in models))

    def test_split_reconstruction_matches_frozen_real_ledgers(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            load_phase4_d1_protocol_a_config,
            reconstruct_d1_protocol_a_inputs,
        )

        config = load_phase4_d1_protocol_a_config(
            ROOT / "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
        )
        inputs = reconstruct_d1_protocol_a_inputs(
            ROOT / "data/unified/bacteria_id_reference",
            config,
        )
        self.assertEqual(inputs.test_values.shape, (3000, 997))
        self.assertEqual(len(inputs.model_seeds), 5)
        self.assertEqual(inputs.source_record_count, 66000)
        self.assertEqual(inputs.model_cell_count, 5)
        self.assertEqual(inputs.role_occurrence_count, 330000)
        self.assertEqual(inputs.source_ledger_sha256, "a44e71531c5a639904390ca73738d035e0c3294766f46a8def82b2f9bd9fec1b")
        self.assertEqual(inputs.model_ledger_sha256, "1b77f2825e25265382597e64218f4a2771011cd9cc8f86c4c3f6e729e6c2c486")
        self.assertEqual(inputs.role_ledger_sha256, "211b087e21f3b511acd6acaa800b693ffef0db7c1aaa354c902d146ee2c2090c")
        self.assertEqual(inputs.model_cells[0]["train_record_ids_sha256"], "ed3ea7448b5737b4122f76cb4b60f95487780c5a68e03f64d48a7b36d39e5433")
        self.assertEqual(inputs.model_cells[0]["validation_record_ids_sha256"], "aa2cf7c7950b877493114d0492410474e0d21998e1081363d3499969092ab7a5")
        self.assertEqual(inputs.model_cells[0]["test_record_ids_sha256"], "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd")
        self.assertEqual(inputs.model_cells[1]["train_record_ids_sha256"], "a5b5e7b791c7e08af1109bb02b0ee6b0cad8b774ba476c3d4720c99291a1371f")
        self.assertEqual(inputs.support_axis_cm1.shape, (997,))
        self.assertEqual(inputs.support_axis_f32_sha256, "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59")
        self.assertEqual(inputs.support_axis_f64_sha256, "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6")
        self.assertFalse(inputs.role_overlap_detected)

    def test_parent_bridge_accepts_only_authorized_step15_payloads_and_rejects_drift(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            Phase4D1ProtocolAError,
            load_phase4_d1_protocol_a_config,
            parse_phase4_d1_protocol_a_config,
            validate_d1_parent_authorities,
        )

        config = load_phase4_d1_protocol_a_config(
            ROOT / "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
        )
        bridge = validate_d1_parent_authorities(config)
        self.assertEqual(
            tuple(bridge["authorized_step15_payloads"]),
            ("record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"),
        )
        self.assertEqual(bridge["condition_bridge_sha256"], "031b4f4f30235a6237ad1a6f89b03fb9f2d1b724f55f7c8ae9d52f402eabaa7c")
        self.assertEqual(bridge["bridge_row_count"], 123000)
        self.assertEqual(bridge["metric_row_count"], 1599000)
        self.assertEqual(bridge["cwt_row_count"], 123000)
        self.assertEqual(bridge["step13_full_domain_state"], "evaluable")
        self.assertEqual(bridge["step13_peak_common_state"], "not_evaluable_coverage")
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "config.json"
            document = json.loads(config.raw_bytes)
            document["authorities"]["step15_parent"]["sha256sums_sha256"] = "0" * 64
            fake.write_bytes(_canonical(document))
            drifted = parse_phase4_d1_protocol_a_config(
                fake, fake.read_bytes(), require_frozen_identity=False
            )
            with self.assertRaisesRegex(Phase4D1ProtocolAError, "Step-15|parent|checksum|authority"):
                validate_d1_parent_authorities(drifted)

    def test_public_build_and_cli_expose_no_outcome_affecting_overrides(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            Phase4D1ProtocolASummary,
            build_phase4_d1_protocol_a,
            make_synthetic_d1_protocol_a_config,
            make_synthetic_d1_protocol_a_inputs,
        )
        from tools.run_phase4_d1_protocol_a import main

        with self.assertRaises(SystemExit):
            main(["build", "--output-root", "x", "--bootstrap-resamples", "8"])
        with self.assertRaises(SystemExit):
            main(["verify", "--run-path", "x", "--seed", "0"])

        inputs = make_synthetic_d1_protocol_a_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
        )
        config = make_synthetic_d1_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory, patch(
            "rpe.runner.phase4_d1_protocol_a.load_phase4_d1_protocol_a_config",
            return_value=config,
        ), patch(
            "rpe.runner.phase4_d1_protocol_a.validate_d1_parent_authorities",
            return_value={"bridge_row_count": len(inputs.record_ids) * 41},
        ), patch(
            "rpe.runner.phase4_d1_protocol_a.reconstruct_d1_protocol_a_inputs",
            return_value=inputs,
        ), patch(
            "rpe.runner.phase4_d1_protocol_a.build_phase4_d1_protocol_a_from_inputs",
            return_value=Phase4D1ProtocolASummary(
                Path(directory),
                "run",
                "complete",
                3000,
                5,
            ),
        ) as build:
            summary = build_phase4_d1_protocol_a(Path(directory), worker_count=1)
        self.assertEqual(summary.run_id, "run")
        build.assert_called_once()

    def test_production_forbids_phase05_results_and_forbidden_step15_outcome_payloads(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            Phase4D1ProtocolAError,
            load_phase4_d1_protocol_a_config,
            reconstruct_d1_protocol_a_inputs,
            validate_d1_parent_authorities,
        )

        config = load_phase4_d1_protocol_a_config(
            ROOT / "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
        )
        with self.assertRaisesRegex(Phase4D1ProtocolAError, "forbidden Phase-0.5"):
            reconstruct_d1_protocol_a_inputs(Path("complete.json"), config)
        with self.assertRaisesRegex(Phase4D1ProtocolAError, "forbidden Step-15|outcome payload"):
            validate_d1_parent_authorities(
                SimpleNamespace(
                    **{
                        **config.__dict__,
                        "document": {
                            **config.document,
                            "step15_forbidden_probe": "predictions.jsonl",
                        },
                    }
                )
            )

    def test_verifier_is_independent_public_has_no_bypass_and_byte_compares_synthetic_artifact(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            build_phase4_d1_protocol_a_from_inputs,
            make_synthetic_d1_protocol_a_config,
            make_synthetic_d1_protocol_a_inputs,
        )
        from rpe.runner.phase4_d1_protocol_a_verifier import (
            verify_phase4_d1_protocol_a_from_inputs,
        )
        import rpe.runner.phase4_d1_protocol_a_verifier as verifier

        tree = ast.parse(inspect.getsource(verifier))
        forbidden = "rpe.runner.phase4_d1_protocol_a"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(forbidden, {alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, forbidden)
        self.assertEqual(
            tuple(inspect.signature(verifier.verify_phase4_d1_protocol_a).parameters),
            ("path", "worker_count"),
        )
        self.assertNotIn("skip", inspect.getsource(verifier.verify_phase4_d1_protocol_a))
        self.assertNotIn("bypass", inspect.getsource(verifier.verify_phase4_d1_protocol_a))

        inputs = make_synthetic_d1_protocol_a_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
        )
        config = make_synthetic_d1_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            built = build_phase4_d1_protocol_a_from_inputs(
                path,
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            verified = verify_phase4_d1_protocol_a_from_inputs(
                built.path,
                inputs=inputs,
                config_path=built.path / "config.json",
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            self.assertEqual(verified.run_id, built.run_id)

    def test_checksum_consistent_semantic_tampering_is_rejected(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            build_phase4_d1_protocol_a_from_inputs,
            make_synthetic_d1_protocol_a_config,
            make_synthetic_d1_protocol_a_inputs,
        )
        from rpe.runner.phase4_d1_protocol_a_verifier import (
            Phase4D1ProtocolAVerifierError,
            verify_phase4_d1_protocol_a_from_inputs,
        )

        inputs = make_synthetic_d1_protocol_a_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
        )
        config = make_synthetic_d1_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            built = build_phase4_d1_protocol_a_from_inputs(
                path,
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            rows = (path / "predictions.jsonl").read_text().splitlines()
            changed = json.loads(rows[0])
            changed["correct"] = not changed["correct"]
            rows[0] = json.dumps(
                changed,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            (path / "predictions.jsonl").write_text("\n".join(rows) + "\n")
            sums = []
            for line in (path / "SHA256SUMS").read_text().splitlines():
                _digest, name = line.split("  ", 1)
                sums.append(
                    f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}"
                )
            (path / "SHA256SUMS").write_text("\n".join(sums) + "\n")
            with self.assertRaisesRegex(
                Phase4D1ProtocolAVerifierError,
                "rebuild|semantic|payload",
            ):
                verify_phase4_d1_protocol_a_from_inputs(
                    built.path,
                    inputs=inputs,
                    config_path=built.path / "config.json",
                    worker_count=1,
                    bootstrap_resamples=8,
                    sign_flip_resamples=16,
                )

    def test_failed_synthetic_state_emits_failed_marker_and_fixed_holm_slots(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            build_phase4_d1_protocol_a_from_inputs,
            make_synthetic_d1_protocol_a_config,
            make_synthetic_d1_protocol_a_inputs,
        )

        inputs = make_synthetic_d1_protocol_a_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=(0, 1, 2, 3, 4),
            force_metric_failure=True,
        )
        config = make_synthetic_d1_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            built = build_phase4_d1_protocol_a_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            self.assertTrue((built.path / "failed.json").exists())
            holm = (built.path / "holm_family.jsonl").read_text().splitlines()
            self.assertEqual(len(holm), 24)

    def test_real_test_rematerialization_uses_spawned_capacity_limited_workers(self) -> None:
        import rpe.runner.phase4_d1_protocol_a as subject
        from rpe.evaluation import Spectrum1D

        axis = np.linspace(350.0, 1830.0, 1000, dtype=np.float64)
        support = np.linspace(386.65, 1792.4, 997, dtype=np.float64)
        spectra = tuple(
            Spectrum1D(
                spectrum_id=f"test::{index}",
                sample_id=str(index),
                axis_cm1=axis,
                intensity=np.linspace(0.0, 1.0, axis.size, dtype=np.float64) + index * 0.001,
            )
            for index in range(3)
        )
        inputs = SimpleNamespace(
            support_axis_cm1=support,
            native_test_labels=np.asarray([0, 1, 0], dtype=np.int64),
            native_test_spectra=spectra,
            record_ids=("a", "b", "c"),
            test_labels=np.asarray([0, 1, 0], dtype=np.int64),
        )
        calls = {}

        class FakeExecutor:
            def __init__(self, *, max_workers, mp_context, initializer, initargs):
                calls.update(
                    max_workers=max_workers,
                    mp_context=mp_context,
                    initializer=initializer,
                    initargs=initargs,
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def map(self, function, jobs):
                calls["function"] = function
                return tuple(function(job) for job in reversed(tuple(jobs)))

        def worker(job):
            order, _record_id, _label, _spectrum = job
            return (
                order,
                {
                    condition: np.asarray([order], dtype=np.float32)
                    for condition, _perturbation, _alpha in subject._condition_specs()
                },
                [],
                [],
                [],
            )

        budget = 64 * 1024**3
        with patch.object(subject, "estimate_p10_peak_bytes", return_value=budget // 2), patch.object(
            subject,
            "ProcessPoolExecutor",
            FakeExecutor,
        ), patch.object(
            subject.multiprocessing,
            "get_context",
            return_value="spawn-context",
        ) as get_context, patch.object(
            subject,
            "_real_science_worker",
            side_effect=worker,
        ), patch.object(
            subject,
            "_load_real_science_authorities",
            return_value=SimpleNamespace(),
        ):
            matrices, conditions, metrics, peaks = subject.rematerialize_d1_protocol_a_science(
                inputs,
                SimpleNamespace(synthetic_fixture=False),
                worker_count=9,
            )
        self.assertEqual(calls["max_workers"], 2)
        self.assertEqual(calls["mp_context"], "spawn-context")
        self.assertIs(calls["initializer"], subject._initialize_real_science_worker)
        self.assertTrue(calls["function"].called)
        get_context.assert_called_once_with("spawn")
        self.assertEqual([float(row[0]) for row in matrices["alpha0"]], [0.0, 1.0, 2.0])
        self.assertEqual((conditions, metrics, peaks), ((), (), ()))

    def test_real_config_and_authority_anchor_are_frozen(self) -> None:
        from rpe.runner.phase4_d1_protocol_a import (
            Phase4D1ProtocolAError,
            load_phase4_d1_protocol_a_config,
        )
        from rpe.runner.phase4_d1_protocol_a_authority import (
            CONFIG_BYTES,
            CONFIG_SHA256,
        )
        import rpe.runner.phase4_d1_protocol_a_verifier as verifier

        path = ROOT / "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
        raw = path.read_bytes()
        self.assertEqual(len(raw), CONFIG_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), CONFIG_SHA256)
        config = load_phase4_d1_protocol_a_config(path)
        self.assertEqual(config.sha256, CONFIG_SHA256)
        self.assertEqual(config.raw_bytes, raw)
        self.assertEqual(verifier._load_config(path, require_frozen_identity=True).sha256, CONFIG_SHA256)

        with tempfile.TemporaryDirectory() as directory:
            drift = Path(directory) / path.name
            document = json.loads(raw)
            document["claim_boundary"] = "drift"
            drift.write_bytes(_canonical(document))
            with self.assertRaisesRegex(Phase4D1ProtocolAError, "frozen config identity"):
                load_phase4_d1_protocol_a_config(drift)
            with self.assertRaisesRegex(verifier.Phase4D1ProtocolAVerifierError, "frozen config identity"):
                verifier._load_config(drift, require_frozen_identity=True)


if __name__ == "__main__":
    unittest.main()
