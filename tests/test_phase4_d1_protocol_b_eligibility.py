from __future__ import annotations

import ast
import hashlib
import importlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# NOTE: This test module is written as a focused contract suite for the
# upcoming D1 Protocol-B all-role eligibility implementation (Phase 4 Step 23).
# At the time of writing, the production module does not exist yet, so the
# expected state for `python -m unittest ...` is RED with an import failure.

from rpe.runner.phase4_d1_protocol_b_eligibility import (  # noqa: E402
    ACTIVE_PERTURBATION_IDS,
    ARTIFACT_PAYLOAD_FILES,
    Phase4D1ProtocolBEligibilityError,
    build_phase4_d1_protocol_b_eligibility_from_inputs,
    evaluate_d1_protocol_b_all_role_gates,
    load_phase4_d1_protocol_b_eligibility_config,
    make_synthetic_d1_protocol_b_inputs,
    make_synthetic_d1_protocol_b_inputs_with_one_failed_cell,
    make_synthetic_d1_protocol_b_config,
    parse_phase4_d1_protocol_b_eligibility_config,
    reconstruct_d1_protocol_b_inputs,
)
from rpe.runner.phase4_d1_protocol_b_eligibility_verifier import (  # noqa: E402
    Phase4D1ProtocolBEligibilityVerifierError,
    verify_phase4_d1_protocol_b_eligibility_from_inputs,
)
from tools.run_phase4_d1_protocol_b_eligibility import (  # noqa: E402
    main as d1_protocol_b_cli_main,
)


REAL_CONFIG_PATH = (
    ROOT / "experiments/phase4/configs/d1_protocol_b_all_role_eligibility_v1.json"
)

ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
MODEL_SEEDS = (0, 1, 2, 3, 4)

# From reports/phase4/step22_d1_protocol_b_all_role_eligibility_design.md
REAL_SOURCE_LEDGER_SHA256 = "a44e71531c5a639904390ca73738d035e0c3294766f46a8def82b2f9bd9fec1b"
REAL_MODEL_LEDGER_SHA256 = "1b77f2825e25265382597e64218f4a2771011cd9cc8f86c4c3f6e729e6c2c486"
REAL_ROLE_LEDGER_SHA256 = "211b087e21f3b511acd6acaa800b693ffef0db7c1aaa354c902d146ee2c2090c"
REAL_TEST_IDS_SHA256 = "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
REAL_SUPPORT_F64_SHA256 = "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6"
REAL_SUPPORT_F32_SHA256 = "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59"
INHERITED_STRUCTURAL_REASON = "structurally_ineligible_missing_explicit_baseline"


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


def _array_sha(value: np.ndarray, *, dtype: str) -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _hex_alpha(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def _artifact_tree(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


class Phase4D1ProtocolBEligibilitySyntheticContractTest(unittest.TestCase):
    def test_synthetic_end_to_end_inventory_is_compact_deduplicated_and_checksummed(self) -> None:
        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=3,
            model_seeds=MODEL_SEEDS,
            shard_source_count=2,
            native_point_count=16,
            support_point_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)

        self.assertEqual(config.endpoint_count, 1)
        self.assertEqual(tuple(config.model_seeds), MODEL_SEEDS)
        self.assertEqual(tuple(config.alpha_grid), ALPHA_GRID)
        self.assertEqual(tuple(config.active_perturbation_ids), ACTIVE_PERTURBATION_IDS)
        self.assertEqual(
            tuple(config.artifact_payload_files),
            tuple(ARTIFACT_PAYLOAD_FILES[:7])
            + tuple(config.condition_matrix_shard_files)
            + tuple(ARTIFACT_PAYLOAD_FILES[7:]),
        )
        self.assertTrue(config.synthetic_fixture)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            summary = build_phase4_d1_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                config=config,
                worker_count=2,
            )
            self.assertEqual(summary.endpoint_count, 1)
            self.assertEqual(summary.model_seed_count, 5)
            self.assertGreater(summary.source_record_count, 0)

            observed = {item.name for item in summary.path.iterdir() if item.is_file()}
            marker_name = next(iter(observed & {"complete.json", "failed.json"}))
            shard_names = {row["filename"] for row in (
                json.loads(line)
                for line in (summary.path / "condition_matrix_shards.jsonl")
                .read_text(encoding="utf-8").splitlines()
            )}
            self.assertEqual(
                observed,
                set(config.artifact_payload_files) | {"SHA256SUMS", marker_name},
            )
            checksums = (summary.path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [line.split("  ", 1)[1] for line in checksums],
                list(config.artifact_payload_files) + [marker_name],
            )

            manifest = json.loads((summary.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["endpoint_count"], 1)
            self.assertEqual(manifest["protocol"], "B")
            self.assertEqual(
                manifest["claim_boundary"],
                "outcome_blind_protocol_b_all_role_eligibility_only",
            )
            self.assertEqual(manifest["artifact_order"], list(config.artifact_payload_files))

            # Synthetic store must remain compact: shard_source_count=2 and 8 support points
            # should never resemble the 66×163,508,000-byte real store.
            shard_rows = [
                json.loads(line)
                for line in (summary.path / "condition_matrix_shards.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all(row["rows_per_shard"] == 2 for row in shard_rows))
            self.assertTrue(all(row["support_point_count"] == 8 for row in shard_rows))

            # Each unique source must be executed once, not repeated per model/role.
            # Contract: operator_cells is source-major and unique by (record_id, perturbation_id).
            operator_rows = [
                json.loads(line)
                for line in (summary.path / "operator_cells.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            pairs = {(row["record_id"], row["perturbation_id"]) for row in operator_rows}
            self.assertEqual(len(pairs), len(operator_rows))

    def test_gate_is_original_denominator_and_single_failure_closes_one_operator(self) -> None:
        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=3,
            records_per_class=2,
            model_seeds=MODEL_SEEDS,
            shard_source_count=2,
            native_point_count=16,
            support_point_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        operator_cells = make_synthetic_d1_protocol_b_inputs_with_one_failed_cell(inputs)

        class_rows, gate = evaluate_d1_protocol_b_all_role_gates(
            inputs.source_records,
            inputs.model_cells,
            inputs.model_role_occurrences,
            operator_cells,
            config,
        )
        self.assertEqual(len(class_rows), 3 * len(ACTIVE_PERTURBATION_IDS))
        self.assertIn("full_domain_core", gate)
        p08 = gate["full_domain_core"]["operators"]["p08"]
        self.assertEqual(p08["required_record_count"], inputs.source_record_count)
        self.assertEqual(p08["required_class_count"], 3)
        self.assertEqual(p08["state"], "not_evaluable_coverage")
        self.assertEqual(p08["complete_record_count"], inputs.source_record_count - 1)

    def test_inherited_p1_p7_rulings_are_not_executed_or_promoted(self) -> None:
        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=MODEL_SEEDS,
            shard_source_count=2,
            native_point_count=16,
            support_point_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            build_phase4_d1_protocol_b_eligibility_from_inputs(
                path,
                inputs=inputs,
                config=config,
                worker_count=1,
            )
            gate = json.loads((path / "gate.json").read_text(encoding="utf-8"))
            inherited = gate["inherited_rulings"]
            self.assertEqual(
                inherited["p06"]["state"],
                INHERITED_STRUCTURAL_REASON,
            )
            self.assertEqual(
                inherited["p07"]["state"],
                INHERITED_STRUCTURAL_REASON,
            )
            self.assertEqual(inherited["p01_p04_full_domain_core"]["state"], "not_evaluable_coverage")
            self.assertEqual(inherited["p05_full_domain_core"]["state"], "not_evaluable_coverage")
            operator_rows = [
                json.loads(line)
                for line in (path / "operator_cells.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all(row["perturbation_id"] in ACTIVE_PERTURBATION_IDS for row in operator_rows),
                "P1–P7 must not appear as execution rows",
            )

    def test_shard_layout_offsets_and_semantic_mutation_is_rejected(self) -> None:
        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=3,
            model_seeds=MODEL_SEEDS,
            shard_source_count=2,
            native_point_count=16,
            support_point_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            built = build_phase4_d1_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                config=config,
                worker_count=2,
            )
            shard_rows = [
                json.loads(line)
                for line in (output / "condition_matrix_shards.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertGreaterEqual(len(shard_rows), 1)
            first = shard_rows[0]
            shard_name = first["filename"]
            shard_path = output / shard_name
            self.assertTrue(shard_path.exists())
            # Condition-major, then shard-local source-major, then support index.
            self.assertEqual(first["byte_count"], first["condition_count"] * first["rows_per_shard"] * first["support_point_count"] * 4)

            # Mutate one float32 value in-place, update SHA256SUMS, ensure verifier rejects.
            raw = bytearray(shard_path.read_bytes())
            raw[0:4] = np.asarray([123.0], dtype="<f4").tobytes()
            shard_path.write_bytes(bytes(raw))
            sums = []
            for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                _digest, name = line.split("  ", 1)
                sums.append(f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}")
            (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                Phase4D1ProtocolBEligibilityVerifierError,
                "semantic|rebuild|payload|shard|matrix",
            ):
                verify_phase4_d1_protocol_b_eligibility_from_inputs(
                    built.path,
                    inputs=inputs,
                    config_path=built.path / "config.json",
                    worker_count=1,
                )

    def test_p10_admission_fails_before_output_directory_is_created(self) -> None:
        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=MODEL_SEEDS,
            shard_source_count=2,
            native_point_count=100000,
            support_point_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            with self.assertRaisesRegex(
                Phase4D1ProtocolBEligibilityError,
                "64 GiB",
            ):
                build_phase4_d1_protocol_b_eligibility_from_inputs(
                    output,
                    inputs=inputs,
                    config=config,
                    worker_count=1,
                )
            self.assertFalse(output.exists(), "admission must fail before creating any output directory")


class Phase4D1ProtocolBEligibilityRealLedgerContractTest(unittest.TestCase):
    def test_real_reconstruction_reproduces_frozen_ledgers_and_support_digests(self) -> None:
        config = load_phase4_d1_protocol_b_eligibility_config(REAL_CONFIG_PATH)
        inputs = reconstruct_d1_protocol_b_inputs(
            ROOT / "data/unified/bacteria_id_reference",
            config,
        )
        self.assertEqual(inputs.source_record_count, 66000)
        self.assertEqual(inputs.model_cell_count, 5)
        self.assertEqual(inputs.role_occurrence_count, 330000)
        self.assertEqual(inputs.source_ledger_sha256, REAL_SOURCE_LEDGER_SHA256)
        self.assertEqual(inputs.model_ledger_sha256, REAL_MODEL_LEDGER_SHA256)
        self.assertEqual(inputs.role_ledger_sha256, REAL_ROLE_LEDGER_SHA256)
        self.assertEqual(inputs.test_record_ids_sha256, REAL_TEST_IDS_SHA256)
        self.assertEqual(inputs.support_axis_f64_sha256, REAL_SUPPORT_F64_SHA256)
        self.assertEqual(inputs.support_axis_f32_sha256, REAL_SUPPORT_F32_SHA256)
        self.assertFalse(inputs.role_overlap_detected)

    def test_real_inherited_bounds_for_p1_p5_and_structural_p6_p7_are_exact(self) -> None:
        document = json.loads(REAL_CONFIG_PATH.read_text(encoding="utf-8"))
        inherited = document["inherited_rulings"]
        self.assertEqual(inherited["p06"]["state"], INHERITED_STRUCTURAL_REASON)
        self.assertEqual(inherited["p07"]["state"], INHERITED_STRUCTURAL_REASON)
        self.assertEqual(inherited["p01_p04_full_domain_core"]["state"], "not_evaluable_coverage")
        self.assertEqual(inherited["p05_full_domain_core"]["state"], "not_evaluable_coverage")
        # Step-22: class upper bound is 0/30 because test-class completeness is 0/30.
        self.assertEqual(
            inherited["p01_p04_full_domain_core"]["bound"]["class_upper_bound_complete"],
            0,
        )
        self.assertEqual(
            inherited["p05_full_domain_core"]["bound"]["class_upper_bound_complete"],
            0,
        )
        # Step-22: optimistic record upper bounds for all-role union.
        self.assertEqual(
            inherited["p01_p04_full_domain_core"]["bound"]["record_upper_bound_complete"],
            63208,
        )
        self.assertEqual(
            inherited["p05_full_domain_core"]["bound"]["record_upper_bound_complete"],
            63002,
        )
        self.assertEqual(inherited["p01_p04_full_domain_core"]["bound"]["required_record_count"], 66000)
        self.assertEqual(inherited["p01_p04_full_domain_core"]["bound"]["required_class_count"], 30)

    def test_real_config_identity_is_frozen_and_drift_is_rejected(self) -> None:
        raw = REAL_CONFIG_PATH.read_bytes()
        document = json.loads(raw)
        drift = json.loads(json.dumps(document))
        drift["claim_boundary"] = "drifted"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / REAL_CONFIG_PATH.name
            path.write_bytes(_canonical(drift))
            with self.assertRaisesRegex(
                Phase4D1ProtocolBEligibilityError,
                "frozen config identity",
            ):
                load_phase4_d1_protocol_b_eligibility_config(path)

    def test_reconstruction_fails_closed_on_support_digest_mismatch(self) -> None:
        raw = REAL_CONFIG_PATH.read_bytes()
        document = json.loads(raw)
        drift = json.loads(json.dumps(document))
        drift["frozen_identities"]["support_axis_f32_sha256"] = "0" * 64
        drift_raw = _canonical(drift)
        config = parse_phase4_d1_protocol_b_eligibility_config(
            REAL_CONFIG_PATH,
            drift_raw,
            require_frozen_identity=False,
        )
        with self.assertRaisesRegex(
            Phase4D1ProtocolBEligibilityError,
            "support_axis_f32_sha256",
        ):
            reconstruct_d1_protocol_b_inputs(
                ROOT / "data/unified/bacteria_id_reference",
                config,
            )


class Phase4D1ProtocolBEligibilityFirewallAndVerifierContractTest(unittest.TestCase):
    def test_recursive_outcome_field_validation_rejects_predictions_metrics_figures(self) -> None:
        # Contract: config parsing must reject any outcome-bearing field recursively.
        raw = REAL_CONFIG_PATH.read_bytes()
        document = json.loads(raw)
        document["outcome"] = {"predictions.jsonl": True, "metric_values.jsonl": True}
        drift_raw = _canonical(document)
        with self.assertRaisesRegex(
            Phase4D1ProtocolBEligibilityError,
            "outcome|prediction|metric|figure|inference",
        ):
            parse_phase4_d1_protocol_b_eligibility_config(
                REAL_CONFIG_PATH,
                drift_raw,
                require_frozen_identity=False,
            )

    def test_verifier_is_structurally_independent_and_byte_compares(self) -> None:
        verifier_path = ROOT / "rpe/runner/phase4_d1_protocol_b_eligibility_verifier.py"
        verifier_source = verifier_path.read_text(encoding="utf-8")
        tree = ast.parse(verifier_source)
        forbidden = "rpe.runner.phase4_d1_protocol_b_eligibility"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == forbidden:
                self.fail("verifier must not import any symbol from phase4_d1_protocol_b_eligibility.py")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden:
                        self.fail("verifier must not import phase4_d1_protocol_b_eligibility.py")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "exec":
                self.fail("verifier must not exec production source")

        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=MODEL_SEEDS,
            shard_source_count=2,
            native_point_count=16,
            support_point_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            built = build_phase4_d1_protocol_b_eligibility_from_inputs(
                output,
                inputs=inputs,
                config=config,
                worker_count=2,
            )
            verified = verify_phase4_d1_protocol_b_eligibility_from_inputs(
                built.path,
                inputs=inputs,
                config_path=built.path / "config.json",
                worker_count=1,
            )
            self.assertEqual(verified.run_id, built.run_id)
            self.assertEqual(verified.status, built.status)
            self.assertEqual(verified.path, built.path)
            self.assertEqual(verified.source_record_count, built.source_record_count)
            self.assertEqual(verified.condition_matrix_shard_count, built.condition_matrix_shard_count)

    def test_worker_pool_uses_spawn_one_blas_thread_and_admission_before_output(self) -> None:
        # Contract: worker pool uses spawn, and BLAS threads are forced to one per worker.
        subject_path = ROOT / "rpe/runner/phase4_d1_protocol_b_eligibility.py"
        source = subject_path.read_text(encoding="utf-8")
        self.assertIn("get_context", source)
        self.assertIn("spawn", source)
        self.assertIn("OMP_NUM_THREADS", source)

        # Admission contract: check memory before creating output directory.
        inputs = make_synthetic_d1_protocol_b_inputs(
            class_count=2,
            records_per_class=2,
            model_seeds=MODEL_SEEDS,
            shard_source_count=2,
            native_point_count=100000,
            support_point_count=8,
        )
        config = make_synthetic_d1_protocol_b_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            with patch(
                "rpe.runner.phase4_d1_protocol_b_eligibility.estimate_p10_peak_bytes",
                return_value=64 * 2**30 + 1,
            ):
                with self.assertRaisesRegex(Phase4D1ProtocolBEligibilityError, "64 GiB"):
                    build_phase4_d1_protocol_b_eligibility_from_inputs(
                        output,
                        inputs=inputs,
                        config=config,
                        worker_count=1,
                    )
            self.assertFalse(output.exists())


class Phase4D1ProtocolBEligibilityCliContractTest(unittest.TestCase):
    def test_cli_rejects_outcome_affecting_overrides(self) -> None:
        # Contract: CLI provides only build/verify and rejects any override of
        # seed/grid/support/parent/resample/skip/outcome.
        with self.assertRaises(SystemExit):
            d1_protocol_b_cli_main(["verify", "--run-path", "x", "--no-reexecute"])
        with self.assertRaises(SystemExit):
            d1_protocol_b_cli_main(["build", "--output-root", "x", "--protocol", "A"])
        with self.assertRaises(SystemExit):
            d1_protocol_b_cli_main(["build", "--output-root", "x", "--seed", "0"])
        with self.assertRaises(SystemExit):
            d1_protocol_b_cli_main(["build", "--output-root", "x", "--support", "x"])
        with self.assertRaises(SystemExit):
            d1_protocol_b_cli_main(["build", "--output-root", "x", "--outcome"])


if __name__ == "__main__":
    unittest.main()

