from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/phase6/configs/baseline_evidence_v1.json"
CATALOG = ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
sys.path.insert(0, str(ROOT))

from rpe.methods import TaskLine, load_classical_catalog  # noqa: E402
import rpe.runner.phase6_baseline_evidence as evidence_module  # noqa: E402
from rpe.runner.phase6_baseline_evidence import (  # noqa: E402
    BaselineEvidenceConfig,
    BaselineEvidenceError,
    BaselineEvidenceInputs,
    BaselineEvidenceSummary,
    build_phase6_baseline_evidence,
    build_phase6_baseline_evidence_from_inputs,
    load_phase6_baseline_evidence_config,
    make_synthetic_baseline_evidence_inputs,
)


def canonical(value: object) -> bytes:
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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        row = json.loads(line)
        if line != canonical(row):
            raise AssertionError(f"{path.name} is not canonical JSONL")
        rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def artifact_files(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def rewrite_sha256sums(path: Path) -> None:
    names = sorted(
        item.name
        for item in path.iterdir()
        if item.is_file() and item.name != "SHA256SUMS"
    )
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )


def fixture_systems(config: BaselineEvidenceConfig):
    catalog = load_classical_catalog(CATALOG)
    by_id = {
        system.system_id: system
        for system in catalog.systems
        if system.task_line is TaskLine.BASELINE_CORRECTION
    }
    selected = []
    seen_families = set()
    for system_id in config.system_ids:
        system = by_id[system_id]
        if system.family_id not in seen_families:
            selected.append(system)
            seen_families.add(system.family_id)
        if len(selected) == 2:
            break
    return tuple(selected)


class Phase6BaselineEvidenceConfigTest(unittest.TestCase):
    def test_public_surface_and_canonical_config_exist(self) -> None:
        self.assertTrue(issubclass(BaselineEvidenceError, ValueError))
        for value in (
            BaselineEvidenceConfig,
            BaselineEvidenceInputs,
            BaselineEvidenceSummary,
            load_phase6_baseline_evidence_config,
            make_synthetic_baseline_evidence_inputs,
            build_phase6_baseline_evidence_from_inputs,
            build_phase6_baseline_evidence,
        ):
            self.assertTrue(callable(value))
        raw = CONFIG.read_bytes()
        self.assertEqual(raw, canonical(json.loads(raw)))

    def test_config_binds_exact_order_families_counts_and_power_failure(self) -> None:
        config = load_phase6_baseline_evidence_config(CONFIG)
        promotion = read_json(ROOT / config.authorities["baseline_promotion"]["path"])
        self.assertEqual(config.schema_version, "phase6-baseline-evidence-v1")
        self.assertEqual(config.system_ids, tuple(promotion["phase5_eligible_system_ids"]))
        self.assertEqual(len(config.system_ids), 112)
        self.assertEqual(
            dict(config.family_counts),
            {"airpls": 15, "asls": 15, "beads": 13, "iasls": 10,
             "imodpoly": 15, "morphological": 14, "psalsa": 15, "snip": 15},
        )
        self.assertEqual(config.document["gate"]["phase5_power_state"], "not_powered_for_phase5")
        self.assertEqual(config.document["gate"]["common_successful_record_count"], 9686)

    def test_config_binds_support_endpoints_direct_gt_and_exact_row_model(self) -> None:
        config = load_phase6_baseline_evidence_config(CONFIG)
        self.assertEqual(config.document["cohorts"]["d5"]["common_support_point_count"], 799)
        self.assertEqual(config.document["cohorts"]["d5"]["common_support"], "204,206,...,1800 cm^-1")
        self.assertEqual(config.direct_gt_state, "not_evaluable_no_validated_clean_gt")
        self.assertEqual(
            tuple(row["endpoint_id"] for row in config.endpoint_manifest),
            ("availability", "direct_gt", "D5-A", "D5-B-matched-reference",
             "D5-reference-free", "D4-A", "D4-B", "D4-reference-free",
             "D4-half-split"),
        )
        self.assertEqual(
            dict(config.expected),
            {"bootstrap_result_row_count": 784, "d4_half_split_well_row_count": 26880,
             "downstream_cluster_row_count": 206304, "family_projection_row_count": 72,
             "fixed_slot_row_count": 1008, "reference_free_cluster_row_count": 103152,
             "system_count": 112, "system_status_row_count": 112,
             "transform_receipt_row_count": 1282400},
        )

    def test_config_binds_models_splits_closures_inventory_and_authority_identity(self) -> None:
        config = load_phase6_baseline_evidence_config(CONFIG)
        document = config.document
        self.assertEqual(tuple(document["pls2"]["component_grid"]), (2, 4, 8, 16, 32))
        self.assertTrue(document["pls2"]["no_train_plus_validation_refit"])
        self.assertEqual(document["half_split"]["records_per_half"], 16)
        self.assertEqual(
            tuple(document["half_split"]["sort_key"]),
            ("source_round", "source_repetition", "record_id"),
        )
        self.assertEqual(tuple(document["component_closure"]["d5_a_matcher_failure"]), ("D5-A",))
        self.assertEqual(len(document["artifact_contract"]["payload_files"]), 12)
        self.assertEqual(document["artifact_contract"]["total_file_count_with_terminal_and_sha256sums"], 14)
        closure = document["authorities"]["phase4_closure_report"]
        self.assertEqual(closure["path"], "reports/phase4/step38_phase4_closure_synthesis.md")
        self.assertEqual(closure["byte_count"], 15853)
        self.assertEqual(closure["sha256"], "d0fe0c8dc5be9791385cc264e6c2764bf3eb96b531102978ba4eb4ed5895d203")

    def test_config_binds_raw_d5_checksum_ledger_exactly(self) -> None:
        config = load_phase6_baseline_evidence_config(CONFIG)
        ledger = config.authorities["d5_raw_sha256sums"]
        self.assertEqual(ledger["path"], "data/unified/rruff_raman_raw/SHA256SUMS")
        self.assertEqual(ledger["byte_count"], 235)
        self.assertEqual(ledger["sha256"], "f0cb09af8d80cdf3d52af9ed01bc1ae48abba94fefdff712c30d6e8c8bfe4995")

    def test_formal_inventory_validator_requires_exact_promotion_order_and_family_population(self) -> None:
        config = load_phase6_baseline_evidence_config(CONFIG)
        with self.assertRaisesRegex(BaselineEvidenceError, "formal.*order|eligible.*order"):
            evidence_module._validate_formal_system_inventory(config, config.system_ids[::-1], dict(config.family_counts))
        with self.assertRaisesRegex(BaselineEvidenceError, "family population"):
            evidence_module._validate_formal_system_inventory(config, config.system_ids, {"airpls": 112})

    def test_verifier_authority_check_requires_phase3_promotion_order(self) -> None:
        from rpe.runner import phase6_baseline_evidence_verifier as verifier

        config = load_phase6_baseline_evidence_config(CONFIG)
        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        document["inventory"]["system_ids"] = list(config.system_ids[::-1])
        with self.assertRaisesRegex(verifier.BaselineEvidenceVerificationError, "eligible system order"):
            verifier._check_authorities(document, ROOT)

    def test_pls_fit_functions_treat_warnings_as_errors(self) -> None:
        from rpe.runner import phase6_baseline_evidence_verifier as verifier_module

        production = ast.parse((ROOT / "rpe/runner/phase6_baseline_evidence.py").read_text(encoding="utf-8"))
        verifier = ast.parse((ROOT / "rpe/runner/phase6_baseline_evidence_verifier.py").read_text(encoding="utf-8"))
        for tree, function_name in ((production, "_fit_pls"), (verifier, "_fit")):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
            source = ast.unparse(function)
            self.assertIn("warnings.catch_warnings", source)
            self.assertIn("warnings.simplefilter('error')", source)
        for module in (evidence_module, verifier_module):
            self.assertEqual(
                module._scientific_finite_or_close(UserWarning("PLS convergence"), "consumer"),
                "not_evaluable_consumer_failure",
            )

    def test_verifier_pls_warning_closes_the_component_grid_like_production(self) -> None:
        from rpe.runner import phase6_baseline_evidence_verifier as verifier

        class WarningPLS:
            def __init__(self, **_: object) -> None:
                pass

            def fit(self, *_: object) -> None:
                warnings.warn("frozen PLS warning", UserWarning)

        x = np.ones((40, 40), dtype=np.float64)
        y = np.ones((40, 4), dtype=np.float64)
        with mock.patch.object(verifier, "PLSRegression", WarningPLS):
            with self.assertRaisesRegex(ValueError, "incomplete PLS grid") as raised:
                verifier._fit(x, y, x, y, 1)
        self.assertEqual(
            verifier._fit_failure_receipt(1, raised.exception),
            {
                "fold": 1,
                "state": "not_evaluable_fit_failure",
                "reason": "BaselineEvidenceError",
            },
        )

    def test_verifier_fixture_mode_requires_the_exact_known_fixture_ids(self) -> None:
        from rpe.runner import phase6_baseline_evidence_verifier as verifier

        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        known_fixture_ids = (
            "0372569fa54794e9f23b37477778de025295f22cd48a52d2aa762991b31119a0",
            "04c18275a33cca6b877282edca72e68808ea658d33b482bc428fbdccf6af8377",
        )
        self.assertEqual(
            verifier._select_rebuild_mode(known_fixture_ids, 2 * 328, document),
            "fixture",
        )
        with self.assertRaisesRegex(
            verifier.BaselineEvidenceVerificationError, "fixture|formal|inventory"
        ):
            verifier._select_rebuild_mode(known_fixture_ids[:1], 328, document)


class Phase6BaselineEvidenceHelperSemanticsTest(unittest.TestCase):
    def test_d5_native_correction_precedes_799_projection(self) -> None:
        native_axis = np.array([203.0, 204.0, 205.0, 206.0, 208.0], dtype=np.float64)
        corrected_native = np.array([10.0, 8.0, 4.0, 2.0, 0.0], dtype=np.float64)
        support = np.array([204.0, 206.0, 208.0], dtype=np.float64)
        projected = evidence_module._project_d5_corrected(native_axis, corrected_native, support)
        self.assertEqual(projected.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(projected, np.array([8.0, 2.0, 0.0], dtype="<f4"))

    def test_d5_protocol_roles_and_class_equal_top1_are_literal(self) -> None:
        roles = evidence_module._d5_protocol_roles()
        self.assertEqual(roles["D5-A"], {"query": "candidate", "library": "identity"})
        self.assertEqual(roles["D5-B-matched-reference"], {"query": "candidate", "library": "candidate"})
        value = evidence_module._class_equal_top1_error(
            ("class-a", "class-a", "class-a", "class-b"),
            (True, True, True, False),
        )
        self.assertEqual(value, 0.5)
        self.assertNotEqual(value, 0.75)

    def test_d4_a_is_fixed_b_is_refit_and_neither_fits_test(self) -> None:
        lifecycle = evidence_module._d4_model_lifecycle()
        self.assertEqual(lifecycle["D4-A"]["representation"], "identity_train_validation")
        self.assertEqual(lifecycle["D4-A"]["candidate_roles"], ("test",))
        self.assertFalse(lifecycle["D4-A"]["refit_per_system"])
        self.assertEqual(lifecycle["D4-B"]["fit_roles"], ("train",))
        self.assertEqual(lifecycle["D4-B"]["selection_roles"], ("validation",))
        self.assertTrue(lifecycle["D4-B"]["refit_per_system"])
        self.assertNotIn("test", lifecycle["D4-B"]["fit_roles"])

    def test_normalized_loss_reference_free_and_half_split_are_literal(self) -> None:
        targets = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
        identity = targets + 0.1
        candidate = targets + 0.02
        self.assertAlmostEqual(
            evidence_module._normalized_squared_loss(candidate, targets),
            float(np.mean(((candidate - targets) ** 2) / (0.32 ** 2))),
        )
        self.assertAlmostEqual(evidence_module._candidate_minus_identity(0.7, 0.2), 0.5)
        records = [(1 + index % 4, 1 + index // 4, f"r{index:02d}") for index in range(32)]
        even, odd = evidence_module._half_split_record_ids(records)
        self.assertEqual((len(even), len(odd)), (16, 16))
        self.assertFalse(set(even) & set(odd))
        self.assertEqual((even[0], odd[0]), ("r00", "r04"))

    def test_component_closure_is_local_and_fixed_denominators_remain(self) -> None:
        self.assertEqual(evidence_module._closed_endpoint_ids("d5_a_matcher_failure"), ("D5-A",))
        self.assertEqual(
            evidence_module._closed_endpoint_ids("d4_transform_failure"),
            ("D4-A", "D4-B", "D4-reference-free", "D4-half-split"),
        )
        rows = evidence_module._fixed_method_rows(
            system_id="fixture-system",
            component_states={"D5-A": (None, "not_evaluable_consumer_failure")},
            phase5_power_state="not_powered_for_phase5",
            direct_gt_state="not_evaluable_no_validated_clean_gt",
        )
        self.assertEqual(len(rows), 9)
        self.assertEqual(next(row for row in rows if row["endpoint_id"] == "D5-A")["value"], None)

    def test_fixed_cluster_closure_keeps_typed_reason_and_canonical_order(self) -> None:
        rows = evidence_module._fixed_cluster_rows(
            [],
            (type("System", (), {"system_id": "s", "family_id": "f", "method_id": "m"})(),),
            ("b", "a"),
            ("D5-A",),
            kind="downstream",
            closure_reasons={("s", "D5-A"): "not_evaluable_metric_domain"},
        )
        self.assertEqual([row["cluster_id"] for row in rows], ["a", "b"])
        self.assertTrue(all(row["state"] == "not_evaluable_metric_domain" for row in rows))

    def test_finite_validator_closes_scientific_output_but_propagates_invariant(self) -> None:
        self.assertEqual(evidence_module._scientific_finite_or_close(float("nan"), "metric"), "not_evaluable_metric_domain")
        with self.assertRaises(KeyError):
            evidence_module._scientific_finite_or_close(KeyError("bad invariant"), "metric")

    def test_shared_cluster_bootstrap_is_seeded_and_uses_equal_cluster_weight(self) -> None:
        values = np.array([0.0, 2.0, 10.0], dtype=np.float64)
        first = evidence_module._bootstrap_cluster_mean(values, resamples=2000, seed=20260817)
        second = evidence_module._bootstrap_cluster_mean(values, resamples=2000, seed=20260817)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["estimate"], 4.0)
        self.assertEqual(first["cluster_count"], 3)

    def test_bootstrap_preserves_the_unique_typed_component_closure(self) -> None:
        system = type(
            "System",
            (),
            {"system_id": "s", "family_id": "f", "method_id": "m"},
        )()
        downstream = [
            {"system_id": "s", "protocol_id": "D5-A", "cluster_id": "c", "effect": None, "state": "not_evaluable_consumer_failure"},
            {"system_id": "s", "protocol_id": "D5-B-matched-reference", "cluster_id": "c", "effect": 0.0, "state": "complete"},
            {"system_id": "s", "protocol_id": "D4-A", "cluster_id": "w", "effect": 0.0, "state": "complete"},
            {"system_id": "s", "protocol_id": "D4-B", "cluster_id": "w", "effect": 0.0, "state": "complete"},
        ]
        reference = [
            {"system_id": "s", "cohort_id": "D5", "cluster_id": "c", "effect": 0.0, "state": "complete"},
            {"system_id": "s", "cohort_id": "D4", "cluster_id": "w", "effect": 0.0, "state": "complete"},
        ]
        half = [
            {"system_id": "s", "well_id": "w", "effect": 0.0, "state": "complete"}
        ]
        rows = evidence_module._bootstrap_rows_from_clusters(
            (system,), downstream, reference, half, resamples=10, seed=20260817
        )
        d5_a = next(row for row in rows if row["endpoint_id"] == "D5-A")
        self.assertEqual(d5_a["state"], "not_evaluable_consumer_failure")


class Phase6BaselineEvidenceArtifactTest(unittest.TestCase):
    def fixture(self):
        config = load_phase6_baseline_evidence_config(CONFIG)
        return config, make_synthetic_baseline_evidence_inputs(config=config), fixture_systems(config)

    def test_tiny_fixture_build_has_fixed_arithmetic_inventory_and_direct_gt_closure(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_baseline_evidence_from_inputs(
                inputs, systems, Path(output_root), config=config, worker_count=1, project_root=ROOT
            )
            self.assertEqual(summary.system_count, len(systems))
            self.assertEqual(summary.transform_receipt_row_count, len(systems) * (inputs.d5_record_count + inputs.d4_record_count))
            self.assertEqual(summary.method_evidence_row_count, len(systems) * 9)
            self.assertEqual(summary.bootstrap_result_row_count, len(systems) * 7)
            expected = set(config.artifact_contract["payload_files"]) | {"complete.json", "SHA256SUMS"}
            self.assertEqual(set(artifact_files(summary.path)), expected)
            rows = read_csv(summary.path / "method_evidence_rows.csv")
            direct = [row for row in rows if row["endpoint_id"] == "direct_gt"]
            self.assertEqual(len(direct), len(systems))
            self.assertTrue(all(row["state"] == "not_evaluable_no_validated_clean_gt" for row in direct))
            self.assertTrue(all(row["phase5_power_state"] == "not_powered_for_phase5" for row in rows))
            typed_closed = [
                row
                for row in rows
                if row["endpoint_id"] == "D4-reference-free"
                and row["state"] != "complete"
            ]
            self.assertEqual(len(typed_closed), 1)
            self.assertEqual(typed_closed[0]["state"], "not_evaluable_metric_domain")

    def test_one_native_correction_receipt_is_reused_by_all_consumers(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_baseline_evidence_from_inputs(
                inputs, systems[:1], Path(output_root), config=config, worker_count=1, project_root=ROOT
            )
            receipts = read_jsonl(summary.path / "transform_receipts.jsonl")
            keys = [(row["system_id"], row["cohort_id"], row["record_id"]) for row in receipts]
            self.assertEqual(len(keys), len(set(keys)))
            status = read_jsonl(summary.path / "system_status.jsonl")[0]
            self.assertEqual(status["d5"]["consumer_input_sha256"]["D5-A"], status["d5"]["consumer_input_sha256"]["D5-B-matched-reference"])
            self.assertEqual(status["d4"]["native_corrected_matrix_sha256"], status["d4"]["consumer_input_sha256"]["D4-B"])

    def test_preflight_records_independent_identity_a_b_receipts(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_baseline_evidence_from_inputs(inputs, systems[:1], Path(output_root), config=config, worker_count=1, project_root=ROOT)
            preflight = read_json(summary.path / "preflight.json")
            self.assertTrue(preflight["identity_d5_matcher"]["exact_equal"])
            self.assertTrue(preflight["identity_d4_models"]["exact_equal"])
            self.assertEqual(preflight["identity_d4_models"]["D4-A"], preflight["identity_d4_models"]["D4-B"])

    def test_worker_count_is_byte_deterministic_and_existing_run_collides(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = build_phase6_baseline_evidence_from_inputs(
                inputs, systems, Path(first_root), config=config, worker_count=1, project_root=ROOT
            )
            second = build_phase6_baseline_evidence_from_inputs(
                inputs, systems, Path(second_root), config=config, worker_count=2, project_root=ROOT
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(artifact_files(first.path), artifact_files(second.path))
            with self.assertRaisesRegex(BaselineEvidenceError, "exist|collision"):
                build_phase6_baseline_evidence_from_inputs(
                    inputs, systems, Path(first_root), config=config, worker_count=1, project_root=ROOT
                )

    def test_independent_verifier_accepts_fixture_and_rejects_checksum_rewritten_tamper(self) -> None:
        from rpe.runner.phase6_baseline_evidence_verifier import verify_phase6_baseline_evidence

        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_baseline_evidence_from_inputs(
                inputs, systems, Path(output_root), config=config, worker_count=1, project_root=ROOT
            )
            verified = verify_phase6_baseline_evidence(summary.path, worker_count=2, project_root=ROOT)
            self.assertEqual(verified.run_id, summary.run_id)
            rows = read_jsonl(summary.path / "downstream_cluster_rows.jsonl")
            numeric = next(row for row in rows if row["effect"] is not None)
            numeric["effect"] = float(numeric["effect"]) + 0.5
            (summary.path / "downstream_cluster_rows.jsonl").write_bytes(
                b"".join(canonical(row) for row in rows)
            )
            rewrite_sha256sums(summary.path)
            with self.assertRaisesRegex(Exception, "mismatch|tamper|byte"):
                verify_phase6_baseline_evidence(summary.path, worker_count=1, project_root=ROOT)

    def test_verifier_source_is_independent_of_production_runner(self) -> None:
        path = ROOT / "rpe/runner/phase6_baseline_evidence_verifier.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                banned.extend(alias.name for alias in node.names if alias.name == "rpe.runner.phase6_baseline_evidence")
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase6_baseline_evidence":
                banned.append(node.module)
        self.assertEqual(banned, [])
        names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertNotIn("build_phase6_baseline_evidence_from_inputs", names)

    def test_verifier_worker_executes_native_transforms_not_identifier_only(self) -> None:
        from rpe.runner import phase6_baseline_evidence_verifier as verifier

        source = Path(verifier.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_evaluate_verifier_system")
        calls = {node.func.id for node in ast.walk(worker) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn("_transform", calls)
        self.assertIn("_transform_native_d5", calls)
        self.assertIn("executor.map(_evaluate_verifier_system, range(len(systems)), chunksize=1)", source)

    def test_verifier_finite_helper_closes_scientific_outputs_and_propagates_invariants(self) -> None:
        from rpe.runner import phase6_baseline_evidence_verifier as verifier

        self.assertEqual(verifier._scientific_finite_or_close(float("nan"), "metric"), "not_evaluable_metric_domain")
        self.assertEqual(verifier._scientific_finite_or_close(np.array([1.0, np.inf]), "consumer"), "not_evaluable_consumer_failure")
        with self.assertRaises(KeyError):
            verifier._scientific_finite_or_close(KeyError("invariant"), "metric")


class Phase6BaselineEvidenceCliTest(unittest.TestCase):
    def test_cli_surface_is_narrow_and_rejects_scientific_overrides(self) -> None:
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "tools/run_phase6_baseline_evidence.py", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("build", result.stdout)
        self.assertIn("verify", result.stdout)
        from tools import run_phase6_baseline_evidence as cli_module

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli_module.main(["build", "--output-root", "unused", "--bootstrap-seed", "7"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
