from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rpe.evaluation import Peak1D
from rpe.methods.catalog import TaskLine, load_classical_catalog
import rpe.runner.phase6_peak_evidence as evidence_module

from rpe.runner.phase6_peak_evidence import (
    PeakEvidenceConfig,
    PeakEvidenceError,
    PeakEvidenceInputs,
    PeakEvidenceSummary,
    build_phase6_peak_evidence,
    build_phase6_peak_evidence_from_inputs,
    derive_peak_evidence_cohort_projection,
    evaluate_mechanism_condition,
    load_phase6_peak_evidence_config,
    make_synthetic_peak_evidence_inputs,
)
from rpe.runner.phase6_peak_evidence_verifier import (
    verify_phase6_peak_evidence,
)
import rpe.runner.phase6_peak_evidence_verifier as verifier_module
import tools.run_phase6_peak_evidence as cli_module


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/phase6/configs/peak_evidence_v1.json"
PHASE1_RUN = ROOT / (
    "data/perturbed/phase1_rruff_core10k/"
    "phase1-rruff-core10k-ca17c81ec5ff89e5cbb9170e2945b5a2a4f10d242f268abef41aae062ad549f4"
)


def peak(position: float, *, fwhm: float = 2.0) -> Peak1D:
    return Peak1D(
        position_cm1=position,
        height=10.0,
        fwhm_cm1=fwhm,
        area=20.0,
        prominence=5.0,
    )


def endpoint(rows, endpoint_id: str):
    return next(row for row in rows if row["endpoint_id"] == endpoint_id)


def files(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def rewrite_sha256sums(path: Path) -> None:
    names = sorted(item.name for item in path.iterdir() if item.is_file() and item.name != "SHA256SUMS")
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )


class Phase6PeakEvidencePublicContractTest(unittest.TestCase):
    def test_public_surface_exists(self) -> None:
        self.assertTrue(issubclass(PeakEvidenceError, ValueError))
        self.assertTrue(callable(PeakEvidenceConfig))
        self.assertTrue(callable(PeakEvidenceInputs))
        self.assertTrue(callable(PeakEvidenceSummary))
        self.assertTrue(callable(load_phase6_peak_evidence_config))
        self.assertTrue(callable(derive_peak_evidence_cohort_projection))
        self.assertTrue(callable(make_synthetic_peak_evidence_inputs))
        self.assertTrue(callable(evaluate_mechanism_condition))
        self.assertTrue(callable(build_phase6_peak_evidence_from_inputs))
        self.assertTrue(callable(build_phase6_peak_evidence))


class Phase6PeakEvidenceConfigTest(unittest.TestCase):
    def test_config_is_canonical_and_binds_frozen_scope(self) -> None:
        raw = CONFIG.read_bytes()
        document = json.loads(raw)
        expected = (
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(raw, expected)
        config = load_phase6_peak_evidence_config(CONFIG)
        self.assertEqual(config.schema_version, "phase6-peak-evidence-v1")
        self.assertEqual(len(config.promoted_system_ids), 21)
        self.assertEqual(len(config.blocked_systems), 15)
        self.assertEqual(config.positive_alpha_grid, (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8))
        self.assertEqual(config.tolerances_cm1, (1.0, 2.0, 4.0, 8.0))
        self.assertEqual(config.bootstrap["resamples"], 2000)
        self.assertEqual(config.bootstrap["seed"], 20260817)
        self.assertEqual(config.cohort["projection_sha256"], "682c3ea2fe5f776be3e12632e66aa3352196b46612889e808ba82f08845473a0")
        self.assertEqual(
            config.expected,
            {
                "blocked_system_count": 15,
                "bootstrap_row_count": 210,
                "detector_call_count": 1937250,
                "detector_receipt_row_count": 47250,
                "family_projection_row_count": 27,
                "intervention_truth_row_count": 11250,
                "method_evidence_row_count": 288,
                "positive_detector_call_count": 1890000,
                "promoted_method_evidence_row_count": 273,
                "promoted_system_count": 21,
                "synthetic_class_row_count": 472500,
                "synthetic_condition_row_count": 1890000,
                "system_status_row_count": 36,
            },
        )
        self.assertEqual(len(config.endpoint_manifest), 13)
        directions = {row["endpoint_id"]: row["preferred_direction"] for row in config.endpoint_manifest}
        self.assertEqual(directions["P3-position-mae"], "lower_is_better")
        self.assertEqual(directions["P3-signed-log2-fwhm-ratio"], "non_monotonic")
        self.assertEqual(directions["P4-deleted-event-disappearance"], "higher_is_better")
        self.assertEqual(directions["P5-inserted-event-detection-gain"], "higher_is_better")

    def test_real_cohort_projection_rederives_exact_frozen_hash(self) -> None:
        payload = derive_peak_evidence_cohort_projection(PHASE1_RUN)
        rows = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual(len(rows), 2250)
        self.assertEqual(tuple(rows[0]), ("class_label", "selection_rank", "source_record_id"))
        self.assertEqual(
            [(row["class_label"], row["selection_rank"]) for row in rows],
            sorted((row["class_label"], row["selection_rank"]) for row in rows),
        )
        self.assertEqual(len({row["class_label"] for row in rows}), 2250)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), "682c3ea2fe5f776be3e12632e66aa3352196b46612889e808ba82f08845473a0")

    def test_phase1_top_level_checksum_ledger_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "payload.bin").write_bytes(b"authoritative")
            digest = hashlib.sha256(b"authoritative").hexdigest()
            (path / "SHA256SUMS").write_text(
                f"{digest}  payload.bin\n", encoding="utf-8"
            )
            (path / "failed.json").write_bytes(b"{}\n")
            evidence_module._verify_phase1_artifact_checksums(path)
            (path / "payload.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(PeakEvidenceError, "checksum.*payload.bin"):
                evidence_module._verify_phase1_artifact_checksums(path)


class Phase6PeakEvidenceEndpointTest(unittest.TestCase):
    def test_p1_retention_uses_inclusive_greedy_matching(self) -> None:
        rows = evaluate_mechanism_condition(
            perturbation_id="p01",
            baseline_peaks=(peak(100.0), peak(200.0)),
            candidate_peaks=(peak(102.0), peak(205.0)),
            truth={},
            tolerance_cm1=2.0,
        )
        row = endpoint(rows, "P1-retention")
        self.assertEqual((row["numerator"], row["denominator"]), (1, 2))
        self.assertEqual(row["value"], 0.5)
        self.assertEqual(row["state"], "complete_numeric")

    def test_p2_matches_globally_before_partitioning_affected_supports(self) -> None:
        rows = evaluate_mechanism_condition(
            perturbation_id="p02",
            baseline_peaks=(peak(100.0), peak(102.0)),
            candidate_peaks=(peak(101.1),),
            truth={"affected_supports_cm1": ((99.0, 101.0),)},
            tolerance_cm1=2.0,
        )
        affected = endpoint(rows, "P2-affected-retention")
        unaffected = endpoint(rows, "P2-unaffected-retention")
        self.assertEqual((affected["numerator"], affected["denominator"], affected["value"]), (0, 1, 0.0))
        self.assertEqual((unaffected["numerator"], unaffected["denominator"], unaffected["value"]), (1, 1, 1.0))

    def test_p3_reuses_one_match_set_for_retention_position_and_width(self) -> None:
        rows = evaluate_mechanism_condition(
            perturbation_id="p03",
            baseline_peaks=(peak(100.0, fwhm=2.0), peak(200.0, fwhm=4.0)),
            candidate_peaks=(peak(101.0, fwhm=4.0), peak(198.5, fwhm=2.0)),
            truth={},
            tolerance_cm1=2.0,
        )
        self.assertEqual(endpoint(rows, "P3-retention")["value"], 1.0)
        self.assertEqual(endpoint(rows, "P3-position-mae")["value"], 1.25)
        self.assertEqual(endpoint(rows, "P3-signed-log2-fwhm-ratio")["value"], 0.0)

    def test_p4_conditions_disappearance_on_baseline_detected_deleted_events(self) -> None:
        rows = evaluate_mechanism_condition(
            perturbation_id="p04",
            baseline_peaks=(peak(100.5), peak(200.0)),
            candidate_peaks=(peak(200.5),),
            truth={"deleted_centers_cm1": (100.0,)},
            tolerance_cm1=2.0,
        )
        disappearance = endpoint(rows, "P4-deleted-event-disappearance")
        retention = endpoint(rows, "P4-undeleted-retention")
        self.assertEqual((disappearance["numerator"], disappearance["denominator"], disappearance["value"]), (1, 1, 1.0))
        self.assertEqual((retention["numerator"], retention["denominator"], retention["value"]), (1, 1, 1.0))

    def test_p4_empty_detector_risk_set_is_null_not_zero(self) -> None:
        rows = evaluate_mechanism_condition(
            perturbation_id="p04",
            baseline_peaks=(peak(200.0),),
            candidate_peaks=(peak(200.5),),
            truth={"deleted_centers_cm1": (100.0,)},
            tolerance_cm1=2.0,
        )
        row = endpoint(rows, "P4-deleted-event-disappearance")
        self.assertIsNone(row["value"])
        self.assertEqual(row["state"], "complete_empty_risk_set")
        self.assertEqual(row["reason_code"], "no_baseline_matched_deleted_event")

    def test_p5_uses_independent_counterfactual_matches_and_preserves_native_set(self) -> None:
        rows = evaluate_mechanism_condition(
            perturbation_id="p05",
            baseline_peaks=(peak(100.0), peak(200.0)),
            candidate_peaks=(peak(100.5), peak(150.5), peak(199.0)),
            truth={"inserted_centers_cm1": (150.0,)},
            tolerance_cm1=2.0,
        )
        gain = endpoint(rows, "P5-inserted-event-detection-gain")
        retention = endpoint(rows, "P5-native-peak-retention")
        self.assertEqual((gain["numerator"], gain["denominator"], gain["value"]), (1, 1, 1.0))
        self.assertEqual((retention["numerator"], retention["denominator"], retention["value"]), (2, 2, 1.0))

    def test_p5_zero_generator_events_keeps_both_rows_with_typed_null(self) -> None:
        rows = evaluate_mechanism_condition(
            perturbation_id="p05",
            baseline_peaks=(peak(100.0),),
            candidate_peaks=(peak(100.0),),
            truth={"inserted_centers_cm1": ()},
            tolerance_cm1=2.0,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNone(row["value"])
            self.assertEqual(row["state"], "not_applicable_zero_synthetic_events")
            self.assertEqual(row["reason_code"], "not_applicable_zero_synthetic_events")


class Phase6PeakEvidenceFixtureContractTest(unittest.TestCase):
    def fixture(self):
        config = load_phase6_peak_evidence_config(CONFIG)
        inputs = make_synthetic_peak_evidence_inputs(config=config)
        catalog = load_classical_catalog(ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json")
        systems = tuple(
            next(
                system
                for system in catalog.systems
                if system.task_line is TaskLine.PEAK_DETECTION
                and system.family_id == family
                and system.system_id in config.promoted_system_ids
            )
            for family in ("find_peaks", "find_peaks_cwt")
        )
        return config, inputs, systems

    def test_synthetic_fixture_build_has_fixed_dynamic_arithmetic(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_peak_evidence_from_inputs(
                inputs,
                systems,
                Path(output_root),
                config=config,
                worker_count=1,
                project_root=ROOT,
            )
            self.assertEqual(summary.system_count, 2)
            self.assertEqual(summary.source_count, inputs.source_count)
            self.assertEqual(summary.detector_call_count, 2 * inputs.source_count * 41)
            self.assertEqual(summary.condition_row_count, 2 * inputs.source_count * 40)
            self.assertEqual(summary.class_row_count, 2 * inputs.source_count * 10)
            self.assertEqual(summary.bootstrap_row_count, 20)
            self.assertEqual(summary.method_evidence_row_count, 26)
            self.assertEqual(len(tuple(summary.path.iterdir())), 14)
            self.assertEqual(
                sorted(path.name for path in summary.path.iterdir()),
                sorted((*config.artifact_contract["payload_files"], "complete.json", "SHA256SUMS")),
            )
            manifest = json.loads((summary.path / "manifest.json").read_bytes())
            self.assertEqual(manifest["detector_call_count"], 164)
            self.assertEqual(manifest["synthetic_condition_row_count"], 160)
            self.assertEqual(manifest["synthetic_class_row_count"], 40)
            self.assertEqual(manifest["bootstrap_row_count"], 20)

    def test_worker_count_does_not_change_artifact_bytes(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = build_phase6_peak_evidence_from_inputs(
                inputs, systems, Path(first_root), config=config, worker_count=1, project_root=ROOT
            )
            second = build_phase6_peak_evidence_from_inputs(
                inputs, systems, Path(second_root), config=config, worker_count=2, project_root=ROOT
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(files(first.path), files(second.path))

    def test_independent_verifier_reexecutes_and_rejects_semantic_tamper(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_peak_evidence_from_inputs(
                inputs, systems, Path(output_root), config=config, worker_count=1, project_root=ROOT
            )
            verified = verify_phase6_peak_evidence(summary.path, worker_count=2, project_root=ROOT)
            self.assertEqual(verified.run_id, summary.run_id)
            rows = (summary.path / "synthetic_condition_rows.jsonl").read_bytes().splitlines()
            changed = json.loads(rows[0])
            changed["candidate_peak_count"] += 1
            rows[0] = (json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n").encode().rstrip(b"\n")
            (summary.path / "synthetic_condition_rows.jsonl").write_bytes(b"\n".join(rows) + b"\n")
            rewrite_sha256sums(summary.path)
            with self.assertRaisesRegex(Exception, "mismatch|tamper|byte"):
                verify_phase6_peak_evidence(summary.path, worker_count=1, project_root=ROOT)

    def test_verifier_module_does_not_import_production_runner(self) -> None:
        path = ROOT / "rpe/runner/phase6_peak_evidence_verifier.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                banned.extend(alias.name for alias in node.names if alias.name == "rpe.runner.phase6_peak_evidence")
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase6_peak_evidence":
                banned.append(node.module)
        self.assertEqual(banned, [])
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("build_phase6_peak_evidence_from_inputs", function_names)

    def test_verifier_bootstrap_matches_finite_mask_mean_order(self) -> None:
        config, _inputs, systems = self.fixture()
        system = systems[0]
        class_rows = []
        for class_label in (1, 2, 3):
            for endpoint_id in evidence_module.SYNTHETIC_ENDPOINT_ORDER:
                responses = []
                for alpha_index, alpha in enumerate(config.positive_alpha_grid):
                    by_tolerance = {}
                    for tolerance in config.tolerances_cm1:
                        key = format(tolerance, "g")
                        if class_label == 2 and alpha_index == 3:
                            cell = {"state": "complete_empty_risk_set", "value": None}
                        else:
                            cell = {
                                "state": "complete_numeric",
                                "value": (class_label * 0.1) + (alpha_index * 0.003) + (tolerance * 0.0001),
                            }
                        by_tolerance[key] = cell
                    responses.append({"alpha": alpha, "by_tolerance_cm1": by_tolerance})
                class_rows.append({
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "class_label": class_label,
                    "endpoint_id": endpoint_id,
                    "responses": responses,
                })
        production = evidence_module._bootstrap_rows(
            class_rows=class_rows, systems=(system,), config=config
        )
        verifier_config = verifier_module._load_config(CONFIG)
        independent = verifier_module._bootstrap(class_rows, (system,), verifier_config)
        self.assertEqual(
            evidence_module._jsonl_bytes(production),
            verifier_module._jsonl(independent),
        )

    def test_formal_count_mismatch_fails_closed_with_terminal_marker(self) -> None:
        config, inputs, systems = self.fixture()
        invalid_formal_inputs = replace(inputs, synthetic_fixture=False)
        with tempfile.TemporaryDirectory() as output_root:
            root = Path(output_root)
            with self.assertRaisesRegex(PeakEvidenceError, "formal.*source|formal.*system|expected"):
                build_phase6_peak_evidence_from_inputs(
                    invalid_formal_inputs, systems, root, config=config, worker_count=1, project_root=ROOT
                )
            runs = tuple(root.iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "failed.json").is_file())
            self.assertFalse((runs[0] / "complete.json").exists())

    def test_runtime_exception_fails_closed_with_terminal_marker(self) -> None:
        config, inputs, systems = self.fixture()
        with tempfile.TemporaryDirectory() as output_root:
            root = Path(output_root)
            with mock.patch.object(evidence_module, "_worker_task", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    build_phase6_peak_evidence_from_inputs(
                        inputs, systems, root, config=config, worker_count=1, project_root=ROOT
                    )
            runs = tuple(root.iterdir())
            self.assertEqual(len(runs), 1)
            failure = json.loads((runs[0] / "failed.json").read_bytes())
            self.assertEqual(failure["status"], "failed")
            self.assertFalse((runs[0] / "complete.json").exists())


class Phase6PeakEvidenceCliTest(unittest.TestCase):
    def test_direct_help_bootstraps_repository_import_path(self) -> None:
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "tools/run_phase6_peak_evidence.py", "--help"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Build or independently verify Phase 6 peak evidence", result.stdout)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_cli_rejects_scientific_overrides_and_dispatches_workers(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli_module.main(["build", "--output-root", "/tmp/o", "--system-id", "x"])
        self.assertEqual(raised.exception.code, 2)
        built = mock.Mock(path=Path("/tmp/p"), run_id="run")
        with mock.patch.object(cli_module, "build_phase6_peak_evidence", return_value=built) as call:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_module.main(["build", "--output-root", "/tmp/o", "--worker-count", "3"]), 0)
        call.assert_called_once_with(Path("/tmp/o"), worker_count=3)
        with mock.patch.object(cli_module, "verify_phase6_peak_evidence", return_value=built) as call:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_module.main(["verify", "--run-path", "/tmp/p", "--worker-count", "4"]), 0)
        call.assert_called_once_with(Path("/tmp/p"), worker_count=4)


if __name__ == "__main__":
    unittest.main()
