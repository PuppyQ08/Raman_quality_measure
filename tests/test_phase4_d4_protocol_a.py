from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


EXPECTED_PAYLOAD_FILES = (
    "config.json",
    "eligibility_bridge.json",
    "preflight.json",
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "record_measurements.jsonl",
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
    "figure1_d4_protocol_a_full_domain.png",
    "figure1_d4_protocol_a_full_domain.svg",
    "figure1_d4_protocol_a_full_domain_data.csv",
    "figure2_d4_protocol_a_full_domain.png",
    "figure2_d4_protocol_a_full_domain.svg",
    "figure2_d4_protocol_a_full_domain_data.csv",
    "d4_protocol_a_full_domain_secondary_table.csv",
    "manifest.json",
)


def _load_subject():
    return importlib.import_module("rpe.runner.phase4_d4_protocol_a")


def _load_verifier():
    return importlib.import_module("rpe.runner.phase4_d4_protocol_a_verifier")


def _load_cli_module():
    return importlib.import_module("tools.run_phase4_d4_protocol_a")


class D4ProtocolARedCheckpointTest(unittest.TestCase):
    def test_red_checkpoint_missing_phase4_d4_protocol_a_module(self) -> None:
        _load_subject()


@unittest.skipUnless(
    importlib.util.find_spec("rpe.runner.phase4_d4_protocol_a") is not None,
    "awaiting Step 29 D4 Protocol-A implementation",
)
class D4ProtocolAContractTest(unittest.TestCase):
    def test_real_config_rejects_malformed_authority_receipt(self) -> None:
        subject = _load_subject()
        path = ROOT / "experiments/phase4/configs/d4_protocol_a_full_domain_v1.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["authorities"]["phase05_d4_protocol_audit"]["sha256"] = "abc"
        raw = (
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(subject.Phase4D4ProtocolAError, "authority|receipt|SHA"):
            subject.parse_phase4_d4_protocol_a_config(
                path, raw, require_frozen_identity=False
            )

    def test_public_surface_matches_step28_contract(self) -> None:
        subject = _load_subject()
        verifier = _load_verifier()

        required = (
            "ARTIFACT_PAYLOAD_FILES",
            "Phase4D4ProtocolAError",
            "Phase4D4ProtocolASummary",
            "parse_phase4_d4_protocol_a_config",
            "load_phase4_d4_protocol_a_config",
            "make_synthetic_d4_protocol_a_inputs",
            "make_synthetic_d4_protocol_a_config",
            "validate_d4_eligibility_parent",
            "fit_d4_protocol_a_models",
            "aggregate_d4_protocol_a",
            "render_d4_protocol_a_figures",
            "build_phase4_d4_protocol_a_from_inputs",
            "build_phase4_d4_protocol_a",
            "verify_phase4_d4_protocol_a_from_inputs",
        )
        missing = [name for name in required if not hasattr(subject, name)]
        self.assertEqual(missing, [])
        self.assertEqual(tuple(subject.ARTIFACT_PAYLOAD_FILES), EXPECTED_PAYLOAD_FILES)
        self.assertEqual(
            tuple(inspect.signature(verifier.verify_phase4_d4_protocol_a).parameters),
            ("path", "worker_count"),
        )
        self.assertNotIn("skip", inspect.getsource(verifier.verify_phase4_d4_protocol_a))
        self.assertNotIn("bypass", inspect.getsource(verifier.verify_phase4_d4_protocol_a))

    def test_failed_parent_marker_requires_evaluable_full_domain_core(self) -> None:
        subject = _load_subject()

        inputs = subject.make_synthetic_d4_protocol_a_inputs(
            well_count=5,
            acquisitions_per_well=2,
            model_folds=5,
        )
        accepted = subject.make_synthetic_d4_protocol_a_config(
            inputs,
            parent_marker_filename="failed.json",
            parent_full_domain_state="evaluable",
        )
        bridge = subject.validate_d4_eligibility_parent(Path("synthetic-parent"), accepted)
        self.assertEqual(bridge["marker_filename"], "failed.json")
        self.assertEqual(bridge["full_domain_state"], "evaluable")

        rejected = subject.make_synthetic_d4_protocol_a_config(
            inputs,
            parent_marker_filename="failed.json",
            parent_full_domain_state="not_evaluable_coverage",
        )
        with self.assertRaisesRegex(
            subject.Phase4D4ProtocolAError,
            "failed.json|full_domain_core|evaluable",
        ):
            subject.validate_d4_eligibility_parent(Path("synthetic-parent"), rejected)

    def test_public_builder_uses_retained_d4_inputs_and_never_touches_synthetic_fixture(self) -> None:
        subject = _load_subject()
        config = subject.load_phase4_d4_protocol_a_config(
            ROOT / "experiments/phase4/configs/d4_protocol_a_full_domain_v1.json"
        )
        parent_path = (
            ROOT
            / "results/phase4/d4_protocol_a_full_domain_eligibility_v1"
            / "phase4-d4-protocol-a-full-domain-eligibility-"
            "0f45ae5a0815ecb852b410549ad4582c4916dbe16fa0e3e20e2079f7d088097f"
        )
        expected_protocol_config = ROOT / "experiments/phase05/configs/d4_sugar_protocol.json"
        expected_archive = ROOT / "data/raw/ramanbench/cache/10779223/Raw data.zip"
        expected_eligibility_config = (
            ROOT / "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
        )

        support_axis = np.linspace(145.83834838867188, 3684.83544921875, 1999, dtype="<f8")
        well_ids = tuple(f"{row}{column}_{plate}" for plate in range(1, 3 + 1) for row in "ABCDEFGH" for column in range(1, 11))
        self.assertEqual(len(well_ids), 240)
        record_ids = tuple(
            f"{well_id}-r{round_index:02d}-rep{repetition:02d}"
            for well_id in well_ids
            for round_index in range(8)
            for repetition in range(4)
        )
        self.assertEqual(len(record_ids), 7680)
        record_well_ids = tuple(well_id for well_id in well_ids for _ in range(32))
        rounds = tuple(round_index for _ in well_ids for round_index in range(8) for _ in range(4))
        repetitions = tuple(repetition for _ in well_ids for _ in range(8) for repetition in range(4))
        blank_record_ids = tuple(f"E1_3-blank-{index:02d}" for index in range(32))
        retained_inputs = subject.D4ProtocolAInputs(
            synthetic_fixture=False,
            well_count=240,
            acquisitions_per_well=32,
            model_folds=5,
            feature_count=1999,
            record_ids=record_ids,
            well_ids=well_ids,
            record_well_ids=record_well_ids,
            fold_by_well={well_id: index // 48 for index, well_id in enumerate(well_ids)},
            rounds=rounds,
            repetitions=repetitions,
            targets=np.zeros((7680, 4), dtype="<f8"),
            target_names=subject.TARGET_NAMES,
            blank_record_ids=blank_record_ids,
            blank_targets=np.zeros((32, 4), dtype="<f8"),
            support_axis_cm1=support_axis,
            validation_macro_nrmse_by_component={2: 0.1, 4: 0.2, 8: 0.3, 16: 0.4, 32: 0.5},
            prediction_fixture=None,
            blank_fixture=None,
        )
        fake_cohort = object()
        fake_eligibility_config = object()
        captured: dict[str, object] = {}

        def fake_load_d4_sugar_cohort(protocol_path: Path, archive_path: Path) -> object:
            captured["cohort_loader_args"] = (protocol_path, archive_path)
            return fake_cohort

        def fake_load_phase4_d4_eligibility_config(path: Path) -> object:
            captured["eligibility_config_path"] = path
            return fake_eligibility_config

        def fake_validate_parent(path: Path, active_config: object) -> Mapping[str, object]:
            captured["parent_path"] = path
            captured["parent_config"] = active_config
            return {
                "full_domain_state": "evaluable",
                "marker_filename": "failed.json",
                "parent_path": str(path),
                "parent_run_id": path.name,
            }

        def fake_reconstruct_inputs(
            cohort: object,
            eligibility_config: object,
            active_config: object,
            parent_bridge: Mapping[str, object],
        ) -> object:
            captured["reconstruct_args"] = {
                "cohort": cohort,
                "eligibility_config": eligibility_config,
                "config": active_config,
                "parent_bridge": parent_bridge,
            }
            return retained_inputs

        def fake_build_from_inputs(
            output_dir: Path,
            *,
            inputs: object,
            config: object,
            worker_count: int,
            bootstrap_resamples: int | None = None,
            sign_flip_resamples: int | None = None,
        ) -> object:
            captured["build_from_inputs"] = {
                "output_dir": output_dir,
                "inputs": inputs,
                "config": config,
                "worker_count": worker_count,
                "bootstrap_resamples": bootstrap_resamples,
                "sign_flip_resamples": sign_flip_resamples,
            }
            expected_run_id = subject.RUN_PREFIX + hashlib.sha256(
                (
                    json.dumps(
                        {"record_count": len(inputs.record_ids), "sha256": config.sha256},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            return subject.Phase4D4ProtocolASummary(
                path=output_dir,
                run_id=expected_run_id,
                status="complete",
                record_count=len(inputs.record_ids),
                prediction_row_count=len(inputs.record_ids) * 41,
            )

        with (
            mock.patch.object(
                subject,
                "make_synthetic_d4_protocol_a_inputs",
                side_effect=AssertionError("synthetic helper must not run for authoritative build"),
            ),
            mock.patch.object(subject, "load_d4_sugar_cohort", side_effect=fake_load_d4_sugar_cohort, create=True),
            mock.patch.object(
                subject,
                "load_phase4_d4_eligibility_config",
                side_effect=fake_load_phase4_d4_eligibility_config,
                create=True,
            ),
            mock.patch.object(subject, "validate_d4_eligibility_parent", side_effect=fake_validate_parent),
            mock.patch.object(
                subject,
                "reconstruct_d4_protocol_a_inputs",
                side_effect=fake_reconstruct_inputs,
                create=True,
            ),
            mock.patch.object(subject, "build_phase4_d4_protocol_a_from_inputs", side_effect=fake_build_from_inputs),
        ):
            with tempfile.TemporaryDirectory() as directory:
                summary = subject.build_phase4_d4_protocol_a(Path(directory), worker_count=3)

        self.assertEqual(captured["cohort_loader_args"], (expected_protocol_config, expected_archive))
        self.assertEqual(captured["eligibility_config_path"], expected_eligibility_config)
        self.assertEqual(captured["parent_path"], parent_path)
        self.assertEqual(captured["parent_config"].sha256, config.sha256)
        self.assertIs(captured["reconstruct_args"]["cohort"], fake_cohort)
        self.assertIs(captured["reconstruct_args"]["eligibility_config"], fake_eligibility_config)
        self.assertEqual(captured["reconstruct_args"]["parent_bridge"]["full_domain_state"], "evaluable")
        self.assertIs(captured["build_from_inputs"]["inputs"], retained_inputs)
        self.assertEqual(captured["build_from_inputs"]["config"].sha256, config.sha256)
        self.assertEqual(captured["build_from_inputs"]["worker_count"], 3)
        self.assertEqual(summary.path.name, summary.run_id)
        self.assertFalse(retained_inputs.synthetic_fixture)
        self.assertEqual(len(retained_inputs.record_ids), 7680)
        self.assertEqual(len(retained_inputs.well_ids), 240)
        self.assertEqual(len(retained_inputs.blank_record_ids), 32)
        self.assertEqual(retained_inputs.support_axis_cm1.shape, (1999,))
        self.assertEqual(summary.record_count, 7680)
        self.assertEqual(summary.prediction_row_count, 7680 * 41)

    def test_pls2_selection_prefers_smallest_exact_tie_without_refit(self) -> None:
        subject = _load_subject()

        inputs = subject.make_synthetic_d4_protocol_a_inputs(
            well_count=10,
            acquisitions_per_well=2,
            model_folds=5,
            validation_macro_nrmse_by_component={
                2: 0.125,
                4: 0.125,
                8: 0.25,
                16: 0.5,
                32: 0.75,
            },
        )
        config = subject.make_synthetic_d4_protocol_a_config(inputs)
        models = subject.fit_d4_protocol_a_models(inputs, config)
        self.assertEqual(len(models), 5)
        for model in models:
            self.assertEqual(
                tuple(row["n_components"] for row in model.validation_scores),
                (2, 4, 8, 16, 32),
            )
            self.assertEqual(model.selected_n_components, 2)
            self.assertFalse(model.refit_with_validation)
            self.assertEqual(model.condition_count, 41)

    def test_hand_derived_regression_metrics_well_loss_and_exact_cv_stitching(self) -> None:
        subject = _load_subject()

        inputs = subject.make_synthetic_d4_protocol_a_inputs(
            well_count=5,
            acquisitions_per_well=2,
            model_folds=5,
            prediction_fixture="hand_derived_regression",
        )
        config = subject.make_synthetic_d4_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            built = subject.build_phase4_d4_protocol_a_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            predictions = [
                json.loads(line)
                for line in (built.path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            stitched = {(row["record_id"], row["condition_id"]) for row in predictions}
            self.assertEqual(len(stitched), len(predictions))
            alpha0_predictions = [row for row in predictions if row["condition_id"] == "alpha0"]
            self.assertEqual(len(alpha0_predictions), len(inputs.record_ids))

            summary_rows = list(
                csv.DictReader((built.path / "condition_summary.csv").read_text(encoding="utf-8").splitlines())
            )
            alpha0 = next(row for row in summary_rows if row["condition_id"] == "alpha0")
            p08_005 = next(
                row for row in summary_rows if row["condition_id"] == "p08:9a9999999999a93f"
            )
            self.assertAlmostEqual(float(alpha0["macro_normalized_rmse"]), 0.09375)
            self.assertAlmostEqual(float(alpha0["macro_mae_mol_l"]), 0.03)
            self.assertAlmostEqual(float(p08_005["macro_normalized_rmse"]), 0.15625)
            self.assertAlmostEqual(float(p08_005["macro_mae_mol_l"]), 0.05)

            well_rows = [
                json.loads(line)
                for line in (built.path / "well_conditions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            alpha0_well = next(
                row for row in well_rows if row["well_id"] == "well-000" and row["condition_id"] == "alpha0"
            )
            p08_well = next(
                row
                for row in well_rows
                if row["well_id"] == "well-000" and row["condition_id"] == "p08:9a9999999999a93f"
            )
            self.assertAlmostEqual(alpha0_well["downstream_harm"], 0.0)
            self.assertAlmostEqual(alpha0_well["loss"], 0.0146484375)
            self.assertAlmostEqual(p08_well["loss"], 0.0390625)
            self.assertAlmostEqual(
                p08_well["downstream_harm"],
                p08_well["loss"] - alpha0_well["loss"],
            )

    def test_blank_lod_loq_formulas_and_auxiliary_failure_isolation(self) -> None:
        subject = _load_subject()

        inputs = subject.make_synthetic_d4_protocol_a_inputs(
            well_count=5,
            acquisitions_per_well=2,
            model_folds=5,
            blank_fixture="isolated_auxiliary_failure",
        )
        config = subject.make_synthetic_d4_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            built = subject.build_phase4_d4_protocol_a_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            lod_rows = [
                json.loads(line)
                for line in (built.path / "technical_lod_loq.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            complete = next(
                row
                for row in lod_rows
                if row["fold"] == 0 and row["condition_id"] == "alpha0" and row["analyte_index"] == 0
            )
            self.assertAlmostEqual(complete["sigma"], 0.02)
            self.assertAlmostEqual(complete["slope"], 0.4)
            self.assertAlmostEqual(complete["ich_lod"], 0.165)
            self.assertAlmostEqual(complete["ich_loq"], 0.5)
            self.assertAlmostEqual(complete["iupac_lod"], 0.15)

            auxiliary_failure = next(
                row
                for row in lod_rows
                if row["fold"] == 1 and row["condition_id"] == "p08:9a9999999999a93f" and row["analyte_index"] == 0
            )
            self.assertEqual(auxiliary_failure["state"], "not_evaluable_nonpositive_slope")
            self.assertIsNone(auxiliary_failure["ich_lod"])
            self.assertIsNone(auxiliary_failure["ich_loq"])

            complete_marker = built.path / "complete.json"
            self.assertTrue(complete_marker.exists())
            endpoint = json.loads(complete_marker.read_text(encoding="utf-8"))
            self.assertEqual(endpoint["status"], "complete")

    def test_small_synthetic_build_has_expected_row_contracts_and_fixed_holm_slots(self) -> None:
        subject = _load_subject()

        inputs = subject.make_synthetic_d4_protocol_a_inputs(
            well_count=5,
            acquisitions_per_well=2,
            model_folds=5,
        )
        config = subject.make_synthetic_d4_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            built = subject.build_phase4_d4_protocol_a_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            record_measurements = (built.path / "record_measurements.jsonl").read_text(encoding="utf-8").splitlines()
            predictions = (built.path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            well_conditions = (built.path / "well_conditions.jsonl").read_text(encoding="utf-8").splitlines()
            well_observations = (built.path / "well_observations.jsonl").read_text(encoding="utf-8").splitlines()
            blanks = (built.path / "blank_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            lod = (built.path / "technical_lod_loq.jsonl").read_text(encoding="utf-8").splitlines()
            holm = (built.path / "holm_family.jsonl").read_text(encoding="utf-8").splitlines()
            figure1_rows = (built.path / "figure1_d4_protocol_a_full_domain_data.csv").read_text(
                encoding="utf-8"
            ).splitlines()
            figure2_rows = (built.path / "figure2_d4_protocol_a_full_domain_data.csv").read_text(
                encoding="utf-8"
            ).splitlines()
            table_rows = (built.path / "d4_protocol_a_full_domain_secondary_table.csv").read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(len(record_measurements), len(inputs.record_ids) * 41)
            self.assertEqual(len(predictions), len(inputs.record_ids) * 41)
            self.assertEqual(len(well_conditions), len(inputs.well_ids) * 41)
            self.assertEqual(len(well_observations), 13 * len(inputs.well_ids) * 40)
            self.assertEqual(len(blanks), 5 * len(inputs.blank_record_ids) * 41)
            self.assertEqual(len(lod), 5 * 41 * 4)
            self.assertEqual(len(holm), 24)
            self.assertEqual(len(figure1_rows) - 1, 13 * 5 * 8)
            self.assertEqual(len(figure2_rows) - 1, 13)
            self.assertEqual(len(table_rows) - 1, 13)

    def test_output_bytes_do_not_depend_on_worker_count(self) -> None:
        subject = _load_subject()

        inputs = subject.make_synthetic_d4_protocol_a_inputs(
            well_count=5,
            acquisitions_per_well=2,
            model_folds=5,
        )
        config = subject.make_synthetic_d4_protocol_a_config(inputs)
        with tempfile.TemporaryDirectory() as directory:
            left_root = Path(directory) / "left"
            right_root = Path(directory) / "right"
            left = subject.build_phase4_d4_protocol_a_from_inputs(
                left_root,
                inputs=inputs,
                config=config,
                worker_count=1,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            right = subject.build_phase4_d4_protocol_a_from_inputs(
                right_root,
                inputs=inputs,
                config=config,
                worker_count=2,
                bootstrap_resamples=8,
                sign_flip_resamples=16,
            )
            self.assertEqual(left.run_id, right.run_id)
            self.assertEqual(
                (left.path / "SHA256SUMS").read_bytes(),
                (right.path / "SHA256SUMS").read_bytes(),
            )
            manifest = json.loads((left.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("worker_count", json.dumps(manifest, sort_keys=True))
            self.assertNotIn(str(left.path.parent), json.dumps(manifest, sort_keys=True))

    def test_verifier_stays_independent_and_cli_rejects_override_flags(self) -> None:
        verifier_path = ROOT / "rpe/runner/phase4_d4_protocol_a_verifier.py"
        cli_path = ROOT / "tools/run_phase4_d4_protocol_a.py"
        if not verifier_path.exists() or not cli_path.exists():
            self.skipTest("awaiting Step 29 verifier/CLI files")

        tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
        forbidden = "rpe.runner.phase4_d4_protocol_a"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(forbidden, {alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, forbidden)

        for arguments in (
            ["build", "--output-root", "x", "--protocol", "A"],
            ["build", "--output-root", "x", "--bootstrap-resamples", "8"],
            ["verify", "--run-path", "x", "--sign-flip-resamples", "16"],
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
