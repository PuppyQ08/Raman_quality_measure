from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import ast
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.cross_decomposition import PLSRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import ReplicatePairInput, SingleSpectrumInput  # noqa: E402
from rpe.methods import TaskLine, load_classical_catalog  # noqa: E402
from rpe.methods.classical.denoising import (  # noqa: E402
    run_stateless_denoising_system,
)
from rpe.metrics.consistency import HalfSplitPearsonConsistencyMetric  # noqa: E402
from rpe.metrics.reference_free import ISLikeStructureToNoiseMetric  # noqa: E402
import rpe.runner.phase6_denoising_evidence as evidence_module  # noqa: E402
import rpe.runner.phase6_denoising_evidence_verifier as verifier_module  # noqa: E402
from rpe.runner.phase6_denoising_evidence import (  # noqa: E402
    DenoisingEvidenceError,
    build_phase6_denoising_evidence_from_inputs,
    load_phase6_denoising_evidence_config,
    make_synthetic_denoising_evidence_inputs,
)
from rpe.runner.phase6_denoising_evidence_verifier import (  # noqa: E402
    verify_phase6_denoising_evidence,
)
import tools.run_phase6_denoising_evidence as cli_module  # noqa: E402


CONFIG = ROOT / "experiments" / "phase6" / "configs" / "denoising_evidence_v1.json"
CATALOG = ROOT / "experiments" / "phase3" / "configs" / "classical_system_catalog_v1.json"


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_bytes().splitlines(keepends=True):
        value = json.loads(line)
        if line != canonical(value):
            raise AssertionError(f"{path.name} is not canonical JSONL")
        rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rewrite_sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(current for current in root.iterdir() if current.is_file() and current.name != "SHA256SUMS"):
        lines.append(f"{sha256_bytes(path.read_bytes())}  {path.name}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def select_fixture_systems():
    catalog = load_classical_catalog(CATALOG)
    stateless = next(
        system
        for system in catalog.systems
        if system.task_line is TaskLine.DENOISING
        and system.system_id == "24e29d6a4baa67e8a1477af40f774d053fa215cc47b52f7f625f344a8c69137a"
    )
    fitted = next(
        system
        for system in catalog.systems
        if system.task_line is TaskLine.DENOISING
        and system.system_id == "6873cc652ebe9f0b95644784cef860c0c10eda5aec3121b7f91108283e783371"
    )
    return (stateless, fitted)


def independent_protocol_a_effect(
    *,
    inputs,
    system,
    well_id: str,
) -> float:
    fold = inputs.fold_by_well[well_id]
    train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
    validation = np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)
    test = np.asarray(inputs.test_indices_by_fold[fold], dtype=np.int64)
    x_train = np.asarray(inputs.native_matrix[train, 1:], dtype=np.float64)
    x_validation = np.asarray(inputs.native_matrix[validation, 1:], dtype=np.float64)
    y_train = np.asarray(inputs.targets[train], dtype=np.float64)
    y_validation = np.asarray(inputs.targets[validation], dtype=np.float64)
    best_score = None
    best_model = None
    for n_components in (2, 4, 8, 16, 32):
        if n_components > min(x_train.shape[0] - 1, x_train.shape[1]):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                model = PLSRegression(
                    n_components=n_components,
                    scale=True,
                    max_iter=500,
                    tol=1e-6,
                    copy=True,
                )
                model.fit(x_train, y_train)
                predicted = np.asarray(model.predict(x_validation), dtype=np.float64)
        except (Warning, ValueError, ArithmeticError, FloatingPointError, StopIteration):
            continue
        score = float(
            np.mean(
                np.sqrt(np.mean((predicted - y_validation) ** 2, axis=0)) / 0.32
            )
        )
        key = (score, n_components)
        if best_score is None or key < best_score:
            best_score = key
            best_model = model
    if best_model is None:
        raise AssertionError("fixture failed to select an identity model")
    well_indexes = [
        index
        for index in test.tolist()
        if inputs.well_ids[index] == well_id
    ]
    identity = np.asarray(best_model.predict(inputs.native_matrix[well_indexes, 1:]), dtype=np.float64)
    candidate_rows = []
    for index in well_indexes:
        result = run_stateless_denoising_system(system, inputs.native_spectra[index])
        if result.denoised_intensity is None:
            raise AssertionError("fixture stateless transform unexpectedly failed")
        candidate_rows.append(np.asarray(result.denoised_intensity[1:], dtype=np.float64))
    candidate = np.asarray(best_model.predict(np.asarray(candidate_rows, dtype=np.float64)), dtype=np.float64)
    targets = np.asarray(inputs.targets[well_indexes], dtype=np.float64)
    identity_loss = float(np.mean(((identity - targets) ** 2) / (0.32 ** 2)))
    candidate_loss = float(np.mean(((candidate - targets) ** 2) / (0.32 ** 2)))
    return candidate_loss - identity_loss


def independent_half_split_effect(
    *,
    inputs,
    system,
    well_id: str,
) -> float:
    metric = HalfSplitPearsonConsistencyMetric()
    indexes = [
        index
        for index, current in enumerate(inputs.well_ids)
        if current == well_id
    ]
    ordered = sorted(
        indexes,
        key=lambda index: (
            int(inputs.rounds[index]),
            int(inputs.repetitions[index]),
            str(inputs.record_ids[index]),
        ),
    )
    even = ordered[0::2]
    odd = ordered[1::2]
    identity_even = np.mean(inputs.native_matrix[even], axis=0)
    identity_odd = np.mean(inputs.native_matrix[odd], axis=0)
    candidate_rows = []
    for index in ordered:
        result = run_stateless_denoising_system(system, inputs.native_spectra[index])
        if result.denoised_intensity is None:
            raise AssertionError("fixture stateless transform unexpectedly failed")
        candidate_rows.append(np.asarray(result.denoised_intensity, dtype=np.float64))
    candidate_rows = np.asarray(candidate_rows, dtype=np.float64)
    candidate_even = np.mean(candidate_rows[0::2], axis=0)
    candidate_odd = np.mean(candidate_rows[1::2], axis=0)
    identity_value = metric.evaluate(
        ReplicatePairInput(
            evidence_module._mean_spectrum(
                "identity-even",
                inputs.axis_cm1,
                identity_even,
            ),
            evidence_module._mean_spectrum(
                "identity-odd",
                inputs.axis_cm1,
                identity_odd,
            ),
        )
    ).outputs[0].value
    candidate_value = metric.evaluate(
        ReplicatePairInput(
            evidence_module._mean_spectrum(
                "candidate-even",
                inputs.axis_cm1,
                candidate_even,
            ),
            evidence_module._mean_spectrum(
                "candidate-odd",
                inputs.axis_cm1,
                candidate_odd,
            ),
        )
    ).outputs[0].value
    return float(candidate_value) - float(identity_value)


class Phase6DenoisingEvidenceConfigTest(unittest.TestCase):
    def test_config_is_canonical_and_binds_frozen_contract(self) -> None:
        raw = CONFIG.read_bytes()
        document = json.loads(raw)
        self.assertEqual(raw, canonical(document))
        config = load_phase6_denoising_evidence_config(CONFIG)
        self.assertEqual(config.schema_version, "phase6-denoising-evidence-v1")
        self.assertEqual(len(config.system_ids), 60)
        self.assertEqual(config.expected["system_count"], 60)
        self.assertEqual(config.expected["stateless_system_count"], 36)
        self.assertEqual(config.expected["fitted_system_count"], 24)
        self.assertEqual(config.expected["method_evidence_row_count"], 360)
        self.assertEqual(config.expected["downstream_well_row_count"], 28800)
        self.assertEqual(config.expected["reference_free_well_row_count"], 14400)
        self.assertEqual(config.expected["half_split_well_row_count"], 14400)
        self.assertEqual(config.expected["bootstrap_row_count"], 240)
        self.assertEqual(config.expected["fit_receipt_count"], 120)
        self.assertEqual(config.expected["transform_receipt_count"], 1198080)
        self.assertEqual(config.expected["model_receipt_count"], 300)
        self.assertEqual(config.expected["system_status_row_count"], 60)
        self.assertEqual(config.expected["family_projection_row_count"], 30)
        self.assertEqual(
            config.authorities["denoising_promotion"]["sha256"],
            "6837091f1e26bfd4f88cd4b4727d6cf8f06803d44fe53e7ba364bcc27d5edc01",
        )
        self.assertEqual(
            config.authorities["phase3_denoising_report"]["sha256"],
            "939a3adbd3610bcb0aa23e9cd43f90ff76d60600a866f6186dd800e5b4846617",
        )
        self.assertEqual(
            tuple(config.artifact_contract["payload_files"]),
            (
                "config.json",
                "authority_bridge.json",
                "preflight.json",
                "fit_receipts.jsonl",
                "transform_receipts.jsonl",
                "model_receipts.jsonl",
                "system_status.jsonl",
                "downstream_well_rows.jsonl",
                "reference_free_well_rows.jsonl",
                "half_split_well_rows.jsonl",
                "bootstrap_results.jsonl",
                "method_evidence_rows.csv",
                "family_projection.csv",
                "manifest.json",
            ),
        )


class Phase6DenoisingEvidenceArtifactTest(unittest.TestCase):
    def test_synthetic_fixture_build_has_expected_row_arithmetic_and_inventory(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_denoising_evidence_from_inputs(
                inputs,
                systems,
                Path(output_root),
                config=config,
                worker_count=1,
                project_root=ROOT,
            )
            self.assertEqual(summary.method_evidence_row_count, len(systems) * 6)
            self.assertEqual(summary.downstream_well_row_count, len(systems) * 2 * inputs.well_count)
            self.assertEqual(summary.reference_free_well_row_count, len(systems) * inputs.well_count)
            self.assertEqual(summary.half_split_well_row_count, len(systems) * inputs.well_count)
            self.assertEqual(summary.bootstrap_row_count, len(systems) * 4)
            files = {
                path.relative_to(summary.path).as_posix()
                for path in summary.path.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                files,
                {
                    "SHA256SUMS",
                    "authority_bridge.json",
                    "bootstrap_results.jsonl",
                    "complete.json",
                    "config.json",
                    "downstream_well_rows.jsonl",
                    "family_projection.csv",
                    "fit_receipts.jsonl",
                    "half_split_well_rows.jsonl",
                    "manifest.json",
                    "method_evidence_rows.csv",
                    "model_receipts.jsonl",
                    "preflight.json",
                    "reference_free_well_rows.jsonl",
                    "system_status.jsonl",
                    "transform_receipts.jsonl",
                },
            )
            preflight = read_json(summary.path / "preflight.json")
            identity = preflight["identity_equivalence"]
            self.assertEqual(identity["status"], "passed")
            self.assertEqual(
                identity["protocol_a_prediction_sha256"],
                identity["protocol_b_prediction_sha256"],
            )
            self.assertEqual(
                identity["protocol_a_model_digest_sha256"],
                identity["protocol_b_model_digest_sha256"],
            )
            method_rows = read_csv(summary.path / "method_evidence_rows.csv")
            self.assertEqual(len(method_rows), len(systems) * 6)
            direct_gt_rows = [
                row
                for row in method_rows
                if row["endpoint_id"] == "direct_gt"
            ]
            self.assertEqual(len(direct_gt_rows), len(systems))
            self.assertTrue(
                all(
                    row["state"] == "not_evaluated_no_defensible_clean_target"
                    for row in direct_gt_rows
                )
            )

    def test_independent_recomputation_matches_stateless_downstream_and_half_split_rows(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        stateless, fitted = select_fixture_systems()
        self.assertIsNotNone(fitted)
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_denoising_evidence_from_inputs(
                inputs,
                (stateless, fitted),
                Path(output_root),
                config=config,
                worker_count=1,
                project_root=ROOT,
            )
            first_well = inputs.unique_well_ids[0]
            downstream_rows = read_jsonl(summary.path / "downstream_well_rows.jsonl")
            half_split_rows = read_jsonl(summary.path / "half_split_well_rows.jsonl")
            downstream_row = next(
                row
                for row in downstream_rows
                if row["system_id"] == stateless.system_id
                and row["protocol_id"] == "D4-denoise-A"
                and row["well_id"] == first_well
            )
            expected_downstream = independent_protocol_a_effect(
                inputs=inputs,
                system=stateless,
                well_id=first_well,
            )
            self.assertAlmostEqual(
                float(downstream_row["effect"]),
                expected_downstream,
                places=12,
            )
            half_split_row = next(
                row
                for row in half_split_rows
                if row["system_id"] == stateless.system_id
                and row["well_id"] == first_well
            )
            expected_half_split = independent_half_split_effect(
                inputs=inputs,
                system=stateless,
                well_id=first_well,
            )
            self.assertAlmostEqual(
                float(half_split_row["effect"]),
                expected_half_split,
                places=12,
            )

    def test_fit_receipts_bind_train_only_roles_and_verifier_accepts(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_denoising_evidence_from_inputs(
                inputs,
                systems,
                Path(output_root),
                config=config,
                worker_count=2,
                project_root=ROOT,
            )
            fit_rows = read_jsonl(summary.path / "fit_receipts.jsonl")
            self.assertEqual(len(fit_rows), 5)
            for row in fit_rows:
                train_wells = set(row["train_well_ids"])
                validation_wells = set(row["validation_well_ids"])
                test_wells = set(row["test_well_ids"])
                self.assertFalse(train_wells & validation_wells)
                self.assertFalse(train_wells & test_wells)
                self.assertFalse(validation_wells & test_wells)
                self.assertEqual(row["status"], "complete")
                self.assertTrue(row["fitted_state_sha256"])
            verified = verify_phase6_denoising_evidence(
                summary.path,
                project_root=ROOT,
            )
            self.assertEqual(verified.run_id, summary.run_id)
            self.assertEqual(verified.path, summary.path)

    def test_protocol_a_reuses_frozen_identity_estimator_without_refit(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        stateless, _fitted = select_fixture_systems()

        class FrozenEstimator:
            def __init__(self) -> None:
                self.predict_calls = 0

            def fit(self, *_args, **_kwargs):
                raise AssertionError("protocol A must not refit the frozen identity estimator")

            def predict(self, value):
                array = np.asarray(value, dtype=np.float64)
                self.predict_calls += 1
                return np.zeros((array.shape[0], inputs.targets.shape[1]), dtype=np.float64)

        frozen = FrozenEstimator()

        def fake_identity_models(_inputs, _config):
            zero_prediction = np.zeros(inputs.targets.shape[1], dtype=np.float64)
            models = {}
            predictions = {}
            for fold in inputs.fold_ids:
                models[int(fold)] = {
                    "fold": int(fold),
                    "selected_n_components": 2,
                    "model_state_digest": f"digest-{fold}",
                    "validation_scores": (),
                    "estimator": frozen,
                }
                for index in inputs.test_indices_by_fold[fold].tolist():
                    predictions[inputs.record_ids[index]] = zero_prediction
            return (
                models,
                predictions,
                {well_id: 0.0 for well_id in inputs.unique_well_ids},
                {well_id: 0.0 for well_id in inputs.unique_well_ids},
                {well_id: 0.0 for well_id in inputs.unique_well_ids},
                {"prediction_sha256": "synthetic"},
            )

        with tempfile.TemporaryDirectory() as output_root:
            with mock.patch.object(evidence_module, "_identity_models", side_effect=fake_identity_models):
                summary = build_phase6_denoising_evidence_from_inputs(
                    inputs,
                    (stateless,),
                    Path(output_root),
                    config=config,
                    worker_count=1,
                    project_root=ROOT,
                )
                self.assertTrue((summary.path / "downstream_well_rows.jsonl").is_file())
        self.assertGreater(frozen.predict_calls, 0)

    def test_failure_isolation_keeps_fixed_row_counts_when_consumer_raises(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        stateless, fitted = select_fixture_systems()
        original_reference = evidence_module._reference_free_well_value
        call_count = {"value": 0}

        def failing_candidate_reference(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] > inputs.well_count:
                raise RuntimeError("synthetic candidate reference-free consumer failure")
            return original_reference(*args, **kwargs)

        with tempfile.TemporaryDirectory() as output_root:
            with mock.patch.object(evidence_module, "_reference_free_well_value", side_effect=failing_candidate_reference):
                summary = build_phase6_denoising_evidence_from_inputs(
                    inputs,
                    (stateless, fitted),
                    Path(output_root),
                    config=config,
                    worker_count=1,
                    project_root=ROOT,
                )
            system_rows = read_jsonl(summary.path / "system_status.jsonl")
            failed_row = next(row for row in system_rows if row["system_id"] == stateless.system_id)
            self.assertIn("consumer", str(failed_row["reason_code"]))
            self.assertEqual(len(read_jsonl(summary.path / "downstream_well_rows.jsonl")), 2 * 2 * inputs.well_count)
            self.assertEqual(len(read_jsonl(summary.path / "reference_free_well_rows.jsonl")), 2 * inputs.well_count)
            failed_reference_rows = [
                row
                for row in read_jsonl(summary.path / "reference_free_well_rows.jsonl")
                if row["system_id"] == stateless.system_id
            ]
            self.assertTrue(all(row["state"] == "not_evaluable_incomplete_system_grid" for row in failed_reference_rows))

    def test_identity_preflight_fails_when_any_grid_candidate_is_not_complete(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        original_fit = evidence_module._fit_selected_pls
        calls = {"value": 0}

        def grid_failure(*args, **kwargs):
            calls["value"] += 1
            estimator, _rows, selected, digest = original_fit(*args, **kwargs)
            if calls["value"] == 1:
                return (
                    estimator,
                    (
                        {
                            "fold": 0,
                            "n_components": 2,
                            "macro_normalized_rmse": None,
                            "state": "failed_model_lifecycle",
                            "failure_category": "RuntimeWarning",
                            "failure_message": "synthetic warning",
                        },
                    ),
                    selected,
                    digest,
                )
            return estimator, _rows, selected, digest

        with tempfile.TemporaryDirectory() as output_root:
            with mock.patch.object(evidence_module, "_fit_selected_pls", side_effect=grid_failure):
                with self.assertRaisesRegex(Exception, "grid|identity|candidate|preflight"):
                    build_phase6_denoising_evidence_from_inputs(
                        inputs,
                        systems,
                        Path(output_root),
                        config=config,
                        worker_count=1,
                        project_root=ROOT,
                    )

    def test_protocol_b_reason_closes_when_candidate_grid_is_not_complete(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        stateless, fitted = select_fixture_systems()
        original_fit = evidence_module._fit_selected_pls
        calls = {"value": 0}

        def candidate_b_grid_failure(*args, **kwargs):
            calls["value"] += 1
            if calls["value"] <= len(inputs.fold_ids):
                return original_fit(*args, **kwargs)
            raise evidence_module.DenoisingEvidenceError("protocol_b_grid", "synthetic candidate-grid failure")

        with tempfile.TemporaryDirectory() as output_root:
            with mock.patch.object(evidence_module, "_fit_selected_pls", side_effect=candidate_b_grid_failure):
                summary = build_phase6_denoising_evidence_from_inputs(
                    inputs,
                    (stateless, fitted),
                    Path(output_root),
                    config=config,
                    worker_count=1,
                    project_root=ROOT,
                )
            downstream_rows = read_jsonl(summary.path / "downstream_well_rows.jsonl")
            protocol_b_rows = [
                row
                for row in downstream_rows
                if row["system_id"] == stateless.system_id and row["protocol_id"] == "D4-denoise-B"
            ]
            self.assertTrue(all(row["effect"] is None for row in protocol_b_rows))
            self.assertTrue(all("protocol_b" in row["state"] for row in protocol_b_rows))
            protocol_a_rows = [
                row
                for row in downstream_rows
                if row["system_id"] == stateless.system_id and row["protocol_id"] == "D4-denoise-A"
            ]
            self.assertTrue(all(row["state"] == "complete" for row in protocol_a_rows))

    def test_worker_count_is_consumed_by_system_executor(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        test_case = self
        seen: list[int] = []
        mapped: list[tuple[int, ...]] = []

        class RecordingExecutor:
            def __init__(self, max_workers=None, *args, **kwargs) -> None:
                seen.append(int(max_workers))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def map(self, func, iterable, chunksize=None, **kwargs):
                test_case.assertEqual(chunksize, 1)
                test_case.assertEqual(kwargs, {})
                items = tuple(int(item) for item in iterable)
                mapped.append(items)
                return [func(item) for item in items]

        with tempfile.TemporaryDirectory() as output_root:
            with mock.patch.object(evidence_module, "ProcessPoolExecutor", RecordingExecutor):
                build_phase6_denoising_evidence_from_inputs(
                    inputs,
                    systems,
                    Path(output_root),
                    config=config,
                    worker_count=3,
                    project_root=ROOT,
                )
        self.assertEqual(seen, [3])
        self.assertEqual(mapped, [tuple(range(len(systems)))])

    def test_verifier_reexecutes_without_delegating_to_builder(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_denoising_evidence_from_inputs(
                inputs,
                systems,
                Path(output_root),
                config=config,
                worker_count=1,
                project_root=ROOT,
            )
            with mock.patch.object(
                verifier_module,
                "build_phase6_denoising_evidence_from_inputs",
                create=True,
                side_effect=AssertionError("verifier must not delegate through the production builder"),
            ):
                verified = verify_phase6_denoising_evidence(
                    summary.path,
                    project_root=ROOT,
                )
            self.assertEqual(verified.run_id, summary.run_id)

    def test_verifier_module_does_not_import_production_runner(self) -> None:
        tree = ast.parse((ROOT / "rpe/runner/phase6_denoising_evidence_verifier.py").read_text(encoding="utf-8"))
        banned = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                banned.extend(alias.name for alias in node.names if alias.name == "rpe.runner.phase6_denoising_evidence")
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase6_denoising_evidence":
                banned.append(node.module)
        self.assertEqual(banned, [])

    def test_verifier_rejects_real_artifact_with_shrunk_system_status_list(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_denoising_evidence_from_inputs(
                inputs,
                systems,
                Path(output_root),
                config=config,
                worker_count=1,
                project_root=ROOT,
            )
            manifest = read_json(summary.path / "manifest.json")
            manifest["synthetic_fixture"] = False
            (summary.path / "manifest.json").write_bytes(canonical(manifest))
            system_rows = read_jsonl(summary.path / "system_status.jsonl")
            (summary.path / "system_status.jsonl").write_bytes(canonical(system_rows[0]))
            rewrite_sha256sums(summary.path)
            with mock.patch.object(verifier_module, "_load_real_inputs", return_value=inputs):
                with self.assertRaisesRegex(Exception, "system_status|system IDs|catalog|config"):
                    verify_phase6_denoising_evidence(
                        summary.path,
                        project_root=ROOT,
                    )

    def test_worker_count_does_not_change_bytes(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = build_phase6_denoising_evidence_from_inputs(
                inputs,
                systems,
                Path(first_root),
                config=config,
                worker_count=1,
                project_root=ROOT,
            )
            second = build_phase6_denoising_evidence_from_inputs(
                inputs,
                systems,
                Path(second_root),
                config=config,
                worker_count=4,
                project_root=ROOT,
            )
            self.assertEqual(first.run_id, second.run_id)
            first_files = {
                path.relative_to(first.path).as_posix(): path.read_bytes()
                for path in first.path.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second.path).as_posix(): path.read_bytes()
                for path in second.path.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)

    def test_verifier_rejects_semantic_tamper_even_if_checksums_are_rewritten(self) -> None:
        config = load_phase6_denoising_evidence_config(CONFIG)
        inputs = make_synthetic_denoising_evidence_inputs()
        systems = select_fixture_systems()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_denoising_evidence_from_inputs(
                inputs,
                systems,
                Path(output_root),
                config=config,
                worker_count=1,
                project_root=ROOT,
            )
            rows = read_jsonl(summary.path / "downstream_well_rows.jsonl")
            rows[0]["effect"] = float(rows[0]["effect"]) + 0.5
            payload = b"".join(canonical(row) for row in rows)
            (summary.path / "downstream_well_rows.jsonl").write_bytes(payload)
            rewrite_sha256sums(summary.path)
            with self.assertRaisesRegex(Exception, "mismatch|tamper|unexpected|byte"):
                verify_phase6_denoising_evidence(
                    summary.path,
                    project_root=ROOT,
                )


class Phase6DenoisingEvidenceCliTest(unittest.TestCase):
    def test_cli_script_help_bootstraps_repo_root_for_direct_execution(self) -> None:
        result = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "tools/run_phase6_denoising_evidence.py",
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Build or independently verify Phase 6 denoising evidence", result.stdout)
        self.assertIn("build", result.stdout)
        self.assertIn("verify", result.stdout)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_cli_rejects_scientific_overrides(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli_module.main(
                    [
                        "build",
                        "--output-root",
                        "results/phase6/denoising_evidence_v1",
                        "--system-id",
                        "24e29d6a4baa67e8a1477af40f774d053fa215cc47b52f7f625f344a8c69137a",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_cli_dispatches_build_and_verify(self) -> None:
        build_summary = mock.Mock(path=Path("/tmp/build"), run_id="run-build")
        verify_summary = mock.Mock(path=Path("/tmp/build"), run_id="run-build")
        stdout = io.StringIO()
        with mock.patch.object(cli_module, "build_phase6_denoising_evidence", return_value=build_summary) as build_mock:
            with contextlib.redirect_stdout(stdout):
                cli_module.main(
                    [
                        "build",
                        "--output-root",
                        "results/phase6/denoising_evidence_v1",
                        "--worker-count",
                        "3",
                    ]
                )
        build_mock.assert_called_once()
        self.assertIn("run-build", stdout.getvalue())
        stdout = io.StringIO()
        with mock.patch.object(cli_module, "verify_phase6_denoising_evidence", return_value=verify_summary) as verify_mock:
            with contextlib.redirect_stdout(stdout):
                cli_module.main(
                    [
                        "verify",
                        "--run-path",
                        "/tmp/build",
                    ]
                )
        verify_mock.assert_called_once()
        self.assertIn("run-build", stdout.getvalue())
