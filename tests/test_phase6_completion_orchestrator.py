from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.complete_phase6_steps_7_11 as orchestrator  # noqa: E402


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def write_kv_status(path: Path, entries: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in entries.items()),
        encoding="utf-8",
    )


def rewrite_sha256sums(path: Path) -> None:
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.name != "SHA256SUMS":
            entries.append(f"{sha256_bytes(item.read_bytes())}  {item.relative_to(path).as_posix()}\n")
    write_text(path / "SHA256SUMS", "".join(entries))


def rewrite_sha256sums_in_order(path: Path, ordered_names: list[str]) -> None:
    write_text(
        path / "SHA256SUMS",
        "".join(
            f"{sha256_bytes((path / name).read_bytes())}  {name}\n"
            for name in ordered_names
        ),
    )


def write_artifact(path: Path, payloads: dict[str, bytes], *, extra_manifest: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for relative, payload in payloads.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    manifest = {
        "status": "complete",
        "payload_files": sorted(payloads),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    write_json(path / "manifest.json", manifest)
    write_json(path / "complete.json", {"status": "complete", "run_id": path.name})
    rewrite_sha256sums(path)


def csv_payload(header: str, rows: list[str]) -> bytes:
    return (header + "\n" + "\n".join(rows) + "\n").encode("utf-8")


STEP7_PAYLOADS = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "audit_status.csv",
    "resampling_alignment.csv",
    "resampling_rank_stability.csv",
    "normalization_alignment.csv",
    "normalization_rank_stability.csv",
    "peak_tolerance_phase4_alignment.csv",
    "peak_tolerance_phase4_rank_stability.csv",
    "peak_tolerance_phase6_system.csv",
    "peak_tolerance_phase6_rank_stability.csv",
    "leakage_boundaries.jsonl",
    "leakage_duplicate_candidates.jsonl",
    "summary.md",
    "manifest.json",
)


def make_step7_run(path: Path) -> None:
    payloads = {
        "config.json": canonical(
            {
                "schema_version": "phase6-appendix-audits-v1",
                "synthetic_fixture": True,
                "artifact_contract": {
                    "configured_payload_count": 16,
                    "payload_files": list(STEP7_PAYLOADS),
                    "terminal_markers": ["complete.json", "SHA256SUMS"],
                    "total_file_count": 18,
                },
            }
        ),
        "authority_bridge.json": canonical({"step6_protocol": "bound"}),
        "preflight.json": canonical({"status": "passed"}),
        "audit_status.csv": csv_payload(
            "component,state,reason_code",
            [
                "resampling,complete,none",
                "normalization,complete,none",
                "peak_tolerance,warning,rank_instability",
                "leakage,complete,none",
            ],
        ),
        "resampling_alignment.csv": csv_payload("panel_id,metric_id,ag,acc_cross", ["d5_a,mse,0.1,0.2", "d4_b,w1,0.3,0.4"]),
        "resampling_rank_stability.csv": csv_payload(
            "panel_id,statistic,tau_b,stable,state,reason_code",
            ["d5_a,ag,0.95,True,complete,none", "d4_b,acc_cross,0.88,False,warning,strict_tau_b"],
        ),
        "normalization_alignment.csv": csv_payload("panel_id,metric_id,ag,acc_cross", ["d5_a,mse,0.1,0.2"]),
        "normalization_rank_stability.csv": csv_payload(
            "panel_id,statistic,tau_b,stable,state,reason_code",
            ["d5_a,ag,1.0,True,complete,none"],
        ),
        "peak_tolerance_phase4_alignment.csv": csv_payload("panel_id,metric_id,ag,acc_cross", ["d5_a,peak_f1,0.5,0.6"]),
        "peak_tolerance_phase4_rank_stability.csv": csv_payload(
            "panel_id,statistic,tau_b,stable,state,reason_code",
            ["d5_a,ag,0.91,True,complete,none"],
        ),
        "peak_tolerance_phase6_system.csv": csv_payload(
            "system_id,endpoint_id,tolerance_cm1,estimate,state,reason_code",
            ["sys-a,endpoint-a,2,0.7,complete,none", "sys-b,endpoint-a,4,0.5,warning,flip"],
        ),
        "peak_tolerance_phase6_rank_stability.csv": csv_payload(
            "endpoint_id,tolerance_cm1,reference_tolerance_cm1,tau_b,stable,state,reason_code",
            ["endpoint-a,4,2,0.83,False,warning,flip"],
        ),
        "leakage_boundaries.jsonl": b"".join(
            canonical(
                {
                    "boundary_id": boundary_id,
                    "status": "complete" if boundary_id != "cross_dataset" else "not_evaluable_insufficient_cross_dataset_identity",
                    "reason_code": "none" if boundary_id != "cross_dataset" else "insufficient_cross_dataset_identity",
                }
            )
            for boundary_id in ("d1_roles", "d2_roles", "d4_roles", "d5_roles", "cross_dataset")
        ),
        "leakage_duplicate_candidates.jsonl": canonical(
            {
                "boundary_id": "d4_roles",
                "left_record_id": "left-1",
                "right_record_id": "right-1",
                "distance": 0.0004,
                "status": "warning",
                "reason_code": "near_duplicate",
            }
        ),
        "summary.md": (
            "# Phase 6 Appendix Audits\n\n"
            "Status: complete.\n"
        ).encode("utf-8"),
    }
    write_artifact(
        path,
        payloads,
        extra_manifest={
            "schema_version": "phase6-appendix-audits-artifact-v1",
            "counts": {
                "audit_status_rows": 4,
                "resampling_alignment_rows": 2,
                "resampling_rank_rows": 2,
                "normalization_alignment_rows": 1,
                "normalization_rank_rows": 1,
                "phase4_peak_alignment_rows": 1,
                "phase4_peak_rank_rows": 1,
                "phase6_peak_system_rows": 2,
                "phase6_peak_rank_rows": 1,
                "leakage_boundary_rows": 5,
                "leakage_duplicate_rows": 1,
            },
        },
    )


def make_phase4_final_figures(path: Path) -> None:
    write_artifact(
        path,
        {
            "figure1_phase4_response.png": b"figure1png\n",
            "figure1_phase4_response.svg": b"<svg>figure1</svg>\n",
            "figure2_phase4_alignment.png": b"figure2png\n",
            "figure2_phase4_alignment.svg": b"<svg>figure2</svg>\n",
        },
    )


STALE_CHECKLIST = """# Active Goal Completion Checklist

Status: active; not complete

- Figure3 / Figure4: RENDER PENDING
- Appendix audits: STEP-7 EXECUTION PENDING
- Release metadata / data card / Croissant: pending
- Paper manuscript: MISSING
- Submission-ready package: MISSING
"""


DEFERRED_STEP7_REPORT = """# Phase 6 Step 7 Appendix Audits

state: deferred_by_owner

- owner decision: defer appendix audits until external gates are resolved
- evidence state: not run and not used as evidence
"""


class FakeRunner:
    def __init__(
        self,
        project_root: Path,
        *,
        fail_label: str | None = None,
        discovery_summary_on_stderr: bool = False,
        verify_outputs_only_run_id: bool = False,
    ) -> None:
        self.project_root = project_root
        self.fail_label = fail_label
        self.discovery_summary_on_stderr = discovery_summary_on_stderr
        self.verify_outputs_only_run_id = verify_outputs_only_run_id
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        label: str,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(
            {
                "command": tuple(command),
                "cwd": str(cwd),
                "env": dict(env or {}),
                "label": label,
            }
        )
        if self.fail_label == label:
            return subprocess.CompletedProcess(command, 2, stdout="", stderr=f"{label} failed\n")
        stdout, stderr = self._dispatch(command, label)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=stderr)

    def _dispatch(self, command: list[str], label: str) -> tuple[str, str]:
        root = self.project_root
        joined = " ".join(command)
        if (
            command[1:] == ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
            or joined.endswith("-m unittest discover -s tests -p test_*.py -v")
        ):
            if self.discovery_summary_on_stderr:
                return (
                    "test_a (tests.fake) ... ok\n",
                    "\nRan 3 tests in 0.125s\n\nOK\n",
                )
            return ("test_a (tests.fake) ... ok\n\nRan 3 tests in 0.125s\n\nOK\n", "")
        if "freeze_phase6_publication_core_config.py" in joined:
            output = Path(command[command.index("--output") + 1])
            if not output.is_absolute():
                output = root / output
            write_json(
                output,
                {
                    "schema_version": "phase6-publication-core-config-v1",
                    "artifact_payload_files": [
                        "table1_metric_validity.csv",
                        "table1_metric_validity.md",
                        "table_s1_full_alignment.csv",
                        "table_s2_protocol_interactions.csv",
                        "table2_method_evidence.csv",
                        "table2_method_evidence.md",
                        "table_s3_system_status.csv",
                        "figure3_model_adequacy_data.csv",
                        "figure3_model_adequacy.png",
                        "figure3_model_adequacy.svg",
                        "figure4_gate_nodes.csv",
                        "figure4_gate_edges.csv",
                        "figure4_gate_map.png",
                        "figure4_gate_map.svg",
                    ],
                },
            )
            return (str(output.relative_to(root)) + "\n", "")
        if "run_phase6_publication_core.py" in joined and " build " in f" {joined} ":
            run_path = root / "results/phase6/publication_core_v1/phase6-publication-core-fixture"
            write_artifact(
                run_path,
                {
                    "config.json": canonical({"schema_version": "phase6-publication-core-config-v1"}),
                    "authority_bridge.json": canonical({"step7_run": "phase6-appendix-audits-fixture"}),
                    "preflight.json": canonical({"status": "passed"}),
                    "table1_metric_validity.csv": csv_payload(
                        "endpoint_id,state,jointly_favorable_metric_ids",
                        ["D5-A,complete,mse", "D1-B-closed,not_evaluable_failed_alpha0_equivalence,"],
                    ),
                    "table1_metric_validity.md": b"| endpoint_id | state |\n",
                    "table_s1_full_alignment.csv": csv_payload("row_id,state", ["1,complete", "2,complete"]),
                    "table_s2_protocol_interactions.csv": csv_payload("row_id,state", ["1,complete"]),
                    "table2_method_evidence.csv": csv_payload("method_id,state", ["mse,complete"]),
                    "table2_method_evidence.md": b"| method_id | state |\n",
                    "table_s3_system_status.csv": csv_payload("system_id,state", ["sys-a,complete", "sys-b,warning"]),
                    "figure3_model_adequacy_data.csv": csv_payload("extractor_id,gate_state", ["airpls,pass", "mor,fail_p95"]),
                    "figure3_model_adequacy.png": b"png3\n",
                    "figure3_model_adequacy.svg": b"<svg>3</svg>\n",
                    "figure4_gate_nodes.csv": csv_payload("node_id,state", ["n1,complete", "n2,pending"]),
                    "figure4_gate_edges.csv": csv_payload("edge_id,state", ["e1,complete"]),
                    "figure4_gate_map.png": b"png4\n",
                    "figure4_gate_map.svg": b"<svg>4</svg>\n",
                },
                extra_manifest={"schema_version": "phase6-publication-core-artifact-v1"},
            )
            return (f"{run_path.name}\t{run_path}\n", "")
        if "run_phase6_publication_core.py" in joined and " verify " in f" {joined} ":
            run_path = Path(command[command.index("--run-path") + 1])
            if self.verify_outputs_only_run_id:
                return (f"{run_path.name}\n", "")
            return (f"{run_path.name}\t{run_path}\n", "")
        if "run_phase6_release_metadata.py" in joined and " build " in f" {joined} ":
            run_path = root / "results/phase6/release_metadata_v1/phase6-release-metadata-fixture"
            write_artifact(
                run_path,
                {
                    "config.json": canonical({"schema_version": "phase6-release-metadata-v1"}),
                    "authority_bridge.json": canonical({"step8_run": "phase6-publication-core-fixture"}),
                    "preflight.json": canonical(
                        {
                            "status": "passed",
                            "croissant_validation": {"status": "passed", "validator": "mlcroissant.Dataset"},
                        }
                    ),
                    "release_matrix.csv": csv_payload(
                        "artifact_id,disposition,repository_license,source_license,reason",
                        [
                            "release_matrix_csv,public_release_candidate,pending_owner_license_selection,pending_owner_license_selection,metadata only",
                            "leaderboard,deferred_no_redistributable_hidden_gt,pending_owner_license_selection,not_applicable,no licensed hidden ground truth",
                        ],
                    ),
                    "data_card.md": (
                        "# Data Card\n\n"
                        "- Repository-wide license selection remains owner-only.\n"
                        "- No hosted leaderboard is released.\n"
                    ).encode("utf-8"),
                    "metadata_index.parquet": b"PAR1fixture\n",
                    "croissant.json": canonical({"conformsTo": "http://mlcommons.org/croissant/1.1"}),
                    "limitations.jsonl": (
                        canonical({"limitation_id": "pending_owner_license_selection", "state": "pending_owner_license_selection"})
                        + canonical({"limitation_id": "leaderboard_state", "state": "deferred_no_redistributable_hidden_gt"})
                    ),
                    "environment.json": canonical({"python": "3.13"}),
                },
                extra_manifest={"schema_version": "phase6-release-metadata-artifact-v1"},
            )
            return (f"{run_path.name}\t{run_path}\n", "")
        if "run_phase6_release_metadata.py" in joined and " verify " in f" {joined} ":
            run_path = Path(command[command.index("--run-path") + 1])
            if self.verify_outputs_only_run_id:
                return (f"{run_path.name}\n", "")
            return (f"{run_path.name}\t{run_path}\n", "")
        if "freeze_phase6_manuscript_config.py" in joined:
            output = root / "experiments/phase6/configs/manuscript_v1.json"
            write_json(
                output,
                {
                    "schema_version": "phase6-manuscript-v1",
                    "artifact_contract": {
                        "payload_files": [
                            "config.json",
                            "authority_bridge.json",
                            "preflight.json",
                            "paper/manuscript.md",
                            "paper/references.bib",
                            "paper/claims_matrix.csv",
                            "manifest.json",
                        ]
                    },
                },
            )
            return ("experiments/phase6/configs/manuscript_v1.json\n", "")
        if "run_phase6_manuscript.py" in joined and " build " in f" {joined} ":
            run_path = root / "results/phase6/manuscript_v1/phase6-manuscript-fixture"
            write_artifact(
                run_path,
                {
                    "config.json": canonical({"schema_version": "phase6-manuscript-v1"}),
                    "authority_bridge.json": canonical({"step8": "ok", "step9": "ok"}),
                    "preflight.json": canonical({"status": "passed"}),
                    "paper/manuscript.md": (
                        "# Fixture Manuscript\n\n"
                        "License and redistribution remain pending owner decisions.\n"
                        "Public leaderboard is deferred.\n"
                    ).encode("utf-8"),
                    "paper/references.bib": b"@article{fixture,\n  title={Fixture}\n}\n",
                    "paper/claims_matrix.csv": csv_payload(
                        "claim_id,verification_state,citation_keys",
                        ["claim-1,verified,fixture", "claim-2,verified,fixture"],
                    ),
                },
                extra_manifest={"schema_version": "phase6-manuscript-artifact-v1"},
            )
            return (f"{run_path.name}\t{run_path}\n", "")
        if "run_phase6_manuscript.py" in joined and " verify " in f" {joined} ":
            run_path = Path(command[command.index("--run-path") + 1])
            if self.verify_outputs_only_run_id:
                return (f"{run_path.name}\n", "")
            return (f"{run_path.name}\t{run_path}\n", "")
        if "freeze_phase6_submission_bundle_config.py" in joined:
            output = root / "experiments/phase6/configs/submission_bundle_v1.json"
            write_json(
                output,
                {
                    "schema_version": "phase6-submission-bundle-v1",
                    "artifact_contract": {
                        "payload_files": [
                            "config.json",
                            "authority_bridge.json",
                            "preflight.json",
                            "metadata/limitations_ledger.json",
                            "paper/manuscript.md",
                            "paper/references.bib",
                            "paper/claims_matrix.csv",
                            "assets/figures/figure1_phase4_response.png",
                            "assets/figures/figure1_phase4_response.svg",
                            "assets/figures/figure2_phase4_alignment.png",
                            "assets/figures/figure2_phase4_alignment.svg",
                            "assets/tables/table1_metric_validity.csv",
                            "manifest.json",
                        ]
                    },
                },
            )
            return (str(output) + "\n", "")
        if "run_phase6_submission_bundle.py" in joined and " build " in f" {joined} ":
            run_path = root / "results/phase6/submission_bundle_v1/phase6-submission-bundle-fixture"
            write_artifact(
                run_path,
                {
                    "config.json": canonical({"schema_version": "phase6-submission-bundle-v1"}),
                    "authority_bridge.json": canonical({"step10": "phase6-manuscript-fixture"}),
                    "preflight.json": canonical({"status": "passed"}),
                    "metadata/limitations_ledger.json": canonical(
                        {
                            "license_gate": "pending_owner_license_selection",
                            "redistribution_gate": "pending_owner_review_rruff_and_bacteria",
                            "renderer_state": "deferred_pending_locked_renderer_and_venue_choice",
                            "venue_state": "pending_owner_choice",
                            "submission_state": "not_submitted_pending_owner_authorization",
                            "leaderboard_state": "deferred_no_redistributable_hidden_gt",
                        }
                    ),
                    "paper/manuscript.md": (
                        "# Submission Bundle\n\n"
                        "License gate remains pending_owner_license_selection.\n"
                        "Leaderboard remains deferred_no_redistributable_hidden_gt.\n"
                    ).encode("utf-8"),
                    "paper/references.bib": b"@article{fixture,\n  title={Fixture}\n}\n",
                    "paper/claims_matrix.csv": csv_payload(
                        "claim_id,verification_state",
                        ["claim-1,verified", "claim-2,verified"],
                    ),
                    "assets/figures/figure1_phase4_response.png": b"bundle-fig1-png\n",
                    "assets/figures/figure1_phase4_response.svg": b"<svg>bundle-fig1</svg>\n",
                    "assets/figures/figure2_phase4_alignment.png": b"bundle-fig2-png\n",
                    "assets/figures/figure2_phase4_alignment.svg": b"<svg>bundle-fig2</svg>\n",
                    "assets/tables/table1_metric_validity.csv": csv_payload(
                        "endpoint_id,state",
                        ["D5-A,complete", "D1-B-closed,not_evaluable_failed_alpha0_equivalence"],
                    ),
                },
                extra_manifest={"schema_version": "phase6-submission-bundle-artifact-v1"},
            )
            return (f"{run_path.name}\t{run_path}\n", "")
        if "run_phase6_submission_bundle.py" in joined and " verify " in f" {joined} ":
            run_path = Path(command[command.index("--run-path") + 1])
            if self.verify_outputs_only_run_id:
                return (f"{run_path.name}\n", "")
            return (f"{run_path.name}\t{run_path}\n", "")
        raise AssertionError(f"Unhandled command: {joined}")


class Phase6CompletionOrchestratorTest(unittest.TestCase):
    def make_project(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, Path]:
        holder = tempfile.TemporaryDirectory()
        root = Path(holder.name)
        write_text(root / "reports/ACTIVE_GOAL_COMPLETION_CHECKLIST.md", STALE_CHECKLIST)
        write_text(root / "PROJECT_LOG.md", "# Project Log\n\nExisting body.\n")
        write_json(root / "experiments/phase6/configs/release_metadata_v1.json", {"schema_version": "phase6-release-metadata-v1"})
        step7_run = root / "results/phase6/appendix_audits_v1/phase6-appendix-audits-fixture"
        make_step7_run(step7_run)
        step4_final = root / "results/phase4/final_figures_v1/phase4-final-figures-fixture"
        make_phase4_final_figures(step4_final)
        build_status = root / ".tmp/input/step7_build_status.json"
        verify_status = root / ".tmp/input/step7_verify_status.json"
        write_json(
            build_status,
            {
                "status": "finished",
                "exit_code": 0,
                "wall_seconds": 10.25,
                "label": "step7_build",
                "worker_count": 8,
            },
        )
        write_json(
            verify_status,
            {
                "status": "finished",
                "exit_code": 0,
                "wall_seconds": 11.5,
                "label": "step7_verify",
                "worker_count": 6,
                "run_path": str(step7_run),
            },
        )
        return holder, root, step7_run, build_status, verify_status

    def run_main(self, root: Path, runner: FakeRunner, *extra: str) -> int:
        argv = [
            "--project-root",
            str(root),
            "--step7-run",
            str(root / "results/phase6/appendix_audits_v1/phase6-appendix-audits-fixture"),
            "--step7-build-status",
            str(root / ".tmp/input/step7_build_status.json"),
            "--step7-verify-status",
            str(root / ".tmp/input/step7_verify_status.json"),
            "--step4-final-figures",
            str(root / "results/phase4/final_figures_v1/phase4-final-figures-fixture"),
        ]
        argv.extend(extra)
        with mock.patch.object(orchestrator, "_run_process", side_effect=runner):
            return orchestrator.main(argv)

    def run_main_deferred(self, root: Path, runner: FakeRunner, *extra: str) -> int:
        argv = [
            "--project-root",
            str(root),
            "--step7-deferred-report",
            str(root / "reports/phase6/step07_appendix_audits.md"),
            "--step4-final-figures",
            str(root / "results/phase4/final_figures_v1/phase4-final-figures-fixture"),
        ]
        argv.extend(extra)
        with mock.patch.object(orchestrator, "_run_process", side_effect=runner):
            return orchestrator.main(argv)

    def test_rejects_skip_full_tests_in_formal_mode(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        with self.assertRaisesRegex(orchestrator.CompletionError, "skip-full-tests.*synthetic"):
            self.run_main(root, FakeRunner(root), "--skip-full-tests")

    def test_deferred_step7_skips_step7_artifacts_and_preserves_owner_report(self) -> None:
        holder, root, step7_run, build_status, verify_status = self.make_project()
        self.addCleanup(holder.cleanup)
        write_text(root / "reports/phase6/step07_appendix_audits.md", DEFERRED_STEP7_REPORT)
        original_report = (root / "reports/phase6/step07_appendix_audits.md").read_text(encoding="utf-8")
        (step7_run / "manifest.json").unlink()
        build_status.unlink()
        verify_status.unlink()
        runner = FakeRunner(root)

        result = self.run_main_deferred(root, runner)

        self.assertEqual(result, 0)
        labels = [str(item["label"]) for item in runner.calls]
        self.assertEqual(
            labels,
            [
                "step8-freeze-config",
                "step8-build",
                "step8-verify",
                "step8-full-tests",
                "step9-build",
                "step9-verify",
                "step9-full-tests",
                "step10-freeze-config",
                "step10-build",
                "step10-verify",
                "step10-full-tests",
                "step11-freeze-config",
                "step11-build",
                "step11-verify",
                "step11-full-tests",
            ],
        )
        discovery_commands = [
            tuple(item["command"])
            for item in runner.calls
            if str(item["label"]).endswith("full-tests") or str(item["label"]) == "step7-close-tests"
        ]
        expected_discovery = (
            str(root / ".venv/bin/python"),
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        )
        self.assertEqual(discovery_commands, [expected_discovery] * 4)

        step8_freeze = next(item for item in runner.calls if str(item["label"]) == "step8-freeze-config")
        step8_command = tuple(str(part) for part in step8_freeze["command"])
        self.assertIn("--step7-deferred-report", step8_command)
        self.assertNotIn("--step7-run-path", step8_command)
        self.assertNotIn(str(step7_run), step8_command)

        self.assertEqual(
            original_report,
            (root / "reports/phase6/step07_appendix_audits.md").read_text(encoding="utf-8"),
        )
        checklist = (root / "reports/ACTIVE_GOAL_COMPLETION_CHECKLIST.md").read_text(encoding="utf-8")
        self.assertIn("DEFERRED BY OWNER — NOT RUN/NOT USED AS EVIDENCE", checklist)
        self.assertNotIn("Appendix audits: Step 7 complete", checklist)
        self.assertIn(
            "internal publication artifact plan complete with Step7 deferred and external owner gates pending",
            checklist,
        )
        self.assertNotIn("Appendix audits: satisfied", checklist)
        project_log = (root / "PROJECT_LOG.md").read_text(encoding="utf-8")
        self.assertIn("STEP 7 DEFERRED BY OWNER", project_log)

    def test_deferred_step7_requires_exact_state_token_line(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        write_text(
            root / "reports/phase6/step07_appendix_audits.md",
            "# Phase 6 Step 7 Appendix Audits\n\nStatus: deferred_by_owner\n",
        )
        with self.assertRaisesRegex(orchestrator.CompletionError, "exact state deferred_by_owner"):
            self.run_main_deferred(root, FakeRunner(root))

    def test_validates_exact_step7_inventory_before_running_commands(self) -> None:
        holder, root, step7_run, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        (step7_run / "summary.md").unlink()
        rewrite_sha256sums(step7_run)
        runner = FakeRunner(root)
        with self.assertRaisesRegex(orchestrator.CompletionError, "Step7 inventory"):
            self.run_main(root, runner)
        self.assertEqual(runner.calls, [])

    def test_runs_expected_command_sequence_and_generates_unique_reports(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        runner = FakeRunner(root)
        result = self.run_main(root, runner)
        self.assertEqual(result, 0)
        labels = [str(item["label"]) for item in runner.calls]
        self.assertEqual(
            labels,
            [
                "step7-close-tests",
                "step8-freeze-config",
                "step8-build",
                "step8-verify",
                "step8-full-tests",
                "step9-build",
                "step9-verify",
                "step9-full-tests",
                "step10-freeze-config",
                "step10-build",
                "step10-verify",
                "step10-full-tests",
                "step11-freeze-config",
                "step11-build",
                "step11-verify",
                "step11-full-tests",
            ],
        )
        for relative in (
            "reports/phase6/step07_appendix_audits.md",
            "reports/phase6/step08_publication_core.md",
            "reports/phase6/step09_release_metadata.md",
            "reports/phase6/step10_manuscript_claim_map.md",
            "reports/phase6/step11_submission_bundle.md",
        ):
            self.assertTrue((root / relative).is_file(), relative)
        step11 = (root / "reports/phase6/step11_submission_bundle.md").read_text(encoding="utf-8")
        self.assertIn("pending_owner_license_selection", step11)
        self.assertIn("deferred_no_redistributable_hidden_gt", step11)
        self.assertNotIn("publish", step11.lower())
        self.assertNotIn("submission pending", step11.lower())
        for call in runner.calls:
            joined = " ".join(str(part) for part in call["command"]).lower()
            self.assertNotIn("publish", joined)
            self.assertNotIn("submit", joined)
            self.assertNotIn("license choose", joined)
        discovery_commands = [
            tuple(item["command"])
            for item in runner.calls
            if str(item["label"]).endswith("full-tests") or str(item["label"]) == "step7-close-tests"
        ]
        expected_discovery = (
            str(root / ".venv/bin/python"),
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        )
        self.assertEqual(len(discovery_commands), 5)
        self.assertEqual(discovery_commands, [expected_discovery] * 5)

    def test_writes_canonical_failure_status_and_stops_immediately_on_nonzero_exit(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        runner = FakeRunner(root, fail_label="step9-build")
        with self.assertRaisesRegex(orchestrator.CommandFailure, "step9-build"):
            self.run_main(root, runner)
        labels = [str(item["label"]) for item in runner.calls]
        self.assertEqual(
            labels,
            [
                "step7-close-tests",
                "step8-freeze-config",
                "step8-build",
                "step8-verify",
                "step8-full-tests",
                "step9-build",
            ],
        )
        status_path = root / ".tmp/phase6-completion/status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["failed_stage"], "step9-build")
        self.assertEqual(status["exit_code"], 2)
        self.assertIn("recovery_command", status)
        self.assertTrue((root / ".tmp/phase6-completion/step9-build.stderr.log").is_file())
        self.assertFalse((root / "reports/phase6/step09_release_metadata.md").exists())

    def test_report_and_block_updates_are_idempotent_across_repeated_successful_runs(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        first_runner = FakeRunner(root)
        second_runner = FakeRunner(root)
        self.assertEqual(self.run_main(root, first_runner), 0)
        first_step11 = (root / "reports/phase6/step11_submission_bundle.md").read_text(encoding="utf-8")
        first_checklist = (root / "reports/ACTIVE_GOAL_COMPLETION_CHECKLIST.md").read_text(encoding="utf-8")
        first_log = (root / "PROJECT_LOG.md").read_text(encoding="utf-8")
        self.assertEqual(self.run_main(root, second_runner), 0)
        self.assertEqual(first_step11, (root / "reports/phase6/step11_submission_bundle.md").read_text(encoding="utf-8"))
        checklist = (root / "reports/ACTIVE_GOAL_COMPLETION_CHECKLIST.md").read_text(encoding="utf-8")
        project_log = (root / "PROJECT_LOG.md").read_text(encoding="utf-8")
        self.assertEqual(checklist.count("phase6-completion-orchestrator:checklist:start"), 1)
        self.assertEqual(project_log.count("phase6-completion-orchestrator:project-log:start"), 1)
        self.assertEqual(checklist, first_checklist)
        self.assertEqual(project_log, first_log)

    def test_accepts_real_step7_key_value_status_files(self) -> None:
        holder, root, _, build_status, verify_status = self.make_project()
        self.addCleanup(holder.cleanup)
        write_kv_status(
            build_status,
            {"status": "finished", "exit_code": "0", "wall_seconds": "10.25"},
        )
        write_kv_status(
            verify_status,
            {"status": "finished", "exit_code": "0", "wall_seconds": "11.50"},
        )
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)

    def test_accepts_unittest_discover_summary_when_emitted_on_stderr(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        self.assertEqual(
            self.run_main(root, FakeRunner(root, discovery_summary_on_stderr=True)),
            0,
        )

    def test_accepts_verify_cli_output_that_only_emits_run_id(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        self.assertEqual(
            self.run_main(root, FakeRunner(root, verify_outputs_only_run_id=True)),
            0,
        )

    def test_rejects_step7_status_when_status_field_is_not_finished(self) -> None:
        holder, root, _, build_status, verify_status = self.make_project()
        self.addCleanup(holder.cleanup)
        write_kv_status(
            build_status,
            {"status": "running", "exit_code": "0", "wall_seconds": "10.25"},
        )
        write_kv_status(
            verify_status,
            {"status": "finished", "exit_code": "0", "wall_seconds": "11.50"},
        )
        with self.assertRaisesRegex(orchestrator.CompletionError, "status.*finished"):
            self.run_main(root, FakeRunner(root))

    def test_rejects_verify_status_when_run_path_does_not_match_step7_run(self) -> None:
        holder, root, _, build_status, verify_status = self.make_project()
        self.addCleanup(holder.cleanup)
        write_kv_status(
            build_status,
            {"status": "finished", "exit_code": "0", "wall_seconds": "10.25"},
        )
        write_kv_status(
            verify_status,
            {
                "status": "finished",
                "exit_code": "0",
                "wall_seconds": "11.50",
                "run_path": str(root / "results/phase6/appendix_audits_v1/wrong-run"),
            },
        )
        with self.assertRaisesRegex(orchestrator.CompletionError, "run_path"):
            self.run_main(root, FakeRunner(root))

    def test_step7_close_uses_formal_worker_counts_for_build_and_verify(self) -> None:
        holder, root, _, build_status, verify_status = self.make_project()
        self.addCleanup(holder.cleanup)
        write_kv_status(
            build_status,
            {
                "status": "finished",
                "exit_code": "0",
                "wall_seconds": "10.25",
                "worker_count": "8",
            },
        )
        write_kv_status(
            verify_status,
            {
                "status": "finished",
                "exit_code": "0",
                "wall_seconds": "11.50",
                "worker_count": "6",
                "run_path": str(root / "results/phase6/appendix_audits_v1/phase6-appendix-audits-fixture"),
            },
        )
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)

    def test_accepts_step7_manifest_and_ledger_in_frozen_non_lexical_order(self) -> None:
        holder, root, step7_run, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        frozen_order = [
            "config.json",
            "authority_bridge.json",
            "preflight.json",
            "audit_status.csv",
            "resampling_alignment.csv",
            "resampling_rank_stability.csv",
            "normalization_alignment.csv",
            "normalization_rank_stability.csv",
            "peak_tolerance_phase4_alignment.csv",
            "peak_tolerance_phase4_rank_stability.csv",
            "peak_tolerance_phase6_system.csv",
            "peak_tolerance_phase6_rank_stability.csv",
            "leakage_boundaries.jsonl",
            "leakage_duplicate_candidates.jsonl",
            "summary.md",
        ]
        write_json(
            step7_run / "manifest.json",
            {
                "status": "complete",
                "payload_files": frozen_order,
                "schema_version": "phase6-appendix-audits-artifact-v1",
                "counts": json.loads((step7_run / "manifest.json").read_text(encoding="utf-8"))["counts"],
            },
        )
        rewrite_sha256sums_in_order(step7_run, [*frozen_order, "manifest.json", "complete.json"])
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)

    def test_accepts_real_frozen_step7_config_shape_without_checksum_entry_count(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)

    def test_step11_report_uses_full_manifest_inventory_and_exact_owner_gate_states(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)
        step11 = (root / "reports/phase6/step11_submission_bundle.md").read_text(encoding="utf-8")
        self.assertIn("- payload files checked: `12`", step11)
        self.assertIn("pending_owner_license_selection", step11)
        self.assertIn("pending_owner_review_rruff_and_bacteria", step11)
        self.assertIn("deferred_pending_locked_renderer_and_venue_choice", step11)
        self.assertIn("pending_owner_choice", step11)
        self.assertIn("not_submitted_pending_owner_authorization", step11)
        self.assertIn("deferred_no_redistributable_hidden_gt", step11)

    def test_step11_report_uses_terminal_complete_status_when_manifest_has_no_status(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)
        run_path = root / "results/phase6/submission_bundle_v1/phase6-submission-bundle-fixture"
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("status", None)
        write_json(run_path / "manifest.json", manifest)
        rewrite_sha256sums(run_path)
        report = orchestrator._render_step11_report(
            project_root=root,
            run_path=run_path,
            test_summary={"test_count": 3, "duration_seconds": 0.125, "status_line": "OK"},
        )
        self.assertIn("- terminal status: `complete`", report)
        self.assertNotIn("manifest status", report)

    def test_validate_complete_artifact_accepts_step11_style_manifest_without_status(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)
        run_path = root / "results/phase6/submission_bundle_v1/phase6-submission-bundle-fixture"
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("status", None)
        write_json(run_path / "manifest.json", manifest)
        rewrite_sha256sums(run_path)
        validated = orchestrator._validate_complete_artifact(
            run_path,
            required_payloads=orchestrator.STEP11_REQUIRED_PAYLOADS,
            name="Step11 build",
        )
        self.assertEqual(validated["schema_version"], "phase6-submission-bundle-artifact-v1")

    def test_step7_report_uses_elapsed_seconds_with_wall_seconds_fallback(self) -> None:
        holder, root, step7_run, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        manifest = json.loads((step7_run / "manifest.json").read_text(encoding="utf-8"))
        report = orchestrator._render_step7_report(
            project_root=root,
            step7_run=step7_run,
            manifest=manifest,
            build_status={"exit_code": 0, "elapsed_seconds": 10.25},
            verify_status={"exit_code": 0, "wall_seconds": 11.5},
            test_summary={"test_count": 3, "duration_seconds": 0.125, "status_line": "OK"},
        )
        self.assertIn("- build status exit_code: `0`; elapsed_seconds: `10.25`", report)
        self.assertIn("- verify status exit_code: `0`; elapsed_seconds: `11.5`", report)
        self.assertNotIn("wall_seconds", report)

    def test_successful_run_rewrites_stale_top_level_checklist_states(self) -> None:
        holder, root, _, _, _ = self.make_project()
        self.addCleanup(holder.cleanup)
        self.assertEqual(self.run_main(root, FakeRunner(root)), 0)
        checklist = (root / "reports/ACTIVE_GOAL_COMPLETION_CHECKLIST.md").read_text(encoding="utf-8")
        self.assertNotIn("Status: active; not complete", checklist)
        self.assertNotIn("RENDER PENDING", checklist)
        self.assertNotIn("STEP-7 EXECUTION PENDING", checklist)
        self.assertNotIn("Release metadata / data card / Croissant: pending", checklist)
        self.assertNotIn("Paper manuscript: MISSING", checklist)
        self.assertNotIn("Submission-ready package: MISSING", checklist)
        self.assertIn(
            "Phase 6 internal artifact plan complete; owner-only external publication/venue/license/redistribution/renderer/leaderboard gates pending/deferred",
            checklist,
        )
        self.assertNotIn("published", checklist.lower())
        self.assertNotIn("submitted", checklist.lower())


if __name__ == "__main__":
    unittest.main()
