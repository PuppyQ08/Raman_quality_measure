from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_submission_bundle import (  # noqa: E402
    SubmissionBundleError,
    build_phase6_submission_bundle,
)
from rpe.runner.phase6_submission_bundle_verifier import (  # noqa: E402
    SubmissionBundleVerificationError,
    verify_phase6_submission_bundle,
)
import tools.run_phase6_submission_bundle as cli_module  # noqa: E402


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


def write_canonical_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def files(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def rewrite_sha256sums(path: Path) -> None:
    payloads = sorted(
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    )
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_bytes((path / relative).read_bytes())}  {relative}\n"
            for relative in payloads
        ),
        encoding="utf-8",
    )


def write_parent_artifact(path: Path, payloads: dict[str, bytes]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for relative, payload in payloads.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    write_canonical_json(
        path / "manifest.json",
        {
            "artifact_schema_version": "fixture-parent-artifact-v1",
            "payload_files": sorted(payloads),
            "status": "complete",
        },
    )
    write_canonical_json(path / "complete.json", {"status": "complete"})
    rewrite_sha256sums(path)


def fixture_project(root: Path) -> Path:
    step8 = root / "fixtures" / "step8_publication_core"
    step9 = root / "fixtures" / "step9_release_metadata"
    step10 = root / "fixtures" / "step10_manuscript"

    table_rows = (
        "endpoint_id,metric_id,estimate,state\n"
        "D5-A,mse,0.125,complete_numeric\n"
        "D4-A,mse,0.375,complete_numeric\n"
    ).encode("utf-8")

    claim_stream = io.StringIO(newline="")
    claim_fields = (
        "claim_id", "manuscript_section", "claim_text", "evidence_class",
        "authority_path", "authority_sha256", "payload_path", "row_selector",
        "allowed_scope", "prohibited_scope", "verification_state",
    )
    claim_writer = csv.DictWriter(
        claim_stream, fieldnames=claim_fields, lineterminator="\n"
    )
    claim_writer.writeheader()
    claim_writer.writerow(
        {
            "claim_id": "claim-d5a",
            "manuscript_section": "Results",
            "claim_text": "D5-A MSE was 0.125 in the frozen table",
            "evidence_class": "numerical_result",
            "authority_path": "fixtures/step8_publication_core/table1_metric_validity.csv",
            "authority_sha256": sha256_bytes(table_rows),
            "payload_path": "assets/tables/table1_metric_validity.csv",
            "row_selector": json.dumps(
                {"endpoint_id": "D5-A", "metric_id": "mse"},
                sort_keys=True, separators=(",", ":"),
            ),
            "allowed_scope": "task_native_metric_validity",
            "prohibited_scope": "method_performance_extrapolation",
            "verification_state": "verified",
        }
    )
    claims_rows = claim_stream.getvalue().encode("utf-8")

    manuscript = (
        "# Raman Bundle Fixture\n\n"
        "Figure link: [Figure 1](../assets/figures/figure1.png)\n\n"
        "Table link: [Table 1](../assets/tables/table1_metric_validity.csv)\n\n"
        "Phase 0.5 original confirmatory gate not_met. The user-authorized internal screening remained separate and was not presented as the original gate passing.\n\n"
        "D5-A MSE was 0.125 in the frozen table.\n\n"
        "The main frozen claim cites [@smith2024].\n"
    ).encode("utf-8")
    references = (
        "@article{smith2024,\n"
        "  title = {Frozen Raman Study},\n"
        "  author = {Smith, Ada},\n"
        "  journal = {Journal of Fixtures},\n"
        "  year = {2024}\n"
        "}\n"
    ).encode("utf-8")

    write_parent_artifact(
        step8,
        {
            "table1_metric_validity.csv": table_rows,
            "figure1.png": b"fixture-figure-1-png\n",
            "figure2.png": b"fixture-figure-2-png\n",
            "figure3.png": b"fixture-figure-3-png\n",
            "figure4.png": b"fixture-figure-4-png\n",
        },
    )
    write_parent_artifact(
        step9,
        {
            "release_metadata.json": canonical(
                {
                    "release_surface": "journal_neutral_source_bundle",
                    "license_gate": "pending_owner_license_selection",
                    "redistribution_gate": "pending_owner_review_rruff_and_bacteria",
                    "leaderboard_state": "deferred_no_redistributable_hidden_gt",
                }
            ),
        },
    )
    write_parent_artifact(
        step10,
        {
            "manuscript.md": manuscript,
            "references.bib": references,
            "claims_matrix.csv": claims_rows,
        },
    )

    write_canonical_json(
        root / "fixtures" / "submission_bundle_fixture_config.json",
        {
            "schema_version": "phase6-submission-bundle-v1",
            "bundle_name": "phase6-submission-bundle",
            "source_mode": "synthetic_fixture",
            "parents": {
                "step8": {
                    "path": "fixtures/step8_publication_core",
                    "required_payloads": [
                        "figure1.png",
                        "figure2.png",
                        "figure3.png",
                        "figure4.png",
                        "table1_metric_validity.csv",
                    ],
                    "required_terminal": "complete.json",
                },
                "step9": {
                    "path": "fixtures/step9_release_metadata",
                    "required_payloads": ["release_metadata.json"],
                    "required_terminal": "complete.json",
                },
                "step10": {
                    "path": "fixtures/step10_manuscript",
                    "required_payloads": ["manuscript.md", "references.bib", "claims_matrix.csv"],
                    "required_terminal": "complete.json",
                },
            },
            "bundle_files": [
                {
                    "parent": "step10",
                    "source_path": "manuscript.md",
                    "bundle_path": "paper/manuscript.md",
                    "role": "manuscript",
                },
                {
                    "parent": "step10",
                    "source_path": "references.bib",
                    "bundle_path": "paper/references.bib",
                    "role": "references",
                },
                {
                    "parent": "step10",
                    "source_path": "claims_matrix.csv",
                    "bundle_path": "paper/claims_matrix.csv",
                    "role": "claims_matrix",
                },
                {
                    "parent": "step8",
                    "source_path": "table1_metric_validity.csv",
                    "bundle_path": "assets/tables/table1_metric_validity.csv",
                    "role": "aggregate_table",
                },
                {
                    "parent": "step8",
                    "source_path": "figure1.png",
                    "bundle_path": "assets/figures/figure1.png",
                    "role": "frozen_figure",
                },
                {
                    "parent": "step8",
                    "source_path": "figure2.png",
                    "bundle_path": "assets/figures/figure2.png",
                    "role": "frozen_figure",
                },
                {
                    "parent": "step8",
                    "source_path": "figure3.png",
                    "bundle_path": "assets/figures/figure3.png",
                    "role": "verified_step8_figure",
                },
                {
                    "parent": "step8",
                    "source_path": "figure4.png",
                    "bundle_path": "assets/figures/figure4.png",
                    "role": "verified_step8_figure",
                },
                {
                    "parent": "step9",
                    "source_path": "release_metadata.json",
                    "bundle_path": "metadata/release_metadata.json",
                    "role": "release_metadata",
                },
            ],
            "artifact_contract": {
                "payload_files": [
                    "config.json",
                    "preflight.json",
                    "paper/manuscript.md",
                    "paper/references.bib",
                    "paper/claims_matrix.csv",
                    "assets/tables/table1_metric_validity.csv",
                    "assets/figures/figure1.png",
                    "assets/figures/figure2.png",
                    "assets/figures/figure3.png",
                    "assets/figures/figure4.png",
                    "metadata/release_metadata.json",
                    "metadata/environment_receipt.json",
                    "metadata/limitations_ledger.json",
                    "metadata/artifact_claim_map.json",
                    "manifest.json",
                ],
                "terminal_marker": "complete.json",
                "archive_format": "directory_only",
            },
        },
    )
    return root / "fixtures" / "submission_bundle_fixture_config.json"


def formal_integration_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    step8 = root / "fixtures" / "step8_publication_core_formal"
    step9 = root / "fixtures" / "step9_release_metadata_formal"
    step10 = root / "fixtures" / "step10_manuscript_formal"
    step4 = root / "fixtures" / "step4_final_figures_authority"

    table1_csv = (
        "endpoint_id,metric_id,estimate,state\n"
        "D5-A,mse,0.125,complete_numeric\n"
    ).encode("utf-8")
    table1_md = (
        "| endpoint_id | metric_id | estimate | state |\n"
        "| --- | --- | --- | --- |\n"
        "| D5-A | mse | 0.125 | complete_numeric |\n"
    ).encode("utf-8")
    table2_csv = (
        "evidence_id,system_id,state\n"
        "baseline-d5,sys-01,not_evaluable_no_validated_clean_gt\n"
    ).encode("utf-8")
    table2_md = (
        "| evidence_id | system_id | state |\n"
        "| --- | --- | --- |\n"
        "| baseline-d5 | sys-01 | not_evaluable_no_validated_clean_gt |\n"
    ).encode("utf-8")
    s1_csv = (
        "endpoint_id,metric_output_id,ag,acc_cross\n"
        "D5-A,mse,0.125,0.875\n"
    ).encode("utf-8")
    s2_csv = (
        "endpoint_id,metric_output_id,interaction_value\n"
        "D5-A,mse,0.05\n"
    ).encode("utf-8")
    s3_csv = (
        "task_line,system_id,publication_disposition\n"
        "baseline_correction,sys-01,local_only_pending_redistribution_review\n"
    ).encode("utf-8")
    figure3_csv = (
        "extractor_id,excitation_stratum,gate_state\n"
        "airpls,green_532,FAIL p95\n"
    ).encode("utf-8")
    figure4_nodes = (
        "node_id,label,state\n"
        "phase0_5,Phase 0.5 original confirmatory gate,not_met\n"
    ).encode("utf-8")
    figure4_edges = (
        "source,target,edge_kind\n"
        "phase0_5,phase6,authorized_fallback\n"
    ).encode("utf-8")

    write_parent_artifact(
        step8,
        {
            "table1_metric_validity.csv": table1_csv,
            "table1_metric_validity.md": table1_md,
            "table_s1_full_alignment.csv": s1_csv,
            "table_s2_protocol_interactions.csv": s2_csv,
            "table2_method_evidence.csv": table2_csv,
            "table2_method_evidence.md": table2_md,
            "table_s3_system_status.csv": s3_csv,
            "figure3_model_adequacy_data.csv": figure3_csv,
            "figure3_model_adequacy.png": b"formal-figure3-png\n",
            "figure3_model_adequacy.svg": b"<svg>formal-figure3</svg>\n",
            "figure4_gate_nodes.csv": figure4_nodes,
            "figure4_gate_edges.csv": figure4_edges,
            "figure4_gate_map.png": b"formal-figure4-png\n",
            "figure4_gate_map.svg": b"<svg>formal-figure4</svg>\n",
        },
    )
    write_parent_artifact(
        step4,
        {
            "figure1_phase4_response.png": b"formal-figure1-png\n",
            "figure1_phase4_response.svg": b"<svg>formal-figure1</svg>\n",
            "figure2_phase4_alignment.png": b"formal-figure2-png\n",
            "figure2_phase4_alignment.svg": b"<svg>formal-figure2</svg>\n",
        },
    )
    write_parent_artifact(
        step9,
        {
            "data_card.md": b"# Data Card\n",
            "croissant.json": canonical({"@type": "sc:Dataset", "name": "formal-fixture"}),
            "metadata_index.parquet": b"PAR1formal-metadata-index",
            "release_matrix.csv": (
                "artifact_id,disposition\n"
                "rruff_records,local_only_pending_redistribution_review\n"
            ).encode("utf-8"),
            "limitations.jsonl": (
                canonical(
                    {
                        "limitation_id": "license_gate",
                        "state": "pending_owner_license_selection",
                    }
                )
                + canonical(
                    {
                        "limitation_id": "redistribution_gate",
                        "state": "pending_owner_review_rruff_and_bacteria",
                    }
                )
            ),
            "environment.json": canonical({"python_version": "3.13.11"}),
        },
    )
    claim_stream = io.StringIO(newline="")
    claim_fields = (
        "claim_id", "manuscript_section", "claim_text", "evidence_class",
        "authority_path", "authority_sha256", "payload_path", "row_selector",
        "allowed_scope", "prohibited_scope", "verification_state",
    )
    claim_writer = csv.DictWriter(
        claim_stream, fieldnames=claim_fields, lineterminator="\n"
    )
    claim_writer.writeheader()
    claim_writer.writerow(
        {
            "claim_id": "claim-formal-table1",
            "manuscript_section": "Results",
            "claim_text": "D5-A MSE was 0.125 in the frozen table",
            "evidence_class": "numerical_result",
            "authority_path": "fixtures/step8_publication_core_formal/table1_metric_validity.csv",
            "authority_sha256": sha256_bytes(table1_csv),
            "payload_path": "fixtures/step8_publication_core_formal/table1_metric_validity.csv",
            "row_selector": json.dumps(
                {"endpoint_id": "D5-A", "metric_id": "mse"},
                sort_keys=True, separators=(",", ":"),
            ),
            "allowed_scope": "task_native_metric_validity",
            "prohibited_scope": "method_performance_extrapolation",
            "verification_state": "verified",
        }
    )
    write_parent_artifact(
        step10,
        {
            "paper/manuscript.md": (
                "# Formal Submission Bundle Fixture\n\n"
                "Figure 1: [Figure 1](../assets/figures/figure1_phase4_response.png)\n\n"
                "Figure 3: [Figure 3](../assets/figures/figure3_model_adequacy.png)\n\n"
                "Table 1: [Table 1](../assets/tables/table1_metric_validity.csv)\n\n"
                "Phase 0.5 original confirmatory gate not_met. The user-authorized internal screening remained separate and was not presented as the original gate passing.\n\n"
                "The main frozen claim cites [@smith2024].\n"
            ).encode("utf-8"),
            "paper/references.bib": (
                "@article{smith2024,\n"
                "  title = {Frozen Raman Study},\n"
                "  author = {Smith, Ada},\n"
                "  journal = {Journal of Fixtures},\n"
                "  year = {2024}\n"
                "}\n"
            ).encode("utf-8"),
            "paper/claims_matrix.csv": claim_stream.getvalue().encode("utf-8"),
        },
    )
    return step4, step8, step9, step10


class Phase6SubmissionBundleTest(unittest.TestCase):
    def make_fixture(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project_root = Path(temporary.name)
        config_path = fixture_project(project_root)
        return project_root, config_path

    def test_build_is_deterministic_and_preserves_frozen_bytes(self) -> None:
        project_root, config_path = self.make_fixture()
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = build_phase6_submission_bundle(
                Path(first_root),
                config_path=config_path,
                project_root=project_root,
            )
            second = build_phase6_submission_bundle(
                Path(second_root),
                config_path=config_path,
                project_root=project_root,
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(files(first.path), files(second.path))
            self.assertEqual(
                (first.path / "assets/figures/figure1.png").read_bytes(),
                (project_root / "fixtures/step8_publication_core/figure1.png").read_bytes(),
            )
            self.assertEqual(
                (first.path / "assets/figures/figure4.png").read_bytes(),
                (project_root / "fixtures/step8_publication_core/figure4.png").read_bytes(),
            )
            limitations = read_json(first.path / "metadata/limitations_ledger.json")
            self.assertEqual(limitations["license_gate"], "pending_owner_license_selection")
            self.assertEqual(
                limitations["redistribution_gate"],
                "pending_owner_review_rruff_and_bacteria",
            )
            self.assertEqual(
                limitations["renderer_state"],
                "deferred_pending_locked_renderer_and_venue_choice",
            )
            self.assertEqual(limitations["venue_state"], "pending_owner_choice")
            self.assertEqual(
                limitations["submission_state"],
                "not_submitted_pending_owner_authorization",
            )
            self.assertEqual(
                limitations["leaderboard_state"],
                "deferred_no_redistributable_hidden_gt",
            )
            self.assertFalse(any(name.endswith(".tar.gz") for name in files(first.path)))
            manifest_text = (first.path / "manifest.json").read_text(encoding="utf-8")
            self.assertNotIn(str(project_root), manifest_text)
            self.assertNotIn(str(first.path), manifest_text)

    def test_build_fails_closed_on_parent_terminal_inventory_or_ledger_mismatch(self) -> None:
        project_root, config_path = self.make_fixture()
        step8 = project_root / "fixtures/step8_publication_core"
        (step8 / "complete.json").unlink()
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(SubmissionBundleError, "terminal|inventory|ledger|complete"):
                build_phase6_submission_bundle(
                    Path(output_root),
                    config_path=config_path,
                    project_root=project_root,
                )

    def test_build_rejects_broken_markdown_link(self) -> None:
        project_root, config_path = self.make_fixture()
        manuscript_path = project_root / "fixtures/step10_manuscript/manuscript.md"
        manuscript_path.write_text(
            manuscript_path.read_text(encoding="utf-8").replace(
                "../assets/figures/figure1.png",
                "../assets/figures/missing.png",
            ),
            encoding="utf-8",
        )
        rewrite_sha256sums(project_root / "fixtures/step10_manuscript")
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(SubmissionBundleError, "Markdown|link|missing"):
                build_phase6_submission_bundle(
                    Path(output_root),
                    config_path=config_path,
                    project_root=project_root,
                )

    def test_build_rejects_missing_bibtex_citation_key(self) -> None:
        project_root, config_path = self.make_fixture()
        manuscript_path = project_root / "fixtures/step10_manuscript/manuscript.md"
        manuscript_path.write_text(
            manuscript_path.read_text(encoding="utf-8").replace("[@smith2024]", "[@missing2024]"),
            encoding="utf-8",
        )
        rewrite_sha256sums(project_root / "fixtures/step10_manuscript")
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(SubmissionBundleError, "citation|BibTeX|missing2024"):
                build_phase6_submission_bundle(
                    Path(output_root),
                    config_path=config_path,
                    project_root=project_root,
                )

    def test_build_rejects_invalid_claim_selector(self) -> None:
        project_root, config_path = self.make_fixture()
        claims_path = project_root / "fixtures/step10_manuscript/claims_matrix.csv"
        rows = read_csv(claims_path)
        rows[0]["row_selector"] = "endpoint_id=DOES_NOT_EXIST"
        with claims_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        rewrite_sha256sums(project_root / "fixtures/step10_manuscript")
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(SubmissionBundleError, "claim|selector|row"):
                build_phase6_submission_bundle(
                    Path(output_root),
                    config_path=config_path,
                    project_root=project_root,
                )

    def test_build_rejects_phase05_wording_boundary_violation(self) -> None:
        project_root, config_path = self.make_fixture()
        manuscript_path = project_root / "fixtures/step10_manuscript/manuscript.md"
        manuscript_path.write_text(
            "# Bad Phase 0.5\n\nPhase 0.5 confirmatory gate passed after internal screening.\n",
            encoding="utf-8",
        )
        rewrite_sha256sums(project_root / "fixtures/step10_manuscript")
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(SubmissionBundleError, "Phase 0.5|confirmatory gate|internal screening"):
                build_phase6_submission_bundle(
                    Path(output_root),
                    config_path=config_path,
                    project_root=project_root,
                )

    def test_build_rejects_forbidden_source_path_patterns(self) -> None:
        project_root, config_path = self.make_fixture()
        raw_dir = project_root / "data/raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        forbidden = raw_dir / "rruff_record_dump.csv"
        forbidden.write_text("restricted\n", encoding="utf-8")
        config = read_json(config_path)
        config["bundle_files"].append(
            {
                "parent": "step9",
                "source_path": "../../data/raw/rruff_record_dump.csv",
                "bundle_path": "assets/tables/forbidden.csv",
                "role": "forbidden",
            }
        )
        write_canonical_json(config_path, config)
        with tempfile.TemporaryDirectory() as output_root:
            with self.assertRaisesRegex(SubmissionBundleError, "forbidden|restricted|raw|rruff"):
                build_phase6_submission_bundle(
                    Path(output_root),
                    config_path=config_path,
                    project_root=project_root,
                )

    def test_independent_verifier_rebuilds_and_rejects_checksum_rewritten_tamper(self) -> None:
        project_root, config_path = self.make_fixture()
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_submission_bundle(
                Path(output_root),
                config_path=config_path,
                project_root=project_root,
            )
            verified = verify_phase6_submission_bundle(summary.path, project_root=project_root)
            self.assertEqual(verified.run_id, summary.run_id)
            manuscript = summary.path / "paper/manuscript.md"
            manuscript.write_text(
                manuscript.read_text(encoding="utf-8").replace("0.125", "0.925"),
                encoding="utf-8",
            )
            rewrite_sha256sums(summary.path)
            with self.assertRaisesRegex(
                SubmissionBundleVerificationError,
                "mismatch|tamper|byte|payload",
            ):
                verify_phase6_submission_bundle(summary.path, project_root=project_root)

    def test_verifier_module_does_not_import_production_runner(self) -> None:
        verifier_path = ROOT / "rpe/runner/phase6_submission_bundle_verifier.py"
        tree = ast.parse(verifier_path.read_text(encoding="utf-8"))
        banned = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                banned.extend(
                    alias.name
                    for alias in node.names
                    if alias.name == "rpe.runner.phase6_submission_bundle"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "rpe.runner.phase6_submission_bundle":
                banned.append(node.module)
        self.assertEqual(banned, [])
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("build_phase6_submission_bundle", names)

    def test_freeze_tool_emits_formal_config_and_build_maps_claim_payload_source_to_bundle_target(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project_root = Path(temporary.name)
        step4, step8, step9, step10 = formal_integration_fixture(project_root)

        from tools import freeze_phase6_submission_bundle_config as freeze_module

        frozen_path = (
            project_root
            / "experiments/phase6/configs/submission_bundle_v1.json"
        )
        self.assertFalse(frozen_path.exists())
        freeze_module.main(
            [
                "--project-root", str(project_root),
                "--step4-final-figures", str(step4),
                "--step8-publication-core", str(step8),
                "--step9-release-metadata", str(step9),
                "--step10-manuscript", str(step10),
            ]
        )
        frozen = frozen_path.read_bytes()
        self.assertEqual(frozen, canonical(json.loads(frozen)))
        document = json.loads(frozen)
        self.assertEqual(document["source_mode"], "formal")
        self.assertEqual(
            document["parents"]["step10"]["required_payloads"],
            ["paper/manuscript.md", "paper/references.bib", "paper/claims_matrix.csv"],
        )
        self.assertIn(
            {
                "parent": "step4_final_figures",
                "source_path": "figure1_phase4_response.png",
                "bundle_path": "assets/figures/figure1_phase4_response.png",
                "role": "frozen_phase4_figure",
            },
            document["bundle_files"],
        )
        self.assertIn(
            {
                "parent": "step8",
                "source_path": "table1_metric_validity.csv",
                "bundle_path": "assets/tables/table1_metric_validity.csv",
                "role": "aggregate_table",
            },
            document["bundle_files"],
        )
        with tempfile.TemporaryDirectory() as output_root:
            summary = build_phase6_submission_bundle(
                Path(output_root),
                config_path=frozen_path,
                project_root=project_root,
            )
            self.assertTrue((summary.path / "assets/figures/figure1_phase4_response.png").is_file())
            self.assertTrue((summary.path / "assets/figures/figure3_model_adequacy.png").is_file())
            claim_map = read_json(summary.path / "metadata/artifact_claim_map.json")
            self.assertEqual(
                claim_map["claims"][0]["payload_path"],
                "assets/tables/table1_metric_validity.csv",
            )


class Phase6SubmissionBundleCliTest(unittest.TestCase):
    def test_cli_surface_is_build_verify_only(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/run_phase6_submission_bundle.py", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("{build,verify}", completed.stdout)
        stderr = io.StringIO()
        with mock.patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as raised:
                cli_module.main(["build", "--output-root", "unused"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--config", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
