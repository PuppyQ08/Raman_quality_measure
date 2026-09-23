from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rpe.runner.phase6_manuscript as manuscript_module  # noqa: E402
from rpe.runner.phase6_manuscript import (  # noqa: E402
    Phase6ManuscriptError,
    Phase6ManuscriptSummary,
    build_phase6_manuscript,
    load_phase6_manuscript_config,
)
from rpe.runner.phase6_manuscript_verifier import (  # noqa: E402
    Phase6ManuscriptVerificationError,
    verify_phase6_manuscript,
)
import tools.freeze_phase6_manuscript_config as freeze_cli_module  # noqa: E402
import tools.run_phase6_manuscript as cli_module  # noqa: E402


FIXED_TITLE = (
    "When Do Raman Fidelity Metrics Track Downstream Utility? "
    "A Task-, Perturbation-, and Protocol-Aware Evaluation"
)
SECTION_HEADINGS = (
    "Abstract",
    "Introduction and the metric-validity problem",
    "Data sources, provenance, and preprocessing-state taxonomy",
    "Metrics, P1-P12 operators, task-native endpoints, Protocol A/B, AG, "
    "Acc-cross, inference, and prospective gates",
    "Gate outcomes and reproducibility limitations",
    "P8-P12 response results and Figures 1/2",
    "Protocol-dependent metric alignment and interactions",
    "Phase-2 adequacy failure and Figure 3",
    "Evidence/fallback map and Figure 4",
    "Discussion, limitations, ethics/RAI, and data/code availability",
    "References plus supplementary methods/tables",
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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stream.getvalue(), encoding="utf-8")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def artifact_files(root: Path) -> dict[str, bytes]:
    return {
        item.relative_to(root).as_posix(): item.read_bytes()
        for item in root.rglob("*")
        if item.is_file()
    }


def rewrite_sha256sums(root: Path) -> None:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = []
    for name in names:
        lines.append(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


class ManuscriptFixture:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.step8 = base / "step8_publication_core"
        self.step9 = base / "step9_release_metadata"
        self.authorities = base / "authorities"
        self.step7_report = self.authorities / "step07_appendix_execution.md"
        self.config_path = base / "manuscript_config.json"
        self.output_root = base / "out"

    def create(self) -> Path:
        step8_payload = self.step8 / "table1_metric_validity.csv"
        write_csv(
            step8_payload,
            ["claim_id", "value", "state", "scope"],
            [
                {
                    "claim_id": "C001",
                    "value": "5/12",
                    "state": "complete",
                    "scope": "protocol_specific",
                },
                {
                    "claim_id": "C002",
                    "value": "112/200",
                    "state": "not_powered_for_phase5",
                    "scope": "baseline_only",
                },
            ],
        )
        write_json(
            self.step8 / "manifest.json",
            {
                "artifact_type": "phase6_publication_core_fixture",
                "payload_files": ["table1_metric_validity.csv"],
                "status": "complete",
            },
        )
        write_json(self.step8 / "complete.json", {"status": "complete"})
        rewrite_sha256sums(self.step8)

        step9_payload = self.step9 / "data_card.md"
        step9_payload.parent.mkdir(parents=True, exist_ok=True)
        step9_payload.write_text(
            "License and redistribution decisions remain pending owner review.\n",
            encoding="utf-8",
        )
        write_json(
            self.step9 / "manifest.json",
            {
                "artifact_type": "phase6_release_metadata_fixture",
                "payload_files": ["data_card.md"],
                "status": "complete",
            },
        )
        write_json(self.step9 / "complete.json", {"status": "complete"})
        rewrite_sha256sums(self.step9)

        phase4 = self.authorities / "phase4_closure.md"
        phase4.parent.mkdir(parents=True, exist_ok=True)
        phase4.write_text(
            "Phase 4 closure: confirmatory success not evaluable; "
            "Phase 5 not run and not powered.\n",
            encoding="utf-8",
        )
        step1 = self.authorities / "step1_design.md"
        step1.write_text(
            "Step 1 froze the title, eleven sections, claim matrix, and renderer "
            "state deferred_pending_locked_renderer_and_venue_choice.\n",
            encoding="utf-8",
        )
        baseline = self.authorities / "step3_baseline.md"
        baseline.write_text(
            "Baseline evidence remains indirect triple evidence and "
            "not_powered_for_phase5.\n",
            encoding="utf-8",
        )
        denoising = self.authorities / "step4_denoising.md"
        denoising.write_text(
            "Denoising evidence is indirect because no defensible clean target "
            "exists.\n",
            encoding="utf-8",
        )
        peak = self.authorities / "step5_peak.md"
        peak.write_text(
            "Peak evidence is synthetic mechanism evidence with no real peak "
            "assignments.\n",
            encoding="utf-8",
        )
        appendix = self.authorities / "step6_appendix.md"
        appendix.write_text(
            "Step 7 robustness and leakage remain dependency-gated pending "
            "authoritative execution.\n",
            encoding="utf-8",
        )
        self.step7_report.write_text(
            "Synthetic Step 7 authority placeholder for focused Step 10 tests only.\n",
            encoding="utf-8",
        )

        sections = []
        for index, heading in enumerate(SECTION_HEADINGS, start=1):
            paragraphs = [
                f"Section {index} preserves the frozen journal-neutral structure.",
                (
                    "The paper distinguishes numerical result, status evidence, "
                    "and unavailable evidence without inventing a public "
                    "leaderboard, venue, DOI, release URL, or license."
                ),
            ]
            citation_keys: list[str] = []
            claim_ids: list[str] = []
            if heading == "Abstract":
                claim_ids = ["C001"]
                citation_keys = ["smith2020"]
            elif heading == "Gate outcomes and reproducibility limitations":
                claim_ids = ["C002"]
            elif heading == "Discussion, limitations, ethics/RAI, and data/code availability":
                paragraphs.append(
                    "Ethics/RAI and data/code availability remain bounded by "
                    "pending owner license and redistribution decisions."
                )
                citation_keys = ["ng2021"]
            sections.append(
                {
                    "heading": heading,
                    "paragraphs": paragraphs,
                    "citation_keys": citation_keys,
                    "claim_ids": claim_ids,
                }
            )

        config = {
            "schema_version": "phase6-manuscript-v1",
            "artifact_contract": {
                "run_prefix": "phase6-manuscript-",
                "result_root_template": "results/phase6/manuscript_v1/<content-run-id>/",
                "checksum_file": "SHA256SUMS",
                "payload_files": [
                    "config.json",
                    "authority_bridge.json",
                    "preflight.json",
                    "paper/manuscript.md",
                    "paper/references.bib",
                    "paper/claims_matrix.csv",
                    "limitations_ledger.json",
                    "claim_lint_receipt.json",
                    "manifest.json",
                ],
                "configured_payload_count": 9,
                "terminal_markers": ["complete.json", "failed.json"],
                "total_file_count_with_terminal_and_sha256sums": 11,
            },
            "title": FIXED_TITLE,
            "renderer_state": "deferred_pending_locked_renderer_and_venue_choice",
            "source_revision_status": "unavailable_no_valid_git_repository",
            "authorities": {
                "step8_publication_core": {
                    "type": "artifact_dir",
                    "path": str(self.step8),
                },
                "step9_release_metadata": {
                    "type": "artifact_dir",
                    "path": str(self.step9),
                },
                "step7_appendix_execution_report": {
                    "type": "file",
                    "path": str(self.step7_report),
                    "sha256": sha256_path(self.step7_report),
                },
                "phase4_closure": {
                    "type": "file",
                    "path": str(phase4),
                    "sha256": sha256_path(phase4),
                },
                "step1_design": {
                    "type": "file",
                    "path": str(step1),
                    "sha256": sha256_path(step1),
                },
                "step3_baseline": {
                    "type": "file",
                    "path": str(baseline),
                    "sha256": sha256_path(baseline),
                },
                "step4_denoising": {
                    "type": "file",
                    "path": str(denoising),
                    "sha256": sha256_path(denoising),
                },
                "step5_peak": {
                    "type": "file",
                    "path": str(peak),
                    "sha256": sha256_path(peak),
                },
                "step6_appendix": {
                    "type": "file",
                    "path": str(appendix),
                    "sha256": sha256_path(appendix),
                },
            },
            "references": [
                {
                    "key": "smith2020",
                    "entry_type": "article",
                    "fields": {
                        "author": "Smith, Jane and Doe, John",
                        "title": "Raman Metric Validity Under Distribution Shift",
                        "journal": "Journal of Raman Studies",
                        "year": "2020",
                        "doi": "10.1000/jrs.2020.001",
                    },
                },
                {
                    "key": "ng2021",
                    "entry_type": "article",
                    "fields": {
                        "author": "Ng, Alice",
                        "title": "Responsible Release Checklists for Scientific Data",
                        "journal": "Science Data Practice",
                        "year": "2021",
                        "doi": "10.1000/sdp.2021.002",
                    },
                },
            ],
            "sections": sections,
            "claims": [
                {
                    "claim_id": "C001",
                    "manuscript_section": "Abstract",
                    "claim_text": "Model adequacy passed in 5/12 cells.",
                    "evidence_class": "status_evidence",
                    "authority_path": str(self.step8),
                    "authority_sha256": sha256_path(self.step8 / "SHA256SUMS"),
                    "payload_path": str(step8_payload),
                    "row_selector": {"claim_id": "C001"},
                    "allowed_scope": ["status_evidence_only"],
                    "prohibited_scope": [
                        "universally superior",
                        "public leaderboard",
                    ],
                    "verification_state": "verified",
                    "citation_keys": ["smith2020"],
                },
                {
                    "claim_id": "C002",
                    "manuscript_section": "Gate outcomes and reproducibility limitations",
                    "claim_text": "Baseline evidence remains not_powered_for_phase5 at 112/200.",
                    "evidence_class": "status_evidence",
                    "authority_path": str(self.step8),
                    "authority_sha256": sha256_path(self.step8 / "SHA256SUMS"),
                    "payload_path": str(step8_payload),
                    "row_selector": {"claim_id": "C002"},
                    "allowed_scope": ["baseline_only", "status_evidence_only"],
                    "prohibited_scope": ["phase5 confirms external validity"],
                    "verification_state": "verified",
                    "citation_keys": [],
                },
            ],
        }
        write_json(self.config_path, config)
        return self.config_path

    def create_freeze_inputs(self) -> tuple[Path, Path, Path]:
        self.step8.mkdir(parents=True, exist_ok=True)
        write_csv(
            self.step8 / "table1_metric_validity.csv",
            [
                "endpoint_id",
                "protocol_id",
                "protocol_label",
                "tier",
                "perturbation_scope",
                "cluster_kind",
                "cluster_count",
                "state",
                "mse_ag",
                "mse_ag_lower",
                "mse_ag_upper",
                "mse_acc_cross",
                "mse_acc_cross_lower",
                "mse_acc_cross_upper",
                "jointly_favorable_metric_ids",
                "ag_only_favorable_metric_ids",
                "acc_only_favorable_metric_ids",
                "holm_family_id",
                "holm_family_size",
                "claim_scope",
                "source_path",
                "source_sha256",
                "extra_frozen_column",
            ],
            [
                {
                    "endpoint_id": "D5-A",
                    "protocol_id": "a",
                    "protocol_label": "Protocol A",
                    "tier": "full_domain_core",
                    "perturbation_scope": "P8-P12",
                    "cluster_kind": "mineral_class",
                    "cluster_count": "681",
                    "state": "complete",
                    "mse_ag": "0.12",
                    "mse_ag_lower": "0.10",
                    "mse_ag_upper": "0.14",
                    "mse_acc_cross": "0.55",
                    "mse_acc_cross_lower": "0.51",
                    "mse_acc_cross_upper": "0.59",
                    "jointly_favorable_metric_ids": "",
                    "ag_only_favorable_metric_ids": "wasserstein_1_cm1",
                    "acc_only_favorable_metric_ids": "",
                    "holm_family_id": "d5_a_family",
                    "holm_family_size": "24",
                    "claim_scope": "secondary_protocol_specific",
                    "source_path": "results/phase4/final_figures_v1/fake/figure2_alignment_data.csv",
                    "source_sha256": "s8-table1-source",
                    "extra_frozen_column": "retained",
                },
                {
                    "endpoint_id": "D1-B-closed",
                    "protocol_id": "b",
                    "protocol_label": "Protocol B",
                    "tier": "full_domain_core",
                    "perturbation_scope": "P8-P12",
                    "cluster_kind": "patient",
                    "cluster_count": "30",
                    "state": "not_evaluable_failed_alpha0_equivalence",
                    "mse_ag": "",
                    "mse_ag_lower": "",
                    "mse_ag_upper": "",
                    "mse_acc_cross": "",
                    "mse_acc_cross_lower": "",
                    "mse_acc_cross_upper": "",
                    "jointly_favorable_metric_ids": "",
                    "ag_only_favorable_metric_ids": "",
                    "acc_only_favorable_metric_ids": "",
                    "holm_family_id": "d1_b_family",
                    "holm_family_size": "24",
                    "claim_scope": "closed_primary_boundary",
                    "source_path": "results/phase4/final_figures_v1/fake/figure2_alignment_data.csv",
                    "source_sha256": "s8-table1-source",
                    "extra_frozen_column": "retained",
                },
            ],
        )
        write_csv(
            self.step8 / "table_s1_full_alignment.csv",
            [
                "endpoint_id",
                "protocol_id",
                "metric_output_id",
                "ag",
                "ag_lower",
                "ag_upper",
                "acc_cross",
                "acc_cross_lower",
                "acc_cross_upper",
                "d_ag_rejected",
                "d_acc_rejected",
                "panel_state",
                "extra_alignment_column",
            ],
            [
                {
                    "endpoint_id": "D5-A",
                    "protocol_id": "a",
                    "metric_output_id": "wasserstein_1_cm1",
                    "ag": "0.30",
                    "ag_lower": "0.25",
                    "ag_upper": "0.35",
                    "acc_cross": "0.70",
                    "acc_cross_lower": "0.64",
                    "acc_cross_upper": "0.76",
                    "d_ag_rejected": "false",
                    "d_acc_rejected": "true",
                    "panel_state": "complete",
                    "extra_alignment_column": "retained",
                }
            ],
        )
        write_csv(
            self.step8 / "table_s2_protocol_interactions.csv",
            [
                "cell_id",
                "metric_output_id",
                "i_ag_state",
                "i_acc_state",
                "state",
                "extra_interaction_column",
            ],
            [
                {
                    "cell_id": "d5",
                    "metric_output_id": "wasserstein_1_cm1",
                    "i_ag_state": "tested",
                    "i_acc_state": "tested",
                    "state": "complete",
                    "extra_interaction_column": "retained",
                }
            ],
        )
        write_csv(
            self.step8 / "figure3_model_adequacy_data.csv",
            [
                "extractor_id",
                "excitation_stratum",
                "input_count",
                "valid_count",
                "invalid_count",
                "median_reconstruction_nrmse",
                "p95_reconstruction_nrmse",
                "p95_threshold",
                "valid_fraction_state",
                "valid_count_state",
                "median_state",
                "p95_state",
                "gate_state",
                "source_receipt_sha256",
                "extra_figure3_column",
            ],
            [
                {
                    "extractor_id": "airpls",
                    "excitation_stratum": "green_514",
                    "input_count": "28092",
                    "gate_state": "5/12 pass",
                    "valid_count": "28092",
                    "invalid_count": "0",
                    "median_reconstruction_nrmse": "0.11",
                    "p95_reconstruction_nrmse": "0.30",
                    "p95_threshold": "0.25",
                    "valid_fraction_state": "pass",
                    "valid_count_state": "pass",
                    "median_state": "pass",
                    "p95_state": "fail",
                    "source_receipt_sha256": "figure3-source",
                    "extra_figure3_column": "retained",
                }
            ],
        )
        write_csv(
            self.step8 / "table2_method_evidence.csv",
            [
                "evidence_id",
                "task_line",
                "system_id",
                "family_id",
                "endpoint_id",
                "protocol_id",
                "cohort_id",
                "evidence_component",
                "metric_output_id",
                "estimate",
                "interval_lower",
                "interval_upper",
                "preferred_direction",
                "state",
                "reason_code",
                "phase5_power_state",
                "source_path",
                "source_sha256",
                "extra_method_column",
            ],
            [
                {
                    "evidence_id": "baseline_family_projection",
                    "task_line": "baseline_correction",
                    "system_id": "family_projection",
                    "family_id": "all_families",
                    "endpoint_id": "availability",
                    "protocol_id": "availability",
                    "cohort_id": "all",
                    "evidence_component": "availability",
                    "metric_output_id": "availability",
                    "estimate": "112/200",
                    "interval_lower": "",
                    "interval_upper": "",
                    "preferred_direction": "not_applicable",
                    "state": "complete",
                    "reason_code": "",
                    "phase5_power_state": "not_powered_for_phase5",
                    "source_path": "results/phase6/baseline_evidence_v1/fake/family_projection.csv",
                    "source_sha256": "table2-source",
                    "extra_method_column": "retained",
                }
            ],
        )
        write_csv(
            self.step8 / "table_s3_system_status.csv",
            [
                "task_line",
                "system_id",
                "family_id",
                "planned_K",
                "runnable_K",
                "coverage_promoted_K",
                "coverage_state",
                "direct_gt_state",
                "downstream_state",
                "reference_free_state",
                "half_split_state",
                "phase5_eligible_K",
                "phase5_power_state",
                "publication_disposition",
                "reason_code",
                "source_path",
                "source_sha256",
                "extra_status_column",
            ],
            [
                {
                    "task_line": "baseline_correction",
                    "system_id": "airpls_family_projection",
                    "family_id": "airpls",
                    "planned_K": "15",
                    "runnable_K": "15",
                    "coverage_promoted_K": "15",
                    "coverage_state": "complete",
                    "direct_gt_state": "not_evaluated",
                    "downstream_state": "complete",
                    "reference_free_state": "complete",
                    "half_split_state": "complete",
                    "phase5_eligible_K": "15",
                    "phase5_power_state": "not_powered_for_phase5",
                    "publication_disposition": "status_only",
                    "reason_code": "",
                    "source_path": "reports/phase6/step03_baseline_evidence.md",
                    "source_sha256": "table-s3-source",
                    "extra_status_column": "retained",
                }
            ],
        )
        write_csv(
            self.step8 / "figure4_gate_nodes.csv",
            ["node_id", "label", "state", "authority_path", "authority_sha256", "extra_gate_column"],
            [
                {
                    "node_id": "phase0_5_original_gate",
                    "label": "Phase 0.5 original confirmatory gate",
                    "state": "not_met",
                    "authority_path": "reports/phase05/phase05_internal_screening_decision.md",
                    "authority_sha256": "phase05-source",
                    "extra_gate_column": "retained",
                },
                {
                    "node_id": "phase2_model_adequacy",
                    "label": "Phase 2 model adequacy",
                    "state": "5/12 pass; 7/12 fail",
                    "authority_path": "results/phase2/background_fit/fake",
                    "authority_sha256": "phase2-source",
                    "extra_gate_column": "retained",
                },
                {
                    "node_id": "phase3_methods",
                    "label": "Phase 3 methods summary",
                    "state": "baseline 112/200 and 8/10 not_powered_for_phase5; denoising/peak applicability_only; DL 0 runnable",
                    "authority_path": "reports/phase6/step01_publication_package_design.md",
                    "authority_sha256": "phase3-source",
                    "extra_gate_column": "retained",
                },
                {
                    "node_id": "phase5",
                    "label": "Phase 5",
                    "state": "not_powered_for_phase5_not_run",
                    "authority_path": "reports/phase4/step38_phase4_closure_synthesis.md",
                    "authority_sha256": "phase5-source",
                    "extra_gate_column": "retained",
                },
            ],
        )
        write_json(
            self.step8 / "manifest.json",
            {
                "artifact_type": "phase6_publication_core_fixture",
                "payload_files": [
                    "table1_metric_validity.csv",
                    "table_s1_full_alignment.csv",
                    "table_s2_protocol_interactions.csv",
                    "figure3_model_adequacy_data.csv",
                    "table2_method_evidence.csv",
                    "table_s3_system_status.csv",
                    "figure4_gate_nodes.csv",
                ],
                "status": "complete",
            },
        )
        write_json(self.step8 / "complete.json", {"status": "complete"})
        rewrite_sha256sums(self.step8)

        self.step9.mkdir(parents=True, exist_ok=True)
        write_csv(
            self.step9 / "release_matrix.csv",
            ["artifact_id", "disposition", "repository_license", "source_license", "reason"],
            [
                {
                    "artifact_id": "rruff_record_level_derivatives",
                    "disposition": "local_only_pending_redistribution_review",
                    "repository_license": "pending_owner_license_selection",
                    "source_license": "redistribution_unconfirmed",
                    "reason": "rruff redistribution unresolved",
                },
                {
                    "artifact_id": "leaderboard",
                    "disposition": "excluded",
                    "repository_license": "pending_owner_license_selection",
                    "source_license": "not_applicable",
                    "reason": "no licensed hidden ground truth",
                },
            ],
        )
        (self.step9 / "data_card.md").write_text(
            "Ethics/RAI and data/code availability remain bounded by pending owner license and redistribution decisions.\n"
            "Croissant metadata remains metadata-only and does not authorize redistribution of restricted records.\n",
            encoding="utf-8",
        )
        write_json(
            self.step9 / "limitations.jsonl",
            [
                {
                    "limitation_id": "licenses",
                    "state": "pending_owner_license_and_redistribution_decisions",
                }
            ],
        )
        write_json(
            self.step9 / "manifest.json",
            {
                "artifact_type": "phase6_release_metadata_fixture",
                "payload_files": ["release_matrix.csv", "data_card.md", "limitations.jsonl"],
                "status": "complete",
            },
        )
        write_json(self.step9 / "complete.json", {"status": "complete"})
        rewrite_sha256sums(self.step9)

        self.step7_report.parent.mkdir(parents=True, exist_ok=True)
        self.step7_report.write_text(
            "Step 7 robustness and leakage execution is not yet available in the real workspace.\n",
            encoding="utf-8",
        )
        return self.step8, self.step9, self.step7_report


class Phase6ManuscriptTest(unittest.TestCase):
    def test_public_surface_exists(self) -> None:
        self.assertTrue(issubclass(Phase6ManuscriptError, ValueError))
        self.assertTrue(callable(load_phase6_manuscript_config))
        self.assertTrue(callable(build_phase6_manuscript))
        self.assertTrue(callable(verify_phase6_manuscript))
        self.assertTrue(callable(Phase6ManuscriptSummary))
        self.assertTrue(callable(cli_module.main))

    def test_valid_synthetic_build_lints_and_writes_claim_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            summary = build_phase6_manuscript(config_path, fixture.output_root)
            self.assertEqual(summary.status, "complete")
            self.assertTrue(summary.run_id.startswith("phase6-manuscript-"))
            self.assertTrue((summary.path / "paper" / "manuscript.md").is_file())
            self.assertTrue((summary.path / "paper" / "references.bib").is_file())
            self.assertTrue((summary.path / "paper" / "claims_matrix.csv").is_file())
            manuscript = (summary.path / "paper" / "manuscript.md").read_text(encoding="utf-8")
            self.assertIn(FIXED_TITLE, manuscript)
            for heading in SECTION_HEADINGS:
                self.assertIn(f"## {heading}", manuscript)
            self.assertIn("Model adequacy passed in 5/12 cells. [@smith2020]", manuscript)
            receipt = json.loads((summary.path / "claim_lint_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["claim_count"], 2)
            self.assertEqual(receipt["citation_keys"], ["ng2021", "smith2020"])
            matrix_rows = read_csv(summary.path / "paper" / "claims_matrix.csv")
            self.assertEqual(
                list(matrix_rows[0]),
                [
                    "claim_id",
                    "manuscript_section",
                    "claim_text",
                    "evidence_class",
                    "authority_path",
                    "authority_sha256",
                    "payload_path",
                    "row_selector",
                    "allowed_scope",
                    "prohibited_scope",
                    "verification_state",
                ],
            )
            self.assertEqual(matrix_rows[0]["claim_id"], "C001")
            verify_summary = verify_phase6_manuscript(summary.path)
            self.assertEqual(verify_summary.run_id, summary.run_id)

    def test_manuscript_is_substantive_and_honest_about_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            summary = build_phase6_manuscript(config_path, fixture.output_root)
            manuscript = (summary.path / "paper" / "manuscript.md").read_text(encoding="utf-8")
            normalized = manuscript.lower()
            self.assertGreaterEqual(len(manuscript.split()), 2500)
            required_phrases = (
                "methods",
                "results",
                "discussion",
                "reproducibility",
                "limitations",
                "gate failures",
                "not powered",
                "no universal metric",
                "redistribution",
                "data/code availability",
                "phase 5 was not run",
                "journal-neutral",
                "claims_matrix",
            )
            for phrase in required_phrases:
                self.assertIn(phrase, normalized)

    def test_missing_evidence_or_non_authoritative_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["claims"][0]["payload_path"] = str(Path(root) / "missing.csv")
            write_json(config_path, config)
            with self.assertRaisesRegex(Phase6ManuscriptError, "missing evidence|non-authoritative"):
                build_phase6_manuscript(config_path, fixture.output_root)

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["authorities"]["phase4_closure"]["sha256"] = "0" * 64
            write_json(config_path, config)
            with self.assertRaisesRegex(Phase6ManuscriptError, "authority hash mismatch"):
                build_phase6_manuscript(config_path, fixture.output_root)

    def test_row_selector_and_numeric_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            step8_payload = Path(config["claims"][0]["payload_path"])
            write_csv(
                step8_payload,
                ["claim_id", "value", "state", "scope"],
                [
                    {"claim_id": "C001", "value": "5/12", "state": "complete", "scope": "a"},
                    {"claim_id": "C001", "value": "5/12", "state": "complete", "scope": "b"},
                ],
            )
            rewrite_sha256sums(fixture.step8)
            config["claims"][0]["authority_sha256"] = sha256_path(fixture.step8 / "SHA256SUMS")
            write_json(config_path, config)
            with self.assertRaisesRegex(Phase6ManuscriptError, "ambiguous row selector"):
                build_phase6_manuscript(config_path, fixture.output_root)

            fixture = ManuscriptFixture(Path(root) / "second")
            config_path = fixture.create()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            step8_payload = Path(config["claims"][0]["payload_path"])
            write_csv(
                step8_payload,
                ["claim_id", "value", "state", "scope"],
                [
                    {"claim_id": "C001", "value": "4/12", "state": "complete", "scope": "protocol_specific"},
                    {"claim_id": "C002", "value": "112/200", "state": "not_powered_for_phase5", "scope": "baseline_only"},
                ],
            )
            rewrite_sha256sums(fixture.step8)
            config["claims"][0]["authority_sha256"] = sha256_path(fixture.step8 / "SHA256SUMS")
            write_json(config_path, config)
            with self.assertRaisesRegex(Phase6ManuscriptError, "numeric token"):
                build_phase6_manuscript(config_path, fixture.output_root)

    def test_unresolved_citation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["sections"][0]["citation_keys"] = ["missing2024"]
            write_json(config_path, config)
            with self.assertRaisesRegex(Phase6ManuscriptError, "unresolved citation"):
                build_phase6_manuscript(config_path, fixture.output_root)

    def test_prohibited_claim_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["claims"][0]["claim_text"] = "W1 is universally superior to MSE."
            write_json(config_path, config)
            with self.assertRaisesRegex(Phase6ManuscriptError, "prohibited-scope wording"):
                build_phase6_manuscript(config_path, fixture.output_root)

    def test_deterministic_bytes_cli_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = ManuscriptFixture(Path(root))
            config_path = fixture.create()
            first = build_phase6_manuscript(config_path, fixture.output_root / "a")
            second = build_phase6_manuscript(config_path, fixture.output_root / "b")
            self.assertEqual(artifact_files(first.path), artifact_files(second.path))

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "run_phase6_manuscript.py"),
                    "verify",
                    "--run-path",
                    str(first.path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(first.run_id, result.stdout)

            manuscript_path = first.path / "paper" / "manuscript.md"
            manuscript_path.write_text(
                manuscript_path.read_text(encoding="utf-8") + "\nTamper.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Phase6ManuscriptVerificationError, "byte mismatch"):
                verify_phase6_manuscript(first.path)

    def test_freeze_cli_requires_step7_and_writes_canonical_relative_config(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "results" / "phase6")) as root:
            workspace = Path(root)
            fixture = ManuscriptFixture(workspace / "freeze-fixture")
            step8_path, step9_path, step7_path = fixture.create_freeze_inputs()
            output_config = ROOT / "experiments" / "phase6" / "configs" / "manuscript_v1.json"
            original_bytes = output_config.read_bytes() if output_config.exists() else None
            try:
                missing_step7 = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "freeze_phase6_manuscript_config.py"),
                        "--step8-run",
                        str(step8_path),
                        "--step9-run",
                        str(step9_path),
                    ],
                    cwd=str(ROOT),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(missing_step7.returncode, 0)
                self.assertRegex(missing_step7.stderr, "step7-report|required")

                success = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "freeze_phase6_manuscript_config.py"),
                        "--step8-run",
                        str(step8_path),
                        "--step9-run",
                        str(step9_path),
                        "--step7-report",
                        str(step7_path),
                    ],
                    cwd=str(ROOT),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertIn("experiments/phase6/configs/manuscript_v1.json", success.stdout)
                config = json.loads(output_config.read_text(encoding="utf-8"))
                self.assertEqual(
                    output_config.read_bytes(),
                    canonical(config),
                )
                self.assertEqual(config["schema_version"], "phase6-manuscript-v1")
                self.assertEqual(config["title"], FIXED_TITLE)
                self.assertEqual(config["renderer_state"], "deferred_pending_locked_renderer_and_venue_choice")
                self.assertEqual(config["source_revision_status"], "unavailable_no_valid_git_repository")
                self.assertEqual(len(config["sections"]), 11)
                self.assertEqual(
                    tuple(section["heading"] for section in config["sections"]),
                    SECTION_HEADINGS,
                )
                self.assertEqual(
                    set(config["authorities"]),
                    {
                        "step8_publication_core",
                        "step9_release_metadata",
                        "step7_appendix_execution_report",
                        "phase4_closure",
                        "step1_design",
                        "step3_baseline",
                        "step4_denoising",
                        "step5_peak",
                        "step6_appendix",
                    },
                )
                for key, authority in config["authorities"].items():
                    self.assertFalse(str(authority["path"]).startswith("/"), key)
                self.assertEqual(
                    config["authorities"]["step8_publication_core"]["path"],
                    step8_path.relative_to(ROOT).as_posix(),
                )
                self.assertEqual(
                    config["authorities"]["step9_release_metadata"]["path"],
                    step9_path.relative_to(ROOT).as_posix(),
                )
                self.assertEqual(
                    config["authorities"]["step7_appendix_execution_report"]["path"],
                    step7_path.relative_to(ROOT).as_posix(),
                )
                claim_ids = {claim["claim_id"] for claim in config["claims"]}
                self.assertTrue(
                    {
                        "table1-d5-a-cluster-count",
                        "table1-d1-b-closed-state",
                        "figure4-phase2-5-pass-7-fail",
                        "figure4-phase3-baseline-112-200",
                        "figure4-phase3-dl-zero-runnable",
                        "figure4-phase05-not-met",
                        "step9-rruff-local-only",
                    }.issubset(claim_ids)
                )
                first_claim = next(claim for claim in config["claims"] if claim["claim_id"] == "figure4-phase2-5-pass-7-fail")
                self.assertEqual(first_claim["authority_path"], step8_path.relative_to(ROOT).as_posix())
                self.assertEqual(
                    first_claim["payload_path"],
                    (step8_path / "figure4_gate_nodes.csv").relative_to(ROOT).as_posix(),
                )
                self.assertEqual(
                    first_claim["row_selector"],
                    {"node_id": "phase2_model_adequacy"},
                )
                self.assertIn("5 pass/7 fail", first_claim["claim_text"])
                manuscript_text = build_phase6_manuscript(output_config, workspace / "manuscript-out").path.joinpath("paper", "manuscript.md").read_text(encoding="utf-8")
                self.assertIn("This journal-neutral manuscript asks when Raman fidelity metrics track downstream utility", manuscript_text)
                self.assertIn("Phase 5 stays not run and not powered", manuscript_text)
                self.assertIn("paper/claims_matrix.csv", manuscript_text)
                self.assertGreater(len(manuscript_text.split()), 180)
                self.assertEqual(
                    {ref["key"] for ref in config["references"]},
                    {
                        "graham2014",
                        "deutsch2021",
                        "steiger1980",
                        "georgiev2024ramanspy",
                    },
                )
                load_phase6_manuscript_config(output_config)
            finally:
                if original_bytes is None:
                    if output_config.exists():
                        output_config.unlink()
                else:
                    output_config.write_bytes(original_bytes)

    def test_freeze_cli_propagates_owner_deferred_step7_boundary_into_sections(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(ROOT / "results" / "phase6")) as root:
            workspace = Path(root)
            fixture = ManuscriptFixture(workspace / "freeze-fixture")
            step8_path, step9_path, step7_path = fixture.create_freeze_inputs()
            step7_path.write_text(
                "---\n"
                "state: deferred_by_owner\n"
                "outcome_execution: not_admitted\n"
                "scientific_claims: not_admitted\n"
                "---\n"
                "\n"
                "# Phase 6 Step 7 Appendix Audits\n"
                "\n"
                "Owner-directed defer for Step 7 appendix sensitivity and leakage audits.\n",
                encoding="utf-8",
            )
            output_config = ROOT / "experiments" / "phase6" / "configs" / "manuscript_v1.json"
            original_bytes = output_config.read_bytes() if output_config.exists() else None
            try:
                freeze_cli_module.freeze_phase6_manuscript_config(
                    step8_path,
                    step9_path,
                    step7_path,
                )
                config = json.loads(output_config.read_text(encoding="utf-8"))
                evidence_section = next(
                    section
                    for section in config["sections"]
                    if section["heading"] == "Evidence/fallback map and Figure 4"
                )
                paragraphs = "\n".join(evidence_section["paragraphs"])
                self.assertIn("deferred by owner", paragraphs.lower())
                self.assertIn("not run", paragraphs.lower())
                self.assertIn("not admitted", paragraphs.lower())
                self.assertRegex(
                    paragraphs.lower(),
                    r"no step 7 sensitivity or leakage result .* support",
                )
                self.assertNotIn(
                    "Step 7 robustness and leakage execution is treated as its own authority",
                    paragraphs,
                )
            finally:
                if original_bytes is None:
                    if output_config.exists():
                        output_config.unlink()
                else:
                    output_config.write_bytes(original_bytes)


if __name__ == "__main__":
    unittest.main()
