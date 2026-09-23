from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_manuscript import (  # noqa: E402
    DEFAULT_CONFIG,
    load_phase6_manuscript_config,
    repo_relative_path,
)


FIXED_TITLE = (
    "When Do Raman Fidelity Metrics Track Downstream Utility? "
    "A Task-, Perturbation-, and Protocol-Aware Evaluation"
)
SECTION_HEADINGS = (
    "Abstract",
    "Introduction and the metric-validity problem",
    "Data sources, provenance, and preprocessing-state taxonomy",
    "Metrics, P1-P12 operators, task-native endpoints, Protocol A/B, AG, Acc-cross, inference, and prospective gates",
    "Gate outcomes and reproducibility limitations",
    "P8-P12 response results and Figures 1/2",
    "Protocol-dependent metric alignment and interactions",
    "Phase-2 adequacy failure and Figure 3",
    "Evidence/fallback map and Figure 4",
    "Discussion, limitations, ethics/RAI, and data/code availability",
    "References plus supplementary methods/tables",
)


class FreezePhase6ManuscriptConfigError(ValueError):
    pass


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


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_artifact_dir(path: Path, *, required_payloads: Sequence[str]) -> str:
    if not path.is_dir():
        raise FreezePhase6ManuscriptConfigError(f"missing artifact directory: {path}")
    complete = path / "complete.json"
    ledger = path / "SHA256SUMS"
    manifest = path / "manifest.json"
    if not complete.is_file() or not ledger.is_file() or not manifest.is_file():
        raise FreezePhase6ManuscriptConfigError(f"artifact missing complete terminal or checksum ledger: {path}")
    manifest_doc = _load_json(manifest)
    if manifest_doc.get("status") != "complete":
        raise FreezePhase6ManuscriptConfigError(f"artifact not complete: {path}")
    listed = set(manifest_doc.get("payload_files", ()))
    missing = [name for name in required_payloads if name not in listed or not (path / name).is_file()]
    if missing:
        raise FreezePhase6ManuscriptConfigError(f"artifact missing required payloads: {', '.join(missing)}")
    entries = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise FreezePhase6ManuscriptConfigError(f"invalid checksum ledger: {path}")
        entries.append((parts[0], parts[1]))
    actual = sorted(
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    )
    listed_names = [name for _, name in entries]
    if len(listed_names) != len(set(listed_names)) or set(listed_names) != set(actual):
        raise FreezePhase6ManuscriptConfigError(f"checksum inventory mismatch: {path}")
    for digest, name in entries:
        if _sha_file(path / name) != digest:
            raise FreezePhase6ManuscriptConfigError(f"checksum mismatch: {path / name}")
    return _sha_file(ledger)


def _validate_step7_report(path: Path) -> str:
    if not path.is_file():
        raise FreezePhase6ManuscriptConfigError("--step7-report must point to an existing report")
    return _sha_file(path)


def _parse_step7_report_tokens(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 5 or lines[0].strip() != "---":
        return {}
    front_matter: dict[str, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        front_matter[key.strip()] = value.strip()
    return front_matter


def _bibliography_entries() -> list[dict[str, object]]:
    plan = (ROOT / "raman_preproc_benchmark_plan_v2.md").read_text(encoding="utf-8")
    project_log = (ROOT / "PROJECT_LOG.md").read_text(encoding="utf-8")
    if "10.3115/v1/D14-1020" not in plan:
        raise FreezePhase6ManuscriptConfigError("missing DOI-backed bibliography fact for graham2014")
    if "10.1162/tacl_a_00417" not in plan:
        raise FreezePhase6ManuscriptConfigError("missing DOI-backed bibliography fact for deutsch2021")
    if "10.1037/0033-2909.87.2.245" not in plan:
        raise FreezePhase6ManuscriptConfigError("missing DOI-backed bibliography fact for steiger1980")
    if "10.1021/acs.analchem.4c00383" not in plan:
        raise FreezePhase6ManuscriptConfigError("missing DOI-backed bibliography fact for georgiev2024ramanspy")
    if "10.5281/zenodo.10779223" not in project_log:
        raise FreezePhase6ManuscriptConfigError("missing DOI-backed bibliography fact for sugar mixtures dataset")
    return [
        {
            "key": "graham2014",
            "entry_type": "article",
            "fields": {
                "author": "Graham, Yvette and Baldwin, Timothy",
                "title": "Testing for Significance of Increased Correlation with Human Judgment",
                "journal": "EMNLP 2014",
                "year": "2014",
                "doi": "10.3115/v1/D14-1020",
            },
        },
        {
            "key": "deutsch2021",
            "entry_type": "article",
            "fields": {
                "author": "Deutsch, Daniel and Dror, Rotem and Roth, Dan",
                "title": "A Statistical Analysis of Summarization Evaluation Metrics Using Resampling Methods",
                "journal": "Transactions of the Association for Computational Linguistics",
                "year": "2021",
                "doi": "10.1162/tacl_a_00417",
            },
        },
        {
            "key": "steiger1980",
            "entry_type": "article",
            "fields": {
                "author": "Steiger, James H.",
                "title": "Tests for Comparing Elements of a Correlation Matrix",
                "journal": "Psychological Bulletin",
                "year": "1980",
                "doi": "10.1037/0033-2909.87.2.245",
            },
        },
        {
            "key": "georgiev2024ramanspy",
            "entry_type": "article",
            "fields": {
                "author": "Georgiev, Dimitar and others",
                "title": "RamanSPy: An open-source Python package for Raman spectroscopy analysis",
                "journal": "Analytical Chemistry",
                "year": "2024",
                "doi": "10.1021/acs.analchem.4c00383",
            },
        },
    ]


def _parse_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    rows = _load_csv(path)
    if not rows:
        raise FreezePhase6ManuscriptConfigError(f"required payload empty: {path}")
    missing = [field for field in expected_fields if field not in rows[0]]
    if missing:
        raise FreezePhase6ManuscriptConfigError(f"unexpected schema for {path.name}")
    return rows


def _append_claim(
    claims: list[dict[str, object]],
    *,
    claim_id: str,
    manuscript_section: str,
    claim_text: str,
    evidence_class: str,
    authority_path: Path,
    authority_sha256: str,
    payload_path: Path,
    row_selector: dict[str, object],
    allowed_scope: list[str],
    prohibited_scope: list[str],
    citation_keys: list[str] | None = None,
) -> None:
    selected_rows = _load_csv(payload_path)
    matches = [
        row
        for row in selected_rows
        if all(row.get(str(key)) == str(value) for key, value in row_selector.items())
    ]
    if len(matches) != 1:
        raise FreezePhase6ManuscriptConfigError(f"claim selector is not unambiguous for {claim_id}")
    selected_text = json.dumps(matches[0], sort_keys=True, separators=(",", ":"))
    for token in re.findall(r"\b\d+(?:/\d+)?(?:\.\d+)?\b", claim_text):
        if token not in selected_text:
            raise FreezePhase6ManuscriptConfigError(f"claim numeric token missing from selected row for {claim_id}")
    claims.append(
        {
            "claim_id": claim_id,
            "manuscript_section": manuscript_section,
            "claim_text": claim_text,
            "evidence_class": evidence_class,
            "authority_path": repo_relative_path(authority_path),
            "authority_sha256": authority_sha256,
            "payload_path": repo_relative_path(payload_path),
            "row_selector": row_selector,
            "allowed_scope": allowed_scope,
            "prohibited_scope": prohibited_scope,
            "verification_state": "verified",
            "citation_keys": list(citation_keys or []),
        }
    )


def _extract_claims_from_step8_and_step9(
    step8_run: Path,
    step8_sha256: str,
    step9_run: Path,
    step9_sha256: str,
) -> list[dict[str, object]]:
    _parse_csv(
        step8_run / "table1_metric_validity.csv",
        (
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
        ),
    )
    _parse_csv(
        step8_run / "table_s1_full_alignment.csv",
        (
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
        ),
    )
    _parse_csv(
        step8_run / "table_s2_protocol_interactions.csv",
        (
            "cell_id",
            "metric_output_id",
            "i_ag_state",
            "i_acc_state",
            "state",
        ),
    )
    _parse_csv(
        step8_run / "table2_method_evidence.csv",
        (
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
        ),
    )
    _parse_csv(
        step8_run / "table_s3_system_status.csv",
        (
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
        ),
    )
    _parse_csv(
        step8_run / "figure3_model_adequacy_data.csv",
        (
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
        ),
    )
    _parse_csv(
        step8_run / "figure4_gate_nodes.csv",
        ("node_id", "label", "state", "authority_path", "authority_sha256"),
    )
    _parse_csv(
        step9_run / "release_matrix.csv",
        ("artifact_id", "disposition", "repository_license", "source_license", "reason"),
    )

    claims: list[dict[str, object]] = []
    _append_claim(
        claims,
        claim_id="table1-d5-a-cluster-count",
        manuscript_section="P8-P12 response results and Figures 1/2",
        claim_text="The D5-A main-table row retains 681 mineral-class clusters.",
        evidence_class="status_evidence",
        authority_path=step8_run,
        authority_sha256=step8_sha256,
        payload_path=step8_run / "table1_metric_validity.csv",
        row_selector={"endpoint_id": "D5-A", "protocol_id": "a"},
        allowed_scope=["task_specific", "status_evidence_only"],
        prohibited_scope=["universally superior", "public leaderboard"],
    )
    _append_claim(
        claims,
        claim_id="table1-d1-b-closed-state",
        manuscript_section="Gate outcomes and reproducibility limitations",
        claim_text="The D1-B row remains not_evaluable_failed_alpha0_equivalence.",
        evidence_class="status_evidence",
        authority_path=step8_run,
        authority_sha256=step8_sha256,
        payload_path=step8_run / "table1_metric_validity.csv",
        row_selector={"endpoint_id": "D1-B-closed", "protocol_id": "b"},
        allowed_scope=["closed_cell_only"],
        prohibited_scope=["primary hypothesis passed"],
    )
    _append_claim(
        claims,
        claim_id="figure4-phase2-5-pass-7-fail",
        manuscript_section="Phase-2 adequacy failure and Figure 3",
        claim_text="The Phase-2 adequacy gate remains 5 pass/7 fail.",
        evidence_class="status_evidence",
        authority_path=step8_run,
        authority_sha256=step8_sha256,
        payload_path=step8_run / "figure4_gate_nodes.csv",
        row_selector={"node_id": "phase2_model_adequacy"},
        allowed_scope=["figure3_gate_only"],
        prohibited_scope=["every possible generator fails"],
    )
    _append_claim(
        claims,
        claim_id="figure4-phase3-baseline-112-200",
        manuscript_section="Gate outcomes and reproducibility limitations",
        claim_text="The Phase 3 methods summary keeps baseline at 112/200 not_powered_for_phase5.",
        evidence_class="status_evidence",
        authority_path=step8_run,
        authority_sha256=step8_sha256,
        payload_path=step8_run / "figure4_gate_nodes.csv",
        row_selector={"node_id": "phase3_methods"},
        allowed_scope=["baseline_only", "status_evidence_only"],
        prohibited_scope=["phase5 confirms external validity"],
    )
    _append_claim(
        claims,
        claim_id="figure4-phase3-dl-zero-runnable",
        manuscript_section="Gate outcomes and reproducibility limitations",
        claim_text="The Phase 3 methods summary keeps dl 0 runnable.",
        evidence_class="status_evidence",
        authority_path=step8_run,
        authority_sha256=step8_sha256,
        payload_path=step8_run / "figure4_gate_nodes.csv",
        row_selector={"node_id": "phase3_methods"},
        allowed_scope=["dl_audit_only"],
        prohibited_scope=["locally reproduced DL-performance claim"],
    )
    _append_claim(
        claims,
        claim_id="figure4-phase05-not-met",
        manuscript_section="Evidence/fallback map and Figure 4",
        claim_text="The evidence-flow map keeps the original Phase 0.5 confirmatory gate at not_met.",
        evidence_class="status_evidence",
        authority_path=step8_run,
        authority_sha256=step8_sha256,
        payload_path=step8_run / "figure4_gate_nodes.csv",
        row_selector={"node_id": "phase0_5_original_gate"},
        allowed_scope=["gate_map_only"],
        prohibited_scope=["original confirmatory gate passed"],
    )
    _append_claim(
        claims,
        claim_id="step9-rruff-local-only",
        manuscript_section="Discussion, limitations, ethics/RAI, and data/code availability",
        claim_text="Release metadata keeps rruff_record_level_derivatives at local_only_pending_redistribution_review.",
        evidence_class="status_evidence",
        authority_path=step9_run,
        authority_sha256=step9_sha256,
        payload_path=step9_run / "release_matrix.csv",
        row_selector={"artifact_id": "rruff_record_level_derivatives"},
        allowed_scope=["release_metadata_only"],
        prohibited_scope=["redistribution approved"],
        citation_keys=["georgiev2024ramanspy"],
    )
    return claims


def _section_claim_ids(claims: list[dict[str, object]], section: str) -> list[str]:
    return [str(claim["claim_id"]) for claim in claims if claim["manuscript_section"] == section]


def _step7_dependency_paragraph(step7_report_tokens: dict[str, str]) -> str:
    if (
        step7_report_tokens.get("state") == "deferred_by_owner"
        and step7_report_tokens.get("outcome_execution") == "not_admitted"
        and step7_report_tokens.get("scientific_claims") == "not_admitted"
    ):
        return (
            "Step 7 appendix audits were deferred by owner; the sensitivity and leakage work was not run for the "
            "present manuscript authority surface, its execution outcome is not admitted, its scientific claims are "
            "not admitted, and no Step 7 sensitivity or leakage result is admitted or available to support any "
            "manuscript claim."
        )
    return (
        "Step 7 robustness and leakage work remains a dependency-governed extension and is referenced for manuscript "
        "dependency only through the supplied report path."
    )


def _sections_from_claims(
    claims: list[dict[str, object]],
    *,
    step7_report_tokens: dict[str, str],
) -> list[dict[str, object]]:
    sections = []
    for heading in SECTION_HEADINGS:
        paragraphs = []
        citation_keys: list[str] = []
        claim_ids = _section_claim_ids(claims, heading)
        if heading == "Abstract":
            paragraphs = [
                "This journal-neutral manuscript asks when Raman fidelity metrics track downstream utility under task, perturbation, and protocol shifts.",
                "It distinguishes numerical results, status evidence, and unavailable evidence and keeps closed cells visibly closed rather than silently repaired.",
            ]
            citation_keys = ["graham2014", "deutsch2021", "steiger1980"]
        elif heading == "Introduction and the metric-validity problem":
            paragraphs = [
                "The central question is bounded metric validity: which fidelity summaries preserve useful downstream ordering information, and where that alignment breaks.",
                "The framing keeps resampling-based significance comparison and dependent-correlation cautions explicit without pretending that manuscript assembly reopens any inferential family.",
            ]
            citation_keys = ["graham2014", "deutsch2021", "steiger1980"]
        elif heading == "Data sources, provenance, and preprocessing-state taxonomy":
            paragraphs = [
                "Data provenance remains source-specific, redistribution-aware, and tied to explicit preprocessing-state labels rather than a single homogeneous Raman corpus.",
                "The local package keeps source DOI evidence and rebuild boundaries explicit, but it does not turn source provenance into an unrestricted redistribution grant.",
            ]
            citation_keys = ["georgiev2024ramanspy"]
        elif heading == "Metrics, P1-P12 operators, task-native endpoints, Protocol A/B, AG, Acc-cross, inference, and prospective gates":
            paragraphs = [
                "Task-native endpoints, Protocol A/B distinctions, AG, Acc-cross, and preregistered gates remain frozen from the executed Phase 4 design and later Step 1 publication contract.",
                "Protocol labels are preserved as different adaptation regimes, so matched-reference D5 and condition-specific refit settings are not collapsed into one generic intervention.",
            ]
            citation_keys = ["graham2014", "deutsch2021", "steiger1980"]
        elif heading == "Gate outcomes and reproducibility limitations":
            paragraphs = [
                "Gate outcomes remain the spine of the paper: confirmatory success stays non-evaluable where coverage or alpha-zero prerequisites failed, and later continuation remains separately authorized internal screening.",
                "Phase 5 stays not run and not powered, while deep-learning and other unavailable lines remain explicit reproducibility findings rather than silent omissions.",
            ]
        elif heading == "P8-P12 response results and Figures 1/2":
            paragraphs = [
                "Secondary Phase 4 result discussion stays within the publication-core authority rows and uses claim-linked sentences for every reported numerical or status fact.",
                "This keeps the journal-neutral manuscript substantive without turning the narrative into an unsupported changelog of all published rows.",
            ]
        elif heading == "Protocol-dependent metric alignment and interactions":
            paragraphs = [
                "Protocol-dependent alignment and interaction discussion is deliberately bounded to endpoint-specific evidence and preserved family structure.",
                "No pooled cross-endpoint family, universal metric winner, or universal MSE failure claim is introduced in this section.",
            ]
        elif heading == "Phase-2 adequacy failure and Figure 3":
            paragraphs = [
                "Phase 2 adequacy failure is carried forward as a gate result and fallback trigger for the paper architecture rather than as a claim that every possible generator or synthetic benchmark route is invalid.",
            ]
        elif heading == "Evidence/fallback map and Figure 4":
            paragraphs = [
                "The evidence/fallback map distinguishes preregistered dependencies from authorized fallback continuations and keeps the original Phase 0.5 gate visibly separate from later internal screening.",
                _step7_dependency_paragraph(step7_report_tokens),
            ]
        elif heading == "Discussion, limitations, ethics/RAI, and data/code availability":
            paragraphs = [
                "Ethics/RAI, data availability, and code availability remain bounded by release-metadata dispositions rather than template ambition or manuscript tone.",
                "Portable rendering remains deferred_pending_locked_renderer_and_venue_choice, and the Step 11 asset surface is referenced only through relative manuscript paths such as paper/claims_matrix.csv and paper/references.bib.",
            ]
            citation_keys = ["georgiev2024ramanspy"]
        else:
            paragraphs = [
                "References and supplementary material are assembled from authority-linked claims, DOI-backed local bibliography facts, and the explicit manuscript-to-asset paths intended for later Step 11 packaging.",
            ]
            citation_keys = ["graham2014", "deutsch2021", "steiger1980", "georgiev2024ramanspy"]
        sections.append(
            {
                "heading": heading,
                "paragraphs": paragraphs,
                "citation_keys": citation_keys,
                "claim_ids": claim_ids,
            }
        )
    return sections


def _report_authority(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FreezePhase6ManuscriptConfigError(f"required report missing: {relative_path}")
    return {
        "type": "file",
        "path": relative_path,
        "sha256": _sha_file(path),
    }


def freeze_phase6_manuscript_config(step8_run: Path, step9_run: Path, step7_report: Path) -> Path:
    step8_sha256 = _validate_artifact_dir(
        step8_run,
        required_payloads=(
            "table1_metric_validity.csv",
            "table_s1_full_alignment.csv",
            "table_s2_protocol_interactions.csv",
            "figure3_model_adequacy_data.csv",
            "table2_method_evidence.csv",
            "table_s3_system_status.csv",
            "figure4_gate_nodes.csv",
        ),
    )
    step9_sha256 = _validate_artifact_dir(
        step9_run,
        required_payloads=(
            "release_matrix.csv",
            "data_card.md",
            "limitations.jsonl",
        ),
    )
    step7_sha256 = _validate_step7_report(step7_report)
    step7_report_tokens = _parse_step7_report_tokens(step7_report)
    claims = _extract_claims_from_step8_and_step9(step8_run, step8_sha256, step9_run, step9_sha256)
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
                "path": repo_relative_path(step8_run),
            },
            "step9_release_metadata": {
                "type": "artifact_dir",
                "path": repo_relative_path(step9_run),
            },
            "step7_appendix_execution_report": {
                "type": "file",
                "path": repo_relative_path(step7_report),
                "sha256": step7_sha256,
            },
            "phase4_closure": _report_authority("reports/phase4/step38_phase4_closure_synthesis.md"),
            "step1_design": _report_authority("reports/phase6/step01_publication_package_design.md"),
            "step3_baseline": _report_authority("reports/phase6/step03_baseline_evidence.md"),
            "step4_denoising": _report_authority("reports/phase6/step04_denoising_evidence.md"),
            "step5_peak": _report_authority("reports/phase6/step05_peak_evidence.md"),
            "step6_appendix": _report_authority("reports/phase6/step06_appendix_sensitivity_leakage_protocol.md"),
        },
        "references": _bibliography_entries(),
        "sections": _sections_from_claims(claims, step7_report_tokens=step7_report_tokens),
        "claims": claims,
    }
    DEFAULT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG.write_bytes(_canonical(config))
    load_phase6_manuscript_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze canonical Phase 6 manuscript config from verified Step 8/9 artifacts")
    parser.add_argument("--step8-run", required=True)
    parser.add_argument("--step9-run", required=True)
    parser.add_argument("--step7-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    config_path = freeze_phase6_manuscript_config(
        Path(args.step8_run).resolve(),
        Path(args.step9_run).resolve(),
        Path(args.step7_report).resolve(),
    )
    print(config_path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreezePhase6ManuscriptConfigError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
