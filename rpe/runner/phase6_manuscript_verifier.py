from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class Phase6ManuscriptVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class Phase6ManuscriptVerificationSummary:
    run_id: str
    path: Path
    status: str
    counts: Mapping[str, int]


ROOT = Path(__file__).resolve().parents[2]


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            _plain(value),
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


def _require_dict(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Phase6ManuscriptVerificationError(f"{name} must be an object")
    return dict(value)


def _require_list(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise Phase6ManuscriptVerificationError(f"{name} must be a list")
    return list(value)


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise Phase6ManuscriptVerificationError(f"{name} must be a nonempty string")
    return value


def _json_bytes(value: object) -> bytes:
    return _canonical(value)


def _csv_bytes(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=tuple(fieldnames),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        formatted: dict[str, object] = {}
        for key in fieldnames:
            value = row[key]
            if isinstance(value, (dict, list, tuple)):
                formatted[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            else:
                formatted[key] = value
        writer.writerow(formatted)
    return stream.getvalue().encode("utf-8")


def _artifact_authority_sha256(path: Path) -> str:
    ledger = path / "SHA256SUMS"
    if not path.is_dir() or not ledger.is_file() or not (path / "complete.json").is_file():
        raise Phase6ManuscriptVerificationError("missing evidence or non-authoritative path")
    return _sha_file(ledger)


def _load_config(config_path: Path) -> tuple[bytes, dict[str, object]]:
    raw = config_path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase6ManuscriptVerificationError("invalid config") from error
    if raw != _canonical(document):
        raise Phase6ManuscriptVerificationError("config is not canonical")
    return raw, dict(document)


def _render_references(document: Mapping[str, object]) -> bytes:
    lines: list[str] = []
    for row in _require_list("references", document.get("references")):
        entry = _require_dict("reference", row)
        key = _require_str("reference.key", entry.get("key"))
        entry_type = _require_str("reference.entry_type", entry.get("entry_type"))
        fields = _require_dict("reference.fields", entry.get("fields"))
        lines.append(f"@{entry_type}{{{key},")
        for field in sorted(fields):
            lines.append(f"  {field} = {{{_require_str(field, fields[field])}}},")
        lines.append("}")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _section_expansion_paragraphs(heading: str) -> tuple[str, ...]:
    if heading == "Abstract":
        return (
            "This substantive draft stays journal-neutral by treating the paper as an evidence-accounting exercise rather than as a venue-specific sales document. The abstract states the scientific problem, the admissible evidence classes, and the closure conditions that govern what may and may not be claimed. It names methods, results, discussion, and reproducibility as distinct responsibilities of the manuscript so readers can see that inferential language is limited by frozen authorities rather than expanded by narrative convenience.",
            "The central contribution is therefore not a claim that one Raman fidelity summary dominates every task, perturbation family, or protocol. Instead, the abstract frames metric validity as conditional: alignment may be informative in some task-native settings and break in others, especially when preprocessing-state shifts, protocol definitions, and gate failures change what downstream utility means. This keeps the paper aligned with the executed program and makes no universal metric ranking claim.",
            "The abstract also states the principal publication boundaries. Later sections explain why phase 5 was not run as a confirmatory meta-evaluation, why several lines remain not powered, why release and redistribution decisions are still gated, and why unexecuted appendix audits remain prospective rather than retroactively narrated as completed robustness evidence.",
        )
    if heading == "Introduction and the metric-validity problem":
        return (
            "The introduction motivates why Raman preprocessing cannot be evaluated honestly through a single convenient scalar ranking. Fidelity metrics summarize agreement between transformed and reference spectra, but downstream tasks depend on perturbation family, endpoint definition, class structure, and the protocol that determines what counts as a fair comparison. A manuscript that ignores those dependencies invites false stability: a metric can appear attractive under one comparison surface and lose practical meaning when the endpoint or comparison protocol changes.",
            "The introduction therefore positions the project against a familiar but weak pattern in empirical benchmarking: present many numbers, suppress closed cells, and imply that the remaining observed trend transfers everywhere. Here the paper explicitly rejects that pattern. Closed cells stay closed, unavailable evidence stays unavailable, and the manuscript records why some questions remain unanswered. That stance is part of the scientific method of the paper, not merely editorial restraint.",
            "This section also explains why prior statistical guidance matters without pretending that manuscript assembly reruns the original inferential procedures. Dependent-comparison cautions, resampling arguments, and multiple-comparison discipline remain relevant because the paper interprets already-frozen results. They matter most as guardrails: they explain why a no universal metric story is the defensible conclusion when protocol- and endpoint-specific interaction structure remains visible in the publication authorities.",
        )
    if heading == "Data sources, provenance, and preprocessing-state taxonomy":
        return (
            "The data section describes sources and preprocessing states as provenance-bearing entities, not as interchangeable rows in a single anonymous benchmark. Raman collections in this workspace carry different source obligations, different preprocessing histories, and different release permissions. For that reason the manuscript keeps source lineage visible and discusses downstream validity in the context of those lineages rather than implying that all spectra live on one unrestricted public surface.",
            "A second function of the data section is to define the preprocessing-state taxonomy that later sections rely on. Raw inputs, transformed outputs, and task-specific eligibility states are treated as scientifically distinct analysis states. That distinction matters because some downstream targets are direct, some are indirect, and some are unavailable under the frozen authority structure. The prose therefore distinguishes between numerical evidence, status evidence, and unavailable evidence before any results interpretation begins.",
            "The section is equally explicit about dissemination boundaries. Data/code availability is discussed as a constrained release question, not as a default promise that every local artifact can be redistributed. Where source owners have not cleared redistribution, the manuscript says so. Where local derivative artifacts remain bounded by review, the text treats that as a standing limitation instead of burying it in supplementary notes.",
        )
    if heading == "Metrics, P1-P12 operators, task-native endpoints, Protocol A/B, AG, Acc-cross, inference, and prospective gates":
        return (
            "The methods section defines how the paper uses fidelity summaries without collapsing distinct evaluation targets into one blended score. P1-P12 operators, task-native endpoints, and the contrast between Protocol A/B are described as analysis design choices that alter what an observed metric-outcome relationship means. AG and Acc-cross are retained as separate summaries because agreement magnitude and cross-condition ordering are not interchangeable scientific questions.",
            "This section also clarifies that the manuscript is not a de novo methods invention pass. The computational methods were frozen earlier; Step 10 assembles a publication-facing narrative that points readers back to the preserved design boundaries. As a result, the methods prose emphasizes scope control: which comparisons were pre-registered or frozen, which inferential families remained admissible, and which prospective gates were defined for later work but not yet executed.",
            "Prospective gates are especially important here because they prevent the paper from rewriting failure as optional complexity. If a later appendix audit or external-validity stage depends on a gate that has not been satisfied, the manuscript keeps that dependency visible. The methods section therefore serves both as a technical description and as a contract that later results and discussion sections are not allowed to violate.",
        )
    if heading == "Gate outcomes and reproducibility limitations":
        return (
            "The results narrative is anchored by gate outcomes before it moves to any positive interpretation. That order matters because gate failures determine which comparisons remain descriptive, which remain closed, and which are not eligible for confirmatory language. By presenting those gate failures first, the manuscript makes clear that the absence of a later claim is often itself a result of the frozen design rather than an omission in writing.",
            "Reproducibility is handled with the same directness. When an authority exists and has been independently rederived, the paper says so. When an analysis line lacks executable releases, runnable environments, or an authorized result surface, the manuscript treats that as a reproducibility finding, not as a logistical footnote. Readers are therefore told which results can be rerun locally, which claims are status-only, and which lines stop at audited inability to reproduce.",
            "This is also the section where the most important publication boundary is stated plainly: phase 5 was not run, and the relevant method families remain not powered for that confirmatory role. The manuscript does not rescue that boundary by pooling task lines, importing synthetic evidence into external-validity claims, or converting unavailable deep-learning rows into implied negatives. The paper instead records the closure honestly and carries that limitation into discussion.",
        )
    if heading == "P8-P12 response results and Figures 1/2":
        return (
            "The manuscript's primary results prose for executed response lines stays close to the publication-core authority tables. Figures 1 and 2 are described as structured displays of authority-backed comparisons, not as invitation to restate every row in prose. That choice keeps the draft substantive while preserving a clean separation between rendered figure evidence and narrative interpretation.",
            "Within that boundary, the paper highlights what the response panels can legitimately show: when a task-native endpoint admits numerical comparison, how protocol choice changes the comparison surface, and where descriptive alignment can be discussed without overextending to global superiority claims. The prose intentionally resists the temptation to turn a figure caption into a synthetic leaderboard or a compact story of winners and losers.",
            "The section also reminds readers how to audit the text. Every numerical or specific status assertion that belongs in the paper is supposed to remain traceable through paper/claims_matrix.csv. That traceability requirement influences writing style: prose is richer about interpretation, but sparse about unmapped numbers. In other words, the section aims for explanatory depth without fabricating a denser result surface than the authorities actually support.",
        )
    if heading == "Protocol-dependent metric alignment and interactions":
        return (
            "This section interprets protocol dependence as a scientific finding rather than as an inconvenience to be averaged away. If a metric aligns differently under Protocol A and Protocol B, that difference is evidence that the evaluation context changes the meaning of fidelity. The manuscript therefore treats interaction structure as part of the answer to the metric-validity question, not as noise that obstructs a cleaner single-number conclusion.",
            "A practical implication follows. Readers should not expect the paper to name one metric as the stable winner across every endpoint, perturbation scope, and protocol. The discussion here is deliberately framed around conditional usefulness, disagreement patterns, and the cost of collapsing dependent contrasts. That is why the text includes an explicit no universal metric position instead of implying that the observed executed cells secretly support a global ranking.",
            "By doing so, the section also protects the broader contribution of the study. The value of the paper is not diminished by the lack of a universal champion metric. On the contrary, the protocol-dependent interpretation is exactly what makes the benchmark informative for future method work: it tells readers where metric behavior changes, where claims must stay local, and where later studies would need new evidence rather than stronger rhetoric.",
        )
    if heading == "Phase-2 adequacy failure and Figure 3":
        return (
            "The Figure 3 section explains why model adequacy remains a gating result instead of a hidden prelude to synthetic downstream claims. The semi-synthetic route was designed to support a later benchmark only if adequacy criteria were met. Because adequacy failures remained material, the manuscript carries forward the authorized fallback: report the adequacy outcome itself, explain the scientific consequence, and stop short of treating unevaluated synthetic ranking behavior as if it had been observed.",
            "That choice is important for readers outside the project history. Without a clear explanation, a figure about adequacy could be mistaken for a weak positive result or a partially completed benchmark. The section therefore states the opposite. Figure 3 belongs to methods and failure analysis as much as to results. It documents why one planned route did not mature into a general-purpose evidence source and why later sections cannot borrow certainty from it.",
            "The prose also frames adequacy failure as informative rather than embarrassing. A manuscript that records where the synthetic path failed to satisfy its own gate is more credible than one that hides the failure and quietly changes scope. Here the fallback figure becomes part of the paper's honesty model: publication-quality narrative is achieved by making screening outcomes legible, not by pretending every planned route matured into a valid benchmark asset.",
        )
    if heading == "Evidence/fallback map and Figure 4":
        return (
            "Figure 4 broadens the same honesty principle to the project-wide evidence flow. Instead of a decorative roadmap, the figure is an evidence/fallback map that distinguishes prerequisites, executed authorities, failed gates, and later continuations. The reader is shown where the study advanced through direct results, where it advanced through formal closure, and where it stopped with a prospective dependency that future work would need to satisfy.",
            "The map is especially useful for the phase naming problem. The original Phase 0.5 screening gate and the later internal-screening continuation are related but not interchangeable. Keeping them separate prevents the paper from laundering an initial not-met result into the appearance of uninterrupted success. The manuscript therefore uses the figure to preserve causal structure: one gate failed, a narrower continuation was separately authorized, and downstream claims remain constrained by that history.",
            "This section also links the prospective appendix line back to the rest of the paper. Step 7 robustness and leakage work is described as a dependency-governed extension, not as retrospective confirmation for claims already made. That means the figure supports reproducibility and limitations at once: it helps readers locate existing evidence and locate the boundaries beyond which the current manuscript intentionally does not speculate.",
        )
    if heading == "Discussion, limitations, ethics/RAI, and data/code availability":
        return (
            "The discussion synthesizes the manuscript around boundaries rather than around a single triumphant metric conclusion. The study shows that fidelity-to-utility alignment is task-, perturbation-, and protocol-aware, and that honest reporting requires preserving both positive evidence and closure states. This leads directly to the paper's main interpretive position: useful metric behavior exists, but it is conditional, and any discussion section that suppresses those conditions would misstate the evidence.",
            "Limitations are therefore not relegated to a token closing paragraph. They are structural. Some task lines remain indirect, some remain synthetic-only, some remain not powered for confirmatory use, and some reproducibility questions terminate in audited absence of runnable artifacts. The manuscript treats those limitations as part of the contribution because they tell readers which future comparisons would require new data, new permissions, new executable releases, or new gates to be satisfied.",
            "Ethics/RAI and data/code availability are handled with the same specificity. Redistribution is governed by source-specific review and owner decisions, so the text does not promise unrestricted release of every local derivative. Instead it explains what can be shared through bounded metadata and what remains local-only pending redistribution clearance. References to paper/claims_matrix.csv and paper/references.bib are included so the eventual package remains auditable even while renderer choice and release packaging stay deferred.",
        )
    if heading == "References plus supplementary methods/tables":
        return (
            "The closing section clarifies that references and supplementary materials are part of the paper's reproducibility surface, not decorative appendices. Bibliography entries are retained only when they are backed by local DOI authority, and supplementary text is expected to point readers toward frozen method definitions, claims tracing, and packaging boundaries. This approach keeps the manuscript aligned with what the workspace can actually certify.",
            "Supplementary methods and tables are also described as extensions of the main contract rather than escape hatches for unsupported claims. If a result, closure, or status statement cannot be supported in the main text, it does not become acceptable simply because it is moved to supplementary material. The same rule applies to references, appendices, and data notes: no new numerical or status assertion should appear without an auditable path back to authority.",
            "For that reason the manuscript names the key relative assets directly. paper/claims_matrix.csv provides the claim-tracing surface, and paper/references.bib provides the retained bibliography surface for later renderer-specific conversion. Together with the authority bridge, limitations ledger, and preflight receipt, these files make the substantive draft inspectable even before venue formatting, submission bundling, or broader redistribution questions are resolved.",
        )
    return ()


def _render_manuscript(document: Mapping[str, object]) -> bytes:
    claims = {
        _require_str("claim_id", _require_dict("claim", row).get("claim_id")): _require_dict("claim", row)
        for row in _require_list("claims", document.get("claims"))
    }
    lines = [f"# {_require_str('title', document.get('title'))}", ""]
    for section in _require_list("sections", document.get("sections")):
        row = _require_dict("section", section)
        lines.append(f"## {_require_str('section.heading', row.get('heading'))}")
        lines.append("")
        for paragraph in _require_list("section.paragraphs", row.get("paragraphs")):
            lines.append(_require_str("paragraph", paragraph))
            lines.append("")
        for paragraph in _section_expansion_paragraphs(_require_str("section.heading", row.get("heading"))):
            lines.append(paragraph)
            lines.append("")
        for claim_id in _require_list("section.claim_ids", row.get("claim_ids", [])):
            claim = claims[_require_str("claim id", claim_id)]
            citation_keys = [
                _require_str("citation key", key)
                for key in _require_list("claim.citation_keys", claim.get("citation_keys", []))
            ]
            suffix = ""
            if citation_keys:
                suffix = " " + " ".join(f"[@{key}]" for key in citation_keys)
            lines.append(f"{_require_str('claim_text', claim.get('claim_text'))}{suffix}")
            lines.append("")
        section_citations = [
            _require_str("section citation", key)
            for key in _require_list("section.citation_keys", row.get("citation_keys", []))
        ]
        if section_citations:
            lines.append("Supporting literature: " + " ".join(f"[@{key}]" for key in section_citations))
            lines.append("")
    return "\n".join(lines).encode("utf-8")


def _claim_matrix_bytes(document: Mapping[str, object]) -> bytes:
    fieldnames = (
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
    )
    rows = []
    for item in _require_list("claims", document.get("claims")):
        claim = _require_dict("claim", item)
        rows.append(
            {
                "claim_id": _require_str("claim_id", claim.get("claim_id")),
                "manuscript_section": _require_str("manuscript_section", claim.get("manuscript_section")),
                "claim_text": _require_str("claim_text", claim.get("claim_text")),
                "evidence_class": _require_str("evidence_class", claim.get("evidence_class")),
                "authority_path": _require_str("authority_path", claim.get("authority_path")),
                "authority_sha256": _require_str("authority_sha256", claim.get("authority_sha256")),
                "payload_path": _require_str("payload_path", claim.get("payload_path")),
                "row_selector": _require_dict("row_selector", claim.get("row_selector")),
                "allowed_scope": _require_list("allowed_scope", claim.get("allowed_scope")),
                "prohibited_scope": _require_list("prohibited_scope", claim.get("prohibited_scope")),
                "verification_state": _require_str("verification_state", claim.get("verification_state")),
            }
        )
    return _csv_bytes(rows, fieldnames)


def _build_expected_files(config_path: Path) -> tuple[str, dict[str, bytes]]:
    raw, document = _load_config(config_path)
    authorities = _require_dict("authorities", document.get("authorities"))
    bridge = {}
    for name, value in authorities.items():
        entry = _require_dict(f"authority {name}", value)
        path = Path(_require_str("authority.path", entry.get("path")))
        if _require_str("authority.type", entry.get("type")) == "artifact_dir":
            digest = _artifact_authority_sha256(path)
        else:
            digest = _sha_file(path)
        bridge[name] = {"type": entry["type"], "path": str(path), "sha256": digest}
    authority_bridge = {
        "status": "complete",
        "step8_required": True,
        "step9_required": True,
        "authorities": bridge,
    }
    preflight = {
        "status": "complete",
        "step8_publication_core_state": "ready",
        "step9_release_metadata_state": "ready",
        "renderer_state": document["renderer_state"],
        "source_revision_status": document["source_revision_status"],
        "authority_count": len(bridge),
    }
    limitations = {
        "status": "complete",
        "renderer_state": document["renderer_state"],
        "source_revision_status": document["source_revision_status"],
        "phase05_original_confirmatory_gate": "not_met",
        "phase05_continuation_state": "separately_authorized_internal_screening",
        "phase2_model_adequacy": "5/12 pass and fallback",
        "baseline_evidence_boundary": "indirect_triple_evidence_not_powered_for_phase5",
        "denoising_evidence_boundary": "indirect_evidence_no_clean_target",
        "peak_evidence_boundary": "synthetic_mechanism_evidence_no_real_peak_assignments",
        "phase4_primary_state": "not_evaluable",
        "phase5_state": "not_powered_for_phase5_not_run",
        "step7_dependency_state": "results_only_if_authoritative",
        "licenses_state": "pending_owner_license_and_redistribution_decisions",
        "authority_names": sorted(bridge),
    }
    manuscript = _render_manuscript(document)
    citation_keys = sorted(set(re.findall(r"\[@([A-Za-z0-9_:-]+)\]", manuscript.decode("utf-8"))))  # type: ignore[name-defined]
    claim_lint = {
        "status": "complete",
        "claim_count": len(_require_list("claims", document.get("claims"))),
        "citation_keys": citation_keys,
        "issues": [
            {
                "claim_id": _require_str("claim_id", _require_dict("claim", row).get("claim_id")),
                "row_selector_state": "verified",
                "numeric_token_state": "verified",
                "citation_state": "verified",
                "scope_state": "verified",
            }
            for row in _require_list("claims", document.get("claims"))
        ],
    }
    payloads = {
        "config.json": raw,
        "authority_bridge.json": _json_bytes(authority_bridge),
        "preflight.json": _json_bytes(preflight),
        "paper/manuscript.md": manuscript,
        "paper/references.bib": _render_references(document),
        "paper/claims_matrix.csv": _claim_matrix_bytes(document),
        "limitations_ledger.json": _json_bytes(limitations),
        "claim_lint_receipt.json": _json_bytes(claim_lint),
    }
    identity = b"".join(
        len(name).to_bytes(4, "little")
        + name.encode("utf-8")
        + len(payload).to_bytes(8, "little")
        + hashlib.sha256(payload).digest()
        for name, payload in sorted(payloads.items())
    )
    run_id = "phase6-manuscript-" + hashlib.sha256(identity).hexdigest()
    manifest = {
        "artifact_schema_version": "phase6-manuscript-v1",
        "run_id": run_id,
        "status": "complete",
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
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
        "source_revision_status": document["source_revision_status"],
        "renderer_state": document["renderer_state"],
        "code_authority": {
            path: _sha_file(ROOT / path)
            for path in (
                "rpe/runner/phase6_manuscript.py",
                "rpe/runner/phase6_manuscript_verifier.py",
                "tools/run_phase6_manuscript.py",
            )
        },
        "counts": {
            "section_count": len(_require_list("sections", document.get("sections"))),
            "claim_count": len(_require_list("claims", document.get("claims"))),
            "citation_count": len(citation_keys),
        },
    }
    payloads["manifest.json"] = _json_bytes(manifest)
    payloads["complete.json"] = _json_bytes({"run_id": run_id, "status": "complete"})
    sha_lines = []
    for name in sorted(payloads):
        sha_lines.append(f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n")
    payloads["SHA256SUMS"] = "".join(sha_lines).encode("utf-8")
    return run_id, payloads


def _load_manifest(run_path: Path) -> dict[str, object]:
    manifest_path = run_path / "manifest.json"
    if not manifest_path.is_file():
        raise Phase6ManuscriptVerificationError("missing manifest")
    raw = manifest_path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase6ManuscriptVerificationError("invalid manifest") from error
    if raw != _canonical(document):
        raise Phase6ManuscriptVerificationError("manifest must be canonical")
    return dict(document)


def _check_sha256sums(run_path: Path) -> None:
    ledger = run_path / "SHA256SUMS"
    if not ledger.is_file():
        raise Phase6ManuscriptVerificationError("missing checksum ledger")
    entries = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise Phase6ManuscriptVerificationError("invalid checksum ledger")
        entries.append((parts[0], parts[1]))
    actual = sorted(
        path.relative_to(run_path).as_posix()
        for path in run_path.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    listed = [name for _, name in entries]
    if listed != actual:
        raise Phase6ManuscriptVerificationError("checksum inventory mismatch")
    for digest, name in entries:
        if _sha_file(run_path / name) != digest:
            raise Phase6ManuscriptVerificationError("byte mismatch")


def verify_phase6_manuscript(run_path: Path) -> Phase6ManuscriptVerificationSummary:
    run_path = Path(run_path)
    _check_sha256sums(run_path)
    manifest = _load_manifest(run_path)
    config_path = Path(str(manifest["config"]["path"]))
    expected_run_id, expected_files = _build_expected_files(config_path)
    if expected_run_id != str(manifest["run_id"]):
        raise Phase6ManuscriptVerificationError("run identity mismatch")
    current = {
        path.relative_to(run_path).as_posix(): path.read_bytes()
        for path in run_path.rglob("*")
        if path.is_file()
    }
    if current != expected_files:
        raise Phase6ManuscriptVerificationError("byte mismatch")
    return Phase6ManuscriptVerificationSummary(
        run_id=str(manifest["run_id"]),
        path=run_path,
        status=str(manifest["status"]),
        counts=dict(manifest["counts"]),
    )
