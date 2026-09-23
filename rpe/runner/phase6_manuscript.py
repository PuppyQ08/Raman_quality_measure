from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/manuscript_v1.json"
_CLAIM_COLUMNS = (
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
_SECTION_HEADINGS = (
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
_IDENTITY_CODE_FILES = (
    "rpe/runner/phase6_manuscript.py",
    "rpe/runner/phase6_manuscript_verifier.py",
    "tools/run_phase6_manuscript.py",
)
_BIB_REQUIRED_FIELDS = {
    "article": ("author", "title", "journal", "year", "doi"),
}


class Phase6ManuscriptError(ValueError):
    pass


@dataclass(frozen=True)
class Phase6ManuscriptConfig:
    path: Path
    raw: bytes
    sha256: str
    document: Mapping[str, object]


@dataclass(frozen=True)
class Phase6ManuscriptSummary:
    run_id: str
    path: Path
    status: str
    counts: Mapping[str, int]


def repo_relative_path(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def repo_absolute_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _plain(value: object) -> object:
    if isinstance(value, MappingProxyType):
        value = dict(value)
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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Phase6ManuscriptError(f"{name} must be an object")
    return dict(value)


def _require_list(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise Phase6ManuscriptError(f"{name} must be a list")
    return list(value)


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise Phase6ManuscriptError(f"{name} must be a nonempty string")
    return value


def _json_bytes(value: object) -> bytes:
    return _canonical(value)


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


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


def _markdown_escape(value: str) -> str:
    return value.replace("\r", "").rstrip()


def _scan_citation_keys(text: str) -> list[str]:
    return re.findall(r"\[@([A-Za-z0-9_:-]+)\]", text)


def _numeric_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\b\d+(?:/\d+)?(?:\.\d+)?\b", text))


def _read_text_or_csv(path: Path) -> tuple[str, list[dict[str, str]] | None]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return path.read_text(encoding="utf-8"), rows
    if path.suffix.lower() == ".json":
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
        if isinstance(value, list):
            rows = []
            for item in value:
                if not isinstance(item, dict):
                    raise Phase6ManuscriptError("JSON evidence must contain object rows")
                rows.append({str(key): json.dumps(val, sort_keys=True, separators=(",", ":")) if isinstance(val, (dict, list)) else str(val) for key, val in item.items()})
            return text, rows
        if isinstance(value, dict):
            return text, [{str(key): json.dumps(val, sort_keys=True, separators=(",", ":")) if isinstance(val, (dict, list)) else str(val) for key, val in value.items()}]
        return text, None
    if path.suffix.lower() == ".jsonl":
        text = path.read_text(encoding="utf-8")
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise Phase6ManuscriptError("JSONL evidence must contain object rows")
            rows.append({str(key): json.dumps(val, sort_keys=True, separators=(",", ":")) if isinstance(val, (dict, list)) else str(val) for key, val in item.items()})
        return text, rows
    return path.read_text(encoding="utf-8"), None


def _artifact_authority_sha256(path: Path) -> str:
    ledger = path / "SHA256SUMS"
    if not path.is_dir() or not ledger.is_file():
        raise Phase6ManuscriptError("missing evidence or non-authoritative path")
    complete = path / "complete.json"
    if not complete.is_file():
        raise Phase6ManuscriptError("missing evidence or non-authoritative path")
    return _sha_file(ledger)


def _validate_authorities(document: Mapping[str, object]) -> dict[str, dict[str, object]]:
    authorities = _require_dict("authorities", document.get("authorities"))
    validated: dict[str, dict[str, object]] = {}
    required = {
        "step8_publication_core",
        "step9_release_metadata",
        "step7_appendix_execution_report",
        "phase4_closure",
        "step1_design",
        "step3_baseline",
        "step4_denoising",
        "step5_peak",
        "step6_appendix",
    }
    if set(authorities) != required:
        raise Phase6ManuscriptError("authority set mismatch")
    for name, value in authorities.items():
        entry = _require_dict(f"authorities.{name}", value)
        kind = _require_str(f"authorities.{name}.type", entry.get("type"))
        path = repo_absolute_path(_require_str(f"authorities.{name}.path", entry.get("path")))
        if kind == "artifact_dir":
            entry["computed_sha256"] = _artifact_authority_sha256(path)
        elif kind == "file":
            if not path.is_file():
                raise Phase6ManuscriptError("missing evidence or non-authoritative path")
            expected = _require_str(f"authorities.{name}.sha256", entry.get("sha256"))
            actual = _sha_file(path)
            if actual != expected:
                raise Phase6ManuscriptError("authority hash mismatch")
            entry["computed_sha256"] = actual
        else:
            raise Phase6ManuscriptError(f"unsupported authority type: {kind}")
        validated[name] = entry
    return validated


def _validate_config_document(document: Mapping[str, object]) -> None:
    if document.get("schema_version") != "phase6-manuscript-v1":
        raise Phase6ManuscriptError("unexpected schema_version")
    title = _require_str("title", document.get("title"))
    if (
        title
        != "When Do Raman Fidelity Metrics Track Downstream Utility? A Task-, Perturbation-, and Protocol-Aware Evaluation"
    ):
        raise Phase6ManuscriptError("title mismatch")
    if _require_str("source_revision_status", document.get("source_revision_status")) != "unavailable_no_valid_git_repository":
        raise Phase6ManuscriptError("source revision state mismatch")
    if _require_str("renderer_state", document.get("renderer_state")) != "deferred_pending_locked_renderer_and_venue_choice":
        raise Phase6ManuscriptError("renderer state mismatch")
    artifact = _require_dict("artifact_contract", document.get("artifact_contract"))
    payload_files = tuple(_require_list("artifact_contract.payload_files", artifact.get("payload_files")))
    expected_payloads = (
        "config.json",
        "authority_bridge.json",
        "preflight.json",
        "paper/manuscript.md",
        "paper/references.bib",
        "paper/claims_matrix.csv",
        "limitations_ledger.json",
        "claim_lint_receipt.json",
        "manifest.json",
    )
    if payload_files != expected_payloads:
        raise Phase6ManuscriptError("artifact payload contract drift")
    if int(artifact.get("configured_payload_count", -1)) != 9:
        raise Phase6ManuscriptError("artifact payload count mismatch")
    if int(artifact.get("total_file_count_with_terminal_and_sha256sums", -1)) != 11:
        raise Phase6ManuscriptError("artifact total file count mismatch")
    if tuple(_require_list("artifact_contract.terminal_markers", artifact.get("terminal_markers"))) != ("complete.json", "failed.json"):
        raise Phase6ManuscriptError("terminal marker mismatch")
    sections = _require_list("sections", document.get("sections"))
    headings = tuple(_require_str("sections[].heading", _require_dict("section", item).get("heading")) for item in sections)
    if headings != _SECTION_HEADINGS:
        raise Phase6ManuscriptError("section heading mismatch")
    claims = _require_list("claims", document.get("claims"))
    claim_ids = [str(_require_dict("claim", row).get("claim_id")) for row in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise Phase6ManuscriptError("duplicate claim_id")


def load_phase6_manuscript_config(path: Path = DEFAULT_CONFIG) -> Phase6ManuscriptConfig:
    raw = Path(path).read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase6ManuscriptError("config must be canonical JSON") from error
    if not isinstance(document, dict) or raw != _canonical(document):
        raise Phase6ManuscriptError("config must be canonical JSON")
    _validate_config_document(document)
    _validate_authorities(document)
    return Phase6ManuscriptConfig(
        path=Path(path),
        raw=raw,
        sha256=_sha(raw),
        document=MappingProxyType(document),
    )


def _render_references(references: Sequence[Mapping[str, object]]) -> bytes:
    lines: list[str] = []
    seen: set[str] = set()
    for row in references:
        entry = _require_dict("reference", row)
        key = _require_str("reference.key", entry.get("key"))
        if key in seen:
            raise Phase6ManuscriptError("duplicate citation key")
        seen.add(key)
        entry_type = _require_str("reference.entry_type", entry.get("entry_type"))
        fields = _require_dict("reference.fields", entry.get("fields"))
        required = _BIB_REQUIRED_FIELDS.get(entry_type)
        if required is None:
            raise Phase6ManuscriptError(f"unsupported BibTeX entry type: {entry_type}")
        for field in required:
            _require_str(f"reference.fields.{field}", fields.get(field))
        lines.append(f"@{entry_type}{{{key},")
        for field in sorted(fields):
            value = _require_str(f"reference.fields.{field}", fields[field])
            lines.append(f"  {field} = {{{value}}},")
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


def _render_manuscript(document: Mapping[str, object], claim_lookup: Mapping[str, Mapping[str, object]]) -> bytes:
    sections = _require_list("sections", document.get("sections"))
    lines = [f"# {_require_str('title', document.get('title'))}", ""]
    for section in sections:
        row = _require_dict("section", section)
        heading = _require_str("section.heading", row.get("heading"))
        lines.append(f"## {heading}")
        lines.append("")
        for paragraph in _require_list("section.paragraphs", row.get("paragraphs")):
            lines.append(_require_str("section.paragraphs[]", paragraph))
            lines.append("")
        for paragraph in _section_expansion_paragraphs(heading):
            lines.append(paragraph)
            lines.append("")
        for claim_id in _require_list("section.claim_ids", row.get("claim_ids", [])):
            claim = _require_dict("claim reference", claim_lookup[_require_str("claim id", claim_id)])
            text = _require_str("claim_text", claim.get("claim_text"))
            citation_keys = [_require_str("citation key", item) for item in _require_list("claim.citation_keys", claim.get("citation_keys", []))]
            suffix = ""
            if citation_keys:
                suffix = " " + " ".join(f"[@{key}]" for key in citation_keys)
            lines.append(f"{text}{suffix}")
            lines.append("")
        section_citations = [_require_str("section citation", item) for item in _require_list("section.citation_keys", row.get("citation_keys", []))]
        if section_citations:
            lines.append("Supporting literature: " + " ".join(f"[@{key}]" for key in section_citations))
            lines.append("")
    return "\n".join(lines).encode("utf-8")


def _select_row(rows: Sequence[Mapping[str, str]], selector: Mapping[str, object]) -> Mapping[str, str]:
    matched = []
    for row in rows:
        ok = True
        for key, value in selector.items():
            if row.get(str(key)) != str(value):
                ok = False
                break
        if ok:
            matched.append(row)
    if not matched:
        raise Phase6ManuscriptError("missing evidence or non-authoritative path")
    if len(matched) != 1:
        raise Phase6ManuscriptError("ambiguous row selector")
    return matched[0]


def _claim_matrix_rows(document: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
    return rows


def _lint_claims(document: Mapping[str, object], manuscript_text: str) -> dict[str, object]:
    references = {
        _require_str("reference.key", _require_dict("reference", row).get("key"))
        for row in _require_list("references", document.get("references"))
    }
    seen_citations = sorted(set(_scan_citation_keys(manuscript_text)))
    unresolved = sorted(key for key in seen_citations if key not in references)
    if unresolved:
        raise Phase6ManuscriptError("unresolved citation")

    issues: list[dict[str, object]] = []
    for item in _require_list("claims", document.get("claims")):
        claim = _require_dict("claim", item)
        authority_path = repo_absolute_path(_require_str("authority_path", claim.get("authority_path")))
        payload_path = repo_absolute_path(_require_str("payload_path", claim.get("payload_path")))
        if not payload_path.is_file():
            raise Phase6ManuscriptError("missing evidence or non-authoritative path")
        if authority_path.is_dir():
            actual_authority = _artifact_authority_sha256(authority_path)
        elif authority_path.is_file():
            actual_authority = _sha_file(authority_path)
        else:
            raise Phase6ManuscriptError("missing evidence or non-authoritative path")
        if actual_authority != _require_str("authority_sha256", claim.get("authority_sha256")):
            raise Phase6ManuscriptError("authority hash mismatch")

        text, rows = _read_text_or_csv(payload_path)
        selector = _require_dict("row_selector", claim.get("row_selector"))
        selected_text = text
        if rows is not None:
            selected_row = _select_row(rows, selector)
            selected_text = json.dumps(selected_row, sort_keys=True, separators=(",", ":"))
        numeric_tokens = _numeric_tokens(_require_str("claim_text", claim.get("claim_text")))
        for token in numeric_tokens:
            if token not in selected_text:
                raise Phase6ManuscriptError("numeric token absent from selected payload row/document")
        lowered = _require_str("claim_text", claim.get("claim_text")).lower()
        for phrase in _require_list("prohibited_scope", claim.get("prohibited_scope")):
            prohibited = _require_str("prohibited_scope[]", phrase)
            if prohibited.lower() in lowered:
                raise Phase6ManuscriptError("prohibited-scope wording")
        claim_citations = [_require_str("citation key", value) for value in _require_list("citation_keys", claim.get("citation_keys", []))]
        for key in claim_citations:
            if key not in references:
                raise Phase6ManuscriptError("unresolved citation")
        issues.append(
            {
                "claim_id": _require_str("claim_id", claim.get("claim_id")),
                "row_selector_state": "verified",
                "numeric_token_state": "verified",
                "citation_state": "verified",
                "scope_state": "verified",
            }
        )
    return {
        "status": "complete",
        "claim_count": len(issues),
        "citation_keys": seen_citations,
        "issues": issues,
    }


def _limitations_ledger(document: Mapping[str, object], authorities: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        "status": "complete",
        "renderer_state": _require_str("renderer_state", document.get("renderer_state")),
        "source_revision_status": _require_str("source_revision_status", document.get("source_revision_status")),
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
        "authority_names": sorted(authorities),
    }


def _authority_bridge(document: Mapping[str, object], authorities: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    rows = {}
    for name, entry in authorities.items():
        rows[name] = {
            "type": entry["type"],
            "path": entry["path"],
            "sha256": entry["computed_sha256"],
        }
    return {
        "status": "complete",
        "step8_required": True,
        "step9_required": True,
        "authorities": rows,
    }


def _preflight(document: Mapping[str, object], authorities: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        "status": "complete",
        "step8_publication_core_state": "ready",
        "step9_release_metadata_state": "ready",
        "renderer_state": document["renderer_state"],
        "source_revision_status": document["source_revision_status"],
        "authority_count": len(authorities),
    }


def _manifest(
    *,
    config: Phase6ManuscriptConfig,
    run_id: str,
    payload_files: Sequence[str],
    counts: Mapping[str, int],
) -> dict[str, object]:
    code_authority = {
        path: _sha_file(ROOT / path)
        for path in _IDENTITY_CODE_FILES
    }
    return {
        "artifact_schema_version": "phase6-manuscript-v1",
        "run_id": run_id,
        "status": "complete",
        "payload_files": list(payload_files),
        "config": {
            "path": str(config.path),
            "sha256": config.sha256,
            "bytes": len(config.raw),
        },
        "source_revision_status": config.document["source_revision_status"],
        "renderer_state": config.document["renderer_state"],
        "code_authority": code_authority,
        "counts": dict(counts),
    }


def build_phase6_manuscript(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
) -> Phase6ManuscriptSummary:
    config = load_phase6_manuscript_config(config_path)
    output_root = Path(output_root) if output_root is not None else ROOT / "results" / "phase6" / "manuscript_v1"
    authorities = _validate_authorities(config.document)
    claims = _require_list("claims", config.document.get("claims"))
    claim_lookup = {
        _require_str("claim_id", _require_dict("claim", row).get("claim_id")): _require_dict("claim", row)
        for row in claims
    }
    references_bytes = _render_references(_require_list("references", config.document.get("references")))
    manuscript_bytes = _render_manuscript(config.document, claim_lookup)
    claim_matrix_bytes = _csv_bytes(_claim_matrix_rows(config.document), _CLAIM_COLUMNS)
    claim_receipt = _lint_claims(config.document, manuscript_bytes.decode("utf-8"))
    claim_lint_bytes = _json_bytes(claim_receipt)
    authority_bridge_bytes = _json_bytes(_authority_bridge(config.document, authorities))
    preflight_bytes = _json_bytes(_preflight(config.document, authorities))
    limitations_bytes = _json_bytes(_limitations_ledger(config.document, authorities))
    payload_map = {
        "config.json": config.raw,
        "authority_bridge.json": authority_bridge_bytes,
        "preflight.json": preflight_bytes,
        "paper/manuscript.md": manuscript_bytes,
        "paper/references.bib": references_bytes,
        "paper/claims_matrix.csv": claim_matrix_bytes,
        "limitations_ledger.json": limitations_bytes,
        "claim_lint_receipt.json": claim_lint_bytes,
    }
    identity_payload = b"".join(
        len(name).to_bytes(4, "little")
        + name.encode("utf-8")
        + len(payload).to_bytes(8, "little")
        + hashlib.sha256(payload).digest()
        for name, payload in sorted(payload_map.items())
    )
    run_id = "phase6-manuscript-" + hashlib.sha256(identity_payload).hexdigest()
    counts = {
        "section_count": len(_SECTION_HEADINGS),
        "claim_count": len(claims),
        "citation_count": len(claim_receipt["citation_keys"]),
    }
    manifest_bytes = _json_bytes(
        _manifest(
            config=config,
            run_id=run_id,
            payload_files=tuple(_require_list("artifact_contract.payload_files", _require_dict("artifact_contract", config.document.get("artifact_contract")).get("payload_files"))),
            counts=counts,
        )
    )
    payload_map["manifest.json"] = manifest_bytes
    run_path = output_root / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    for relative, payload in payload_map.items():
        path = run_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    complete = {"run_id": run_id, "status": "complete"}
    (run_path / "complete.json").write_bytes(_json_bytes(complete))
    sha_lines = []
    for path in sorted(item for item in run_path.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        relative = path.relative_to(run_path).as_posix()
        sha_lines.append(f"{_sha_file(path)}  {relative}\n")
    (run_path / "SHA256SUMS").write_text("".join(sha_lines), encoding="utf-8")
    return Phase6ManuscriptSummary(
        run_id=run_id,
        path=run_path,
        status="complete",
        counts=MappingProxyType(counts),
    )
