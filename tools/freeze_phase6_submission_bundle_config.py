from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_submission_bundle import repo_relative_path


CONFIG_RELATIVE_PATH = Path("experiments/phase6/configs/submission_bundle_v1.json")


class FreezePhase6SubmissionBundleConfigError(ValueError):
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_artifact_dir(path: Path, *, required_payloads: Sequence[str]) -> str:
    if not path.is_dir():
        raise FreezePhase6SubmissionBundleConfigError(f"missing artifact directory: {path}")
    complete = path / "complete.json"
    ledger = path / "SHA256SUMS"
    manifest = path / "manifest.json"
    if not complete.is_file() or not ledger.is_file() or not manifest.is_file():
        raise FreezePhase6SubmissionBundleConfigError(
            f"artifact missing terminal, manifest, or checksum ledger: {path}"
        )
    complete_doc = _load_json(complete)
    manifest_doc = _load_json(manifest)
    if "complete" not in str(complete_doc.get("status", "")):
        raise FreezePhase6SubmissionBundleConfigError(f"artifact terminal is not complete: {path}")
    listed = list(manifest_doc.get("payload_files", ()))
    if not isinstance(listed, list):
        raise FreezePhase6SubmissionBundleConfigError(f"manifest payload inventory is invalid: {path}")
    missing = [name for name in required_payloads if name not in listed or not (path / name).is_file()]
    if missing:
        raise FreezePhase6SubmissionBundleConfigError(
            f"artifact missing required payloads: {', '.join(missing)}"
        )
    entries: list[tuple[str, str]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise FreezePhase6SubmissionBundleConfigError(f"invalid checksum ledger: {path}")
        entries.append((parts[0], parts[1]))
    actual = sorted(
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    )
    listed_names = [name for _, name in entries]
    if len(listed_names) != len(set(listed_names)) or set(listed_names) != set(actual):
        raise FreezePhase6SubmissionBundleConfigError(f"checksum inventory mismatch: {path}")
    for digest, relative in entries:
        if _sha_file(path / relative) != digest:
            raise FreezePhase6SubmissionBundleConfigError(f"checksum mismatch: {path / relative}")
    return _sha_file(ledger)


def freeze_phase6_submission_bundle_config(
    *,
    project_root: Path,
    step4_final_figures: Path,
    step8_publication_core: Path,
    step9_release_metadata: Path,
    step10_manuscript: Path,
) -> Path:
    project_root = Path(project_root).resolve()
    step4_final_figures = Path(step4_final_figures).resolve()
    step8_publication_core = Path(step8_publication_core).resolve()
    step9_release_metadata = Path(step9_release_metadata).resolve()
    step10_manuscript = Path(step10_manuscript).resolve()

    _validate_artifact_dir(
        step4_final_figures,
        required_payloads=(
            "figure1_phase4_response.png",
            "figure1_phase4_response.svg",
            "figure2_phase4_alignment.png",
            "figure2_phase4_alignment.svg",
        ),
    )
    _validate_artifact_dir(
        step8_publication_core,
        required_payloads=(
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
        ),
    )
    _validate_artifact_dir(
        step9_release_metadata,
        required_payloads=(
            "data_card.md",
            "croissant.json",
            "metadata_index.parquet",
            "release_matrix.csv",
            "limitations.jsonl",
            "environment.json",
        ),
    )
    _validate_artifact_dir(
        step10_manuscript,
        required_payloads=(
            "paper/manuscript.md",
            "paper/references.bib",
            "paper/claims_matrix.csv",
        ),
    )

    document = {
        "schema_version": "phase6-submission-bundle-v1",
        "bundle_name": "phase6-submission-bundle",
        "source_mode": "formal",
        "parents": {
            "step4_final_figures": {
                "path": repo_relative_path(step4_final_figures, project_root),
                "required_payloads": [
                    "figure1_phase4_response.png",
                    "figure1_phase4_response.svg",
                    "figure2_phase4_alignment.png",
                    "figure2_phase4_alignment.svg",
                ],
                "required_terminal": "complete.json",
            },
            "step8": {
                "path": repo_relative_path(step8_publication_core, project_root),
                "required_payloads": [
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
                "required_terminal": "complete.json",
            },
            "step9": {
                "path": repo_relative_path(step9_release_metadata, project_root),
                "required_payloads": [
                    "data_card.md",
                    "croissant.json",
                    "metadata_index.parquet",
                    "release_matrix.csv",
                    "limitations.jsonl",
                    "environment.json",
                ],
                "required_terminal": "complete.json",
            },
            "step10": {
                "path": repo_relative_path(step10_manuscript, project_root),
                "required_payloads": [
                    "paper/manuscript.md",
                    "paper/references.bib",
                    "paper/claims_matrix.csv",
                ],
                "required_terminal": "complete.json",
            },
        },
        "bundle_files": [
            {
                "parent": "step10",
                "source_path": "paper/manuscript.md",
                "bundle_path": "paper/manuscript.md",
                "role": "manuscript",
            },
            {
                "parent": "step10",
                "source_path": "paper/references.bib",
                "bundle_path": "paper/references.bib",
                "role": "references",
            },
            {
                "parent": "step10",
                "source_path": "paper/claims_matrix.csv",
                "bundle_path": "paper/claims_matrix.csv",
                "role": "claims_matrix",
            },
            {
                "parent": "step4_final_figures",
                "source_path": "figure1_phase4_response.png",
                "bundle_path": "assets/figures/figure1_phase4_response.png",
                "role": "frozen_phase4_figure",
            },
            {
                "parent": "step4_final_figures",
                "source_path": "figure1_phase4_response.svg",
                "bundle_path": "assets/figures/figure1_phase4_response.svg",
                "role": "frozen_phase4_figure",
            },
            {
                "parent": "step4_final_figures",
                "source_path": "figure2_phase4_alignment.png",
                "bundle_path": "assets/figures/figure2_phase4_alignment.png",
                "role": "frozen_phase4_figure",
            },
            {
                "parent": "step4_final_figures",
                "source_path": "figure2_phase4_alignment.svg",
                "bundle_path": "assets/figures/figure2_phase4_alignment.svg",
                "role": "frozen_phase4_figure",
            },
            {
                "parent": "step8",
                "source_path": "table1_metric_validity.csv",
                "bundle_path": "assets/tables/table1_metric_validity.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "table1_metric_validity.md",
                "bundle_path": "assets/tables/table1_metric_validity.md",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "table_s1_full_alignment.csv",
                "bundle_path": "assets/tables/table_s1_full_alignment.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "table_s2_protocol_interactions.csv",
                "bundle_path": "assets/tables/table_s2_protocol_interactions.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "table2_method_evidence.csv",
                "bundle_path": "assets/tables/table2_method_evidence.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "table2_method_evidence.md",
                "bundle_path": "assets/tables/table2_method_evidence.md",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "table_s3_system_status.csv",
                "bundle_path": "assets/tables/table_s3_system_status.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "figure3_model_adequacy_data.csv",
                "bundle_path": "assets/tables/figure3_model_adequacy_data.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "figure4_gate_nodes.csv",
                "bundle_path": "assets/tables/figure4_gate_nodes.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "figure4_gate_edges.csv",
                "bundle_path": "assets/tables/figure4_gate_edges.csv",
                "role": "aggregate_table",
            },
            {
                "parent": "step8",
                "source_path": "figure3_model_adequacy.png",
                "bundle_path": "assets/figures/figure3_model_adequacy.png",
                "role": "verified_step8_figure",
            },
            {
                "parent": "step8",
                "source_path": "figure3_model_adequacy.svg",
                "bundle_path": "assets/figures/figure3_model_adequacy.svg",
                "role": "verified_step8_figure",
            },
            {
                "parent": "step8",
                "source_path": "figure4_gate_map.png",
                "bundle_path": "assets/figures/figure4_gate_map.png",
                "role": "verified_step8_figure",
            },
            {
                "parent": "step8",
                "source_path": "figure4_gate_map.svg",
                "bundle_path": "assets/figures/figure4_gate_map.svg",
                "role": "verified_step8_figure",
            },
            {
                "parent": "step9",
                "source_path": "data_card.md",
                "bundle_path": "metadata/data_card.md",
                "role": "release_metadata",
            },
            {
                "parent": "step9",
                "source_path": "croissant.json",
                "bundle_path": "metadata/croissant.json",
                "role": "release_metadata",
            },
            {
                "parent": "step9",
                "source_path": "metadata_index.parquet",
                "bundle_path": "metadata/metadata_index.parquet",
                "role": "release_metadata",
            },
            {
                "parent": "step9",
                "source_path": "release_matrix.csv",
                "bundle_path": "metadata/release_matrix.csv",
                "role": "release_metadata",
            },
            {
                "parent": "step9",
                "source_path": "limitations.jsonl",
                "bundle_path": "metadata/source_limitations.jsonl",
                "role": "release_metadata",
            },
            {
                "parent": "step9",
                "source_path": "environment.json",
                "bundle_path": "metadata/source_environment.json",
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
                "assets/figures/figure1_phase4_response.png",
                "assets/figures/figure1_phase4_response.svg",
                "assets/figures/figure2_phase4_alignment.png",
                "assets/figures/figure2_phase4_alignment.svg",
                "assets/figures/figure3_model_adequacy.png",
                "assets/figures/figure3_model_adequacy.svg",
                "assets/figures/figure4_gate_map.png",
                "assets/figures/figure4_gate_map.svg",
                "assets/tables/table1_metric_validity.csv",
                "assets/tables/table1_metric_validity.md",
                "assets/tables/table_s1_full_alignment.csv",
                "assets/tables/table_s2_protocol_interactions.csv",
                "assets/tables/table2_method_evidence.csv",
                "assets/tables/table2_method_evidence.md",
                "assets/tables/table_s3_system_status.csv",
                "assets/tables/figure3_model_adequacy_data.csv",
                "assets/tables/figure4_gate_nodes.csv",
                "assets/tables/figure4_gate_edges.csv",
                "metadata/data_card.md",
                "metadata/croissant.json",
                "metadata/metadata_index.parquet",
                "metadata/release_matrix.csv",
                "metadata/source_limitations.jsonl",
                "metadata/source_environment.json",
                "metadata/environment_receipt.json",
                "metadata/limitations_ledger.json",
                "metadata/artifact_claim_map.json",
                "manifest.json",
            ],
            "terminal_marker": "complete.json",
            "archive_format": "directory_only",
        },
    }
    target = project_root / CONFIG_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical(document))
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the formal Phase 6 Step 11 submission bundle config"
    )
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--step4-final-figures", required=True)
    parser.add_argument("--step8-publication-core", required=True)
    parser.add_argument("--step9-release-metadata", required=True)
    parser.add_argument("--step10-manuscript", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = freeze_phase6_submission_bundle_config(
        project_root=Path(args.project_root),
        step4_final_figures=Path(args.step4_final_figures),
        step8_publication_core=Path(args.step8_publication_core),
        step9_release_metadata=Path(args.step9_release_metadata),
        step10_manuscript=Path(args.step10_manuscript),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
