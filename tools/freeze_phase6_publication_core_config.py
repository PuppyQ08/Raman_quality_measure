from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_publication_core import derive_figure3_model_adequacy_rows  # noqa: E402
from rpe.runner.phase6_publication_core_authority import (  # noqa: E402
    CLAIM_BOUNDARY,
    CODE_RECEIPT_PATHS,
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    FIXED_AUTHORITY_RUNS,
    PAYLOAD_FILES,
    REPORT_AUTHORITY_PATHS,
    ROOT as PROJECT_ROOT,
    SCHEMA_VERSION,
    validate_step7_deferred_report,
    canonical_json_bytes,
    file_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze Phase 6 publication-core config")
    step7_group = parser.add_mutually_exclusive_group(required=True)
    step7_group.add_argument("--step7-run-path")
    step7_group.add_argument("--step7-deferred-report")
    parser.add_argument("--output", default=str(DEFAULT_CONFIG))
    return parser


def _validate_step7_path(path: Path) -> None:
    if not path.is_dir():
        raise ValueError("step7_run_path: not a directory")
    required = {"config.json", "manifest.json", "complete.json", "SHA256SUMS"}
    found = {item.name for item in path.iterdir() if item.is_file()}
    missing = sorted(required - found)
    if missing:
        raise ValueError(f"step7_run_path: missing required files {missing}")


def _document(*, step7_path: Path | None, step7_deferred_report: Path | None) -> dict[str, object]:
    figure3_rows = derive_figure3_model_adequacy_rows(PROJECT_ROOT / FIXED_AUTHORITY_RUNS["phase2_background_fit"])
    if step7_deferred_report is not None:
        step7_dependency = validate_step7_deferred_report(step7_deferred_report)
    elif step7_path is not None:
        _validate_step7_path(step7_path)
        step7_dependency = {
            "mode": "completed_run",
            "state": "complete",
            "run": file_identity(step7_path / "SHA256SUMS"),
            "run_path": str(step7_path.relative_to(PROJECT_ROOT)),
        }
    else:
        raise ValueError("one Step7 authority mode is required")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "claim_boundary": CLAIM_BOUNDARY,
        "synthetic_fixture": False,
        "artifact_payload_files": list(PAYLOAD_FILES),
        "expected": {
            "table1_row_count": 12,
            "table_s1_row_count": 143,
            "table_s2_row_count": 65,
            "table2_row_count": 1656,
            "table_s3_row_count": 316,
            "figure3_row_count": len(figure3_rows),
            "configured_payload_count": len(PAYLOAD_FILES),
            "artifact_file_count": len(PAYLOAD_FILES) + 2,
        },
        "figure3": {
            "width_px": 3600,
            "height_px": 2700,
            "dpi": 300,
            "p95_threshold": 0.25,
            "extractor_order": ["airpls", "arpls", "mor"],
            "excitation_order": ["green_514", "green_532", "nir_780", "nir_785"],
        },
        "figure4": {
            "width_px": 3600,
            "height_px": 2400,
            "dpi": 300,
        },
        "authorities": {
            key: file_identity(PROJECT_ROOT / relative_path)
            for key, relative_path in FIXED_AUTHORITY_RUNS.items()
        },
        "report_authorities": {
            key: file_identity(PROJECT_ROOT / relative_path)
            for key, relative_path in REPORT_AUTHORITY_PATHS.items()
        },
        "code_receipts": {
            key: file_identity(PROJECT_ROOT / relative_path)
            for key, relative_path in CODE_RECEIPT_PATHS.items()
        },
        "step7_dependency": step7_dependency,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    step7_path = Path(args.step7_run_path).resolve() if args.step7_run_path else None
    step7_deferred_report = (
        Path(args.step7_deferred_report).resolve()
        if args.step7_deferred_report
        else None
    )
    document = _document(
        step7_path=step7_path,
        step7_deferred_report=step7_deferred_report,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(document))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
