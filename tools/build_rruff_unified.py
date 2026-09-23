from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.io.rruff import (  # noqa: E402
    RruffConversionSummary,
    RruffRuntimeMetrics,
    RruffValidationError,
    _build_rruff_unified_with_metrics,
)
from rpe.io.schema import SchemaValidationError  # noqa: E402
from rpe.io.store import DatasetValidationError  # noqa: E402


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dataset_document(summary) -> dict[str, object]:
    return {
        "dataset_id": summary.dataset_id,
        "record_count": summary.record_count,
        "eligible_records": summary.eligible_records,
        "axis_group_count": summary.axis_group_count,
        "output_bytes": summary.output_bytes,
        "files": {
            name: dict(details)
            for name, details in summary.files.items()
        },
    }


def _summary_document(
    summary: RruffConversionSummary,
    metrics: RruffRuntimeMetrics,
) -> dict[str, object]:
    return {
        "status": "written",
        "raw": _dataset_document(summary.raw),
        "processed": _dataset_document(summary.processed),
        "pairing": {
            "path": summary.pair_index_path.as_posix(),
            "rows": summary.pair_index_rows,
            "bytes": summary.pair_index_path.stat().st_size,
            "sha256": summary.pair_index_sha256,
            "status_counts": dict(summary.pair_status_counts),
            "axis_relation_counts": dict(
                summary.axis_relation_counts
            ),
        },
        "receipt": {
            "path": summary.receipt_path.as_posix(),
            "bytes": summary.receipt_path.stat().st_size,
            "sha256": summary.receipt_sha256,
            "rejected_records": summary.rejected_records,
        },
        "metrics": asdict(metrics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary, metrics = _build_rruff_unified_with_metrics(
            args.raw_root,
            args.output_root,
            overwrite=args.overwrite,
        )
    except (
        RruffValidationError,
        DatasetValidationError,
        SchemaValidationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            _canonical_json(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 1

    print(_canonical_json(_summary_document(summary, metrics)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
