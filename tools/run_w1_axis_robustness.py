from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.w1_axis_robustness import (  # noqa: E402
    DEFAULT_CONFIG,
    build_w1_axis_robustness,
)
from rpe.runner.w1_axis_statistics import (  # noqa: E402
    DEFAULT_COMMON_GRID_ARTIFACT,
    build_statistics_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build W1 axis robustness inventory, baseline reproduction, and native summaries"
    )
    parser.add_argument(
        "--stage",
        choices=("inventory", "reproduce", "native", "resample", "summarize", "infer", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--max-records-per-dataset", type=int, default=None)
    parser.add_argument("--common-grid-artifact", type=Path, default=DEFAULT_COMMON_GRID_ARTIFACT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.stage in {"summarize", "infer"}:
            path = build_statistics_run(
                config_path=arguments.config,
                common_grid_artifact=arguments.common_grid_artifact,
                output_root=arguments.output_root,
                stage=arguments.stage,
                resume=arguments.resume,
                worker_count=1 if arguments.worker_count is None else arguments.worker_count,
            )
            summary = type("StatisticsSummary", (), {
                "path": path, "run_id": path.name, "stage": arguments.stage, "status": "complete",
            })()
        else:
            summary = build_w1_axis_robustness(
                config_path=arguments.config,
                artifact_root=arguments.artifact_root,
                output_root=arguments.output_root,
                stage=arguments.stage,
                worker_count=arguments.worker_count,
                max_records_per_dataset=arguments.max_records_per_dataset,
                resume=arguments.resume,
            )
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "status": "failed",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "path": str(summary.path),
                "run_id": summary.run_id,
                "stage": summary.stage,
                "status": summary.status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
