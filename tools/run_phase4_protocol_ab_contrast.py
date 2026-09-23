from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_protocol_ab_contrast import (  # noqa: E402
    build_phase4_protocol_ab_contrast,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or independently verify the Phase 4 Protocol-A/B contrast"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--worker-count", type=int, default=5)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    verify.add_argument("--worker-count", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build":
            summary = build_phase4_protocol_ab_contrast(
                arguments.output_root, worker_count=arguments.worker_count
            )
        else:
            from rpe.runner.phase4_protocol_ab_contrast_verifier import (
                verify_phase4_protocol_ab_contrast,
            )

            summary = verify_phase4_protocol_ab_contrast(
                arguments.run_path, worker_count=arguments.worker_count
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
                "bootstrap_result_count": summary.bootstrap_result_count,
                "endpoint_status_count": summary.endpoint_status_count,
                "holm_slot_count": summary.holm_slot_count,
                "path": str(summary.path),
                "run_id": summary.run_id,
                "status": summary.status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
