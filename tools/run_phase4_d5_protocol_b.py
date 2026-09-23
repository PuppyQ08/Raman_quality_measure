from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_d5_protocol_b import (  # noqa: E402
    Phase4D5ProtocolBError,
    build_phase4_d5_protocol_b,
)
from rpe.runner.phase4_d5_protocol_b_verifier import (  # noqa: E402
    verify_phase4_d5_protocol_b,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or independently verify Phase 4 D5 Protocol B")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--worker-count", type=int, default=16)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    verify.add_argument("--worker-count", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build":
            summary = build_phase4_d5_protocol_b(
                arguments.output_root, worker_count=arguments.worker_count
            )
        else:
            summary = verify_phase4_d5_protocol_b(
                arguments.run_path, worker_count=arguments.worker_count
            )
    except Phase4D5ProtocolBError as error:
        print(json.dumps({"error": str(error), "status": "error"}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "class_observation_count": summary.class_observation_count,
                "full_record_count": summary.full_record_count,
                "path": str(summary.path),
                "prediction_row_count": summary.prediction_row_count,
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
