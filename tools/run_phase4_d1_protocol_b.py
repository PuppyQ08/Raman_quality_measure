from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_d1_protocol_b import Phase4D1ProtocolBError, build_phase4_d1_protocol_b
from rpe.runner.phase4_d1_protocol_b_verifier import (
    Phase4D1ProtocolBVerifierError,
    verify_phase4_d1_protocol_b,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify Phase 4 D1 Protocol B")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--worker-count", type=int, default=5)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    verify.add_argument("--worker-count", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            summary = build_phase4_d1_protocol_b(args.output_root, worker_count=args.worker_count)
        else:
            summary = verify_phase4_d1_protocol_b(args.run_path, worker_count=args.worker_count)
    except (OSError, ValueError, Phase4D1ProtocolBError, Phase4D1ProtocolBVerifierError) as error:
        print(
            json.dumps(
                {"error": str(error), "error_type": type(error).__name__, "status": "failed"},
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
                "status": summary.status,
                "prediction_row_count": summary.prediction_row_count,
                "class_observation_count": summary.class_observation_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
