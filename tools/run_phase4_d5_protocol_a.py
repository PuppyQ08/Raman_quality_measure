from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_d5_protocol_a import Phase4D5ProtocolAError, build_phase4_d5_protocol_a
from rpe.runner.phase4_d5_protocol_a_verifier import verify_phase4_d5_protocol_a


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--worker-count", type=int, default=16)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    verify.add_argument("--worker-count", type=int, default=12)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build":
            summary = build_phase4_d5_protocol_a(arguments.output_root, worker_count=arguments.worker_count)
        else:
            summary = verify_phase4_d5_protocol_a(arguments.run_path, worker_count=arguments.worker_count)
    except (OSError, ValueError, Phase4D5ProtocolAError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 1
    print(json.dumps({"path": str(summary.path), "run_id": summary.run_id, "status": summary.status, "prediction_row_count": summary.prediction_row_count, "metric_row_count": summary.metric_row_count, "class_observation_count": summary.class_observation_count}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
