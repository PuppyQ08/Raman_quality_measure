from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rpe.runner.phase4_d2_protocol_b import Phase4D2ProtocolBError, build_phase4_d2_protocol_b

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or independently verify Phase 4 D2 Protocol B")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("--output-root", type=Path, required=True); build.add_argument("--worker-count", type=int, default=16)
    verify = sub.add_parser("verify"); verify.add_argument("--run-path", type=Path, required=True); verify.add_argument("--worker-count", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            summary = build_phase4_d2_protocol_b(args.output_root, worker_count=args.worker_count)
        else:
            from rpe.runner.phase4_d2_protocol_b_verifier import verify_phase4_d2_protocol_b

            summary = verify_phase4_d2_protocol_b(args.run_path, worker_count=args.worker_count)
    except (Phase4D2ProtocolBError, ValueError) as error:
        print(json.dumps({"error": str(error), "status": "error"}, sort_keys=True), file=sys.stderr); return 2
    print(json.dumps({"path": str(summary.path), "run_id": summary.run_id, "status": summary.status, "prediction_row_count": summary.prediction_row_count, "class_observation_count": summary.class_observation_count}, sort_keys=True, separators=(",", ":")))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
