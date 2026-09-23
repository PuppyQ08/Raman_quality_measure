from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase3_dl_audit import DlAuditError, build_phase3_dl_audit
from rpe.runner.phase3_dl_audit_verifier import (
    DlAuditVerificationError,
    verify_phase3_dl_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the Phase 3 DL reproducibility audit")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            summary = build_phase3_dl_audit(args.output_root)
            status = "built"
        else:
            summary = verify_phase3_dl_audit(args.run_path, project_root=ROOT)
            status = "verified"
    except (DlAuditError, DlAuditVerificationError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error), "error_type": type(error).__name__, "status": "error"}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps({"path": str(summary.path), "run_id": summary.run_id, "status": status}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
