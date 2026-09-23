from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_denoising_evidence import (
    build_phase6_denoising_evidence,
)
from rpe.runner.phase6_denoising_evidence_verifier import (
    verify_phase6_denoising_evidence,
)  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or independently verify Phase 6 denoising evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output-root", required=True)
    build.add_argument("--worker-count", type=int, default=8)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "build":
        summary = build_phase6_denoising_evidence(
            Path(args.output_root),
            worker_count=int(args.worker_count),
        )
    else:
        summary = verify_phase6_denoising_evidence(
            Path(args.run_path),
        )
    print(f"{summary.run_id}\t{summary.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
