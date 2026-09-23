from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_manuscript import build_phase6_manuscript  # noqa: E402
from rpe.runner.phase6_manuscript_verifier import (  # noqa: E402
    verify_phase6_manuscript,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or independently verify Phase 6 manuscript tooling",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--output-root", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "build":
        summary = build_phase6_manuscript(
            Path(args.config),
            Path(args.output_root),
        )
    else:
        summary = verify_phase6_manuscript(Path(args.run_path))
    print(f"{summary.run_id}\t{summary.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
