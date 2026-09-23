from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from raman_data.datasets import get_dataset_info, load_dataset

from tools.source_audit import validate_dataset_arrays


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "data" / "raw" / "ramanbench"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt_path(raw_root: Path, dataset_id: str) -> Path:
    return raw_root / "receipts" / f"{dataset_id}.json"


def _write_receipt(raw_root: Path, dataset_id: str, receipt: dict[str, Any]) -> None:
    path = _receipt_path(raw_root, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def audit_one(dataset_id: str, raw_root: Path) -> int:
    cache_root = raw_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
    info = get_dataset_info(dataset_id)
    try:
        dataset = load_dataset(dataset_id, cache_dir=str(cache_root), load_data=True)
        if dataset is None:
            raise RuntimeError("loader returned None")
        array_details = validate_dataset_arrays(
            dataset.spectra,
            dataset.targets,
            dataset.raman_shifts,
        )
        receipt = {
            "dataset_id": dataset_id,
            "status": "verified",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "loader_version": "raman-data==1.2.6",
            "source_url": info.metadata.get("source"),
            "data_license_claim": info.license,
            "task_type": info.task_type.name,
            "application_type": info.application_type.name,
            "checks": [
                "loader_completed",
                "nonempty_2d_spectra",
                "sample_target_count_match",
                "feature_wavenumber_count_match",
                "finite_spectra",
                "finite_wavenumbers",
            ],
            "array_details": array_details,
        }
        _write_receipt(raw_root, dataset_id, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        receipt = {
            "dataset_id": dataset_id,
            "status": "failed",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "loader_version": "raman-data==1.2.6",
            "source_url": info.metadata.get("source"),
            "data_license_claim": info.license,
            "task_type": info.task_type.name,
            "application_type": info.application_type.name,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_receipt(raw_root, dataset_id, receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


def configured_keys(raw_root: Path) -> list[str]:
    official = raw_root / "evidence" / "official"
    paths = [
        official / "configs__datasets__classification_all.json",
        official / "configs__datasets__regression_all.json",
    ]
    keys: set[str] = set()
    for path in paths:
        keys.update(json.loads(path.read_text(encoding="utf-8")))
    return sorted(keys)


def audit_all(raw_root: Path, timeout_seconds: int) -> int:
    logs = raw_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    keys = configured_keys(raw_root)
    if len(keys) != 77:
        raise RuntimeError(f"RamanBench v0.1.1 expected 77 configured keys, found {len(keys)}")

    for index, dataset_id in enumerate(keys, start=1):
        receipt_path = _receipt_path(raw_root, dataset_id)
        if receipt_path.exists():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing.get("status") == "verified":
                print(f"[{index:02d}/{len(keys)}] SKIP verified {dataset_id}", flush=True)
                continue

        print(f"[{index:02d}/{len(keys)}] START {dataset_id}", flush=True)
        log_path = logs / f"{dataset_id}.log"
        command = [
            sys.executable,
            "-m",
            "tools.audit_ramanbench",
            "--one",
            dataset_id,
            "--raw-root",
            str(raw_root),
        ]
        environment = dict(os.environ)
        environment["MPLCONFIGDIR"] = "/tmp/raman-preproc-eval-mpl"
        environment["HF_HOME"] = str(raw_root / "cache")
        environment["KAGGLEHUB_CACHE"] = str(raw_root / "cache")
        try:
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                    text=True,
                )
            print(
                f"[{index:02d}/{len(keys)}] EXIT {result.returncode} {dataset_id}",
                flush=True,
            )
        except subprocess.TimeoutExpired:
            _write_receipt(
                raw_root,
                dataset_id,
                {
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "started_at": _timestamp(),
                    "finished_at": _timestamp(),
                    "loader_version": "raman-data==1.2.6",
                    "error_type": "TimeoutExpired",
                    "error": f"loader exceeded {timeout_seconds} seconds",
                },
            )
            print(f"[{index:02d}/{len(keys)}] TIMEOUT {dataset_id}", flush=True)

    receipts = [_receipt_path(raw_root, key) for key in keys]
    missing = [path.stem for path in receipts if not path.exists()]
    if missing:
        raise RuntimeError(f"missing receipts: {missing}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    if args.one:
        return audit_one(args.one, raw_root)
    return audit_all(raw_root, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
