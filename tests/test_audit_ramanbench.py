import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ramanbench import (  # noqa: E402
    _receipt_path,
    _write_receipt,
    configured_keys,
)


class RamanBenchAuditTest(unittest.TestCase):
    def test_configured_keys_returns_sorted_union(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_root = Path(temporary_directory)
            official = raw_root / "evidence" / "official"
            official.mkdir(parents=True)
            (official / "configs__datasets__classification_all.json").write_text(
                json.dumps(["zeta", "alpha"]),
                encoding="utf-8",
            )
            (official / "configs__datasets__regression_all.json").write_text(
                json.dumps(["beta", "alpha"]),
                encoding="utf-8",
            )

            self.assertEqual(configured_keys(raw_root), ["alpha", "beta", "zeta"])

    def test_write_receipt_replaces_existing_document(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_root = Path(temporary_directory)
            _write_receipt(raw_root, "sample", {"status": "failed"})
            _write_receipt(raw_root, "sample", {"status": "verified"})

            receipt = json.loads(
                _receipt_path(raw_root, "sample").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt, {"status": "verified"})
            self.assertFalse(
                _receipt_path(raw_root, "sample").with_suffix(".json.tmp").exists()
            )


if __name__ == "__main__":
    unittest.main()
