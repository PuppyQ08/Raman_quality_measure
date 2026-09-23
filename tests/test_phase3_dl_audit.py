from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rpe.runner.phase3_dl_audit import (
    DlAuditError,
    build_phase3_dl_audit_from_documents,
    derive_dl_candidate_audit,
    load_phase3_dl_audit_config,
)
from rpe.runner.phase3_dl_audit_verifier import verify_phase3_dl_audit

ROOT = Path(__file__).resolve().parents[1]


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def candidate(candidate_id: str, family: str | None = None) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "family_id": family or candidate_id.lower(),
        "label": candidate_id,
        "variants": ["default"],
    }


def evidence(candidate_id: str, *, runnable: bool) -> dict[str, object]:
    present = "verified" if runnable else "absent"
    return {
        "candidate_id": candidate_id,
        "bibliographic_identity_state": "verified",
        "task_role": "spectral_preprocessor",
        "code_state": present,
        "weight_state": present,
        "data_state": present,
        "code_license_state": present,
        "weight_license_state": present,
        "data_license_state": present,
        "environment_state": "constructible" if runnable else "not_attempted_blocked_upstream",
        "current_host_state": "compatible" if runnable else "not_evaluated",
        "smoke_level": "real_input_repeat_deterministic" if runnable else "not_run_blocked_upstream",
        "blocker_codes": [] if runnable else ["weights_missing"],
        "evidence_ids": [f"{candidate_id}-source"],
    }


class DlAuditDerivationTest(unittest.TestCase):
    def test_formal_config_binds_exact_roster_evidence_and_authorities(self) -> None:
        path = ROOT / "experiments/phase3/configs/dl_reproducibility_audit_v1.json"
        raw = path.read_bytes()
        self.assertEqual(raw, canonical(json.loads(raw)))
        config = load_phase3_dl_audit_config(path, project_root=ROOT)
        self.assertEqual(config.candidate_ids, tuple(f"DL{i:02d}" for i in range(1, 11)))
        self.assertEqual(config.minimum_runnable_family_count, 3)
        self.assertEqual(
            set(config.authorities),
            {"parent_plan", "phase0_source_audit", "phase3_registry_design", "step11_design", "source_versions", "source_manifest", "raw_checksums", "phase3_lock"},
        )
        self.assertEqual(set(config.evidence), {"source_checks", "artifact_checks", "environment_checks", "smoke_checks"})

    def test_runnable_requires_all_eight_conditions(self) -> None:
        row = derive_dl_candidate_audit(candidate("DL01", "deepr"), evidence("DL01", runnable=True))
        self.assertEqual(row["availability"], "runnable")
        self.assertEqual(row["publication_disposition"], "executable_candidate")

        missing = evidence("DL01", runnable=True)
        missing["weight_state"] = "absent"
        missing["blocker_codes"] = ["weights_missing"]
        row = derive_dl_candidate_audit(candidate("DL01", "deepr"), missing)
        self.assertEqual(row["availability"], "weights_missing")
        self.assertEqual(row["publication_disposition"], "reported_only")

    def test_hardware_and_artifact_blockers_are_not_conflated(self) -> None:
        row = evidence("DL02", runnable=True)
        row["current_host_state"] = "gpu_unavailable"
        row["smoke_level"] = "not_run_current_host"
        row["blocker_codes"] = ["current_host_gpu_unavailable"]
        derived = derive_dl_candidate_audit(candidate("DL02", "dscf"), row)
        self.assertEqual(derived["availability"], "dependency_missing")
        self.assertEqual(derived["code_state"], "verified")
        self.assertEqual(derived["weight_state"], "verified")
        self.assertIn("current_host_gpu_unavailable", derived["blocker_codes"])

    def test_deeper_paper_gate_never_follows_from_shape_smoke(self) -> None:
        row = evidence("DL01", runnable=True)
        row["smoke_level"] = "synthetic_shape_smoke"
        row["data_state"] = "absent"
        row["blocker_codes"] = ["official_paired_data_missing"]
        derived = derive_dl_candidate_audit(candidate("DL01", "deepr"), row)
        self.assertEqual(derived["availability"], "data_missing")
        self.assertEqual(derived["deepr_reproduction_state"], "not_evaluable_data_missing")

    def test_ambiguous_identity_is_reported_only_before_role_exclusion(self) -> None:
        row = evidence("DL06", runnable=False)
        row["bibliographic_identity_state"] = "ambiguous_multiple_methods"
        row["task_role"] = "representation_model"
        derived = derive_dl_candidate_audit(candidate("DL06", "smae_ramanmae"), row)
        self.assertEqual(derived["availability"], "reported_only")
        self.assertEqual(derived["publication_disposition"], "reported_only")


class DlAuditArtifactTest(unittest.TestCase):
    def documents(self) -> tuple[dict[str, object], list[dict[str, object]]]:
        candidates = [candidate(f"DL{index:02d}") for index in range(1, 11)]
        config = {
            "schema_version": "phase3-dl-reproducibility-audit-config-v1",
            "experiment_id": "phase3-dl-reproducibility-audit-v1",
            "candidate_ids": [value["candidate_id"] for value in candidates],
            "minimum_runnable_family_count": 3,
            "claim_boundary": "reproducibility_audit_not_benchmark_result",
            "fixture": True,
        }
        rows = [evidence(f"DL{index:02d}", runnable=index <= 2) for index in range(1, 11)]
        return config, candidates, rows

    def test_complete_audit_does_not_imply_three_family_gate(self) -> None:
        config, candidates, rows = self.documents()
        with tempfile.TemporaryDirectory() as temporary:
            result = build_phase3_dl_audit_from_documents(
                Path(temporary), config=config, candidates=candidates, evidence_rows=rows
            )
            marker = json.loads((result.path / "complete.json").read_bytes())
            gate = json.loads((result.path / "gate.json").read_bytes())
            self.assertTrue(marker["audit_complete"])
            self.assertEqual(gate["runnable_family_count"], 2)
            self.assertFalse(gate["parent_minimum_three_runnable_pass"])
            self.assertEqual(
                {path.name for path in result.path.iterdir() if path.is_file()},
                {
                    "config.json", "candidates.jsonl", "source_checks.jsonl",
                    "artifact_checks.jsonl", "environment_checks.jsonl",
                    "smoke_checks.jsonl", "reproducibility_table.csv",
                    "summary.json", "gate.json", "manifest.json",
                    "complete.json", "SHA256SUMS",
                },
            )
            self.assertEqual(verify_phase3_dl_audit(result.path).run_id, result.run_id)

    def test_duplicate_roster_and_unsupported_positive_claim_fail_closed(self) -> None:
        config, candidates, rows = self.documents()
        duplicate = copy.deepcopy(candidates)
        duplicate[-1]["candidate_id"] = "DL09"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(DlAuditError, "candidate roster"):
                build_phase3_dl_audit_from_documents(Path(temporary), config=config, candidates=duplicate, evidence_rows=rows)
        rows[2]["code_state"] = "verified"
        rows[2]["evidence_ids"] = []
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(DlAuditError, "evidence_ids"):
                build_phase3_dl_audit_from_documents(Path(temporary), config=config, candidates=candidates, evidence_rows=rows)

    def test_verifier_rejects_checksum_consistent_status_tampering(self) -> None:
        config, candidates, rows = self.documents()
        with tempfile.TemporaryDirectory() as temporary:
            result = build_phase3_dl_audit_from_documents(Path(temporary), config=config, candidates=candidates, evidence_rows=rows)
            table = result.path / "reproducibility_table.csv"
            parsed = list(csv.DictReader(io.StringIO(table.read_text())))
            parsed[2]["availability"] = "runnable"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=list(parsed[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(parsed)
            old = table.read_bytes(); new = output.getvalue().encode(); table.write_bytes(new)
            sums = result.path / "SHA256SUMS"
            sums.write_text(sums.read_text().replace(hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()))
            with self.assertRaisesRegex(ValueError, "reproducibility table"):
                verify_phase3_dl_audit(result.path)


if __name__ == "__main__":
    unittest.main()
