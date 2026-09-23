from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import scipy
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from scipy import stats
import numpy as np

from rpe.runner.phase6_appendix_verify_bacteria import v_build_bacteria_sensitivity
from rpe.runner.phase6_appendix_verify_leakage import v_build_leakage
from rpe.runner.phase6_appendix_verify_rruff import v_build_rruff_sensitivity
from rpe.runner.phase6_appendix_verify_sugar import v_build_sugar_sensitivity


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/appendix_audits_v1.json"
FIXTURE_ID = "phase6-appendix-audits-tiny-verifier-fixture-v1"
_IDENTITY_CODE_FILES = (
    "rpe/runner/phase6_appendix_audits.py", "rpe/runner/phase6_appendix_audits_verifier.py",
    "rpe/runner/phase6_appendix_bacteria.py", "rpe/runner/phase6_appendix_sugar.py",
    "rpe/runner/phase6_appendix_rruff.py", "rpe/runner/phase6_appendix_leakage.py",
    "rpe/runner/phase6_appendix_metric_core.py", "rpe/runner/phase6_appendix_verifier_science.py",
    "rpe/runner/phase6_appendix_verify_bacteria.py", "rpe/runner/phase6_appendix_verify_sugar.py",
    "rpe/runner/phase6_appendix_verify_rruff.py", "rpe/runner/phase6_appendix_verify_leakage.py",
    "tools/run_phase6_appendix_audits.py", "experiments/phase6/configs/appendix_audits_v1.json",
)


class AppendixAuditsVerificationError(ValueError):
    """Raised when independent reconstruction or byte comparison fails."""


@dataclass(frozen=True)
class AppendixAuditsVerificationSummary:
    run_id: str
    path: Path
    status: str
    counts: Mapping[str, int]


@dataclass(frozen=True)
class _Config:
    path: Path
    raw: Mapping[str, object]
    raw_bytes: bytes
    sha256: str


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            _plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, AbcMapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _load_config(path: Path) -> _Config:
    raw_bytes = Path(path).read_bytes()
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AppendixAuditsVerificationError("invalid verifier config bytes") from error
    if not isinstance(document, dict) or raw_bytes != _canonical(document):
        raise AppendixAuditsVerificationError("verifier config must be canonical JSON")
    if document.get("schema_version") != "phase6-appendix-audits-v1":
        raise AppendixAuditsVerificationError("unexpected verifier config schema")
    artifact = document.get("artifact_contract", {})
    if (
        artifact.get("configured_payload_count") != 16
        or tuple(artifact.get("payload_files", ())) != (
            "config.json",
            "authority_bridge.json",
            "preflight.json",
            "audit_status.csv",
            "resampling_alignment.csv",
            "resampling_rank_stability.csv",
            "normalization_alignment.csv",
            "normalization_rank_stability.csv",
            "peak_tolerance_phase4_alignment.csv",
            "peak_tolerance_phase4_rank_stability.csv",
            "peak_tolerance_phase6_system.csv",
            "peak_tolerance_phase6_rank_stability.csv",
            "leakage_boundaries.jsonl",
            "leakage_duplicate_candidates.jsonl",
            "summary.md",
            "manifest.json",
        )
    ):
        raise AppendixAuditsVerificationError("verifier payload contract drift")
    return _Config(
        path=Path(path),
        raw=MappingProxyType(document),
        raw_bytes=raw_bytes,
        sha256=_sha(raw_bytes),
    )


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=tuple(fields),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else format(value, ".17g")
                    if isinstance(value, float)
                    else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _tie_count(values: Sequence[float]) -> int:
    counts: dict[float, int] = {}
    for value in values:
        counts[float(value)] = counts.get(float(value), 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def _rank_stability(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    statistic: str,
    metric_ids: Sequence[str],
) -> dict[str, object]:
    ids = tuple(metric_ids)
    base = {
        "tau_b": None,
        "reference_tie_count": None,
        "candidate_tie_count": None,
        "max_abs_rank_displacement": None,
        "stable": False,
    }
    if set(reference) != set(ids) or set(candidate) != set(ids):
        return {
            **base,
            "state": "not_evaluable",
            "reason_code": "not_evaluable_incomplete_metric_set",
        }
    direction = 1.0 if statistic == "ag" else -1.0
    ref_values = [direction * float(reference[item]) for item in ids]
    cur_values = [direction * float(candidate[item]) for item in ids]
    ref_rank = stats.rankdata(ref_values, method="average")
    cur_rank = stats.rankdata(cur_values, method="average")
    ref_ties = _tie_count(ref_values)
    cur_ties = _tie_count(cur_values)
    if ref_ties == len(ids) - 1 or cur_ties == len(ids) - 1:
        return {
            **base,
            "reference_tie_count": ref_ties,
            "candidate_tie_count": cur_ties,
            "state": "not_evaluable",
            "reason_code": "not_evaluable_all_tied_ranking",
        }
    tau = float(
        stats.kendalltau(ref_rank, cur_rank, variant="b", nan_policy="propagate").statistic
    )
    stable = tau > 0.9
    return {
        **base,
        "tau_b": tau,
        "reference_tie_count": ref_ties,
        "candidate_tie_count": cur_ties,
        "max_abs_rank_displacement": float(
            max(abs(float(left) - float(right)) for left, right in zip(ref_rank, cur_rank, strict=True))
        ),
        "stable": stable,
        "state": "complete_numeric",
        "reason_code": "" if stable else "complete_ranking_instability_or_reversal",
    }


def _curve_value(row: Mapping[str, object], tolerance: int) -> tuple[object, object, object, object, object, object]:
    value = row["curve_by_tolerance_cm1"][str(tolerance)]
    if isinstance(value, Mapping):
        return (
            value["estimate"],
            value["interval_lower"],
            value["interval_upper"],
            value.get("alpha_estimates", ()),
            value.get("contributing_class_counts", ()),
            value.get("state", row.get("state", "complete")),
        )
    interval = row["interval_by_tolerance_cm1"][str(tolerance)]
    return (
        value,
        interval[0],
        interval[1],
        row.get("alpha_estimates_json", {}),
        row.get("contributing_class_counts_json", {}),
        row.get("state", "complete"),
    )


def _project_phase6_peak_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    system_ids: Sequence[str],
    endpoint_manifest: Sequence[Mapping[str, object]],
    tolerances: Sequence[int],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    systems = tuple(system_ids)
    endpoints = tuple(endpoint_manifest)
    by_key = {
        (str(row["system_id"]), str(row["endpoint_id"])): row
        for row in rows
    }
    expected = {
        (system_id, str(endpoint["endpoint_id"]))
        for system_id in systems
        for endpoint in endpoints
    }
    if set(by_key) != expected:
        raise AppendixAuditsVerificationError("phase6 peak verifier rows must cover every system/endpoint pair")
    system_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for system_id in systems:
        for endpoint in endpoints:
            source = by_key[(system_id, str(endpoint["endpoint_id"]))]
            for tolerance in tolerances:
                estimate, lower, upper, alpha, counts, state = _curve_value(source, tolerance)
                system_rows.append(
                    {
                        "system_id": system_id,
                        "family_id": source["family_id"],
                        "method_id": source["method_id"],
                        "endpoint_id": endpoint["endpoint_id"],
                        "preferred_direction": endpoint["preferred_direction"],
                        "tolerance_cm1": tolerance,
                        "estimate": estimate,
                        "interval_lower": lower,
                        "interval_upper": upper,
                        "design_class_count": source["design_class_count"],
                        "alpha_estimates_json": alpha,
                        "contributing_class_counts_json": counts,
                        "state": state,
                        "reason_code": source.get("reason_code", ""),
                    }
                )
    for endpoint in endpoints:
        direction = str(endpoint["preferred_direction"])
        if direction == "non_monotonic":
            continue
        endpoint_id = str(endpoint["endpoint_id"])
        reference = {
            system_id: float(_curve_value(by_key[(system_id, endpoint_id)], 2)[0])
            for system_id in systems
        }
        statistic = "acc_cross" if direction == "higher_is_better" else "ag"
        for tolerance in tolerances:
            if tolerance == 2:
                continue
            candidate = {
                system_id: float(_curve_value(by_key[(system_id, endpoint_id)], tolerance)[0])
                for system_id in systems
            }
            rank = _rank_stability(reference, candidate, statistic=statistic, metric_ids=systems)
            rank_rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "preferred_direction": direction,
                    "tolerance_cm1": tolerance,
                    "reference_tolerance_cm1": 2,
                    "tau_b": rank["tau_b"],
                    "reference_tie_count": rank["reference_tie_count"],
                    "candidate_tie_count": rank["candidate_tie_count"],
                    "max_abs_rank_displacement": rank["max_abs_rank_displacement"],
                    "stable": rank["stable"],
                    "system_count": len(systems),
                    "state": rank["state"],
                    "reason_code": rank["reason_code"],
                }
            )
    return tuple(system_rows), tuple(rank_rows)


def _condition_sort(component: str, condition_id: object) -> tuple[int, int]:
    value = str(condition_id)
    if component == "resampling":
        interpolation, spacing = value.split(":", 1)
        return (
            ("linear", "cubic", "pchip").index(interpolation),
            ("0.5", "1", "2").index(f"{float(spacing):g}"),
        )
    if component == "normalization":
        return (("none", "maximum", "area", "snv").index(value), 0)
    if component == "phase4_peak_tolerance":
        return ((1, 2, 4, 8).index(int(value.split(":", 1)[1])), 0)
    return (0, 0)


def _validate_ledger(
    run: Path,
    *,
    expected_names: Sequence[str] | None,
    terminal_name: str,
) -> None:
    ledger = run / "SHA256SUMS"
    if not ledger.is_file():
        raise AppendixAuditsVerificationError("artifact SHA256SUMS missing")
    lines = ledger.read_text(encoding="utf-8").splitlines()
    if expected_names is not None and len(lines) != len(expected_names):
        raise AppendixAuditsVerificationError("artifact ledger must contain exactly seventeen entries")
    names: list[str] = []
    for line in lines:
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise AppendixAuditsVerificationError("artifact ledger is malformed")
        digest, name = fields
        names.append(name)
        member = run / name
        if not member.is_file() or _sha_file(member) != digest:
            raise AppendixAuditsVerificationError(f"artifact ledger digest mismatch for {name}")
    if expected_names is not None and tuple(names) != tuple(expected_names):
        raise AppendixAuditsVerificationError("artifact ledger inventory/order mismatch")
    if terminal_name not in {"complete.json", "failed.json"}:
        raise AppendixAuditsVerificationError(
            "artifact terminal_name must be complete.json or failed.json"
        )
    terminal = run / terminal_name
    other_terminal = run / (
        "failed.json" if terminal_name == "complete.json" else "complete.json"
    )
    if not terminal.is_file() or other_terminal.exists():
        raise AppendixAuditsVerificationError(
            "artifact must have exactly one declared terminal"
        )


def _validate_artifact_inventory(path: Path, config: _Config) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    expected_payloads = tuple(str(name) for name in config.raw["artifact_contract"]["payload_files"])
    expected_names = expected_payloads + ("complete.json",)
    if not path.is_dir():
        raise AppendixAuditsVerificationError("artifact run path is missing")
    observed = {
        item.name: item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }
    if set(observed) != set(expected_names) | {"SHA256SUMS"}:
        raise AppendixAuditsVerificationError("artifact inventory mismatch")
    _validate_ledger(path, expected_names=expected_names, terminal_name="complete.json")
    if observed["config.json"] != config.raw_bytes:
        raise AppendixAuditsVerificationError("artifact config.json bytes differ from canonical project config")
    try:
        manifest = json.loads(observed["manifest.json"])
        authority_bridge = json.loads(observed["authority_bridge.json"])
    except json.JSONDecodeError as error:
        raise AppendixAuditsVerificationError("artifact manifest or authority bridge is not valid JSON") from error
    return observed, manifest, authority_bridge


def _validate_authorities(config: _Config, project_root: Path) -> Mapping[str, object]:
    receipts: dict[str, object] = {}
    for name, raw in config.raw["authorities"].items():
        identity = dict(raw)
        path = project_root / str(identity["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(identity["byte_count"])
            or _sha_file(path) != str(identity["sha256"])
        ):
            raise AppendixAuditsVerificationError(f"authority byte identity mismatch: {name}")
        receipts[str(name)] = identity
    index_path = project_root / str(config.raw["authorities"]["final_phase4_parent_panel_index"]["path"])
    with index_path.open(newline="", encoding="utf-8") as stream:
        index_rows = tuple(csv.DictReader(stream))
    panels = tuple(dict(row) for row in config.raw["panel_manifest"])
    if tuple(row["panel_id"] for row in index_rows) != tuple(row["panel_id"] for row in panels):
        raise AppendixAuditsVerificationError("phase4 parent panel index order drift")
    phase4_parent_panels: list[dict[str, object]] = []
    for declared, indexed in zip(panels, index_rows, strict=True):
        if (
            declared["parent_path"] != indexed["relative_path"]
            or declared["parent_key"] != indexed["parent_key"]
        ):
            raise AppendixAuditsVerificationError("phase4 parent panel index contents drift")
        run = project_root / str(declared["parent_path"])
        terminal_name = str(indexed["terminal_name"])
        _validate_ledger(
            run,
            expected_names=None,
            terminal_name=terminal_name,
        )
        marker = run / terminal_name
        if _sha_file(run / "SHA256SUMS") != indexed["ledger_sha256"] or _sha_file(marker) != indexed["terminal_sha256"]:
            raise AppendixAuditsVerificationError("phase4 parent panel ledger or terminal identity drift")
        phase4_parent_panels.append(
            {
                "panel_id": indexed["panel_id"],
                "parent_path": indexed["relative_path"],
                "ledger_sha256": indexed["ledger_sha256"],
                "terminal_name": indexed["terminal_name"],
                "terminal_sha256": indexed["terminal_sha256"],
            }
        )
    peak_root = (project_root / str(config.raw["authorities"]["phase6_peak_checksum_ledger"]["path"])).parent
    expected_inventory = dict(config.raw["phase6_peak_retained_payloads"]["verified_inventory_sha256"])
    peak_ledger = peak_root / "SHA256SUMS"
    lines = peak_ledger.read_text(encoding="utf-8").splitlines()
    observed_inventory: dict[str, str] = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        observed_inventory[name] = digest
        if _sha_file(peak_root / name) != digest:
            raise AppendixAuditsVerificationError("phase6 peak retained payload digest drift")
    if observed_inventory != expected_inventory:
        raise AppendixAuditsVerificationError("phase6 peak retained payload inventory drift")
    receipts["phase4_parent_panels"] = tuple(phase4_parent_panels)
    return MappingProxyType(receipts)


def _panel_meta(config: _Config) -> dict[str, dict[str, object]]:
    return {str(row["panel_id"]): dict(row) for row in config.raw["panel_manifest"]}


def _numeric_panels(config: _Config) -> tuple[str, ...]:
    return tuple(
        str(panel_id)
        for panel_id in config.raw["fixed_panel_order"]
        if str(panel_id) != "d1_b_closed"
    )


def _fixture_resampling_conditions(config: _Config) -> tuple[str, ...]:
    return tuple(
        f"{interpolator}:{float(spacing):g}"
        for interpolator in ("linear", "cubic", "pchip")
        for spacing in config.raw["resampling_conditions"]["spacing_cm1"]
    )


def _fixture_normalization_conditions(config: _Config) -> tuple[str, ...]:
    return tuple(str(value) for value in config.raw["normalization_conditions"])


def _fixture_phase4_tolerances() -> tuple[str, ...]:
    return ("tolerance:1", "tolerance:2", "tolerance:4", "tolerance:8")


def _fixture_phase4_tables(
    config: _Config,
    *,
    component: str,
    conditions: tuple[str, ...],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    lookup = _panel_meta(config)
    reference = {
        "resampling": "native",
        "normalization": "none",
        "phase4_peak_tolerance": "tolerance:2",
    }[component]
    alignments: list[dict[str, object]] = []
    ranks: list[dict[str, object]] = []
    for panel_index, panel_id in enumerate(_numeric_panels(config)):
        meta = lookup[panel_id]
        for condition_index, condition_id in enumerate(conditions):
            basis = (panel_index + 1) * 1000 + (condition_index + 1) * 100
            for metric_index, metric_id in enumerate(config.raw["metric_ids"]):
                alignments.append(
                    {
                        "panel_id": panel_id,
                        "endpoint_id": meta["endpoint_id"],
                        "protocol_id": meta["protocol_id"],
                        "condition_id": condition_id,
                        "metric_output_id": metric_id,
                        "ag": float(basis + metric_index) / 100.0,
                        "acc_cross": float(500000 - basis - metric_index) / 1000000.0,
                        "state": "complete_numeric",
                        "reason_code": "",
                    }
                )
            for statistic in ("ag", "acc_cross"):
                ranks.append(
                    {
                        "panel_id": panel_id,
                        "endpoint_id": meta["endpoint_id"],
                        "protocol_id": meta["protocol_id"],
                        "condition_id": condition_id,
                        "statistic": statistic,
                        "reference_condition_id": reference,
                        "tau_b": 1.0,
                        "reference_tie_count": 0,
                        "candidate_tie_count": 0,
                        "max_abs_rank_displacement": 0.0,
                        "stable": True,
                        "state": "complete_numeric",
                        "reason_code": "",
                    }
                )
    return tuple(alignments), tuple(ranks)


def _fixture_system_ids() -> tuple[str, ...]:
    return tuple(f"fixture-system-{index:02d}" for index in range(21))


def _fixture_phase6_rows(config: _Config, system_ids: Sequence[str]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for system_index, system_id in enumerate(system_ids):
        for endpoint_index, endpoint in enumerate(config.raw["phase6_endpoint_manifest"]):
            base = float((system_index + 1) * 100 + endpoint_index) / 1000.0
            direction = str(endpoint["preferred_direction"])
            if direction == "higher_is_better":
                curve = {"1": base - 0.01, "2": base, "4": base - 0.001, "8": base - 0.002}
            elif direction == "lower_is_better":
                curve = {"1": base + 0.01, "2": base, "4": base + 0.001, "8": base + 0.002}
            else:
                curve = {"1": base, "2": base, "4": base, "8": base}
            rows.append(
                {
                    "system_id": system_id,
                    "family_id": f"fixture-family-{system_index % 3}",
                    "method_id": f"fixture-method-{system_index:02d}",
                    "endpoint_id": endpoint["endpoint_id"],
                    "curve_by_tolerance_cm1": curve,
                    "interval_by_tolerance_cm1": {
                        key: [value - 0.05, value + 0.05] for key, value in curve.items()
                    },
                    "design_class_count": 4,
                    "alpha_estimates_json": {"0.05": curve["2"]},
                    "contributing_class_counts_json": {"fixture-class": 4},
                    "state": "complete_numeric",
                    "reason_code": "",
                }
            )
    return tuple(rows)


def _fixture_leakage_rows(config: _Config) -> tuple[dict[str, object], ...]:
    inherited = {
        "phase3_denoising_fit_vs_transform",
        "phase6_baseline_d4_fit_selection_test_lifecycle",
        "phase6_denoising_identity_equivalence",
        "phase6_peak_claim_boundary",
    }
    rows: list[dict[str, object]] = []
    for index, definition in enumerate(config.raw["leakage_boundary_definitions"]):
        boundary_id = str(definition["boundary_id"])
        if boundary_id == "cross_dataset_provenance_d1_d2_d4_d5":
            status = "not_evaluable"
            reason = "not_evaluable_insufficient_cross_dataset_identity"
        elif boundary_id in inherited:
            status = "pass_via_parent_hash"
            reason = ""
        else:
            status = "pass"
            reason = ""
        rows.append(
            {
                "boundary_id": boundary_id,
                "producer_authorities": [],
                "consumer_authorities": [],
                "fit_roles": list(definition.get("fit_roles", ())),
                "selection_roles": list(definition.get("selection_roles", ())),
                "evaluation_roles": list(definition.get("evaluation_roles", ())),
                "leakage_entity": definition.get("leakage_entity", ""),
                "expected_overlap_semantics": definition.get("expected_overlap_semantics", ""),
                "recomputed_digests": {"fixture_digest": f"fixture-{index:02d}"},
                "recomputed_counts": {"fixture_count": index},
                "exact_overlap_count": 0,
                "near_candidate_count": 0,
                "status": status,
                "reason_code": reason,
                "evidence_paths": [f"fixture://{boundary_id}"],
            }
        )
    return tuple(rows)


def _fixture_audit_status(
    config: _Config,
    *,
    system_ids: Sequence[str],
    leakage_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    lookup = _panel_meta(config)
    statuses: list[dict[str, object]] = []
    order = 0
    for component, conditions in (
        ("resampling", _fixture_resampling_conditions(config)),
        ("normalization", _fixture_normalization_conditions(config)),
        ("phase4_peak_tolerance", _fixture_phase4_tolerances()),
    ):
        for panel_id in config.raw["fixed_panel_order"]:
            meta = lookup[str(panel_id)]
            for condition_id in conditions:
                if str(panel_id) == "d1_b_closed":
                    statuses.append(
                        {
                            "component": component,
                            "status_order": order,
                            "panel_id": str(panel_id),
                            "endpoint_id": meta["endpoint_id"],
                            "protocol_id": meta["protocol_id"],
                            "condition_id": condition_id,
                            "state": "not_evaluable",
                            "reason_code": "not_evaluable_failed_alpha0_equivalence",
                            "numerical_row_count": 0,
                            "rank_row_count": 0,
                            "evidence_key": "fixture_d1_b_closure",
                        }
                    )
                else:
                    statuses.append(
                        {
                            "component": component,
                            "status_order": order,
                            "panel_id": str(panel_id),
                            "endpoint_id": meta["endpoint_id"],
                            "protocol_id": meta["protocol_id"],
                            "condition_id": condition_id,
                            "state": "complete_numeric",
                            "reason_code": "",
                            "numerical_row_count": 13,
                            "rank_row_count": 2,
                            "evidence_key": f"fixture_{component}",
                        }
                    )
                order += 1
    for endpoint in config.raw["phase6_endpoint_manifest"]:
        for tolerance in (1, 2, 4, 8):
            statuses.append(
                {
                    "component": "phase6_peak_tolerance",
                    "status_order": order,
                    "panel_id": "",
                    "endpoint_id": endpoint["endpoint_id"],
                    "protocol_id": "",
                    "condition_id": f"tolerance:{tolerance}",
                    "state": "complete_numeric",
                    "reason_code": "",
                    "numerical_row_count": len(tuple(system_ids)),
                    "rank_row_count": 0 if endpoint["preferred_direction"] == "non_monotonic" or tolerance == 2 else 1,
                    "evidence_key": "fixture_step5_projection",
                }
            )
            order += 1
    for row in leakage_rows:
        statuses.append(
            {
                "component": "leakage_boundaries",
                "status_order": order,
                "panel_id": "",
                "endpoint_id": "",
                "protocol_id": "",
                "condition_id": row["boundary_id"],
                "state": row["status"],
                "reason_code": row["reason_code"],
                "numerical_row_count": 0,
                "rank_row_count": 0,
                "evidence_key": "fixture_leakage",
            }
        )
        order += 1
    return tuple(statuses)


def _fixture_inputs(config: _Config) -> dict[str, object]:
    system_ids = _fixture_system_ids()
    leakage_rows = _fixture_leakage_rows(config)
    resampling_alignment, resampling_rank = _fixture_phase4_tables(
        config,
        component="resampling",
        conditions=_fixture_resampling_conditions(config),
    )
    normalization_alignment, normalization_rank = _fixture_phase4_tables(
        config,
        component="normalization",
        conditions=_fixture_normalization_conditions(config),
    )
    peak_alignment, peak_rank = _fixture_phase4_tables(
        config,
        component="phase4_peak_tolerance",
        conditions=_fixture_phase4_tolerances(),
    )
    return {
        "authority_bridge": {
            "fixture_kind": FIXTURE_ID,
            "fixture_version": 1,
            "schema_version": config.raw["schema_version"],
            "expected_rows": dict(config.raw["expected_rows"]),
        },
        "identity": {
            "fixture_kind": FIXTURE_ID,
            "fixture_version": 1,
            "worker_count_identity_excluded": True,
        },
        "phase6_system_ids": system_ids,
        "phase6_rows": _fixture_phase6_rows(config, system_ids),
        "tables": {
            "audit_status": _fixture_audit_status(
                config,
                system_ids=system_ids,
                leakage_rows=leakage_rows,
            ),
            "resampling_alignment": resampling_alignment,
            "resampling_rank_stability": resampling_rank,
            "normalization_alignment": normalization_alignment,
            "normalization_rank_stability": normalization_rank,
            "peak_tolerance_phase4_alignment": peak_alignment,
            "peak_tolerance_phase4_rank_stability": peak_rank,
            "leakage_duplicate_candidates": (),
            "leakage_boundaries": leakage_rows,
        },
    }


def _adapter_tables(results: Sequence[object]) -> dict[str, tuple[Mapping[str, object], ...]]:
    names = (
        "audit_status",
        "resampling_alignment",
        "resampling_rank_stability",
        "normalization_alignment",
        "normalization_rank_stability",
        "peak_tolerance_phase4_alignment",
        "peak_tolerance_phase4_rank_stability",
    )
    return {
        name: tuple(dict(row) for result in results for row in getattr(result, name))
        for name in names
    }


def _canonicalize_formal_tables(
    config: _Config,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    leakage: object,
    phase6_rows: Sequence[Mapping[str, object]],
    system_ids: Sequence[str],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    panel_order = {
        str(value): index for index, value in enumerate(config.raw["fixed_panel_order"])
    }
    metric_order = {
        str(value): index for index, value in enumerate(config.raw["metric_ids"])
    }
    system_order = {str(value): index for index, value in enumerate(system_ids)}
    endpoint_order = {
        str(row["endpoint_id"]): index
        for index, row in enumerate(config.raw["phase6_endpoint_manifest"])
    }
    boundary_order = {
        str(row["boundary_id"]): index
        for index, row in enumerate(config.raw["leakage_boundary_definitions"])
    }
    result = {
        name: tuple(dict(row) for row in rows) for name, rows in tables.items()
    }
    for component, alignment_name, rank_name in (
        ("resampling", "resampling_alignment", "resampling_rank_stability"),
        ("normalization", "normalization_alignment", "normalization_rank_stability"),
        ("phase4_peak_tolerance", "peak_tolerance_phase4_alignment", "peak_tolerance_phase4_rank_stability"),
    ):
        def base(row: Mapping[str, object]) -> tuple[int, int, int]:
            return (
                panel_order.get(str(row.get("panel_id")), 999),
                *_condition_sort(component, row.get("condition_id")),
            )

        result[alignment_name] = tuple(
            sorted(
                result[alignment_name],
                key=lambda row: (
                    *base(row),
                    metric_order.get(str(row.get("metric_output_id")), 999),
                ),
            )
        )
        result[rank_name] = tuple(
            sorted(
                result[rank_name],
                key=lambda row: (
                    *base(row),
                    ("ag", "acc_cross").index(str(row.get("statistic"))),
                ),
            )
        )
    system_rows, rank_rows = _project_phase6_peak_rows(
        phase6_rows,
        system_ids=system_ids,
        endpoint_manifest=tuple(config.raw["phase6_endpoint_manifest"]),
        tolerances=(1, 2, 4, 8),
    )
    result["peak_tolerance_phase6_system"] = tuple(
        sorted(
            system_rows,
            key=lambda row: (
                system_order[str(row["system_id"])],
                endpoint_order[str(row["endpoint_id"])],
                int(row["tolerance_cm1"]),
            ),
        )
    )
    result["peak_tolerance_phase6_rank_stability"] = tuple(
        sorted(
            rank_rows,
            key=lambda row: (
                endpoint_order[str(row["endpoint_id"])],
                int(row["tolerance_cm1"]),
            ),
        )
    )
    result["leakage_boundaries"] = tuple(
        sorted(
            (dict(row) for row in leakage.boundary_rows),
            key=lambda row: boundary_order[str(row["boundary_id"])],
        )
    )
    result["leakage_duplicate_candidates"] = tuple(
        sorted(
            (dict(row) for row in leakage.candidate_rows),
            key=lambda row: (
                boundary_order[str(row["boundary_id"])],
                str(row["left_role_id"]),
                str(row["right_role_id"]),
                str(row["left_record_id"]),
                str(row["right_record_id"]),
            ),
        )
    )
    statuses: list[dict[str, object]] = []
    for row in tables["audit_status"]:
        current = dict(row)
        current["condition_id"] = str(current.get("condition_id", ""))
        statuses.append(current)
    for row in leakage.status_rows:
        current = dict(row)
        statuses.append(
            {
                "component": "leakage_boundaries",
                "panel_id": "",
                "endpoint_id": "",
                "protocol_id": "",
                "condition_id": str(current["boundary_id"]),
                "state": current["state"],
                "reason_code": current["reason_code"],
                "numerical_row_count": current["numerical_row_count"],
                "rank_row_count": current["rank_row_count"],
                "evidence_key": current["evidence_key"],
            }
        )
    for endpoint in config.raw["phase6_endpoint_manifest"]:
        for tolerance in (1, 2, 4, 8):
            statuses.append(
                {
                    "component": "phase6_peak_tolerance",
                    "panel_id": "",
                    "endpoint_id": endpoint["endpoint_id"],
                    "protocol_id": "",
                    "condition_id": f"tolerance:{tolerance}",
                    "state": "complete_numeric",
                    "reason_code": "",
                    "numerical_row_count": len(tuple(system_ids)),
                    "rank_row_count": 0 if endpoint["preferred_direction"] == "non_monotonic" or tolerance == 2 else 1,
                    "evidence_key": "step5_bootstrap_projection",
                }
            )
    component_order = {
        "resampling": 0,
        "normalization": 1,
        "phase4_peak_tolerance": 2,
        "phase6_peak_tolerance": 3,
        "leakage_boundaries": 4,
    }

    def status_key(row: Mapping[str, object]) -> tuple[object, ...]:
        component = str(row["component"])
        condition_id = str(row["condition_id"])
        panel_id = str(row.get("panel_id", ""))
        if component in {"resampling", "normalization", "phase4_peak_tolerance"}:
            return (
                component_order[component],
                panel_order[panel_id],
                *_condition_sort(component, condition_id),
            )
        if component == "phase6_peak_tolerance":
            return (
                component_order[component],
                endpoint_order[str(row["endpoint_id"])],
                _condition_sort("phase4_peak_tolerance", condition_id)[0],
            )
        return (component_order[component], boundary_order[condition_id])

    result["audit_status"] = tuple(
        {
            **row,
            "status_order": index,
        }
        for index, row in enumerate(sorted(statuses, key=status_key))
    )
    return result


def _production_like_sugar_identity(result: object) -> Mapping[str, object]:
    panel_ids = {
        str(row["panel_id"])
        for row in getattr(result, "audit_status")
        if str(row["panel_id"]).startswith("d4_")
    }
    identity: dict[str, dict[str, str]] = {}
    for panel_id in sorted(panel_ids):
        identity[panel_id] = {
            "normalization_none": "not_run",
            "tolerance_2": "not_run",
        }
    for row in getattr(result, "audit_status"):
        panel_id = str(row["panel_id"])
        if panel_id not in identity:
            continue
        if (
            str(row["component"]) == "normalization"
            and str(row["condition_id"]) == "none"
            and str(row["state"]) == "complete_numeric"
        ):
            identity[panel_id]["normalization_none"] = "passed"
        if (
            str(row["component"]) == "phase4_peak_tolerance"
            and str(row["condition_id"]) == "tolerance:2"
            and str(row["state"]) == "complete_numeric"
        ):
            identity[panel_id]["tolerance_2"] = "passed"
    return MappingProxyType(identity)


def _production_like_leakage_identity(config: _Config, result: object, authorities: Mapping[str, object]) -> Mapping[str, object]:
    leakage_entity = {
        str(row["boundary_id"]): str(row["leakage_entity"])
        for row in config.raw["leakage_boundary_definitions"]
    }
    direct_cells: dict[str, object] = {}
    for boundary_id, receipts in dict(result.identity["direct_cells"]).items():
        direct_cells[str(boundary_id)] = tuple(
            {
                **dict(receipt),
                "entity_key": leakage_entity[str(boundary_id)],
            }
            for receipt in receipts
        )
    scans = {
        str(boundary_id): dict(receipt)
        for boundary_id, receipt in dict(result.identity["scans"]).items()
    }
    return MappingProxyType(
        {
            "authorities": authorities,
            "phase2_rruff": dict(result.identity["phase2"]),
            "phase3_denoising": dict(result.identity["phase3"]),
            "direct_cells": MappingProxyType(
                {
                    str(boundary_id): direct_cells[str(boundary_id)]
                    for boundary_id in (
                        str(row["boundary_id"])
                        for row in config.raw["leakage_boundary_definitions"]
                        if str(row["boundary_id"]) in direct_cells
                    )
                }
            ),
            "scans": MappingProxyType(
                {
                    str(boundary_id): scans[str(boundary_id)]
                    for boundary_id in (
                        str(row["boundary_id"])
                        for row in config.raw["leakage_boundary_definitions"]
                        if str(row["boundary_id"]) in scans
                    )
                }
            ),
            "candidate_count": int(result.identity["candidate_count"]),
            "boundary_status": MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in dict(result.identity["boundary_status"]).items()
                }
            ),
        }
    )


def _formal_inputs(config: _Config, project_root: Path, *, worker_count: int) -> dict[str, object]:
    authorities = _validate_authorities(config, project_root)
    bacteria = v_build_bacteria_sensitivity(config.raw, project_root, worker_count=worker_count)
    sugar = v_build_sugar_sensitivity(config.raw, project_root, worker_count=worker_count)
    rruff = v_build_rruff_sensitivity(config.raw, project_root, worker_count=worker_count)
    leakage = v_build_leakage(config.raw, project_root, worker_count=worker_count)
    peak_root = (project_root / str(config.raw["authorities"]["phase6_peak_checksum_ledger"]["path"])).parent
    peak_config = json.loads((peak_root / "config.json").read_text(encoding="utf-8"))
    system_ids = tuple(str(value) for value in peak_config["promoted_system_ids"])
    phase6_rows = tuple(
        json.loads(line)
        for line in (peak_root / "bootstrap_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    )
    condition_receipt = _validate_step5_condition_rows(peak_root / "synthetic_condition_rows.jsonl", phase6_rows)
    tables = _canonicalize_formal_tables(
        config,
        _adapter_tables((bacteria, sugar, rruff)),
        leakage,
        phase6_rows,
        system_ids,
    )
    bacteria_identity = {
        "record_count": int(bacteria.identity["record_count"]),
        "class_count": int(bacteria.identity["class_count"]),
        "detector_call_count": int(bacteria.identity["detector_call_count"]),
    }
    sugar_identity = dict(_production_like_sugar_identity(sugar))
    rruff_identity = {
        "adapter": "rruff",
        "cohort_record_count": int(rruff.identity["cohort_record_count"]),
        "query_union_record_count": int(rruff.identity["query_union_record_count"]),
        "query_occurrence_count": int(rruff.identity["query_occurrence_count"]),
    }
    leakage_identity = _production_like_leakage_identity(config, leakage, authorities)
    authority_bridge = MappingProxyType(
        {
            **dict(authorities),
            "phase2_direct_split": dict(leakage_identity["phase2_rruff"]),
            "phase3_formal_denoising": dict(leakage_identity["phase3_denoising"]),
            "leakage_receipts": leakage_identity,
        }
    )
    run_identity = _formal_identity_receipt(config, project_root, {"bacteria": bacteria_identity, "sugar": sugar_identity, "rruff": rruff_identity}, leakage_identity, peak_root, system_ids, condition_receipt)
    return {
        "authority_bridge": authority_bridge,
        "identity": run_identity,
        "phase6_system_ids": system_ids,
        "phase6_rows": phase6_rows,
        "tables": tables,
    }


def _validate_step5_condition_rows(path: Path, bootstrap_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not path.is_file(): raise AppendixAuditsVerificationError("missing Step-5 synthetic condition rows")
    expected = {(str(row["system_id"]), str(row["endpoint_id"]), tolerance): [] for row in bootstrap_rows for tolerance in ("1", "2", "4", "8")}
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            for tolerance, endpoints in dict(row.get("endpoints_by_tolerance_cm1", {})).items():
                for endpoint, state in dict(endpoints).items():
                    key = (str(row["system_id"]), str(endpoint), str(tolerance))
                    if key not in expected or not isinstance(state, Mapping) or "state" not in state or "denominator" not in state:
                        raise AppendixAuditsVerificationError("invalid Step-5 condition curve row")
                    expected[key].append(state); count += 1
    for (system, endpoint, tolerance), rows in expected.items():
        source = next(row for row in bootstrap_rows if str(row["system_id"]) == system and str(row["endpoint_id"]) == endpoint)
        contributing = tuple(source["curve_by_tolerance_cm1"][tolerance].get("contributing_class_counts", ()))
        if contributing and sum(contributing) != sum(str(row["state"]) == "complete_numeric" for row in rows):
            raise AppendixAuditsVerificationError("Step-5 contributing-count mismatch")
    return MappingProxyType({"synthetic_condition_rows_sha256": _sha_file(path), "validated_endpoint_rows": count})


def _formal_identity_receipt(config: _Config, project_root: Path, adapters: Mapping[str, object], leakage: Mapping[str, object], peak_root: Path, system_ids: Sequence[str], condition_receipt: Mapping[str, object]) -> Mapping[str, object]:
    code_manifest = {relative: _sha_file(project_root / relative) for relative in _IDENTITY_CODE_FILES}
    return MappingProxyType({"identity_schema_version": "phase6-step7-formal-identity-v1", "adapter_identities": adapters, "leakage_identity": leakage, "step5": {"bootstrap_results_sha256": _sha_file(peak_root / "bootstrap_results.jsonl"), "config_sha256": _sha_file(peak_root / "config.json"), "promoted_system_ids": tuple(system_ids), "condition_validation": condition_receipt}, "authorities": config.raw["authorities"], "code_manifest": code_manifest, "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform_system": platform.system(), "platform_machine": platform.machine()}, "fixed_ordering": {"panels": tuple(config.raw["fixed_panel_order"]), "metrics": tuple(config.raw["metric_ids"]), "boundaries": tuple(row["boundary_id"] for row in config.raw["leakage_boundary_definitions"])}})


def _assemble_expected_files(config: _Config, rebuilt: Mapping[str, object]) -> tuple[dict[str, bytes], Mapping[str, int], str]:
    payload_files = tuple(config.raw["artifact_contract"]["payload_files"])
    tables = rebuilt["tables"]
    system_rows, rank_rows = _project_phase6_peak_rows(
        rebuilt["phase6_rows"],
        system_ids=tuple(rebuilt["phase6_system_ids"]),
        endpoint_manifest=tuple(config.raw["phase6_endpoint_manifest"]),
        tolerances=(1, 2, 4, 8),
    )
    ordered_tables: dict[str, Sequence[Mapping[str, object]]] = {
        "audit_status": tuple(tables["audit_status"]),
        "resampling_alignment": tuple(tables["resampling_alignment"]),
        "resampling_rank_stability": tuple(tables["resampling_rank_stability"]),
        "normalization_alignment": tuple(tables["normalization_alignment"]),
        "normalization_rank_stability": tuple(tables["normalization_rank_stability"]),
        "peak_tolerance_phase4_alignment": tuple(tables["peak_tolerance_phase4_alignment"]),
        "peak_tolerance_phase4_rank_stability": tuple(tables["peak_tolerance_phase4_rank_stability"]),
        "leakage_duplicate_candidates": tuple(tables["leakage_duplicate_candidates"]),
        "leakage_boundaries": tuple(tables["leakage_boundaries"]),
        "peak_tolerance_phase6_system": system_rows,
        "peak_tolerance_phase6_rank_stability": rank_rows,
    }
    counts = {name: len(rows) for name, rows in ordered_tables.items()}
    identity = {
        "schema_version": config.raw["schema_version"],
        "config_sha256": config.sha256,
        "authority_bridge": rebuilt["authority_bridge"],
        "identity": rebuilt["identity"],
        "schemas": config.raw["schemas"],
        "payload_order": payload_files,
    }
    run_id = str(config.raw["artifact_contract"]["run_prefix"]) + _sha(_canonical(identity))
    payloads: dict[str, bytes] = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": _canonical(rebuilt["authority_bridge"]),
        "preflight.json": _canonical(
            {
                "status": "passed",
                "d1_b_state": "not_evaluable_failed_alpha0_equivalence",
                "worker_count_identity_excluded": True,
            }
        ),
        "audit_status.csv": _csv_bytes(
            ordered_tables["audit_status"],
            config.raw["schemas"]["audit_status_csv_columns"],
        ),
        "resampling_alignment.csv": _csv_bytes(
            ordered_tables["resampling_alignment"],
            config.raw["schemas"]["phase4_alignment_csv_columns"],
        ),
        "resampling_rank_stability.csv": _csv_bytes(
            ordered_tables["resampling_rank_stability"],
            config.raw["schemas"]["phase4_rank_csv_columns"],
        ),
        "normalization_alignment.csv": _csv_bytes(
            ordered_tables["normalization_alignment"],
            config.raw["schemas"]["phase4_alignment_csv_columns"],
        ),
        "normalization_rank_stability.csv": _csv_bytes(
            ordered_tables["normalization_rank_stability"],
            config.raw["schemas"]["phase4_rank_csv_columns"],
        ),
        "peak_tolerance_phase4_alignment.csv": _csv_bytes(
            ordered_tables["peak_tolerance_phase4_alignment"],
            config.raw["schemas"]["phase4_alignment_csv_columns"],
        ),
        "peak_tolerance_phase4_rank_stability.csv": _csv_bytes(
            ordered_tables["peak_tolerance_phase4_rank_stability"],
            config.raw["schemas"]["phase4_rank_csv_columns"],
        ),
        "peak_tolerance_phase6_system.csv": _csv_bytes(
            ordered_tables["peak_tolerance_phase6_system"],
            config.raw["schemas"]["phase6_system_csv_columns"],
        ),
        "peak_tolerance_phase6_rank_stability.csv": _csv_bytes(
            ordered_tables["peak_tolerance_phase6_rank_stability"],
            config.raw["schemas"]["phase6_rank_csv_columns"],
        ),
        "leakage_boundaries.jsonl": _jsonl(ordered_tables["leakage_boundaries"]),
        "leakage_duplicate_candidates.jsonl": _jsonl(
            ordered_tables["leakage_duplicate_candidates"]
        ),
    }
    outcomes: dict[str, dict[str, int]] = {}
    for name, rows in ordered_tables.items():
        states: dict[str, int] = {}
        for row in rows:
            state = str(row.get("state", row.get("status", "")))
            states[state] = states.get(state, 0) + 1
        outcomes[name] = states
    payloads["summary.md"] = (
        "# Phase 6 Appendix Audits\n\n"
        "Status: complete.\n\n"
        "## Tables\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in counts.items())
        + "\n\n## Outcomes\n\n"
        + "\n".join(
            f"- {key}: {json.dumps(value, sort_keys=True)}" for key, value in outcomes.items()
        )
        + "\n"
    ).encode("utf-8")
    payloads["manifest.json"] = _canonical(
        {
            "schema_version": "phase6-appendix-audits-artifact-v1",
            "run_id": run_id,
            "status": "complete",
            "counts": counts,
            "fixed_expected_counts": config.raw["expected_rows"],
            "outcome_counts": outcomes,
            "identity": _plain(identity),
            "payload_files": payload_files,
        }
    )
    all_files = {
        **payloads,
        "complete.json": _canonical({"run_id": run_id, "status": "complete"}),
    }
    ledger = b"".join(
        f"{_sha(all_files[name])}  {name}\n".encode("utf-8")
        for name in (*payload_files, "complete.json")
    )
    return {**all_files, "SHA256SUMS": ledger}, counts, run_id


def _compare_bytes(path: Path, expected: Mapping[str, bytes]) -> None:
    actual = {
        item.name: item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }
    differing = sorted(
        name
        for name in set(actual) | set(expected)
        if actual.get(name) != expected.get(name)
    )
    if differing:
        raise AppendixAuditsVerificationError(
            f"rebuilt artifact bytes differ for payloads: {', '.join(differing)}"
        )


def verify_phase6_appendix_audits(
    path: Path,
    *,
    worker_count: int,
    project_root: Path = ROOT,
) -> AppendixAuditsVerificationSummary:
    if not isinstance(worker_count, int) or worker_count <= 0:
        raise AppendixAuditsVerificationError("worker_count must be a positive integer")
    run_path = Path(path)
    config = _load_config(Path(project_root) / DEFAULT_CONFIG.relative_to(ROOT))
    _files, manifest, authority_bridge = _validate_artifact_inventory(run_path, config)
    fixture = (
        authority_bridge.get("fixture_kind") == FIXTURE_ID
        or manifest.get("identity", {}).get("identity", {}).get("fixture_kind") == FIXTURE_ID
    )
    if fixture:
        rebuilt = _fixture_inputs(config)
    else:
        rebuilt = _formal_inputs(config, Path(project_root), worker_count=worker_count)
    expected_files, counts, run_id = _assemble_expected_files(config, rebuilt)
    _compare_bytes(run_path, expected_files)
    return AppendixAuditsVerificationSummary(
        run_id=run_id,
        path=run_path,
        status="complete",
        counts=counts,
    )


__all__ = [
    "AppendixAuditsVerificationError",
    "AppendixAuditsVerificationSummary",
    "verify_phase6_appendix_audits",
]
