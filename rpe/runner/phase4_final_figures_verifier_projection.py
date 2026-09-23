"""Verifier-owned reconstruction of Phase-4 final-figure inputs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Mapping

from rpe.runner.phase4_final_figures_authority import (
    ALPHA_GRID,
    ENDPOINT_ORDER,
    METRIC_OUTPUT_IDS,
    PERTURBATION_IDS,
    PROTOCOL_ORDER,
)


class VerifierProjectionError(ValueError):
    pass


SCOPE_ROWS = (
    ("p01_p05_not_evaluable_coverage", "not_evaluable_coverage"),
    ("p06_p07_structurally_ineligible_missing_explicit_baseline", "structurally_ineligible_missing_explicit_baseline"),
    ("d3_inactive", "inactive"),
    ("d1_protocol_b_failed_alpha0_equivalence", "not_evaluable_failed_alpha0_equivalence"),
    ("d5_protocol_a_peak_common_primary_not_evaluable_coverage", "not_evaluable_coverage"),
    ("phase4_confirmatory_success_not_evaluable", "not_evaluable"),
)
CELL_MAP = {
    "d5_full_domain_core": "d5", "d2_5shot": "d2_5", "d2_10shot": "d2_10",
    "d2_20shot": "d2_20", "d4_full_domain_core": "d4",
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise VerifierProjectionError(f"{path}: expected object")
    return value


def _csv_rows(raw: bytes, path: str) -> list[dict[str, str]]:
    try:
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise VerifierProjectionError(f"{path}: invalid CSV") from error


def _json_rows(raw: bytes, path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        for line in raw.decode("utf-8").splitlines():
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise VerifierProjectionError(f"{path}: JSONL row is not an object")
                rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerifierProjectionError(f"{path}: invalid JSONL") from error
    return rows


def _finite(value: object, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise VerifierProjectionError(f"{path}: expected finite number") from error
    if not math.isfinite(result):
        raise VerifierProjectionError(f"{path}: expected finite number")
    return result


def _integer(value: object, path: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise VerifierProjectionError(f"{path}: expected integer") from error


def _boolean(value: object, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise VerifierProjectionError(f"{path}: expected boolean")


def _interval(value: object, path: str) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return _finite(value[0], path), _finite(value[1], path)
    if isinstance(value, str):
        match = re.fullmatch(r"\(\s*([^,]+),\s*([^\)]+)\)", value)
        if match:
            return _finite(match.group(1), path), _finite(match.group(2), path)
    raise VerifierProjectionError(f"{path}: invalid interval")


def _admit_parent(key: str, spec: Mapping[str, object], root: Path) -> tuple[Path, dict[str, str], dict[str, bytes], dict[str, object]]:
    relative = Path(str(spec.get("relative_path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise VerifierProjectionError(f"parents.{key}.relative_path: invalid")
    path = root / relative
    if not path.is_dir() or path.name != str(spec.get("directory_basename", "")):
        raise VerifierProjectionError(f"parents.{key}: exact directory mismatch")
    ledger_raw = (path / "SHA256SUMS").read_bytes()
    if _sha256(ledger_raw) != str(spec.get("ledger_sha256", "")):
        raise VerifierProjectionError(f"parents.{key}.SHA256SUMS: digest mismatch")
    entries: dict[str, str] = {}
    for line in ledger_raw.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None or match.group(2) in entries:
            raise VerifierProjectionError(f"parents.{key}.SHA256SUMS: malformed entry")
        entries[match.group(2)] = match.group(1)
    required = tuple(str(item) for item in spec.get("required_inventory", ()))
    if tuple(entries) != required:
        raise VerifierProjectionError(f"parents.{key}.required_inventory: order mismatch")
    if {item.name for item in path.iterdir() if item.is_file()} != set(entries) | {"SHA256SUMS"}:
        raise VerifierProjectionError(f"parents.{key}: inventory mismatch")
    for name, expected_digest in entries.items():
        if _sha256_file(path / name) != expected_digest:
            raise VerifierProjectionError(f"parents.{key}.{name}: full-inventory digest mismatch")
    names = {"manifest.json", str(spec.get("terminal", ""))}
    allowed = _object(f"parents.{key}.allowed_payloads", spec.get("allowed_payloads"))
    for value in allowed.values():
        names.add(str(_object(f"parents.{key}.allowed_payload", value).get("path", "")))
    shot_payloads: list[Mapping[str, object]] = []
    for group_name in ("figure1_by_shot", "figure2_by_shot"):
        group = spec.get(group_name)
        if group is not None:
            for value in _object(f"parents.{key}.{group_name}", group).values():
                item = _object(f"parents.{key}.{group_name}", value)
                names.add(str(item.get("path", "")))
                shot_payloads.append(item)
    payloads: dict[str, bytes] = {}
    for name in names:
        if not name or name not in entries:
            raise VerifierProjectionError(f"parents.{key}: unbound payload {name!r}")
        raw = (path / name).read_bytes()
        if _sha256(raw) != entries[name]:
            raise VerifierProjectionError(f"parents.{key}.{name}: ledger mismatch")
        payloads[name] = raw
    for alias, value in allowed.items():
        allowed_spec = _object(f"parents.{key}.allowed_payloads.{alias}", value)
        name = str(allowed_spec.get("path", ""))
        if _sha256(payloads[name]) != str(allowed_spec.get("sha256", "")):
            raise VerifierProjectionError(f"parents.{key}.allowed_payloads.{alias}: digest mismatch")
    for item in shot_payloads:
        name = str(item.get("path", ""))
        if _sha256(payloads[name]) != str(item.get("sha256", "")):
            raise VerifierProjectionError(f"parents.{key}.{name}: digest mismatch")
    manifest = _object(f"parents.{key}.manifest", json.loads(payloads["manifest.json"]))
    terminal_name = str(spec.get("terminal", ""))
    terminal = _object(f"parents.{key}.{terminal_name}", json.loads(payloads[terminal_name]))
    expected_status = str(spec.get("status", ""))
    if str(manifest.get("status")) != expected_status or str(terminal.get("status")) != expected_status:
        raise VerifierProjectionError(f"parents.{key}: terminal status mismatch")
    manifest_run = str(manifest.get("run_id", manifest.get("run", "")))
    terminal_run = str(terminal.get("run_id", terminal.get("run", "")))
    if manifest_run != str(spec.get("manifest_run_id", "")):
        raise VerifierProjectionError(f"parents.{key}.manifest: run identity mismatch")
    if terminal_run != str(spec.get("terminal_run_id", "")):
        raise VerifierProjectionError(f"parents.{key}.{terminal_name}: run identity mismatch")
    receipt = {"relative_path": relative.as_posix(), "ledger_sha256": str(spec["ledger_sha256"]), "terminal_name": terminal_name, "terminal_sha256": entries[terminal_name], "status": expected_status, "allowed_payloads": {alias: {"path": str(_object("allowed", value)["path"]), "sha256": str(_object("allowed", value)["sha256"])} for alias, value in allowed.items()}}
    return path, entries, payloads, receipt


def _allowed_name(spec: Mapping[str, object], alias: str) -> str:
    return str(_object(f"allowed_payloads.{alias}", _object("allowed_payloads", spec.get("allowed_payloads")).get(alias)).get("path", ""))


def _parent_figure1(key: str, spec: Mapping[str, object], payloads: Mapping[str, bytes]) -> dict[tuple[str, float], tuple[float, float]]:
    rows, shot, result = _csv_rows(payloads[_allowed_name(spec, "figure1")], f"{key}.{_allowed_name(spec, 'figure1')}"), spec.get("shot_count"), {}
    for row in rows:
        if shot is not None and _integer(row.get("shot_count"), f"{key}.shot_count") != int(shot): continue
        if row.get("metric_output_id") != "mse": continue
        pair = (str(row.get("perturbation_id", "")), _finite(row.get("alpha"), f"{key}.alpha"))
        if pair in result: raise VerifierProjectionError(f"{key}.figure1: duplicate MSE key {pair}")
        result[pair] = (_finite(row.get("mean_metric_harm"), f"{key}.mean_metric_harm"), _finite(row.get("mean_downstream_harm"), f"{key}.mean_downstream_harm"))
    expected = {(p, a) for p in PERTURBATION_IDS for a in ALPHA_GRID}
    if set(result) != expected: raise VerifierProjectionError(f"{key}.figure1: expected exact 40-row MSE grid")
    return result


def _parent_holm(key: str, spec: Mapping[str, object], payloads: Mapping[str, bytes]) -> dict[tuple[str, str], dict[str, object]]:
    name, shot, result = _allowed_name(spec, "holm"), spec.get("shot_count"), {}
    for row in _json_rows(payloads[name], f"{key}.{name}"):
        if shot is not None and _integer(row.get("shot_count"), f"{key}.shot_count") != int(shot): continue
        metric, statistic = str(row.get("metric_output_id", "")), str(row.get("statistic", ""))
        if metric in METRIC_OUTPUT_IDS and statistic in {"d_ag", "d_acc"}: result[(metric, statistic)] = row
    if len(result) != 24: raise VerifierProjectionError(f"{key}.holm: expected 24 candidate slots, got {len(result)}")
    return result


def _holm_bundle(row: Mapping[str, object], *, family_id: str, state: str) -> dict[str, object]:
    favorable, rejected = _boolean(row.get("favorable"), "holm.favorable"), _boolean(row.get("rejected"), "holm.rejected")
    return {"raw_p_value": _finite(row.get("raw_p_value"), "holm.raw_p_value"), "adjusted_p_value": _finite(row.get("adjusted_p_value"), "holm.adjusted_p_value"), "rank": _integer(row.get("rank"), "holm.rank"), "family_size": _integer(row.get("family_size"), "holm.family_size"), "family_id": str(row.get("family_id", family_id)) or family_id, "favorable": favorable, "rejected": rejected, "direction": "favorable" if favorable else "adverse", "state": str(row.get("state", row.get("family_state", state))) or state, "glyph": "favorable_rejected" if rejected and favorable else "adverse_rejected" if rejected else "not_rejected"}


def _parent_figure2(key: str, spec: Mapping[str, object], payloads: Mapping[str, bytes]) -> list[dict[str, object]]:
    name, shot = _allowed_name(spec, "figure2"), spec.get("shot_count")
    selected = [row for row in _csv_rows(payloads[name], f"{key}.{name}") if shot is None or _integer(row.get("shot_count"), f"{key}.shot_count") == int(shot)]
    by_metric = {str(row.get("metric_output_id", "")): row for row in selected}
    if len(selected) != 13 or set(by_metric) != set(METRIC_OUTPUT_IDS): raise VerifierProjectionError(f"{key}.figure2: expected exact 13-metric panel")
    holm, endpoint, protocol, panel, output = _parent_holm(key, spec, payloads), str(spec.get("endpoint_id", "")), str(spec.get("protocol_id", "")), str(spec.get("panel_id", "")), []
    for metric in METRIC_OUTPUT_IDS:
        row = by_metric[metric]
        if "ag_lower" in row:
            ag_lower, ag_upper, acc, acc_lower, acc_upper, panel_state = _finite(row.get("ag_lower"), f"{key}.{metric}.ag_lower"), _finite(row.get("ag_upper"), f"{key}.{metric}.ag_upper"), _finite(row.get("acc"), f"{key}.{metric}.acc"), _finite(row.get("acc_lower"), f"{key}.{metric}.acc_lower"), _finite(row.get("acc_upper"), f"{key}.{metric}.acc_upper"), str(row.get("metric_state", "complete"))
        else:
            ag_lower, ag_upper = _interval(row.get("ag_interval"), f"{key}.{metric}.ag_interval")
            acc, (acc_lower, acc_upper), panel_state = _finite(row.get("acc_cross"), f"{key}.{metric}.acc_cross"), _interval(row.get("acc_interval"), f"{key}.{metric}.acc_interval"), str(row.get("state", "complete"))
        item: dict[str, object] = {"panel_id": panel, "endpoint_id": endpoint, "protocol_id": protocol, "metric_output_id": metric, "ag": _finite(row.get("ag"), f"{key}.{metric}.ag"), "ag_lower": ag_lower, "ag_upper": ag_upper, "acc_cross": acc, "acc_cross_lower": acc_lower, "acc_cross_upper": acc_upper, "panel_state": panel_state}
        for statistic in ("d_ag", "d_acc"):
            if metric == "mse":
                item.update({statistic: "", f"{statistic}_lower": "", f"{statistic}_upper": "", f"{statistic}_raw_p_value": "", f"{statistic}_adjusted_p_value": "", f"{statistic}_rank": "", f"{statistic}_family_size": 24, f"{statistic}_family_id": f"{endpoint}:{protocol}:within_protocol", f"{statistic}_favorable": "", f"{statistic}_rejected": "", f"{statistic}_direction": "reference", f"{statistic}_state": "reference_no_contrast", f"{statistic}_glyph": "not_rejected"})
            else:
                if row.get(statistic, "") == "": raise VerifierProjectionError(f"{key}.{metric}.{statistic}: missing contrast")
                lower, upper = _interval(row[f"{statistic}_interval"], f"{key}.{metric}.{statistic}_interval") if f"{statistic}_interval" in row else (_finite(row.get(f"{statistic}_lower"), f"{key}.{metric}.{statistic}_lower"), _finite(row.get(f"{statistic}_upper"), f"{key}.{metric}.{statistic}_upper"))
                item[statistic], item[f"{statistic}_lower"], item[f"{statistic}_upper"] = _finite(row[statistic], f"{key}.{metric}.{statistic}"), lower, upper
                item.update({f"{statistic}_{field}": value for field, value in _holm_bundle(holm[(metric, statistic)], family_id=f"{endpoint}:{protocol}:within_protocol", state="tested").items()})
        output.append(item)
    return output


def _contrast_holm(spec: Mapping[str, object], payloads: Mapping[str, bytes]) -> dict[tuple[str, str], dict[str, object]]:
    name, result = _allowed_name(spec, "holm"), {}
    for row in _json_rows(payloads[name], f"contrast.{name}"):
        cell, slot = str(row.get("cell_id", "")), str(row.get("slot_id", ""))
        if cell in CELL_MAP: result[(cell, slot)] = row
    if len(result) != 145: raise VerifierProjectionError(f"contrast.holm: expected 145 slots, got {len(result)}")
    return result


def _contrast_bundle(row: Mapping[str, object]) -> dict[str, object]:
    favorable = str(row.get("direction", "")) in {"attenuated_by_b", "candidate_advantage_strengthened_under_b"}
    rejected = _boolean(row.get("rejected"), "contrast.rejected")
    return {"raw_p_value": _finite(row.get("raw_p_value"), "contrast.raw_p_value"), "adjusted_p_value": _finite(row.get("adjusted_p_value"), "contrast.adjusted_p_value"), "rank": _integer(row.get("rank"), "contrast.rank"), "family_size": _integer(row.get("family_size"), "contrast.family_size"), "family_id": str(row.get("family_id", "")), "favorable": favorable, "rejected": rejected, "direction": "favorable" if favorable else "adverse", "state": str(row.get("state", "tested")), "glyph": "favorable_rejected" if rejected and favorable else "adverse_rejected" if rejected else "not_rejected"}


def _contrast_effects(spec: Mapping[str, object], payloads: Mapping[str, bytes], holm: Mapping[tuple[str, str], Mapping[str, object]]) -> list[dict[str, object]]:
    name, output = _allowed_name(spec, "downstream"), []
    for row in _csv_rows(payloads[name], f"contrast.{name}"):
        raw_cell, perturbation, summary_type = str(row.get("cell_id", "")), str(row.get("perturbation_id", "")), str(row.get("summary_type", ""))
        if raw_cell not in CELL_MAP: raise VerifierProjectionError(f"contrast.downstream: unexpected cell {raw_cell!r}")
        bundle = _contrast_bundle(holm[(raw_cell, f"downstream:{perturbation}")])
        output.append({"cell_id": CELL_MAP[raw_cell], "perturbation_id": perturbation, "alpha": None if summary_type == "integrated" else _finite(row.get("alpha"), "contrast.alpha"), "summary_type": summary_type, "g_harm": _finite(row.get("gap"), "contrast.gap"), "g_harm_lower": _finite(row.get("interval_lower"), "contrast.interval_lower"), "g_harm_upper": _finite(row.get("interval_upper"), "contrast.interval_upper"), **bundle, "classification": "attenuation_by_b_rejected" if bundle["rejected"] and bundle["favorable"] else "amplification_by_b_rejected" if bundle["rejected"] else "not_rejected"})
    if len(output) != 225: raise VerifierProjectionError(f"contrast.downstream: expected 225 rows, got {len(output)}")
    return output


def _interaction_rows(spec: Mapping[str, object], payloads: Mapping[str, bytes], holm: Mapping[tuple[str, str], Mapping[str, object]]) -> list[dict[str, object]]:
    name, output = _allowed_name(spec, "interaction"), []
    for row in _csv_rows(payloads[name], f"contrast.{name}"):
        raw_cell, metric = str(row.get("cell_id", "")), str(row.get("metric_output_id", ""))
        item: dict[str, object] = {"cell_id": CELL_MAP.get(raw_cell, raw_cell), "metric_output_id": metric, "delta_ag": _finite(row.get("delta_ag"), "interaction.delta_ag"), "delta_ag_lower": _finite(row.get("delta_ag_lower"), "interaction.delta_ag_lower"), "delta_ag_upper": _finite(row.get("delta_ag_upper"), "interaction.delta_ag_upper"), "delta_acc": _finite(row.get("delta_acc"), "interaction.delta_acc"), "delta_acc_lower": _finite(row.get("delta_acc_lower"), "interaction.delta_acc_lower"), "delta_acc_upper": _finite(row.get("delta_acc_upper"), "interaction.delta_acc_upper")}
        for statistic in ("i_ag", "i_acc"):
            if metric == "mse": item.update({statistic: "", f"{statistic}_lower": "", f"{statistic}_upper": "", f"{statistic}_raw_p_value": "", f"{statistic}_adjusted_p_value": "", f"{statistic}_rank": "", f"{statistic}_family_size": 29, f"{statistic}_family_id": f"{raw_cell}:protocol_ab", f"{statistic}_favorable": "", f"{statistic}_rejected": "", f"{statistic}_direction": "reference", f"{statistic}_state": "reference_no_interaction", f"{statistic}_glyph": "not_rejected"})
            else:
                for suffix in ("", "_lower", "_upper"): item[f"{statistic}{suffix}"] = _finite(row.get(f"{statistic}{suffix}"), f"interaction.{statistic}{suffix}")
                item.update({f"{statistic}_{field}": value for field, value in _contrast_bundle(holm[(raw_cell, f"{statistic}:{metric}")]).items()})
        item["state"] = str(row.get("state", "complete")); output.append(item)
    if len(output) != 65: raise VerifierProjectionError(f"contrast.interaction: expected 65 rows, got {len(output)}")
    return output


def project_real_final_figure_inputs(config_document: Mapping[str, object], *, root: Path) -> dict[str, object]:
    parents = _object("config.parents", config_document.get("parents")); expected_keys = ("d5_a", "d5_b", "d2_a", "d2_b", "d1_a", "d1_b", "d4_a", "d4_b", "contrast")
    if set(parents) != set(expected_keys): raise VerifierProjectionError(f"config.parents: expected exact keys {expected_keys!r}")
    admitted: dict[str, tuple[Mapping[str, object], dict[str, bytes], dict[str, object]]] = {}
    for key in expected_keys:
        spec = _object(f"config.parents.{key}", parents[key]); _, _, payloads, receipt = _admit_parent(key, spec, root); admitted[key] = (spec, payloads, receipt)
    panel_specs: list[tuple[str, Mapping[str, object], dict[str, bytes]]] = []
    for key in ("d5_a", "d5_b"): spec, payloads, _ = admitted[key]; panel_specs.append((key, spec, payloads))
    for key in ("d2_a", "d2_b"):
        parent_spec, payloads, _ = admitted[key]
        for shot in (5, 10, 20):
            spec = dict(parent_spec); spec["shot_count"] = shot; spec["endpoint_id"] = f"d2_{shot}"; spec["panel_id"] = f"d2_{shot}_{spec['protocol_id']}"; allowed = dict(_object("allowed_payloads", spec["allowed_payloads"])); allowed["figure1"] = _object("figure1_by_shot", spec["figure1_by_shot"])[str(shot)]; allowed["figure2"] = _object("figure2_by_shot", spec["figure2_by_shot"])[str(shot)]; spec["allowed_payloads"] = allowed; panel_specs.append((f"{key}_{shot}", spec, payloads))
    for key in ("d1_a", "d4_a", "d4_b"): spec, payloads, _ = admitted[key]; panel_specs.append((key, spec, payloads))
    figure1_grids: dict[tuple[str, str], dict[tuple[str, float], tuple[float, float]]] = {}; figure2_rows: list[dict[str, object]] = []
    for key, spec, payloads in panel_specs:
        endpoint, protocol = str(spec["endpoint_id"]), str(spec["protocol_id"]); figure1_grids[(endpoint, protocol)] = _parent_figure1(key, spec, payloads); figure2_rows.extend(_parent_figure2(key, spec, payloads))
    figure1_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINT_ORDER:
        protocols, a_grid = (("a",) if endpoint == "d1" else PROTOCOL_ORDER), figure1_grids[(endpoint, "a")]
        for protocol in protocols:
            grid = figure1_grids[(endpoint, protocol)]
            for perturbation in PERTURBATION_IDS:
                for alpha in ALPHA_GRID:
                    metric_x, downstream = grid[(perturbation, alpha)]; a_metric, _ = a_grid[(perturbation, alpha)]
                    if endpoint in {"d5", "d2_5", "d2_10", "d2_20"} and protocol == "b" and metric_x != a_metric: raise VerifierProjectionError(f"{endpoint}: Protocol-B MSE coordinate mismatch")
                    figure1_rows.append({"cell_id": endpoint, "protocol_id": protocol, "perturbation_id": perturbation, "alpha": alpha, "metric_output_id": "mse", "protocol_a_mse_x": a_metric, "metric_x": metric_x, "downstream_harm": downstream})
    contrast_spec, contrast_payloads, _ = admitted["contrast"]; contrast_holm = _contrast_holm(contrast_spec, contrast_payloads); effects, interactions = _contrast_effects(contrast_spec, contrast_payloads, contrast_holm), _interaction_rows(contrast_spec, contrast_payloads, contrast_holm)
    parent_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINT_ORDER:
        for protocol in PROTOCOL_ORDER:
            closed, base_key = endpoint == "d1" and protocol == "b", f"d2_{protocol}" if endpoint.startswith("d2_") else f"{endpoint}_{protocol}"
            spec, _, receipt = admitted[base_key]; allowed = _object("allowed_payloads", spec.get("allowed_payloads"))
            if endpoint.startswith("d2_"):
                shot = endpoint.split("_")[1]; figure1_spec, figure2_spec = _object("figure1_by_shot", spec["figure1_by_shot"])[shot], _object("figure2_by_shot", spec["figure2_by_shot"])[shot]
            else: figure1_spec, figure2_spec = allowed.get("figure1", {}), allowed.get("figure2", {})
            holm_spec = allowed.get("holm", {})
            parent_rows.append({"panel_id": "d1_b_closed" if closed else f"{endpoint}_{protocol}", "endpoint_id": endpoint, "protocol_id": protocol, "state": "not_evaluable_failed_alpha0_equivalence" if closed else "complete", "numerical_source_allowed": not closed, "parent_key": base_key, "relative_path": receipt["relative_path"], "ledger_sha256": receipt["ledger_sha256"], "terminal_name": receipt["terminal_name"], "terminal_sha256": receipt["terminal_sha256"], "figure1_path": str(_object("figure1", figure1_spec).get("path", "")), "figure1_sha256": str(_object("figure1", figure1_spec).get("sha256", "")), "figure2_path": str(_object("figure2", figure2_spec).get("path", "")), "figure2_sha256": str(_object("figure2", figure2_spec).get("sha256", "")), "holm_path": str(_object("holm", holm_spec).get("path", "")), "holm_sha256": str(_object("holm", holm_spec).get("sha256", ""))})
    if len(figure1_rows) != 440 or len(figure2_rows) != 143: raise VerifierProjectionError("projection counts do not match frozen design")
    captions = _object("config.captions", config_document.get("captions"))
    return {"scope_status_rows": [{"scope_id": key, "state": state} for key, state in SCOPE_ROWS], "parent_panel_rows": parent_rows, "figure1_response_rows": figure1_rows, "figure1_protocol_effect_rows": effects, "figure2_alignment_rows": figure2_rows, "figure2_interaction_rows": interactions, "captions": {"figure1": str(captions["figure1"]), "figure2": str(captions["figure2"])}, "parent_receipts": {key: receipt for key, (_, _, receipt) in admitted.items()}}


__all__ = ["VerifierProjectionError", "project_real_final_figure_inputs"]
