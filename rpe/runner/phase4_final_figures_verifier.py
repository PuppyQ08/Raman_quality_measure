from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rpe-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "phase4-final-figures-v1"

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, PngImagePlugin

from rpe.runner.phase4_final_figures_authority import (
    ALPHA_GRID,
    ARTIFACT_PAYLOAD_FILES,
    ARTIFACT_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CONFIG_BYTES,
    CONFIG_SHA256,
    ENDPOINT_ORDER,
    EXPERIMENT_ID,
    FIGURE_TEXT,
    FIGURE1_PROTOCOL_EFFECT_FIELDS,
    FIGURE1_RESPONSE_FIELDS,
    FIGURE1_SIZE,
    FIGURE2_ALIGNMENT_FIELDS,
    FIGURE2_PROTOCOL_INTERACTION_FIELDS,
    FIGURE2_SIZE,
    METRIC_OUTPUT_IDS,
    PAIRED_CELL_ORDER,
    PARENT_PANEL_FIELDS,
    PERTURBATION_IDS,
    PROTOCOL_EFFECT_CLASSIFICATIONS,
    PROTOCOL_ORDER,
    ROOT,
    SCHEMA_VERSION,
    SCOPE_IDS,
    SCOPE_STATUS_FIELDS,
    STYLE,
    WITHIN_PROTOCOL_GLYPHS,
)


class Phase4FinalFiguresVerifierError(ValueError):
    pass


@dataclass(frozen=True)
class Phase4FinalFiguresVerifierSummary:
    path: Path
    run_id: str
    status: str
    verified_file_count: int


@dataclass(frozen=True)
class _VerifierConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    scope_ids: tuple[str, ...]
    endpoint_order: tuple[str, ...]
    paired_cell_order: tuple[str, ...]
    protocol_order: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_output_ids: tuple[str, ...]
    artifact_payload_files: tuple[str, ...]


@dataclass(frozen=True)
class _VerifierInputs:
    document: Mapping[str, object]
    scope_status_rows: tuple[dict[str, object], ...]
    parent_panel_rows: tuple[dict[str, object], ...]
    figure1_response_rows: tuple[dict[str, object], ...]
    figure1_protocol_effect_rows: tuple[dict[str, object], ...]
    figure2_alignment_rows: tuple[dict[str, object], ...]
    figure2_interaction_rows: tuple[dict[str, object], ...]
    captions: Mapping[str, str]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_run_id(
    *,
    config_sha256: str,
    payload_projection: Mapping[str, object],
) -> str:
    return "phase4-final-figures-" + _sha256_hex(
        _canonical_json_bytes(
            {
                "config_sha256": config_sha256,
                "payload_projection": payload_projection,
            }
        )
    )


def _write_sha256sums(
    payloads: Mapping[str, bytes],
    terminal_name: str,
    terminal_bytes: bytes,
) -> bytes:
    rows = [f"{_sha256_hex(payloads[name])}  {name}" for name in ARTIFACT_PAYLOAD_FILES]
    rows.append(f"{_sha256_hex(terminal_bytes)}  {terminal_name}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise Phase4FinalFiguresVerifierError(f"{path}: must be an object")
    return value


def _list(path: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise Phase4FinalFiguresVerifierError(f"{path}: must be a list")
    return value


def _tuple_of_str(path: str, value: object) -> tuple[str, ...]:
    result = tuple(str(item) for item in _list(path, value))
    if any(not item for item in result):
        raise Phase4FinalFiguresVerifierError(f"{path}: must not contain empty values")
    return result


def _tuple_of_float(path: str, value: object) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in _list(path, value))
    except (TypeError, ValueError) as error:
        raise Phase4FinalFiguresVerifierError(f"{path}: must contain numeric values") from error


def _int_at(path: str, value: object) -> int:
    if not isinstance(value, int):
        raise Phase4FinalFiguresVerifierError(f"{path}: must be an integer")
    return value


def _validate_exact(path: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise Phase4FinalFiguresVerifierError(
            f"{path}: expected {expected!r}, got {observed!r}"
        )


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "True":
            return True
        if value == "False":
            return False
    raise Phase4FinalFiguresVerifierError(f"bool: invalid boolean value {value!r}")


def _require_frozen_identity(config_doc: Mapping[str, object]) -> bool:
    if bool(config_doc.get("synthetic_fixture", True)):
        return False
    return CONFIG_BYTES > 0 and CONFIG_SHA256 != ("0" * 64)


def _parse_config(path: Path, raw_bytes: bytes) -> _VerifierConfig:
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise Phase4FinalFiguresVerifierError(f"config: invalid JSON: {error}") from error
    doc = _object("config", document)
    synthetic_fixture = bool(doc.get("synthetic_fixture", True))
    _validate_exact("config.schema_version", doc.get("schema_version"), SCHEMA_VERSION)
    _validate_exact("config.experiment_id", doc.get("experiment_id"), EXPERIMENT_ID)
    _validate_exact("config.claim_boundary", doc.get("claim_boundary"), CLAIM_BOUNDARY)
    scope_ids = _tuple_of_str("config.scope_ids", doc.get("scope_ids"))
    endpoint_order = _tuple_of_str("config.endpoint_order", doc.get("endpoint_order"))
    paired_cell_order = _tuple_of_str(
        "config.paired_cell_order",
        doc.get("paired_cell_order"),
    )
    protocol_order = _tuple_of_str("config.protocol_order", doc.get("protocol_order"))
    perturbation_ids = _tuple_of_str(
        "config.perturbation_ids",
        doc.get("perturbation_ids"),
    )
    alpha_grid = _tuple_of_float("config.alpha_grid", doc.get("alpha_grid"))
    metric_output_ids = _tuple_of_str(
        "config.metric_output_ids",
        doc.get("metric_output_ids"),
    )
    artifact_payload_files = _tuple_of_str(
        "config.artifact_payload_files",
        doc.get("artifact_payload_files"),
    )
    _validate_exact("config.scope_ids", scope_ids, SCOPE_IDS)
    _validate_exact("config.endpoint_order", endpoint_order, ENDPOINT_ORDER)
    _validate_exact("config.paired_cell_order", paired_cell_order, PAIRED_CELL_ORDER)
    _validate_exact("config.protocol_order", protocol_order, PROTOCOL_ORDER)
    _validate_exact("config.perturbation_ids", perturbation_ids, PERTURBATION_IDS)
    _validate_exact("config.alpha_grid", alpha_grid, ALPHA_GRID)
    _validate_exact("config.metric_output_ids", metric_output_ids, METRIC_OUTPUT_IDS)
    _validate_exact(
        "config.artifact_payload_files",
        artifact_payload_files,
        ARTIFACT_PAYLOAD_FILES,
    )
    if _require_frozen_identity(doc):
        if len(raw_bytes) != CONFIG_BYTES or _sha256_hex(raw_bytes) != CONFIG_SHA256:
            raise Phase4FinalFiguresVerifierError("config: frozen identity mismatch")
    expected = _object("config.expected", doc.get("expected"))
    _int_at("config.expected.scope_status_row_count", expected.get("scope_status_row_count"))
    _int_at(
        "config.expected.parent_panel_index_row_count",
        expected.get("parent_panel_index_row_count"),
    )
    _int_at(
        "config.expected.figure1_response_row_count",
        expected.get("figure1_response_row_count"),
    )
    _int_at(
        "config.expected.figure1_protocol_effect_row_count",
        expected.get("figure1_protocol_effect_row_count"),
    )
    _int_at(
        "config.expected.figure2_alignment_row_count",
        expected.get("figure2_alignment_row_count"),
    )
    _int_at(
        "config.expected.figure2_protocol_interaction_row_count",
        expected.get("figure2_protocol_interaction_row_count"),
    )
    figure1 = _object("config.figure1", doc.get("figure1"))
    figure2 = _object("config.figure2", doc.get("figure2"))
    _int_at("config.figure1.width_px", figure1.get("width_px"))
    _int_at("config.figure1.height_px", figure1.get("height_px"))
    _int_at("config.figure2.width_px", figure2.get("width_px"))
    _int_at("config.figure2.height_px", figure2.get("height_px"))
    return _VerifierConfig(
        path=path,
        raw_bytes=raw_bytes,
        sha256=_sha256_hex(raw_bytes),
        document=doc,
        synthetic_fixture=synthetic_fixture,
        scope_ids=scope_ids,
        endpoint_order=endpoint_order,
        paired_cell_order=paired_cell_order,
        protocol_order=protocol_order,
        perturbation_ids=perturbation_ids,
        alpha_grid=alpha_grid,
        metric_output_ids=metric_output_ids,
        artifact_payload_files=artifact_payload_files,
    )


def _validate_config_receipts(config: _VerifierConfig) -> None:
    if config.synthetic_fixture:
        return
    for group_name in ("authorities", "code_authority"):
        group = _object(f"config.{group_name}", config.document.get(group_name))
        for key, raw_receipt in group.items():
            receipt = _object(f"config.{group_name}.{key}", raw_receipt)
            relative = Path(str(receipt.get("path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise Phase4FinalFiguresVerifierError(f"config.{group_name}.{key}.path: invalid")
            target = ROOT / relative
            if not target.is_file():
                raise Phase4FinalFiguresVerifierError(f"config.{group_name}.{key}.path: missing")
            raw = target.read_bytes()
            if len(raw) != int(receipt.get("bytes", -1)) or _sha256_hex(raw) != str(receipt.get("sha256", "")):
                raise Phase4FinalFiguresVerifierError(f"config.{group_name}.{key}: receipt mismatch")
    import PIL
    environment = _object("config.rendering_environment", config.document.get("rendering_environment"))
    observed = {
        "python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__, "machine": platform.machine(),
        "system": platform.system(), "backend": "Agg",
        "font": "DejaVu Sans", "svg_hashsalt": "phase4-final-figures-v1",
    }
    if dict(environment) != observed:
        raise Phase4FinalFiguresVerifierError(
            f"config.rendering_environment: expected {dict(environment)!r}, got {observed!r}"
        )


def _inputs_from_object(inputs: object) -> _VerifierInputs:
    return _VerifierInputs(
        document=dict(getattr(inputs, "document", {})),
        scope_status_rows=tuple(
            dict(row) for row in getattr(inputs, "scope_status_rows")
        ),
        parent_panel_rows=tuple(
            dict(row) for row in getattr(inputs, "parent_panel_rows")
        ),
        figure1_response_rows=tuple(
            dict(row) for row in getattr(inputs, "figure1_response_rows")
        ),
        figure1_protocol_effect_rows=tuple(
            dict(row) for row in getattr(inputs, "figure1_protocol_effect_rows")
        ),
        figure2_alignment_rows=tuple(
            dict(row) for row in getattr(inputs, "figure2_alignment_rows")
        ),
        figure2_interaction_rows=tuple(
            dict(row) for row in getattr(inputs, "figure2_interaction_rows")
        ),
        captions=dict(getattr(inputs, "captions")),
    )


def _reconstruct_inputs(config: _VerifierConfig) -> _VerifierInputs:
    _validate_config_receipts(config)
    if not config.synthetic_fixture:
        try:
            from rpe.runner.phase4_final_figures_verifier_projection import (
                VerifierProjectionError,
                project_real_final_figure_inputs,
            )

            document = project_real_final_figure_inputs(
                config.document,
                root=Path(__file__).resolve().parents[2],
            )
        except (ImportError, VerifierProjectionError) as error:
            raise Phase4FinalFiguresVerifierError(
                f"config.parents: {error}"
            ) from error
        return _VerifierInputs(
            document=dict(document),
            scope_status_rows=tuple(dict(row) for row in document["scope_status_rows"]),
            parent_panel_rows=tuple(dict(row) for row in document["parent_panel_rows"]),
            figure1_response_rows=tuple(
                dict(row) for row in document["figure1_response_rows"]
            ),
            figure1_protocol_effect_rows=tuple(
                dict(row) for row in document["figure1_protocol_effect_rows"]
            ),
            figure2_alignment_rows=tuple(
                dict(row) for row in document["figure2_alignment_rows"]
            ),
            figure2_interaction_rows=tuple(
                dict(row) for row in document["figure2_interaction_rows"]
            ),
            captions=dict(document["captions"]),
        )
    document = {
        "scope_status_rows": [
            {"scope_id": scope_id, "state": "closed"} for scope_id in config.scope_ids
        ],
        "parent_panel_rows": [
            {
                "panel_id": "d5_a",
                "endpoint_id": "d5",
                "protocol_id": "a",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d5_b",
                "endpoint_id": "d5",
                "protocol_id": "b",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d2_5_a",
                "endpoint_id": "d2_5",
                "protocol_id": "a",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d2_5_b",
                "endpoint_id": "d2_5",
                "protocol_id": "b",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d2_10_a",
                "endpoint_id": "d2_10",
                "protocol_id": "a",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d2_10_b",
                "endpoint_id": "d2_10",
                "protocol_id": "b",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d2_20_a",
                "endpoint_id": "d2_20",
                "protocol_id": "a",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d2_20_b",
                "endpoint_id": "d2_20",
                "protocol_id": "b",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d1_a",
                "endpoint_id": "d1",
                "protocol_id": "a",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d1_b_closed",
                "endpoint_id": "d1",
                "protocol_id": "b",
                "numerical_source_allowed": False,
            },
            {
                "panel_id": "d4_a",
                "endpoint_id": "d4",
                "protocol_id": "a",
                "numerical_source_allowed": True,
            },
            {
                "panel_id": "d4_b",
                "endpoint_id": "d4",
                "protocol_id": "b",
                "numerical_source_allowed": True,
            },
        ],
        "figure1_response_rows": [],
        "figure1_protocol_effect_rows": [],
        "figure2_alignment_rows": [],
        "figure2_interaction_rows": [],
        "captions": {"figure1": FIGURE_TEXT[3], "figure2": FIGURE_TEXT[4]},
    }
    for endpoint_index, cell_id in enumerate(config.endpoint_order):
        protocols = ("a",) if cell_id == "d1" else config.protocol_order
        for protocol in protocols:
            for perturbation_index, perturbation_id in enumerate(config.perturbation_ids):
                for alpha_index, alpha in enumerate(config.alpha_grid):
                    metric_x = round(
                        0.1 + endpoint_index + perturbation_index + alpha_index / 100.0,
                        6,
                    )
                    document["figure1_response_rows"].append(
                        {
                            "cell_id": cell_id,
                            "protocol_id": protocol,
                            "perturbation_id": perturbation_id,
                            "alpha": alpha,
                            "metric_output_id": "mse",
                            "protocol_a_mse_x": metric_x,
                            "metric_x": round(
                                metric_x
                                + (0.5 if protocol == "b" and cell_id == "d4" else 0.0),
                                6,
                            ),
                            "downstream_harm": round(
                                2.0
                                + endpoint_index / 10.0
                                + perturbation_index
                                + alpha_index / 10.0,
                                6,
                            ),
                        }
                    )
    for cell_index, cell_id in enumerate(config.paired_cell_order):
        for perturbation_index, perturbation_id in enumerate(config.perturbation_ids):
            for alpha_index, alpha in enumerate(config.alpha_grid):
                document["figure1_protocol_effect_rows"].append(
                    {
                        "cell_id": cell_id,
                        "perturbation_id": perturbation_id,
                        "alpha": alpha,
                        "summary_type": "alpha_specific",
                        "g_harm": round((cell_index - perturbation_index) / 10.0, 6),
                        "g_harm_lower": "",
                        "g_harm_upper": "",
                        "raw_p_value": "",
                        "adjusted_p_value": "",
                        "rank": "",
                        "family_size": "",
                        "family_id": "",
                        "favorable": "",
                        "rejected": "",
                        "direction": "",
                        "state": "",
                        "classification": PROTOCOL_EFFECT_CLASSIFICATIONS[
                            (cell_index + perturbation_index + alpha_index)
                            % len(PROTOCOL_EFFECT_CLASSIFICATIONS)
                        ],
                    }
                )
            document["figure1_protocol_effect_rows"].append(
                {
                    "cell_id": cell_id,
                    "perturbation_id": perturbation_id,
                    "alpha": None,
                    "summary_type": "integrated",
                    "g_harm": round((cell_index - perturbation_index) / 10.0, 6),
                    "g_harm_lower": "",
                    "g_harm_upper": "",
                    "raw_p_value": "",
                    "adjusted_p_value": "",
                    "rank": "",
                    "family_size": "",
                    "family_id": "",
                    "favorable": "",
                    "rejected": "",
                    "direction": "",
                    "state": "",
                    "classification": PROTOCOL_EFFECT_CLASSIFICATIONS[
                        (cell_index + perturbation_index)
                        % len(PROTOCOL_EFFECT_CLASSIFICATIONS)
                    ],
                }
            )
    for endpoint_index, endpoint_id in enumerate(config.endpoint_order):
        for protocol_id in (("a",) if endpoint_id == "d1" else config.protocol_order):
            for metric_index, metric_output_id in enumerate(config.metric_output_ids):
                favorable = (metric_index + (protocol_id == "b")) % 3 == 0
                rejected = metric_index % 3 != 2
                direction = "favorable" if favorable else "adverse" if rejected else "neutral"
                document["figure2_alignment_rows"].append(
                    {
                        "panel_id": f"{endpoint_id}_{protocol_id}",
                        "endpoint_id": endpoint_id,
                        "protocol_id": protocol_id,
                        "metric_output_id": metric_output_id,
                        "ag": round(
                            0.50
                            + endpoint_index / 100
                            + metric_index / 1000
                            + (0.03 if protocol_id == "b" else 0),
                            6,
                        ),
                        "ag_lower": round(
                            0.45
                            + endpoint_index / 100
                            + metric_index / 1000
                            + (0.03 if protocol_id == "b" else 0),
                            6,
                        ),
                        "ag_upper": round(
                            0.55
                            + endpoint_index / 100
                            + metric_index / 1000
                            + (0.03 if protocol_id == "b" else 0),
                            6,
                        ),
                        "acc_cross": round(0.60 + endpoint_index / 100 + metric_index / 1000, 6),
                        "acc_cross_lower": round(0.57 + endpoint_index / 100 + metric_index / 1000, 6),
                        "acc_cross_upper": round(0.63 + endpoint_index / 100 + metric_index / 1000, 6),
                        "d_ag": round((metric_index - 1) / 100, 6),
                        "d_ag_lower": round((metric_index - 1) / 100 - 0.01, 6),
                        "d_ag_upper": round((metric_index - 1) / 100 + 0.01, 6),
                        "d_ag_raw_p_value": round(0.001 + metric_index / 1000, 6),
                        "d_ag_adjusted_p_value": round(0.011 + metric_index / 1000, 6),
                        "d_ag_rank": metric_index + 1,
                        "d_ag_family_size": 24,
                        "d_ag_family_id": f"{endpoint_id}:{protocol_id}:within_protocol",
                        "d_ag_favorable": favorable,
                        "d_ag_rejected": rejected,
                        "d_ag_direction": direction,
                        "d_ag_state": "complete",
                        "d_ag_glyph": "favorable_rejected" if rejected and favorable else "adverse_rejected" if rejected else "not_rejected",
                        "d_acc": round((1 - metric_index) / 100, 6),
                        "d_acc_lower": round((1 - metric_index) / 100 - 0.01, 6),
                        "d_acc_upper": round((1 - metric_index) / 100 + 0.01, 6),
                        "d_acc_raw_p_value": round(0.002 + metric_index / 1000, 6),
                        "d_acc_adjusted_p_value": round(0.022 + metric_index / 1000, 6),
                        "d_acc_rank": metric_index + 1,
                        "d_acc_family_size": 24,
                        "d_acc_family_id": f"{endpoint_id}:{protocol_id}:within_protocol",
                        "d_acc_favorable": not favorable,
                        "d_acc_rejected": rejected,
                        "d_acc_direction": "favorable" if not favorable else "adverse",
                        "d_acc_state": "complete",
                        "d_acc_glyph": "favorable_rejected" if rejected and not favorable else "adverse_rejected" if rejected else "not_rejected",
                        "panel_state": "complete",
                    }
                )
    for cell_index, cell_id in enumerate(config.paired_cell_order):
        for metric_index, metric_output_id in enumerate(config.metric_output_ids):
            document["figure2_interaction_rows"].append(
                {
                    "cell_id": cell_id,
                    "metric_output_id": metric_output_id,
                    "delta_ag": round((cell_index - metric_index) / 20.0, 6),
                    "delta_ag_lower": round((cell_index - metric_index) / 20.0 - 0.03, 6),
                    "delta_ag_upper": round((cell_index - metric_index) / 20.0 + 0.03, 6),
                    "delta_acc": round((metric_index - cell_index) / 30.0, 6),
                    "delta_acc_lower": round((metric_index - cell_index) / 30.0 - 0.02, 6),
                    "delta_acc_upper": round((metric_index - cell_index) / 30.0 + 0.02, 6),
                    "i_ag": round((cell_index - metric_index) / 40.0, 6),
                    "i_ag_lower": round((cell_index - metric_index) / 40.0 - 0.04, 6),
                    "i_ag_upper": round((cell_index - metric_index) / 40.0 + 0.04, 6),
                    "i_acc": round((metric_index - cell_index) / 50.0, 6),
                    "i_acc_lower": round((metric_index - cell_index) / 50.0 - 0.05, 6),
                    "i_acc_upper": round((metric_index - cell_index) / 50.0 + 0.05, 6),
                    **{
                        f"{stat}_{field}": value
                        for stat, value in (
                            (
                                "i_ag",
                                {
                                    "raw_p_value": round(0.003 + metric_index / 1000, 6),
                                    "adjusted_p_value": round(0.033 + metric_index / 1000, 6),
                                    "rank": metric_index + 1,
                                    "family_size": 29,
                                    "family_id": f"{cell_id}:protocol_interaction",
                                    "favorable": metric_index % 3 == 0,
                                    "rejected": metric_index % 3 != 2,
                                    "direction": "favorable" if metric_index % 3 == 0 else "adverse",
                                    "state": "complete",
                                    "glyph": "favorable_rejected" if metric_index % 3 == 0 else "adverse_rejected" if metric_index % 3 == 1 else "not_rejected",
                                },
                            ),
                            (
                                "i_acc",
                                {
                                    "raw_p_value": round(0.004 + metric_index / 1000, 6),
                                    "adjusted_p_value": round(0.034 + metric_index / 1000, 6),
                                    "rank": metric_index + 1,
                                    "family_size": 29,
                                    "family_id": f"{cell_id}:protocol_interaction",
                                    "favorable": metric_index % 3 == 1,
                                    "rejected": metric_index % 3 != 2,
                                    "direction": "favorable" if metric_index % 3 == 1 else "adverse",
                                    "state": "complete",
                                    "glyph": "favorable_rejected" if metric_index % 3 == 1 else "adverse_rejected" if metric_index % 3 == 0 else "not_rejected",
                                },
                            ),
                        )
                        for field, value in value.items()
                    },
                    "state": "complete",
                }
            )
    return _VerifierInputs(
        document=dict(document),
        scope_status_rows=tuple(dict(row) for row in document["scope_status_rows"]),
        parent_panel_rows=tuple(dict(row) for row in document["parent_panel_rows"]),
        figure1_response_rows=tuple(dict(row) for row in document["figure1_response_rows"]),
        figure1_protocol_effect_rows=tuple(
            dict(row) for row in document["figure1_protocol_effect_rows"]
        ),
        figure2_alignment_rows=tuple(dict(row) for row in document["figure2_alignment_rows"]),
        figure2_interaction_rows=tuple(
            dict(row) for row in document["figure2_interaction_rows"]
        ),
        captions=dict(document["captions"]),
    )


def _sort_scope_rows(
    rows: Sequence[Mapping[str, object]],
    scope_order: Sequence[str],
) -> list[dict[str, object]]:
    order = {scope_id: index for index, scope_id in enumerate(scope_order)}
    normalized = []
    for row in rows:
        scope_id = str(row.get("scope_id", ""))
        if scope_id not in order:
            raise Phase4FinalFiguresVerifierError(
                f"scope_status_rows.scope_id: unexpected {scope_id!r}"
            )
        normalized.append({"scope_id": scope_id, "state": str(row.get("state", ""))})
    normalized.sort(key=lambda row: order[row["scope_id"]])
    return normalized


def _sort_parent_rows(
    rows: Sequence[Mapping[str, object]],
    endpoint_order: Sequence[str],
    protocol_order: Sequence[str],
) -> list[dict[str, object]]:
    endpoint_index = {value: index for index, value in enumerate(endpoint_order)}
    protocol_index = {value: index for index, value in enumerate(protocol_order)}
    normalized = []
    for row in rows:
        endpoint_id = str(row.get("endpoint_id", ""))
        protocol_id = str(row.get("protocol_id", ""))
        if endpoint_id not in endpoint_index:
            raise Phase4FinalFiguresVerifierError(
                f"parent_panel_rows.endpoint_id: unexpected {endpoint_id!r}"
            )
        if protocol_id not in protocol_index:
            raise Phase4FinalFiguresVerifierError(
                f"parent_panel_rows.protocol_id: unexpected {protocol_id!r}"
            )
        normalized.append({
            field: (
                str(row.get(field, ""))
                if field != "numerical_source_allowed"
                else bool(row.get(field, False))
            )
            for field in PARENT_PANEL_FIELDS
        } | {"endpoint_id": endpoint_id, "protocol_id": protocol_id})
    normalized.sort(
        key=lambda row: (
            endpoint_index[row["endpoint_id"]],
            protocol_index[row["protocol_id"]],
            row["panel_id"],
        )
    )
    return normalized


def _sort_figure1_response_rows(
    rows: Sequence[Mapping[str, object]],
    config: _VerifierConfig,
) -> list[dict[str, object]]:
    endpoint_index = {value: index for index, value in enumerate(config.endpoint_order)}
    protocol_index = {value: index for index, value in enumerate(config.protocol_order)}
    perturbation_index = {
        value: index for index, value in enumerate(config.perturbation_ids)
    }
    alpha_index = {value: index for index, value in enumerate(config.alpha_grid)}
    normalized = []
    for row in rows:
        cell_id = str(row.get("cell_id", ""))
        protocol_id = str(row.get("protocol_id", ""))
        perturbation_id = str(row.get("perturbation_id", ""))
        alpha = float(row.get("alpha"))
        metric_output_id = str(row.get("metric_output_id", ""))
        if cell_id not in endpoint_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_response_rows.cell_id: unexpected {cell_id!r}"
            )
        if protocol_id not in protocol_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_response_rows.protocol_id: unexpected {protocol_id!r}"
            )
        if perturbation_id not in perturbation_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_response_rows.perturbation_id: unexpected {perturbation_id!r}"
            )
        if alpha not in alpha_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_response_rows.alpha: unexpected {alpha!r}"
            )
        if metric_output_id != "mse":
            raise Phase4FinalFiguresVerifierError(
                "figure1_response_rows.metric_output_id: must be mse"
            )
        protocol_a_mse_x = float(row.get("protocol_a_mse_x"))
        metric_x = float(row.get("metric_x"))
        x_render_value = protocol_a_mse_x if protocol_id == "b" and cell_id != "d1" else metric_x
        normalized.append(
            {
                "cell_id": cell_id,
                "protocol_id": protocol_id,
                "perturbation_id": perturbation_id,
                "alpha": alpha,
                "metric_output_id": metric_output_id,
                "protocol_a_mse_x": protocol_a_mse_x,
                "metric_x": metric_x,
                "x_render_value": x_render_value,
                "downstream_harm": float(row.get("downstream_harm")),
            }
        )
    normalized.sort(
        key=lambda row: (
            endpoint_index[row["cell_id"]],
            protocol_index[row["protocol_id"]],
            perturbation_index[row["perturbation_id"]],
            alpha_index[row["alpha"]],
        )
    )
    return normalized


def _sort_figure1_protocol_effect_rows(
    rows: Sequence[Mapping[str, object]],
    config: _VerifierConfig,
) -> list[dict[str, object]]:
    cell_index = {value: index for index, value in enumerate(config.paired_cell_order)}
    perturbation_index = {
        value: index for index, value in enumerate(config.perturbation_ids)
    }
    alpha_index = {value: index for index, value in enumerate(config.alpha_grid)}
    normalized = []
    for row in rows:
        cell_id = str(row.get("cell_id", ""))
        perturbation_id = str(row.get("perturbation_id", ""))
        classification = str(row.get("classification", ""))
        summary_type = str(row.get("summary_type", ""))
        if cell_id not in cell_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_protocol_effect_rows.cell_id: unexpected {cell_id!r}"
            )
        if perturbation_id not in perturbation_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_protocol_effect_rows.perturbation_id: unexpected {perturbation_id!r}"
            )
        if classification not in PROTOCOL_EFFECT_CLASSIFICATIONS:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_protocol_effect_rows.classification: unexpected {classification!r}"
            )
        if summary_type == "alpha_specific":
            alpha = float(row.get("alpha"))
            if alpha not in alpha_index:
                raise Phase4FinalFiguresVerifierError(
                    f"figure1_protocol_effect_rows.alpha: unexpected {alpha!r}"
                )
        elif summary_type == "integrated":
            alpha = ""
        else:
            raise Phase4FinalFiguresVerifierError(
                f"figure1_protocol_effect_rows.summary_type: unexpected {summary_type!r}"
            )
        normalized.append(
            {
                "cell_id": cell_id,
                "perturbation_id": perturbation_id,
                "alpha": alpha,
                "summary_type": summary_type,
                "g_harm": float(row.get("g_harm")),
                "g_harm_lower": "" if row.get("g_harm_lower", "") == "" else float(row.get("g_harm_lower")),
                "g_harm_upper": "" if row.get("g_harm_upper", "") == "" else float(row.get("g_harm_upper")),
                "raw_p_value": "" if row.get("raw_p_value", "") == "" else float(row.get("raw_p_value")),
                "adjusted_p_value": "" if row.get("adjusted_p_value", "") == "" else float(row.get("adjusted_p_value")),
                "rank": "" if row.get("rank", "") == "" else int(row.get("rank")),
                "family_size": "" if row.get("family_size", "") == "" else int(row.get("family_size")),
                "family_id": str(row.get("family_id", "")),
                "favorable": "" if row.get("favorable", "") == "" else _boolish(row.get("favorable")),
                "rejected": "" if row.get("rejected", "") == "" else _boolish(row.get("rejected")),
                "direction": str(row.get("direction", "")),
                "state": str(row.get("state", "")),
                "classification": classification,
            }
        )
    normalized.sort(
        key=lambda row: (
            cell_index[row["cell_id"]],
            perturbation_index[row["perturbation_id"]],
            0 if row["summary_type"] == "alpha_specific" else 1,
            alpha_index.get(row["alpha"], len(alpha_index)),
        )
    )
    return normalized


def _sort_figure2_alignment_rows(
    rows: Sequence[Mapping[str, object]],
    parent_rows: Sequence[Mapping[str, object]],
    config: _VerifierConfig,
) -> list[dict[str, object]]:
    panel_lookup = {
        (str(row["endpoint_id"]), str(row["protocol_id"])): str(row["panel_id"])
        for row in parent_rows
        if bool(row["numerical_source_allowed"])
    }
    endpoint_index = {value: index for index, value in enumerate(config.endpoint_order)}
    metric_index = {
        value: index for index, value in enumerate(config.metric_output_ids)
    }
    panel_rank = {"a": 0, "b": 1}
    normalized = []
    for row in rows:
        endpoint_id = str(row.get("endpoint_id", ""))
        protocol_id = str(row.get("protocol_id", ""))
        metric_output_id = str(row.get("metric_output_id", ""))
        if endpoint_id not in endpoint_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure2_alignment_rows.endpoint_id: unexpected {endpoint_id!r}"
            )
        if metric_output_id not in metric_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure2_alignment_rows.metric_output_id: unexpected {metric_output_id!r}"
            )
        if protocol_id not in config.protocol_order or (endpoint_id == "d1" and protocol_id != "a"):
            raise Phase4FinalFiguresVerifierError(
                f"figure2_alignment_rows.protocol_id: unexpected {protocol_id!r}"
            )
        if str(row.get("panel_id", "")) != panel_lookup.get((endpoint_id, protocol_id)):
            raise Phase4FinalFiguresVerifierError(
                "figure2_alignment_rows.panel_id: does not bind parent panel"
            )

        def value(name: str) -> object:
            current = row.get(name, "")
            if current == "":
                raise Phase4FinalFiguresVerifierError(
                    f"figure2_alignment_rows.{name}: missing"
                )
            return current

        def glyph(name: str) -> str:
            current = str(value(name))
            if current not in WITHIN_PROTOCOL_GLYPHS:
                raise Phase4FinalFiguresVerifierError(
                    f"figure2_alignment_rows.{name}: unexpected {current!r}"
                )
            return current

        base_row = {
            "panel_id": str(row["panel_id"]),
            "endpoint_id": endpoint_id,
            "protocol_id": protocol_id,
            "metric_output_id": metric_output_id,
        }
        for prefix in ("ag", "acc"):
            stat = "d_ag" if prefix == "ag" else "d_acc"
            if metric_output_id == "mse":
                for name in (stat, f"{stat}_lower", f"{stat}_upper", f"{stat}_raw_p_value", f"{stat}_adjusted_p_value", f"{stat}_rank"):
                    base_row[name] = ""
                base_row[f"{stat}_family_size"] = int(value(f"{stat}_family_size"))
                base_row[f"{stat}_family_id"] = str(value(f"{stat}_family_id"))
                base_row[f"{stat}_favorable"] = ""; base_row[f"{stat}_rejected"] = ""
                base_row[f"{stat}_direction"] = "reference"; base_row[f"{stat}_state"] = "reference_no_contrast"; base_row[f"{stat}_glyph"] = "not_rejected"
                continue
            for name in (
                stat,
                f"{stat}_lower",
                f"{stat}_upper",
                f"{stat}_raw_p_value",
                f"{stat}_adjusted_p_value",
            ):
                base_row[name] = float(value(name))
            for name in (f"{stat}_rank", f"{stat}_family_size"):
                base_row[name] = int(value(name))
            base_row[f"{stat}_family_id"] = str(value(f"{stat}_family_id"))
            for name in (f"{stat}_favorable", f"{stat}_rejected"):
                raw = value(name)
                base_row[name] = raw if isinstance(raw, bool) else _boolish(raw)
            base_row[f"{stat}_direction"] = str(value(f"{stat}_direction"))
            base_row[f"{stat}_state"] = str(value(f"{stat}_state"))
            base_row[f"{stat}_glyph"] = glyph(f"{stat}_glyph")
        for name in (
            "ag",
            "ag_lower",
            "ag_upper",
            "acc_cross",
            "acc_cross_lower",
            "acc_cross_upper",
        ):
            base_row[name] = float(value(name))
        base_row["panel_state"] = str(value("panel_state"))
        normalized.append(base_row)
    normalized.sort(
        key=lambda row: (
            endpoint_index[row["endpoint_id"]],
            metric_index[row["metric_output_id"]],
            panel_rank[row["protocol_id"]],
        )
    )
    return normalized


def _sort_figure2_interaction_rows(
    rows: Sequence[Mapping[str, object]],
    config: _VerifierConfig,
) -> list[dict[str, object]]:
    cell_index = {value: index for index, value in enumerate(config.paired_cell_order)}
    metric_index = {
        value: index for index, value in enumerate(config.metric_output_ids)
    }
    normalized = []
    for row in rows:
        cell_id = str(row.get("cell_id", ""))
        metric_output_id = str(row.get("metric_output_id", ""))
        if cell_id not in cell_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure2_interaction_rows.cell_id: unexpected {cell_id!r}"
            )
        if metric_output_id not in metric_index:
            raise Phase4FinalFiguresVerifierError(
                f"figure2_interaction_rows.metric_output_id: unexpected {metric_output_id!r}"
            )
        item = {
            "cell_id": cell_id,
            "metric_output_id": metric_output_id,
            "delta_ag": float(row.get("delta_ag")),
            "delta_ag_lower": float(row.get("delta_ag_lower")),
            "delta_ag_upper": float(row.get("delta_ag_upper")),
            "delta_acc": float(row.get("delta_acc")),
            "delta_acc_lower": float(row.get("delta_acc_lower")),
            "delta_acc_upper": float(row.get("delta_acc_upper")),
                "i_ag": "" if metric_output_id == "mse" else float(row.get("i_ag")),
                "i_ag_lower": "" if metric_output_id == "mse" else float(row.get("i_ag_lower")),
                "i_ag_upper": "" if metric_output_id == "mse" else float(row.get("i_ag_upper")),
                "i_acc": "" if metric_output_id == "mse" else float(row.get("i_acc")),
                "i_acc_lower": "" if metric_output_id == "mse" else float(row.get("i_acc_lower")),
                "i_acc_upper": "" if metric_output_id == "mse" else float(row.get("i_acc_upper")),
        }
        for statistic in ("i_ag", "i_acc"):
            if metric_output_id == "mse":
                for name in (statistic, f"{statistic}_lower", f"{statistic}_upper", f"{statistic}_raw_p_value", f"{statistic}_adjusted_p_value", f"{statistic}_rank"):
                    item[name] = ""
                item[f"{statistic}_family_size"] = int(row.get(f"{statistic}_family_size")); item[f"{statistic}_family_id"] = str(row.get(f"{statistic}_family_id")); item[f"{statistic}_favorable"] = ""; item[f"{statistic}_rejected"] = ""; item[f"{statistic}_direction"] = "reference"; item[f"{statistic}_state"] = "reference_no_interaction"; item[f"{statistic}_glyph"] = "not_rejected"
                continue
            for name in (f"{statistic}_raw_p_value", f"{statistic}_adjusted_p_value"):
                item[name] = float(row.get(name))
            for name in (f"{statistic}_rank", f"{statistic}_family_size"):
                item[name] = int(row.get(name))
            item[f"{statistic}_family_id"] = str(row.get(f"{statistic}_family_id"))
            for name in (f"{statistic}_favorable", f"{statistic}_rejected"):
                raw = row.get(name)
                item[name] = raw if isinstance(raw, bool) else _boolish(raw)
            item[f"{statistic}_direction"] = str(row.get(f"{statistic}_direction"))
            item[f"{statistic}_state"] = str(row.get(f"{statistic}_state"))
            glyph = str(row.get(f"{statistic}_glyph"))
            if glyph not in WITHIN_PROTOCOL_GLYPHS:
                raise Phase4FinalFiguresVerifierError(
                    f"figure2_interaction_rows.{statistic}_glyph: unexpected {glyph!r}"
                )
            item[f"{statistic}_glyph"] = glyph
        item["state"] = str(row.get("state", "complete"))
        normalized.append(item)
    normalized.sort(
        key=lambda row: (
            cell_index[row["cell_id"]],
            metric_index[row["metric_output_id"]],
        )
    )
    return normalized


def _validate_counts(
    config: _VerifierConfig,
    *,
    scope_rows: Sequence[Mapping[str, object]],
    parent_rows: Sequence[Mapping[str, object]],
    figure1_rows: Sequence[Mapping[str, object]],
    effect_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    interaction_rows: Sequence[Mapping[str, object]],
) -> None:
    expected = _object("config.expected", config.document["expected"])
    observed = {
        "scope_status_row_count": len(scope_rows),
        "parent_panel_index_row_count": len(parent_rows),
        "figure1_response_row_count": len(figure1_rows),
        "figure1_protocol_effect_row_count": len(effect_rows),
        "figure2_alignment_row_count": len(alignment_rows),
        "figure2_protocol_interaction_row_count": len(interaction_rows),
    }
    for key, value in observed.items():
        if int(expected[key]) != value:
            raise Phase4FinalFiguresVerifierError(
                f"{key}: expected {expected[key]!r}, got {value!r}"
            )
    if any(
        row["cell_id"] == "d1" and row["protocol_id"] == "b"
        for row in figure1_rows
    ):
        raise Phase4FinalFiguresVerifierError(
            "figure1_response_rows: D1-B must remain closed"
        )


def _canonicalize_svg_ids(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    pattern = re.compile(
        r'(?P<prefix>id="|url\(#|xlink:href="#)'
        r'(?P<identifier>[A-Za-z_][A-Za-z0-9_.:-]*)'
        r'(?=["\)])'
    )
    identifiers: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        identifier = match.group("identifier")
        canonical = identifiers.setdefault(
            identifier, f"canonical-{len(identifiers):04d}"
        )
        return f"{match.group('prefix')}{canonical}"

    return pattern.sub(replace, text).encode("utf-8")


def _save_figure_bytes(
    figure: plt.Figure,
    *,
    width_px: int,
    height_px: int,
) -> tuple[bytes, bytes]:
    dpi = 300
    figure.set_size_inches(width_px / dpi, height_px / dpi, forward=True)
    png_path = Path("/tmp") / "phase4-final-figures-verifier-temp.png"
    svg_path = Path("/tmp") / "phase4-final-figures-verifier-temp.svg"
    try:
        figure.savefig(
            png_path,
            dpi=dpi,
            format="png",
            metadata={"Software": "matplotlib", "Date": None},
            facecolor="white",
        )
        figure.savefig(
            svg_path,
            dpi=dpi,
            format="svg",
            metadata={"Date": None},
            facecolor="white",
        )
        return _canonicalize_png(png_path.read_bytes()), _canonicalize_svg_ids(svg_path.read_bytes())
    finally:
        plt.close(figure)
        if png_path.exists():
            png_path.unlink()
        if svg_path.exists():
            svg_path.unlink()


def _canonicalize_png(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as image:
        rgba = image.convert("RGBA")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Software", "matplotlib")
        output = io.BytesIO()
        rgba.save(
            output,
            format="PNG",
            compress_level=9,
            optimize=False,
            dpi=(300, 300),
            pnginfo=metadata,
        )
    return output.getvalue()


def _render_figure1(
    figure1_rows: Sequence[Mapping[str, object]],
    effect_rows: Sequence[Mapping[str, object]],
) -> tuple[bytes, bytes]:
    endpoint_titles = {
        "d5": "D5",
        "d2_5": "D2-5",
        "d2_10": "D2-10",
        "d2_20": "D2-20",
        "d1": "D1",
        "d4": "D4",
    }
    figure = plt.Figure(constrained_layout=True)
    grid = figure.add_gridspec(5, 5, height_ratios=[0.24, 1.0, 1.0, 0.24, 0.72])
    ribbon = figure.add_subplot(grid[0, :])
    ribbon.set_axis_off()
    ribbon.text(0.01, 0.72, FIGURE_TEXT[0], fontsize=18, family="DejaVu Sans")
    ribbon.text(
        0.35,
        0.72,
        "P1-P5 not_evaluable_coverage",
        fontsize=14,
        family="DejaVu Sans",
    )
    ribbon.text(
        0.68,
        0.72,
        "P6/P7 structurally_ineligible_missing_explicit_baseline",
        fontsize=14,
        family="DejaVu Sans",
    )
    ribbon.text(0.01, 0.18, "D3 inactive", fontsize=14, family="DejaVu Sans")
    cell_axes: dict[tuple[str, str], plt.Axes] = {}
    for row_index, protocol in enumerate(("a", "b"), start=1):
        for col_index, cell_id in enumerate(PAIRED_CELL_ORDER):
            axis = figure.add_subplot(grid[row_index, col_index])
            cell_axes[(cell_id, protocol)] = axis
            axis.axhline(0.0, color="#cccccc", linewidth=0.6)
            axis.axvline(0.0, color="#cccccc", linewidth=0.6)
            if row_index == 1:
                axis.set_title(
                    endpoint_titles[cell_id],
                    fontsize=16,
                    family="DejaVu Sans",
                )
            axis.set_xlabel("ΔMSE" if row_index == 2 else "", fontsize=10)
            axis.set_ylabel("Δ downstream harm" if col_index == 0 else "", fontsize=10)
            if cell_id == "d5" and protocol == "b":
                axis.text(
                    0.02,
                    0.96,
                    FIGURE_TEXT[1],
                    transform=axis.transAxes,
                    va="top",
                    fontsize=10,
                    family="DejaVu Sans",
                )
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in figure1_rows:
        grouped.setdefault(
            (
                str(row["cell_id"]),
                str(row["protocol_id"]),
                str(row["perturbation_id"]),
            ),
            [],
        ).append(row)
    for (cell_id, protocol, perturbation_id), group in grouped.items():
        if (cell_id, protocol) not in cell_axes:
            continue
        axis = cell_axes[(cell_id, protocol)]
        ordered = sorted(group, key=lambda item: float(item["alpha"]))
        xs = [float(item["x_render_value"]) for item in ordered]
        ys = [float(item["downstream_harm"]) for item in ordered]
        style = STYLE["perturbation_palette"][perturbation_id]
        axis.plot(xs, ys, color=style["color"], linewidth=1.4, alpha=0.95)
        axis.scatter(xs, ys, color=style["color"], marker=style["marker"], s=[size * size for size in (3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)], label=perturbation_id.upper(), zorder=3)
    for cell_id in PAIRED_CELL_ORDER:
        values = [(float(row["x_render_value"]), float(row["downstream_harm"])) for row in figure1_rows if row["cell_id"] == cell_id]
        xs, ys = zip(*values, strict=True)
        for axis, data in ((cell_axes[(cell_id, "a")], xs), (cell_axes[(cell_id, "b")], xs)):
            low, high = min(0.0, min(data)), max(0.0, max(data)); pad = max((high - low) * 0.05, 1e-12); axis.set_xlim(low - pad, high + pad)
        for axis in (cell_axes[(cell_id, "a")], cell_axes[(cell_id, "b")]):
            low, high = min(0.0, min(ys)), max(0.0, max(ys)); pad = max((high - low) * 0.05, 1e-12); axis.set_ylim(low - pad, high + pad)
    footer_axes = {}
    for col_index, cell_id in enumerate(PAIRED_CELL_ORDER):
        axis = figure.add_subplot(grid[3, col_index])
        footer_axes[cell_id] = axis
        axis.set_axis_off()
        axis.set_title(endpoint_titles[cell_id], fontsize=14, family="DejaVu Sans")
    integrated_rows = [row for row in effect_rows if row["summary_type"] == "integrated"]
    perturbation_order = {value: index for index, value in enumerate(PERTURBATION_IDS)}
    integrated_rows.sort(
        key=lambda row: (
            PAIRED_CELL_ORDER.index(str(row["cell_id"])),
            perturbation_order[str(row["perturbation_id"])],
        )
    )
    color_map = STYLE["protocol_effect_colors"]
    for row in integrated_rows:
        axis = footer_axes[str(row["cell_id"])]
        x = perturbation_order[str(row["perturbation_id"])] * 0.18 + 0.04
        axis.add_patch(
            Rectangle(
                (x, 0.18),
                0.15,
                0.52,
                facecolor=color_map[str(row["classification"])],
                edgecolor="#444444",
                linewidth=0.8,
            )
        )
        axis.text(
            x + 0.075,
            0.44,
            str(row["perturbation_id"]).upper(),
            ha="center",
            va="center",
            fontsize=9,
            family="DejaVu Sans",
        )
    bottom = grid[4, :].subgridspec(1, 5, width_ratios=[1.15, 1.15, 0.8, 1.35, 1.35])
    d1_axis = figure.add_subplot(bottom[0, 0:2])
    d1_axis.set_title("D1", fontsize=15, family="DejaVu Sans")
    d1_axis.set_xlabel("ΔMSE", fontsize=10)
    d1_axis.set_ylabel("Δ downstream harm", fontsize=10)
    d1_rows = [row for row in figure1_rows if row["cell_id"] == "d1"]
    for perturbation_id in PERTURBATION_IDS:
        ordered = sorted(
            (row for row in d1_rows if row["perturbation_id"] == perturbation_id),
            key=lambda item: float(item["alpha"]),
        )
        style = STYLE["perturbation_palette"][perturbation_id]
        xs = [float(item["x_render_value"]) for item in ordered]
        ys = [float(item["downstream_harm"]) for item in ordered]
        d1_axis.plot(xs, ys, color=style["color"], linewidth=1.4)
        d1_axis.scatter(xs, ys, color=style["color"], marker=style["marker"], s=[size * size for size in (3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)], label=perturbation_id.upper(), zorder=3)
    d1_closed = figure.add_subplot(bottom[0, 2])
    d1_closed.set_axis_off()
    d1_closed.add_patch(
        Rectangle(
            (0.02, 0.08),
            0.96,
            0.84,
            facecolor="#efefef",
            edgecolor="#666666",
            hatch="///",
        )
    )
    d1_closed.text(
        0.5,
        0.5,
        "closed: failed exact\nalpha-zero equivalence;\nno positive-alpha outcome",
        ha="center",
        va="center",
        fontsize=9,
        family="DejaVu Sans",
    )
    ledger = figure.add_subplot(bottom[0, 3:5])
    ledger.set_axis_off()
    ledger.text(0.0, 0.9, FIGURE_TEXT[3], fontsize=13, family="DejaVu Sans")
    ledger.text(0.0, 0.66, "P08  P09  P10  P11  P12", fontsize=11, family="DejaVu Sans")
    ledger.text(
        0.0,
        0.44,
        "D5  D2-5  D2-10  D2-20  D1  D4",
        fontsize=11,
        family="DejaVu Sans",
    )
    ledger.text(
        0.0,
        0.22,
        "matched reference | closed: failed exact alpha-zero equivalence; no positive-alpha outcome",
        fontsize=10,
        family="DejaVu Sans",
        wrap=True,
    )
    ledger.text(
        0.0,
        0.04,
        "AG  Acc-cross  I_AG  I_Acc  Protocol A  Protocol B",
        fontsize=9,
        family="DejaVu Sans",
    )
    return _save_figure_bytes(figure, width_px=FIGURE1_SIZE[0], height_px=FIGURE1_SIZE[1])


def _render_figure2(
    alignment_rows: Sequence[Mapping[str, object]],
    interaction_rows: Sequence[Mapping[str, object]],
) -> tuple[bytes, bytes]:
    endpoint_titles = {
        "d5": "D5",
        "d2_5": "D2-5",
        "d2_10": "D2-10",
        "d2_20": "D2-20",
        "d1": "D1",
        "d4": "D4",
    }
    figure = plt.Figure(constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=[0.16, 1.0, 1.0])
    title_ax = figure.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    title_ax.text(0.01, 0.7, FIGURE_TEXT[4], fontsize=18, family="DejaVu Sans")
    title_ax.text(
        0.48,
        0.7,
        "Protocol A ○ (hollow)   Protocol B ■ (filled)   ↑ favorable   ↓ adverse",
        fontsize=11,
        family="DejaVu Sans",
    )
    title_ax.text(0.58, 0.28, "† D4-B standalone parent coordinate; I_AG/I_Acc use Protocol-A coordinate.", fontsize=9, family="DejaVu Sans")
    grouped_alignment: dict[str, list[Mapping[str, object]]] = {}
    for row in alignment_rows:
        grouped_alignment.setdefault(str(row["endpoint_id"]), []).append(row)
    grouped_interaction: dict[str, list[Mapping[str, object]]] = {}
    for row in interaction_rows:
        grouped_interaction.setdefault(str(row["cell_id"]), []).append(row)
    ordered_endpoints = list(ENDPOINT_ORDER)
    for index, endpoint_id in enumerate(ordered_endpoints):
        panel = grid[1 + index // 3, index % 3].subgridspec(1, 4, wspace=0.08)
        axes = [figure.add_subplot(panel[0, column]) for column in range(4)]
        axes[0].set_title(endpoint_titles[endpoint_id] + ("†" if endpoint_id == "d4" else ""), fontsize=15, family="DejaVu Sans", loc="left")
        rows = sorted(
            grouped_alignment.get(endpoint_id, []),
            key=lambda item: (
                METRIC_OUTPUT_IDS.index(str(item["metric_output_id"])),
                str(item["protocol_id"]),
            ),
        )
        metric_rows = [row for row in rows if row["protocol_id"] == "a"]
        y_positions = list(range(len(metric_rows)))
        for column, (axis, label) in enumerate(zip(axes, ("AG", "Acc-cross", "I_AG", "I_Acc"), strict=True)):
            axis.set_yticks(y_positions)
            axis.set_yticklabels([str(row["metric_output_id"]) for row in metric_rows] if column == 0 else [], fontsize=7, family="DejaVu Sans")
            axis.invert_yaxis(); axis.axvline(0.0, color="#d0d0d0", linewidth=0.7); axis.set_xlabel(label, fontsize=8); axis.axhspan(-0.5, 0.5, color="#f2f2f2", zorder=-2)
        ag_axis, acc_axis, i_ag_axis, i_acc_axis = axes
        by_metric_protocol = {
            (str(row["metric_output_id"]), str(row["protocol_id"])): row for row in rows
        }
        for position, row in zip(y_positions, metric_rows, strict=True):
            a_ag = float(row["ag"])
            a_lo = float(row["ag_lower"])
            a_hi = float(row["ag_upper"])
            a_acc = float(row["acc_cross"])
            a_acc_lo = float(row["acc_cross_lower"])
            a_acc_hi = float(row["acc_cross_upper"])
            ag_axis.errorbar(
                a_ag,
                position - 0.12,
                xerr=[[a_ag - a_lo], [a_hi - a_ag]],
                fmt="o", mfc="white",
                color="#1f77b4",
                markersize=4,
            )
            acc_axis.errorbar(
                a_acc,
                position - 0.12,
                xerr=[[a_acc - a_acc_lo], [a_acc_hi - a_acc]],
                fmt="o", mfc="white", color="#1f77b4",
                markersize=4,
            )
            if row["d_ag_glyph"] != "not_rejected":
                ag_axis.scatter(a_ag, position - 0.30, marker="^" if row["d_ag_glyph"] == "favorable_rejected" else "v", color="#009E73" if row["d_ag_glyph"] == "favorable_rejected" else "#D55E00", s=16, zorder=4)
            if row["d_acc_glyph"] != "not_rejected":
                acc_axis.scatter(a_acc, position - 0.30, marker="^" if row["d_acc_glyph"] == "favorable_rejected" else "v", color="#009E73" if row["d_acc_glyph"] == "favorable_rejected" else "#D55E00", s=16, zorder=4)
            b_row = by_metric_protocol.get((str(row["metric_output_id"]), "b"))
            if b_row is not None and b_row["panel_state"] == "complete":
                b_ag = float(b_row["ag"])
                b_lo = float(b_row["ag_lower"])
                b_hi = float(b_row["ag_upper"])
                b_acc = float(b_row["acc_cross"])
                b_acc_lo = float(b_row["acc_cross_lower"])
                b_acc_hi = float(b_row["acc_cross_upper"])
                ag_axis.errorbar(
                    b_ag,
                    position + 0.12,
                    xerr=[[b_ag - b_lo], [b_hi - b_ag]],
                    fmt="s",
                    color="#d62728",
                    markersize=4,
                )
                acc_axis.errorbar(
                    b_acc,
                    position + 0.12,
                    xerr=[[b_acc - b_acc_lo], [b_acc_hi - b_acc]],
                    fmt="s", color="#d62728",
                    markersize=4,
                )
                if b_row["d_ag_glyph"] != "not_rejected":
                    ag_axis.scatter(b_ag, position + 0.30, marker="^" if b_row["d_ag_glyph"] == "favorable_rejected" else "v", color="#009E73" if b_row["d_ag_glyph"] == "favorable_rejected" else "#D55E00", s=16, zorder=4)
                if b_row["d_acc_glyph"] != "not_rejected":
                    acc_axis.scatter(b_acc, position + 0.30, marker="^" if b_row["d_acc_glyph"] == "favorable_rejected" else "v", color="#009E73" if b_row["d_acc_glyph"] == "favorable_rejected" else "#D55E00", s=16, zorder=4)
        if endpoint_id in grouped_interaction:
            rows_i = sorted(
                grouped_interaction[endpoint_id],
                key=lambda item: METRIC_OUTPUT_IDS.index(str(item["metric_output_id"])),
            )
            for position, row in zip(range(len(rows_i)), rows_i, strict=True):
                if str(row["metric_output_id"]) == "mse":
                    continue
                i_ag = float(row["i_ag"])
                i_ag_lo = float(row["i_ag_lower"])
                i_ag_hi = float(row["i_ag_upper"])
                i_acc = float(row["i_acc"])
                i_acc_lo = float(row["i_acc_lower"])
                i_acc_hi = float(row["i_acc_upper"])
                ag_color = "#2166AC" if row["i_ag_glyph"] == "favorable_rejected" else "#B2182B" if row["i_ag_glyph"] == "adverse_rejected" else "#777777"
                acc_color = "#2166AC" if row["i_acc_glyph"] == "favorable_rejected" else "#B2182B" if row["i_acc_glyph"] == "adverse_rejected" else "#777777"
                i_ag_axis.errorbar(
                    i_ag,
                    position,
                    xerr=[[i_ag - i_ag_lo], [i_ag_hi - i_ag]],
                    fmt="o", color=ag_color,
                    markersize=3.5,
                )
                i_acc_axis.errorbar(
                    i_acc,
                    position,
                    xerr=[[i_acc - i_acc_lo], [i_acc_hi - i_acc]],
                    fmt="o", color=acc_color,
                    markersize=3.5,
                )
        else:
            for axis in (i_ag_axis, i_acc_axis):
                axis.set_facecolor("#efefef"); axis.patch.set_hatch("///")
            i_ag_axis.text(0.5, 0.5, "closed", transform=i_ag_axis.transAxes, ha="center", va="center", fontsize=8)
            i_acc_axis.text(0.5, 0.5, "closed", transform=i_acc_axis.transAxes, ha="center", va="center", fontsize=8)
    return _save_figure_bytes(figure, width_px=FIGURE2_SIZE[0], height_px=FIGURE2_SIZE[1])


def _render_payloads(
    figure1_rows: Sequence[Mapping[str, object]],
    effect_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    interaction_rows: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    figure1_png, figure1_svg = _render_figure1(figure1_rows, effect_rows)
    figure2_png, figure2_svg = _render_figure2(alignment_rows, interaction_rows)
    return {
        "figure1_phase4_response.png": figure1_png,
        "figure1_phase4_response.svg": figure1_svg,
        "figure2_phase4_alignment.png": figure2_png,
        "figure2_phase4_alignment.svg": figure2_svg,
    }


def _payload_projection(
    config: _VerifierConfig,
    *,
    scope_rows: Sequence[Mapping[str, object]],
    parent_rows: Sequence[Mapping[str, object]],
    figure1_rows: Sequence[Mapping[str, object]],
    effect_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    interaction_rows: Sequence[Mapping[str, object]],
    captions: Mapping[str, object],
) -> dict[str, object]:
    return {
        "scope_sha256": _sha256_hex(_csv_bytes(scope_rows, SCOPE_STATUS_FIELDS)),
        "parent_sha256": _sha256_hex(_csv_bytes(parent_rows, PARENT_PANEL_FIELDS)),
        "figure1_sha256": _sha256_hex(
            _csv_bytes(figure1_rows, FIGURE1_RESPONSE_FIELDS)
        ),
        "effect_sha256": _sha256_hex(
            _csv_bytes(effect_rows, FIGURE1_PROTOCOL_EFFECT_FIELDS)
        ),
        "alignment_sha256": _sha256_hex(
            _csv_bytes(alignment_rows, FIGURE2_ALIGNMENT_FIELDS)
        ),
        "interaction_sha256": _sha256_hex(
            _csv_bytes(interaction_rows, FIGURE2_PROTOCOL_INTERACTION_FIELDS)
        ),
        "captions_sha256": _sha256_hex(_canonical_json_bytes(captions)),
    }


def _rebuild_payloads(
    *,
    config: _VerifierConfig,
    inputs: _VerifierInputs,
    worker_count: int,
) -> tuple[str, dict[str, bytes], bytes, bytes]:
    if worker_count < 1:
        raise Phase4FinalFiguresVerifierError("worker_count: must be at least 1")
    scope_rows = _sort_scope_rows(inputs.scope_status_rows, config.scope_ids)
    parent_rows = _sort_parent_rows(
        inputs.parent_panel_rows,
        config.endpoint_order,
        config.protocol_order,
    )
    figure1_rows = _sort_figure1_response_rows(inputs.figure1_response_rows, config)
    effect_rows = _sort_figure1_protocol_effect_rows(
        inputs.figure1_protocol_effect_rows,
        config,
    )
    alignment_rows = _sort_figure2_alignment_rows(
        inputs.figure2_alignment_rows,
        parent_rows,
        config,
    )
    interaction_rows = _sort_figure2_interaction_rows(
        inputs.figure2_interaction_rows,
        config,
    )
    captions = {
        "figure1": str(inputs.captions["figure1"]),
        "figure2": str(inputs.captions["figure2"]),
        "scope_ribbon": FIGURE_TEXT[0],
        "closure_card": FIGURE_TEXT[2],
        "matched_reference_label": FIGURE_TEXT[1],
    }
    _validate_counts(
        config,
        scope_rows=scope_rows,
        parent_rows=parent_rows,
        figure1_rows=figure1_rows,
        effect_rows=effect_rows,
        alignment_rows=alignment_rows,
        interaction_rows=interaction_rows,
    )
    projection = _payload_projection(
        config,
        scope_rows=scope_rows,
        parent_rows=parent_rows,
        figure1_rows=figure1_rows,
        effect_rows=effect_rows,
        alignment_rows=alignment_rows,
        interaction_rows=interaction_rows,
        captions=captions,
    )
    run_id = _stable_run_id(config_sha256=config.sha256, payload_projection=projection)
    figures = _render_payloads(figure1_rows, effect_rows, alignment_rows, interaction_rows)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "status": "complete",
        "figure1_size_px": list(FIGURE1_SIZE),
        "figure2_size_px": list(FIGURE2_SIZE),
        "payload_files": list(config.artifact_payload_files),
        "counts": {
            "scope_status": len(scope_rows),
            "parent_panel_index": len(parent_rows),
            "figure1_response_data": len(figure1_rows),
            "figure1_protocol_effect_data": len(effect_rows),
            "figure2_alignment_data": len(alignment_rows),
            "figure2_protocol_interaction_data": len(interaction_rows),
            "configured_payloads": len(config.artifact_payload_files),
            "artifact_files": len(config.artifact_payload_files) + 2,
        },
    }
    authority_bridge = {
        "state": "synthetic_complete" if config.synthetic_fixture else "real_complete",
        "protocol_a_x_authority": "mse",
        "d1_protocol_b_state": "closed_failed_alpha0_equivalence",
        "claim_boundary": CLAIM_BOUNDARY,
        "parent_receipts": inputs.document.get("parent_receipts", {}),
    }
    preflight = {
        "execution_mode": "deterministic_projection_only",
        "synthetic_fixture": config.synthetic_fixture,
        "validated_counts": manifest["counts"],
    }
    payloads = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": _canonical_json_bytes(authority_bridge),
        "preflight.json": _canonical_json_bytes(preflight),
        "scope_status.csv": _csv_bytes(scope_rows, SCOPE_STATUS_FIELDS),
        "parent_panel_index.csv": _csv_bytes(parent_rows, PARENT_PANEL_FIELDS),
        "figure1_response_data.csv": _csv_bytes(
            figure1_rows,
            FIGURE1_RESPONSE_FIELDS,
        ),
        "figure1_protocol_effect_data.csv": _csv_bytes(
            effect_rows,
            FIGURE1_PROTOCOL_EFFECT_FIELDS,
        ),
        "figure2_alignment_data.csv": _csv_bytes(
            alignment_rows,
            FIGURE2_ALIGNMENT_FIELDS,
        ),
        "figure2_protocol_interaction_data.csv": _csv_bytes(
            interaction_rows,
            FIGURE2_PROTOCOL_INTERACTION_FIELDS,
        ),
        "captions.json": _canonical_json_bytes(captions),
        "manifest.json": _canonical_json_bytes(manifest),
        **figures,
    }
    complete = {
        "schema_version": "phase4-final-figures-marker-v1",
        "run_id": run_id,
        "status": "complete",
    }
    complete_bytes = _canonical_json_bytes(complete)
    sha256sums = _write_sha256sums(payloads, "complete.json", complete_bytes)
    return run_id, payloads, complete_bytes, sha256sums


def verify_phase4_final_figures_from_inputs(
    run_path: Path | str,
    *,
    inputs: object,
    config_path: Path | str,
    worker_count: int,
) -> Phase4FinalFiguresVerifierSummary:
    artifact_path = Path(run_path)
    if not artifact_path.is_dir():
        raise Phase4FinalFiguresVerifierError(f"run_path: not a directory: {artifact_path}")
    actual_files = {
        path.name: path.read_bytes()
        for path in artifact_path.iterdir()
        if path.is_file()
    }
    expected_inventory = set(ARTIFACT_PAYLOAD_FILES) | {"complete.json", "SHA256SUMS"}
    if set(actual_files) != expected_inventory:
        raise Phase4FinalFiguresVerifierError("payload inventory mismatch")
    config_file = Path(config_path)
    config_raw = config_file.read_bytes()
    if actual_files["config.json"] != config_raw:
        raise Phase4FinalFiguresVerifierError("payload config.json mismatch")
    config = _parse_config(config_file, config_raw)
    verifier_inputs = _inputs_from_object(inputs)
    run_id, payloads, complete_bytes, sha256sums = _rebuild_payloads(
        config=config,
        inputs=verifier_inputs,
        worker_count=worker_count,
    )
    rebuilt_files = dict(payloads)
    rebuilt_files["complete.json"] = complete_bytes
    rebuilt_files["SHA256SUMS"] = sha256sums
    comparison_order = (*ARTIFACT_PAYLOAD_FILES, "complete.json", "SHA256SUMS")
    for name in comparison_order:
        if actual_files[name] != rebuilt_files[name]:
            raise Phase4FinalFiguresVerifierError(
                f"payload rebuild semantic mismatch: {name}"
            )
    return Phase4FinalFiguresVerifierSummary(
        path=artifact_path,
        run_id=run_id,
        status="complete",
        verified_file_count=len(expected_inventory),
    )


def verify_phase4_final_figures(
    run_path: Path | str,
    *,
    worker_count: int,
) -> Phase4FinalFiguresVerifierSummary:
    artifact_path = Path(run_path)
    config_path = artifact_path / "config.json"
    config = _parse_config(config_path, config_path.read_bytes())
    inputs = _reconstruct_inputs(config)
    return verify_phase4_final_figures_from_inputs(
        artifact_path,
        inputs=inputs,
        config_path=config_path,
        worker_count=worker_count,
    )


__all__ = [
    "Phase4FinalFiguresVerifierError",
    "Phase4FinalFiguresVerifierSummary",
    "verify_phase4_final_figures",
    "verify_phase4_final_figures_from_inputs",
]
