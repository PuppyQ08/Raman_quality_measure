from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.alignment import (
    AlignmentObservation,
    AlignmentValidationError,
    alignment_gap,
    cross_perturbation_accuracy,
)
from rpe.runner.phase4_protocol_ab_contrast import (
    Phase4ProtocolABContrastError,
    validate_parent_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments" / "robustness" / "w1_axis_v1.json"
SCHEMA_VERSION = "w1-axis-robustness-config-v1"
EXPERIMENT_ID = "w1-axis-robustness-v1"
CONDITION_IDS = (
    "native_all5",
    "native_no_axis",
    "common_grid_all5",
    "common_grid_no_axis",
)


class W1AxisRobustnessError(ValueError):
    pass


@dataclass(frozen=True)
class W1AxisRobustnessConfig:
    path: Path
    document: Mapping[str, object]
    metric_output_ids: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    no_axis_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    paper_alignment_csv: Path
    baseline_atol: float
    baseline_rtol: float
    bootstrap_resamples: int
    sign_flip_resamples: int
    artifact_root: Path
    default_output_root: Path
    parent_phase4_protocol_ab_contrast_config: Path
    panels: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class NativePanelTable:
    panel_id: str
    endpoint_id: str
    protocol_id: str
    parent_key: str
    cluster_kind: str
    cluster_ids: tuple[str, ...]
    metric_output_ids: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_harm: np.ndarray
    downstream_harm: np.ndarray
    within_cluster_counts: np.ndarray | None
    parent_status: str
    common_grid_ready: bool

    def __post_init__(self) -> None:
        metric = np.asarray(self.metric_harm, dtype=np.float64)
        downstream = np.asarray(self.downstream_harm, dtype=np.float64)
        expected_metric = (
            len(self.metric_output_ids),
            len(self.cluster_ids),
            len(self.perturbation_ids),
            len(self.alpha_grid),
        )
        expected_downstream = expected_metric[1:]
        if metric.shape != expected_metric:
            raise W1AxisRobustnessError("metric_harm shape mismatch")
        if downstream.shape != expected_downstream:
            raise W1AxisRobustnessError("downstream_harm shape mismatch")
        if not np.isfinite(metric).all() or not np.isfinite(downstream).all():
            raise W1AxisRobustnessError("native table arrays must be finite")
        metric.setflags(write=False)
        downstream.setflags(write=False)
        object.__setattr__(self, "metric_harm", metric)
        object.__setattr__(self, "downstream_harm", downstream)
        if self.within_cluster_counts is None:
            return
        counts = np.asarray(self.within_cluster_counts, dtype=np.int64)
        if counts.shape != expected_downstream:
            raise W1AxisRobustnessError("within_cluster_counts shape mismatch")
        if np.any(counts <= 0):
            raise W1AxisRobustnessError("within_cluster_counts must be positive")
        counts.setflags(write=False)
        object.__setattr__(self, "within_cluster_counts", counts)


@dataclass(frozen=True)
class W1AxisRobustnessInputs:
    inventory_document: Mapping[str, object]
    coverage_rows: tuple[Mapping[str, object], ...]
    tables: tuple[NativePanelTable, ...]
    paper_rows: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class W1AxisRobustnessSummary:
    path: Path
    run_id: str
    stage: str
    status: str


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


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise W1AxisRobustnessError(f"non-object JSONL row: {path}:{line_number}")
                rows.append(item)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W1AxisRobustnessError(f"cannot read JSONL: {path}") from error
    return rows


def _resolve_path(base: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [(base / path).resolve(), (ROOT / path).resolve()]
    if path.parts and path.parts[0] == ROOT.name:
        candidates.append((ROOT.joinpath(*path.parts[1:])).resolve())
    candidates.append((ROOT.parent / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_output_path(value: object) -> Path:
    """Resolve generated outputs from the project root, never the config folder."""
    path = Path(str(value))
    return path if path.is_absolute() else (ROOT / path).resolve()


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _paper_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row["panel_id"]), str(row["metric_output_id"]))


def _condition_family(condition_id: str) -> str:
    if condition_id.startswith("native_"):
        return "native"
    if condition_id.startswith("common_grid_"):
        return "common_grid"
    raise W1AxisRobustnessError(f"unknown condition: {condition_id}")


def _validate_panel_specs(panels: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    required_ids = (
        "d1_a",
        "d2_5_a",
        "d2_5_b",
        "d2_10_a",
        "d2_10_b",
        "d2_20_a",
        "d2_20_b",
        "d4_a",
        "d4_b",
        "d5_a",
        "d5_b",
    )
    if tuple(str(panel.get("panel_id", "")) for panel in panels) != required_ids:
        raise W1AxisRobustnessError("panel order mismatch")
    if any(str(panel.get("panel_id", "")) == "d1_b" for panel in panels):
        raise W1AxisRobustnessError("D1-B must not be accepted")
    return tuple(MappingProxyType(dict(panel)) for panel in panels)


def parse_w1_axis_robustness_config(path: Path, raw_bytes: bytes) -> W1AxisRobustnessConfig:
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise W1AxisRobustnessError("config is not valid JSON") from error
    if document.get("schema_version") != SCHEMA_VERSION:
        raise W1AxisRobustnessError("config schema mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise W1AxisRobustnessError("config experiment mismatch")
    metric_output_ids = tuple(str(value) for value in document.get("metric_output_ids", ()))
    if len(metric_output_ids) != 13 or metric_output_ids[0] != "mse":
        raise W1AxisRobustnessError("frozen metric grid mismatch")
    perturbation_ids = tuple(str(value) for value in document.get("perturbation_ids", ()))
    if perturbation_ids != ("p08", "p09", "p10", "p11", "p12"):
        raise W1AxisRobustnessError("frozen perturbation grid mismatch")
    no_axis_ids = tuple(str(value) for value in document.get("no_axis_perturbation_ids", ()))
    if no_axis_ids != ("p08", "p09", "p10"):
        raise W1AxisRobustnessError("no-axis perturbation grid mismatch")
    alpha_grid = tuple(float(value) for value in document.get("alpha_grid", ()))
    if alpha_grid != (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8):
        raise W1AxisRobustnessError("frozen positive alpha grid mismatch")
    bootstrap_resamples = int(document.get("bootstrap_resamples", -1))
    sign_flip_resamples = int(document.get("sign_flip_resamples", -1))
    if bootstrap_resamples != 2000 or sign_flip_resamples != 100000:
        raise W1AxisRobustnessError("resample contract mismatch")
    baseline_atol = float(document.get("baseline_atol", "nan"))
    baseline_rtol = float(document.get("baseline_rtol", "nan"))
    if baseline_atol != 1e-10 or baseline_rtol != 1e-8:
        raise W1AxisRobustnessError("baseline tolerance mismatch")
    base = path.parent
    paper_alignment_csv = _resolve_path(base, document["paper_alignment_csv"])
    parent_config = _resolve_path(base, document["parent_phase4_protocol_ab_contrast_config"])
    artifact_root = _resolve_path(base, document.get("artifact_root", ROOT))
    output_root = _resolve_output_path(
        document.get("output_root", ROOT / "results" / "robustness" / "w1_axis_v1")
    )
    panels = _validate_panel_specs(document.get("panels", ()))
    return W1AxisRobustnessConfig(
        path=path.resolve(),
        document=MappingProxyType(document),
        metric_output_ids=metric_output_ids,
        perturbation_ids=perturbation_ids,
        no_axis_perturbation_ids=no_axis_ids,
        alpha_grid=alpha_grid,
        paper_alignment_csv=paper_alignment_csv,
        baseline_atol=baseline_atol,
        baseline_rtol=baseline_rtol,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
        artifact_root=artifact_root,
        default_output_root=output_root,
        parent_phase4_protocol_ab_contrast_config=parent_config,
        panels=panels,
    )


def load_w1_axis_robustness_config(path: Path = DEFAULT_CONFIG) -> W1AxisRobustnessConfig:
    config_path = Path(path)
    return parse_w1_axis_robustness_config(config_path, config_path.read_bytes())


def derive_endpoint_seed(endpoint_id: str, operation_id: str) -> int:
    if operation_id not in {"bootstrap", "sign_flip"}:
        raise W1AxisRobustnessError("operation_id must be bootstrap or sign_flip")
    if endpoint_id.endswith(("_a", "_b")):
        raise W1AxisRobustnessError("endpoint_id must exclude protocol suffixes")
    raw = f"20260920|{endpoint_id}|{operation_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def condition_perturbation_ids(
    config: W1AxisRobustnessConfig,
    condition_id: str,
) -> tuple[str, ...]:
    if condition_id.endswith("all5"):
        return config.perturbation_ids
    if condition_id.endswith("no_axis"):
        return config.no_axis_perturbation_ids
    raise W1AxisRobustnessError(f"unknown condition: {condition_id}")


def condition_observation_count_per_cluster(
    table: NativePanelTable,
    config: W1AxisRobustnessConfig,
    condition_id: str,
) -> int:
    return len(condition_perturbation_ids(config, condition_id)) * len(table.alpha_grid)


def count_oc_family_pairs(
    perturbation_ids: Sequence[str],
    alpha_grid: Sequence[float],
) -> int:
    condition_labels = [
        perturbation_id
        for perturbation_id in perturbation_ids
        for _alpha in alpha_grid
    ]
    total = 0
    for left_index, left in enumerate(condition_labels):
        for right in condition_labels[left_index + 1 :]:
            if left != right:
                total += 1
    return total


def summarize_oc_pair_decomposition(
    *,
    strict_agreement_count: int,
    strict_disagreement_count: int,
    metric_tie_count: int,
    downstream_tie_count: int,
    double_tie_count: int,
) -> Mapping[str, float | int]:
    pair_count = (
        strict_agreement_count
        + strict_disagreement_count
        + metric_tie_count
        + downstream_tie_count
        + double_tie_count
    )
    if pair_count <= 0:
        raise W1AxisRobustnessError("pair_count must be positive")
    tie_count = metric_tie_count + downstream_tie_count + double_tie_count
    accuracy = (
        strict_agreement_count + 0.5 * tie_count
    ) / pair_count
    return MappingProxyType(
        {
            "pair_count": pair_count,
            "strict_agreement_count": strict_agreement_count,
            "strict_disagreement_count": strict_disagreement_count,
            "metric_tie_count": metric_tie_count,
            "downstream_tie_count": downstream_tie_count,
            "double_tie_count": double_tie_count,
            "agreement_fraction": strict_agreement_count / pair_count,
            "tie_fraction": tie_count / pair_count,
            "disagreement_fraction": strict_disagreement_count / pair_count,
            "accuracy": accuracy,
        }
    )


def summarize_task_harm(values: Sequence[float]) -> Mapping[str, float | int]:
    data = [float(value) for value in values]
    positive = [value for value in data if value > 0.0]
    negative = [value for value in data if value < 0.0]
    zeros = [value for value in data if value == 0.0]
    return MappingProxyType(
        {
            "count": len(data),
            "positive_count": len(positive),
            "negative_count": len(negative),
            "zero_count": len(zeros),
            "signed_sum": float(sum(data)),
            "positive_sum": float(sum(positive)),
            "negative_sum": float(sum(negative)),
            "absolute_sum": float(sum(abs(value) for value in data)),
        }
    )


def _observation_state(row: Mapping[str, object], *, legacy_missing_state: bool) -> str:
    state = row.get("state")
    if state is None:
        if legacy_missing_state:
            return "complete"
        raise W1AxisRobustnessError("missing observation state")
    if state != "complete":
        raise W1AxisRobustnessError("observation state must be complete")
    return "complete"


def _load_native_panel_table(
    *,
    panel: Mapping[str, object],
    config: W1AxisRobustnessConfig,
    parent_path: Path,
    parent_status: str,
    common_grid_ready: bool,
) -> NativePanelTable:
    observation_rows = _read_jsonl(parent_path / str(panel["observation_file"]))
    shot_count = panel.get("shot_count")
    if shot_count is not None:
        observation_rows = [
            row
            for row in observation_rows
            if int(row.get("shot_count", -1)) == int(shot_count)
        ]
    cluster_field = str(panel["cluster_field"])
    count_field = panel.get("count_field")
    metric_output_ids = config.metric_output_ids
    perturbation_ids = config.perturbation_ids
    alpha_grid = config.alpha_grid
    metric_index = {value: index for index, value in enumerate(metric_output_ids)}
    perturbation_index = {value: index for index, value in enumerate(perturbation_ids)}
    alpha_index = {float(value).hex(): index for index, value in enumerate(alpha_grid)}
    cluster_ids = tuple(dict.fromkeys(str(row[cluster_field]) for row in observation_rows))
    if not cluster_ids:
        raise W1AxisRobustnessError(f"no clusters reconstructed for {panel['panel_id']}")
    cluster_index = {value: index for index, value in enumerate(cluster_ids)}
    expected_row_count = (
        len(metric_output_ids)
        * len(perturbation_ids)
        * len(alpha_grid)
        * len(cluster_ids)
    )
    if len(observation_rows) != expected_row_count:
        raise W1AxisRobustnessError(f"observation grid row count mismatch: {panel['panel_id']}")
    metric_harm = np.empty(
        (len(metric_output_ids), len(cluster_ids), len(perturbation_ids), len(alpha_grid)),
        dtype=np.float64,
    )
    downstream_harm = np.empty(
        (len(cluster_ids), len(perturbation_ids), len(alpha_grid)),
        dtype=np.float64,
    )
    within_cluster_counts = (
        None
        if count_field is None
        else np.empty((len(cluster_ids), len(perturbation_ids), len(alpha_grid)), dtype=np.int64)
    )
    seen_metric_keys: set[tuple[str, str, str, str]] = set()
    seen_downstream: dict[tuple[str, str, str], float] = {}
    seen_counts: dict[tuple[str, str, str], int] = {}
    for row in observation_rows:
        _observation_state(row, legacy_missing_state=bool(panel.get("legacy_missing_state", False)))
        try:
            metric_output_id = str(row["metric_output_id"])
            perturbation_id = str(row["perturbation_id"])
            alpha_key = float(row["alpha"]).hex()
            cluster_id = str(row[cluster_field])
            metric_value = float(row["metric_harm"])
            downstream_value = float(row["downstream_harm"])
        except (KeyError, TypeError, ValueError) as error:
            raise W1AxisRobustnessError(f"observation schema mismatch: {panel['panel_id']}") from error
        if metric_output_id not in metric_index or perturbation_id not in perturbation_index or alpha_key not in alpha_index or cluster_id not in cluster_index:
            raise W1AxisRobustnessError(f"observation outside frozen grid: {panel['panel_id']}")
        metric_key = (metric_output_id, perturbation_id, alpha_key, cluster_id)
        if metric_key in seen_metric_keys:
            raise W1AxisRobustnessError(f"duplicate observation key: {panel['panel_id']}")
        seen_metric_keys.add(metric_key)
        if not math.isfinite(metric_value) or not math.isfinite(downstream_value):
            raise W1AxisRobustnessError(f"non-finite observation value: {panel['panel_id']}")
        mi = metric_index[metric_output_id]
        ci = cluster_index[cluster_id]
        pi = perturbation_index[perturbation_id]
        ai = alpha_index[alpha_key]
        metric_harm[mi, ci, pi, ai] = metric_value
        condition_key = (cluster_id, perturbation_id, alpha_key)
        prior = seen_downstream.setdefault(condition_key, downstream_value)
        if prior.hex() != downstream_value.hex():
            raise W1AxisRobustnessError(f"inconsistent downstream copies: {panel['panel_id']}")
        downstream_harm[ci, pi, ai] = downstream_value
        if count_field is not None:
            try:
                count_value = int(row[count_field])
            except (KeyError, TypeError, ValueError) as error:
                raise W1AxisRobustnessError(f"invalid count field: {panel['panel_id']}") from error
            if count_value <= 0:
                raise W1AxisRobustnessError(f"non-positive count field: {panel['panel_id']}")
            previous_count = seen_counts.setdefault(condition_key, count_value)
            if previous_count != count_value:
                raise W1AxisRobustnessError(f"inconsistent within-cluster count: {panel['panel_id']}")
            if within_cluster_counts is not None:
                within_cluster_counts[ci, pi, ai] = count_value
    return NativePanelTable(
        panel_id=str(panel["panel_id"]),
        endpoint_id=str(panel["endpoint_id"]),
        protocol_id=str(panel["protocol_id"]),
        parent_key=str(panel["parent_key"]),
        cluster_kind=str(panel["cluster_kind"]),
        cluster_ids=cluster_ids,
        metric_output_ids=metric_output_ids,
        perturbation_ids=perturbation_ids,
        alpha_grid=alpha_grid,
        metric_harm=metric_harm,
        downstream_harm=downstream_harm,
        within_cluster_counts=within_cluster_counts,
        parent_status=parent_status,
        common_grid_ready=common_grid_ready,
    )


def _load_protocol_ab_parent_receipts(config: W1AxisRobustnessConfig) -> Mapping[str, Mapping[str, object]]:
    document = json.loads(config.parent_phase4_protocol_ab_contrast_config.read_text(encoding="utf-8"))
    parents = document.get("parent_artifacts")
    if not isinstance(parents, dict):
        raise W1AxisRobustnessError("phase4 protocol-ab parent registry missing")
    return parents


def load_paper_alignment_rows(path: Path) -> tuple[Mapping[str, str], ...]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise W1AxisRobustnessError(f"cannot read paper alignment CSV: {path}") from error


def build_endpoint_seed_rows(
    panels: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    endpoint_ids = tuple(sorted({str(panel["endpoint_id"]) for panel in panels}))
    rows = []
    for endpoint_id in endpoint_ids:
        for operation_id in ("bootstrap", "sign_flip"):
            rows.append(
                MappingProxyType(
                    {
                        "endpoint_id": endpoint_id,
                        "operation_id": operation_id,
                        "seed": derive_endpoint_seed(endpoint_id, operation_id),
                    }
                )
            )
    return tuple(rows)


def _panel_inventory_row(
    *,
    panel: Mapping[str, object],
    expected: Mapping[str, object],
    config: W1AxisRobustnessConfig,
) -> Mapping[str, object]:
    parent_key = str(panel["parent_key"])
    parent_path = config.artifact_root / str(expected["relative_path"])
    try:
        receipt = validate_parent_artifact(parent_path, expected)
        parent_status = str(receipt["status"])
        native_ready = parent_status == "complete"
        native_status = "ready" if native_ready else f"parent_{parent_status}"
    except (Phase4ProtocolABContrastError, OSError) as error:
        receipt = MappingProxyType(
            {
                "relative_path": str(expected["relative_path"]),
                "status": "missing_or_invalid_parent",
            }
        )
        parent_status = "missing_or_invalid_parent"
        native_ready = False
        native_status = "missing_or_invalid_parent"
    parent_config_path = config.artifact_root / str(panel["parent_config_path"])
    support_grid = None
    support_grid_keys: tuple[str, ...] = ()
    support_grid_status = "missing_support_grid"
    try:
        parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
        support_grid = parent_config.get("support_grid")
        if isinstance(support_grid, dict) and support_grid:
            support_grid_keys = tuple(sorted(support_grid.keys()))
            support_grid_status = "support_grid_present"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        support_grid_status = "parent_config_unreadable"
    if parent_status != "complete":
        common_grid_ready = False
        common_grid_status = (
            "parent_failed"
            if parent_status == "failed"
            else native_status
        )
    elif isinstance(support_grid, dict) and support_grid:
        common_grid_ready = True
        common_grid_status = "ready_not_built"
    else:
        common_grid_ready = False
        common_grid_status = (
            "parent_config_unreadable"
            if support_grid_status == "parent_config_unreadable"
            else "missing_support_grid"
        )
    return MappingProxyType(
        {
            "panel_id": str(panel["panel_id"]),
            "endpoint_id": str(panel["endpoint_id"]),
            "protocol_id": str(panel["protocol_id"]),
            "parent_key": parent_key,
            "parent_relative_path": str(expected["relative_path"]),
            "parent_status": parent_status,
            "native_ready": native_ready,
            "native_status": native_status,
            "common_grid_ready": common_grid_ready,
            "common_grid_status": common_grid_status,
            "support_grid_keys": support_grid_keys,
            "receipt": dict(receipt),
        }
    )


def build_inventory_only_inputs(
    config: W1AxisRobustnessConfig,
) -> W1AxisRobustnessInputs:
    parent_registry = _load_protocol_ab_parent_receipts(config)
    inventory_panels: dict[str, dict[str, object]] = {}
    for panel in config.panels:
        parent_key = str(panel["parent_key"])
        if parent_key == "d1_b":
            raise W1AxisRobustnessError("D1-B must not be accepted")
        expected = parent_registry.get(parent_key)
        if not isinstance(expected, dict):
            raise W1AxisRobustnessError(f"missing parent receipt: {parent_key}")
        inventory_panels[str(panel["panel_id"])] = dict(
            _panel_inventory_row(panel=panel, expected=expected, config=config)
        )
    endpoint_seed_rows = build_endpoint_seed_rows(config.panels)
    inventory_document = {
        "schema_version": "w1-axis-robustness-input-inventory-v1",
        "experiment_id": EXPERIMENT_ID,
        "excluded_parent_keys": ["d1_b"],
        "parent_phase4_protocol_ab_contrast_config": str(config.parent_phase4_protocol_ab_contrast_config),
        "panels": inventory_panels,
        "endpoint_seed_rows": [dict(row) for row in endpoint_seed_rows],
    }
    coverage_rows = tuple(build_coverage_status_rows(config, inventory_panels))
    return W1AxisRobustnessInputs(
        inventory_document=MappingProxyType(inventory_document),
        coverage_rows=coverage_rows,
        tables=(),
        paper_rows=(),
    )


def reconstruct_w1_axis_robustness_inputs(
    config: W1AxisRobustnessConfig,
) -> W1AxisRobustnessInputs:
    inventory_inputs = build_inventory_only_inputs(config)
    tables: list[NativePanelTable] = []
    for panel in config.panels:
        panel_id = str(panel["panel_id"])
        inventory_row = inventory_inputs.inventory_document["panels"][panel_id]
        if not inventory_row["native_ready"]:
            raise W1AxisRobustnessError(f"native reconstruction blocked: {panel_id}:{inventory_row['native_status']}")
        expected = _load_protocol_ab_parent_receipts(config)[str(panel["parent_key"])]
        parent_path = config.artifact_root / str(expected["relative_path"])
        tables.append(
            _load_native_panel_table(
                panel=panel,
                config=config,
                parent_path=parent_path,
                parent_status=str(inventory_row["parent_status"]),
                common_grid_ready=bool(inventory_row["common_grid_ready"]),
            )
        )
    paper_rows = load_paper_alignment_rows(config.paper_alignment_csv)
    return W1AxisRobustnessInputs(
        inventory_document=inventory_inputs.inventory_document,
        coverage_rows=inventory_inputs.coverage_rows,
        tables=tuple(tables),
        paper_rows=paper_rows,
    )


def build_coverage_status_rows(
    config: W1AxisRobustnessConfig,
    inventory: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for panel in config.panels:
        panel_id = str(panel["panel_id"])
        details = inventory[panel_id]
        for condition_id in CONDITION_IDS:
            family = _condition_family(condition_id)
            ready_key = f"{family}_ready"
            status_key = f"{family}_status"
            rows.append(
                {
                    "panel_id": panel_id,
                    "endpoint_id": str(panel["endpoint_id"]),
                    "protocol_id": str(panel["protocol_id"]),
                    "parent_key": str(panel["parent_key"]),
                    "condition_id": condition_id,
                    "family": family,
                    "ready": bool(details[ready_key]),
                    "status": str(details[status_key]),
                    "parent_status": str(details["parent_status"]),
                }
            )
    return rows


def _selected_indices(
    table: NativePanelTable,
    perturbation_ids: Sequence[str],
) -> tuple[int, ...]:
    index = {value: position for position, value in enumerate(table.perturbation_ids)}
    return tuple(index[value] for value in perturbation_ids)


def _observations_for_metric(
    table: NativePanelTable,
    metric_index: int,
    perturbation_positions: Sequence[int],
) -> tuple[AlignmentObservation, ...]:
    return tuple(
        AlignmentObservation(
            cluster_id=cluster_id,
            perturbation_id=table.perturbation_ids[pi],
            alpha=table.alpha_grid[ai],
            metric_harm=float(table.metric_harm[metric_index, ci, pi, ai]),
            downstream_harm=float(table.downstream_harm[ci, pi, ai]),
        )
        for ci, cluster_id in enumerate(table.cluster_ids)
        for pi in perturbation_positions
        for ai in range(len(table.alpha_grid))
    )


def _constant_metric_identity(
    table: NativePanelTable,
    metric_index: int,
    perturbation_positions: Sequence[int],
) -> bool:
    values = table.metric_harm[metric_index, :, perturbation_positions, :]
    return bool(np.all(values == values.reshape(-1)[0]))


def compute_native_point_estimates(
    table: NativePanelTable,
    config: W1AxisRobustnessConfig,
    condition_id: str,
) -> Mapping[str, list[dict[str, object]]]:
    perturbation_ids = condition_perturbation_ids(config, condition_id)
    perturbation_positions = _selected_indices(table, perturbation_ids)
    observations_per_cluster = condition_observation_count_per_cluster(table, config, condition_id)
    pair_count_per_cluster = count_oc_family_pairs(perturbation_ids, table.alpha_grid)
    point_rows: list[dict[str, object]] = []
    ag_detail_rows: list[dict[str, object]] = []
    oc_detail_rows: list[dict[str, object]] = []
    task_harm_rows: list[dict[str, object]] = []
    ag_values: list[float] = []
    acc_values: list[float] = []
    metric_results = []
    for metric_index, metric_output_id in enumerate(table.metric_output_ids):
        observations = _observations_for_metric(table, metric_index, perturbation_positions)
        try:
            acc = cross_perturbation_accuracy(observations)
            if _constant_metric_identity(table, metric_index, perturbation_positions):
                ag = MappingProxyType(
                    {
                        "alignment_gap": 0.0,
                        "cluster_contributions": tuple(
                            MappingProxyType({"cluster_id": cluster_id, "value": 0.0})
                            for cluster_id in table.cluster_ids
                        ),
                    }
                )
            else:
                ag = alignment_gap(observations)
        except AlignmentValidationError as error:
            raise W1AxisRobustnessError(f"{table.panel_id}/{metric_output_id}: {error}") from error
        metric_results.append((metric_output_id, ag, acc))
        ag_value = float(ag["alignment_gap"] if isinstance(ag, Mapping) else ag.alignment_gap)
        ag_values.append(ag_value)
        acc_values.append(float(acc.accuracy))
        point_rows.append(
            {
                "panel_id": table.panel_id,
                "endpoint_id": table.endpoint_id,
                "protocol_id": table.protocol_id,
                "condition_id": condition_id,
                "metric_output_id": metric_output_id,
                "ag": ag_value,
                "acc_cross": float(acc.accuracy),
                "d_ag": None,
                "d_acc": None,
                "state": "complete",
                "observation_count_per_cluster": observations_per_cluster,
                "pair_count_per_cluster": pair_count_per_cluster,
            }
        )
        contributions = (
            ag["cluster_contributions"]
            if isinstance(ag, Mapping)
            else ag.cluster_contributions
        )
        for contribution in contributions:
            ag_detail_rows.append(
                {
                    "panel_id": table.panel_id,
                    "endpoint_id": table.endpoint_id,
                    "protocol_id": table.protocol_id,
                    "condition_id": condition_id,
                    "metric_output_id": metric_output_id,
                    "cluster_id": str(contribution["cluster_id"] if isinstance(contribution, Mapping) else contribution.cluster_id),
                    "ag_cluster_contribution": float(contribution["value"] if isinstance(contribution, Mapping) else contribution.value),
                }
            )
        for cluster in acc.cluster_contributions:
            summary = summarize_oc_pair_decomposition(
                strict_agreement_count=cluster.strict_agreement_count,
                strict_disagreement_count=cluster.strict_disagreement_count,
                metric_tie_count=cluster.metric_tie_count,
                downstream_tie_count=cluster.downstream_tie_count,
                double_tie_count=cluster.double_tie_count,
            )
            oc_detail_rows.append(
                {
                    "panel_id": table.panel_id,
                    "endpoint_id": table.endpoint_id,
                    "protocol_id": table.protocol_id,
                    "condition_id": condition_id,
                    "metric_output_id": metric_output_id,
                    "cluster_id": cluster.cluster_id,
                    "pair_count": int(summary["pair_count"]),
                    "strict_agreement_count": int(summary["strict_agreement_count"]),
                    "strict_disagreement_count": int(summary["strict_disagreement_count"]),
                    "metric_tie_count": int(summary["metric_tie_count"]),
                    "downstream_tie_count": int(summary["downstream_tie_count"]),
                    "double_tie_count": int(summary["double_tie_count"]),
                    "agreement_fraction": float(summary["agreement_fraction"]),
                    "tie_fraction": float(summary["tie_fraction"]),
                    "disagreement_fraction": float(summary["disagreement_fraction"]),
                    "accuracy": float(summary["accuracy"]),
                }
            )
    reference_ag = ag_values[0]
    reference_acc = acc_values[0]
    for row, value_ag, value_acc in zip(point_rows[1:], ag_values[1:], acc_values[1:]):
        row["d_ag"] = reference_ag - value_ag
        row["d_acc"] = value_acc - reference_acc
    selected_harm = table.downstream_harm[:, perturbation_positions, :]
    for cluster_index, cluster_id in enumerate(table.cluster_ids):
        summary = summarize_task_harm(selected_harm[cluster_index].reshape(-1))
        task_harm_rows.append(
            {
                "panel_id": table.panel_id,
                "endpoint_id": table.endpoint_id,
                "protocol_id": table.protocol_id,
                "condition_id": condition_id,
                "cluster_id": cluster_id,
                **dict(summary),
            }
        )
    overall = summarize_task_harm(selected_harm.reshape(-1))
    task_harm_rows.append(
        {
            "panel_id": table.panel_id,
            "endpoint_id": table.endpoint_id,
            "protocol_id": table.protocol_id,
            "condition_id": condition_id,
            "cluster_id": "__all__",
            **dict(overall),
        }
    )
    return MappingProxyType(
        {
            "point_rows": point_rows,
            "ag_detail_rows": ag_detail_rows,
            "oc_detail_rows": oc_detail_rows,
            "task_harm_rows": task_harm_rows,
        }
    )


def compute_baseline_reproduction_rows(
    *,
    point_rows: Sequence[Mapping[str, object]],
    paper_rows: Sequence[Mapping[str, str]],
    atol: float,
    rtol: float,
) -> list[dict[str, object]]:
    point_by_key = {
        (str(row["panel_id"]), str(row["metric_output_id"])): row
        for row in point_rows
    }
    paper_by_key = {_paper_key(row): row for row in paper_rows}
    ordered_keys: list[tuple[str, str]] = []
    for row in paper_rows:
        ordered_keys.append(_paper_key(row))
    for key in sorted(set(point_by_key) - set(paper_by_key)):
        ordered_keys.append(key)
    rows: list[dict[str, object]] = []
    for key in ordered_keys:
        point = point_by_key.get(key)
        paper = paper_by_key.get(key)
        status = "matched"
        mismatches: list[str] = []
        actual_ag = None if point is None else float(point["ag"])
        actual_acc = None if point is None else float(point["acc_cross"])
        actual_d_ag = None if point is None else point.get("d_ag")
        actual_d_acc = None if point is None else point.get("d_acc")
        paper_ag = None if paper is None else _float_or_none(paper.get("ag"))
        paper_acc = None if paper is None else _float_or_none(paper.get("acc_cross"))
        paper_d_ag = None if paper is None else _float_or_none(paper.get("d_ag"))
        paper_d_acc = None if paper is None else _float_or_none(paper.get("d_acc"))
        if point is None:
            status = "missing_native_row"
        elif paper is None:
            status = "missing_reference_row"
        else:
            for name, left, right in (
                ("ag", actual_ag, paper_ag),
                ("acc_cross", actual_acc, paper_acc),
                ("d_ag", None if actual_d_ag is None else float(actual_d_ag), paper_d_ag),
                ("d_acc", None if actual_d_acc is None else float(actual_d_acc), paper_d_acc),
            ):
                if left is None and right is None:
                    continue
                if left is None or right is None or not np.isclose(left, right, atol=atol, rtol=rtol):
                    mismatches.append(name)
            if mismatches:
                status = "mismatched"
        rows.append(
            {
                "panel_id": key[0],
                "metric_output_id": key[1],
                "status": status,
                "mismatch_fields": ",".join(mismatches),
                "actual_ag": actual_ag,
                "paper_ag": paper_ag,
                "actual_acc_cross": actual_acc,
                "paper_acc_cross": paper_acc,
                "actual_d_ag": None if actual_d_ag is None else float(actual_d_ag),
                "paper_d_ag": paper_d_ag,
                "actual_d_acc": None if actual_d_acc is None else float(actual_d_acc),
                "paper_d_acc": paper_d_acc,
                "native_state": None if point is None else point.get("state"),
                "paper_panel_state": "missing_reference_row" if paper is None else paper.get("panel_state"),
            }
        )
    return rows


def _rows_to_csv(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _run_id(
    config: W1AxisRobustnessConfig, stage: str, *, max_records_per_dataset: int | None = None
) -> str:
    document = {
        "config_path": str(config.path),
        "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest() if config.path.is_file() else "synthetic",
        "stage": stage,
        "execution_mode": "full" if max_records_per_dataset is None else "canary",
        "max_records_per_dataset": max_records_per_dataset,
    }
    return "w1-axis-robustness-" + hashlib.sha256(_canonical_json_bytes(document)).hexdigest()[:16]


def _marker_name(stage: str, status: str) -> str:
    suffix = "complete" if status == "complete" else "incomplete"
    return f"{stage}.{suffix}.json"


def build_common_grid_resample_stage_payloads(
    *,
    config: W1AxisRobustnessConfig,
    inputs: W1AxisRobustnessInputs,
    resume: bool,
    max_records_per_dataset: int | None = None,
    worker_count: int = 1,
) -> tuple[dict[str, bytes], Mapping[str, object]]:
    stage_started = time.monotonic()
    from rpe.runner.w1_axis_common_grid import (
        common_grid_numerical_code_identity,
        evaluate_or_load_real_common_grid_dataset,
        finalize_common_grid_panel_result,
        load_common_grid_support_specs,
        materialize_real_common_grid_dataset,
    )

    specs = load_common_grid_support_specs()
    ready_panel_ids = [
        str(row["panel_id"])
        for row in inputs.coverage_rows
        if str(row["condition_id"]) == "common_grid_all5" and bool(row["ready"])
    ]
    support_by_panel_id: dict[str, Mapping[str, object]] = {}
    for panel in config.panels:
        panel_id = str(panel["panel_id"])
        if panel_id.startswith("d2_") or panel_id.startswith("d1_"):
            support_by_panel_id[panel_id] = {
                "dataset_id": "bacteria",
                "point_count": specs["bacteria"].point_count,
                "axis_sha256_f64": specs["bacteria"].axis_sha256_f64,
            }
        elif panel_id.startswith("d4_"):
            support_by_panel_id[panel_id] = {
                "dataset_id": "sugar",
                "point_count": specs["sugar"].point_count,
                "axis_sha256_f64": specs["sugar"].axis_sha256_f64,
            }
        elif panel_id.startswith("d5_"):
            support_by_panel_id[panel_id] = {
                "dataset_id": "d5",
                "point_count": specs["d5"].point_count,
                "axis_sha256_f64": specs["d5"].axis_sha256_f64,
            }
    code_sha256, numerical_code_dependency_sha256s = common_grid_numerical_code_identity(ROOT)
    cache_root = config.default_output_root / "record_caches"
    result_by_dataset = {}
    end_to_end_elapsed_by_dataset = {}
    for dataset_id in ("bacteria", "sugar", "d5"):
        dataset_started = time.monotonic()
        dataset = materialize_real_common_grid_dataset(
            dataset_id, root=config.artifact_root, max_records_per_dataset=max_records_per_dataset,
        )
        result_by_dataset[dataset_id] = evaluate_or_load_real_common_grid_dataset(
            dataset, cache_root=cache_root / dataset_id, resume=resume, code_sha256=code_sha256, worker_count=worker_count,
            progress_started_at=dataset_started,
        )
        end_to_end_elapsed_by_dataset[dataset_id] = time.monotonic() - dataset_started
    parent_by_panel = {table.panel_id: table for table in inputs.tables}
    panel_payloads: list[dict[str, object]] = []
    panel_status_by_id: dict[str, str] = {}
    for panel_id in ready_panel_ids:
        dataset_id = str(support_by_panel_id[panel_id]["dataset_id"])
        dataset_result = result_by_dataset[dataset_id]
        parent = parent_by_panel.get(panel_id)
        failure_rows = list(dataset_result.failure_rows)
        if dataset_result.status == "canary_partial":
            failure_rows.append({
                "panel_id": panel_id, "reason_code": "canary_record_limit",
                "processed_record_count": dataset_result.processed_record_count,
                "planned_record_count": dataset_result.planned_record_count,
            })
        if parent is None:
            panel_payloads.append({"panel_id": panel_id, "status": "parent_table_missing", "failure_rows": failure_rows})
            panel_status_by_id[panel_id] = "parent_table_missing"
            continue
        finalized = finalize_common_grid_panel_result(
            panel_id=panel_id, parent_table=parent, aggregated_rows=dataset_result.aggregated_rows, failure_rows=failure_rows
        )
        entry: dict[str, object] = {
            "panel_id": panel_id, "status": finalized.status, "failure_rows": list(finalized.failure_rows),
            "complete_metric_output_ids": list(finalized.complete_metric_output_ids),
            "incomplete_metric_output_ids": list(finalized.incomplete_metric_output_ids),
        }
        if finalized.table is not None:
            entry["table"] = {
                "cluster_ids": list(finalized.table.cluster_ids),
                "metric_output_ids": list(finalized.table.metric_output_ids),
                "perturbation_ids": list(finalized.table.perturbation_ids),
                "alpha_grid": list(finalized.table.alpha_grid),
                "metric_harm": finalized.table.metric_harm.tolist(),
                "downstream_harm": finalized.table.downstream_harm.tolist(),
                "within_cluster_counts": None if finalized.table.within_cluster_counts is None else finalized.table.within_cluster_counts.tolist(),
            }
        panel_payloads.append(entry)
        panel_status_by_id[panel_id] = finalized.status
    incomplete_panel_ids = {
        panel_id
        for panel_id in ready_panel_ids
        if panel_status_by_id.get(panel_id) != "complete"
    }
    if max_records_per_dataset is not None:
        # A record cap is evidence-only by definition.  Never publish it as a
        # complete analysis even if a synthetic/empty panel registry is used.
        incomplete_panel_ids.update(ready_panel_ids)
    payload = {
        "schema_version": "w1-axis-common-grid-stage-v1",
        "resume": bool(resume),
        "max_records_per_dataset": None if max_records_per_dataset is None else int(max_records_per_dataset),
        "worker_count": int(worker_count),
        "application_elapsed_seconds": time.monotonic() - stage_started,
        "panel_count": len(ready_panel_ids),
        "complete_panel_count": len(ready_panel_ids) - len(incomplete_panel_ids),
        "incomplete_panel_count": len(incomplete_panel_ids) if ready_panel_ids else (1 if max_records_per_dataset is not None else 0),
        "dataset_results": {
            dataset_id: {
                "status": result.status,
                "processed_record_count": result.processed_record_count,
                "planned_record_count": result.planned_record_count,
                "elapsed_seconds": result.elapsed_seconds,
                "end_to_end_elapsed_seconds": end_to_end_elapsed_by_dataset[dataset_id],
                "cache_hit_count": result.cache_hit_count,
                "cache_hit": result.cache_hit,
                "requested_worker_count": result.requested_worker_count,
                "effective_worker_count": result.effective_worker_count,
                "p10_worker_capacity": result.p10_worker_capacity,
                "worker_process_ids": list(result.worker_process_ids),
                "worker_process_count": len(result.worker_process_ids),
                "worker_blas_thread_limits": list(result.worker_blas_thread_limits),
                "aggregated_rows": list(result.aggregated_rows),
                "failure_rows": list(result.failure_rows),
            }
            for dataset_id, result in result_by_dataset.items()
        },
        "numerical_code_sha256": code_sha256,
        "numerical_code_dependency_sha256s": dict(numerical_code_dependency_sha256s),
        "panel_tables": panel_payloads,
        "panels": [
            {
                "panel_id": panel_id,
                **dict(support_by_panel_id[panel_id]),
                "status": panel_status_by_id.get(panel_id, "incomplete_grid"),
            }
            for panel_id in ready_panel_ids
        ],
    }
    return (
        {"common_grid_tables.json": _canonical_json_bytes(payload)},
        payload,
    )


def _stage_payloads(
    *,
    config: W1AxisRobustnessConfig,
    inputs: W1AxisRobustnessInputs,
    stage: str,
    resume: bool,
    max_records_per_dataset: int | None,
    worker_count: int,
) -> tuple[dict[str, bytes], str, str, Mapping[str, object]]:
    endpoint_seed_rows = tuple(
        dict(row)
        for row in inputs.inventory_document.get("endpoint_seed_rows", ())
    )
    payloads: dict[str, bytes] = {
        "config.json": _canonical_json_bytes(dict(config.document)),
        "input_inventory.json": _canonical_json_bytes(dict(inputs.inventory_document)),
        "coverage_status.csv": _rows_to_csv(
            inputs.coverage_rows,
            (
                "panel_id",
                "endpoint_id",
                "protocol_id",
                "parent_key",
                "condition_id",
                "family",
                "ready",
                "status",
                "parent_status",
            ),
        ),
        "endpoint_seeds.csv": _rows_to_csv(
            endpoint_seed_rows,
            ("endpoint_id", "operation_id", "seed"),
        ),
    }
    manifest: dict[str, object] = {
        "schema_version": "w1-axis-robustness-manifest-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "execution_mode": "full" if max_records_per_dataset is None else "canary",
        "max_records_per_dataset": max_records_per_dataset,
        "worker_count_requested": worker_count,
        "status": "complete",
        "payload_files": [],
        "endpoint_seed_count": len(endpoint_seed_rows),
    }
    status = "complete"
    if stage in {"resample", "all"}:
        resample_payloads, resample_manifest = build_common_grid_resample_stage_payloads(
            config=config,
            inputs=inputs,
            resume=resume,
            max_records_per_dataset=max_records_per_dataset,
            worker_count=worker_count,
        )
        payloads.update(resample_payloads)
        manifest["common_grid_panel_count"] = int(resample_manifest["panel_count"])
        manifest["common_grid_complete_panel_count"] = int(resample_manifest["complete_panel_count"])
        manifest["common_grid_incomplete_panel_count"] = int(resample_manifest["incomplete_panel_count"])
        if int(resample_manifest["incomplete_panel_count"]) > 0:
            status = "incomplete"
            manifest["status"] = "incomplete"
            manifest["reason"] = "common_grid_partial_or_failed"
        elif stage == "all":
            # The resample data is available, but Task 3 owns inference and
            # statistical summaries; keep all-stage honest until it consumes
            # the complete common-grid tables.
            status = "incomplete"
            manifest["status"] = "incomplete"
            manifest["reason"] = "common_grid_resample_and_summary_deferred"
    if stage in {"reproduce", "native", "all"}:
        native_all5_point_rows: list[dict[str, object]] = []
        native_all5_ag_details: list[dict[str, object]] = []
        native_all5_oc_details: list[dict[str, object]] = []
        native_all5_task_harm: list[dict[str, object]] = []
        native_no_axis_point_rows: list[dict[str, object]] = []
        native_no_axis_ag_details: list[dict[str, object]] = []
        native_no_axis_oc_details: list[dict[str, object]] = []
        native_no_axis_task_harm: list[dict[str, object]] = []
        for table in inputs.tables:
            native_all5 = compute_native_point_estimates(table, config, "native_all5")
            native_all5_point_rows.extend(native_all5["point_rows"])
            native_all5_ag_details.extend(native_all5["ag_detail_rows"])
            native_all5_oc_details.extend(native_all5["oc_detail_rows"])
            native_all5_task_harm.extend(native_all5["task_harm_rows"])
            if stage in {"native", "all"}:
                native_no_axis = compute_native_point_estimates(table, config, "native_no_axis")
                native_no_axis_point_rows.extend(native_no_axis["point_rows"])
                native_no_axis_ag_details.extend(native_no_axis["ag_detail_rows"])
                native_no_axis_oc_details.extend(native_no_axis["oc_detail_rows"])
                native_no_axis_task_harm.extend(native_no_axis["task_harm_rows"])
        baseline_rows = compute_baseline_reproduction_rows(
            point_rows=native_all5_point_rows,
            paper_rows=inputs.paper_rows,
            atol=config.baseline_atol,
            rtol=config.baseline_rtol,
        )
        payloads["baseline_reproduction.csv"] = _rows_to_csv(
            baseline_rows,
            (
                "panel_id",
                "metric_output_id",
                "status",
                "mismatch_fields",
                "actual_ag",
                "paper_ag",
                "actual_acc_cross",
                "paper_acc_cross",
                "actual_d_ag",
                "paper_d_ag",
                "actual_d_acc",
                "paper_d_acc",
                "native_state",
                "paper_panel_state",
            ),
        )
        manifest["baseline_row_count"] = len(baseline_rows)
        if stage in {"native", "all"}:
            payloads["native_all5_point_estimates.csv"] = _rows_to_csv(
                native_all5_point_rows,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "metric_output_id",
                    "ag",
                    "acc_cross",
                    "d_ag",
                    "d_acc",
                    "state",
                    "observation_count_per_cluster",
                    "pair_count_per_cluster",
                ),
            )
            payloads["native_no_axis_point_estimates.csv"] = _rows_to_csv(
                native_no_axis_point_rows,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "metric_output_id",
                    "ag",
                    "acc_cross",
                    "d_ag",
                    "d_acc",
                    "state",
                    "observation_count_per_cluster",
                    "pair_count_per_cluster",
                ),
            )
            payloads["native_all5_ag_details.csv"] = _rows_to_csv(
                native_all5_ag_details,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "metric_output_id",
                    "cluster_id",
                    "ag_cluster_contribution",
                ),
            )
            payloads["native_no_axis_ag_details.csv"] = _rows_to_csv(
                native_no_axis_ag_details,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "metric_output_id",
                    "cluster_id",
                    "ag_cluster_contribution",
                ),
            )
            payloads["native_all5_oc_details.csv"] = _rows_to_csv(
                native_all5_oc_details,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "metric_output_id",
                    "cluster_id",
                    "pair_count",
                    "strict_agreement_count",
                    "strict_disagreement_count",
                    "metric_tie_count",
                    "downstream_tie_count",
                    "double_tie_count",
                    "agreement_fraction",
                    "tie_fraction",
                    "disagreement_fraction",
                    "accuracy",
                ),
            )
            payloads["native_no_axis_oc_details.csv"] = _rows_to_csv(
                native_no_axis_oc_details,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "metric_output_id",
                    "cluster_id",
                    "pair_count",
                    "strict_agreement_count",
                    "strict_disagreement_count",
                    "metric_tie_count",
                    "downstream_tie_count",
                    "double_tie_count",
                    "agreement_fraction",
                    "tie_fraction",
                    "disagreement_fraction",
                    "accuracy",
                ),
            )
            payloads["native_all5_task_harm.csv"] = _rows_to_csv(
                native_all5_task_harm,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "cluster_id",
                    "count",
                    "positive_count",
                    "negative_count",
                    "zero_count",
                    "signed_sum",
                    "positive_sum",
                    "negative_sum",
                    "absolute_sum",
                ),
            )
            payloads["native_no_axis_task_harm.csv"] = _rows_to_csv(
                native_no_axis_task_harm,
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "condition_id",
                    "cluster_id",
                    "count",
                    "positive_count",
                    "negative_count",
                    "zero_count",
                    "signed_sum",
                    "positive_sum",
                    "negative_sum",
                    "absolute_sum",
                ),
            )
            manifest["native_all5_row_count"] = len(native_all5_point_rows)
            manifest["native_no_axis_row_count"] = len(native_no_axis_point_rows)
    manifest["payload_files"] = tuple(sorted((*payloads.keys(), "manifest.json")))
    payloads["manifest.json"] = _canonical_json_bytes(manifest)
    marker = {
        "schema_version": "w1-axis-robustness-marker-v1",
        "stage": stage,
        "status": status,
    }
    if status != "complete":
        marker["reason"] = str(manifest.get("reason", "common_grid_partial_or_failed"))
    marker_name = _marker_name(stage, status)
    return payloads, marker_name, status, marker


def _write_stage_payloads(
    staging_path: Path,
    payloads: Mapping[str, bytes],
) -> None:
    for name, raw in payloads.items():
        (staging_path / name).write_bytes(raw)


def _existing_summary(
    run_path: Path, run_id: str, stage: str, *, max_records_per_dataset: int | None
) -> W1AxisRobustnessSummary | None:
    for status in ("complete", "incomplete"):
        marker_path = run_path / _marker_name(stage, status)
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
            if (
                manifest.get("execution_mode") != ("full" if max_records_per_dataset is None else "canary")
                or manifest.get("max_records_per_dataset") != max_records_per_dataset
            ):
                raise W1AxisRobustnessError("resume execution identity mismatch")
            return W1AxisRobustnessSummary(
                path=run_path,
                run_id=run_id,
                stage=stage,
                status=str(marker["status"]),
            )
    return None


def build_w1_axis_robustness_from_inputs(
    output_root: Path,
    *,
    config: W1AxisRobustnessConfig,
    inputs: W1AxisRobustnessInputs,
    stage: str,
    resume: bool,
    max_records_per_dataset: int | None = None,
    worker_count: int = 1,
) -> W1AxisRobustnessSummary:
    if stage not in {"inventory", "reproduce", "native", "resample", "all"}:
        raise W1AxisRobustnessError("invalid stage")
    run_id = _run_id(config, stage, max_records_per_dataset=max_records_per_dataset)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / run_id
    existing = _existing_summary(
        run_path, run_id, stage, max_records_per_dataset=max_records_per_dataset
    )
    if existing is not None:
        if resume:
            return existing
        raise W1AxisRobustnessError("append-only run path already exists")
    if run_path.exists():
        raise W1AxisRobustnessError("append-only run path already exists")
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.staging-", dir=str(root))
    )
    staging_path = staging_parent / run_id
    staging_path.mkdir()
    try:
        payloads, marker_name, status, marker = _stage_payloads(
            config=config,
            inputs=inputs,
            stage=stage,
            resume=resume,
            max_records_per_dataset=max_records_per_dataset,
            worker_count=worker_count,
        )
        _write_stage_payloads(staging_path, payloads)
        (staging_path / marker_name).write_bytes(_canonical_json_bytes(marker))
        staging_path.rename(run_path)
    except Exception:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
        raise
    if staging_parent.exists():
        shutil.rmtree(staging_parent)
    return W1AxisRobustnessSummary(path=run_path, run_id=run_id, stage=stage, status=status)


def build_w1_axis_robustness(
    *,
    config_path: Path = DEFAULT_CONFIG,
    artifact_root: Path | None = None,
    output_root: Path | None = None,
    stage: str = "all",
    worker_count: int | None = None,
    max_records_per_dataset: int | None = None,
    resume: bool = False,
) -> W1AxisRobustnessSummary:
    if worker_count is not None and worker_count <= 0:
        raise W1AxisRobustnessError("worker_count must be positive")
    if max_records_per_dataset is not None and max_records_per_dataset <= 0:
        raise W1AxisRobustnessError("max_records_per_dataset must be positive")
    config = load_w1_axis_robustness_config(config_path)
    if artifact_root is not None or output_root is not None:
        config = W1AxisRobustnessConfig(
            path=config.path,
            document=config.document,
            metric_output_ids=config.metric_output_ids,
            perturbation_ids=config.perturbation_ids,
            no_axis_perturbation_ids=config.no_axis_perturbation_ids,
            alpha_grid=config.alpha_grid,
            paper_alignment_csv=config.paper_alignment_csv,
            baseline_atol=config.baseline_atol,
            baseline_rtol=config.baseline_rtol,
            bootstrap_resamples=config.bootstrap_resamples,
            sign_flip_resamples=config.sign_flip_resamples,
            artifact_root=Path(artifact_root) if artifact_root is not None else config.artifact_root,
            default_output_root=Path(output_root) if output_root is not None else config.default_output_root,
            parent_phase4_protocol_ab_contrast_config=config.parent_phase4_protocol_ab_contrast_config,
            panels=config.panels,
        )
    if stage == "inventory":
        inputs = build_inventory_only_inputs(config)
    else:
        inputs = reconstruct_w1_axis_robustness_inputs(config)
    return build_w1_axis_robustness_from_inputs(
        config.default_output_root,
        config=config,
        inputs=inputs,
        stage=stage,
        max_records_per_dataset=max_records_per_dataset,
        worker_count=1 if worker_count is None else worker_count,
        resume=resume,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "EXPERIMENT_ID",
    "NativePanelTable",
    "W1AxisRobustnessConfig",
    "W1AxisRobustnessError",
    "W1AxisRobustnessInputs",
    "W1AxisRobustnessSummary",
    "build_endpoint_seed_rows",
    "build_inventory_only_inputs",
    "build_coverage_status_rows",
    "build_w1_axis_robustness",
    "build_w1_axis_robustness_from_inputs",
    "compute_baseline_reproduction_rows",
    "compute_native_point_estimates",
    "condition_observation_count_per_cluster",
    "condition_perturbation_ids",
    "count_oc_family_pairs",
    "derive_endpoint_seed",
    "load_paper_alignment_rows",
    "load_w1_axis_robustness_config",
    "parse_w1_axis_robustness_config",
    "reconstruct_w1_axis_robustness_inputs",
    "summarize_oc_pair_decomposition",
    "summarize_task_harm",
]
