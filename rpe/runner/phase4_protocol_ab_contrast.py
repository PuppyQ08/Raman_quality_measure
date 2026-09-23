from __future__ import annotations

import json
import math
import platform
import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import isotonic_regression
from sklearn import __version__ as sklearn_version
from threadpoolctl import __version__ as threadpoolctl_version

from rpe.alignment import (
    AlignmentObservation,
    AlignmentValidationError,
    alignment_gap,
    cross_perturbation_accuracy,
    holm_step_down,
    paired_contribution_sign_flip,
)
from rpe.runner.phase4_protocol_ab_contrast_authority import (
    ALPHAS,
    ARTIFACT_PAYLOAD_FILES,
    ARTIFACT_SCHEMA_VERSION,
    CELL_IDS,
    CLAIM_BOUNDARY,
    CODE_RELATIVE_PATHS,
    CONFIG_BYTES,
    CONFIG_SHA256,
    DEFAULT_CONFIG,
    DOWNSTREAM_CSV_FIELDS,
    EVALUABLE_CELL_IDS,
    EXPERIMENT_ID,
    MARKER_SCHEMA_VERSION,
    METRIC_CSV_FIELDS,
    METRIC_OUTPUT_IDS,
    PERTURBATION_IDS,
    ROOT,
    SCHEMA_VERSION,
    canonical_json_bytes,
    csv_bytes,
    jsonl_bytes,
    sha256_file,
    sha256_hex,
    stable_run_id,
    write_sha256sums,
)


class Phase4ProtocolABContrastError(ValueError):
    pass


@dataclass(frozen=True)
class Phase4ProtocolABContrastConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    artifact_payload_files: tuple[str, ...]
    cell_ids: tuple[str, ...]
    metric_output_ids: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    synthetic_fixture: bool


@dataclass(frozen=True)
class ProtocolABCellInput:
    cell_id: str
    cluster_kind: str
    cluster_ids: tuple[str, ...]
    metric_output_ids: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_harm: np.ndarray
    downstream_a: np.ndarray
    downstream_b: np.ndarray
    metric_reconciliation: Mapping[str, object]
    parent_alignment_a: Mapping[str, Mapping[str, object]] | None = None
    parent_alignment_b: Mapping[str, Mapping[str, object]] | None = None

    def __post_init__(self) -> None:
        metric = np.asarray(self.metric_harm, dtype=np.float64)
        a = np.asarray(self.downstream_a, dtype=np.float64)
        b = np.asarray(self.downstream_b, dtype=np.float64)
        expected_metric = (
            len(self.metric_output_ids), len(self.cluster_ids),
            len(self.perturbation_ids), len(self.alpha_grid),
        )
        expected_downstream = expected_metric[1:]
        if metric.shape != expected_metric or a.shape != expected_downstream or b.shape != expected_downstream:
            raise Phase4ProtocolABContrastError("cell arrays have incompatible shapes")
        if not np.isfinite(metric).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise Phase4ProtocolABContrastError("cell arrays must be finite")
        if len(set(self.cluster_ids)) != len(self.cluster_ids):
            raise Phase4ProtocolABContrastError("cluster IDs must be unique")
        metric.setflags(write=False)
        a.setflags(write=False)
        b.setflags(write=False)
        object.__setattr__(self, "metric_harm", metric)
        object.__setattr__(self, "downstream_a", a)
        object.__setattr__(self, "downstream_b", b)
        object.__setattr__(self, "metric_reconciliation", MappingProxyType(dict(self.metric_reconciliation)))
        object.__setattr__(self, "parent_alignment_a", MappingProxyType({
            str(key): MappingProxyType(dict(value))
            for key, value in (self.parent_alignment_a or {}).items()
        }))
        object.__setattr__(self, "parent_alignment_b", MappingProxyType({
            str(key): MappingProxyType(dict(value))
            for key, value in (self.parent_alignment_b or {}).items()
        }))


@dataclass(frozen=True)
class Phase4ProtocolABContrastInputs:
    cells: tuple[ProtocolABCellInput, ...]
    endpoint_status_rows: tuple[Mapping[str, object], ...]
    authority_bridge: Mapping[str, object]
    preflight: Mapping[str, object]


@dataclass(frozen=True)
class Phase4ProtocolABContrastSummary:
    path: Path
    run_id: str
    status: str
    endpoint_status_count: int
    bootstrap_result_count: int
    holm_slot_count: int


def _environment_authority() -> dict[str, str]:
    import scipy
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl_version,
    }


def _code_authority() -> dict[str, dict[str, object]]:
    result = {}
    for relative in CODE_RELATIVE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise Phase4ProtocolABContrastError(f"missing code authority: {relative}")
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def parse_phase4_protocol_ab_contrast_config(
    path: Path,
    raw_bytes: bytes,
    *,
    require_frozen_identity: bool = True,
) -> Phase4ProtocolABContrastConfig:
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4ProtocolABContrastError("config is not valid JSON") from error
    if canonical_json_bytes(document) != raw_bytes:
        raise Phase4ProtocolABContrastError("config is not canonical JSON")
    digest = sha256_hex(raw_bytes)
    synthetic = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity:
        if CONFIG_BYTES <= 0 or not CONFIG_SHA256:
            raise Phase4ProtocolABContrastError("real config identity is not frozen")
        if len(raw_bytes) != CONFIG_BYTES or digest != CONFIG_SHA256:
            raise Phase4ProtocolABContrastError("config frozen identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4ProtocolABContrastError("config schema mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4ProtocolABContrastError("config experiment mismatch")
    if document.get("claim_boundary") != CLAIM_BOUNDARY:
        raise Phase4ProtocolABContrastError("config claim boundary mismatch")
    payloads = tuple(document.get("artifact_payload_files", ()))
    if payloads != ARTIFACT_PAYLOAD_FILES:
        raise Phase4ProtocolABContrastError("artifact payload order mismatch")
    cells = tuple(document.get("cell_ids", ()))
    metrics = tuple(document.get("metric_output_ids", ()))
    perturbations = tuple(document.get("perturbation_ids", ()))
    alphas = tuple(float(value) for value in document.get("alpha_grid", ()))
    if not synthetic:
        if cells != CELL_IDS or metrics != METRIC_OUTPUT_IDS or perturbations != PERTURBATION_IDS or alphas != ALPHAS:
            raise Phase4ProtocolABContrastError("real scientific grid mismatch")
        inference = document.get("inference", {})
        if inference != {
            "bootstrap_resamples": 2000, "confidence_level": 0.95,
            "holm_alpha": 0.05, "holm_slots_per_cell": 29,
            "random_seed": 20260817, "sign_flip_resamples": 100000,
        }:
            raise Phase4ProtocolABContrastError("inference contract mismatch")
        if document.get("code_authority") != _code_authority():
            raise Phase4ProtocolABContrastError("code authority mismatch")
        if document.get("environment_authority") != _environment_authority():
            raise Phase4ProtocolABContrastError("environment authority mismatch")
        trust = document.get("trust_anchor")
        if trust != {
            "config_authority_relative_path": "rpe/runner/phase4_protocol_ab_contrast_authority.py",
            "config_binds_authority": False,
            "direction": "authority_to_config_only",
        }:
            raise Phase4ProtocolABContrastError("trust anchor mismatch")
        expected = document.get("expected")
        if expected != {
            "artifact_file_count": 14, "bootstrap_result_count": 475,
            "configured_payload_count": 12, "downstream_effect_row_count": 225,
            "endpoint_status_count": 6, "holm_slot_count": 145,
            "metric_effect_row_count": 65, "paired_observation_row_count": 525720,
            "sign_flip_result_count": 145,
        }:
            raise Phase4ProtocolABContrastError("expected count contract mismatch")
        authorities = document.get("authorities")
        if not isinstance(authorities, dict) or not authorities:
            raise Phase4ProtocolABContrastError("scientific authorities are missing")
        for name, receipt in authorities.items():
            try:
                source = ROOT / str(receipt["path"])
                expected_bytes = int(receipt["bytes"])
                expected_sha = str(receipt["sha256"])
            except (KeyError, TypeError, ValueError) as error:
                raise Phase4ProtocolABContrastError(f"authority schema mismatch: {name}") from error
            if not source.is_file() or source.stat().st_size != expected_bytes or sha256_file(source) != expected_sha:
                raise Phase4ProtocolABContrastError(f"scientific authority mismatch: {name}")
    if not cells or not metrics or metrics[0] != "mse" or not perturbations or not alphas:
        raise Phase4ProtocolABContrastError("config grid is empty or lacks MSE")
    return Phase4ProtocolABContrastConfig(
        Path(path), raw_bytes, digest, MappingProxyType(document), payloads,
        cells, metrics, perturbations, alphas, synthetic,
    )


def load_phase4_protocol_ab_contrast_config(
    path: Path = DEFAULT_CONFIG,
) -> Phase4ProtocolABContrastConfig:
    return parse_phase4_protocol_ab_contrast_config(Path(path), Path(path).read_bytes())


def _observation_state(row: Mapping[str, object], *, legacy: bool, parents_complete: bool) -> str:
    if "state" not in row:
        if legacy and parents_complete:
            return "complete"
        raise Phase4ProtocolABContrastError("legacy state requires complete parents")
    state = row["state"]
    if state != "complete":
        raise Phase4ProtocolABContrastError("observation state must be complete")
    return "complete"


def validate_parent_artifact(
    path: Path, expected: Mapping[str, object]
) -> Mapping[str, object]:
    parent = Path(path)
    ledger_path = parent / "SHA256SUMS"
    if not parent.is_dir() or not ledger_path.is_file():
        raise Phase4ProtocolABContrastError("parent artifact or checksum ledger is missing")
    ledger_sha = sha256_file(ledger_path)
    if ledger_sha != str(expected.get("sha256sums_sha256", "")):
        raise Phase4ProtocolABContrastError("parent ledger identity mismatch")
    entries = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4ProtocolABContrastError("parent checksum ledger is malformed") from error
        if name in entries or len(digest) != 64:
            raise Phase4ProtocolABContrastError("parent checksum ledger is malformed")
        entries[name] = digest
    actual_inventory = sorted(item.name for item in parent.iterdir() if item.is_file())
    required_inventory = sorted(str(value) for value in expected.get("required_inventory", ()))
    if actual_inventory != required_inventory or sorted([*entries, "SHA256SUMS"]) != required_inventory:
        raise Phase4ProtocolABContrastError("parent artifact inventory mismatch")
    for name, digest in entries.items():
        if sha256_file(parent / name) != digest:
            raise Phase4ProtocolABContrastError(f"parent payload checksum mismatch: {name}")
    if parent.name != str(expected.get("directory_name", "")):
        raise Phase4ProtocolABContrastError("parent directory identity mismatch")
    manifest = json.loads((parent / "manifest.json").read_bytes())
    manifest_sha = sha256_file(parent / "manifest.json")
    if manifest_sha != str(expected.get("manifest_sha256", manifest_sha)):
        raise Phase4ProtocolABContrastError("parent manifest identity mismatch")
    manifest_run_id = str(manifest.get("run_id", manifest.get("run", "")))
    if manifest_run_id != str(expected.get("manifest_run_id", "")):
        raise Phase4ProtocolABContrastError("parent manifest run ID mismatch")
    terminal_name = str(expected.get("terminal_name", ""))
    if terminal_name not in {"complete.json", "failed.json"}:
        raise Phase4ProtocolABContrastError("parent terminal name is invalid")
    terminal = json.loads((parent / terminal_name).read_bytes())
    terminal_sha = sha256_file(parent / terminal_name)
    if terminal_sha != str(expected.get("terminal_sha256", terminal_sha)):
        raise Phase4ProtocolABContrastError("parent terminal identity mismatch")
    terminal_run_id = str(terminal.get("run_id", terminal.get("run", "")))
    if terminal_run_id != str(expected.get("terminal_run_id", "")):
        raise Phase4ProtocolABContrastError("parent terminal run ID mismatch")
    if manifest.get("status") != expected.get("status") or terminal.get("status") != expected.get("status"):
        raise Phase4ProtocolABContrastError("parent terminal status mismatch")
    expected_endpoint_state = expected.get("endpoint_state")
    if expected_endpoint_state is not None:
        manifest_state = manifest.get("endpoint_state", manifest.get("failure", {}).get("state"))
        terminal_state = terminal.get("endpoint_state", terminal.get("failure", {}).get("state"))
        if manifest_state != expected_endpoint_state or terminal_state != expected_endpoint_state:
            raise Phase4ProtocolABContrastError("parent endpoint state mismatch")
    return MappingProxyType({
        "checksum_entries": entries,
        "directory_name": parent.name,
        "manifest_run_id": manifest_run_id,
        "manifest_sha256": manifest_sha,
        "relative_path": str(expected.get("relative_path", "")),
        "sha256sums_sha256": ledger_sha,
        "status": str(terminal["status"]),
        "terminal_name": terminal_name,
        "terminal_run_id": terminal_run_id,
        "terminal_sha256": terminal_sha,
    })


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Phase4ProtocolABContrastError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4ProtocolABContrastError(f"cannot read canonical JSONL: {path}") from error
    return rows


def _alpha0_csv_row(path: Path) -> Mapping[str, str]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise Phase4ProtocolABContrastError(f"cannot read condition summary: {path}") from error
    selected = [row for row in rows if row.get("condition_id") == "alpha0"]
    if len(selected) != 1:
        raise Phase4ProtocolABContrastError("D5 alpha-zero row count mismatch")
    return MappingProxyType(selected[0])


def _parent_alignment_rows(path: Path, shot: int | None) -> Mapping[str, Mapping[str, object]]:
    rows = _read_jsonl(path / "alignment_results.jsonl")
    if shot is not None:
        rows = [row for row in rows if int(row.get("shot_count", -1)) == shot]
    if len(rows) != len(METRIC_OUTPUT_IDS):
        raise Phase4ProtocolABContrastError("parent alignment row count mismatch")
    result = {}
    for row in rows:
        metric = str(row.get("metric_output_id", ""))
        if metric in result or metric not in METRIC_OUTPUT_IDS:
            raise Phase4ProtocolABContrastError("parent alignment metric identity mismatch")
        if row.get("state", row.get("metric_state")) != "complete":
            raise Phase4ProtocolABContrastError("parent alignment state mismatch")
        comparison = row.get("comparison")
        d_ag = row.get("d_ag")
        d_acc = row.get("d_acc")
        if isinstance(comparison, dict):
            d_ag = comparison.get("d_ag")
            d_acc = comparison.get("d_acc")
        acc = row.get("acc_cross", row.get("acc"))
        ag = row.get("ag")
        values = [ag, acc] + ([] if metric == "mse" else [d_ag, d_acc])
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise Phase4ProtocolABContrastError("parent alignment numerical schema mismatch")
        result[metric] = {
            "acc_cross": float(acc), "ag": float(ag),
            "d_acc": None if metric == "mse" else float(d_acc),
            "d_ag": None if metric == "mse" else float(d_ag),
        }
    if tuple(result) != METRIC_OUTPUT_IDS:
        raise Phase4ProtocolABContrastError("parent alignment order mismatch")
    return MappingProxyType({key: MappingProxyType(value) for key, value in result.items()})


def _projection_identity(
    cell_id: str,
    rows: Sequence[Mapping[str, object]],
    *,
    cluster_field: str,
    count_field: str | None,
    metric_output_ids: Sequence[str],
    perturbation_ids: Sequence[str],
    alpha_grid: Sequence[float],
    include_metric: bool,
) -> Mapping[str, object]:
    cluster_order = {}
    for row in rows:
        cluster_order.setdefault(str(row[cluster_field]), len(cluster_order))
    metric_order = {value: index for index, value in enumerate(metric_output_ids)}
    perturbation_order = {value: index for index, value in enumerate(perturbation_ids)}
    alpha_order = {float(value).hex(): index for index, value in enumerate(alpha_grid)}
    ordered = sorted(
        rows,
        key=lambda row: (
            metric_order[str(row["metric_output_id"])],
            perturbation_order[str(row["perturbation_id"])],
            alpha_order[float(row["alpha"]).hex()],
            cluster_order[str(row[cluster_field])],
        ),
    )
    payload = bytearray()
    for row in ordered:
        item = {
            "alpha_hex": float(row["alpha"]).hex(),
            "cell_id": cell_id,
            "cluster_id": str(row[cluster_field]),
            "metric_output_id": str(row["metric_output_id"]),
            "perturbation_id": str(row["perturbation_id"]),
            "state": str(row.get("state", "complete")),
        }
        if include_metric:
            item["metric_harm_hex"] = float(row["metric_harm"]).hex()
        if count_field is not None:
            item["within_cluster_count"] = int(row[count_field])
        payload.extend(canonical_json_bytes(item))
    return MappingProxyType({"bytes": len(payload), "sha256": sha256_hex(bytes(payload))})


def validate_real_parent_authorities(
    config: Phase4ProtocolABContrastConfig,
) -> Mapping[str, Mapping[str, object]]:
    parents = config.document.get("parent_artifacts")
    if not isinstance(parents, dict) or set(parents) != {
        "d5_a", "d5_b", "d2_a", "d2_b",
        "d1_a", "d1_b", "d4_a", "d4_b",
    }:
        raise Phase4ProtocolABContrastError("parent artifact order mismatch")
    receipts = {}
    for name, expected in parents.items():
        if not isinstance(expected, dict):
            raise Phase4ProtocolABContrastError(f"parent artifact schema mismatch: {name}")
        receipts[name] = validate_parent_artifact(ROOT / str(expected["relative_path"]), expected)
    return MappingProxyType(receipts)


def adapt_protocol_observation_rows(
    *,
    cell_id: str,
    rows_a: Sequence[Mapping[str, object]],
    rows_b: Sequence[Mapping[str, object]],
    cluster_field: str,
    cluster_kind: str,
    count_field: str | None = None,
    metric_output_ids: Sequence[str],
    perturbation_ids: Sequence[str],
    alpha_grid: Sequence[float],
    parents_complete: bool,
    legacy_missing_state: bool,
    use_protocol_a_metric: bool,
) -> ProtocolABCellInput:
    if not parents_complete:
        if legacy_missing_state:
            raise Phase4ProtocolABContrastError("legacy state requires complete parents")
        raise Phase4ProtocolABContrastError("parents must be complete")
    metrics = tuple(metric_output_ids)
    perturbations = tuple(perturbation_ids)
    alphas = tuple(float(value) for value in alpha_grid)
    metric_index = {value: index for index, value in enumerate(metrics)}
    perturbation_index = {value: index for index, value in enumerate(perturbations)}
    alpha_index = {float(value).hex(): index for index, value in enumerate(alphas)}
    cluster_ids = tuple(dict.fromkeys(str(row[cluster_field]) for row in rows_a))
    cluster_index = {value: index for index, value in enumerate(cluster_ids)}
    expected = len(metrics) * len(perturbations) * len(alphas) * len(cluster_ids)
    if len(rows_a) != expected or len(rows_b) != expected:
        raise Phase4ProtocolABContrastError("observation grid row count mismatch")

    def keyed(rows: Sequence[Mapping[str, object]], side: str):
        result = {}
        for row in rows:
            _observation_state(row, legacy=legacy_missing_state, parents_complete=parents_complete)
            try:
                key = (
                    str(row["metric_output_id"]), str(row["perturbation_id"]),
                    float(row["alpha"]).hex(), str(row[cluster_field]),
                )
                metric_value = float(row["metric_harm"])
                downstream_value = float(row["downstream_harm"])
            except (KeyError, TypeError, ValueError) as error:
                raise Phase4ProtocolABContrastError(f"{side} observation schema mismatch") from error
            if key in result:
                raise Phase4ProtocolABContrastError(f"{side} duplicate observation key")
            if key[0] not in metric_index or key[1] not in perturbation_index or key[2] not in alpha_index or key[3] not in cluster_index:
                raise Phase4ProtocolABContrastError(f"{side} observation outside frozen grid")
            if not math.isfinite(metric_value) or not math.isfinite(downstream_value):
                raise Phase4ProtocolABContrastError(f"{side} observation must be finite")
            count_value = None
            if count_field is not None:
                try:
                    count_value = int(row[count_field])
                except (KeyError, TypeError, ValueError) as error:
                    raise Phase4ProtocolABContrastError(f"{side} within-cluster count is invalid") from error
                if count_value <= 0:
                    raise Phase4ProtocolABContrastError(f"{side} within-cluster count is invalid")
            result[key] = (metric_value, downstream_value, count_value)
        return result

    a_map = keyed(rows_a, "protocol A")
    b_map = keyed(rows_b, "protocol B")
    if set(a_map) != set(b_map):
        raise Phase4ProtocolABContrastError("A/B observation keys must be identical")
    metric = np.empty((len(metrics), len(cluster_ids), len(perturbations), len(alphas)), dtype=np.float64)
    downstream_a = np.empty((len(cluster_ids), len(perturbations), len(alphas)), dtype=np.float64)
    downstream_b = np.empty_like(downstream_a)
    mismatch_count = 0
    max_abs = 0.0
    seen_downstream_a = {}
    seen_downstream_b = {}
    for metric_id in metrics:
        for perturbation in perturbations:
            for alpha in alphas:
                alpha_key = float(alpha).hex()
                for cluster_id in cluster_ids:
                    key = (metric_id, perturbation, alpha_key, cluster_id)
                    av, ay, acount = a_map[key]
                    bv, by, bcount = b_map[key]
                    if acount != bcount:
                        raise Phase4ProtocolABContrastError("A/B within-cluster count mismatch")
                    mi = metric_index[metric_id]
                    ci = cluster_index[cluster_id]
                    pi = perturbation_index[perturbation]
                    ai = alpha_index[alpha_key]
                    metric[mi, ci, pi, ai] = av if use_protocol_a_metric else bv
                    condition_key = (cluster_id, perturbation, alpha_key)
                    for seen, value, side in ((seen_downstream_a, ay, "A"), (seen_downstream_b, by, "B")):
                        prior = seen.setdefault(condition_key, value)
                        if prior.hex() != value.hex():
                            raise Phase4ProtocolABContrastError(f"protocol {side} downstream copies differ")
                    downstream_a[ci, pi, ai] = ay
                    downstream_b[ci, pi, ai] = by
                    if av.hex() != bv.hex():
                        mismatch_count += 1
                        max_abs = max(max_abs, abs(av - bv))
    return ProtocolABCellInput(
        cell_id=cell_id,
        cluster_kind=cluster_kind,
        cluster_ids=cluster_ids,
        metric_output_ids=metrics,
        perturbation_ids=perturbations,
        alpha_grid=alphas,
        metric_harm=metric,
        downstream_a=downstream_a,
        downstream_b=downstream_b,
        metric_reconciliation={
            "mismatch_count": mismatch_count,
            "max_abs_difference": max_abs,
            "metric_authority": "protocol_a" if use_protocol_a_metric else "protocol_b",
        },
    )


def _observations(cell: ProtocolABCellInput, metric_index: int, downstream: np.ndarray):
    return tuple(
        AlignmentObservation(
            cluster_id=cluster_id,
            perturbation_id=perturbation,
            alpha=alpha,
            metric_harm=float(cell.metric_harm[metric_index, ci, pi, ai]),
            downstream_harm=float(downstream[ci, pi, ai]),
        )
        for ci, cluster_id in enumerate(cell.cluster_ids)
        for pi, perturbation in enumerate(cell.perturbation_ids)
        for ai, alpha in enumerate(cell.alpha_grid)
    )


@dataclass(frozen=True)
class _AGMetricCache:
    pooled_inverse: np.ndarray
    pooled_unique_count: int
    separate_inverse: tuple[np.ndarray, ...]
    separate_unique_counts: tuple[int, ...]


def _prepare_ag_cache(metric: np.ndarray) -> _AGMetricCache:
    pooled_inverse = np.unique(metric.reshape(-1), return_inverse=True)[1]
    separate_inverse = tuple(
        np.unique(metric[:, index, :].reshape(-1), return_inverse=True)[1]
        for index in range(metric.shape[1])
    )
    pooled_inverse.setflags(write=False)
    for value in separate_inverse:
        value.setflags(write=False)
    return _AGMetricCache(
        pooled_inverse=pooled_inverse,
        pooled_unique_count=int(pooled_inverse.max()) + 1,
        separate_inverse=separate_inverse,
        separate_unique_counts=tuple(int(value.max()) + 1 for value in separate_inverse),
    )


def _weighted_isotonic_sse(
    inverse: np.ndarray,
    unique_count: int,
    downstream: np.ndarray,
    weights: np.ndarray,
) -> float:
    summed_weights = np.bincount(inverse, weights=weights, minlength=unique_count)
    summed_y = np.bincount(inverse, weights=weights * downstream, minlength=unique_count)
    summed_y_squared = np.bincount(
        inverse, weights=weights * downstream * downstream, minlength=unique_count
    )
    positive = summed_weights > 0
    fitted = np.asarray(
        isotonic_regression(
            y=summed_y[positive] / summed_weights[positive],
            weights=summed_weights[positive],
            increasing=True,
        ).x,
        dtype=np.float64,
    )
    return float(
        np.sum(
            summed_y_squared[positive]
            - 2.0 * fitted * summed_y[positive]
            + fitted * fitted * summed_weights[positive]
        )
    )


def _weighted_ag(
    metric: np.ndarray,
    downstream: np.ndarray,
    cluster_weights: np.ndarray,
    cache: _AGMetricCache | None = None,
) -> float:
    clusters, perturbations, alphas = downstream.shape
    prepared = _prepare_ag_cache(metric) if cache is None else cache
    observation_weights = np.repeat(cluster_weights, perturbations * alphas).astype(np.float64)
    y = downstream.reshape(-1)
    total = float(np.sum(observation_weights))
    mean = float(np.sum(observation_weights * y) / total)
    sst = float(np.sum(observation_weights * (y - mean) ** 2))
    if not math.isfinite(sst) or sst <= 0.0:
        raise Phase4ProtocolABContrastError("bootstrap downstream SST is not positive")
    pooled_sse = _weighted_isotonic_sse(
        prepared.pooled_inverse, prepared.pooled_unique_count, y, observation_weights
    )
    separate_sse = 0.0
    for pi in range(perturbations):
        dy = downstream[:, pi, :].reshape(-1)
        pw = np.repeat(cluster_weights, alphas).astype(np.float64)
        separate_sse += _weighted_isotonic_sse(
            prepared.separate_inverse[pi],
            prepared.separate_unique_counts[pi],
            dy,
            pw,
        )
    raw = (pooled_sse - separate_sse) / sst
    if raw < -1e-12:
        raise Phase4ProtocolABContrastError("bootstrap alignment gap is negative")
    return 0.0 if raw < 0.0 else float(raw)


def _cross_pair_indices(perturbation_count: int, alpha_count: int):
    left, right = [], []
    labels = [pi for pi in range(perturbation_count) for _ in range(alpha_count)]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                left.append(i)
                right.append(j)
    return np.asarray(left), np.asarray(right)


def _cluster_acc(metric: np.ndarray, downstream: np.ndarray) -> np.ndarray:
    clusters, perturbations, alphas = downstream.shape
    left, right = _cross_pair_indices(perturbations, alphas)
    mx = metric.reshape(clusters, -1)
    dy = downstream.reshape(clusters, -1)
    md = mx[:, left] - mx[:, right]
    yd = dy[:, left] - dy[:, right]
    ties = (md == 0.0) | (yd == 0.0)
    agree = (~ties) & ((md > 0.0) == (yd > 0.0))
    return np.mean(agree.astype(np.float64) + 0.5 * ties.astype(np.float64), axis=1)


def _interval(values: np.ndarray, confidence: float = 0.95) -> list[float]:
    q = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [q, 1.0 - q], method="linear")
    return [float(lo), float(hi)]


def build_protocol_ab_holm(
    cell_id: str, slots: Sequence[tuple[str, float]], *, alpha: float = 0.05
) -> list[dict[str, object]]:
    if len({slot for slot, _ in slots}) != len(slots):
        raise Phase4ProtocolABContrastError("Holm slot IDs must be unique")
    results = {row.hypothesis_id: row for row in holm_step_down(dict(slots), alpha=alpha)}
    return [
        {
            "adjusted_p_value": results[slot].adjusted_p_value,
            "alpha": results[slot].alpha,
            "cell_id": cell_id,
            "family_id": f"{cell_id}:protocol_ab",
            "family_size": results[slot].family_size,
            "rank": results[slot].rank,
            "raw_p_value": results[slot].raw_p_value,
            "rejected": results[slot].rejected,
            "slot_id": slot,
            "slot_order": index,
        }
        for index, (slot, _) in enumerate(slots)
    ]


def compute_protocol_ab_cell(
    cell: ProtocolABCellInput,
    *,
    bootstrap_resamples: int = 2000,
    sign_flip_resamples: int = 100000,
    random_seed: int = 20260817,
) -> Mapping[str, list[dict[str, object]]]:
    if bootstrap_resamples <= 0 or sign_flip_resamples <= 0:
        raise Phase4ProtocolABContrastError("resample counts must be positive")
    metric_count, cluster_count, perturbation_count, alpha_count = cell.metric_harm.shape
    gap = cell.downstream_a - cell.downstream_b
    point_ag_a = np.empty(metric_count)
    point_ag_b = np.empty(metric_count)
    point_acc_a = np.empty(metric_count)
    point_acc_b = np.empty(metric_count)
    ag_contrib_a = np.empty((metric_count, cluster_count))
    ag_contrib_b = np.empty((metric_count, cluster_count))
    acc_contrib_a = np.empty((metric_count, cluster_count))
    acc_contrib_b = np.empty((metric_count, cluster_count))
    for mi in range(metric_count):
        a_gap = alignment_gap(_observations(cell, mi, cell.downstream_a))
        b_gap = alignment_gap(_observations(cell, mi, cell.downstream_b))
        a_acc = cross_perturbation_accuracy(_observations(cell, mi, cell.downstream_a))
        b_acc = cross_perturbation_accuracy(_observations(cell, mi, cell.downstream_b))
        point_ag_a[mi], point_ag_b[mi] = a_gap.alignment_gap, b_gap.alignment_gap
        point_acc_a[mi], point_acc_b[mi] = a_acc.accuracy, b_acc.accuracy
        a_ag_map = {row.cluster_id: row.value for row in a_gap.cluster_contributions}
        b_ag_map = {row.cluster_id: row.value for row in b_gap.cluster_contributions}
        a_acc_map = {row.cluster_id: row.accuracy for row in a_acc.cluster_contributions}
        b_acc_map = {row.cluster_id: row.accuracy for row in b_acc.cluster_contributions}
        ag_contrib_a[mi] = [a_ag_map[value] for value in cell.cluster_ids]
        ag_contrib_b[mi] = [b_ag_map[value] for value in cell.cluster_ids]
        acc_contrib_a[mi] = [a_acc_map[value] for value in cell.cluster_ids]
        acc_contrib_b[mi] = [b_acc_map[value] for value in cell.cluster_ids]
    delta_ag = point_ag_b - point_ag_a
    delta_acc = point_acc_b - point_acc_a
    parent_d_ag_a = point_ag_a[0] - point_ag_a
    parent_d_ag_b = point_ag_b[0] - point_ag_b
    parent_d_acc_a = point_acc_a - point_acc_a[0]
    parent_d_acc_b = point_acc_b - point_acc_b[0]
    interaction_ag = parent_d_ag_b - parent_d_ag_a
    interaction_acc = parent_d_acc_b - parent_d_acc_a

    generator = np.random.Generator(np.random.PCG64(random_seed))
    draws = generator.integers(0, cluster_count, size=(bootstrap_resamples, cluster_count))
    weights = np.zeros((bootstrap_resamples, cluster_count), dtype=np.int64)
    np.add.at(weights, (np.repeat(np.arange(bootstrap_resamples), cluster_count), draws.reshape(-1)), 1)
    condition_boot = np.empty((bootstrap_resamples, perturbation_count, alpha_count))
    integrated_boot = np.empty((bootstrap_resamples, perturbation_count))
    ag_a_boot = np.empty((bootstrap_resamples, metric_count))
    ag_b_boot = np.empty_like(ag_a_boot)
    acc_a_clusters = np.asarray([_cluster_acc(cell.metric_harm[mi], cell.downstream_a) for mi in range(metric_count)])
    acc_b_clusters = np.asarray([_cluster_acc(cell.metric_harm[mi], cell.downstream_b) for mi in range(metric_count)])
    acc_a_boot = weights @ acc_a_clusters.T / cluster_count
    acc_b_boot = weights @ acc_b_clusters.T / cluster_count
    ag_caches = tuple(_prepare_ag_cache(cell.metric_harm[index]) for index in range(metric_count))
    for bi, current in enumerate(weights):
        condition_boot[bi] = np.tensordot(current, gap, axes=(0, 0)) / cluster_count
        integrated_boot[bi] = np.mean(condition_boot[bi], axis=1)
        for mi in range(metric_count):
            ag_a_boot[bi, mi] = _weighted_ag(
                cell.metric_harm[mi], cell.downstream_a, current, ag_caches[mi]
            )
            ag_b_boot[bi, mi] = _weighted_ag(
                cell.metric_harm[mi], cell.downstream_b, current, ag_caches[mi]
            )
    delta_ag_boot = ag_b_boot - ag_a_boot
    delta_acc_boot = acc_b_boot - acc_a_boot
    interaction_ag_boot = (ag_b_boot[:, [0]] - ag_b_boot) - (ag_a_boot[:, [0]] - ag_a_boot)
    interaction_acc_boot = (acc_b_boot - acc_b_boot[:, [0]]) - (acc_a_boot - acc_a_boot[:, [0]])

    downstream_rows = []
    bootstrap_rows = []
    for pi, perturbation in enumerate(cell.perturbation_ids):
        for ai, alpha in enumerate(cell.alpha_grid):
            value = float(np.mean(gap[:, pi, ai]))
            interval = _interval(condition_boot[:, pi, ai])
            downstream_rows.append({
                "alpha": alpha, "cell_id": cell.cell_id, "cluster_count": cluster_count,
                "gap": value, "interval": interval,
                "mean_harm_a": float(np.mean(cell.downstream_a[:, pi, ai])),
                "mean_harm_b": float(np.mean(cell.downstream_b[:, pi, ai])),
                "perturbation_id": perturbation, "state": "complete",
                "summary_type": "alpha_specific",
            })
            bootstrap_rows.append({
                "cell_id": cell.cell_id, "confidence_level": 0.95,
                "interval": interval, "metric_output_id": None,
                "perturbation_id": perturbation, "alpha": alpha,
                "random_seed": random_seed, "resamples": bootstrap_resamples,
                "state": "complete", "statistic": "gap",
            })
        integrated = float(np.mean(gap[:, pi, :]))
        interval = _interval(integrated_boot[:, pi])
        downstream_rows.append({
            "alpha": None, "cell_id": cell.cell_id, "cluster_count": cluster_count,
            "gap": integrated, "interval": interval,
            "mean_harm_a": float(np.mean(cell.downstream_a[:, pi, :])),
            "mean_harm_b": float(np.mean(cell.downstream_b[:, pi, :])),
            "perturbation_id": perturbation, "state": "complete",
            "summary_type": "integrated",
        })
        bootstrap_rows.append({
            "cell_id": cell.cell_id, "confidence_level": 0.95,
            "interval": interval, "metric_output_id": None,
            "perturbation_id": perturbation, "alpha": None,
            "random_seed": random_seed, "resamples": bootstrap_resamples,
            "state": "complete", "statistic": "g",
        })

    metric_rows = []
    for mi, metric_id in enumerate(cell.metric_output_ids):
        parent_a = cell.parent_alignment_a.get(metric_id, {})
        parent_b = cell.parent_alignment_b.get(metric_id, {})
        parent_acc_a = parent_a.get("acc_cross", parent_a.get("acc"))
        parent_acc_b = parent_b.get("acc_cross", parent_b.get("acc"))
        parent_ag_a = parent_a.get("ag")
        parent_ag_b = parent_b.get("ag")
        parent_ag_a_difference = None if parent_ag_a is None else float(point_ag_a[mi] - float(parent_ag_a))
        parent_ag_b_difference = None if parent_ag_b is None else float(point_ag_b[mi] - float(parent_ag_b))
        parent_acc_a_difference = None if parent_acc_a is None else float(point_acc_a[mi] - float(parent_acc_a))
        parent_acc_b_difference = None if parent_acc_b is None else float(point_acc_b[mi] - float(parent_acc_b))
        row = {
            "acc_a": float(point_acc_a[mi]), "acc_b": float(point_acc_b[mi]),
            "ag_a": float(point_ag_a[mi]), "ag_b": float(point_ag_b[mi]),
            "cell_id": cell.cell_id, "delta_acc": float(delta_acc[mi]),
            "delta_acc_interval": _interval(delta_acc_boot[:, mi]),
            "delta_ag": float(delta_ag[mi]), "delta_ag_interval": _interval(delta_ag_boot[:, mi]),
            "i_acc": None if mi == 0 else float(interaction_acc[mi]),
            "i_acc_interval": None if mi == 0 else _interval(interaction_acc_boot[:, mi]),
            "i_ag": None if mi == 0 else float(interaction_ag[mi]),
            "i_ag_interval": None if mi == 0 else _interval(interaction_ag_boot[:, mi]),
            "metric_output_id": metric_id,
            "canonical_d_acc_a": None if mi == 0 else float(parent_d_acc_a[mi]),
            "canonical_d_acc_b": None if mi == 0 else float(parent_d_acc_b[mi]),
            "canonical_d_ag_a": None if mi == 0 else float(parent_d_ag_a[mi]),
            "canonical_d_ag_b": None if mi == 0 else float(parent_d_ag_b[mi]),
            "parent_acc_a": None if parent_acc_a is None else float(parent_acc_a),
            "parent_acc_a_difference": parent_acc_a_difference,
            "parent_acc_b": None if parent_acc_b is None else float(parent_acc_b),
            "parent_acc_b_difference": parent_acc_b_difference,
            "parent_ag_a": None if parent_ag_a is None else float(parent_ag_a),
            "parent_ag_a_difference": parent_ag_a_difference,
            "parent_ag_b": None if parent_ag_b is None else float(parent_ag_b),
            "parent_ag_b_difference": parent_ag_b_difference,
            "parent_a_exact": None if parent_ag_a is None else parent_ag_a_difference == 0.0 and parent_acc_a_difference == 0.0,
            "parent_b_exact": None if parent_ag_b is None else parent_ag_b_difference == 0.0 and parent_acc_b_difference == 0.0,
            "parent_d_acc_a": None if mi == 0 else (None if not parent_a else float(parent_a.get("d_acc"))),
            "parent_d_acc_b": None if mi == 0 else (None if not parent_b else float(parent_b.get("d_acc"))),
            "parent_d_ag_a": None if mi == 0 else (None if not parent_a else float(parent_a.get("d_ag"))),
            "parent_d_ag_b": None if mi == 0 else (None if not parent_b else float(parent_b.get("d_ag"))),
            "state": "complete",
        }
        metric_rows.append(row)
    for statistic, values_by_metric in (
        ("delta_ag", delta_ag_boot),
        ("delta_acc", delta_acc_boot),
    ):
        for mi, metric_id in enumerate(cell.metric_output_ids):
            bootstrap_rows.append({
                "alpha": None, "cell_id": cell.cell_id, "confidence_level": 0.95,
                "interval": _interval(values_by_metric[:, mi]), "metric_output_id": metric_id,
                "perturbation_id": None, "random_seed": random_seed,
                "resamples": bootstrap_resamples, "state": "complete", "statistic": statistic,
            })
    for statistic, values_by_metric in (
        ("i_ag", interaction_ag_boot),
        ("i_acc", interaction_acc_boot),
    ):
        for mi, metric_id in enumerate(cell.metric_output_ids[1:], start=1):
                bootstrap_rows.append({
                    "alpha": None, "cell_id": cell.cell_id, "confidence_level": 0.95,
                    "interval": _interval(values_by_metric[:, mi]), "metric_output_id": metric_id,
                    "perturbation_id": None, "random_seed": random_seed,
                    "resamples": bootstrap_resamples, "state": "complete", "statistic": statistic,
                })

    sign_flip_rows = []
    slots = []
    for pi, perturbation in enumerate(cell.perturbation_ids):
        slot = f"downstream:{perturbation}"
        result = paired_contribution_sign_flip(
            tuple(float(value) for value in np.mean(gap[:, pi, :], axis=1)),
            aggregation="mean", resamples=sign_flip_resamples, random_seed=random_seed,
        )
        direction = "attenuated_by_b" if result.observed > 0 else "amplified_by_b" if result.observed < 0 else "zero"
        sign_flip_rows.append({
            "aggregation": "mean", "cell_id": cell.cell_id, "direction": direction,
            "extreme_resamples": result.extreme_resamples, "metric_output_id": None,
            "observed": result.observed, "perturbation_id": perturbation,
            "p_value": result.p_value, "random_seed": random_seed,
            "resamples": sign_flip_resamples, "slot_id": slot, "state": "tested",
        })
        slots.append((slot, result.p_value))
    for statistic in ("i_ag", "i_acc"):
        for mi, metric_id in enumerate(cell.metric_output_ids[1:], start=1):
            if statistic == "i_ag":
                values = ag_contrib_b[0] - ag_contrib_b[mi] - ag_contrib_a[0] + ag_contrib_a[mi]
                aggregation = "sum"
                observed = interaction_ag[mi]
            else:
                values = acc_contrib_b[mi] - acc_contrib_b[0] - acc_contrib_a[mi] + acc_contrib_a[0]
                aggregation = "mean"
                observed = interaction_acc[mi]
            result = paired_contribution_sign_flip(
                tuple(float(value) for value in values), aggregation=aggregation,
                resamples=sign_flip_resamples, random_seed=random_seed,
            )
            if not math.isclose(result.observed, float(observed), rel_tol=0.0, abs_tol=1e-12):
                raise Phase4ProtocolABContrastError("interaction contributions do not match point estimate")
            slot = f"{statistic}:{metric_id}"
            direction = "candidate_advantage_strengthened_under_b" if observed > 0 else "candidate_advantage_weakened_under_b" if observed < 0 else "zero"
            sign_flip_rows.append({
                "aggregation": aggregation, "cell_id": cell.cell_id, "direction": direction,
                "extreme_resamples": result.extreme_resamples, "metric_output_id": metric_id,
                "observed": float(observed), "perturbation_id": None,
                "p_value": result.p_value, "random_seed": random_seed,
                "resamples": sign_flip_resamples, "slot_id": slot, "state": "tested",
            })
            slots.append((slot, result.p_value))
    holm_rows = build_protocol_ab_holm(cell.cell_id, slots)
    sign_by_slot = {row["slot_id"]: row for row in sign_flip_rows}
    for row in holm_rows:
        row["direction"] = sign_by_slot[row["slot_id"]]["direction"]
        row["directional_result"] = (
            sign_by_slot[row["slot_id"]]["direction"] if row["rejected"] else "not_rejected"
        )
        row["state"] = "tested"
    if cell.parent_alignment_a:
        if not all(row["parent_a_exact"] is True for row in metric_rows):
            raise Phase4ProtocolABContrastError("Protocol-A parent alignment recomputation mismatch")
        if cell.cell_id != "d4_full_domain_core" and not all(
            row["parent_b_exact"] is True for row in metric_rows
        ):
            raise Phase4ProtocolABContrastError("Protocol-B parent alignment recomputation mismatch")
    return MappingProxyType({
        "bootstrap_rows": bootstrap_rows,
        "downstream_rows": downstream_rows,
        "holm_rows": holm_rows,
        "metric_rows": metric_rows,
        "sign_flip_rows": sign_flip_rows,
    })


def make_synthetic_protocol_ab_contrast_config() -> Phase4ProtocolABContrastConfig:
    document = {
        "alpha_grid": [0.1, 0.2],
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "cell_ids": list(CELL_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
        "experiment_id": EXPERIMENT_ID,
        "metric_output_ids": ["mse", "candidate"],
        "parent_artifacts": {"synthetic": True},
        "perturbation_ids": ["p01", "p02"],
        "schema_version": SCHEMA_VERSION,
        "synthetic_fixture": True,
    }
    raw = canonical_json_bytes(document)
    return parse_phase4_protocol_ab_contrast_config(Path("synthetic-config.json"), raw, require_frozen_identity=False)


def _hand_cell(cell_id: str) -> ProtocolABCellInput:
    metric = np.asarray([
        [[[0., 1.], [0., 1.]], [[0., 1.], [0., 1.]]],
        [[[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]]],
    ])
    a = np.asarray([[[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]]])
    b = np.asarray([[[0., 1.], [0., 1.]], [[0., 1.], [0., 1.]]])
    return ProtocolABCellInput(
        cell_id, "class_label", ("c1", "c2"), ("mse", "candidate"),
        ("p01", "p02"), (0.1, 0.2), metric, a, b,
        {"mismatch_count": 0, "max_abs_difference": 0.0, "metric_authority": "protocol_a"},
    )


def make_synthetic_protocol_ab_contrast_inputs() -> Phase4ProtocolABContrastInputs:
    cells = tuple(_hand_cell(cell_id) for cell_id in EVALUABLE_CELL_IDS)
    statuses = tuple(
        {
            "cell_id": cell_id,
            "cluster_count": 0 if cell_id == "d1_full_domain_core" else 2,
            "cluster_kind": "class_label",
            "paired_row_count": 0 if cell_id == "d1_full_domain_core" else 16,
            "reason": "failed_alpha0_equivalence" if cell_id == "d1_full_domain_core" else None,
            "state": "not_evaluable_failed_alpha0_equivalence" if cell_id == "d1_full_domain_core" else "evaluable",
        }
        for cell_id in CELL_IDS
    )
    return Phase4ProtocolABContrastInputs(
        cells=cells,
        endpoint_status_rows=statuses,
        authority_bridge=MappingProxyType({"parents": {"synthetic": True}, "state": "complete"}),
        preflight=MappingProxyType({"paired_row_count": 80, "state": "complete", "synthetic_fixture": True}),
    )


def _csv_downstream(rows: Sequence[Mapping[str, object]]) -> bytes:
    projected = []
    for row in rows:
        interval = row["interval"]
        projected.append({
            "cell_id": row["cell_id"], "perturbation_id": row["perturbation_id"],
            "summary_type": row["summary_type"], "alpha": row["alpha"],
            "mean_harm_a": row["mean_harm_a"], "mean_harm_b": row["mean_harm_b"],
            "gap": row["gap"], "interval_lower": interval[0], "interval_upper": interval[1],
            "cluster_count": row["cluster_count"], "state": row["state"],
        })
    return csv_bytes(projected, DOWNSTREAM_CSV_FIELDS)


def _csv_metrics(rows: Sequence[Mapping[str, object]]) -> bytes:
    projected = []
    for row in rows:
        item = dict(row)
        for name in ("delta_ag", "delta_acc", "i_ag", "i_acc"):
            interval = item.pop(f"{name}_interval")
            item[f"{name}_lower"] = None if interval is None else interval[0]
            item[f"{name}_upper"] = None if interval is None else interval[1]
        projected.append(item)
    return csv_bytes(projected, METRIC_CSV_FIELDS)


def _closed_cell_projection(
    *,
    cell_id: str,
    cluster_count: int,
    metric_output_ids: Sequence[str],
    perturbation_ids: Sequence[str],
    alpha_grid: Sequence[float],
    bootstrap_resamples: int,
    sign_flip_resamples: int,
    random_seed: int,
    state: str,
    reason: str,
) -> Mapping[str, list[dict[str, object]]]:
    downstream_rows = []
    bootstrap_rows = []
    for perturbation_id in perturbation_ids:
        for alpha in alpha_grid:
            downstream_rows.append({
                "alpha": float(alpha), "cell_id": cell_id,
                "cluster_count": cluster_count, "gap": None,
                "interval": [None, None], "mean_harm_a": None,
                "mean_harm_b": None, "perturbation_id": perturbation_id,
                "reason": reason, "state": state,
                "summary_type": "alpha_specific",
            })
            bootstrap_rows.append({
                "alpha": float(alpha), "cell_id": cell_id,
                "confidence_level": 0.95, "interval": [None, None],
                "metric_output_id": None, "perturbation_id": perturbation_id,
                "random_seed": random_seed, "reason": reason,
                "resamples": bootstrap_resamples, "state": state,
                "statistic": "gap",
            })
        downstream_rows.append({
            "alpha": None, "cell_id": cell_id, "cluster_count": cluster_count,
            "gap": None, "interval": [None, None], "mean_harm_a": None,
            "mean_harm_b": None, "perturbation_id": perturbation_id,
            "reason": reason, "state": state, "summary_type": "integrated",
        })
        bootstrap_rows.append({
            "alpha": None, "cell_id": cell_id, "confidence_level": 0.95,
            "interval": [None, None], "metric_output_id": None,
            "perturbation_id": perturbation_id, "random_seed": random_seed,
            "reason": reason, "resamples": bootstrap_resamples,
            "state": state, "statistic": "g",
        })
    metric_rows = []
    for metric_output_id in metric_output_ids:
        metric_rows.append({
            "acc_a": None, "acc_b": None, "ag_a": None, "ag_b": None,
            "canonical_d_acc_a": None, "canonical_d_acc_b": None,
            "canonical_d_ag_a": None, "canonical_d_ag_b": None,
            "cell_id": cell_id, "delta_acc": None,
            "delta_acc_interval": [None, None], "delta_ag": None,
            "delta_ag_interval": [None, None], "i_acc": None,
            "i_acc_interval": None if metric_output_id == "mse" else [None, None],
            "i_ag": None,
            "i_ag_interval": None if metric_output_id == "mse" else [None, None],
            "metric_output_id": metric_output_id, "parent_a_exact": None,
            "parent_acc_a": None, "parent_acc_a_difference": None,
            "parent_acc_b": None, "parent_acc_b_difference": None,
            "parent_ag_a": None, "parent_ag_a_difference": None,
            "parent_ag_b": None, "parent_ag_b_difference": None,
            "parent_b_exact": None, "parent_d_acc_a": None,
            "parent_d_acc_b": None, "parent_d_ag_a": None,
            "parent_d_ag_b": None, "reason": reason, "state": state,
        })
    for statistic in ("delta_ag", "delta_acc"):
        for metric_output_id in metric_output_ids:
            bootstrap_rows.append({
                "alpha": None, "cell_id": cell_id, "confidence_level": 0.95,
                "interval": [None, None], "metric_output_id": metric_output_id,
                "perturbation_id": None, "random_seed": random_seed,
                "reason": reason, "resamples": bootstrap_resamples,
                "state": state, "statistic": statistic,
            })
    for statistic in ("i_ag", "i_acc"):
        for metric_output_id in metric_output_ids[1:]:
            bootstrap_rows.append({
                "alpha": None, "cell_id": cell_id, "confidence_level": 0.95,
                "interval": [None, None], "metric_output_id": metric_output_id,
                "perturbation_id": None, "random_seed": random_seed,
                "reason": reason, "resamples": bootstrap_resamples,
                "state": state, "statistic": statistic,
            })
    slots = [
        *(f"downstream:{value}" for value in perturbation_ids),
        *(f"i_ag:{value}" for value in metric_output_ids[1:]),
        *(f"i_acc:{value}" for value in metric_output_ids[1:]),
    ]
    sign_flip_rows = [{
        "aggregation": "mean" if slot.startswith(("downstream:", "i_acc:")) else "sum",
        "cell_id": cell_id, "direction": None, "extreme_resamples": None,
        "metric_output_id": None if slot.startswith("downstream:") else slot.split(":", 1)[1],
        "observed": None,
        "perturbation_id": slot.split(":", 1)[1] if slot.startswith("downstream:") else None,
        "p_value": 1.0, "random_seed": random_seed, "reason": reason,
        "resamples": sign_flip_resamples, "slot_id": slot, "state": state,
    } for slot in slots]
    holm_rows = build_protocol_ab_holm(cell_id, [(slot, 1.0) for slot in slots])
    for row in holm_rows:
        row.update({"direction": None, "directional_result": "not_rejected", "reason": reason, "state": state})
    return MappingProxyType({
        "bootstrap_rows": bootstrap_rows, "downstream_rows": downstream_rows,
        "holm_rows": holm_rows, "metric_rows": metric_rows,
        "sign_flip_rows": sign_flip_rows,
    })


def build_phase4_protocol_ab_contrast_from_inputs(
    output_root: Path,
    *,
    inputs: Phase4ProtocolABContrastInputs,
    config: Phase4ProtocolABContrastConfig,
    worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
    _precomputed_identity: tuple[str, Mapping[str, object], Mapping[str, object]] | None = None,
) -> Phase4ProtocolABContrastSummary:
    if worker_count < 1:
        raise Phase4ProtocolABContrastError("worker_count must be positive")
    if (bootstrap_resamples is not None or sign_flip_resamples is not None) and not config.synthetic_fixture:
        raise Phase4ProtocolABContrastError("inference overrides are synthetic-test only")
    bootstrap_count = bootstrap_resamples or int(config.document["inference"]["bootstrap_resamples"])
    sign_count = sign_flip_resamples or int(config.document["inference"]["sign_flip_resamples"])
    seed = int(config.document.get("inference", {}).get("random_seed", 20260817))
    if tuple(cell.cell_id for cell in inputs.cells) != tuple(value for value in config.cell_ids if value != "d1_full_domain_core"):
        raise Phase4ProtocolABContrastError("input cell order mismatch")
    parent_identity = dict(config.document.get("parent_artifacts", {}))
    if _precomputed_identity is None:
        code = {} if config.synthetic_fixture else _code_authority()
        environment = _environment_authority()
        run_id = stable_run_id(
            config_sha256=config.sha256,
            parent_identity_projection=parent_identity,
            code_authority=code,
            environment_authority=environment,
        )
    else:
        run_id, code_raw, environment_raw = _precomputed_identity
        code = dict(code_raw)
        environment = dict(environment_raw)
        expected_run_id = stable_run_id(
            config_sha256=config.sha256,
            parent_identity_projection=parent_identity,
            code_authority=code,
            environment_authority=environment,
        )
        if run_id != expected_run_id:
            raise Phase4ProtocolABContrastError("precomputed run identity mismatch")
    run_path = Path(output_root) / run_id
    if run_path.exists():
        raise Phase4ProtocolABContrastError("append-only run path already exists")
    projections = []
    failure = None
    failed_cell_id = None
    for cell_index, cell in enumerate(inputs.cells):
        if failure is None:
            try:
                projection = compute_protocol_ab_cell(
                    cell, bootstrap_resamples=bootstrap_count,
                    sign_flip_resamples=sign_count, random_seed=seed,
                )
            except (AlignmentValidationError, Phase4ProtocolABContrastError) as error:
                message = str(error)
                failure_state = (
                    "failed_bootstrap_or_sign_flip"
                    if "bootstrap" in message or "sign flip" in message
                    else "failed_alignment_recomputation"
                )
                failure = {
                    "cell_id": cell.cell_id, "error": message,
                    "error_type": type(error).__name__, "state": failure_state,
                }
                failed_cell_id = cell.cell_id
                projection = _closed_cell_projection(
                    cell_id=cell.cell_id, cluster_count=len(cell.cluster_ids),
                    metric_output_ids=config.metric_output_ids,
                    perturbation_ids=config.perturbation_ids,
                    alpha_grid=config.alpha_grid, bootstrap_resamples=bootstrap_count,
                    sign_flip_resamples=sign_count, random_seed=seed,
                    state=failure_state, reason=message,
                )
        else:
            projection = _closed_cell_projection(
                cell_id=cell.cell_id, cluster_count=len(cell.cluster_ids),
                metric_output_ids=config.metric_output_ids,
                perturbation_ids=config.perturbation_ids, alpha_grid=config.alpha_grid,
                bootstrap_resamples=bootstrap_count, sign_flip_resamples=sign_count,
                random_seed=seed, state="not_tested_endpoint_closed",
                reason=f"closed after {failed_cell_id}",
            )
        projections.append(projection)
    downstream_rows = [row for result in projections for row in result["downstream_rows"]]
    metric_rows = [row for result in projections for row in result["metric_rows"]]
    bootstrap_rows = [row for result in projections for row in result["bootstrap_rows"]]
    sign_flip_rows = [row for result in projections for row in result["sign_flip_rows"]]
    holm_rows = [row for result in projections for row in result["holm_rows"]]
    endpoint_status_rows = [dict(row) for row in inputs.endpoint_status_rows]
    if failure is not None:
        failed_seen = False
        for row in endpoint_status_rows:
            if row["cell_id"] == "d1_full_domain_core":
                continue
            if row["cell_id"] == failed_cell_id:
                row.update({"reason": failure["error"], "state": failure["state"]})
                failed_seen = True
            elif failed_seen:
                row.update({"reason": f"closed after {failed_cell_id}", "state": "not_tested_endpoint_closed"})
    status = "failed" if failure is not None else "complete"
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": config.document.get("claim_boundary", CLAIM_BOUNDARY),
        "code_authority": code,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "artifact_files": len(ARTIFACT_PAYLOAD_FILES) + 2,
            "bootstrap_results": len(bootstrap_rows),
            "configured_payloads": len(ARTIFACT_PAYLOAD_FILES),
            "downstream_protocol_effects": len(downstream_rows),
            "endpoint_status": len(endpoint_status_rows),
            "holm_family": len(holm_rows),
            "metric_protocol_effects": len(metric_rows),
            "sign_flip_results": len(sign_flip_rows),
        },
        "environment_authority": environment,
        "experiment_id": EXPERIMENT_ID,
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "run_id": run_id,
        "failure": failure,
        "status": status,
        "synthetic_fixture": config.synthetic_fixture,
    }
    payloads = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": canonical_json_bytes(dict(inputs.authority_bridge)),
        "preflight.json": canonical_json_bytes(dict(inputs.preflight)),
        "endpoint_status.jsonl": jsonl_bytes(endpoint_status_rows),
        "downstream_protocol_effects.jsonl": jsonl_bytes(downstream_rows),
        "metric_protocol_effects.jsonl": jsonl_bytes(metric_rows),
        "bootstrap_results.jsonl": jsonl_bytes(bootstrap_rows),
        "sign_flip_results.jsonl": jsonl_bytes(sign_flip_rows),
        "holm_family.jsonl": jsonl_bytes(holm_rows),
        "protocol_downstream_contrast_table.csv": _csv_downstream(downstream_rows),
        "protocol_metric_interaction_table.csv": _csv_metrics(metric_rows),
        "manifest.json": canonical_json_bytes(manifest),
    }
    terminal_name = "failed.json" if failure is not None else "complete.json"
    terminal_document = {
        "endpoint_state": failure["state"] if failure is not None else "complete",
        "run": run_id, "run_id": run_id, "schema": MARKER_SCHEMA_VERSION,
        "status": status,
    }
    if failure is not None:
        terminal_document["failure"] = failure
    terminal = canonical_json_bytes(terminal_document)
    run_path.mkdir(parents=True)
    for name in ARTIFACT_PAYLOAD_FILES:
        (run_path / name).write_bytes(payloads[name])
    (run_path / terminal_name).write_bytes(terminal)
    (run_path / "SHA256SUMS").write_bytes(write_sha256sums(payloads, terminal_name, terminal))
    return Phase4ProtocolABContrastSummary(
        run_path, run_id, status, len(endpoint_status_rows),
        len(bootstrap_rows), len(holm_rows),
    )


def reconstruct_phase4_protocol_ab_contrast_inputs(
    config: Phase4ProtocolABContrastConfig,
) -> Phase4ProtocolABContrastInputs:
    receipts = validate_real_parent_authorities(config)
    parents = config.document["parent_artifacts"]
    parent_paths = {
        name: ROOT / str(value["relative_path"])
        for name, value in parents.items()
    }

    d2_alpha = json.loads((parent_paths["d2_b"] / "alpha0_equivalence.json").read_bytes())
    if (
        sha256_file(parent_paths["d2_b"] / "alpha0_equivalence.json")
        != str(config.document["alpha0_admission"]["d2_sha256"])
        or int(d2_alpha.get("mismatch_count", -1)) != 0
        or any(int(value) != 0 for value in d2_alpha.get("mismatch_counts", {}).values())
    ):
        raise Phase4ProtocolABContrastError("D2 alpha-zero admission failed")
    d4_alpha = json.loads((parent_paths["d4_b"] / "alpha0_equivalence.json").read_bytes())
    if (
        sha256_file(parent_paths["d4_b"] / "alpha0_equivalence.json")
        != str(config.document["alpha0_admission"]["d4_sha256"])
        or d4_alpha.get("status") != "passed"
        or int(d4_alpha.get("mismatch_count", -1)) != 0
        or d4_alpha.get("expected_digests") != d4_alpha.get("observed_digests")
    ):
        raise Phase4ProtocolABContrastError("D4 alpha-zero admission failed")
    d1_alpha = json.loads((parent_paths["d1_b"] / "alpha0_equivalence.json").read_bytes())
    if (
        sha256_file(parent_paths["d1_b"] / "alpha0_equivalence.json")
        != str(config.document["alpha0_admission"]["d1_sha256"])
        or d1_alpha.get("status") != "failed"
        or int(d1_alpha.get("mismatch_count", -1)) != 3
    ):
        raise Phase4ProtocolABContrastError("D1 closed alpha-zero disposition mismatch")
    d5_bridge = json.loads((parent_paths["d5_b"] / "authority_bridge.json").read_bytes())
    if (
        d5_bridge.get("condition_bridge", {}).get("state") != "complete"
        or int(d5_bridge.get("condition_bridge", {}).get("mismatch_count", -1)) != 0
        or d5_bridge.get("operator_bridge", {}).get("state") != "complete"
        or int(d5_bridge.get("operator_bridge", {}).get("mismatch_count", -1)) != 0
    ):
        raise Phase4ProtocolABContrastError("D5 authority bridge admission failed")
    d5_alpha_a = _alpha0_csv_row(parent_paths["d5_a"] / "condition_summary.csv")
    d5_alpha_b = _alpha0_csv_row(parent_paths["d5_b"] / "condition_summary.csv")
    if dict(d5_alpha_a) != dict(d5_alpha_b):
        raise Phase4ProtocolABContrastError("D5 shared alpha-zero row mismatch")
    d5_alpha_raw = canonical_json_bytes(dict(d5_alpha_a))
    expected_d5 = config.document["alpha0_admission"]["d5_shared_summary"]
    if len(d5_alpha_raw) != int(expected_d5["bytes"]) or sha256_hex(d5_alpha_raw) != str(expected_d5["sha256"]):
        raise Phase4ProtocolABContrastError("D5 shared alpha-zero identity mismatch")

    cell_specs = (
        ("d5_full_domain_core", "d5_a", "d5_b", "class_observations.jsonl", None, "cluster_id", "mineral_class", "occurrence_count"),
        ("d2_5shot", "d2_a", "d2_b", "class_observations.jsonl", 5, "class_label", "class_label", None),
        ("d2_10shot", "d2_a", "d2_b", "class_observations.jsonl", 10, "class_label", "class_label", None),
        ("d2_20shot", "d2_a", "d2_b", "class_observations.jsonl", 20, "class_label", "class_label", None),
        ("d4_full_domain_core", "d4_a", "d4_b", "well_observations.jsonl", None, "well_id", "physical_well", "acquisition_count"),
    )
    loaded = {}
    cells = []
    status_rows = []
    pairing_receipts = {}
    for cell_id, a_name, b_name, filename, shot, cluster_field, cluster_kind, count_field in cell_specs:
        observation_contract = config.document.get("observation_payloads", {})
        alignment_contract = config.document.get("parent_alignment", {}).get(cell_id, {})
        for parent_name in (a_name, b_name):
            observation_path = parent_paths[parent_name] / filename
            expected_observation = observation_contract.get(parent_name, {})
            if (
                observation_path.stat().st_size != int(expected_observation.get("bytes", -1))
                or sha256_file(observation_path) != str(expected_observation.get("sha256", ""))
            ):
                raise Phase4ProtocolABContrastError(f"observation payload identity mismatch: {parent_name}")
        if (
            alignment_contract.get("protocol_a_parent") != a_name
            or alignment_contract.get("protocol_b_parent") != b_name
        ):
            raise Phase4ProtocolABContrastError(f"parent alignment mapping mismatch: {cell_id}")
        for side, parent_name in (("a", a_name), ("b", b_name)):
            alignment_path = parent_paths[parent_name] / "alignment_results.jsonl"
            if sha256_file(alignment_path) != str(alignment_contract.get(f"protocol_{side}_payload_sha256", "")):
                raise Phase4ProtocolABContrastError(f"parent alignment payload mismatch: {cell_id}/{side}")
        if a_name not in loaded:
            loaded[a_name] = _read_jsonl(parent_paths[a_name] / filename)
            loaded[b_name] = _read_jsonl(parent_paths[b_name] / filename)
        rows_a = loaded[a_name]
        rows_b = loaded[b_name]
        if shot is not None:
            rows_a = [row for row in rows_a if int(row.get("shot_count", -1)) == shot]
            rows_b = [row for row in rows_b if int(row.get("shot_count", -1)) == shot]
        expected_identity = config.document["pairing_identities"][cell_id]
        key_identity = _projection_identity(
            cell_id, rows_a, cluster_field=cluster_field, count_field=count_field,
            metric_output_ids=config.metric_output_ids,
            perturbation_ids=config.perturbation_ids, alpha_grid=config.alpha_grid,
            include_metric=False,
        )
        metric_identity = _projection_identity(
            cell_id, rows_a, cluster_field=cluster_field, count_field=count_field,
            metric_output_ids=config.metric_output_ids,
            perturbation_ids=config.perturbation_ids, alpha_grid=config.alpha_grid,
            include_metric=True,
        )
        if dict(key_identity) != dict(expected_identity["key_projection"]) or dict(metric_identity) != dict(expected_identity["metric_projection"]):
            raise Phase4ProtocolABContrastError(f"pairing projection identity mismatch: {cell_id}")
        cell = adapt_protocol_observation_rows(
            cell_id=cell_id, rows_a=rows_a, rows_b=rows_b,
            cluster_field=cluster_field, cluster_kind=cluster_kind,
            count_field=count_field,
            metric_output_ids=config.metric_output_ids,
            perturbation_ids=config.perturbation_ids, alpha_grid=config.alpha_grid,
            parents_complete=True, legacy_missing_state=cell_id.startswith("d2_"),
            use_protocol_a_metric=True,
        )
        parent_alignment_a = _parent_alignment_rows(parent_paths[a_name], shot)
        parent_alignment_b = _parent_alignment_rows(parent_paths[b_name], shot)
        cell = ProtocolABCellInput(
            cell_id=cell.cell_id, cluster_kind=cell.cluster_kind,
            cluster_ids=cell.cluster_ids, metric_output_ids=cell.metric_output_ids,
            perturbation_ids=cell.perturbation_ids, alpha_grid=cell.alpha_grid,
            metric_harm=cell.metric_harm, downstream_a=cell.downstream_a,
            downstream_b=cell.downstream_b,
            metric_reconciliation=cell.metric_reconciliation,
            parent_alignment_a=parent_alignment_a,
            parent_alignment_b=parent_alignment_b,
        )
        expected_rows = int(expected_identity["row_count"])
        if expected_rows != len(rows_a) or expected_rows != len(rows_b):
            raise Phase4ProtocolABContrastError(f"pairing row count mismatch: {cell_id}")
        if len(cell.cluster_ids) != int(expected_identity["cluster_count"]):
            raise Phase4ProtocolABContrastError(f"cluster count mismatch: {cell_id}")
        expected_reconciliation = config.document["metric_reconciliation"][cell_id]
        if int(cell.metric_reconciliation["mismatch_count"]) != int(expected_reconciliation["mismatch_count"]):
            raise Phase4ProtocolABContrastError(f"metric reconciliation count mismatch: {cell_id}")
        if float(cell.metric_reconciliation["max_abs_difference"]).hex() != float(expected_reconciliation["max_abs_difference"]).hex():
            raise Phase4ProtocolABContrastError(f"metric reconciliation maximum mismatch: {cell_id}")
        cells.append(cell)
        pairing_receipts[cell_id] = {
            "cluster_count": len(cell.cluster_ids),
            "key_projection": dict(key_identity),
            "metric_projection": dict(metric_identity),
            "metric_reconciliation": dict(cell.metric_reconciliation),
            "row_count": expected_rows,
        }
        alpha0_kind = (
            "shared_alpha0_summary" if cell_id == "d5_full_domain_core"
            else "exact_projection_receipt"
        )
        status_rows.append({
            "alpha0_state": "passed", "cell_id": cell_id,
            "cluster_count": len(cell.cluster_ids), "cluster_kind": cluster_kind,
            "key_projection": dict(key_identity),
            "metric_count": len(config.metric_output_ids),
            "metric_projection": dict(metric_identity),
            "metric_reconciliation": dict(cell.metric_reconciliation),
            "paired_row_count": expected_rows, "perturbation_count": len(config.perturbation_ids),
            "protocol_a_parent": dict(receipts[a_name]),
            "protocol_b_parent": dict(receipts[b_name]),
            "reason": None, "state": "evaluable",
            "alpha0_evidence_type": alpha0_kind,
        })
        if shot is None:
            del loaded[a_name]
            del loaded[b_name]
    d1_status = {
        "alpha0_state": "failed_alpha0_equivalence",
        "alpha0_evidence_type": "exact_projection_receipt",
        "cell_id": "d1_full_domain_core", "cluster_count": 30,
        "cluster_kind": "class_label", "metric_count": 13,
        "paired_row_count": 0, "perturbation_count": 5,
        "reason": "protocol_b_failed_alpha0_equivalence",
        "protocol_a_parent": dict(receipts["d1_a"]),
        "protocol_b_parent": dict(receipts["d1_b"]),
        "state": "not_evaluable_failed_alpha0_equivalence",
    }
    ordered_status = []
    by_id = {row["cell_id"]: row for row in status_rows}
    by_id["d1_full_domain_core"] = d1_status
    for cell_id in config.cell_ids:
        ordered_status.append(by_id[cell_id])
    authority_bridge = {
        "alpha0_admission": {
            "d2": {"mismatch_count": 0, "sha256": sha256_file(parent_paths["d2_b"] / "alpha0_equivalence.json")},
            "d4": {"mismatch_count": 0, "sha256": sha256_file(parent_paths["d4_b"] / "alpha0_equivalence.json"), "status": "passed"},
            "d5": {"canonical_bytes": len(d5_alpha_raw), "canonical_sha256": sha256_hex(d5_alpha_raw), "evidence": "shared_alpha0_summary"},
        },
        "parents": {name: dict(value) for name, value in receipts.items()},
        "state": "complete",
    }
    preflight = {
        "cell_states": {row["cell_id"]: row["state"] for row in ordered_status},
        "evaluable_cell_count": 5,
        "fixed_cell_count": 6,
        "metric_authority": "protocol_a",
        "paired_observation_row_count": sum(int(value["row_count"]) for value in pairing_receipts.values()),
        "pairing_receipts": pairing_receipts,
        "state": "complete",
        "synthetic_fixture": False,
    }
    return Phase4ProtocolABContrastInputs(
        cells=tuple(cells), endpoint_status_rows=tuple(ordered_status),
        authority_bridge=MappingProxyType(authority_bridge),
        preflight=MappingProxyType(preflight),
    )


def build_phase4_protocol_ab_contrast(
    output_root: Path, *, worker_count: int = 5
) -> Phase4ProtocolABContrastSummary:
    config = load_phase4_protocol_ab_contrast_config()
    code = _code_authority()
    environment = _environment_authority()
    run_id = stable_run_id(
        config_sha256=config.sha256,
        parent_identity_projection=dict(config.document["parent_artifacts"]),
        code_authority=code,
        environment_authority=environment,
    )
    if (Path(output_root) / run_id).exists():
        raise Phase4ProtocolABContrastError("append-only run path already exists")
    inputs = reconstruct_phase4_protocol_ab_contrast_inputs(config)
    return build_phase4_protocol_ab_contrast_from_inputs(
        output_root, inputs=inputs, config=config, worker_count=worker_count,
        _precomputed_identity=(run_id, code, environment),
    )


__all__ = [
    "ARTIFACT_PAYLOAD_FILES", "Phase4ProtocolABContrastConfig",
    "Phase4ProtocolABContrastError", "Phase4ProtocolABContrastInputs",
    "Phase4ProtocolABContrastSummary", "ProtocolABCellInput",
    "adapt_protocol_observation_rows", "build_phase4_protocol_ab_contrast",
    "build_phase4_protocol_ab_contrast_from_inputs", "build_protocol_ab_holm",
    "compute_protocol_ab_cell", "load_phase4_protocol_ab_contrast_config",
    "make_synthetic_protocol_ab_contrast_config",
    "make_synthetic_protocol_ab_contrast_inputs",
    "parse_phase4_protocol_ab_contrast_config",
    "reconstruct_phase4_protocol_ab_contrast_inputs",
    "validate_real_parent_authorities",
    "validate_parent_artifact",
]
