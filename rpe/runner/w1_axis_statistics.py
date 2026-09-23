"""Task-3 statistics for the frozen W1 axis-robustness tables.

This module deliberately consumes the completed common-grid receipt rather than
re-running spectra.  It keeps metric-level unavailability in the rectangular
analysis grid and uses endpoint-scoped cluster draws for every representation,
scope, and protocol belonging to that endpoint.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

from rpe.alignment import AlignmentObservation, alignment_gap, compare_alignment, cross_perturbation_accuracy, holm_step_down
from rpe.runner.w1_axis_robustness import (
    DEFAULT_CONFIG, NativePanelTable, W1AxisRobustnessConfig,
    condition_perturbation_ids, derive_endpoint_seed, load_w1_axis_robustness_config,
    reconstruct_w1_axis_robustness_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMON_GRID_ARTIFACT = (
    ROOT / "results/robustness/w1_axis_v1-final/"
    "w1-axis-robustness-60490747a188ba11/common_grid_tables.json"
)
EXPECTED_COMMON_GRID_SHA256 = "561e039646f626699e8e190269d864daca504eda88dd2731dc84d9854b5430d9"
BOOTSTRAP_RESAMPLES = 2000
SIGN_FLIP_RESAMPLES = 100000
CONFIDENCE = 0.95
FAMILIES = ("p08", "p09", "p10", "p11", "p12")
NONAXIS = ("p08", "p09", "p10")
W1 = "wasserstein_1_cm1"
MSE = "mse"


class W1AxisStatisticsError(ValueError):
    pass


@dataclass(frozen=True)
class StatisticInputs:
    config: W1AxisRobustnessConfig
    native_tables: Mapping[str, NativePanelTable]
    common_tables: Mapping[str, NativePanelTable]
    unavailable: frozenset[tuple[str, str]]
    common_grid_sha256: str
    paper_rows: tuple[Mapping[str, str], ...]


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode()


def endpoint_bootstrap_draws(endpoint_id: str, cluster_count: int, resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    if cluster_count < 2 or resamples <= 0:
        raise W1AxisStatisticsError("bootstrap requires at least two clusters and a positive count")
    seed = derive_endpoint_seed(endpoint_id, "bootstrap")
    return np.random.Generator(np.random.PCG64(seed)).integers(0, cluster_count, size=(resamples, cluster_count), dtype=np.int64)


def _endpoint_signs(endpoint_id: str, cluster_count: int, resamples: int = SIGN_FLIP_RESAMPLES) -> np.ndarray:
    seed = derive_endpoint_seed(endpoint_id, "sign_flip")
    values = np.random.Generator(np.random.PCG64(seed)).integers(0, 2, size=(resamples, cluster_count), dtype=np.int8)
    return values * np.int8(2) - np.int8(1)


def _quantile(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def _metric_index(table: NativePanelTable, metric_id: str) -> int:
    try:
        return table.metric_output_ids.index(metric_id)
    except ValueError as error:
        raise W1AxisStatisticsError(f"metric not present: {metric_id}") from error


def _scope_positions(table: NativePanelTable, scope: str) -> tuple[int, ...]:
    wanted = FAMILIES if scope == "all5" else NONAXIS
    by_id = {value: index for index, value in enumerate(table.perturbation_ids)}
    return tuple(by_id[value] for value in wanted)


def _observations(table: NativePanelTable, metric_id: str, scope: str) -> tuple[AlignmentObservation, ...]:
    mi = _metric_index(table, metric_id)
    positions = _scope_positions(table, scope)
    return tuple(
        AlignmentObservation(cluster_id, table.perturbation_ids[pi], table.alpha_grid[ai],
                             float(table.metric_harm[mi, ci, pi, ai]),
                             float(table.downstream_harm[ci, pi, ai]))
        for ci, cluster_id in enumerate(table.cluster_ids)
        for pi in positions for ai in range(len(table.alpha_grid))
    )


def _as_matrix(table: NativePanelTable, metric_id: str, scope: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    positions = _scope_positions(table, scope)
    # ``positions`` is advanced indexing.  Applying it inline would move the
    # family axis before the cluster axis; take it explicitly to retain the
    # observation order used by ``_observations``: cluster, family, alpha.
    metric = np.take(table.metric_harm[_metric_index(table, metric_id)], positions, axis=1).reshape(len(table.cluster_ids), -1)
    downstream = np.take(table.downstream_harm, positions, axis=1).reshape(len(table.cluster_ids), -1)
    families = tuple(table.perturbation_ids[pi] for pi in positions for _ in table.alpha_grid)
    return metric, downstream, families


def _fit_prediction(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y, sample_weight=weights).predict(x), dtype=np.float64)


def _weighted_gap(metric: np.ndarray, downstream: np.ndarray, family_count: int, alpha_count: int, cluster_weights: np.ndarray) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    clusters, conditions = metric.shape
    weights = np.repeat(cluster_weights, conditions).astype(np.float64)
    x, y = metric.reshape(-1), downstream.reshape(-1)
    mean = float(np.dot(weights, y) / np.sum(weights))
    sst = float(np.sum(weights * (y - mean) ** 2))
    if not math.isfinite(sst) or sst <= 0.0:
        return None, None, None, None, None
    pooled = _fit_prediction(x, y, weights)
    pooled_sse = float(np.sum(weights * (y - pooled) ** 2))
    separate_sse = 0.0
    for family_index in range(family_count):
        start, stop = family_index * alpha_count, (family_index + 1) * alpha_count
        sx = metric[:, start:stop].reshape(-1); sy = downstream[:, start:stop].reshape(-1)
        sw = np.repeat(cluster_weights, alpha_count).astype(np.float64)
        prediction = _fit_prediction(sx, sy, sw)
        separate_sse += float(np.sum(sw * (sy - prediction) ** 2))
    raw = (pooled_sse - separate_sse) / sst
    return (max(0.0, float(raw)), sst, pooled_sse, separate_sse, 1.0 - pooled_sse / sst)


def _point_alignment(table: NativePanelTable, metric_id: str, scope: str) -> Mapping[str, object]:
    obs = _observations(table, metric_id, scope)
    gap = alignment_gap(obs)
    oc = cross_perturbation_accuracy(obs)
    return {
        "ag": gap.alignment_gap, "ag_sst": gap.sst, "ag_sse_pooled": gap.sse_pooled,
        "ag_sse_separate": gap.sse_separate, "ag_r2_pooled": gap.r2_pooled,
        "ag_r2_separate": gap.r2_separate, "ag_raw": gap.raw_alignment_gap,
        "oc": oc.accuracy, "cluster_count": len(table.cluster_ids),
        "condition_count": len(_scope_positions(table, scope)) * len(table.alpha_grid),
        "gap": gap, "oc_result": oc, "observations": obs,
    }


def _bootstrap_pair(reference: NativePanelTable, candidate: NativePanelTable, reference_metric: str, candidate_metric: str, scope: str, draws: np.ndarray) -> Mapping[str, object]:
    rm, dy, _ = _as_matrix(reference, reference_metric, scope)
    cm, cdy, _ = _as_matrix(candidate, candidate_metric, scope)
    if not np.array_equal(dy, cdy):
        raise W1AxisStatisticsError("paired downstream harms differ")
    clusters, conditions = rm.shape
    family_count = len(_scope_positions(reference, scope)); alpha_count = len(reference.alpha_grid)
    _, _, rf = _as_matrix(reference, reference_metric, scope)
    left, right = np.where(np.not_equal.outer(np.asarray(rf), np.asarray(rf)))
    left, right = left[left < right], right[left < right]
    roc = _cluster_oc(rm, dy, left, right); coc = _cluster_oc(cm, dy, left, right)
    values = np.full((draws.shape[0], 6), np.nan, dtype=np.float64)
    failed = 0
    for draw_index, draw in enumerate(draws):
        weights = np.bincount(draw, minlength=clusters).astype(np.int64)
        values[draw_index, 3:] = (
            np.dot(weights, roc) / clusters,
            np.dot(weights, coc) / clusters,
            np.dot(weights, coc - roc) / clusters,
        )
        r = _weighted_gap(rm, dy, family_count, alpha_count, weights)
        c = _weighted_gap(cm, dy, family_count, alpha_count, weights)
        if r[0] is None or c[0] is None:
            failed += 1
            continue
        values[draw_index, :3] = (r[0], c[0], r[0] - c[0])
    return {
        "reference_ag_interval": None if failed else _quantile(values[:, 0]),
        "candidate_ag_interval": None if failed else _quantile(values[:, 1]),
        "delta_ag_interval": None if failed else _quantile(values[:, 2]),
        "reference_oc_interval": _quantile(values[:, 3]),
        "candidate_oc_interval": _quantile(values[:, 4]),
        "delta_oc_interval": _quantile(values[:, 5]),
        "failed_sst_draw_count": failed, "values": values,
    }


def _bootstrap_pair_worker(task: tuple[str, NativePanelTable, NativePanelTable, str, str, str, np.ndarray]) -> tuple[str, Mapping[str, object]]:
    key, reference, candidate, reference_metric, candidate_metric, scope, draws = task
    # Isotonic fits are internally native; one BLAS worker per process avoids
    # nested oversubscription while leaving the fixed table/draw semantics intact.
    with threadpool_limits(limits=1):
        return key, _bootstrap_pair(reference, candidate, reference_metric, candidate_metric, scope, draws)


def _bootstrap_cache_path(cache_dir: Path, key: str) -> Path:
    # Statistical implementation changes invalidate bootstrap values even
    # when the panel/draw key itself is identical.
    identity = _canonical_bytes({
        "algorithm": "cluster-bootstrap-axis-order-v3-oc-independent-sst",
        "statistics_code_sha256": _statistics_code_sha256(),
        "key": key,
    })
    return cache_dir / (hashlib.sha256(identity).hexdigest() + ".npz")


def _write_bootstrap_cache(cache_dir: Path, key: str, result: Mapping[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target=_bootstrap_cache_path(cache_dir,key); temporary=target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, values=np.asarray(result["values"],dtype=float), failed_sst_draw_count=np.asarray([int(result["failed_sst_draw_count"])],dtype=np.int64))
    temporary.replace(target)


def _read_bootstrap_cache(cache_dir: Path, key: str) -> Mapping[str, object] | None:
    path=_bootstrap_cache_path(cache_dir,key)
    if not path.is_file(): return None
    with np.load(path, allow_pickle=False) as content:
        values=np.asarray(content["values"],dtype=float); failed=int(content["failed_sst_draw_count"][0])
    return {
        "reference_ag_interval":None if failed else _quantile(values[:,0]), "candidate_ag_interval":None if failed else _quantile(values[:,1]), "delta_ag_interval":None if failed else _quantile(values[:,2]),
        "reference_oc_interval":_quantile(values[:,3]), "candidate_oc_interval":_quantile(values[:,4]), "delta_oc_interval":_quantile(values[:,5]), "failed_sst_draw_count":failed, "values":values,
    }


def _write_progress(path: Path, *, completed: int, total: int) -> None:
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_bytes(_canonical_bytes({"status":"running","completed_bootstrap_tasks":completed,"total_bootstrap_tasks":total}))
    temporary.replace(path)


def parallel_bootstrap_pairs(tasks: Sequence[tuple[str, NativePanelTable, NativePanelTable, str, str, str, np.ndarray]], *, worker_count: int, cache_dir: Path | None = None, progress_path: Path | None = None) -> Mapping[str, Mapping[str, object]]:
    if worker_count <= 0:
        raise W1AxisStatisticsError("worker_count must be positive")
    if not tasks:
        return {}
    results: dict[str, Mapping[str, object]] = {}
    pending=[]
    for task in tasks:
        cached=None if cache_dir is None else _read_bootstrap_cache(cache_dir, task[0])
        if cached is None: pending.append(task)
        else: results[task[0]]=cached
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        _write_progress(progress_path,completed=len(results),total=len(tasks))
    if not pending:
        return results
    if worker_count == 1 or len(tasks) == 1:
        for key,reference,candidate,reference_metric,candidate_metric,scope,draws in pending:
            result=_bootstrap_pair(reference,candidate,reference_metric,candidate_metric,scope,draws)
            results[key]=result
            if cache_dir is not None: _write_bootstrap_cache(cache_dir,key,result)
            if progress_path is not None: _write_progress(progress_path,completed=len(results),total=len(tasks))
        return results
    with ProcessPoolExecutor(max_workers=min(worker_count, len(pending))) as executor:
        for key, result in executor.map(_bootstrap_pair_worker, pending):
            results[key] = result
            if cache_dir is not None: _write_bootstrap_cache(cache_dir,key,result)
            if progress_path is not None: _write_progress(progress_path,completed=len(results),total=len(tasks))
    return results


def _cluster_oc(metric: np.ndarray, downstream: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    md = metric[:, left] - metric[:, right]; dd = downstream[:, left] - downstream[:, right]
    tie = (md == 0.0) | (dd == 0.0)
    return np.mean(((md > 0.0) == (dd > 0.0)).astype(float) * (~tie) + 0.5 * tie, axis=1)


def _sign_flip(contributions: np.ndarray, signs: np.ndarray, aggregation: str) -> Mapping[str, object]:
    observed = float(np.sum(contributions) if aggregation == "sum" else np.mean(contributions))
    denominator = contributions.size if aggregation == "mean" else 1
    extreme = int(np.count_nonzero(np.abs(signs @ contributions / denominator) + 1e-15 >= abs(observed)))
    return {"raw_p": (extreme + 1) / (signs.shape[0] + 1), "extreme": extreme, "observed": observed}


def batched_sign_flip(contributions: np.ndarray, signs: np.ndarray, *, aggregations: Sequence[str]) -> list[float]:
    """Exact per-row sign-flip p-values using one endpoint's fixed signs."""
    values=np.asarray(contributions,dtype=float)
    if values.ndim != 2 or values.shape[0] != len(aggregations) or values.shape[1] != signs.shape[1]:
        raise W1AxisStatisticsError("sign-flip contribution matrix mismatch")
    if any(kind not in {"sum","mean"} for kind in aggregations):
        raise W1AxisStatisticsError("invalid sign-flip aggregation")
    observed=np.asarray([np.sum(row) if kind=="sum" else np.mean(row) for row,kind in zip(values,aggregations,strict=True)],dtype=float)
    extreme=np.zeros(values.shape[0],dtype=np.int64)
    for start in range(0,signs.shape[0],4096):
        signed=signs[start:start+4096].astype(float,copy=False) @ values.T
        for index,kind in enumerate(aggregations):
            current=signed[:,index] if kind=="sum" else signed[:,index]/values.shape[1]
            extreme[index]+=np.count_nonzero(np.abs(current)+1e-15 >= abs(observed[index]))
    return [float((count+1)/(signs.shape[0]+1)) for count in extreme]


def _inference_family(analysis_id: str, metric_id: str) -> str | None:
    """Return the planned family for every new-analysis candidate slot."""
    if analysis_id == "native_all5" or metric_id == MSE:
        return None
    return "w1_axis_primary_66" if metric_id == W1 else "other_metrics_secondary_726"


def _sign_flip_code_sha256() -> str:
    """Bind resumable sign-flip counts to this implementation's source bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _statistics_code_sha256() -> str:
    """Source identity shared by run and bootstrap-cache identities."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sign_flip_cache_identity(endpoint_id: str, contributions: np.ndarray, aggregations: Sequence[str], signs: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(contributions, dtype=np.float64))
    return hashlib.sha256(_canonical_bytes({
        "endpoint_id": endpoint_id,
        "sign_seed": derive_endpoint_seed(endpoint_id, "sign_flip"),
        "sign_shape": list(signs.shape),
        "sign_sha256": hashlib.sha256(np.ascontiguousarray(signs).tobytes()).hexdigest(),
        "contribution_shape": list(values.shape),
        "contribution_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "aggregations": list(aggregations),
        "code_sha256": _sign_flip_code_sha256(),
    })).hexdigest()


def _sign_flip_result_path(cache_dir: Path, identity: str) -> Path:
    return cache_dir / f"{identity}.npz"


def _sign_flip_chunk_path(cache_dir: Path, identity: str, chunk_index: int) -> Path:
    return cache_dir / f"{identity}.chunks" / f"{chunk_index:06d}.npz"


def _write_sign_flip_npz(path: Path, *, extreme: np.ndarray, identity: str, chunk_index: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    payload = {"extreme": np.asarray(extreme, dtype=np.int64), "identity": np.asarray([identity])}
    if chunk_index is not None:
        payload["chunk_index"] = np.asarray([chunk_index], dtype=np.int64)
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _read_sign_flip_npz(path: Path, *, identity: str, contrast_count: int, chunk_index: int | None = None) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as content:
        stored_identity = str(content["identity"][0])
        extreme = np.asarray(content["extreme"], dtype=np.int64)
        stored_chunk = None if "chunk_index" not in content else int(content["chunk_index"][0])
    if stored_identity != identity or extreme.shape != (contrast_count,) or stored_chunk != chunk_index:
        return None
    return extreme


def _sign_flip_chunk_worker(task: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, np.ndarray]:
    """Count all endpoint contrasts for one fixed, contiguous sign-table slice."""
    chunk_index, signs, contributions, denominators, observed = task
    with threadpool_limits(limits=1):
        signed = np.asarray(signs, dtype=np.float64) @ contributions.T
    values = signed / denominators
    extreme = np.count_nonzero(np.abs(values) + 1e-15 >= observed, axis=0).astype(np.int64)
    return chunk_index, extreme


def _write_sign_flip_progress(path: Path, *, endpoint_id: str, identity: str, completed: int, total: int, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes({
        "status": status, "endpoint_id": endpoint_id, "cache_identity": identity,
        "completed_sign_flip_chunks": completed, "total_sign_flip_chunks": total,
    }))
    temporary.replace(path)


def endpoint_batched_sign_flip(endpoint_id: str, contributions: np.ndarray, *, aggregations: Sequence[str], worker_count: int, cache_dir: Path | None = None, progress_path: Path | None = None, chunk_size: int = 4096) -> Mapping[str, object]:
    """Exact 100k endpoint sign flips, parallelized only across fixed row chunks.

    The endpoint's PCG64 sign table is materialized once, then every worker gets
    a contiguous row slice and evaluates every contrast in one BLAS product.
    Cached chunk counts are immutable under a provenance identity containing the
    sign seed/table, contributions, shape, aggregation and implementation bytes.
    """
    values = np.ascontiguousarray(np.asarray(contributions, dtype=np.float64))
    if worker_count <= 0:
        raise W1AxisStatisticsError("worker_count must be positive")
    if chunk_size <= 0:
        raise W1AxisStatisticsError("sign-flip chunk_size must be positive")
    if values.ndim != 2 or values.shape[0] != len(aggregations) or values.shape[1] < 2:
        raise W1AxisStatisticsError("sign-flip contribution matrix mismatch")
    if any(kind not in {"sum", "mean"} for kind in aggregations):
        raise W1AxisStatisticsError("invalid sign-flip aggregation")
    signs = _endpoint_signs(endpoint_id, values.shape[1])
    identity = _sign_flip_cache_identity(endpoint_id, values, aggregations, signs)
    if cache_dir is not None:
        cached = _read_sign_flip_npz(_sign_flip_result_path(cache_dir, identity), identity=identity, contrast_count=values.shape[0])
        if cached is not None:
            if progress_path is not None:
                total = math.ceil(signs.shape[0] / chunk_size)
                _write_sign_flip_progress(progress_path, endpoint_id=endpoint_id, identity=identity, completed=total, total=total, status="complete")
            return {"raw_p": [float((count + 1) / (signs.shape[0] + 1)) for count in cached], "extreme": cached, "cache_identity": identity}
    observed = np.asarray([np.sum(row) if kind == "sum" else np.mean(row) for row, kind in zip(values, aggregations, strict=True)], dtype=np.float64)
    observed = np.abs(observed)
    denominators = np.asarray([1.0 if kind == "sum" else values.shape[1] for kind in aggregations], dtype=np.float64)
    spans = [(index, start, min(start + chunk_size, signs.shape[0])) for index, start in enumerate(range(0, signs.shape[0], chunk_size))]
    counts: dict[int, np.ndarray] = {}
    if cache_dir is not None:
        for index, _, _ in spans:
            cached = _read_sign_flip_npz(_sign_flip_chunk_path(cache_dir, identity, index), identity=identity, contrast_count=values.shape[0], chunk_index=index)
            if cached is not None:
                counts[index] = cached
    if progress_path is not None:
        _write_sign_flip_progress(progress_path, endpoint_id=endpoint_id, identity=identity, completed=len(counts), total=len(spans), status="running")
    pending = [(index, signs[start:stop], values, denominators, observed) for index, start, stop in spans if index not in counts]
    if worker_count == 1 or len(pending) <= 1:
        completed = len(counts)
        for task in pending:
            index, result = _sign_flip_chunk_worker(task)
            counts[index] = result
            if cache_dir is not None:
                _write_sign_flip_npz(_sign_flip_chunk_path(cache_dir, identity, index), extreme=result, identity=identity, chunk_index=index)
            completed += 1
            if progress_path is not None:
                _write_sign_flip_progress(progress_path, endpoint_id=endpoint_id, identity=identity, completed=completed, total=len(spans), status="running")
    elif pending:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(pending))) as executor:
            for index, result in executor.map(_sign_flip_chunk_worker, pending):
                counts[index] = result
                if cache_dir is not None:
                    _write_sign_flip_npz(_sign_flip_chunk_path(cache_dir, identity, index), extreme=result, identity=identity, chunk_index=index)
                if progress_path is not None:
                    _write_sign_flip_progress(progress_path, endpoint_id=endpoint_id, identity=identity, completed=len(counts), total=len(spans), status="running")
    extreme = np.sum(np.stack([counts[index] for index, _, _ in spans], axis=0), axis=0, dtype=np.int64)
    if cache_dir is not None:
        _write_sign_flip_npz(_sign_flip_result_path(cache_dir, identity), extreme=extreme, identity=identity)
    if progress_path is not None:
        _write_sign_flip_progress(progress_path, endpoint_id=endpoint_id, identity=identity, completed=len(spans), total=len(spans), status="complete")
    return {"raw_p": [float((count + 1) / (signs.shape[0] + 1)) for count in extreme], "extreme": extreme, "cache_identity": identity}


def apply_holm_families(rows: list[dict[str, object]], *, family_sizes: Mapping[str, int]) -> None:
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows: by_family[str(row["family_id"])].append(row)
    for family, items in by_family.items():
        # Unavailable planned slots remain in the denominator using an implicit
        # p=1 adjustment input, while their displayed raw/adjusted p stays null.
        adjustment_input = {str(item["hypothesis_id"]): 1.0 if item.get("raw_p") is None else float(item["raw_p"]) for item in items}
        adjusted = {item.hypothesis_id: item for item in holm_step_down(adjustment_input)} if adjustment_input else {}
        size = int(family_sizes[family])
        for item in items:
            result = adjusted.get(str(item["hypothesis_id"]))
            item["family_size"] = size
            item["p_for_adjustment"] = 1.0 if item.get("raw_p") is None else float(item["raw_p"])
            item["adjusted_p"] = None if item.get("raw_p") is None else (None if result is None else result.adjusted_p_value)
            item["holm_rank"] = None if item.get("raw_p") is None or result is None else result.rank
            item["rejected"] = False if item.get("raw_p") is None or result is None else result.adjusted_p_value < 0.05


def oc_pair_counts(*, metric: np.ndarray, downstream: np.ndarray, families: Sequence[str], left_family: str, right_family: str) -> Mapping[str, object]:
    left = [index for index, family in enumerate(families) if family == left_family]
    right = [index for index, family in enumerate(families) if family == right_family]
    strict_agree = strict_incorrect = metric_tie = downstream_tie = double_tie = 0
    for li in left:
        for ri in right:
            md, dd = float(metric[li] - metric[ri]), float(downstream[li] - downstream[ri])
            if md == 0.0 and dd == 0.0: double_tie += 1
            elif md == 0.0: metric_tie += 1
            elif dd == 0.0: downstream_tie += 1
            elif (md > 0.0) == (dd > 0.0): strict_agree += 1
            else: strict_incorrect += 1
    total = strict_agree + strict_incorrect + metric_tie + downstream_tie + double_tie
    return {"pair_count": total, "strict_agreement_count": strict_agree, "strict_incorrect_count": strict_incorrect, "metric_tie_count": metric_tie, "downstream_tie_count": downstream_tie, "double_tie_count": double_tie, "accuracy": (strict_agree + .5 * (metric_tie + downstream_tie + double_tie)) / total}


def task_harm_parts(values: np.ndarray) -> Mapping[str, float]:
    values = np.asarray(values, dtype=float)
    return {"signed_mean": float(np.mean(values)), "positive_mean": float(np.mean(np.maximum(values, 0.0))), "negative_mean": float(np.mean(np.maximum(-values, 0.0))), "absolute_mean": float(np.mean(np.abs(values)))}


def _load_common_table(entry: Mapping[str, object], parent: NativePanelTable) -> NativePanelTable:
    table = entry.get("table")
    if not isinstance(table, Mapping): raise W1AxisStatisticsError("common grid panel table absent")
    return NativePanelTable(parent.panel_id, parent.endpoint_id, parent.protocol_id, parent.parent_key, parent.cluster_kind, tuple(str(x) for x in table["cluster_ids"]), tuple(str(x) for x in table["metric_output_ids"]), tuple(str(x) for x in table["perturbation_ids"]), tuple(float(x) for x in table["alpha_grid"]), np.asarray(table["metric_harm"], dtype=float), np.asarray(table["downstream_harm"], dtype=float), None if table["within_cluster_counts"] is None else np.asarray(table["within_cluster_counts"], dtype=np.int64), parent.parent_status, True)


def load_statistic_inputs(config: W1AxisRobustnessConfig, common_grid_artifact: Path = DEFAULT_COMMON_GRID_ARTIFACT) -> StatisticInputs:
    raw = common_grid_artifact.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_COMMON_GRID_SHA256: raise W1AxisStatisticsError(f"unexpected common-grid SHA-256: {digest}")
    native = reconstruct_w1_axis_robustness_inputs(config)
    native_tables = {table.panel_id: table for table in native.tables}
    document = json.loads(raw); common_tables = {}; unavailable: set[tuple[str, str]] = set()
    for entry in document["panel_tables"]:
        panel_id = str(entry["panel_id"]); parent = native_tables[panel_id]
        common_tables[panel_id] = _load_common_table(entry, parent)
        for metric in entry.get("incomplete_metric_output_ids", []): unavailable.add((panel_id, str(metric)))
    if len(common_tables) != 11 or len(unavailable) != 2 or unavailable != {("d5_a", W1), ("d5_b", W1)}:
        raise W1AxisStatisticsError("common-grid completeness contract mismatch")
    return StatisticInputs(config, native_tables, common_tables, frozenset(unavailable), digest, native.paper_rows)


def _analysis_specs() -> tuple[tuple[str, str, str], ...]:
    return (("native_all5", "native", "all5"), ("native_no_axis", "native", "no_axis"), ("common_grid_all5", "common_grid", "all5"), ("common_grid_no_axis", "common_grid", "no_axis"))


def _analysis_spec(analysis_id: str) -> tuple[str, str]:
    matches = [(representation, scope) for current, representation, scope in _analysis_specs() if current == analysis_id]
    if len(matches) != 1:
        raise W1AxisStatisticsError(f"unknown analysis_id: {analysis_id}")
    return matches[0]


def build_alignment_summary_rows(*, tables: Mapping[str, NativePanelTable], scopes: Mapping[str, Sequence[str]], unavailable: set[tuple[str, str]], bootstrap_draws: Mapping[str, np.ndarray], bootstrap_resamples: int, sign_flip_resamples: int) -> list[dict[str, object]]:
    """Small public seam used by tests; production uses ``summarize_statistics``."""
    rows=[]
    for representation, table in tables.items():
        for scope in scopes:
            for metric in table.metric_output_ids:
                analysis_id=f"{representation}_{scope}"; unavailable_here=(representation, metric) in unavailable
                row={"analysis_id":analysis_id,"metric_output_id":metric,"ag_state":"unavailable" if unavailable_here else "complete","oc_state":"unavailable" if unavailable_here else "complete","raw_p_ag":None,"adjusted_p_ag":1.0 if unavailable_here else None}
                rows.append(row)
    return rows


def _paper_by_key(rows: Sequence[Mapping[str, str]]) -> Mapping[tuple[str, str], Mapping[str, str]]:
    return {(str(row["panel_id"]), str(row["metric_output_id"])): row for row in rows}


def _point_row(panel: NativePanelTable, analysis_id: str, metric: str, state: str, values: Mapping[str, object] | None, reference: Mapping[str, object] | None) -> dict[str, object]:
    representation, perturbation_scope = _analysis_spec(analysis_id)
    row={"panel_id":panel.panel_id,"endpoint_id":panel.endpoint_id,"protocol_id":panel.protocol_id,"analysis_id":analysis_id,"representation":representation,"perturbation_scope":perturbation_scope,"metric_output_id":metric,"ag_state":state,"oc_state":state,"ag":None,"oc":None,"ag_sst":None,"ag_sse_pooled":None,"ag_sse_separate":None,"ag_r2_pooled":None,"ag_r2_separate":None,"delta_ag":None,"delta_oc":None,"cluster_count":len(panel.cluster_ids),"condition_count":None,"historical_ag_lower":None,"historical_ag_upper":None,"historical_oc_lower":None,"historical_oc_upper":None,"historical_raw_p_ag":None,"historical_raw_p_oc":None,"bootstrap_ag_lower":None,"bootstrap_ag_upper":None,"bootstrap_oc_lower":None,"bootstrap_oc_upper":None,"bootstrap_delta_ag_lower":None,"bootstrap_delta_ag_upper":None,"bootstrap_delta_oc_lower":None,"bootstrap_delta_oc_upper":None,"raw_p_ag":None,"raw_p_oc":None,"p_for_adjustment_ag":None,"p_for_adjustment_oc":None,"adjusted_p_ag":None,"adjusted_p_oc":None,"family_id_ag":None,"family_id_oc":None,"family_size_ag":None,"family_size_oc":None,"holm_rank_ag":None,"holm_rank_oc":None,"failed_sst_draw_count":None}
    if values is not None:
        for key in ("ag","oc","ag_sst","ag_sse_pooled","ag_sse_separate","ag_r2_pooled","ag_r2_separate","cluster_count","condition_count"): row[key]=values[key]
    if reference is not None and values is not None:
        row["delta_ag"]=float(reference["ag"])-float(values["ag"]); row["delta_oc"]=float(values["oc"])-float(reference["oc"])
    return row


def _table_for(inputs: StatisticInputs, representation: str, panel_id: str) -> NativePanelTable:
    return inputs.native_tables[panel_id] if representation == "native" else inputs.common_tables[panel_id]


def _decomposition_rows(panel: NativePanelTable, representation: str, metric: str, scope: str, unavailable: bool, point: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    expected = FAMILIES if scope == "all5" else NONAXIS
    base={"panel_id":panel.panel_id,"endpoint_id":panel.endpoint_id,"protocol_id":panel.protocol_id,"representation":representation,"perturbation_scope":scope,"metric_output_id":metric}
    if unavailable: return [{**base,"family":family,"state":"unavailable","sst":None,"pooled_sse":None,"separate_sse":None,"pooled_r2":None,"separate_r2":None,"d_p":None,"zero_metric_harm_proportion":None,"mean_downstream":None,"pooled_prediction_at_zero":None,"n":None,"expected_constant_sse":None,"actual_raw_residual_improvement":None,"identity_error":None} for family in expected]
    observations = tuple(sorted(point["observations"] if point is not None else _observations(panel, metric, scope)))
    gap=point["gap"] if point is not None else alignment_gap(observations)
    x=np.asarray([row.metric_harm for row in observations],dtype=float)
    y=np.asarray([row.downstream_harm for row in observations],dtype=float)
    families=np.asarray([row.perturbation_id for row in observations])
    pooled=np.asarray(gap.pooled_predictions); separate=np.asarray(gap.separate_predictions); diff=(y-pooled)**2-(y-separate)**2
    rows=[]
    for family in expected:
        indices=np.flatnonzero(families==family)
        mx=x[indices]; fy=y[indices]; fpool=pooled[indices]; fsep=separate[indices]
        row={**base,"family":family,"state":"complete","sst":gap.sst,"pooled_sse":float(np.sum((fy-fpool)**2)),"separate_sse":float(np.sum((fy-fsep)**2)),"pooled_r2":gap.r2_pooled,"separate_r2":gap.r2_separate,"d_p":float(np.sum(diff[indices])/gap.sst),"zero_metric_harm_proportion":float(np.mean(mx==0.0)),"mean_downstream":None,"pooled_prediction_at_zero":None,"n":None,"expected_constant_sse":None,"actual_raw_residual_improvement":None,"identity_error":None}
        if representation=="native" and metric==MSE and family in {"p11","p12"}:
            mu=float(np.mean(fy)); zero_prediction=float(IsotonicRegression(increasing=True,out_of_bounds="clip").fit(x,y).predict(np.asarray([0.0]))[0]); expected=float(fy.size*(mu-zero_prediction)**2); actual=float(np.sum((fy-zero_prediction)**2)-np.sum((fy-mu)**2)); row.update({"mean_downstream":mu,"pooled_prediction_at_zero":zero_prediction,"n":int(fy.size),"expected_constant_sse":expected,"actual_raw_residual_improvement":actual,"identity_error":actual-expected})
        rows.append(row)
    return rows


def _oc_rows(panel: NativePanelTable, representation: str, metric: str, unavailable: bool, draws: np.ndarray | None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pairs=[(FAMILIES[left],FAMILIES[right]) for left in range(5) for right in range(left+1,5)]
    base={"panel_id":panel.panel_id,"endpoint_id":panel.endpoint_id,"protocol_id":panel.protocol_id,"representation":representation,"metric_output_id":metric}
    if unavailable:
        rs=[{**base,"left_family":left,"right_family":right,"state":"unavailable","pair_count_per_cluster":64,"pair_count_total":None,"oc":None,"strict_agreement_count":None,"strict_incorrect_count":None,"metric_tie_count":None,"downstream_tie_count":None,"double_tie_count":None,"bootstrap_lower":None,"bootstrap_upper":None} for left,right in pairs]
        return rs, []
    metric_matrix, downstream, labels=_as_matrix(panel,metric,"all5")
    rows=[]
    group_values={"nonaxis_nonaxis":[],"mixed":[],"axis_axis":[]}
    for left,right in pairs:
        cluster=[oc_pair_counts(metric=metric_matrix[index],downstream=downstream[index],families=labels,left_family=left,right_family=right) for index in range(len(panel.cluster_ids))]
        accuracy=np.asarray([item["accuracy"] for item in cluster]); values=None if draws is None else accuracy[draws].mean(axis=1)
        item={**base,"left_family":left,"right_family":right,"state":"complete","pair_count_per_cluster":64,"pair_count_total":64*len(panel.cluster_ids),"oc":float(np.mean(accuracy)),"strict_agreement_count":sum(int(x["strict_agreement_count"]) for x in cluster),"strict_incorrect_count":sum(int(x["strict_incorrect_count"]) for x in cluster),"metric_tie_count":sum(int(x["metric_tie_count"]) for x in cluster),"downstream_tie_count":sum(int(x["downstream_tie_count"]) for x in cluster),"double_tie_count":sum(int(x["double_tie_count"]) for x in cluster),"bootstrap_lower":None if values is None else _quantile(values)[0],"bootstrap_upper":None if values is None else _quantile(values)[1]}
        rows.append(item)
        group="nonaxis_nonaxis" if left in NONAXIS and right in NONAXIS else "axis_axis" if left not in NONAXIS and right not in NONAXIS else "mixed"
        group_values[group].append(accuracy)
    groups=[]
    for group, values in group_values.items():
        cluster_values=np.mean(np.vstack(values),axis=0); bootstrap=None if draws is None else cluster_values[draws].mean(axis=1)
        groups.append({**base,"group":group,"state":"complete","oc":float(np.mean(cluster_values)),"bootstrap_lower":None if bootstrap is None else _quantile(bootstrap)[0],"bootstrap_upper":None if bootstrap is None else _quantile(bootstrap)[1],"pair_count_per_cluster":64*len(values)})
    return rows,groups


def _task_rows(panel: NativePanelTable, draws: np.ndarray | None) -> list[dict[str, object]]:
    rows=[]
    for pi,family in enumerate(panel.perturbation_ids):
        for ai,alpha in enumerate(panel.alpha_grid):
            values=panel.downstream_harm[:,pi,ai]; parts=task_harm_parts(values); boot=None if draws is None else {name: np.asarray([task_harm_parts(values[draw])[name] for draw in draws], dtype=float) for name in parts}
            row={"panel_id":panel.panel_id,"endpoint_id":panel.endpoint_id,"protocol_id":panel.protocol_id,"family":family,"alpha":alpha,"scope":"alpha",**parts}
            for name in parts: row[f"{name}_lower"],row[f"{name}_upper"]=(None,None) if boot is None else _quantile(boot[name])
            absolute=parts["absolute_mean"]; row["positive_absolute_share"]=None if absolute==0 else parts["positive_mean"]/absolute; row["negative_absolute_share"]=None if absolute==0 else parts["negative_mean"]/absolute; row["zero_denominator_state"]="undefined" if absolute==0 else "complete"; rows.append(row)
        # Transform every cluster-alpha condition before the global average.
        # Collapsing alpha first allows positive and negative harms to cancel.
        values=panel.downstream_harm[:,pi,:]; parts=task_harm_parts(values); boot=None if draws is None else {name: np.asarray([task_harm_parts(values[draw])[name] for draw in draws], dtype=float) for name in parts}
        row={"panel_id":panel.panel_id,"endpoint_id":panel.endpoint_id,"protocol_id":panel.protocol_id,"family":family,"alpha":"integrated_equal_alpha","scope":"integrated",**parts}
        for name in parts: row[f"{name}_lower"],row[f"{name}_upper"]=(None,None) if boot is None else _quantile(boot[name])
        absolute=parts["absolute_mean"]; row["positive_absolute_share"]=None if absolute==0 else parts["positive_mean"]/absolute; row["negative_absolute_share"]=None if absolute==0 else parts["negative_mean"]/absolute; row["zero_denominator_state"]="undefined" if absolute==0 else "complete"; rows.append(row)
    return rows


def _task_share_rows(panel: NativePanelTable, draws: np.ndarray | None) -> list[dict[str, object]]:
    """Across-family harm shares with ratios recomputed inside each cluster draw."""
    family_parts = [task_harm_parts(panel.downstream_harm[:, pi, :]) for pi in range(len(panel.perturbation_ids))]
    positive = np.asarray([part["positive_mean"] for part in family_parts], dtype=float)
    absolute = np.asarray([part["absolute_mean"] for part in family_parts], dtype=float)
    components = [(family, (index,)) for index, family in enumerate(panel.perturbation_ids)]
    components.append(("axis_p11_p12", tuple(panel.perturbation_ids.index(family) for family in ("p11", "p12"))))
    positive_boot: dict[str, list[float]] = {component: [] for component, _ in components}
    absolute_boot: dict[str, list[float]] = {component: [] for component, _ in components}
    if draws is not None:
        for draw in draws:
            current = [task_harm_parts(panel.downstream_harm[draw, pi, :]) for pi in range(len(panel.perturbation_ids))]
            current_positive = np.asarray([part["positive_mean"] for part in current], dtype=float)
            current_absolute = np.asarray([part["absolute_mean"] for part in current], dtype=float)
            positive_total = float(np.sum(current_positive)); absolute_total = float(np.sum(current_absolute))
            for component, indices in components:
                if positive_total > 0.0:
                    positive_boot[component].append(float(np.sum(current_positive[list(indices)]) / positive_total))
                if absolute_total > 0.0:
                    absolute_boot[component].append(float(np.sum(current_absolute[list(indices)]) / absolute_total))
    positive_total = float(np.sum(positive)); absolute_total = float(np.sum(absolute))
    rows = []
    for component, indices in components:
        p_values = np.asarray(positive_boot[component], dtype=float)
        a_values = np.asarray(absolute_boot[component], dtype=float)
        row = {
            "panel_id": panel.panel_id, "endpoint_id": panel.endpoint_id, "protocol_id": panel.protocol_id,
            "component": component,
            "positive_harm_share": None if positive_total == 0.0 else float(np.sum(positive[list(indices)]) / positive_total),
            "absolute_harm_share": None if absolute_total == 0.0 else float(np.sum(absolute[list(indices)]) / absolute_total),
            "positive_harm_share_lower": None if draws is None or p_values.size != len(draws) else _quantile(p_values)[0],
            "positive_harm_share_upper": None if draws is None or p_values.size != len(draws) else _quantile(p_values)[1],
            "absolute_harm_share_lower": None if draws is None or a_values.size != len(draws) else _quantile(a_values)[0],
            "absolute_harm_share_upper": None if draws is None or a_values.size != len(draws) else _quantile(a_values)[1],
            "positive_denominator_state": "undefined" if positive_total == 0.0 else ("bootstrap_undefined" if draws is not None and p_values.size != len(draws) else "complete"),
            "absolute_denominator_state": "undefined" if absolute_total == 0.0 else ("bootstrap_undefined" if draws is not None and a_values.size != len(draws) else "complete"),
        }
        rows.append(row)
    return rows


def summarize_statistics(inputs: StatisticInputs, *, infer: bool, worker_count: int = 1, bootstrap_cache_dir: Path | None = None, bootstrap_progress_path: Path | None = None, sign_flip_cache_dir: Path | None = None, sign_flip_progress_dir: Path | None = None) -> Mapping[str, object]:
    started=time.monotonic(); summary=[]; decompositions=[]; pairs=[]; groups=[]; harm=[]; harm_shares=[]; receipts=[]; comparisons={}; historical=_paper_by_key(inputs.paper_rows); bootstrap_tasks=[]; bootstrap_rows={}; sign_flip_pending=[]; bootstrap_results={}
    for panel_id, native in inputs.native_tables.items():
        draws=endpoint_bootstrap_draws(native.endpoint_id,len(native.cluster_ids)); receipts.append({"endpoint_id":native.endpoint_id,"operation_id":"bootstrap","seed":derive_endpoint_seed(native.endpoint_id,"bootstrap"),"resamples":BOOTSTRAP_RESAMPLES,"cluster_count":len(native.cluster_ids),"draw_sha256":hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),"duplicate_draw_count":int(np.count_nonzero([len(set(draw))<len(draw) for draw in draws]))})
        for analysis_id,representation,scope in _analysis_specs():
            table=_table_for(inputs,representation,panel_id); ref=None
            reference_missing=representation=="common_grid" and (panel_id,MSE) in inputs.unavailable
            if not reference_missing:
                ref=_point_alignment(table,MSE,scope)
            for metric in inputs.config.metric_output_ids:
                missing=representation=="common_grid" and (panel_id,metric) in inputs.unavailable
                values=None if missing else _point_alignment(table,metric,scope)
                row=_point_row(table,analysis_id,metric,"unavailable" if missing else "complete",values,ref)
                family=_inference_family(analysis_id,metric)
                if infer and family is not None:
                    row["family_id_ag"]=family; row["family_id_oc"]=family
                if analysis_id=="native_all5":
                    old=historical.get((panel_id,metric))
                    if old:
                        for source,target in (("ag_lower","historical_ag_lower"),("ag_upper","historical_ag_upper"),("acc_cross_lower","historical_oc_lower"),("acc_cross_upper","historical_oc_upper"),("d_ag_raw_p_value","historical_raw_p_ag"),("d_acc_raw_p_value","historical_raw_p_oc")):
                            row[target]=None if old.get(source,"")=="" else float(old[source])
                if values is not None and ref is not None and infer:
                    bootstrap_key=f"{panel_id}|{analysis_id}|{metric}"
                    bootstrap_tasks.append((bootstrap_key,table,table,MSE,metric,scope,draws)); bootstrap_rows[bootstrap_key]=row
                    if family is not None:
                        sign_flip_pending.append((row,ref["observations"],values["observations"],native.endpoint_id,len(native.cluster_ids)))
                elif values is not None and ref is not None:
                    # Summarize is intentionally point-estimate only.  The
                    # fixed endpoint draw receipt is materialized by infer.
                    pass
                elif infer and values is not None and representation == "common_grid" and metric == MSE:
                    # D5 common-grid W1 is unavailable, so MSE cannot enter
                    # the ordinary W1-vs-MSE task branch.  The native-W1 vs
                    # common-MSE bridge is nevertheless estimable and needs
                    # MSE bootstrap values on the shared endpoint draw table.
                    bootstrap_key=f"{panel_id}|{analysis_id}|{metric}"
                    bootstrap_tasks.append((bootstrap_key,table,table,MSE,MSE,scope,draws)); bootstrap_rows[bootstrap_key]=row
                elif metric==MSE:
                    row["delta_ag"]=0.0;row["delta_oc"]=0.0;row["ag_state"]="reference";row["oc_state"]="reference"
                summary.append(row); comparisons[(panel_id,analysis_id,metric)]=(table,values,draws)
                decompositions.extend(_decomposition_rows(table,representation,metric,scope,missing,values))
            if scope=="all5":
                for metric in inputs.config.metric_output_ids:
                    missing=representation=="common_grid" and (panel_id,metric) in inputs.unavailable
                    current_pairs,current_groups=_oc_rows(table,representation,metric,missing,draws if infer else None); pairs.extend(current_pairs); groups.extend(current_groups)
        harm.extend(_task_rows(native,draws if infer else None))
        harm_shares.extend(_task_share_rows(native,draws if infer else None))
    if infer:
        bootstrap_results=parallel_bootstrap_pairs(bootstrap_tasks,worker_count=worker_count,cache_dir=bootstrap_cache_dir,progress_path=bootstrap_progress_path)
        for key,boot in bootstrap_results.items():
            row=bootstrap_rows[key]
            for prefix,interval in (("bootstrap_ag",boot["candidate_ag_interval"]),("bootstrap_oc",boot["candidate_oc_interval"]),("bootstrap_delta_ag",boot["delta_ag_interval"]),("bootstrap_delta_oc",boot["delta_oc_interval"])):
                if interval is not None: row[prefix+"_lower"],row[prefix+"_upper"]=interval
            row["failed_sst_draw_count"]=boot["failed_sst_draw_count"]
        signs_by_endpoint={}
        for _,_,_,endpoint,cluster_count in sign_flip_pending:
            if endpoint not in signs_by_endpoint:
                signs=_endpoint_signs(endpoint,cluster_count); signs_by_endpoint[endpoint]=signs
                receipts.append({"endpoint_id":endpoint,"operation_id":"sign_flip","seed":derive_endpoint_seed(endpoint,"sign_flip"),"resamples":SIGN_FLIP_RESAMPLES,"cluster_count":cluster_count,"draw_sha256":hashlib.sha256(np.ascontiguousarray(signs).tobytes()).hexdigest()})
        pending_by_endpoint: dict[str,list[tuple[dict[str,object],Sequence[AlignmentObservation],Sequence[AlignmentObservation]]]] = defaultdict(list)
        for row,reference_observations,candidate_observations,endpoint,_ in sign_flip_pending:
            pending_by_endpoint[endpoint].append((row,reference_observations,candidate_observations))
        for endpoint,items in pending_by_endpoint.items():
            contribution_rows=[]; aggregations=[]; owners=[]
            for row,reference_observations,candidate_observations in items:
                comparison=compare_alignment(reference_observations,candidate_observations)
                contribution_rows.extend((np.asarray([value.value for value in comparison.ag_contribution_differences]),np.asarray([value.value for value in comparison.acc_contribution_differences])))
                aggregations.extend(("sum","mean")); owners.append(row)
            sign_result=endpoint_batched_sign_flip(
                endpoint, np.vstack(contribution_rows), aggregations=aggregations, worker_count=worker_count,
                cache_dir=sign_flip_cache_dir,
                progress_path=None if sign_flip_progress_dir is None else sign_flip_progress_dir/f"{endpoint}.json",
            )
            p_values=sign_result["raw_p"]
            for index,row in enumerate(owners):
                row["raw_p_ag"],row["raw_p_oc"]=p_values[2*index],p_values[2*index+1]
        hypotheses=[]
        for row in summary:
            for outcome in ("ag","oc"):
                raw=row[f"raw_p_{outcome}"]; family=row[f"family_id_{outcome}"]
                if family is not None: hypotheses.append({"hypothesis_id":f"{row['panel_id']}|{row['analysis_id']}|{row['metric_output_id']}|{outcome}","family_id":family,"raw_p":raw,"state":row[f"{outcome}_state"]})
        apply_holm_families(hypotheses,family_sizes={"w1_axis_primary_66":66,"other_metrics_secondary_726":726})
        by_id={item["hypothesis_id"]:item for item in hypotheses}
        for row in summary:
            for outcome in ("ag","oc"):
                key=f"{row['panel_id']}|{row['analysis_id']}|{row['metric_output_id']}|{outcome}"
                if key in by_id:
                    item=by_id[key];row[f"adjusted_p_{outcome}"]=item["adjusted_p"];row[f"p_for_adjustment_{outcome}"]=item["p_for_adjustment"];row[f"family_size_{outcome}"]=item["family_size"];row[f"holm_rank_{outcome}"]=item["holm_rank"]
    return {"alignment_summary":summary,"oc_pair_summary":pairs,"oc_group_summary":groups,"ag_family_decomposition":decompositions,"task_harm_summary":harm,"task_harm_shares":harm_shares,"bootstrap_receipts":receipts,"protocol_interactions":_interaction_rows(inputs,comparisons,infer,bootstrap_results),"bridge_comparisons":_bridge_rows(inputs,comparisons,infer,bootstrap_results),"elapsed_seconds":time.monotonic()-started}


def _require_shared_draws(left: tuple[NativePanelTable, object, np.ndarray], right: tuple[NativePanelTable, object, np.ndarray]) -> None:
    """Reject a pairing that cannot represent one endpoint-level resample."""
    left_table, _, left_draws = left
    right_table, _, right_draws = right
    if left_table.cluster_ids != right_table.cluster_ids:
        raise W1AxisStatisticsError("cluster mismatch for paired bootstrap")
    if not np.array_equal(left_draws, right_draws):
        raise W1AxisStatisticsError("independent draw tables forbidden for paired bootstrap")


def _paired_interval(left: np.ndarray, right: np.ndarray, *, sign: float = 1.0) -> tuple[float | None, float | None]:
    """Interval `sign * (right - left)` directly on shared bootstrap rows."""
    values = sign * (right - left)
    if not np.isfinite(values).all():
        return None, None
    return _quantile(values)


def _interaction_rows(inputs: StatisticInputs | None, comparisons: Mapping[tuple[str,str,str],object], infer: bool, bootstrap_results: Mapping[str, Mapping[str, object]] | None = None) -> list[dict[str,object]]:
    rows=[]
    endpoints=("d2_5","d2_10","d2_20","d4","d5")
    for endpoint in endpoints:
        a=f"{endpoint}_a";b=f"{endpoint}_b"
        for analysis_id,representation,scope in _analysis_specs():
            for outcome in ("ag","oc"):
                a_w1=comparisons.get((a,analysis_id,W1),(None,None,None)); b_w1=comparisons.get((b,analysis_id,W1),(None,None,None))
                av=a_w1[1];bv=b_w1[1]
                unavailable=av is None or bv is None
                amse=comparisons.get((a,analysis_id,MSE),(None,None,None))[1]; bmse=comparisons.get((b,analysis_id,MSE),(None,None,None))[1]
                unavailable=unavailable or amse is None or bmse is None
                # Match the column-level convention: AG is MSE-W1 while OC is W1-MSE.
                delta_a=None if unavailable else ((amse["ag"]-av["ag"]) if outcome=="ag" else (av["oc"]-amse["oc"]))
                delta_b=None if unavailable else ((bmse["ag"]-bv["ag"]) if outcome=="ag" else (bv["oc"]-bmse["oc"]))
                lower=upper=None
                if infer and not unavailable:
                    _require_shared_draws(a_w1,b_w1)
                    if bootstrap_results is None:
                        raise W1AxisStatisticsError("missing paired bootstrap results")
                    a_values=np.asarray(bootstrap_results[f"{a}|{analysis_id}|{W1}"]["values"],dtype=float)
                    b_values=np.asarray(bootstrap_results[f"{b}|{analysis_id}|{W1}"]["values"],dtype=float)
                    # Bootstrap columns are MSE-W1 for AG and W1-MSE for OC.
                    column,sign=(2,1.0) if outcome=="ag" else (5,1.0)
                    lower,upper=_paired_interval(a_values[:,column],b_values[:,column],sign=sign)
                rows.append({"endpoint_id":endpoint,"analysis_id":analysis_id,"representation":representation,"perturbation_scope":scope,"metric_output_id":W1,"outcome":outcome,"state":"unavailable" if unavailable else "complete","delta_a":delta_a,"delta_b":delta_b,"interaction":None if unavailable else delta_b-delta_a,"bootstrap_lower":lower,"bootstrap_upper":upper,"shared_endpoint_draws":True})
    return rows


def _bridge_rows(inputs: StatisticInputs, comparisons: Mapping[tuple[str,str,str],object], infer: bool, bootstrap_results: Mapping[str, Mapping[str, object]] | None = None) -> list[dict[str,object]]:
    rows=[]
    for panel_id in inputs.native_tables:
        for scope in ("all5","no_axis"):
            native_entry=comparisons[(panel_id,f"native_{scope}",W1)]; common_entry=comparisons[(panel_id,f"common_grid_{scope}",MSE)]
            native=native_entry[1]; common=common_entry[1]; ag_lower=ag_upper=oc_lower=oc_upper=None
            _require_shared_draws(native_entry,common_entry)
            if infer:
                if bootstrap_results is None:
                    raise W1AxisStatisticsError("missing paired bootstrap results")
                try:
                    native_values=np.asarray(bootstrap_results[f"{panel_id}|native_{scope}|{W1}"]["values"],dtype=float)
                    common_values=np.asarray(bootstrap_results[f"{panel_id}|common_grid_{scope}|{MSE}"]["values"],dtype=float)
                except KeyError as error:
                    raise W1AxisStatisticsError("missing bridge bootstrap result") from error
                ag_lower,ag_upper=_paired_interval(native_values[:,1],common_values[:,1])
                oc_lower,oc_upper=_paired_interval(common_values[:,4],native_values[:,4])
            rows.append({"panel_id":panel_id,"endpoint_id":inputs.native_tables[panel_id].endpoint_id,"perturbation_scope":scope,"reference":"native_w1","candidate":"common_grid_mse","state":"complete","native_w1_ag":native["ag"],"common_grid_mse_ag":common["ag"],"delta_ag":common["ag"]-native["ag"],"native_w1_oc":native["oc"],"common_grid_mse_oc":common["oc"],"delta_oc":native["oc"]-common["oc"],"bootstrap_ag_lower":ag_lower,"bootstrap_ag_upper":ag_upper,"bootstrap_oc_lower":oc_lower,"bootstrap_oc_upper":oc_upper})
    return rows


_FIELDS={
    "alignment_summary":("panel_id","endpoint_id","protocol_id","analysis_id","representation","perturbation_scope","metric_output_id","ag_state","oc_state","ag","oc","ag_sst","ag_sse_pooled","ag_sse_separate","ag_r2_pooled","ag_r2_separate","delta_ag","delta_oc","cluster_count","condition_count","historical_ag_lower","historical_ag_upper","historical_oc_lower","historical_oc_upper","historical_raw_p_ag","historical_raw_p_oc","bootstrap_ag_lower","bootstrap_ag_upper","bootstrap_oc_lower","bootstrap_oc_upper","bootstrap_delta_ag_lower","bootstrap_delta_ag_upper","bootstrap_delta_oc_lower","bootstrap_delta_oc_upper","raw_p_ag","raw_p_oc","p_for_adjustment_ag","p_for_adjustment_oc","adjusted_p_ag","adjusted_p_oc","family_id_ag","family_id_oc","family_size_ag","family_size_oc","holm_rank_ag","holm_rank_oc","failed_sst_draw_count"),
    "oc_pair_summary":("panel_id","endpoint_id","protocol_id","representation","metric_output_id","left_family","right_family","state","pair_count_per_cluster","pair_count_total","oc","strict_agreement_count","strict_incorrect_count","metric_tie_count","downstream_tie_count","double_tie_count","bootstrap_lower","bootstrap_upper"),
    "oc_group_summary":("panel_id","endpoint_id","protocol_id","representation","metric_output_id","group","state","oc","bootstrap_lower","bootstrap_upper","pair_count_per_cluster"),
    "ag_family_decomposition":("panel_id","endpoint_id","protocol_id","representation","perturbation_scope","metric_output_id","family","state","sst","pooled_sse","separate_sse","pooled_r2","separate_r2","d_p","zero_metric_harm_proportion","mean_downstream","pooled_prediction_at_zero","n","expected_constant_sse","actual_raw_residual_improvement","identity_error"),
    "task_harm_summary":("panel_id","endpoint_id","protocol_id","family","alpha","scope","signed_mean","positive_mean","negative_mean","absolute_mean","signed_mean_lower","signed_mean_upper","positive_mean_lower","positive_mean_upper","negative_mean_lower","negative_mean_upper","absolute_mean_lower","absolute_mean_upper","positive_absolute_share","negative_absolute_share","zero_denominator_state"),
    "task_harm_shares":("panel_id","endpoint_id","protocol_id","component","positive_harm_share","positive_harm_share_lower","positive_harm_share_upper","positive_denominator_state","absolute_harm_share","absolute_harm_share_lower","absolute_harm_share_upper","absolute_denominator_state"),
    "protocol_interactions":("endpoint_id","analysis_id","representation","perturbation_scope","metric_output_id","outcome","state","delta_a","delta_b","interaction","bootstrap_lower","bootstrap_upper","shared_endpoint_draws"),
    "bridge_comparisons":("panel_id","endpoint_id","perturbation_scope","reference","candidate","state","native_w1_ag","common_grid_mse_ag","delta_ag","native_w1_oc","common_grid_mse_oc","delta_oc","bootstrap_ag_lower","bootstrap_ag_upper","bootstrap_oc_lower","bootstrap_oc_upper"),
    "bootstrap_receipts":("endpoint_id","operation_id","seed","resamples","cluster_count","draw_sha256","duplicate_draw_count"),
}


def build_statistics_run(*, config_path: Path = DEFAULT_CONFIG, common_grid_artifact: Path = DEFAULT_COMMON_GRID_ARTIFACT, output_root: Path | None = None, stage: str = "infer", resume: bool = False, worker_count: int = 1) -> Path:
    if stage not in {"summarize","infer","all"}: raise W1AxisStatisticsError("statistics stage must be summarize, infer, or all")
    config=load_w1_axis_robustness_config(config_path); inputs=load_statistic_inputs(config,common_grid_artifact)
    actual_stage="infer" if stage=="all" else stage; root=Path(output_root or config.default_output_root)/"statistics"
    if worker_count <= 0: raise W1AxisStatisticsError("worker_count must be positive")
    identity={"stage":actual_stage,"common_grid_sha256":inputs.common_grid_sha256,"config_sha256":hashlib.sha256(config.path.read_bytes()).hexdigest(),"statistics_code_sha256":_statistics_code_sha256(),"bootstrap_algorithm":"cluster-bootstrap-axis-order-v3-oc-independent-sst","task_harm_algorithm":"transform-before-alpha-average-v3-cross-family-shares","bootstrap_resamples":BOOTSTRAP_RESAMPLES,"sign_flip_resamples":SIGN_FLIP_RESAMPLES,"worker_count":worker_count}
    run_id="w1-axis-statistics-"+hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:16]; path=root/run_id
    if path.exists():
        if resume: return path
        raise W1AxisStatisticsError("append-only statistics run path already exists")
    root.mkdir(parents=True,exist_ok=True); staging_parent=Path(tempfile.mkdtemp(prefix=f".{run_id}.staging-",dir=root)); staging=staging_parent/run_id; staging.mkdir()
    try:
        # A new source-identified run owns fresh caches.  This avoids both
        # consuming and mixing artifacts made by an earlier implementation.
        cache_dir=root/f".{run_id}.bootstrap-cache" if actual_stage=="infer" else None
        progress_path=root/f".{run_id}.infer-progress.json" if actual_stage=="infer" else None
        sign_flip_cache_dir=root/f".{run_id}.sign-flip-cache" if actual_stage=="infer" else None
        sign_flip_progress_dir=root/f".{run_id}.sign-flip-progress" if actual_stage=="infer" else None
        result=summarize_statistics(inputs,infer=actual_stage=="infer",worker_count=worker_count,bootstrap_cache_dir=cache_dir,bootstrap_progress_path=progress_path,sign_flip_cache_dir=sign_flip_cache_dir,sign_flip_progress_dir=sign_flip_progress_dir); payloads={}
        for name,fields in _FIELDS.items(): payloads[name+".csv"]=_csv_bytes(result[name],fields)
        checksums={name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()}
        checks={"alignment_summary_rows":len(result["alignment_summary"]),"oc_pair_summary_rows":len(result["oc_pair_summary"]),"ag_family_decomposition_rows":len(result["ag_family_decomposition"]),"task_harm_shares_rows":len(result["task_harm_shares"]),"protocol_interactions_rows":len(result["protocol_interactions"]),"bridge_comparisons_rows":len(result["bridge_comparisons"]),"d5_common_w1_unavailable_rows":sum(r["ag_state"]=="unavailable" for r in result["alignment_summary"])}
        expected={"alignment_summary_rows":572,"oc_pair_summary_rows":2860,"ag_family_decomposition_rows":2288,"task_harm_shares_rows":66,"protocol_interactions_rows":40,"bridge_comparisons_rows":22}
        if any(checks[key]!=value for key,value in expected.items()): raise W1AxisStatisticsError(f"output grid mismatch: {checks}")
        manifest={"schema_version":"w1-axis-statistics-v3","statistics_algorithm_version":"review-corrections-v3","bootstrap_algorithm":identity["bootstrap_algorithm"],"task_harm_algorithm":identity["task_harm_algorithm"],"statistics_code_sha256":identity["statistics_code_sha256"],"run_id":run_id,"stage":actual_stage,"status":"complete","common_grid_artifact":str(common_grid_artifact),"common_grid_sha256":inputs.common_grid_sha256,"bootstrap_resamples":BOOTSTRAP_RESAMPLES,"sign_flip_resamples":SIGN_FLIP_RESAMPLES,"worker_count":worker_count,"elapsed_seconds":result["elapsed_seconds"],"checks":checks,"checksums":checksums}
        payloads["manifest.json"]=_canonical_bytes(manifest); payloads[f"{actual_stage}.complete.json"]=_canonical_bytes({"status":"complete","stage":actual_stage})
        for name,raw in payloads.items(): (staging/name).write_bytes(raw)
        staging.rename(path)
        if progress_path is not None:
            progress_path.write_bytes(_canonical_bytes({"status":"complete"}))
    except Exception:
        shutil.rmtree(staging_parent,ignore_errors=True); raise
    shutil.rmtree(staging_parent,ignore_errors=True); return path
