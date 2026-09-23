from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import struct
import tempfile
from collections import defaultdict
from dataclasses import asdict, asdict as dataclass_asdict, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from rpe.alignment import (
    AlignmentObservation,
    AlignmentValidationError,
    alignment_gap,
    bulk_paired_cluster_bootstrap,
    compare_alignment,
    cross_perturbation_accuracy,
    holm_step_down,
    orient_harm,
    paired_contribution_sign_flip,
)
from rpe.downstream.rruff import (
    D5LibraryQuerySplit,
    D5RawCohort,
    load_d5_native_spectra,
    load_d5_raw_cohort,
)
from rpe.downstream.rruff_matching import D5MatchingResult, match_d5_protocol_a_values
from rpe.evaluation import PeakPairInput, PreferredDirection, SingleSpectrumInput, Spectrum1D, SpectrumPairInput, evaluate_metric
from rpe.methods.catalog import Phase3ClassicalCatalog, Phase3System, load_classical_catalog
from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system
from rpe.metrics import ISLikeStructureToNoiseMetric, MAEMetric, MSEMetric, NMSEMetric, PeakDetectionCurvesMetric, PearsonRMetric, RMSEMetric, SAMMetric, Wasserstein1Metric
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d5_protocol_a import (
    ALPHA_GRID,
    ARTIFACT_PAYLOAD_FILES,
    ARTIFACT_SCHEMA_VERSION,
    CANDIDATE_OUTPUT_IDS,
    CATALOG_RELATIVE_PATH,
    CODE_RELATIVE_PATHS,
    CONDITION_IDS,
    CONFIG_AUTHORITY_RELATIVE_PATH,
    CONFIG_RELATIVE_PATH,
    CWT_SYSTEM_ID,
    D5_CONFIG_RELATIVE_PATH,
    DATASET_RELATIVE_PATH,
    EXPERIMENT_ID,
    METRIC_DIRECTIONS,
    METRIC_OUTPUT_IDS,
    PERTURBATION_IDS,
    PHASE1_CONFIG_RELATIVE_PATH,
    POSITIVE_ALPHAS,
    ROOT,
    RUN_PREFIX,
    STRUCTURE_OUTPUT_IDS,
    SWEEP_RELATIVE_PATH,
    Phase4D5ProtocolAConfig,
    Phase4D5ProtocolAError,
    Phase4D5ProtocolASummary,
    parse_phase4_d5_protocol_a_config,
)


def _json_ready(value: object) -> object:
    if is_dataclass(value):
        return _json_ready(dataclass_asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D5ProtocolAError("json", f"unsupported value {type(value).__name__}")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _positive_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase4D5ProtocolAError(path, "must be a positive integer")
    return value


def _environment() -> dict[str, object]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _code_document() -> dict[str, object]:
    return {
        relative: {"bytes": (ROOT / relative).stat().st_size, "sha256": _sha_file(ROOT / relative)}
        for relative in CODE_RELATIVE_PATHS
    }


def _authority_document() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _grid(config: Phase4D5ProtocolAConfig) -> np.ndarray:
    values = np.arange(
        config.support_start_cm1,
        config.support_stop_cm1 + config.support_step_cm1 / 2,
        config.support_step_cm1,
        dtype="<f8",
    )
    if values.size != config.support_point_count:
        raise Phase4D5ProtocolAError("support grid", "point count mismatch")
    return values


def _project(spectrum: Spectrum1D, grid: np.ndarray, max_gap: float) -> np.ndarray:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5ProtocolAError("downstream support", "extrapolation required")
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    support = axis[left : right + 1]
    if left < 0 or right >= axis.size or support.size < 2 or float(np.max(np.diff(support))) > max_gap:
        raise Phase4D5ProtocolAError("downstream support", "invalid bounds or gap")
    output = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(output).all() or float(np.linalg.norm(output.astype(np.float64))) <= 0:
        raise Phase4D5ProtocolAError("downstream projection", "nonfinite or zero norm")
    return output


def match_d5_protocol_a_799(
    cohort: D5RawCohort,
    split: D5LibraryQuerySplit,
    *,
    condition_id: str,
    query_record_ids: tuple[str, ...],
    library_record_ids: tuple[str, ...],
    query_values: np.ndarray,
    library_values: np.ndarray,
) -> D5MatchingResult:
    query = np.asarray(query_values)
    library = np.asarray(library_values)
    if query.ndim != 2 or library.ndim != 2 or query.shape[1] != 799 or library.shape[1] != 799:
        raise Phase4D5ProtocolAError("799 matcher", "must use exactly 799 features")
    return match_d5_protocol_a_values(
        cohort,
        split,
        condition_id=condition_id,
        query_record_ids=query_record_ids,
        library_record_ids=library_record_ids,
        query_values=query,
        library_values=library,
    )


def metric_preflight_states(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, object]]:
    if tuple(counts) != METRIC_OUTPUT_IDS:
        raise Phase4D5ProtocolAError("metric preflight", "must follow exact manifest")
    states: dict[str, dict[str, object]] = {}
    for output_id in METRIC_OUTPUT_IDS:
        value = counts[output_id]
        complete = int(value["complete"])
        planned = int(value["planned"])
        state = "complete" if complete == planned else "not_evaluable_incomplete_grid"
        states[output_id] = {"complete": complete, "planned": planned, "state": state}
    if states["mse"]["state"] != "complete":
        raise Phase4D5ProtocolAError("MSE comparator", "must be complete on the full grid")
    return states


def fixed_holm_family(
    observed_p_values: Mapping[str, float],
    metric_states: Mapping[str, Mapping[str, object]],
    observed_contrasts: Mapping[str, float],
) -> list[dict[str, object]]:
    multiplicity: dict[str, float] = {}
    state_by_hypothesis: dict[str, str] = {}
    for metric_id in CANDIDATE_OUTPUT_IDS:
        eligible = metric_states[metric_id]["state"] == "complete"
        for statistic in ("d_ag", "d_acc"):
            hypothesis = f"{metric_id}:{statistic}"
            if eligible and hypothesis in observed_p_values:
                multiplicity[hypothesis] = float(observed_p_values[hypothesis])
                state_by_hypothesis[hypothesis] = "tested"
            else:
                multiplicity[hypothesis] = 1.0
                state_by_hypothesis[hypothesis] = "not_tested_metric_incomplete"
    adjusted = {row.hypothesis_id: row for row in holm_step_down(multiplicity, alpha=0.05)}
    output = []
    for hypothesis in sorted(multiplicity):
        tested = state_by_hypothesis[hypothesis] == "tested"
        if tested and hypothesis not in observed_contrasts:
            raise Phase4D5ProtocolAError("Holm family", f"missing observed contrast for {hypothesis}")
        contrast = None if not tested else float(observed_contrasts[hypothesis])
        if contrast is not None and not np.isfinite(contrast):
            raise Phase4D5ProtocolAError("Holm family", f"nonfinite observed contrast for {hypothesis}")
        favorable = None if contrast is None else contrast > 0.0
        output.append({
            "metric_output_id": hypothesis.rsplit(":", 1)[0],
            "statistic": hypothesis.rsplit(":", 1)[1],
            "hypothesis_id": hypothesis,
            "state": state_by_hypothesis[hypothesis],
            "raw_p_value": observed_p_values.get(hypothesis),
            "multiplicity_p_value": multiplicity[hypothesis],
            "adjusted_p_value": adjusted[hypothesis].adjusted_p_value,
            "rank": adjusted[hypothesis].rank,
            "family_size": adjusted[hypothesis].family_size,
            "observed_contrast": contrast,
            "favorable": favorable,
            "rejected": adjusted[hypothesis].rejected and tested and bool(favorable),
        })
    return output


def aggregate_class_observations(
    occurrence_rows: Sequence[Mapping[str, object]],
    metric_rows: Mapping[tuple[str, str, str], float],
    *,
    metric_output_id: str,
    preferred_direction: str,
    positive_conditions: Sequence[str],
) -> list[dict[str, object]]:
    direction = PreferredDirection(preferred_direction)
    by_class_condition: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in occurrence_rows:
        by_class_condition[(int(row["class_label"]), str(row["condition_id"]))].append(row)
    classes = sorted({key[0] for key in by_class_condition})
    output = []
    for class_label in classes:
        baseline_rows = by_class_condition[(class_label, "alpha0")]
        baseline_error = float(np.mean([not bool(row["top1_correct"]) for row in baseline_rows]))
        for condition_id in positive_conditions:
            rows = by_class_condition[(class_label, condition_id)]
            downstream_error = float(np.mean([not bool(row["top1_correct"]) for row in rows]))
            metric_harms = [
                orient_harm(
                    metric_rows[(str(row["record_id"]), "alpha0", metric_output_id)],
                    metric_rows[(str(row["record_id"]), condition_id, metric_output_id)],
                    direction,
                )
                for row in rows
            ]
            perturbation_id, alpha_hex = condition_id.split(":", 1)
            output.append(
                {
                    "cluster_id": str(class_label),
                    "perturbation_id": perturbation_id,
                    "alpha": struct.unpack("<d", bytes.fromhex(alpha_hex))[0],
                    "metric_output_id": metric_output_id,
                    "metric_harm": float(np.mean(metric_harms)),
                    "downstream_harm": downstream_error - baseline_error,
                    "occurrence_count": len(rows),
                    "state": "complete",
                }
            )
    return output


def _padding(values: Sequence[float]) -> tuple[float, float]:
    minimum = min(0.0, *values)
    maximum = max(0.0, *values)
    width = maximum - minimum
    pad = 0.05 * (width if width > 0 else max(1.0, abs(minimum), abs(maximum)))
    return minimum - pad, maximum + pad


def render_figure_payloads(
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    if len(figure1_rows) != 520 or len(figure2_rows) != 13:
        raise Phase4D5ProtocolAError("figure source", "must contain 520 and 13 rows")
    figure1_fields = tuple(figure1_rows[0])
    figure2_fields = tuple(figure2_rows[0])
    f1_csv = _csv_bytes(figure1_rows, figure1_fields)
    f2_csv = _csv_bytes(figure2_rows, figure2_fields)
    rc = {
        "font.family": "DejaVu Sans",
        "svg.hashsalt": "rpe-phase4-d5-v1",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharey=True)
        y_values = [float(row["mean_downstream_harm"]) for row in figure1_rows if row["mean_downstream_harm"] is not None]
        y_lim = _padding(y_values)
        for metric_index, metric_id in enumerate(METRIC_OUTPUT_IDS):
            ax = axes.flat[metric_index]
            selected_metric = [row for row in figure1_rows if row["metric_output_id"] == metric_id]
            for perturbation_id in PERTURBATION_IDS:
                selected = [row for row in selected_metric if row["perturbation_id"] == perturbation_id]
                if selected and selected[0]["metric_state"] == "complete":
                    ax.plot(
                        [float(row["mean_metric_harm"]) for row in selected],
                        [float(row["mean_downstream_harm"]) for row in selected],
                        marker="o",
                        linewidth=1.5,
                        color=dict(zip(PERTURBATION_IDS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))[perturbation_id],
                        label=perturbation_id.upper(),
                    )
            x_values = [float(row["mean_metric_harm"]) for row in selected_metric if row["metric_state"] == "complete"]
            if x_values:
                ax.set_xlim(*_padding(x_values))
            ax.set_ylim(*y_lim)
            ax.set_title(metric_id)
            ax.axhline(0.0, color="#999999", linewidth=0.5)
            ax.axvline(0.0, color="#999999", linewidth=0.5)
        for index in range(len(METRIC_OUTPUT_IDS), 16):
            axes.flat[index].axis("off")
        axes.flat[0].legend(loc="best", fontsize=7)
        fig.suptitle("D5 / Protocol A / full_domain_core / P8–P12")
        fig.tight_layout()
        f1_png = io.BytesIO()
        f1_svg = io.BytesIO()
        fig.savefig(f1_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"})
        fig.savefig(f1_svg, format="svg", metadata={"Date": None})
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        y = np.arange(len(METRIC_OUTPUT_IDS))
        specs = (
            ("ag", "ag_lower", "ag_upper", "AG"),
            ("acc", "acc_lower", "acc_upper", "Acc-cross"),
            ("d_ag", "d_ag_lower", "d_ag_upper", "D_AG"),
            ("d_acc", "d_acc_lower", "d_acc_upper", "D_Acc"),
        )
        for ax, (key, low_key, high_key, title) in zip(axes, specs, strict=True):
            finite = []
            for index, row in enumerate(figure2_rows):
                value = row.get(key)
                if value is None:
                    if row.get("metric_state") != "complete":
                        ax.scatter(0.0, index, marker="x", color="#777777", s=12)
                        ax.text(0.02, index, str(row["metric_state"]), transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=4.5, color="#777777", clip_on=True)
                    continue
                value = float(value)
                low = float(row[low_key])
                high = float(row[high_key])
                finite.extend((low, value, high))
                ax.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color="#1f77b4")
                if key in {"d_ag", "d_acc"}:
                    raw = row[f"{key}_raw_p"]
                    adjusted_p = row[f"{key}_adjusted_p"]
                    direction = "favorable" if bool(row[f"{key}_favorable"]) else "unfavorable"
                    label = f"raw={float(raw):.6g}; adj={float(adjusted_p):.6g}; {direction}"
                    ax.text(0.02, index, label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=4.5, color="#333333", clip_on=True)
            if finite:
                ax.set_xlim(*_padding(finite))
            ax.axvline(0.0, color="#999999", linewidth=0.5)
            ax.set_title(title)
        axes[0].set_yticks(y, METRIC_OUTPUT_IDS)
        fig.suptitle("D5 / Protocol A / full_domain_core / secondary")
        fig.tight_layout()
        f2_png = io.BytesIO()
        f2_svg = io.BytesIO()
        fig.savefig(f2_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"})
        fig.savefig(f2_svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    return {
        "figure1_d5_protocol_a_full_domain.png": f1_png.getvalue(),
        "figure1_d5_protocol_a_full_domain.svg": f1_svg.getvalue(),
        "figure1_d5_protocol_a_full_domain_data.csv": f1_csv,
        "figure2_d5_protocol_a_full_domain.png": f2_png.getvalue(),
        "figure2_d5_protocol_a_full_domain.svg": f2_svg.getvalue(),
        "figure2_d5_protocol_a_full_domain_data.csv": f2_csv,
    }


def _phase1_source(record_order: int, record_id: str, class_label: int, mineral_name: str, spectrum: Spectrum1D) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            record_order,
            record_id,
            spectrum.sample_id or record_id,
            class_label,
            mineral_name,
            f"native::{_array_sha(spectrum.axis_cm1)}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, "<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, "<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1),
        normalized_intensity_float64_sha256=_array_sha(spectrum.intensity),
        provenance=MappingProxyType(
            {
                "license": None,
                "license_status": "not_stated",
                "retrieved_date": "2026-08-19",
                "sha256": "0" * 64,
                "source_artifact": "rruff",
                "source_url": "https://rruff.info",
            }
        ),
    )


def _condition_id(perturbation_id: str, alpha: float) -> str:
    return f"{perturbation_id}:{struct.pack('<d', float(alpha)).hex()}"


def _metric_objects() -> dict[str, object]:
    return {
        "mse": MSEMetric(),
        "rmse": RMSEMetric(),
        "mae": MAEMetric(),
        "sam": SAMMetric(),
        "pearson_r": PearsonRMetric(),
        "nmse": NMSEMetric(),
        "wasserstein_1_cm1": Wasserstein1Metric(),
        "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
    }


def _resolve_cwt(catalog: Phase3ClassicalCatalog) -> Phase3System:
    matches = [system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D5ProtocolAError("CWT system", "must resolve exactly once")
    return matches[0]


def _run_identity(config: Phase4D5ProtocolAConfig, code: Mapping[str, object], environment: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": config.authorities,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config_authority": _authority_document(),
        "config_sha256": config.sha256,
        "cwt_system_id": config.cwt_system_id,
        "environment": environment,
        "figure_contract": config.figure_contract,
        "frozen_identities": config.frozen_identities,
        "inference": {
            "bootstrap_resamples": config.bootstrap_resamples,
            "sign_flip_resamples": config.sign_flip_resamples,
            "random_seed": config.random_seed,
            "holm_slots": config.expected_holm_slot_count,
        },
        "metric_output_ids": config.metric_output_ids,
        "protocol": "A",
        "tier": "full_domain_core",
    }
    return RUN_PREFIX + _sha_bytes(_canonical(identity)), identity


@threadpool_limits.wrap(limits=1, user_api="blas")
def _payloads_from_inputs(
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    classical_catalog: Phase3ClassicalCatalog,
    config: Phase4D5ProtocolAConfig,
    inference_resamples: int | None,
) -> tuple[dict[str, bytes], Phase4D5ProtocolASummary]:
    if not config.synthetic_fixture and inference_resamples is not None:
        raise Phase4D5ProtocolAError("inference_resamples", "test override forbidden for frozen run")
    if tuple(sweep.alpha_grid) != ALPHA_GRID or sweep.sha256 != config.authorities.get("sweep_sha256"):
        raise Phase4D5ProtocolAError("sweep", "identity mismatch")
    if phase1_config.file_sha256 != config.authorities.get("phase1_core_config_sha256"):
        raise Phase4D5ProtocolAError("phase1 config", "identity mismatch")
    if len(cohort.record_ids) != config.cohort_record_count or len(native_spectra) != len(cohort.record_ids):
        raise Phase4D5ProtocolAError("cohort", "count mismatch")
    expected_spectrum_ids = tuple(f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids)
    if tuple(value.spectrum_id for value in native_spectra) != expected_spectrum_ids:
        raise Phase4D5ProtocolAError("native spectra", "order mismatch")

    query_positions: dict[int, list[dict[str, int]]] = defaultdict(list)
    split_occurrences = []
    for split in cohort.splits:
        for query_order, raw_index in enumerate(split.query_indices):
            index = int(raw_index)
            query_positions[index].append({"split_seed": int(split.seed), "query_order": query_order})
            split_occurrences.append((int(split.seed), query_order, index))
    unique_indices = sorted(query_positions, key=lambda index: cohort.record_ids[index])
    if len(unique_indices) != config.query_record_count or len(split_occurrences) != config.query_occurrence_count:
        raise Phase4D5ProtocolAError("query ledgers", "count mismatch")

    grid = _grid(config)
    projected: dict[tuple[str, str], np.ndarray] = {}
    downstream_rows = []
    for index, spectrum in enumerate(native_spectra):
        values = _project(spectrum, grid, config.support_max_gap_cm1)
        record_id = cohort.record_ids[index]
        projected[(record_id, "alpha0")] = values
        downstream_rows.append(
            {"record_id": record_id, "condition_id": "alpha0", "values_sha256": _array_sha(values, "<f4")}
        )

    p10_admission = P10MemoryAdmission(64 * 2**30)
    operator_cells = []
    conditions: dict[tuple[str, str], Spectrum1D] = {}
    record_condition_rows = []
    for record_order, cohort_index in enumerate(unique_indices):
        source_spectrum = native_spectra[cohort_index]
        record_id = cohort.record_ids[cohort_index]
        source = _phase1_source(
            record_order,
            record_id,
            int(cohort.class_labels[cohort_index]),
            cohort.mineral_names[cohort_index],
            source_spectrum,
        )
        conditions[(record_id, "alpha0")] = source_spectrum
        record_condition_rows.append(
            {
                "record_order": record_order,
                "record_id": record_id,
                "condition_id": "alpha0",
                "source_spectrum_id": source_spectrum.spectrum_id,
                "output_spectrum_id": source_spectrum.spectrum_id,
                "axis_sha256": _array_sha(source_spectrum.axis_cm1),
                "intensity_sha256": _array_sha(source_spectrum.intensity),
                "projection_sha256": _array_sha(projected[(record_id, "alpha0")], "<f4"),
            }
        )
        alpha_zero_hashes = []
        for perturbation_id in PERTURBATION_IDS:
            cell = run_perturbation_cell(source, perturbation_id, phase1_config, sweep, p10_admission=p10_admission)
            if cell.status is not CellStatus.COMPLETE:
                raise Phase4D5ProtocolAError("operator cell", f"{record_id}/{perturbation_id} is not complete")
            outputs = []
            for item in cell.records:
                output = item.result.output
                outputs.append(
                    {
                        "alpha": item.result.alpha,
                        "alpha_float64_le_hex": item.alpha_float64_le_hex,
                        "output_spectrum_id": output.spectrum_id,
                        "axis_sha256": _array_sha(output.axis_cm1),
                        "intensity_sha256": _array_sha(output.intensity),
                        "diagnostics": _json_ready(item.result.diagnostics),
                    }
                )
                if item.result.alpha == 0.0:
                    alpha_zero_hashes.append((_array_sha(output.axis_cm1), _array_sha(output.intensity)))
                else:
                    condition_id = _condition_id(perturbation_id, item.result.alpha)
                    conditions[(record_id, condition_id)] = output
                    projected_values = _project(output, grid, config.support_max_gap_cm1)
                    projected[(record_id, condition_id)] = projected_values
                    record_condition_rows.append(
                        {
                            "record_order": record_order,
                            "record_id": record_id,
                            "condition_id": condition_id,
                            "source_spectrum_id": source_spectrum.spectrum_id,
                            "output_spectrum_id": output.spectrum_id,
                            "axis_sha256": _array_sha(output.axis_cm1),
                            "intensity_sha256": _array_sha(output.intensity),
                            "projection_sha256": _array_sha(projected_values, "<f4"),
                        }
                    )
                    downstream_rows.append(
                        {
                            "record_id": record_id,
                            "condition_id": condition_id,
                            "values_sha256": _array_sha(projected_values, "<f4"),
                        }
                    )
            operator_cells.append(
                {
                    "record_id": record_id,
                    "record_order": record_order,
                    "perturbation_id": perturbation_id,
                    "state_digest": cell.state.state_digest if cell.state else None,
                    "native_gate": _json_ready(cell.evidence.native_gate),
                    "outputs": outputs,
                }
            )
        source_hash = (_array_sha(source_spectrum.axis_cm1), _array_sha(source_spectrum.intensity))
        if any(value != source_hash for value in alpha_zero_hashes) or len(alpha_zero_hashes) != 5:
            raise Phase4D5ProtocolAError("alpha-zero collapse", "operator identities differ")
    if len(record_condition_rows) != config.expected_query_condition_count or len(downstream_rows) != config.expected_projected_row_count:
        raise Phase4D5ProtocolAError("condition counts", "mismatch")

    metric_values: dict[tuple[str, str, str], float] = {}
    metric_failures: dict[str, list[dict[str, object]]] = {output_id: [] for output_id in METRIC_OUTPUT_IDS}
    metric_objects = _metric_objects()
    metric_rows = []
    for record_order, cohort_index in enumerate(unique_indices):
        record_id = cohort.record_ids[cohort_index]
        source = native_spectra[cohort_index]
        for condition_id in CONDITION_IDS:
            candidate = conditions[(record_id, condition_id)]
            for output_id in METRIC_OUTPUT_IDS[:8]:
                try:
                    if output_id == "is_like_structure_to_noise":
                        result = evaluate_metric(metric_objects[output_id], SingleSpectrumInput(candidate))
                    else:
                        result = evaluate_metric(metric_objects[output_id], SpectrumPairInput(source, candidate))
                    scalar = next(value for value in result.outputs if value.output_id == output_id)
                    metric_values[(record_id, condition_id, output_id)] = float(scalar.value)
                    metric_rows.append(
                        {
                            "record_order": record_order,
                            "record_id": record_id,
                            "condition_id": condition_id,
                            "metric_output_id": output_id,
                            "state": "complete",
                            "value": float(scalar.value),
                            "diagnostics": _json_ready(result.diagnostics),
                            "exception": None,
                        }
                    )
                except Exception as error:
                    metric_failures[output_id].append(
                        {
                            "record_id": record_id,
                            "condition_id": condition_id,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    metric_rows.append(
                        {
                            "record_order": record_order,
                            "record_id": record_id,
                            "condition_id": condition_id,
                            "metric_output_id": output_id,
                            "state": "failed",
                            "value": None,
                            "diagnostics": {},
                            "exception": {"type": type(error).__name__, "message": str(error)},
                        }
                    )

    cwt_system = _resolve_cwt(classical_catalog)
    peak_metric = PeakDetectionCurvesMetric()
    peak_receipts = []
    for record_order, cohort_index in enumerate(unique_indices):
        record_id = cohort.record_ids[cohort_index]
        reference_result = run_peak_detection_system(cwt_system, native_spectra[cohort_index])
        for condition_id in CONDITION_IDS:
            result = reference_result if condition_id == "alpha0" else run_peak_detection_system(cwt_system, conditions[(record_id, condition_id)])
            successful = result.status in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}
            peak_receipts.append(
                {
                    "record_order": record_order,
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "status": result.status.value,
                    "peaks_sha256": result.peaks_sha256,
                    "peak_count": len(result.peaks),
                    "warnings": [asdict(value) for value in result.warnings],
                    "diagnostics": _json_ready(result.diagnostics),
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                }
            )
            if successful and reference_result.status in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}:
                try:
                    metric_result = evaluate_metric(
                        peak_metric,
                        PeakPairInput(
                            tuple(value.to_peak1d() for value in reference_result.peaks),
                            tuple(value.to_peak1d() for value in result.peaks),
                            2.0,
                            (0.0,),
                        ),
                    )
                    by_id = {value.output_id: value for value in metric_result.outputs}
                    for output_id in STRUCTURE_OUTPUT_IDS:
                        scalar = by_id[output_id]
                        metric_values[(record_id, condition_id, output_id)] = float(scalar.value)
                        metric_rows.append(
                            {
                                "record_order": record_order,
                                "record_id": record_id,
                                "condition_id": condition_id,
                                "metric_output_id": output_id,
                                "state": "complete",
                                "value": float(scalar.value),
                                "diagnostics": {},
                                "exception": None,
                            }
                        )
                except Exception as error:
                    for output_id in STRUCTURE_OUTPUT_IDS:
                        metric_failures[output_id].append(
                            {
                                "record_id": record_id,
                                "condition_id": condition_id,
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                        metric_rows.append(
                            {
                                "record_order": record_order,
                                "record_id": record_id,
                                "condition_id": condition_id,
                                "metric_output_id": output_id,
                                "state": "failed",
                                "value": None,
                                "diagnostics": {},
                                "exception": {"type": type(error).__name__, "message": str(error)},
                            }
                        )
            else:
                for output_id in STRUCTURE_OUTPUT_IDS:
                    metric_failures[output_id].append(
                        {
                            "record_id": record_id,
                            "condition_id": condition_id,
                            "type": "PeakRunStatus",
                            "message": result.status.value,
                        }
                    )
                    metric_rows.append(
                        {
                            "record_order": record_order,
                            "record_id": record_id,
                            "condition_id": condition_id,
                            "metric_output_id": output_id,
                            "state": "failed",
                            "value": None,
                            "diagnostics": {},
                            "exception": {"type": "PeakRunStatus", "message": result.status.value},
                        }
                    )

    planned_metric_rows = config.expected_query_condition_count
    counts = {
        output_id: {"complete": planned_metric_rows - len(metric_failures[output_id]), "planned": planned_metric_rows}
        for output_id in METRIC_OUTPUT_IDS
    }
    metric_states = metric_preflight_states(counts)
    condition_order = {condition_id: index for index, condition_id in enumerate(CONDITION_IDS)}
    metric_order = {output_id: index for index, output_id in enumerate(METRIC_OUTPUT_IDS)}
    metric_rows.sort(
        key=lambda row: (
            int(row["record_order"]),
            condition_order[str(row["condition_id"])],
            metric_order[str(row["metric_output_id"])],
        )
    )
    preflight = {
        "claim_boundary": "outcome_blind_preflight_no_predictions_correctness_scores_alignment_or_pvalues",
        "downstream_role_state": "complete",
        "metric_states": metric_states,
        "operator_cell_count": len(operator_cells),
        "record_condition_count": len(record_condition_rows),
        "projected_row_count": len(downstream_rows),
    }

    prediction_rows = []
    for split in cohort.splits:
        query_indices = tuple(int(value) for value in split.query_indices)
        library_indices = tuple(int(value) for value in split.library_indices)
        query_ids = tuple(cohort.record_ids[index] for index in query_indices)
        library_ids = tuple(cohort.record_ids[index] for index in library_indices)
        library = np.stack([projected[(record_id, "alpha0")] for record_id in library_ids])
        for condition_id in CONDITION_IDS:
            query = np.stack([projected[(record_id, condition_id)] for record_id in query_ids])
            result = match_d5_protocol_a_799(
                cohort,
                split,
                condition_id=condition_id,
                query_record_ids=query_ids,
                library_record_ids=library_ids,
                query_values=query,
                library_values=library,
            )
            for query_order, cohort_index in enumerate(query_indices):
                top_k = min(5, result.ranked_class_labels.shape[1])
                prediction_rows.append(
                    {
                        "split_seed": int(split.seed),
                        "split_sha256": split.split_sha256,
                        "query_order": query_order,
                        "cohort_index": cohort_index,
                        "record_id": cohort.record_ids[cohort_index],
                        "group_id": cohort.group_ids[cohort_index],
                        "class_label": int(cohort.class_labels[cohort_index]),
                        "condition_id": condition_id,
                        "top1_class_label": int(result.ranked_class_labels[query_order, 0]),
                        "top1_score": float(result.ranked_class_scores[query_order, 0]),
                        "top1_correct": bool(result.top1_correct[query_order]),
                        "top5_class_labels": [int(value) for value in result.ranked_class_labels[query_order, :top_k]],
                        "top5_scores": [float(value) for value in result.ranked_class_scores[query_order, :top_k]],
                        "top5_correct": bool(result.top5_correct[query_order]),
                    }
                )
    if len(prediction_rows) != config.expected_prediction_row_count:
        raise Phase4D5ProtocolAError("prediction rows", "count mismatch")

    class_rows = []
    eligible_metric_ids = [output_id for output_id in METRIC_OUTPUT_IDS if metric_states[output_id]["state"] == "complete"]
    for metric_id in METRIC_OUTPUT_IDS:
        if metric_id in eligible_metric_ids:
            current = aggregate_class_observations(
                prediction_rows,
                metric_values,
                metric_output_id=metric_id,
                preferred_direction=config.metric_directions[metric_id].value,
                positive_conditions=CONDITION_IDS[1:],
            )
            class_rows.extend(current)
        else:
            for class_label in sorted(set(int(value) for value in cohort.class_labels)):
                for condition_id in CONDITION_IDS[1:]:
                    perturbation_id, alpha_hex = condition_id.split(":", 1)
                    class_rows.append(
                        {
                            "cluster_id": str(class_label),
                            "perturbation_id": perturbation_id,
                            "alpha": struct.unpack("<d", bytes.fromhex(alpha_hex))[0],
                            "metric_output_id": metric_id,
                            "metric_harm": None,
                            "downstream_harm": None,
                            "occurrence_count": 0,
                            "state": "not_evaluable_incomplete_grid",
                        }
                    )

    by_metric: dict[str, list[AlignmentObservation]] = defaultdict(list)
    for row in class_rows:
        if row.get("metric_harm") is not None:
            by_metric[row["metric_output_id"]].append(
                AlignmentObservation(
                    row["cluster_id"],
                    row["perturbation_id"],
                    row["alpha"],
                    row["metric_harm"],
                    row["downstream_harm"],
                )
            )

    reference = tuple(by_metric["mse"])
    metric_results: dict[str, dict[str, object]] = {}
    bootstrap_rows = []
    sign_rows = []
    raw_p_values: dict[str, float] = {}
    observed_contrasts: dict[str, float] = {}
    effective_resamples = config.bootstrap_resamples if inference_resamples is None else inference_resamples
    effective_sign_flips = config.sign_flip_resamples if inference_resamples is None else max(64, inference_resamples)
    downstream_constant = False
    try:
        reference_gap = alignment_gap(reference)
        reference_acc = cross_perturbation_accuracy(reference)
        mse_intervals = bulk_paired_cluster_bootstrap(
            reference,
            reference,
            resamples=effective_resamples,
            confidence_level=config.confidence_level,
            random_seed=config.random_seed,
        )
        metric_results["mse"] = {
            "metric_output_id": "mse",
            "metric_state": "complete",
            "ag": reference_gap.alignment_gap,
            "acc": reference_acc.accuracy,
            "comparison": None,
        }
    except AlignmentValidationError as error:
        if error.path != "constant downstream":
            raise
        downstream_constant = True
        mse_intervals = None
        metric_results["mse"] = {
            "metric_output_id": "mse",
            "metric_state": "not_evaluable_constant_downstream",
            "ag": None,
            "acc": None,
            "comparison": None,
        }
    for metric_id in CANDIDATE_OUTPUT_IDS:
        if metric_id not in by_metric:
            metric_results[metric_id] = {
                "metric_output_id": metric_id,
                "metric_state": metric_states[metric_id]["state"],
                "ag": None,
                "acc": None,
                "comparison": None,
            }
            continue
        if downstream_constant:
            metric_results[metric_id] = {
                "metric_output_id": metric_id,
                "metric_state": "not_evaluable_constant_downstream",
                "ag": None,
                "acc": None,
                "comparison": None,
            }
            continue
        candidate = tuple(by_metric[metric_id])
        comparison = compare_alignment(reference, candidate)
        bootstrap = bulk_paired_cluster_bootstrap(
            reference,
            candidate,
            resamples=effective_resamples,
            confidence_level=config.confidence_level,
            random_seed=config.random_seed,
        )
        if mse_intervals is None:
            mse_intervals = bootstrap
        ag_flip = paired_contribution_sign_flip(
            tuple(value.value for value in comparison.ag_contribution_differences),
            aggregation="sum",
            resamples=effective_sign_flips,
            random_seed=config.random_seed,
        )
        acc_flip = paired_contribution_sign_flip(
            tuple(value.value for value in comparison.acc_contribution_differences),
            aggregation="mean",
            resamples=effective_sign_flips,
            random_seed=config.random_seed,
        )
        raw_p_values[f"{metric_id}:d_ag"] = ag_flip.p_value
        raw_p_values[f"{metric_id}:d_acc"] = acc_flip.p_value
        observed_contrasts[f"{metric_id}:d_ag"] = comparison.d_ag
        observed_contrasts[f"{metric_id}:d_acc"] = comparison.d_acc
        metric_results[metric_id] = {
            "metric_output_id": metric_id,
            "metric_state": "complete",
            "ag": comparison.candidate_gap.alignment_gap,
            "acc": comparison.candidate_accuracy.accuracy,
            "comparison": _json_ready(comparison),
        }
        bootstrap_rows.append({"metric_output_id": metric_id, **_json_ready(bootstrap)})
        sign_rows.extend(
            (
                {"metric_output_id": metric_id, "statistic": "d_ag", **_json_ready(ag_flip)},
                {"metric_output_id": metric_id, "statistic": "d_acc", **_json_ready(acc_flip)},
            )
        )

    family = fixed_holm_family(raw_p_values, metric_states, observed_contrasts)

    figure2_rows = []
    bootstrap_by_metric = {row["metric_output_id"]: row for row in bootstrap_rows}
    family_by_key = {(row["metric_output_id"], row["statistic"]): row for row in family}
    for metric_id in METRIC_OUTPUT_IDS:
        result = metric_results[metric_id]
        if metric_id == "mse" and mse_intervals is not None:
            figure2_rows.append(
                {
                    "metric_output_id": metric_id,
                    "preferred_direction": config.metric_directions[metric_id].value,
                    "metric_state": "complete",
                    "ag": result["ag"],
                    "ag_lower": mse_intervals.reference_ag_interval[0],
                    "ag_upper": mse_intervals.reference_ag_interval[1],
                    "acc": result["acc"],
                    "acc_lower": mse_intervals.reference_acc_interval[0],
                    "acc_upper": mse_intervals.reference_acc_interval[1],
                    "pair_count": reference_acc.pair_count,
                    "strict_agreement_count": reference_acc.strict_agreement_count,
                    "strict_disagreement_count": reference_acc.strict_disagreement_count,
                    "metric_tie_count": reference_acc.metric_tie_count,
                    "downstream_tie_count": reference_acc.downstream_tie_count,
                    "double_tie_count": reference_acc.double_tie_count,
                    "d_ag": None,
                    "d_ag_lower": None,
                    "d_ag_upper": None,
                    "d_ag_favorable": None,
                    "d_acc": None,
                    "d_acc_lower": None,
                    "d_acc_upper": None,
                    "d_acc_favorable": None,
                    "d_ag_raw_p": None,
                    "d_ag_adjusted_p": None,
                    "d_ag_holm_rank": None,
                    "d_ag_rejected": None,
                    "d_acc_raw_p": None,
                    "d_acc_adjusted_p": None,
                    "d_acc_holm_rank": None,
                    "d_acc_rejected": None,
                }
            )
        elif result["metric_state"] == "complete":
            boot = bootstrap_by_metric[metric_id]
            comparison = result["comparison"]
            ag_holm = family_by_key[(metric_id, "d_ag")]
            acc_holm = family_by_key[(metric_id, "d_acc")]
            candidate_acc = comparison["candidate_accuracy"]
            figure2_rows.append(
                {
                    "metric_output_id": metric_id,
                    "preferred_direction": config.metric_directions[metric_id].value,
                    "metric_state": "complete",
                    "ag": result["ag"],
                    "ag_lower": boot["candidate_ag_interval"][0],
                    "ag_upper": boot["candidate_ag_interval"][1],
                    "acc": result["acc"],
                    "acc_lower": boot["candidate_acc_interval"][0],
                    "acc_upper": boot["candidate_acc_interval"][1],
                    "pair_count": candidate_acc["pair_count"],
                    "strict_agreement_count": candidate_acc["strict_agreement_count"],
                    "strict_disagreement_count": candidate_acc["strict_disagreement_count"],
                    "metric_tie_count": candidate_acc["metric_tie_count"],
                    "downstream_tie_count": candidate_acc["downstream_tie_count"],
                    "double_tie_count": candidate_acc["double_tie_count"],
                    "d_ag": comparison["d_ag"],
                    "d_ag_lower": boot["d_ag_interval"][0],
                    "d_ag_upper": boot["d_ag_interval"][1],
                    "d_ag_favorable": comparison["d_ag"] > 0.0,
                    "d_acc": comparison["d_acc"],
                    "d_acc_lower": boot["d_acc_interval"][0],
                    "d_acc_upper": boot["d_acc_interval"][1],
                    "d_acc_favorable": comparison["d_acc"] > 0.0,
                    "d_ag_raw_p": ag_holm["raw_p_value"],
                    "d_ag_adjusted_p": ag_holm["adjusted_p_value"],
                    "d_ag_holm_rank": ag_holm["rank"],
                    "d_ag_rejected": ag_holm["rejected"],
                    "d_acc_raw_p": acc_holm["raw_p_value"],
                    "d_acc_adjusted_p": acc_holm["adjusted_p_value"],
                    "d_acc_holm_rank": acc_holm["rank"],
                    "d_acc_rejected": acc_holm["rejected"],
                }
            )
        else:
            ag_holm = family_by_key.get((metric_id, "d_ag"))
            acc_holm = family_by_key.get((metric_id, "d_acc"))
            figure2_rows.append(
                {
                    "metric_output_id": metric_id,
                    "preferred_direction": config.metric_directions[metric_id].value,
                    "metric_state": result["metric_state"],
                    "ag": None,
                    "ag_lower": None,
                    "ag_upper": None,
                    "acc": None,
                    "acc_lower": None,
                    "acc_upper": None,
                    "pair_count": None,
                    "strict_agreement_count": None,
                    "strict_disagreement_count": None,
                    "metric_tie_count": None,
                    "downstream_tie_count": None,
                    "double_tie_count": None,
                    "d_ag": None,
                    "d_ag_lower": None,
                    "d_ag_upper": None,
                    "d_ag_favorable": None,
                    "d_acc": None,
                    "d_acc_lower": None,
                    "d_acc_upper": None,
                    "d_acc_favorable": None,
                    "d_ag_raw_p": None if ag_holm is None else ag_holm["raw_p_value"],
                    "d_ag_adjusted_p": 1.0 if ag_holm is None else ag_holm["adjusted_p_value"],
                    "d_ag_holm_rank": None if ag_holm is None else ag_holm["rank"],
                    "d_ag_rejected": False,
                    "d_acc_raw_p": None if acc_holm is None else acc_holm["raw_p_value"],
                    "d_acc_adjusted_p": 1.0 if acc_holm is None else acc_holm["adjusted_p_value"],
                    "d_acc_holm_rank": None if acc_holm is None else acc_holm["rank"],
                    "d_acc_rejected": False,
                }
            )

    figure1_rows = []
    for metric_id in METRIC_OUTPUT_IDS:
        current_rows = [row for row in class_rows if row["metric_output_id"] == metric_id and row.get("metric_harm") is not None]
        for perturbation_id in PERTURBATION_IDS:
            for alpha in POSITIVE_ALPHAS:
                selected = [row for row in current_rows if row["perturbation_id"] == perturbation_id and row["alpha"] == alpha]
                figure1_rows.append(
                    {
                        "metric_output_id": metric_id,
                        "perturbation_id": perturbation_id,
                        "alpha": alpha,
                        "mean_metric_harm": None if not selected else float(np.mean([row["metric_harm"] for row in selected])),
                        "mean_downstream_harm": None if not selected else float(np.mean([row["downstream_harm"] for row in selected])),
                        "metric_state": metric_states[metric_id]["state"],
                    }
                )

    figure_payloads = render_figure_payloads(figure1_rows, figure2_rows)
    condition_summary_rows = []
    for condition_id in CONDITION_IDS:
        selected = [row for row in prediction_rows if row["condition_id"] == condition_id]
        labels = sorted({row["class_label"] for row in selected})
        macro1 = float(np.mean([np.mean([row["top1_correct"] for row in selected if row["class_label"] == label]) for label in labels]))
        macro5 = float(np.mean([np.mean([row["top5_correct"] for row in selected if row["class_label"] == label]) for label in labels]))
        condition_summary_rows.append(
            {
                "condition_id": condition_id,
                "query_occurrence_count": len(selected),
                "top1_macro_class_accuracy": macro1,
                "top5_macro_class_accuracy": macro5,
                "top1_micro_accuracy": float(np.mean([row["top1_correct"] for row in selected])),
                "top5_micro_accuracy": float(np.mean([row["top5_correct"] for row in selected])),
            }
        )

    code = {str(key): dict(value) for key, value in config.code_authority.items()} if config.synthetic_fixture else _code_document()
    environment = dict(config.environment_authority) if config.environment_authority else _environment()
    run_id, run_identity = _run_identity(config, code, environment)
    alignment_rows = [metric_results[metric_id] for metric_id in METRIC_OUTPUT_IDS]
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "operator_cells": len(operator_cells),
            "record_conditions": len(record_condition_rows),
            "metric_values": len(metric_rows),
            "peak_receipts": len(peak_receipts),
            "downstream_rows": len(downstream_rows),
            "matcher_predictions": len(prediction_rows),
            "class_observations": len(class_rows),
            "alignment_results": len(alignment_rows),
            "bootstrap_results": len(bootstrap_rows),
            "sign_flip_results": len(sign_rows),
            "holm_family": len(family),
        },
        "environment": environment,
        "experiment_id": EXPERIMENT_ID,
        "metric_states": metric_states,
        "perturbation_ids": list(PERTURBATION_IDS),
        "protocol": "A",
        "run_id": run_id,
        "run_identity": run_identity,
        "status": "complete",
        "synthetic_fixture": config.synthetic_fixture,
        "tier": "full_domain_core",
    }
    complete = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "run_id": run_id, "status": "complete"}
    condition_fields = tuple(condition_summary_rows[0])
    table_fields = tuple(figure2_rows[0])
    payloads = {
        "config.json": config.raw_bytes,
        "preflight.json": _canonical(preflight),
        "operator_cells.jsonl": _jsonl(operator_cells),
        "record_conditions.jsonl": _jsonl(record_condition_rows),
        "metric_values.jsonl": _jsonl(metric_rows),
        "peak_receipts.jsonl": _jsonl(peak_receipts),
        "downstream_rows.jsonl": _jsonl(downstream_rows),
        "matcher_predictions.jsonl": _jsonl(prediction_rows),
        "condition_summary.csv": _csv_bytes(condition_summary_rows, condition_fields),
        "class_observations.jsonl": _jsonl(class_rows),
        "alignment_results.jsonl": _jsonl(alignment_rows),
        "bootstrap_results.jsonl": _jsonl(bootstrap_rows),
        "sign_flip_results.jsonl": _jsonl(sign_rows),
        "holm_family.jsonl": _jsonl(family),
        "d5_full_domain_secondary_table.csv": _csv_bytes(figure2_rows, table_fields),
        "manifest.json": _canonical(manifest),
        "complete.json": _canonical(complete),
        **figure_payloads,
    }
    summary = Phase4D5ProtocolASummary(
        Path("."),
        run_id,
        "complete",
        config.query_record_count,
        len(prediction_rows),
        len(metric_rows),
        len(class_rows),
    )
    return payloads, summary


def _write_payloads(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ARTIFACT_PAYLOAD_FILES:
        output_dir.joinpath(name).write_bytes(payloads[name])
    sha256s = "".join(f"{_sha_bytes(payloads[name])}  {name}\n" for name in ARTIFACT_PAYLOAD_FILES)
    output_dir.joinpath("SHA256SUMS").write_text(sha256s, encoding="utf-8")


def _load_run_config(run_path: Path) -> Phase4D5ProtocolAConfig:
    config_path = run_path / "config.json"
    if not config_path.is_file():
        raise Phase4D5ProtocolAError("config.json", "missing from run artifact")
    raw = config_path.read_bytes()
    config = parse_phase4_d5_protocol_a_config(config_path, raw, require_frozen_identity=False)
    frozen_path = ROOT / CONFIG_RELATIVE_PATH
    if not config.synthetic_fixture and frozen_path.is_file() and raw != frozen_path.read_bytes():
        raise Phase4D5ProtocolAError("config.json", "does not match frozen experiment config")
    return config


def _compare_artifacts(observed_path: Path, rebuilt_path: Path) -> None:
    expected_names = set(ARTIFACT_PAYLOAD_FILES) | {"SHA256SUMS"}
    observed_names = {item.name for item in observed_path.iterdir() if item.is_file()}
    rebuilt_names = {item.name for item in rebuilt_path.iterdir() if item.is_file()}
    if observed_names != expected_names:
        raise Phase4D5ProtocolAError("artifact inventory", f"unexpected observed files: {sorted(observed_names)!r}")
    if rebuilt_names != expected_names:
        raise Phase4D5ProtocolAError("artifact inventory", f"unexpected rebuilt files: {sorted(rebuilt_names)!r}")
    for name in sorted(expected_names):
        observed = observed_path.joinpath(name).read_bytes()
        rebuilt = rebuilt_path.joinpath(name).read_bytes()
        if observed != rebuilt:
            raise Phase4D5ProtocolAError(name, "byte mismatch")
        if name != "SHA256SUMS":
            checksum = _sha_bytes(observed)
            line = f"{checksum}  {name}\n".encode("utf-8")
            if line not in observed_path.joinpath("SHA256SUMS").read_bytes():
                raise Phase4D5ProtocolAError("SHA256SUMS", f"missing checksum line for {name}")


def verify_phase4_d5_protocol_a_from_inputs(
    path: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    classical_catalog: Phase3ClassicalCatalog,
    worker_count: int,
    inference_resamples: int | None = None,
) -> Phase4D5ProtocolASummary:
    _positive_int("worker_count", worker_count)
    run_path = Path(path)
    if not run_path.is_dir():
        raise Phase4D5ProtocolAError("path", "must be an existing run directory")
    config = _load_run_config(run_path)
    payloads, rebuilt_summary = _payloads_from_inputs(
        cohort=cohort,
        native_spectra=native_spectra,
        sweep=sweep,
        phase1_config=phase1_config,
        classical_catalog=classical_catalog,
        config=config,
        inference_resamples=inference_resamples,
    )
    with tempfile.TemporaryDirectory(prefix="phase4-d5-protocol-a-verify-") as temporary:
        rebuilt_path = Path(temporary) / rebuilt_summary.run_id
        _write_payloads(rebuilt_path, payloads)
        _compare_artifacts(run_path, rebuilt_path)
    return Phase4D5ProtocolASummary(
        run_path,
        rebuilt_summary.run_id,
        rebuilt_summary.status,
        rebuilt_summary.query_record_count,
        rebuilt_summary.prediction_row_count,
        rebuilt_summary.metric_row_count,
        rebuilt_summary.class_observation_count,
    )


def verify_phase4_d5_protocol_a(path: Path, *, worker_count: int = 12) -> Phase4D5ProtocolASummary:
    _positive_int("worker_count", worker_count)
    run_path = Path(path)
    config = _load_run_config(run_path)
    cohort = load_d5_raw_cohort(ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH)
    native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1 = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    catalog = load_classical_catalog(ROOT / CATALOG_RELATIVE_PATH, project_root=ROOT)
    if config.synthetic_fixture:
        raise Phase4D5ProtocolAError("config", "synthetic fixture must use verify_phase4_d5_protocol_a_from_inputs")
    return verify_phase4_d5_protocol_a_from_inputs(
        run_path,
        cohort=cohort,
        native_spectra=native,
        sweep=sweep,
        phase1_config=phase1,
        classical_catalog=catalog,
        worker_count=worker_count,
        inference_resamples=None,
    )


__all__ = [
    "aggregate_class_observations",
    "fixed_holm_family",
    "match_d5_protocol_a_799",
    "metric_preflight_states",
    "render_figure_payloads",
    "verify_phase4_d5_protocol_a",
    "verify_phase4_d5_protocol_a_from_inputs",
]
