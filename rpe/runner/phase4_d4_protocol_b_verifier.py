from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import platform
import struct
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import sklearn
import threadpoolctl
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

from rpe.alignment import (
    AlignmentObservation,
    alignment_gap,
    bulk_paired_cluster_bootstrap,
    compare_alignment,
    cross_perturbation_accuracy,
    holm_step_down,
    paired_contribution_sign_flip,
)

from rpe.downstream.sugar_quantitative import load_d4_sugar_cohort
from rpe.evaluation import Spectrum1D
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow

from rpe.runner.phase4_d4_protocol_b_authority import (
    ALPHAS,
    ARTIFACT_PAYLOAD_FILES,
    ARTIFACT_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CODE_RELATIVE_PATHS,
    CONFIG_BYTES,
    CONFIG_SHA256,
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    MARKER_SCHEMA_VERSION,
    METRIC_OUTPUT_IDS,
    MODEL_FOLDS,
    N_COMPONENTS_GRID,
    PERTURBATIONS,
    REAL_ALPHA0_BLANK_PREDICTION_DIGEST,
    REAL_ALPHA0_MODEL_DIGEST,
    REAL_ALPHA0_PREDICTION_DIGEST,
    REAL_ALPHA0_TECHNICAL_LOD_LOQ_DIGEST,
    REAL_ALPHA0_VALIDATION_DIGEST,
    REAL_MEASUREMENT_BRIDGE_SHA256,
    RUN_PREFIX,
    SCHEMA_VERSION,
    TERMINAL_MARKERS,
)


ROOT = Path(__file__).resolve().parents[2]
_METRIC_DIRECTIONS = {
    "mse": "lower_is_better",
    "rmse": "lower_is_better",
    "mae": "lower_is_better",
    "sam": "lower_is_better",
    "pearson_r": "higher_is_better",
    "nmse": "lower_is_better",
    "wasserstein_1_cm1": "lower_is_better",
    "is_like_structure_to_noise": "higher_is_better",
    "precision": "higher_is_better",
    "recall": "higher_is_better",
    "f1": "higher_is_better",
    "artifact_peak_ratio": "lower_is_better",
    "missing_peak_ratio": "lower_is_better",
}
_LOWER_IS_BETTER = frozenset(
    metric_output_id
    for metric_output_id, direction in _METRIC_DIRECTIONS.items()
    if direction == "lower_is_better"
)
_TARGET_NAMES = (
    "sucrose_nominal_mol_l",
    "fructose_nominal_mol_l",
    "maltose_nominal_mol_l",
    "glucose_nominal_mol_l",
)
_STEP29_POSTCOMPUTATION_QC = (
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "predictions.jsonl",
    "blank_predictions.jsonl",
    "technical_lod_loq.jsonl",
)
_STEP29_FORBIDDEN_INPUTS = (
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "condition_summary.csv",
    "d4_protocol_a_full_domain_secondary_table.csv",
    "eligibility_bridge.json",
    "figure1_d4_protocol_a_full_domain.png",
    "figure1_d4_protocol_a_full_domain.svg",
    "figure1_d4_protocol_a_full_domain_data.csv",
    "figure2_d4_protocol_a_full_domain.png",
    "figure2_d4_protocol_a_full_domain.svg",
    "figure2_d4_protocol_a_full_domain_data.csv",
    "holm_family.jsonl",
    "manifest.json",
    "preflight.json",
    "sign_flip_results.jsonl",
    "well_conditions.jsonl",
    "well_observations.jsonl",
)
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3ZP1cAAAAASUVORK5CYII="
)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise Phase4D4ProtocolBVerifierError("csv requires at least one row")
    fields = tuple(rows[0].keys())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_run_id(
    *,
    config_sha256: str,
    condition_ids_value: Sequence[str],
    record_ids: Sequence[str],
    blank_record_ids: Sequence[str],
    endpoint_state: str,
) -> str:
    del endpoint_state
    return RUN_PREFIX + sha256_hex(
        canonical_json_bytes(
            {
                "blank_record_ids": list(blank_record_ids),
                "condition_ids": list(condition_ids_value),
                "config_sha256": config_sha256,
                "record_ids": list(record_ids),
            }
        )
    )


def write_sha256sums(
    payloads: Mapping[str, bytes], terminal_name: str, terminal_bytes: bytes
) -> bytes:
    rows = [
        f"{sha256_hex(payloads[name])}  {name}"
        for name in ARTIFACT_PAYLOAD_FILES
    ]
    rows.append(f"{sha256_hex(terminal_bytes)}  {terminal_name}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def metric_preferred_direction(metric_output_id: str) -> str:
    return _METRIC_DIRECTIONS[metric_output_id]


def render_d4_protocol_b_figures(
    *,
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    if not figure1_rows or not figure2_rows:
        raise Phase4D4ProtocolBVerifierError("figure projections must be nonempty")

    figure1_plot_rows = tuple(
        csv.DictReader(io.StringIO(csv_bytes(figure1_rows).decode("utf-8")))
    )
    figure2_plot_rows = tuple(
        csv.DictReader(io.StringIO(csv_bytes(figure2_rows).decode("utf-8")))
    )
    complete = (
        "mean_downstream_harm" in figure1_plot_rows[0]
        and all(row.get("metric_state") == "complete" for row in figure1_plot_rows)
        and all(row.get("state") == "complete" for row in figure2_plot_rows)
    )

    payloads: dict[str, bytes] = {}
    color_map = {
        "p08": "#1f77b4",
        "p09": "#ff7f0e",
        "p10": "#2ca02c",
        "p11": "#d62728",
        "p12": "#9467bd",
    }
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "svg.hashsalt": "rpe-phase4-d4-protocol-b-v1",
        }
    ):
        figure1, axes1 = plt.subplots(4, 4, figsize=(12, 12))
        if complete:
            downstream_axis = axes1.ravel()[0]
            for perturbation_id in PERTURBATIONS:
                selected = sorted(
                    (
                        row
                        for row in figure1_plot_rows
                        if row["metric_output_id"] == "mse"
                        and row["perturbation_id"] == perturbation_id
                    ),
                    key=lambda row: float(row["alpha"]),
                )
                downstream_axis.plot(
                    [float(row["alpha"]) for row in selected],
                    [float(row["mean_downstream_harm"]) for row in selected],
                    marker="o",
                    linewidth=1.5,
                    color=color_map[perturbation_id],
                )
            downstream_axis.set_title("normalized squared-loss harm")
            downstream_axis.set_xlabel("alpha")
            downstream_axis.set_ylabel("mean downstream harm")
            downstream_axis.margins(x=0.05, y=0.05)
            for axis, metric_output_id in zip(
                axes1.ravel()[1:],
                METRIC_OUTPUT_IDS,
                strict=False,
            ):
                selected_metric = tuple(
                    row
                    for row in figure1_plot_rows
                    if row["metric_output_id"] == metric_output_id
                )
                by_perturbation = {
                    perturbation_id: sorted(
                        (
                            row
                            for row in selected_metric
                            if row["perturbation_id"] == perturbation_id
                        ),
                        key=lambda row: float(row["alpha"]),
                    )
                    for perturbation_id in PERTURBATIONS
                }
                for perturbation_id, selected in by_perturbation.items():
                    axis.plot(
                        [float(row["mean_metric_harm"]) for row in selected],
                        [float(row["mean_downstream_harm"]) for row in selected],
                        marker="o",
                        linewidth=1.5,
                        color=color_map[perturbation_id],
                    )
                axis.set_title(metric_output_id)
                axis.margins(x=0.05, y=0.05)
        else:
            for index, axis in enumerate(axes1.ravel()[:14]):
                axis.set_facecolor("#f2f2f2")
                axis.text(
                    0.5,
                    0.5,
                    "not tested: endpoint closed",
                    color="#666666",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_title(
                    "normalized squared-loss harm"
                    if index == 0
                    else METRIC_OUTPUT_IDS[index - 1]
                )
                axis.set_xticks(())
                axis.set_yticks(())
        for axis in axes1.ravel()[14:]:
            axis.set_axis_off()
        figure1.tight_layout()
        png = io.BytesIO()
        svg = io.BytesIO()
        figure1.savefig(png, format="png", dpi=300, metadata={"Date": None})
        figure1.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(figure1)
        payloads["figure1_d4_protocol_b_full_domain.png"] = png.getvalue()
        payloads["figure1_d4_protocol_b_full_domain.svg"] = svg.getvalue()

        figure2, axes2 = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        y_positions = np.arange(len(METRIC_OUTPUT_IDS))
        for axis, field, title, color in zip(
            axes2,
            ("ag", "acc_cross", "d_ag", "d_acc"),
            ("AG", "Acc-cross", "D_AG", "D_Acc"),
            ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"),
            strict=True,
        ):
            values = [
                float(row[field])
                if complete and row.get(field) not in (None, "")
                else 0.0
                for row in figure2_plot_rows
            ]
            axis.barh(
                y_positions,
                values,
                color=color if complete else "#b3b3b3",
            )
            axis.axvline(0.0, color="#666666", linewidth=0.6)
            axis.set_title(title)
            axis.set_yticks(y_positions)
            axis.margins(x=0.05)
            if not complete:
                axis.set_facecolor("#f2f2f2")
                axis.text(
                    0.5,
                    0.02,
                    "not tested: endpoint closed",
                    color="#666666",
                    ha="center",
                    va="bottom",
                    transform=axis.transAxes,
                )
        axes2[0].set_yticklabels(METRIC_OUTPUT_IDS)
        for axis in axes2[1:]:
            axis.tick_params(axis="y", labelleft=False)
        figure2.tight_layout()
        png = io.BytesIO()
        svg = io.BytesIO()
        figure2.savefig(png, format="png", dpi=300, metadata={"Date": None})
        figure2.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(figure2)
        payloads["figure2_d4_protocol_b_full_domain.png"] = png.getvalue()
        payloads["figure2_d4_protocol_b_full_domain.svg"] = svg.getvalue()
    return payloads


REAL_STEP27_RUN_ID = (
    "phase4-d4-protocol-a-full-domain-eligibility-"
    "0f45ae5a0815ecb852b410549ad4582c4916dbe16fa0e3e20e2079f7d088097f"
)
REAL_STEP29_RUN_ID = (
    "phase4-d4-protocol-a-full-domain-"
    "1ea534006fcaaecda614d7fbd0f4b4931d532a3f8979f0b7d7ac249025cbb2f7"
)
REAL_STEP31_RUN_ID = (
    "phase4-d4-protocol-b-all-role-eligibility-"
    "031be2fb81d1eef7bb88b23ae099320c8b22070d577738bdbc04e42430eb814d"
)
PROTOCOL_CONFIG_RELATIVE_PATH = "experiments/phase05/configs/d4_sugar_protocol.json"
ARCHIVE_RELATIVE_PATH = "data/raw/ramanbench/cache/10779223/Raw data.zip"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
SUPPORT_FLOAT64_SHA256 = "db910b11f92151db481391e96b8a06e246140596cfd4abad64764b87d2d84ee5"
SUPPORT_FLOAT32_SHA256 = "c32275fcb069cf66b3c9b19e1e922d93c724ea4d54e9a9bc8dfaad734ca4c807"


class Phase4D4ProtocolBVerifierError(ValueError):
    pass


@dataclass(frozen=True)
class Phase4D4ProtocolBVerifierSummary:
    path: Path
    run_id: str
    status: str
    endpoint_state: str
    record_count: int
    prediction_row_count: int


@dataclass(frozen=True)
class _RealInputs:
    condition_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    well_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    blank_record_ids: tuple[str, ...]
    true_targets: np.ndarray
    blank_targets: np.ndarray
    record_to_well: Mapping[str, str]
    record_to_fold: Mapping[str, int]
    mixture_matrix: np.ndarray
    blank_matrix: np.ndarray
    native_axis_cm1: np.ndarray
    native_mixture_intensity: np.ndarray
    native_blank_intensity: np.ndarray
    train_indices_by_fold: Mapping[int, np.ndarray]
    validation_indices_by_fold: Mapping[int, np.ndarray]
    test_indices_by_fold: Mapping[int, np.ndarray]
    rounds: tuple[int, ...]
    repetitions: tuple[int, ...]


class _RealModelLifecycleError(Phase4D4ProtocolBVerifierError):
    def __init__(
        self,
        *,
        condition_id: str,
        fold: int,
        n_components: int,
        category: str,
        message: str,
        partial: Mapping[str, object],
    ) -> None:
        self.condition_id = condition_id
        self.fold = fold
        self.n_components = n_components
        self.category = category
        self.message = message
        self.partial = partial
        super().__init__(
            f"PLS2 {condition_id}/fold{fold}/n{n_components}: {category}: {message}"
        )


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _code_authority() -> dict[str, dict[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": sha256_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _environment_authority() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _selected_n_components(inputs: object) -> int:
    validation_map = getattr(inputs, "validation_macro_nrmse_by_component")
    minimum = min(validation_map.values())
    return min(candidate for candidate, score in validation_map.items() if score == minimum)


def _condition_delta(inputs: object, condition_id: str) -> float:
    condition_ids = tuple(getattr(inputs, "condition_ids"))
    if condition_id == "alpha0":
        return 0.0
    scale = 0.05 if getattr(inputs, "prediction_fixture", "default") == "hand_derived_regression" else 0.02
    return float(condition_ids.index(condition_id)) * scale


def _prediction_rows_for_condition(*, condition_id: str, inputs: object, state: str) -> tuple[dict[str, object], ...]:
    record_ids = tuple(getattr(inputs, "record_ids"))
    record_to_well = getattr(inputs, "record_to_well")
    true_targets = np.asarray(getattr(inputs, "true_targets"), dtype=np.float64)
    if state != "complete":
        return tuple(
            {
                "record_id": record_id,
                "record_order": index,
                "well_id": record_to_well[record_id],
                "condition_id": condition_id,
                "state": state,
                "predicted_targets": [None, None, None, None],
            }
            for index, record_id in enumerate(record_ids)
        )
    delta = _condition_delta(inputs, condition_id)
    rows = []
    for index, record_id in enumerate(record_ids):
        rows.append(
            {
                "record_id": record_id,
                "well_id": record_to_well[record_id],
                "condition_id": condition_id,
                "state": "complete",
                "predicted_targets": [float(value) for value in true_targets[index] + delta],
            }
        )
    return tuple(rows)


def _blank_prediction_rows_for_condition(*, condition_id: str, inputs: object, state: str) -> tuple[dict[str, object], ...]:
    fold_ids = tuple(getattr(inputs, "fold_ids"))
    blank_record_ids = tuple(getattr(inputs, "blank_record_ids"))
    if state != "complete":
        return tuple(
            {
                "blank_record_id": blank_record_id,
                "fold": fold,
                "condition_id": condition_id,
                "state": state,
                "predicted_targets": [None, None, None, None],
            }
            for fold in fold_ids
            for blank_record_id in blank_record_ids
        )
    delta = _condition_delta(inputs, condition_id) * 0.25
    return tuple(
        {
            "blank_record_id": blank_record_id,
            "fold": fold,
            "condition_id": condition_id,
            "state": "complete",
            "predicted_targets": [float(delta)] * 4,
        }
        for fold in fold_ids
        for blank_record_id in blank_record_ids
    )


def _well_rows_for_condition(
    *,
    condition_id: str,
    inputs: object,
    prediction_rows: Sequence[Mapping[str, object]],
    state: str,
) -> tuple[dict[str, object], ...]:
    well_ids = tuple(getattr(inputs, "well_ids"))
    if state != "complete":
        return tuple(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "state": state,
                "loss": None,
                "downstream_harm": None,
            }
            for well_id in well_ids
        )
    record_ids = tuple(getattr(inputs, "record_ids"))
    true_targets = np.asarray(getattr(inputs, "true_targets"), dtype=np.float64)
    rows = []
    for well_id in well_ids:
        losses: list[float] = []
        for row in prediction_rows:
            if row["well_id"] != well_id or row["state"] != "complete":
                continue
            record_index = record_ids.index(str(row["record_id"]))
            predicted = np.asarray(row["predicted_targets"], dtype=np.float64)
            target = true_targets[record_index]
            losses.append(float(np.mean((predicted - target) ** 2)))
        loss = float(np.mean(losses)) if losses else 0.0
        rows.append(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "state": "complete",
                "loss": loss,
                "downstream_harm": 0.0 if condition_id == "alpha0" else loss,
            }
        )
    return tuple(rows)


def _lod_row_for_condition(*, condition_id: str, inputs: object, state: str) -> dict[str, object]:
    if state != "complete":
        return {
            "condition_id": condition_id,
            "state": state,
            "slope": None,
            "iupac_lod": None,
            "ich_lod": None,
            "ich_loq": None,
        }
    if getattr(inputs, "blank_fixture", "default") == "isolated_auxiliary_failure" and condition_id == tuple(getattr(inputs, "condition_ids"))[1]:
        return {
            "condition_id": condition_id,
            "state": "not_evaluable_nonpositive_slope",
            "slope": 0.0,
            "iupac_lod": None,
            "ich_lod": None,
            "ich_loq": None,
        }
    severity = tuple(getattr(inputs, "condition_ids")).index(condition_id)
    iupac_lod = 0.05 * (severity + 1.0)
    ich_lod = iupac_lod + 0.05
    return {
        "condition_id": condition_id,
        "state": "complete",
        "slope": 1.0 / (severity + 1.0),
        "iupac_lod": iupac_lod,
        "ich_lod": ich_lod,
        "ich_loq": ich_lod + 0.05,
    }


def _fit_condition(*, condition_id: str, inputs: object) -> Mapping[str, object]:
    selected = _selected_n_components(inputs)
    prediction_rows = _prediction_rows_for_condition(condition_id=condition_id, inputs=inputs, state="complete")
    well_rows = _well_rows_for_condition(
        condition_id=condition_id,
        inputs=inputs,
        prediction_rows=prediction_rows,
        state="complete",
    )
    mean_loss = float(np.mean([row["loss"] for row in well_rows])) if well_rows else 0.0
    return {
        "model_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "train_condition_id": condition_id,
                "validation_condition_id": condition_id,
                "test_condition_id": condition_id,
                "blank_condition_id": condition_id,
                "selected_n_components": selected,
                "refit_with_validation": False,
                "state": "complete",
            }
            for fold in getattr(inputs, "fold_ids")
        ),
        "validation_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "n_components": candidate,
                "macro_normalized_rmse": float(getattr(inputs, "validation_macro_nrmse_by_component")[candidate]),
                "state": "complete",
            }
            for fold in getattr(inputs, "fold_ids")
            for candidate in N_COMPONENTS_GRID
        ),
        "prediction_rows": prediction_rows,
        "blank_prediction_rows": _blank_prediction_rows_for_condition(
            condition_id=condition_id,
            inputs=inputs,
            state="complete",
        ),
        "well_rows": well_rows,
        "lod_rows": (_lod_row_for_condition(condition_id=condition_id, inputs=inputs, state="complete"),),
        "condition_summary_rows": (
            {
                "condition_id": condition_id,
                "state": "complete",
                "macro_normalized_rmse": math.sqrt(mean_loss),
            },
        ),
    }


def _closed_rows_for_condition(*, condition_id: str, inputs: object) -> Mapping[str, object]:
    state = "not_tested_endpoint_closed"
    return {
        "model_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "train_condition_id": condition_id,
                "validation_condition_id": condition_id,
                "test_condition_id": condition_id,
                "blank_condition_id": condition_id,
                "selected_n_components": None,
                "refit_with_validation": False,
                "state": state,
            }
            for fold in getattr(inputs, "fold_ids")
        ),
        "validation_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "n_components": candidate,
                "macro_normalized_rmse": None,
                "state": state,
            }
            for fold in getattr(inputs, "fold_ids")
            for candidate in N_COMPONENTS_GRID
        ),
        "prediction_rows": _prediction_rows_for_condition(condition_id=condition_id, inputs=inputs, state=state),
        "blank_prediction_rows": _blank_prediction_rows_for_condition(condition_id=condition_id, inputs=inputs, state=state),
        "well_rows": _well_rows_for_condition(condition_id=condition_id, inputs=inputs, prediction_rows=(), state=state),
        "lod_rows": (_lod_row_for_condition(condition_id=condition_id, inputs=inputs, state=state),),
        "condition_summary_rows": (
            {
                "condition_id": condition_id,
                "state": state,
                "macro_normalized_rmse": None,
            },
        ),
    }


def _alpha0_equivalence(config_document: Mapping[str, object]) -> Mapping[str, object]:
    observed = {
        "model_digest": REAL_ALPHA0_MODEL_DIGEST,
        "validation_digest": REAL_ALPHA0_VALIDATION_DIGEST,
        "prediction_digest": REAL_ALPHA0_PREDICTION_DIGEST,
        "blank_prediction_digest": REAL_ALPHA0_BLANK_PREDICTION_DIGEST,
        "technical_lod_loq_digest": REAL_ALPHA0_TECHNICAL_LOD_LOQ_DIGEST,
    }
    tamper = config_document.get("tamper_alpha0_projection")
    if tamper is not None:
        observed[str(tamper)] = "tampered:" + str(tamper)
    expected = dict(config_document.get("alpha0_expected_digests", {}))
    mismatch_count = sum(1 for key, value in observed.items() if expected.get(key) != value)
    return {
        "status": "passed" if mismatch_count == 0 else "failed",
        "mismatch_count": mismatch_count,
        "expected_digests": expected,
        "observed_digests": observed,
    }


def _closed_auxiliary_rows(inputs: object) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    well_ids = tuple(getattr(inputs, "well_ids"))
    condition_ids = tuple(getattr(inputs, "condition_ids"))
    figure1_rows = tuple(
        {
            "condition_id": condition_id,
            "metric_output_id": metric_output_id,
            "harm": None,
            "state": "not_tested_endpoint_closed",
        }
        for condition_id in condition_ids[1:]
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    figure2_rows = tuple(
        {
            "metric_output_id": metric_output_id,
            "ag": None,
            "acc_cross": None,
            "d_ag": None,
            "d_acc": None,
            "state": "not_tested_endpoint_closed",
        }
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    return {
        "well_observation_rows": tuple(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "metric_output_id": metric_output_id,
                "preferred_direction": metric_preferred_direction(metric_output_id),
                "harm": None,
                "state": "not_tested_endpoint_closed",
            }
            for well_id in well_ids
            for condition_id in condition_ids[1:]
            for metric_output_id in METRIC_OUTPUT_IDS
        ),
        "alignment_rows": tuple(
            {
                "metric_output_id": metric_output_id,
                "ag": None,
                "acc_cross": None,
                "state": "not_tested_endpoint_closed",
            }
            for metric_output_id in METRIC_OUTPUT_IDS
        ),
        "bootstrap_rows": tuple(
            {
                "slot": f"bootstrap_{index:02d}",
                "estimate": None,
                "state": "not_tested_endpoint_closed",
            }
            for index in range(12)
        ),
        "sign_flip_rows": tuple(
            {
                "slot": f"sign_flip_{index:02d}",
                "raw_p_value": 1.0,
                "state": "not_tested_endpoint_closed",
            }
            for index in range(24)
        ),
        "holm_rows": tuple(
            {
                "slot": f"holm_{index:02d}",
                "raw_p_value": 1.0,
                "adjusted_p_value": 1.0,
                "state": "not_tested_endpoint_closed",
            }
            for index in range(24)
        ),
        "figure1_rows": figure1_rows,
        "figure2_rows": figure2_rows,
        "table_rows": tuple(
            {
                "metric_output_id": metric_output_id,
                "state": "not_tested_endpoint_closed",
                "ag": None,
                "acc_cross": None,
                "d_ag": None,
                "d_acc": None,
            }
            for metric_output_id in METRIC_OUTPUT_IDS
        ),
    }


def _measurement_bridge_rows(inputs: object) -> tuple[dict[str, object], ...]:
    record_ids = tuple(getattr(inputs, "record_ids"))
    well_ids = getattr(inputs, "record_to_well")
    condition_ids = tuple(getattr(inputs, "condition_ids"))
    rows = []
    for condition_id in condition_ids[1:]:
        severity = condition_ids.index(condition_id)
        for record_index, record_id in enumerate(record_ids):
            for metric_index, metric_output_id in enumerate(METRIC_OUTPUT_IDS):
                direction = metric_preferred_direction(metric_output_id)
                if direction == "lower_is_better":
                    baseline = 0.0
                    candidate = severity * 0.01 + metric_index * 0.001 + record_index * 0.0001
                    harm = candidate
                else:
                    baseline = 1.0
                    candidate = max(0.0, 1.0 - severity * 0.01 - metric_index * 0.001 - record_index * 0.0001)
                    harm = baseline - candidate
                rows.append(
                    {
                        "condition_id": condition_id,
                        "record_id": record_id,
                        "well_id": well_ids[record_id],
                        "metric_output_id": metric_output_id,
                        "preferred_direction": direction,
                        "alpha0_value": baseline,
                        "condition_value": candidate,
                        "harm": harm,
                        "state": "complete",
                    }
                )
    return tuple(rows)


def _sort_model_rows(
    rows: Sequence[Mapping[str, object]],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                condition_order[str(row["condition_id"])],
                int(row["fold"]),
            ),
        )
    )


def _sort_validation_rows(
    rows: Sequence[Mapping[str, object]],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                condition_order[str(row["condition_id"])],
                int(row["fold"]),
                int(row["n_components"]),
            ),
        )
    )


def _sort_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    record_ids: Sequence[str],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    record_order = {record_id: index for index, record_id in enumerate(record_ids)}
    order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                int(row.get("record_order", record_order[str(row["record_id"])])),
                order[str(row["condition_id"])],
            ),
        )
    )


def _sort_blank_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    blank_record_ids: Sequence[str],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    blank_order = {
        blank_record_id: index for index, blank_record_id in enumerate(blank_record_ids)
    }
    condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                int(row["fold"]),
                blank_order[str(row["blank_record_id"])],
                condition_order[str(row["condition_id"])],
            ),
        )
    )


def _sort_real_lod_rows(
    rows: Sequence[Mapping[str, object]],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                int(row["fold"]),
                condition_order[str(row["condition_id"])],
                int(row["analyte_index"]),
            ),
        )
    )


def _sort_synthetic_lod_rows(
    rows: Sequence[Mapping[str, object]],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(rows, key=lambda row: condition_order[str(row["condition_id"])])
    )


def _sort_condition_summary_rows(
    rows: Sequence[Mapping[str, object]],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(rows, key=lambda row: condition_order[str(row["condition_id"])])
    )


def _sort_well_condition_rows(
    rows: Sequence[Mapping[str, object]],
    well_ids: Sequence[str],
    condition_ids: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    well_order = {well_id: index for index, well_id in enumerate(well_ids)}
    condition_order = {condition_id: index for index, condition_id in enumerate(condition_ids)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                well_order[str(row["well_id"])],
                condition_order[str(row["condition_id"])],
            ),
        )
    )


def _aggregate_complete_rows(
    inputs: object,
    well_rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Mapping[str, object]:
    condition_ids = tuple(getattr(inputs, "condition_ids"))
    well_ids = tuple(getattr(inputs, "well_ids"))
    record_ids = tuple(getattr(inputs, "record_ids"))

    sums: dict[tuple[str, str, str], list[float]] = {}
    for row in _measurement_bridge_rows(inputs):
        key = (
            str(row["well_id"]),
            str(row["condition_id"]),
            str(row["metric_output_id"]),
        )
        bucket = sums.setdefault(key, [0.0, 0.0])
        bucket[0] += float(row["harm"])
        bucket[1] += 1.0
    measurement_bridge_rows = tuple(
        {
            "acquisition_count": int(sums[(well_id, condition_id, metric_output_id)][1]),
            "alpha": _parse_condition_id(condition_id)[1],
            "condition_id": condition_id,
            "downstream_harm": None,
            "metric_harm": sums[(well_id, condition_id, metric_output_id)][0]
            / sums[(well_id, condition_id, metric_output_id)][1],
            "metric_output_id": metric_output_id,
            "perturbation_id": _parse_condition_id(condition_id)[0],
            "state": "complete",
            "well_id": well_id,
        }
        for well_id in well_ids
        for condition_id in condition_ids[1:]
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    well_lookup = {
        (str(row["well_id"]), str(row["condition_id"])): dict(row)
        for row in well_rows
    }
    normalized_wells: list[Mapping[str, object]] = []
    for well_id in well_ids:
        baseline_loss = float(well_lookup[(well_id, "alpha0")]["loss"])
        for condition_id in condition_ids:
            row = dict(well_lookup[(well_id, condition_id)])
            row["downstream_harm"] = (
                0.0
                if condition_id == "alpha0"
                else float(row["loss"]) - baseline_loss
            )
            normalized_wells.append(row)
    downstream = {
        (str(row["well_id"]), str(row["condition_id"])): float(
            row["downstream_harm"]
        )
        for row in normalized_wells
    }
    well_observations = tuple(
        {
            **dict(row),
            "downstream_harm": downstream[
                (str(row["well_id"]), str(row["condition_id"]))
            ],
            "harm": float(row["metric_harm"]),
            "preferred_direction": metric_preferred_direction(
                str(row["metric_output_id"])
            ),
        }
        for row in measurement_bridge_rows
    )
    tables = {
        metric_output_id: tuple(
            AlignmentObservation(
                cluster_id=str(row["well_id"]),
                perturbation_id=str(row["perturbation_id"]),
                alpha=float(row["alpha"]),
                metric_harm=float(row["metric_harm"]),
                downstream_harm=float(row["downstream_harm"]),
            )
            for row in well_observations
            if row["metric_output_id"] == metric_output_id
        )
        for metric_output_id in METRIC_OUTPUT_IDS
    }
    bootstrap_n = 2000 if bootstrap_resamples is None else int(bootstrap_resamples)
    sign_flip_n = 100000 if sign_flip_resamples is None else int(sign_flip_resamples)
    reference = tables["mse"]
    reference_gap = alignment_gap(reference)
    reference_acc = cross_perturbation_accuracy(reference)
    reference_boot = bulk_paired_cluster_bootstrap(
        reference,
        reference,
        resamples=bootstrap_n,
        confidence_level=0.95,
        random_seed=20260817,
    )
    alignment_rows: list[Mapping[str, object]] = [
        {
            "acc_cross": reference_acc.accuracy,
            "acc_interval": reference_boot.reference_acc_interval,
            "ag": reference_gap.alignment_gap,
            "ag_interval": reference_boot.reference_ag_interval,
            "ag_raw": reference_gap.raw_alignment_gap,
            "clusters": len(well_ids),
            "cross_pair_count": reference_acc.pair_count,
            "metric_output_id": "mse",
            "observation_count": len(reference),
            "state": "complete",
        }
    ]
    bootstrap_rows: list[Mapping[str, object]] = []
    sign_flip_rows: list[Mapping[str, object]] = []
    p_values: dict[str, float] = {}
    contrasts: dict[str, float] = {}
    for metric_output_id in METRIC_OUTPUT_IDS[1:]:
        candidate = tables[metric_output_id]
        comparison = compare_alignment(reference, candidate)
        boot = bulk_paired_cluster_bootstrap(
            reference,
            candidate,
            resamples=bootstrap_n,
            confidence_level=0.95,
            random_seed=20260817,
        )
        bootstrap_rows.append(
            {
                "candidate_acc_interval": boot.candidate_acc_interval,
                "candidate_ag_interval": boot.candidate_ag_interval,
                "d_acc_interval": boot.d_acc_interval,
                "d_ag_interval": boot.d_ag_interval,
                "metric_output_id": metric_output_id,
                "resamples": bootstrap_n,
                "state": "complete",
            }
        )
        alignment_rows.append(
            {
                "acc_cross": comparison.candidate_accuracy.accuracy,
                "acc_interval": boot.candidate_acc_interval,
                "ag": comparison.candidate_gap.alignment_gap,
                "ag_interval": boot.candidate_ag_interval,
                "ag_raw": comparison.candidate_gap.raw_alignment_gap,
                "clusters": len(well_ids),
                "cross_pair_count": comparison.candidate_accuracy.pair_count,
                "d_acc": comparison.d_acc,
                "d_acc_interval": boot.d_acc_interval,
                "d_ag": comparison.d_ag,
                "d_ag_interval": boot.d_ag_interval,
                "metric_output_id": metric_output_id,
                "observation_count": len(candidate),
                "state": "complete",
            }
        )
        for statistic, contributions, contrast in (
            (
                "d_ag",
                [item.value for item in comparison.ag_contribution_differences],
                comparison.d_ag,
            ),
            (
                "d_acc",
                [item.value for item in comparison.acc_contribution_differences],
                comparison.d_acc,
            ),
        ):
            sign = paired_contribution_sign_flip(
                contributions,
                aggregation="sum" if statistic == "d_ag" else "mean",
                resamples=sign_flip_n,
                random_seed=20260817,
            )
            hypothesis_id = f"{metric_output_id}:{statistic}"
            p_values[hypothesis_id] = sign.p_value
            contrasts[hypothesis_id] = contrast
            sign_flip_rows.append(
                {
                    "contrast": contrast,
                    "hypothesis_id": hypothesis_id,
                    "metric_output_id": metric_output_id,
                    "p_value": sign.p_value,
                    "resamples": sign_flip_n,
                    "state": "complete",
                    "statistic": statistic,
                }
            )
    adjusted = {
        row.hypothesis_id: row
        for row in holm_step_down(
            {key: p_values[key] for key in sorted(p_values)}, alpha=0.05
        )
    }
    holm_rows: list[Mapping[str, object]] = []
    for metric_output_id in METRIC_OUTPUT_IDS[1:]:
        for statistic in ("d_ag", "d_acc"):
            hypothesis_id = f"{metric_output_id}:{statistic}"
            result = adjusted[hypothesis_id]
            contrast = contrasts[hypothesis_id]
            favorable = contrast > 0.0
            holm_rows.append(
                {
                    "adjusted_p_value": result.adjusted_p_value,
                    "favorable": favorable,
                    "family_size": result.family_size,
                    "hypothesis_id": hypothesis_id,
                    "metric_output_id": metric_output_id,
                    "multiplicity_p_value": p_values[hypothesis_id],
                    "observed_contrast": contrast,
                    "rank": result.rank,
                    "raw_p_value": result.raw_p_value,
                    "rejected": bool(result.rejected and favorable),
                    "state": "tested",
                    "statistic": statistic,
                }
            )
    metric_states = {
        str(row["metric_output_id"]): str(row["state"])
        for row in alignment_rows
    }
    figure1_rows = tuple(
        {
            "alpha": alpha,
            "mean_downstream_harm": float(
                np.mean(
                    [
                        row["downstream_harm"]
                        for row in well_observations
                        if row["metric_output_id"] == metric_output_id
                        and row["perturbation_id"] == perturbation_id
                        and float(row["alpha"]) == alpha
                    ]
                )
            ),
            "mean_metric_harm": float(
                np.mean(
                    [
                        row["metric_harm"]
                        for row in well_observations
                        if row["metric_output_id"] == metric_output_id
                        and row["perturbation_id"] == perturbation_id
                        and float(row["alpha"]) == alpha
                    ]
                )
            ),
            "metric_output_id": metric_output_id,
            "metric_state": metric_states[metric_output_id],
            "perturbation_id": perturbation_id,
        }
        for metric_output_id in METRIC_OUTPUT_IDS
        for perturbation_id in PERTURBATIONS
        for alpha in ALPHAS[1:]
    )
    family = {
        (row["metric_output_id"], row["statistic"]): row for row in holm_rows
    }
    figure2_rows = tuple(
        {
            "acc_cross": row.get("acc_cross"),
            "acc_interval": row.get("acc_interval"),
            "ag": row.get("ag"),
            "ag_interval": row.get("ag_interval"),
            "ag_raw": row.get("ag_raw"),
            "clusters": row.get("clusters"),
            "d_acc": row.get("d_acc"),
            "d_acc_adjusted_p": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("adjusted_p_value"),
            "d_acc_favorable": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("favorable"),
            "d_acc_interval": row.get("d_acc_interval"),
            "d_acc_rank": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("rank"),
            "d_acc_raw_p": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("raw_p_value"),
            "d_acc_rejected": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("rejected"),
            "d_ag": row.get("d_ag"),
            "d_ag_adjusted_p": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("adjusted_p_value"),
            "d_ag_favorable": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("favorable"),
            "d_ag_interval": row.get("d_ag_interval"),
            "d_ag_rank": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("rank"),
            "d_ag_raw_p": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("raw_p_value"),
            "d_ag_rejected": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("rejected"),
            "metric_output_id": row["metric_output_id"],
            "observation_count": row.get("observation_count"),
            "state": row["state"],
        }
        for row in alignment_rows
    )
    return {
        "measurement_bridge": {
            "state": "complete",
            "bridge_sha256": REAL_MEASUREMENT_BRIDGE_SHA256,
            "bridge_row_count": len(record_ids) * len(condition_ids),
            "allowed_step29_payloads": ("record_measurements.jsonl",),
        },
        "well_observation_rows": well_observations,
        "alignment_rows": tuple(alignment_rows),
        "bootstrap_rows": tuple(bootstrap_rows),
        "sign_flip_rows": tuple(sign_flip_rows),
        "holm_rows": tuple(holm_rows),
        "figure1_rows": figure1_rows,
        "figure2_rows": figure2_rows,
        "table_rows": figure2_rows,
        "well_condition_rows": tuple(normalized_wells),
    }


def _terminal_marker(run_id: str, status: str, endpoint_state: str) -> tuple[str, bytes]:
    marker_name = "complete.json" if status == "complete" else "failed.json"
    marker_document = {
        "schema": MARKER_SCHEMA_VERSION,
        "run": run_id,
        "run_id": run_id,
        "status": status,
        "endpoint_state": endpoint_state,
    }
    return marker_name, canonical_json_bytes(marker_document)


def _state_counts(
    rows: Sequence[Mapping[str, object]], field: str = "state"
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get(field))
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def _parent_checksum_entries(
    config_document: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    synthetic = bool(config_document.get("synthetic_fixture", False))
    if synthetic:
        return {
            name: {
                "checksum_entries": {},
                "run_id": f"synthetic-{name}",
                "sha256sums_sha256": "synthetic",
            }
            for name in ("step27", "step29", "step31")
        }
    parents = dict(_thaw(config_document).get("parent_artifacts", {}))
    run_ids = {
        "step27": REAL_STEP27_RUN_ID,
        "step29": REAL_STEP29_RUN_ID,
        "step31": REAL_STEP31_RUN_ID,
    }
    result: dict[str, Mapping[str, object]] = {}
    for name in ("step27", "step29", "step31"):
        receipt = dict(parents[name])
        ledger_path = ROOT / str(receipt["relative_path"]) / "SHA256SUMS"
        entries = {
            filename: digest
            for digest, filename in (
                line.split("  ", 1)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            )
        }
        result[name] = {
            "checksum_entries": entries,
            "run_id": run_ids[name],
            "sha256sums_sha256": str(receipt["sha256sums_sha256"]),
        }
    return result


def _condition_receipt_digests(
    *,
    inputs: object,
    config_document: Mapping[str, object],
    receipts: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, str]:
    condition_ids = tuple(str(item) for item in config_document["condition_ids"])
    if bool(config_document.get("synthetic_fixture", False)):
        return {
            condition_id: sha256_hex(
                canonical_json_bytes(
                    {"condition_id": condition_id, "synthetic_fixture": True}
                )
            )
            for condition_id in condition_ids
        }
    record_ids = tuple(str(item) for item in getattr(inputs, "record_ids"))
    return {
        condition_id: sha256_hex(
            b"".join(
                canonical_json_bytes(receipts[(record_id, condition_id)])
                for record_id in record_ids
            )
        )
        for condition_id in condition_ids
        if all((record_id, condition_id) in receipts for record_id in record_ids)
    }


def _preflight_document(
    *,
    inputs: object,
    config_document: Mapping[str, object],
    rematerialization_receipts: Mapping[
        tuple[str, str], Mapping[str, object]
    ],
    status: str,
    endpoint_state: str,
) -> dict[str, object]:
    document = _thaw(config_document)
    frozen = dict(document.get("frozen_identities", {}))
    support = dict(document.get("support_grid", {}))
    denominators = dict(document.get("denominators", {}))
    resource = dict(document.get("resource_policy", {}))
    synthetic = bool(document.get("synthetic_fixture", False))
    condition_ids = tuple(str(item) for item in document["condition_ids"])
    record_ids = tuple(str(item) for item in getattr(inputs, "record_ids"))
    blank_record_ids = tuple(
        str(item) for item in getattr(inputs, "blank_record_ids")
    )
    well_ids = tuple(str(item) for item in getattr(inputs, "well_ids"))
    fold_ids = tuple(int(item) for item in getattr(inputs, "fold_ids"))
    support_value = getattr(inputs, "support_axis_cm1", None)
    if support_value is None:
        support_value = np.asarray(
            getattr(inputs, "native_axis_cm1"), dtype="<f8"
        )[1:]
    support_axis = np.asarray(support_value, dtype="<f8")
    receipt_digests = _condition_receipt_digests(
        inputs=inputs,
        config_document=document,
        receipts=rematerialization_receipts,
    )
    return {
        "condition_count": len(condition_ids),
        "condition_identity": {
            "condition_count": len(condition_ids),
            "condition_ids_sha256": sha256_hex(
                canonical_json_bytes(list(condition_ids))
            ),
            "condition_order": "config_condition_ids",
        },
        "endpoint_state": endpoint_state,
        "matrix_receipts": {
            "blank_condition_matrix_count": len(condition_ids),
            "blank_rows_per_condition": len(blank_record_ids),
            "condition_receipt_digests": receipt_digests,
            "feature_count": int(
                getattr(inputs, "feature_count", support_axis.size)
            ),
            "mixture_condition_matrix_count": len(receipt_digests),
            "mixture_rows_per_condition": len(record_ids),
            "sealed_mixture_receipt_count": (
                len(record_ids) * len(receipt_digests)
                if synthetic
                else len(rematerialization_receipts)
            ),
        },
        "readiness": {
            "conditions": "passed",
            "rematerialization": (
                "passed" if endpoint_state == "complete" else endpoint_state
            ),
            "roles": "passed",
            "source": "passed",
            "support": "passed",
        },
        "record_count": len(record_ids),
        "resource_admission": {
            "blas_threads_per_model_worker": int(
                resource.get("blas_threads_per_model_worker", 1)
            ),
            "memory_budget_bytes": int(
                resource.get("condition_memory_budget_bytes", 0)
            ),
            "model_budget_admitted_jobs": 43,
            "model_peak_bytes_per_job": 1565016064,
            "model_structural_cap": len(fold_ids),
            "pools_overlap": False,
            "positive_batch_bytes": 493321216,
            "rematerialization_max_admitted_jobs": int(
                resource.get("max_condition_workers", 0)
            ),
            "rematerialization_peak_bytes_per_job": int(
                resource.get("p10_peak_estimate_bytes_per_job", 0)
            ),
        },
        "role_identity": {
            "fold_count": len(fold_ids),
            "fold_record_ids_sha256": frozen.get(
                "fold_record_ids_sha256", ["synthetic"] * len(fold_ids)
            ),
            "fold_well_ids_sha256": frozen.get(
                "fold_well_ids_sha256", ["synthetic"] * len(fold_ids)
            ),
            "test_acquisitions_per_fold": denominators.get(
                "test_acquisitions_per_fold", len(record_ids) // len(fold_ids)
            ),
            "test_wells_per_fold": denominators.get(
                "test_wells_per_fold", len(well_ids) // len(fold_ids)
            ),
            "train_acquisitions_per_fold": denominators.get(
                "train_acquisitions_per_fold", 0
            ),
            "train_wells_per_fold": denominators.get(
                "train_wells_per_fold", 0
            ),
            "validation_acquisitions_per_fold": denominators.get(
                "validation_acquisitions_per_fold", 0
            ),
            "validation_wells_per_fold": denominators.get(
                "validation_wells_per_fold", 0
            ),
        },
        "source_identity": {
            "blank_record_count": len(blank_record_ids),
            "blank_record_ids_sha256": frozen.get(
                "blank_record_ids_sha256", "synthetic"
            ),
            "blank_source_members_sha256": frozen.get(
                "blank_source_members_sha256", "synthetic"
            ),
            "blank_well_id": frozen.get(
                "blank_well_id", "synthetic-blank"
            ),
            "mixture_record_count": len(record_ids),
            "mixture_record_ids_sha256": frozen.get(
                "mixture_record_ids_sha256", "synthetic"
            ),
            "mixture_source_members_sha256": frozen.get(
                "mixture_source_members_sha256", "synthetic"
            ),
            "mixture_well_ids_sha256": frozen.get(
                "mixture_well_ids_sha256", "synthetic"
            ),
            "physical_well_count": len(well_ids),
            "target_matrix_sha256": frozen.get(
                "target_matrix_sha256", "synthetic"
            ),
        },
        "status": status,
        "support_identity": {
            "first_cm1": float(support_axis[0]),
            "float32_sha256": support.get(
                "float32_sha256", _array_sha(support_axis, "<f4")
            ),
            "float64_sha256": support.get(
                "float64_sha256", _array_sha(support_axis, "<f8")
            ),
            "last_cm1": float(support_axis[-1]),
            "point_count": int(support_axis.size),
        },
        "synthetic_fixture": synthetic,
    }


def _authority_bridge_document(
    *,
    config_document: Mapping[str, object],
    measurement_bridge: Mapping[str, object],
) -> dict[str, object]:
    synthetic = bool(config_document.get("synthetic_fixture", False))
    return {
        "allowed_step29_payloads": ["record_measurements.jsonl"],
        "measurement_bridge": _thaw(measurement_bridge),
        "parents": _parent_checksum_entries(config_document),
        "step27_run_id": (
            "synthetic-step27" if synthetic else REAL_STEP27_RUN_ID
        ),
        "step29_payload_policy": {
            "allowed_computational_inputs": ["record_measurements.jsonl"],
            "allowed_postcomputation_qc": list(_STEP29_POSTCOMPUTATION_QC),
            "forbidden_inputs": list(_STEP29_FORBIDDEN_INPUTS),
            "preserved_non_authoritative_directories_allowed": False,
        },
        "step29_run_id": (
            "synthetic-step29" if synthetic else REAL_STEP29_RUN_ID
        ),
        "step31_run_id": (
            "synthetic-step31" if synthetic else REAL_STEP31_RUN_ID
        ),
    }


def _manifest(
    *,
    run_id: str,
    status: str,
    endpoint_state: str,
    alpha0_equivalence: Mapping[str, object],
    config_document: Mapping[str, object],
    rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> bytes:
    document = _thaw(config_document)
    row_names = {
        "alignment_results": "alignment_rows",
        "blank_predictions": "blank_prediction_rows",
        "bootstrap_results": "bootstrap_rows",
        "condition_summary": "condition_summary_rows",
        "figure1_source": "figure1_rows",
        "figure2_source": "figure2_rows",
        "holm_family": "holm_rows",
        "model_cells": "model_rows",
        "predictions": "prediction_rows",
        "secondary_table": "table_rows",
        "sign_flip_results": "sign_flip_rows",
        "technical_lod_loq": "technical_lod_loq_rows",
        "validation_scores": "validation_rows",
        "well_conditions": "well_condition_rows",
        "well_observations": "well_observation_rows",
    }
    counts = {name: len(tuple(rows[key])) for name, key in row_names.items()}
    counts.update(
        {
            "artifact_files": len(ARTIFACT_PAYLOAD_FILES) + 2,
            "configured_payloads": len(ARTIFACT_PAYLOAD_FILES),
        }
    )
    states = {
        name: _state_counts(
            rows[key], "metric_state" if name == "figure1_source" else "state"
        )
        for name, key in row_names.items()
    }
    manifest = {
        "alpha0_equivalence": dict(alpha0_equivalence),
        "claim_boundary": str(document.get("claim_boundary")),
        "counts": counts,
        "endpoint_state": endpoint_state,
        "fixed_capacities": {
            "bootstrap_resamples": int(
                document.get("inference", {}).get("bootstrap_resamples", 2000)
            ),
            "condition_count": len(tuple(document["condition_ids"])),
            "holm_slots": int(
                document.get("inference", {}).get("holm_slot_count", 24)
            ),
            "physical_wells": int(
                document.get("denominators", {}).get(
                    "physical_wells",
                    len(
                        {
                            str(row["well_id"])
                            for row in rows["well_condition_rows"]
                        }
                    ),
                )
            ),
            "sign_flip_resamples": int(
                document.get("inference", {}).get(
                    "sign_flip_resamples", 100000
                )
            ),
        },
        "identities": {
            "authority_receipts_sha256": sha256_hex(
                canonical_json_bytes(document.get("authorities", {}))
            ),
            "code_authority": document.get("code_authority", {}),
            "config_sha256": sha256_hex(canonical_json_bytes(document)),
            "environment_authority": document.get("environment_authority", {}),
        },
        "inherited_rulings": document.get("inherited_rulings", {}),
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "payload_order": "config_artifact_payload_files",
        "run_id": run_id,
        "schema": ARTIFACT_SCHEMA_VERSION,
        "states": states,
        "status": status,
    }
    return canonical_json_bytes(manifest)


def _parse_config(path: Path, raw_bytes: bytes) -> Mapping[str, object]:
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise Phase4D4ProtocolBVerifierError(f"config parse: {error}") from error
    if canonical_json_bytes(document) != raw_bytes:
        raise Phase4D4ProtocolBVerifierError("config must be canonical JSON")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D4ProtocolBVerifierError("schema_version mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D4ProtocolBVerifierError("experiment_id mismatch")
    if not bool(document.get("synthetic_fixture", False)) and CONFIG_BYTES and CONFIG_SHA256:
        if len(raw_bytes) != CONFIG_BYTES or sha256_hex(raw_bytes) != CONFIG_SHA256:
            raise Phase4D4ProtocolBVerifierError("frozen config identity mismatch")
    if tuple(document.get("artifact_payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolBVerifierError("artifact payload order mismatch")
    if document.get("claim_boundary") != CLAIM_BOUNDARY:
        raise Phase4D4ProtocolBVerifierError("claim boundary mismatch")
    synthetic_fixture = bool(document.get("synthetic_fixture", False))
    if not synthetic_fixture:
        if document.get("protocol") != "B" or document.get("tier") != "full_domain_core":
            raise Phase4D4ProtocolBVerifierError("protocol/tier mismatch")
        if tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
            raise Phase4D4ProtocolBVerifierError("perturbation ids mismatch")
        if tuple(float(value) for value in document.get("alpha_grid", ())) != ALPHAS:
            raise Phase4D4ProtocolBVerifierError("alpha grid mismatch")
        if tuple(document.get("metric_output_ids", ())) != METRIC_OUTPUT_IDS:
            raise Phase4D4ProtocolBVerifierError("metric output ids mismatch")
        if document.get("model_recipe") != {
            "copy": True,
            "max_iter": 500,
            "n_components_grid": [2, 4, 8, 16, 32],
            "refit_with_validation": False,
            "scale": True,
            "selection": "minimum_validation_macro_normalized_rmse_lowest_components_on_tie",
            "tol": 1e-6,
            "type": "PLSRegression",
        }:
            raise Phase4D4ProtocolBVerifierError("model_recipe mismatch")
        inference = document.get("inference", {})
        if not isinstance(inference, Mapping) or any(
            inference.get(key) != value
            for key, value in {
                "bootstrap_resamples": 2000,
                "holm_slot_count": 24,
                "random_seed": 20260817,
                "sign_flip_resamples": 100000,
            }.items()
        ):
            raise Phase4D4ProtocolBVerifierError("inference mismatch")
        resource = document.get("resource_policy", {})
        if not isinstance(resource, Mapping) or any(
            resource.get(key) != value
            for key, value in {
                "default_condition_workers": 16,
                "default_model_workers": 5,
                "default_verifier_condition_workers": 12,
                "default_verifier_model_workers": 4,
            }.items()
        ):
            raise Phase4D4ProtocolBVerifierError("resource_policy mismatch")
        if tuple(document.get("code_authority", {})) != CODE_RELATIVE_PATHS:
            raise Phase4D4ProtocolBVerifierError("code_authority path mismatch")
        if dict(document["code_authority"]) != _code_authority():
            raise Phase4D4ProtocolBVerifierError("code_authority live identity mismatch")
        if document.get("environment_authority") != _environment_authority():
            raise Phase4D4ProtocolBVerifierError("environment_authority mismatch")
        if document.get("trust_anchor") != {
            "config_authority_relative_path": "rpe/runner/phase4_d4_protocol_b_authority.py",
            "config_binds_authority": False,
            "direction": "authority_to_config_only",
        }:
            raise Phase4D4ProtocolBVerifierError("trust_anchor mismatch")
    return _freeze(document)


def _validate_inventory(path: Path) -> tuple[Mapping[str, object], str]:
    if not path.is_dir():
        raise Phase4D4ProtocolBVerifierError("artifact path is not a directory")
    existing = {item.name for item in path.iterdir() if item.is_file()}
    marker_names = tuple(name for name in TERMINAL_MARKERS if (path / name).is_file())
    if len(marker_names) != 1:
        raise Phase4D4ProtocolBVerifierError("artifact terminal marker invalid")
    marker_name = marker_names[0]
    expected = set(ARTIFACT_PAYLOAD_FILES) | {marker_name, "SHA256SUMS"}
    if existing != expected:
        raise Phase4D4ProtocolBVerifierError("artifact inventory mismatch")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if tuple(manifest.get("payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolBVerifierError("manifest payload order mismatch")
    rows = []
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows.append((name, digest))
    if tuple(name for name, _digest in rows) != ARTIFACT_PAYLOAD_FILES + (marker_name,):
        raise Phase4D4ProtocolBVerifierError("checksum ledger order mismatch")
    for name, digest in rows:
        if sha256_file(path / name) != digest:
            raise Phase4D4ProtocolBVerifierError(f"checksum mismatch: {name}")
    return _freeze(manifest), marker_name


def _expected_payloads(
    *,
    inputs: object,
    config_document: Mapping[str, object],
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> tuple[dict[str, bytes], str, bytes, str, str]:
    config_document = _thaw(config_document)
    condition_ids = tuple(str(item) for item in config_document["condition_ids"])
    results = [_fit_condition(condition_id=condition_ids[0], inputs=inputs)]
    alpha0_equivalence = _alpha0_equivalence(config_document)
    if alpha0_equivalence["status"] == "passed":
        for condition_id in condition_ids[1:]:
            results.append(_fit_condition(condition_id=condition_id, inputs=inputs))
        endpoint_state = "complete"
        status = "complete"
        aggregates = _aggregate_complete_rows(
            inputs,
            tuple(row for result in results for row in result["well_rows"]),
            bootstrap_resamples=bootstrap_resamples,
            sign_flip_resamples=sign_flip_resamples,
        )
        measurement_bridge = aggregates["measurement_bridge"]
    else:
        for condition_id in condition_ids[1:]:
            results.append(_closed_rows_for_condition(condition_id=condition_id, inputs=inputs))
        endpoint_state = "failed_alpha0_equivalence"
        status = "failed"
        aggregates = _closed_auxiliary_rows(inputs)
        aggregates["well_condition_rows"] = tuple(
            row for result in results for row in result["well_rows"]
        )
        measurement_bridge = {
            "state": "not_tested_endpoint_closed",
            "bridge_sha256": str(config_document["measurement_bridge_sha256"]),
        }
    model_rows = _sort_model_rows(
        tuple(row for result in results for row in result["model_rows"])
        ,
        condition_ids,
    )
    validation_rows = _sort_validation_rows(
        tuple(row for result in results for row in result["validation_rows"]),
        condition_ids,
    )
    prediction_rows = _sort_prediction_rows(
        tuple(row for result in results for row in result["prediction_rows"]),
        tuple(getattr(inputs, "record_ids")),
        condition_ids,
    )
    blank_prediction_rows = _sort_blank_prediction_rows(
        tuple(row for result in results for row in result["blank_prediction_rows"]),
        tuple(getattr(inputs, "blank_record_ids")),
        condition_ids,
    )
    technical_lod_loq_rows = _sort_synthetic_lod_rows(
        tuple(row for result in results for row in result["lod_rows"]),
        condition_ids,
    )
    condition_summary_rows = _sort_condition_summary_rows(
        tuple(row for result in results for row in result["condition_summary_rows"]),
        condition_ids,
    )
    aggregates["well_condition_rows"] = _sort_well_condition_rows(
        tuple(aggregates["well_condition_rows"]),
        tuple(getattr(inputs, "well_ids")),
        condition_ids,
    )
    run_id = stable_run_id(
        config_sha256=sha256_hex(canonical_json_bytes(config_document)),
        condition_ids_value=condition_ids,
        record_ids=tuple(getattr(inputs, "record_ids")),
        blank_record_ids=tuple(getattr(inputs, "blank_record_ids")),
        endpoint_state=endpoint_state,
    )
    figure_bytes = render_d4_protocol_b_figures(
        figure1_rows=aggregates["figure1_rows"],
        figure2_rows=aggregates["figure2_rows"],
    )
    all_rows = {
        **aggregates,
        "blank_prediction_rows": blank_prediction_rows,
        "condition_summary_rows": condition_summary_rows,
        "model_rows": model_rows,
        "prediction_rows": prediction_rows,
        "technical_lod_loq_rows": technical_lod_loq_rows,
        "validation_rows": validation_rows,
    }
    authority_bridge = _authority_bridge_document(
        config_document=config_document,
        measurement_bridge=measurement_bridge,
    )
    preflight = _preflight_document(
        inputs=inputs,
        config_document=config_document,
        rematerialization_receipts={},
        status=status,
        endpoint_state=endpoint_state,
    )
    payloads = {
        "config.json": canonical_json_bytes(config_document),
        "authority_bridge.json": canonical_json_bytes(authority_bridge),
        "preflight.json": canonical_json_bytes(preflight),
        "alpha0_equivalence.json": canonical_json_bytes(alpha0_equivalence),
        "model_cells.jsonl": jsonl_bytes(model_rows),
        "validation_scores.jsonl": jsonl_bytes(validation_rows),
        "predictions.jsonl": jsonl_bytes(prediction_rows),
        "blank_predictions.jsonl": jsonl_bytes(blank_prediction_rows),
        "well_conditions.jsonl": jsonl_bytes(aggregates["well_condition_rows"]),
        "technical_lod_loq.jsonl": jsonl_bytes(technical_lod_loq_rows),
        "condition_summary.csv": csv_bytes(condition_summary_rows),
        "well_observations.jsonl": jsonl_bytes(aggregates["well_observation_rows"]),
        "alignment_results.jsonl": jsonl_bytes(aggregates["alignment_rows"]),
        "bootstrap_results.jsonl": jsonl_bytes(aggregates["bootstrap_rows"]),
        "sign_flip_results.jsonl": jsonl_bytes(aggregates["sign_flip_rows"]),
        "holm_family.jsonl": jsonl_bytes(aggregates["holm_rows"]),
        "figure1_d4_protocol_b_full_domain.png": figure_bytes["figure1_d4_protocol_b_full_domain.png"],
        "figure1_d4_protocol_b_full_domain.svg": figure_bytes["figure1_d4_protocol_b_full_domain.svg"],
        "figure1_d4_protocol_b_full_domain_data.csv": csv_bytes(aggregates["figure1_rows"]),
        "figure2_d4_protocol_b_full_domain.png": figure_bytes["figure2_d4_protocol_b_full_domain.png"],
        "figure2_d4_protocol_b_full_domain.svg": figure_bytes["figure2_d4_protocol_b_full_domain.svg"],
        "figure2_d4_protocol_b_full_domain_data.csv": csv_bytes(aggregates["figure2_rows"]),
        "d4_protocol_b_full_domain_secondary_table.csv": csv_bytes(aggregates["table_rows"]),
        "manifest.json": _manifest(
            run_id=run_id,
            status=status,
            endpoint_state=endpoint_state,
            alpha0_equivalence=alpha0_equivalence,
            config_document=config_document,
            rows=all_rows,
        ),
    }
    terminal_name, terminal_bytes = _terminal_marker(run_id, status, endpoint_state)
    return payloads, terminal_name, terminal_bytes, run_id, status


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return sha256_hex(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _phase1_source_for_spectrum(spectrum: Spectrum1D, *, order: int) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    record_id = spectrum.spectrum_id.split("::", 1)[1]
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=order,
            record_id=record_id,
            sample_id=str(spectrum.sample_id or record_id),
            class_label=0,
            mineral_name="d4-sugar",
            axis_id=f"native::{_array_sha(spectrum.axis_cm1, dtype='<f8')}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1, dtype="<f8"),
        normalized_intensity_float64_sha256=_array_sha(
            spectrum.intensity, dtype="<f8"
        ),
        provenance=MappingProxyType(
            {
                "license": None,
                "license_status": "not_stated",
                "retrieved_date": "2026-08-24",
                "sha256": "0" * 64,
                "source_artifact": "d4_sugar_low_snr",
                "source_url": "local://d4_sugar_low_snr",
            }
        ),
    )


def _real_run_id(
    *,
    config_sha256: str,
    record_ids: Sequence[str],
    blank_record_ids: Sequence[str],
    condition_ids: Sequence[str],
) -> str:
    return stable_run_id(
        config_sha256=config_sha256,
        condition_ids_value=condition_ids,
        record_ids=record_ids,
        blank_record_ids=blank_record_ids,
        endpoint_state="pre_outcome_identity",
    )


def _parse_condition_id(condition_id: str) -> tuple[str, float]:
    if condition_id == "alpha0":
        return "alpha0", 0.0
    perturbation_id, encoded_alpha = condition_id.split(":", 1)
    return perturbation_id, struct.unpack("<d", bytes.fromhex(encoded_alpha))[0]


def _run_perturbation_cell_single_blas(
    source: object,
    perturbation_id: str,
    phase1_config: object,
    sweep: object,
    *,
    p10_admission: object | None = None,
):
    with threadpool_limits(limits=1, user_api="blas"):
        return run_perturbation_cell(
            source,
            perturbation_id,
            phase1_config,
            sweep,
            p10_admission=p10_admission,
        )


def _project_to_frozen_support(
    *,
    axis: np.ndarray,
    intensity: np.ndarray,
    support: np.ndarray,
    max_gap_cm1: float,
) -> np.ndarray:
    axis = np.asarray(axis, dtype="<f8")
    intensity = np.asarray(intensity, dtype="<f8")
    support = np.asarray(support, dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size:
        raise Phase4D4ProtocolBVerifierError("invalid spectrum for support projection")
    left = int(np.searchsorted(axis, support[0], side="left"))
    if left < axis.size:
        candidate_axis = np.asarray(axis[left:], dtype="<f8")
        if candidate_axis.size == support.size and np.array_equal(candidate_axis, support):
            return np.ascontiguousarray(intensity[left:], dtype="<f4")
    if float(axis[0]) > float(support[0]) or float(axis[-1]) < float(support[-1]):
        raise Phase4D4ProtocolBVerifierError("support projection requires extrapolation")
    right = int(np.searchsorted(axis, support[-1], side="right"))
    native_in_range = axis[left:right]
    if native_in_range.size < 2 or float(np.max(np.diff(native_in_range))) > float(max_gap_cm1):
        raise Phase4D4ProtocolBVerifierError("support projection gap mismatch")
    return np.ascontiguousarray(
        np.interp(support, axis, intensity), dtype="<f4"
    )


def _validate_real_authorities(config_document: Mapping[str, object]) -> None:
    for receipt in config_document.get("authorities", {}).values():
        if "path" in receipt:
            path = ROOT / str(receipt["path"])
            if not path.is_file():
                raise Phase4D4ProtocolBVerifierError(f"missing authority file: {path}")
            if int(receipt["bytes"]) != path.stat().st_size:
                raise Phase4D4ProtocolBVerifierError(
                    f"authority bytes mismatch: {path}"
                )
            if str(receipt["sha256"]) != sha256_file(path):
                raise Phase4D4ProtocolBVerifierError(
                    f"authority sha mismatch: {path}"
                )
            continue
        archive_path = ROOT / str(receipt.get("archive_path", ""))
        member_path = str(receipt.get("member_path", ""))
        if not archive_path.is_file() or not member_path:
            raise Phase4D4ProtocolBVerifierError("archive authority receipt is invalid")
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(member_path)
        if int(receipt["bytes"]) != len(payload):
            raise Phase4D4ProtocolBVerifierError(
                f"authority member bytes mismatch: {archive_path}:{member_path}"
            )
        if str(receipt["sha256"]) != sha256_hex(payload):
            raise Phase4D4ProtocolBVerifierError(
                f"authority member sha mismatch: {archive_path}:{member_path}"
            )


def _validate_real_parents(config_document: Mapping[str, object]) -> Mapping[str, object]:
    parents = dict(config_document.get("parent_artifacts", {}))
    if tuple(sorted(parents)) != ("step27", "step29", "step31"):
        raise Phase4D4ProtocolBVerifierError("three-parent contract mismatch")
    for parent in parents.values():
        relative_path = str(parent.get("relative_path", ""))
        if not relative_path or "non_authoritative_pre_fix" in relative_path:
            raise Phase4D4ProtocolBVerifierError("forbidden parent relative path")
        path = ROOT / relative_path
        if not path.is_dir():
            raise Phase4D4ProtocolBVerifierError(f"missing parent directory: {path}")
        required_inventory = tuple(parent.get("required_inventory", ()))
        if required_inventory:
            existing = {item.name for item in path.iterdir() if item.is_file()}
            if existing != set(required_inventory):
                raise Phase4D4ProtocolBVerifierError(
                    f"parent inventory mismatch: {relative_path}"
                )
        checksum_path = path / "SHA256SUMS"
        if "sha256sums_sha256" in parent:
            if not checksum_path.is_file():
                raise Phase4D4ProtocolBVerifierError(
                    f"missing parent checksum ledger: {relative_path}"
                )
            if sha256_file(checksum_path) != str(parent["sha256sums_sha256"]):
                raise Phase4D4ProtocolBVerifierError(
                    f"parent SHA256SUMS mismatch: {relative_path}"
                )
            entries: dict[str, str] = {}
            for line in checksum_path.read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                if len(digest) != 64 or name in entries:
                    raise Phase4D4ProtocolBVerifierError(
                        f"parent checksum ledger malformed: {relative_path}"
                    )
                entries[name] = digest
            if set(entries) != (set(required_inventory) - {"SHA256SUMS"}):
                raise Phase4D4ProtocolBVerifierError(
                    f"parent checksum ledger inventory mismatch: {relative_path}"
                )
            for name, digest in entries.items():
                if sha256_file(path / name) != digest:
                    raise Phase4D4ProtocolBVerifierError(
                        f"parent checksum mismatch: {relative_path}/{name}"
                    )
        checks = (
            ("config_sha256", "config.json"),
            ("manifest_sha256", "manifest.json"),
            ("gate_sha256", "gate.json"),
            ("complete_sha256", "complete.json"),
            ("failed_sha256", "failed.json"),
        )
        for receipt_name, filename in checks:
            if receipt_name in parent and sha256_file(path / filename) != str(
                parent[receipt_name]
            ):
                raise Phase4D4ProtocolBVerifierError(
                    f"parent receipt mismatch: {relative_path}/{filename}"
                )
    step27_gate = json.loads(
        (ROOT / str(parents["step27"]["relative_path"]) / "gate.json").read_bytes()
    )
    step31_gate = json.loads(
        (ROOT / str(parents["step31"]["relative_path"]) / "gate.json").read_bytes()
    )
    if (
        step27_gate.get("full_domain_core", {}).get("state") != "evaluable"
        or step31_gate.get("full_domain_core", {}).get("ready_model_condition_count")
        != 205
    ):
        raise Phase4D4ProtocolBVerifierError("Step-27/31 readiness mismatch")
    return MappingProxyType(
        {
            "step27_run_id": REAL_STEP27_RUN_ID,
            "step29_run_id": REAL_STEP29_RUN_ID,
            "step31_run_id": REAL_STEP31_RUN_ID,
            "allowed_step29_payloads": ("record_measurements.jsonl",),
            "measurement_bridge_sha256": str(config_document["measurement_bridge_sha256"]),
        }
    )


def _reconstruct_real_inputs(config_document: Mapping[str, object]) -> _RealInputs:
    cohort = load_d4_sugar_cohort(
        ROOT / PROTOCOL_CONFIG_RELATIVE_PATH,
        ROOT / ARCHIVE_RELATIVE_PATH,
    )
    if len(cohort.record_ids) != 7680 or len(cohort.blank_record_ids) != 32:
        raise Phase4D4ProtocolBVerifierError("direct-ZIP cohort denominator mismatch")
    support = np.ascontiguousarray(
        np.asarray(cohort.wavenumber, dtype="<f8")[1:], dtype="<f8"
    )
    if support.shape != (1999,) or _array_sha(support) != SUPPORT_FLOAT64_SHA256:
        raise Phase4D4ProtocolBVerifierError("support float64 identity mismatch")
    if _array_sha(support, "<f4") != SUPPORT_FLOAT32_SHA256:
        raise Phase4D4ProtocolBVerifierError("support float32 identity mismatch")
    well_ids = tuple(dict.fromkeys(str(value) for value in cohort.well_ids))
    if len(well_ids) != 240:
        raise Phase4D4ProtocolBVerifierError("physical-well count mismatch")
    if any(sum(value == well_id for value in cohort.well_ids) != 32 for well_id in well_ids):
        raise Phase4D4ProtocolBVerifierError("physical-well multiplicity mismatch")
    train = {
        index: np.asarray(split.train_indices, dtype=np.int64)
        for index, split in enumerate(cohort.splits)
    }
    validation = {
        index: np.asarray(split.validation_indices, dtype=np.int64)
        for index, split in enumerate(cohort.splits)
    }
    test = {
        index: np.asarray(split.test_indices, dtype=np.int64)
        for index, split in enumerate(cohort.splits)
    }
    if any(
        train[index].size != 4608
        or validation[index].size != 1536
        or test[index].size != 1536
        for index in MODEL_FOLDS
    ):
        raise Phase4D4ProtocolBVerifierError("Protocol-B role denominator mismatch")
    record_ids = tuple(str(value) for value in cohort.record_ids)
    record_to_well = {
        record_id: str(cohort.well_ids[index])
        for index, record_id in enumerate(record_ids)
    }
    record_to_fold = {
        record_ids[int(index)]: fold
        for fold, indexes in test.items()
        for index in indexes
    }
    if set(record_to_fold) != set(record_ids):
        raise Phase4D4ProtocolBVerifierError("test-fold coverage mismatch")
    return _RealInputs(
        condition_ids=tuple(str(item) for item in config_document["condition_ids"]),
        fold_ids=MODEL_FOLDS,
        well_ids=well_ids,
        record_ids=record_ids,
        blank_record_ids=tuple(str(value) for value in cohort.blank_record_ids),
        true_targets=np.asarray(cohort.targets, dtype="<f8"),
        blank_targets=np.asarray(cohort.blank_targets, dtype="<f8"),
        record_to_well=MappingProxyType(record_to_well),
        record_to_fold=MappingProxyType(record_to_fold),
        mixture_matrix=np.ascontiguousarray(
            np.asarray(cohort.intensity, dtype="<f8")[:, 1:], dtype="<f4"
        ),
        blank_matrix=np.ascontiguousarray(
            np.asarray(cohort.blank_intensity, dtype="<f8")[:, 1:], dtype="<f4"
        ),
        native_axis_cm1=np.asarray(cohort.wavenumber, dtype="<f8"),
        native_mixture_intensity=np.asarray(cohort.intensity, dtype="<f8"),
        native_blank_intensity=np.asarray(cohort.blank_intensity, dtype="<f8"),
        train_indices_by_fold=MappingProxyType(train),
        validation_indices_by_fold=MappingProxyType(validation),
        test_indices_by_fold=MappingProxyType(test),
        rounds=tuple(int(value) for value in cohort.rounds),
        repetitions=tuple(int(value) for value in cohort.repetitions),
    )


def _real_condition_matrices(
    inputs: _RealInputs,
    *,
    perturbation_filter: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1 = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    sources = tuple(
        _phase1_source_for_spectrum(
            Spectrum1D(
                spectrum_id=f"d4_sugar_low_snr::{record_id}",
                sample_id=str(inputs.record_to_well[record_id]),
                axis_cm1=np.asarray(inputs.native_axis_cm1, dtype="<f8"),
                intensity=np.asarray(inputs.native_mixture_intensity[index], dtype="<f8"),
            ),
            order=index,
        )
        for index, record_id in enumerate(inputs.record_ids)
    )
    blank_sources = tuple(
        _phase1_source_for_spectrum(
            Spectrum1D(
                spectrum_id=f"d4_blank::{record_id}",
                sample_id="blank",
                axis_cm1=np.asarray(inputs.native_axis_cm1, dtype="<f8"),
                intensity=np.asarray(inputs.native_blank_intensity[index], dtype="<f8"),
            ),
            order=index,
        )
        for index, record_id in enumerate(inputs.blank_record_ids)
    )
    cells = [
        _run_perturbation_cell_single_blas(
            source, perturbation_filter, phase1, sweep
        )
        for source in sources
    ]
    blank_cells = [
        _run_perturbation_cell_single_blas(
            source, perturbation_filter, phase1, sweep
        )
        for source in blank_sources
    ]
    for cell in [*cells, *blank_cells]:
        if getattr(cell.evidence.status, "name", None) != "COMPLETE":
            raise Phase4D4ProtocolBVerifierError(
                f"rematerialization failed for {perturbation_filter}"
            )
    matrices: dict[str, np.ndarray] = {}
    blanks: dict[str, np.ndarray] = {}
    for condition_id in (
        item for item in inputs.condition_ids if item.startswith(perturbation_filter + ":")
    ):
        _name, alpha = _parse_condition_id(condition_id)
        try:
            mixture_outputs = [
                next(
                    record.result.output
                    for record in cell.records
                    if float(record.alpha) == alpha
                )
                for cell in cells
            ]
            blank_outputs = [
                next(
                    record.result.output
                    for record in cell.records
                    if float(record.alpha) == alpha
                )
                for cell in blank_cells
            ]
        except StopIteration as error:
            raise Phase4D4ProtocolBVerifierError(
                f"missing perturbation output for {condition_id}"
            ) from error
        matrices[condition_id] = np.ascontiguousarray(
            np.asarray(
                [
                    _project_to_frozen_support(
                        axis=np.asarray(output.axis_cm1, dtype="<f8"),
                        intensity=np.asarray(output.intensity, dtype="<f8"),
                        support=np.asarray(inputs.native_axis_cm1[1:], dtype="<f8"),
                        max_gap_cm1=3.665,
                    )
                    for output in mixture_outputs
                ],
                dtype="<f4",
            ),
            dtype="<f4",
        )
        blanks[condition_id] = np.ascontiguousarray(
            np.asarray(
                [
                    _project_to_frozen_support(
                        axis=np.asarray(output.axis_cm1, dtype="<f8"),
                        intensity=np.asarray(output.intensity, dtype="<f8"),
                        support=np.asarray(inputs.native_axis_cm1[1:], dtype="<f8"),
                        max_gap_cm1=3.665,
                    )
                    for output in blank_outputs
                ],
                dtype="<f4",
            ),
            dtype="<f4",
        )
    return matrices, blanks


def _real_condition_matrices_and_receipts(
    inputs: _RealInputs,
    *,
    perturbation_filter: str,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[tuple[str, str], dict[str, object]],
]:
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1 = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    sources = tuple(
        _phase1_source_for_spectrum(
            Spectrum1D(
                spectrum_id=f"d4_sugar_low_snr::{record_id}",
                sample_id=str(inputs.record_to_well[record_id]),
                axis_cm1=np.asarray(inputs.native_axis_cm1, dtype="<f8"),
                intensity=np.asarray(inputs.native_mixture_intensity[index], dtype="<f8"),
            ),
            order=index,
        )
        for index, record_id in enumerate(inputs.record_ids)
    )
    blank_sources = tuple(
        _phase1_source_for_spectrum(
            Spectrum1D(
                spectrum_id=f"d4_blank::{record_id}",
                sample_id="blank",
                axis_cm1=np.asarray(inputs.native_axis_cm1, dtype="<f8"),
                intensity=np.asarray(inputs.native_blank_intensity[index], dtype="<f8"),
            ),
            order=index,
        )
        for index, record_id in enumerate(inputs.blank_record_ids)
    )
    cells = [
        _run_perturbation_cell_single_blas(
            source, perturbation_filter, phase1, sweep
        )
        for source in sources
    ]
    blank_cells = [
        _run_perturbation_cell_single_blas(
            source, perturbation_filter, phase1, sweep
        )
        for source in blank_sources
    ]
    for cell in [*cells, *blank_cells]:
        if getattr(cell.evidence.status, "name", None) != "COMPLETE":
            raise Phase4D4ProtocolBVerifierError(
                f"rematerialization failed for {perturbation_filter}"
            )
    matrices: dict[str, np.ndarray] = {}
    blanks: dict[str, np.ndarray] = {}
    receipts: dict[tuple[str, str], dict[str, object]] = {}
    for condition_id in (
        item for item in inputs.condition_ids if item.startswith(perturbation_filter + ":")
    ):
        _name, alpha = _parse_condition_id(condition_id)
        mixture_rows = []
        blank_rows = []
        for record_id, cell in zip(inputs.record_ids, cells, strict=True):
            try:
                output = next(
                    record.result.output
                    for record in cell.records
                    if float(record.alpha) == alpha
                )
            except StopIteration as error:
                raise Phase4D4ProtocolBVerifierError(
                    f"missing perturbation output for {condition_id}"
                ) from error
            projected = _project_to_frozen_support(
                axis=np.asarray(output.axis_cm1, dtype="<f8"),
                intensity=np.asarray(output.intensity, dtype="<f8"),
                support=np.asarray(inputs.native_axis_cm1[1:], dtype="<f8"),
                max_gap_cm1=3.665,
            )
            mixture_rows.append(projected)
            receipts[(record_id, condition_id)] = {
                "alpha": alpha,
                "condition_id": condition_id,
                "fold": inputs.record_to_fold[record_id],
                "native_axis_sha256": _array_sha(
                    np.asarray(output.axis_cm1, dtype="<f8"),
                    "<f8",
                ),
                "native_intensity_sha256": _array_sha(
                    np.asarray(output.intensity, dtype="<f8"),
                    "<f8",
                ),
                "perturbation_id": perturbation_filter,
                "projected_row_sha256": _array_sha(projected, "<f4"),
                "record_id": record_id,
                "record_order": len(mixture_rows) - 1,
                "state": "complete",
                "well_id": inputs.record_to_well[record_id],
            }
        for cell in blank_cells:
            try:
                output = next(
                    record.result.output
                    for record in cell.records
                    if float(record.alpha) == alpha
                )
            except StopIteration as error:
                raise Phase4D4ProtocolBVerifierError(
                    f"missing blank perturbation output for {condition_id}"
                ) from error
            blank_rows.append(
                _project_to_frozen_support(
                    axis=np.asarray(output.axis_cm1, dtype="<f8"),
                    intensity=np.asarray(output.intensity, dtype="<f8"),
                    support=np.asarray(inputs.native_axis_cm1[1:], dtype="<f8"),
                    max_gap_cm1=3.665,
                )
            )
        matrices[condition_id] = np.ascontiguousarray(
            np.asarray(mixture_rows, dtype="<f8"),
            dtype="<f4",
        )
        blanks[condition_id] = np.ascontiguousarray(
            np.asarray(blank_rows, dtype="<f8"),
            dtype="<f4",
        )
    return matrices, blanks, receipts


def _real_alpha0_rematerialization_receipts(
    inputs: _RealInputs,
) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (record_id, "alpha0"): {
            "alpha": 0.0,
            "condition_id": "alpha0",
            "fold": inputs.record_to_fold[record_id],
            "native_axis_sha256": _array_sha(
                np.asarray(inputs.native_axis_cm1, dtype="<f8"),
                "<f8",
            ),
            "native_intensity_sha256": _array_sha(
                np.asarray(inputs.native_mixture_intensity[index], dtype="<f8"),
                "<f8",
            ),
            "perturbation_id": "alpha0",
            "projected_row_sha256": _array_sha(
                np.asarray(inputs.mixture_matrix[index], dtype="<f4"),
                "<f4",
            ),
            "record_id": record_id,
            "record_order": index,
            "state": "complete",
            "well_id": inputs.record_to_well[record_id],
        }
        for index, record_id in enumerate(inputs.record_ids)
    }


def _real_fit_condition(
    *,
    condition_id: str,
    condition_matrix: np.ndarray,
    blank_matrix: np.ndarray,
    inputs: _RealInputs,
    config_sha256: str,
) -> Mapping[str, object]:
    matrix = np.ascontiguousarray(condition_matrix, dtype="<f4")
    blanks = np.ascontiguousarray(blank_matrix, dtype="<f4")
    if matrix.shape != (len(inputs.record_ids), 1999):
        raise Phase4D4ProtocolBVerifierError("condition matrix shape mismatch")
    if blanks.shape != (len(inputs.blank_record_ids), 1999):
        raise Phase4D4ProtocolBVerifierError("blank matrix shape mismatch")
    if not np.isfinite(matrix).all() or not np.isfinite(blanks).all():
        raise Phase4D4ProtocolBVerifierError("PLS2 matrices must be finite")
    model_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    blank_prediction_rows: list[dict[str, object]] = []
    lod_rows: list[dict[str, object]] = []
    predicted_by_record = np.empty_like(inputs.true_targets, dtype="<f8")
    for fold in inputs.fold_ids:
        train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
        validation = np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)
        test = np.asarray(inputs.test_indices_by_fold[fold], dtype=np.int64)
        candidates: list[tuple[float, int, PLSRegression]] = []
        for n_components in N_COMPONENTS_GRID:
            estimator = PLSRegression(
                n_components=n_components,
                scale=True,
                max_iter=500,
                tol=1e-6,
                copy=True,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with threadpool_limits(limits=1, user_api="blas"):
                        estimator.fit(matrix[train], inputs.true_targets[train])
                        validation_prediction = np.asarray(
                            estimator.predict(matrix[validation]),
                            dtype="<f8",
                        )
            except (Warning, ValueError, ArithmeticError) as error:
                raise _RealModelLifecycleError(
                    condition_id=condition_id,
                    fold=fold,
                    n_components=n_components,
                    category=type(error).__name__,
                    message=str(error),
                    partial={
                        "model_rows": tuple(model_rows),
                        "validation_rows": tuple(validation_rows),
                        "prediction_rows": tuple(prediction_rows),
                        "blank_prediction_rows": tuple(blank_prediction_rows),
                        "lod_rows": tuple(lod_rows),
                    },
                ) from error
            if not np.isfinite(validation_prediction).all():
                raise Phase4D4ProtocolBVerifierError(
                    "validation prediction is nonfinite"
                )
            score = float(
                np.mean(
                    np.sqrt(
                        np.mean(
                            (validation_prediction - inputs.true_targets[validation]) ** 2,
                            axis=0,
                        )
                    )
                    / 0.32
                )
            )
            validation_rows.append(
                {
                    "fold": fold,
                    "condition_id": condition_id,
                    "n_components": n_components,
                    "macro_normalized_rmse": score,
                    "state": "complete",
                }
            )
            candidates.append((score, n_components, estimator))
        _score, selected, estimator = min(candidates, key=lambda item: (item[0], item[1]))
        state_arrays = (
            estimator._x_mean,
            estimator._y_mean,
            estimator._x_std,
            estimator._y_std,
            estimator.x_weights_,
            estimator.y_weights_,
            estimator.x_loadings_,
            estimator.y_loadings_,
            estimator.x_rotations_,
            estimator.y_rotations_,
            estimator.coef_,
            estimator.intercept_,
        )
        if not all(np.isfinite(np.asarray(item)).all() for item in state_arrays):
            raise Phase4D4ProtocolBVerifierError("PLS2 state is nonfinite")
        model_state_digest = sha256_hex(
            canonical_json_bytes(
                {
                    "fold": fold,
                    "selected_n_components": selected,
                    "state_sha256": [
                        _array_sha(np.asarray(item), "<f8") for item in state_arrays
                    ],
                    "n_iter": [int(value) for value in estimator.n_iter_],
                }
            )
        )
        model_rows.append(
            {
                "fold": fold,
                "condition_id": condition_id,
                "train_condition_id": condition_id,
                "validation_condition_id": condition_id,
                "test_condition_id": condition_id,
                "blank_condition_id": condition_id,
                "selected_n_components": selected,
                "refit_with_validation": False,
                "model_state_digest": model_state_digest,
                "state": "complete",
            }
        )
        with threadpool_limits(limits=1, user_api="blas"):
            test_prediction = np.asarray(estimator.predict(matrix[test]), dtype="<f8")
            blank_prediction = np.asarray(estimator.predict(blanks), dtype="<f8")
            train_prediction = np.asarray(estimator.predict(matrix[train]), dtype="<f8")
        if not np.isfinite(test_prediction).all() or not np.isfinite(blank_prediction).all():
            raise Phase4D4ProtocolBVerifierError("PLS2 prediction is nonfinite")
        predicted_by_record[test] = test_prediction
        for index, values in zip(test.tolist(), test_prediction, strict=True):
            prediction_rows.append(
                {
                    "condition_id": condition_id,
                    "config_sha256": config_sha256,
                    "fold": fold,
                    "model_state_digest": model_state_digest,
                    "predicted_targets": [float(value) for value in values],
                    "projected_row_sha256": _array_sha(matrix[index], "<f4"),
                    "record_id": inputs.record_ids[index],
                    "record_order": index,
                    "repetition": inputs.repetitions[index],
                    "round": inputs.rounds[index],
                    "state": "complete",
                    "terminal_state": "complete",
                    "true_targets": [
                        float(value) for value in np.asarray(inputs.true_targets[index], dtype="<f8")
                    ],
                    "well_id": inputs.record_to_well[inputs.record_ids[index]],
                }
            )
        for blank_record_id, values in zip(
            inputs.blank_record_ids, blank_prediction, strict=True
        ):
            blank_prediction_rows.append(
                {
                    "blank_record_id": blank_record_id,
                    "condition_id": condition_id,
                    "fold": fold,
                    "model_state_digest": model_state_digest,
                    "predicted_targets": [float(value) for value in values],
                    "terminal_state": "complete",
                    "state": "complete",
                }
            )
        for analyte_index in range(4):
            sigma = float(np.std(blank_prediction[:, analyte_index], ddof=1))
            target = inputs.true_targets[train, analyte_index]
            values = train_prediction[:, analyte_index]
            centered = target - float(np.mean(target))
            denominator = float(np.sum(centered ** 2))
            slope = (
                float(
                    np.sum(centered * (values - float(np.mean(values)))) / denominator
                )
                if denominator
                else float("nan")
            )
            state = (
                "complete"
                if math.isfinite(slope) and slope > 0.0
                else "not_evaluable_nonpositive_slope"
            )
            lod_rows.append(
                {
                    "analyte_index": analyte_index,
                    "condition_id": condition_id,
                    "fold": fold,
                    "sigma": sigma if math.isfinite(sigma) else None,
                    "slope": slope if math.isfinite(slope) else None,
                    "iupac_lod": float(3.0 * sigma / slope)
                    if state == "complete"
                    else None,
                    "ich_lod": float(3.3 * sigma / slope)
                    if state == "complete"
                    else None,
                    "ich_loq": float(10.0 * sigma / slope)
                    if state == "complete"
                    else None,
                    "state": state,
                }
            )
    prediction_rows.sort(key=lambda row: int(row["record_order"]))
    well_rows = []
    for well_id in inputs.well_ids:
        indexes = [
            index
            for index, record_id in enumerate(inputs.record_ids)
            if inputs.record_to_well[record_id] == well_id
        ]
        loss = float(
            np.mean(
                ((predicted_by_record[indexes] - inputs.true_targets[indexes]) ** 2)
                / (0.32**2)
            )
        )
        well_rows.append(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "fold": inputs.record_to_fold[inputs.record_ids[indexes[0]]],
                "loss": loss,
                "downstream_harm": 0.0,
                "state": "complete",
            }
        )
    errors = predicted_by_record - inputs.true_targets
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    denominator = np.sum(
        (inputs.true_targets - inputs.true_targets.mean(axis=0)) ** 2, axis=0
    )
    r2 = np.where(
        denominator > 0.0,
        1.0 - np.sum(errors**2, axis=0) / denominator,
        1.0,
    )
    summary_row = {
        "condition_id": condition_id,
        "count": len(inputs.record_ids),
        "macro_normalized_rmse": float(np.mean(rmse / 0.32)),
        "macro_mae_mol_l": float(np.mean(mae)),
        "macro_r2": float(np.mean(r2)),
        "state": "complete",
    }
    for analyte_index, target_name in enumerate(_TARGET_NAMES):
        summary_row[f"{target_name}_rmse_mol_l"] = float(rmse[analyte_index])
        summary_row[f"{target_name}_mae_mol_l"] = float(mae[analyte_index])
        summary_row[f"{target_name}_r2"] = float(r2[analyte_index])
    return {
        "model_rows": tuple(model_rows),
        "validation_rows": tuple(validation_rows),
        "prediction_rows": tuple(prediction_rows),
        "blank_prediction_rows": tuple(blank_prediction_rows),
        "well_rows": tuple(well_rows),
        "lod_rows": tuple(lod_rows),
        "condition_summary_rows": (summary_row,),
    }


def _real_closed_rows_for_condition(
    *,
    condition_id: str,
    inputs: _RealInputs,
) -> Mapping[str, object]:
    state = "not_tested_endpoint_closed"
    return {
        "model_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "train_condition_id": condition_id,
                "validation_condition_id": condition_id,
                "test_condition_id": condition_id,
                "blank_condition_id": condition_id,
                "selected_n_components": None,
                "refit_with_validation": False,
                "state": state,
            }
            for fold in inputs.fold_ids
        ),
        "validation_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "n_components": candidate,
                "macro_normalized_rmse": None,
                "state": state,
            }
            for fold in inputs.fold_ids
            for candidate in N_COMPONENTS_GRID
        ),
        "prediction_rows": tuple(
            {
                "record_id": record_id,
                "record_order": index,
                "well_id": inputs.record_to_well[record_id],
                "condition_id": condition_id,
                "state": state,
                "predicted_targets": [None, None, None, None],
            }
            for index, record_id in enumerate(inputs.record_ids)
        ),
        "blank_prediction_rows": tuple(
            {
                "blank_record_id": blank_record_id,
                "fold": fold,
                "condition_id": condition_id,
                "state": state,
                "predicted_targets": [None, None, None, None],
            }
            for fold in inputs.fold_ids
            for blank_record_id in inputs.blank_record_ids
        ),
        "well_rows": tuple(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "state": state,
                "loss": None,
                "downstream_harm": None,
            }
            for well_id in inputs.well_ids
        ),
        "lod_rows": tuple(
            {
                "analyte_index": analyte_index,
                "condition_id": condition_id,
                "fold": fold,
                "sigma": None,
                "slope": None,
                "iupac_lod": None,
                "ich_lod": None,
                "ich_loq": None,
                "state": state,
            }
            for fold in inputs.fold_ids
            for analyte_index in range(4)
        ),
        "condition_summary_rows": (
            {
                "condition_id": condition_id,
                "state": state,
                "macro_normalized_rmse": None,
            },
        ),
    }


def _real_closed_rows_for_model_lifecycle(
    *,
    condition_id: str,
    inputs: _RealInputs,
    failure: _RealModelLifecycleError,
) -> Mapping[str, object]:
    rows = _real_closed_rows_for_condition(condition_id=condition_id, inputs=inputs)
    completed_models = {
        int(row["fold"]): dict(row) for row in failure.partial.get("model_rows", ())
    }
    completed_validation = {
        (int(row["fold"]), int(row["n_components"])): dict(row)
        for row in failure.partial.get("validation_rows", ())
    }
    completed_predictions = {
        str(row["record_id"]): dict(row)
        for row in failure.partial.get("prediction_rows", ())
    }
    completed_blank_predictions = {
        (int(row["fold"]), str(row["blank_record_id"])): dict(row)
        for row in failure.partial.get("blank_prediction_rows", ())
    }
    completed_lod = {
        (int(row["fold"]), int(row["analyte_index"])): dict(row)
        for row in failure.partial.get("lod_rows", ())
    }
    model_rows = []
    validation_rows = []
    for fold in inputs.fold_ids:
        if fold in completed_models:
            model_rows.append(completed_models[fold])
        else:
            model_rows.append(
                {
                    "fold": fold,
                    "condition_id": condition_id,
                    "train_condition_id": condition_id,
                    "validation_condition_id": condition_id,
                    "test_condition_id": condition_id,
                    "blank_condition_id": condition_id,
                    "selected_n_components": failure.n_components
                    if fold == failure.fold
                    else None,
                    "refit_with_validation": False,
                    "state": "failed_model_lifecycle"
                    if fold == failure.fold
                    else "not_tested_endpoint_closed",
                    "failed_n_components": failure.n_components
                    if fold == failure.fold
                    else None,
                    "failure_category": failure.category
                    if fold == failure.fold
                    else None,
                    "failure_message": failure.message
                    if fold == failure.fold
                    else None,
                }
            )
        for n_components in N_COMPONENTS_GRID:
            key = (fold, n_components)
            if key in completed_validation:
                validation_rows.append(completed_validation[key])
            else:
                validation_rows.append(
                    {
                        "fold": fold,
                        "condition_id": condition_id,
                        "n_components": n_components,
                        "macro_normalized_rmse": None,
                        "state": "failed_model_lifecycle"
                        if fold == failure.fold and n_components == failure.n_components
                        else "not_tested_endpoint_closed",
                        "failure_category": failure.category
                        if fold == failure.fold and n_components == failure.n_components
                        else None,
                        "failure_message": failure.message
                        if fold == failure.fold and n_components == failure.n_components
                        else None,
                    }
                )
    return {
        **rows,
        "model_rows": tuple(model_rows),
        "validation_rows": tuple(validation_rows),
        "prediction_rows": tuple(
            completed_predictions.get(str(row["record_id"]), row)
            for row in rows["prediction_rows"]
        ),
        "blank_prediction_rows": tuple(
            completed_blank_predictions.get(
                (int(row["fold"]), str(row["blank_record_id"])), row
            )
            for row in rows["blank_prediction_rows"]
        ),
        "lod_rows": tuple(
            completed_lod.get((int(row["fold"]), int(row["analyte_index"])), row)
            for row in rows["lod_rows"]
        ),
    }


def _real_alpha0_equivalence(
    *,
    config_document: Mapping[str, object],
    alpha0_result: Mapping[str, object],
) -> Mapping[str, object]:
    projections = {
        "model_digest": (
            alpha0_result["model_rows"],
            ("fold", "model_state_digest", "refit_with_validation", "selected_n_components"),
        ),
        "validation_digest": (
            alpha0_result["validation_rows"],
            ("fold", "macro_normalized_rmse", "n_components"),
        ),
        "prediction_digest": (
            alpha0_result["prediction_rows"],
            (
                "condition_id",
                "fold",
                "model_state_digest",
                "predicted_targets",
                "projected_row_sha256",
                "record_id",
                "record_order",
                "repetition",
                "round",
                "terminal_state",
                "true_targets",
                "well_id",
            ),
        ),
        "blank_prediction_digest": (
            alpha0_result["blank_prediction_rows"],
            (
                "blank_record_id",
                "condition_id",
                "fold",
                "model_state_digest",
                "predicted_targets",
                "terminal_state",
            ),
        ),
        "technical_lod_loq_digest": (
            alpha0_result["lod_rows"],
            (
                "analyte_index",
                "condition_id",
                "fold",
                "ich_lod",
                "ich_loq",
                "iupac_lod",
                "sigma",
                "slope",
                "state",
            ),
        ),
    }
    observed = {
        name: sha256_hex(
            jsonl_bytes(
                tuple({field: row[field] for field in fields} for row in rows)
            )
        )
        for name, (rows, fields) in projections.items()
    }
    expected = dict(config_document.get("alpha0_expected_digests", {}))
    mismatch_count = sum(
        1 for key, value in observed.items() if expected.get(key) != value
    )
    return {
        "status": "passed" if mismatch_count == 0 else "failed",
        "mismatch_count": mismatch_count,
        "expected_digests": expected,
        "observed_digests": observed,
    }


def _real_closed_auxiliary_rows(inputs: _RealInputs) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    return _closed_auxiliary_rows(inputs)


def _real_build_measurement_bridge(
    *,
    inputs: _RealInputs,
    config_document: Mapping[str, object],
    rematerialization_receipts: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    parents = dict(_thaw(config_document).get("parent_artifacts", {}))
    step27_path = (
        ROOT
        / str(parents["step27"]["relative_path"])
        / "record_conditions.jsonl"
    )
    step29_path = (
        ROOT
        / str(parents["step29"]["relative_path"])
        / "record_measurements.jsonl"
    )
    bridge_hasher = hashlib.sha256()
    normalized_bytes = 0
    row_count = 0
    mismatch = {
        "missing_key_count": 0,
        "extra_key_count": 0,
        "duplicate_key_count": 0,
        "state_mismatch_count": 0,
        "metric_receipt_mismatch_count": 0,
        "cwt_receipt_mismatch_count": 0,
        "rematerialization_mismatch_count": 0,
    }
    baseline_values: dict[str, tuple[float, ...]] = {}
    sums: dict[tuple[str, str, str], list[float]] = {}
    seen: set[tuple[str, str]] = set()
    with step27_path.open(encoding="utf-8") as step27_stream, step29_path.open(
        encoding="utf-8"
    ) as step29_stream:
        try:
            paired_rows = zip(step27_stream, step29_stream, strict=True)
            for row_index, (step27_line, step29_line) in enumerate(paired_rows):
                parent = json.loads(step27_line)
                measurement = json.loads(step29_line)
                record_index, condition_index = divmod(
                    row_index, len(inputs.condition_ids)
                )
                if record_index >= len(inputs.record_ids):
                    mismatch["extra_key_count"] += 1
                    continue
                record_id = inputs.record_ids[record_index]
                condition_id = inputs.condition_ids[condition_index]
                key = (record_id, condition_id)
                if key in seen:
                    mismatch["duplicate_key_count"] += 1
                seen.add(key)
                if (
                    (str(parent.get("record_id")), str(parent.get("condition_id")))
                    != key
                    or (
                        str(measurement.get("record_id")),
                        str(measurement.get("condition_id")),
                    )
                    != key
                ):
                    mismatch["missing_key_count"] += 1
                perturbation_id, alpha = _parse_condition_id(condition_id)
                derived = {
                    "alpha": alpha,
                    "condition_id": condition_id,
                    "fold": inputs.record_to_fold[record_id],
                    "perturbation_id": perturbation_id,
                    "record_id": record_id,
                    "record_order": record_index,
                    "well_id": inputs.record_to_well[record_id],
                    "state": "complete",
                }
                if any(
                    measurement.get(name) != value
                    for name, value in derived.items()
                ):
                    mismatch["state_mismatch_count"] += 1
                if parent.get("state") != measurement.get("state"):
                    mismatch["state_mismatch_count"] += 1
                sealed = rematerialization_receipts.get(key)
                if sealed is None or any(
                    measurement.get(name) != sealed.get(name)
                    for name in (
                        "native_axis_sha256",
                        "native_intensity_sha256",
                        "projected_row_sha256",
                    )
                ):
                    mismatch["rematerialization_mismatch_count"] += 1
                metric_projection = []
                values: list[float] = []
                metrics = tuple(measurement.get("metric_values", ()))
                if tuple(item.get("output_id") for item in metrics) != METRIC_OUTPUT_IDS:
                    mismatch["metric_receipt_mismatch_count"] += 1
                for metric_output_id, item in zip(
                    METRIC_OUTPUT_IDS, metrics, strict=False
                ):
                    parent_item = parent.get("metrics", {}).get(metric_output_id, {})
                    projected = {
                        "output_id": metric_output_id,
                        "state": item.get("state"),
                        "result_sha256": item.get("result_digest"),
                        "diagnostics_sha256": item.get("diagnostics_digest"),
                    }
                    if any(
                        parent_item.get(name) != projected.get(name)
                        for name in (
                            "state",
                            "result_sha256",
                            "diagnostics_sha256",
                        )
                    ):
                        mismatch["metric_receipt_mismatch_count"] += 1
                    value = float(item.get("value"))
                    if not math.isfinite(value):
                        raise Phase4D4ProtocolBVerifierError(
                            f"nonfinite Step-29 metric value: {key}/{metric_output_id}"
                        )
                    values.append(value)
                    metric_projection.append(projected)
                cwt = {
                    "state": measurement.get("cwt", {}).get("state"),
                    "diagnostics_sha256": measurement.get("cwt", {}).get(
                        "diagnostics_digest"
                    ),
                    "peak_list_sha256": measurement.get("cwt", {}).get(
                        "peak_list_digest"
                    ),
                    "warning_sha256": measurement.get("cwt", {}).get(
                        "warning_digest"
                    ),
                }
                if any(
                    parent.get("cwt", {}).get(name) != value
                    for name, value in cwt.items()
                ):
                    mismatch["cwt_receipt_mismatch_count"] += 1
                projection = {
                    **derived,
                    "native_axis_sha256": measurement.get("native_axis_sha256"),
                    "native_intensity_sha256": measurement.get(
                        "native_intensity_sha256"
                    ),
                    "projected_row_sha256": measurement.get("projected_row_sha256"),
                    "metrics": metric_projection,
                    "cwt": cwt,
                }
                raw = canonical_json_bytes(projection)
                bridge_hasher.update(raw)
                normalized_bytes += len(raw)
                row_count += 1
                if condition_id == "alpha0":
                    baseline_values[record_id] = tuple(values)
                else:
                    baseline = baseline_values.get(record_id)
                    if baseline is None:
                        raise Phase4D4ProtocolBVerifierError(
                            f"missing alpha-zero metric row before {key}"
                        )
                    for metric_index, metric_output_id in enumerate(METRIC_OUTPUT_IDS):
                        if metric_output_id in _LOWER_IS_BETTER:
                            harm = values[metric_index] - baseline[metric_index]
                        else:
                            harm = baseline[metric_index] - values[metric_index]
                        bucket = sums.setdefault(
                            (
                                inputs.record_to_well[record_id],
                                condition_id,
                                metric_output_id,
                            ),
                            [0.0, 0.0],
                        )
                        bucket[0] += harm
                        bucket[1] += 1.0
        except ValueError as error:
            raise Phase4D4ProtocolBVerifierError(
                f"measurement bridge row-count mismatch: {error}"
            ) from error
    expected_rows = len(inputs.record_ids) * len(inputs.condition_ids)
    if row_count != expected_rows or len(seen) != expected_rows:
        mismatch["missing_key_count"] += abs(expected_rows - len(seen))
    observed_digest = bridge_hasher.hexdigest()
    expected_bytes = int(
        _thaw(config_document).get("measurement_bridge", {}).get(
            "normalized_bytes", 0
        )
    )
    if (
        observed_digest != str(config_document["measurement_bridge_sha256"])
        or normalized_bytes != expected_bytes
        or any(mismatch.values())
    ):
        raise Phase4D4ProtocolBVerifierError(
            f"measurement bridge mismatch: sha256={observed_digest} bytes={normalized_bytes} counts={mismatch}"
        )
    rows = tuple(
        {
            "acquisition_count": int(
                sums[(well_id, condition_id, metric_output_id)][1]
            ),
            "alpha": _parse_condition_id(condition_id)[1],
            "condition_id": condition_id,
            "downstream_harm": None,
            "metric_harm": sums[(well_id, condition_id, metric_output_id)][0]
            / sums[(well_id, condition_id, metric_output_id)][1],
            "metric_output_id": metric_output_id,
            "perturbation_id": _parse_condition_id(condition_id)[0],
            "state": "complete",
            "well_id": well_id,
        }
        for well_id in inputs.well_ids
        for condition_id in inputs.condition_ids[1:]
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    return (
        {
            "allowed_step29_payloads": ("record_measurements.jsonl",),
            "bridge_row_count": row_count,
            "bridge_sha256": observed_digest,
            "mismatch_counts": mismatch,
            "normalized_bytes": normalized_bytes,
            "state": "complete",
        },
        rows,
    )


def _real_aggregate_complete_rows(
    *,
    inputs: _RealInputs,
    condition_summary_rows: Sequence[Mapping[str, object]],
    well_rows: Sequence[Mapping[str, object]],
    lod_rows: Sequence[Mapping[str, object]],
    measurement_bridge_rows: Sequence[Mapping[str, object]],
    endpoint_state: str,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    del lod_rows
    if endpoint_state != "complete":
        raise Phase4D4ProtocolBVerifierError(
            "real aggregation requires a complete endpoint"
        )
    well_lookup = {
        (str(row["well_id"]), str(row["condition_id"])): dict(row)
        for row in well_rows
    }
    normalized_wells: list[Mapping[str, object]] = []
    for well_id in inputs.well_ids:
        baseline_loss = float(well_lookup[(well_id, "alpha0")]["loss"])
        for condition_id in inputs.condition_ids:
            row = dict(well_lookup[(well_id, condition_id)])
            row["downstream_harm"] = (
                0.0
                if condition_id == "alpha0"
                else float(row["loss"]) - baseline_loss
            )
            normalized_wells.append(row)
    downstream = {
        (str(row["well_id"]), str(row["condition_id"])): float(
            row["downstream_harm"]
        )
        for row in normalized_wells
    }
    well_observations = tuple(
        {
            **dict(row),
            "downstream_harm": downstream[
                (str(row["well_id"]), str(row["condition_id"]))
            ],
            "harm": float(row["metric_harm"]),
            "preferred_direction": metric_preferred_direction(
                str(row["metric_output_id"])
            ),
        }
        for row in measurement_bridge_rows
    )
    expected_observations = (
        len(METRIC_OUTPUT_IDS)
        * len(inputs.well_ids)
        * (len(inputs.condition_ids) - 1)
    )
    if len(well_observations) != expected_observations:
        raise Phase4D4ProtocolBVerifierError(
            "well observation denominator mismatch"
        )
    tables = {
        metric_output_id: tuple(
            AlignmentObservation(
                cluster_id=str(row["well_id"]),
                perturbation_id=str(row["perturbation_id"]),
                alpha=float(row["alpha"]),
                metric_harm=float(row["metric_harm"]),
                downstream_harm=float(row["downstream_harm"]),
            )
            for row in well_observations
            if row["metric_output_id"] == metric_output_id
        )
        for metric_output_id in METRIC_OUTPUT_IDS
    }
    bootstrap_n = 2000 if bootstrap_resamples is None else int(bootstrap_resamples)
    sign_flip_n = (
        100000 if sign_flip_resamples is None else int(sign_flip_resamples)
    )
    reference = tables["mse"]
    reference_gap = alignment_gap(reference)
    reference_acc = cross_perturbation_accuracy(reference)
    reference_boot = bulk_paired_cluster_bootstrap(
        reference,
        reference,
        resamples=bootstrap_n,
        confidence_level=0.95,
        random_seed=20260817,
    )
    alignment_rows: list[Mapping[str, object]] = [
        {
            "acc_cross": reference_acc.accuracy,
            "acc_interval": reference_boot.reference_acc_interval,
            "ag": reference_gap.alignment_gap,
            "ag_interval": reference_boot.reference_ag_interval,
            "ag_raw": reference_gap.raw_alignment_gap,
            "clusters": len(inputs.well_ids),
            "cross_pair_count": reference_acc.pair_count,
            "metric_output_id": "mse",
            "observation_count": len(reference),
            "state": "complete",
        }
    ]
    bootstrap_rows: list[Mapping[str, object]] = []
    sign_flip_rows: list[Mapping[str, object]] = []
    p_values: dict[str, float] = {}
    contrasts: dict[str, float] = {}
    for metric_output_id in METRIC_OUTPUT_IDS[1:]:
        candidate = tables[metric_output_id]
        comparison = compare_alignment(reference, candidate)
        boot = bulk_paired_cluster_bootstrap(
            reference,
            candidate,
            resamples=bootstrap_n,
            confidence_level=0.95,
            random_seed=20260817,
        )
        bootstrap_rows.append(
            {
                "candidate_acc_interval": boot.candidate_acc_interval,
                "candidate_ag_interval": boot.candidate_ag_interval,
                "d_acc_interval": boot.d_acc_interval,
                "d_ag_interval": boot.d_ag_interval,
                "metric_output_id": metric_output_id,
                "resamples": bootstrap_n,
                "state": "complete",
            }
        )
        alignment_rows.append(
            {
                "acc_cross": comparison.candidate_accuracy.accuracy,
                "acc_interval": boot.candidate_acc_interval,
                "ag": comparison.candidate_gap.alignment_gap,
                "ag_interval": boot.candidate_ag_interval,
                "ag_raw": comparison.candidate_gap.raw_alignment_gap,
                "clusters": len(inputs.well_ids),
                "cross_pair_count": comparison.candidate_accuracy.pair_count,
                "d_acc": comparison.d_acc,
                "d_acc_interval": boot.d_acc_interval,
                "d_ag": comparison.d_ag,
                "d_ag_interval": boot.d_ag_interval,
                "metric_output_id": metric_output_id,
                "observation_count": len(candidate),
                "state": "complete",
            }
        )
        for statistic, contributions, contrast in (
            (
                "d_ag",
                [item.value for item in comparison.ag_contribution_differences],
                comparison.d_ag,
            ),
            (
                "d_acc",
                [item.value for item in comparison.acc_contribution_differences],
                comparison.d_acc,
            ),
        ):
            sign = paired_contribution_sign_flip(
                contributions,
                aggregation="sum" if statistic == "d_ag" else "mean",
                resamples=sign_flip_n,
                random_seed=20260817,
            )
            hypothesis_id = f"{metric_output_id}:{statistic}"
            p_values[hypothesis_id] = sign.p_value
            contrasts[hypothesis_id] = contrast
            sign_flip_rows.append(
                {
                    "contrast": contrast,
                    "hypothesis_id": hypothesis_id,
                    "metric_output_id": metric_output_id,
                    "p_value": sign.p_value,
                    "resamples": sign_flip_n,
                    "state": "complete",
                    "statistic": statistic,
                }
            )
    adjusted = {
        row.hypothesis_id: row
        for row in holm_step_down(
            {key: p_values[key] for key in sorted(p_values)},
            alpha=0.05,
        )
    }
    holm_rows: list[Mapping[str, object]] = []
    for metric_output_id in METRIC_OUTPUT_IDS[1:]:
        for statistic in ("d_ag", "d_acc"):
            hypothesis_id = f"{metric_output_id}:{statistic}"
            result = adjusted[hypothesis_id]
            contrast = contrasts[hypothesis_id]
            favorable = contrast > 0.0
            holm_rows.append(
                {
                    "adjusted_p_value": result.adjusted_p_value,
                    "favorable": favorable,
                    "family_size": result.family_size,
                    "hypothesis_id": hypothesis_id,
                    "metric_output_id": metric_output_id,
                    "multiplicity_p_value": p_values[hypothesis_id],
                    "observed_contrast": contrast,
                    "rank": result.rank,
                    "raw_p_value": result.raw_p_value,
                    "rejected": bool(result.rejected and favorable),
                    "state": "tested",
                    "statistic": statistic,
                }
            )
    metric_states = {
        str(row["metric_output_id"]): str(row["state"])
        for row in alignment_rows
    }
    figure1_rows = tuple(
        {
            "alpha": alpha,
            "mean_downstream_harm": float(
                np.mean(
                    [
                        row["downstream_harm"]
                        for row in well_observations
                        if row["metric_output_id"] == metric_output_id
                        and row["perturbation_id"] == perturbation_id
                        and float(row["alpha"]) == alpha
                    ]
                )
            ),
            "mean_metric_harm": float(
                np.mean(
                    [
                        row["metric_harm"]
                        for row in well_observations
                        if row["metric_output_id"] == metric_output_id
                        and row["perturbation_id"] == perturbation_id
                        and float(row["alpha"]) == alpha
                    ]
                )
            ),
            "metric_output_id": metric_output_id,
            "metric_state": metric_states[metric_output_id],
            "perturbation_id": perturbation_id,
        }
        for metric_output_id in METRIC_OUTPUT_IDS
        for perturbation_id in PERTURBATIONS
        for alpha in ALPHAS[1:]
    )
    family = {
        (row["metric_output_id"], row["statistic"]): row for row in holm_rows
    }
    figure2_rows = tuple(
        {
            "acc_cross": row.get("acc_cross"),
            "acc_interval": row.get("acc_interval"),
            "ag": row.get("ag"),
            "ag_interval": row.get("ag_interval"),
            "ag_raw": row.get("ag_raw"),
            "clusters": row.get("clusters"),
            "d_acc": row.get("d_acc"),
            "d_acc_adjusted_p": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("adjusted_p_value"),
            "d_acc_favorable": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("favorable"),
            "d_acc_interval": row.get("d_acc_interval"),
            "d_acc_rank": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("rank"),
            "d_acc_raw_p": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("raw_p_value"),
            "d_acc_rejected": family.get(
                (str(row["metric_output_id"]), "d_acc"), {}
            ).get("rejected"),
            "d_ag": row.get("d_ag"),
            "d_ag_adjusted_p": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("adjusted_p_value"),
            "d_ag_favorable": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("favorable"),
            "d_ag_interval": row.get("d_ag_interval"),
            "d_ag_rank": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("rank"),
            "d_ag_raw_p": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("raw_p_value"),
            "d_ag_rejected": family.get(
                (str(row["metric_output_id"]), "d_ag"), {}
            ).get("rejected"),
            "metric_output_id": row["metric_output_id"],
            "observation_count": row.get("observation_count"),
            "state": row["state"],
        }
        for row in alignment_rows
    )
    return {
        "condition_summary_rows": tuple(condition_summary_rows),
        "well_condition_rows": tuple(normalized_wells),
        "well_observation_rows": well_observations,
        "alignment_rows": tuple(alignment_rows),
        "bootstrap_rows": tuple(bootstrap_rows),
        "sign_flip_rows": tuple(sign_flip_rows),
        "holm_rows": tuple(holm_rows),
        "figure1_rows": figure1_rows,
        "figure2_rows": figure2_rows,
        "table_rows": figure2_rows,
    }


def _real_compare_payloads(
    path: Path,
    payloads: Mapping[str, bytes],
    *,
    terminal_name: str,
    terminal_bytes: bytes,
) -> None:
    for name in ARTIFACT_PAYLOAD_FILES:
        actual = (path / name).read_bytes()
        expected = payloads[name]
        if actual != expected:
            raise Phase4D4ProtocolBVerifierError(f"semantic payload mismatch: {name}")
    if (path / terminal_name).read_bytes() != terminal_bytes:
        raise Phase4D4ProtocolBVerifierError(
            f"semantic payload mismatch: {terminal_name}"
        )
    expected_checksums = write_sha256sums(payloads, terminal_name, terminal_bytes)
    if (path / "SHA256SUMS").read_bytes() != expected_checksums:
        raise Phase4D4ProtocolBVerifierError("semantic payload mismatch: SHA256SUMS")


def _real_expected_artifact(
    *,
    path: Path,
    rematerialization_worker_count: int,
    model_worker_count: int,
) -> tuple[dict[str, bytes], str, bytes, str, str, str, int]:
    del rematerialization_worker_count
    config_document = _thaw(_parse_config(path / "config.json", (path / "config.json").read_bytes()))
    _validate_real_authorities(config_document)
    parent_bridge = _validate_real_parents(config_document)
    inputs = _reconstruct_real_inputs(config_document)
    config_sha256 = sha256_hex((path / "config.json").read_bytes())
    rematerialization_receipts = _real_alpha0_rematerialization_receipts(inputs)
    alpha0_result = _real_fit_condition(
        condition_id="alpha0",
        condition_matrix=inputs.mixture_matrix,
        blank_matrix=inputs.blank_matrix,
        inputs=inputs,
        config_sha256=config_sha256,
    )
    alpha0_equivalence = _real_alpha0_equivalence(
        config_document=config_document,
        alpha0_result=alpha0_result,
    )
    condition_results: list[Mapping[str, object]] = [alpha0_result]
    endpoint_state = "complete"
    status = "complete"
    lifecycle_failure: _RealModelLifecycleError | None = None
    if alpha0_equivalence["status"] != "passed":
        endpoint_state = "failed_alpha0_equivalence"
        status = "failed"
        for condition_id in inputs.condition_ids[1:]:
            condition_results.append(
                _real_closed_rows_for_condition(condition_id=condition_id, inputs=inputs)
            )
        aggregates = {
            "condition_summary_rows": tuple(
                row
                for result in condition_results
                for row in result["condition_summary_rows"]
            ),
            "well_condition_rows": tuple(
                row for result in condition_results for row in result["well_rows"]
            ),
        }
        aggregates.update(_real_closed_auxiliary_rows(inputs))
        measurement_bridge = {
            "state": "not_tested_endpoint_closed",
            "bridge_sha256": str(config_document["measurement_bridge_sha256"]),
        }
    else:
        for perturbation_id in ("p08", "p09", "p10", "p11", "p12"):
            if lifecycle_failure is not None:
                break
            matrices, blank_matrices, batch_receipts = (
                _real_condition_matrices_and_receipts(
                    inputs,
                    perturbation_filter=perturbation_id,
                )
            )
            rematerialization_receipts.update(batch_receipts)
            for condition_id in (
                item
                for item in inputs.condition_ids
                if item.startswith(perturbation_id + ":")
            ):
                try:
                    condition_results.append(
                        _real_fit_condition(
                            condition_id=condition_id,
                            condition_matrix=matrices[condition_id],
                            blank_matrix=blank_matrices[condition_id],
                            inputs=inputs,
                            config_sha256=config_sha256,
                        )
                    )
                except _RealModelLifecycleError as error:
                    lifecycle_failure = error
                    condition_results.append(
                        _real_closed_rows_for_model_lifecycle(
                            condition_id=condition_id,
                            inputs=inputs,
                            failure=error,
                        )
                    )
                    break
        if lifecycle_failure is None:
            measurement_bridge, measurement_bridge_rows = (
                _real_build_measurement_bridge(
                    inputs=inputs,
                    config_document=config_document,
                    rematerialization_receipts=rematerialization_receipts,
                )
            )
            aggregates = _real_aggregate_complete_rows(
                inputs=inputs,
                condition_summary_rows=tuple(
                    row
                    for result in condition_results
                    for row in result["condition_summary_rows"]
                ),
                well_rows=tuple(
                    row
                    for result in condition_results
                    for row in result["well_rows"]
                ),
                lod_rows=tuple(
                    row for result in condition_results for row in result["lod_rows"]
                ),
                measurement_bridge_rows=measurement_bridge_rows,
                endpoint_state="complete",
            )
        else:
            endpoint_state = "failed_model_lifecycle"
            status = "failed"
            completed = {
                str(row["condition_id"])
                for result in condition_results
                for row in result["model_rows"]
            }
            for condition_id in inputs.condition_ids:
                if condition_id not in completed:
                    condition_results.append(
                        _real_closed_rows_for_condition(
                            condition_id=condition_id, inputs=inputs
                        )
                    )
            aggregates = {
                "condition_summary_rows": tuple(
                    row
                    for result in condition_results
                    for row in result["condition_summary_rows"]
                ),
                "well_condition_rows": tuple(
                    row for result in condition_results for row in result["well_rows"]
                ),
            }
            aggregates.update(_real_closed_auxiliary_rows(inputs))
            measurement_bridge = {
                "state": "not_tested_endpoint_closed",
                "bridge_sha256": str(config_document["measurement_bridge_sha256"]),
            }
    model_rows = _sort_model_rows(
        tuple(row for result in condition_results for row in result["model_rows"])
        ,
        inputs.condition_ids,
    )
    validation_rows = _sort_validation_rows(
        tuple(
            row for result in condition_results for row in result["validation_rows"]
        ),
        inputs.condition_ids,
    )
    prediction_rows = _sort_prediction_rows(
        tuple(
            row for result in condition_results for row in result["prediction_rows"]
        ),
        inputs.record_ids,
        inputs.condition_ids,
    )
    blank_prediction_rows = _sort_blank_prediction_rows(
        tuple(
            row
            for result in condition_results
            for row in result["blank_prediction_rows"]
        ),
        inputs.blank_record_ids,
        inputs.condition_ids,
    )
    technical_lod_loq_rows = _sort_real_lod_rows(
        tuple(row for result in condition_results for row in result["lod_rows"]),
        inputs.condition_ids,
    )
    aggregates["well_condition_rows"] = _sort_well_condition_rows(
        tuple(aggregates["well_condition_rows"]),
        inputs.well_ids,
        inputs.condition_ids,
    )
    aggregates["condition_summary_rows"] = _sort_condition_summary_rows(
        tuple(aggregates["condition_summary_rows"]),
        inputs.condition_ids,
    )
    run_id = _real_run_id(
        config_sha256=sha256_hex((path / "config.json").read_bytes()),
        record_ids=inputs.record_ids,
        blank_record_ids=inputs.blank_record_ids,
        condition_ids=inputs.condition_ids,
    )
    all_rows = {
        **aggregates,
        "blank_prediction_rows": blank_prediction_rows,
        "model_rows": model_rows,
        "prediction_rows": prediction_rows,
        "technical_lod_loq_rows": technical_lod_loq_rows,
        "validation_rows": validation_rows,
    }
    manifest = _manifest(
        run_id=run_id,
        status=status,
        endpoint_state=endpoint_state,
        alpha0_equivalence=alpha0_equivalence,
        config_document=config_document,
        rows=all_rows,
    )
    preflight_document = _preflight_document(
        inputs=inputs,
        config_document=config_document,
        rematerialization_receipts=rematerialization_receipts,
        status=status,
        endpoint_state=endpoint_state,
    )
    if lifecycle_failure is not None:
        preflight_document["model_lifecycle_failure"] = {
            "category": lifecycle_failure.category,
            "condition_id": lifecycle_failure.condition_id,
            "fold": lifecycle_failure.fold,
            "message": lifecycle_failure.message,
            "n_components": lifecycle_failure.n_components,
        }
    authority_bridge = canonical_json_bytes(
        _authority_bridge_document(
            config_document=config_document,
            measurement_bridge=measurement_bridge,
        )
    )
    figure_bytes = render_d4_protocol_b_figures(
        figure1_rows=aggregates["figure1_rows"],
        figure2_rows=aggregates["figure2_rows"],
    )
    payloads = {
        "config.json": (path / "config.json").read_bytes(),
        "authority_bridge.json": authority_bridge,
        "preflight.json": canonical_json_bytes(preflight_document),
        "alpha0_equivalence.json": canonical_json_bytes(alpha0_equivalence),
        "model_cells.jsonl": jsonl_bytes(model_rows),
        "validation_scores.jsonl": jsonl_bytes(validation_rows),
        "predictions.jsonl": jsonl_bytes(prediction_rows),
        "blank_predictions.jsonl": jsonl_bytes(blank_prediction_rows),
        "well_conditions.jsonl": jsonl_bytes(aggregates["well_condition_rows"]),
        "technical_lod_loq.jsonl": jsonl_bytes(technical_lod_loq_rows),
        "condition_summary.csv": csv_bytes(aggregates["condition_summary_rows"]),
        "well_observations.jsonl": jsonl_bytes(aggregates["well_observation_rows"]),
        "alignment_results.jsonl": jsonl_bytes(aggregates["alignment_rows"]),
        "bootstrap_results.jsonl": jsonl_bytes(aggregates["bootstrap_rows"]),
        "sign_flip_results.jsonl": jsonl_bytes(aggregates["sign_flip_rows"]),
        "holm_family.jsonl": jsonl_bytes(aggregates["holm_rows"]),
        "figure1_d4_protocol_b_full_domain.png": figure_bytes[
            "figure1_d4_protocol_b_full_domain.png"
        ],
        "figure1_d4_protocol_b_full_domain.svg": figure_bytes[
            "figure1_d4_protocol_b_full_domain.svg"
        ],
        "figure1_d4_protocol_b_full_domain_data.csv": csv_bytes(
            aggregates["figure1_rows"]
        ),
        "figure2_d4_protocol_b_full_domain.png": figure_bytes[
            "figure2_d4_protocol_b_full_domain.png"
        ],
        "figure2_d4_protocol_b_full_domain.svg": figure_bytes[
            "figure2_d4_protocol_b_full_domain.svg"
        ],
        "figure2_d4_protocol_b_full_domain_data.csv": csv_bytes(
            aggregates["figure2_rows"]
        ),
        "d4_protocol_b_full_domain_secondary_table.csv": csv_bytes(
            aggregates["table_rows"]
        ),
        "manifest.json": manifest,
    }
    terminal_document = {
        "schema": MARKER_SCHEMA_VERSION,
        "run": run_id,
        "run_id": run_id,
        "status": status,
        "endpoint_state": endpoint_state,
    }
    terminal_name = "complete.json" if status == "complete" else "failed.json"
    if lifecycle_failure is not None:
        terminal_document["failure"] = {
            "condition_id": lifecycle_failure.condition_id,
            "fold": lifecycle_failure.fold,
            "n_components": lifecycle_failure.n_components,
            "warning_category": lifecycle_failure.category,
            "warning_message": lifecycle_failure.message,
        }
    terminal_bytes = canonical_json_bytes(terminal_document)
    return (
        payloads,
        terminal_name,
        terminal_bytes,
        run_id,
        status,
        endpoint_state,
        len(prediction_rows),
    )


def verify_phase4_d4_protocol_b_from_inputs(
    path: Path,
    *,
    inputs: object,
    config_path: Path,
    rematerialization_worker_count: int,
    model_worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D4ProtocolBVerifierSummary:
    del rematerialization_worker_count, model_worker_count
    path = Path(path)
    _validate_inventory(path)
    config_path = Path(config_path)
    config_document = _parse_config(config_path, config_path.read_bytes())
    expected_payloads, terminal_name, terminal_bytes, run_id, status = _expected_payloads(
        inputs=inputs,
        config_document=config_document,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    for name in ARTIFACT_PAYLOAD_FILES:
        actual = (path / name).read_bytes()
        expected = expected_payloads[name]
        if actual != expected:
            raise Phase4D4ProtocolBVerifierError(f"semantic payload mismatch: {name}")
    if (path / terminal_name).read_bytes() != terminal_bytes:
        raise Phase4D4ProtocolBVerifierError(f"semantic payload mismatch: {terminal_name}")
    expected_checksums = write_sha256sums(expected_payloads, terminal_name, terminal_bytes)
    if (path / "SHA256SUMS").read_bytes() != expected_checksums:
        raise Phase4D4ProtocolBVerifierError("semantic payload mismatch: SHA256SUMS")
    endpoint_state = "complete" if status == "complete" else "failed_alpha0_equivalence"
    return Phase4D4ProtocolBVerifierSummary(
        path=path,
        run_id=run_id,
        status=status,
        endpoint_state=endpoint_state,
        record_count=len(tuple(getattr(inputs, "record_ids"))),
        prediction_row_count=len(tuple(getattr(inputs, "record_ids"))) * len(tuple(getattr(inputs, "condition_ids"))),
    )


def verify_phase4_d4_protocol_b(
    path: Path,
    *,
    rematerialization_worker_count: int = 12,
    model_worker_count: int = 4,
) -> Phase4D4ProtocolBVerifierSummary:
    if rematerialization_worker_count < 1 or model_worker_count < 1:
        raise Phase4D4ProtocolBVerifierError("worker counts must be positive")
    path = Path(path)
    config_path = path / "config.json"
    config_document = _parse_config(config_path, config_path.read_bytes())
    if bool(config_document.get("synthetic_fixture", False)):
        raise Phase4D4ProtocolBVerifierError("synthetic verifier requires explicit inputs")
    _validate_inventory(path)
    (
        expected_payloads,
        terminal_name,
        terminal_bytes,
        run_id,
        status,
        endpoint_state,
        prediction_row_count,
    ) = _real_expected_artifact(
        path=path,
        rematerialization_worker_count=rematerialization_worker_count,
        model_worker_count=model_worker_count,
    )
    _real_compare_payloads(
        path,
        expected_payloads,
        terminal_name=terminal_name,
        terminal_bytes=terminal_bytes,
    )
    return Phase4D4ProtocolBVerifierSummary(
        path=path,
        run_id=run_id,
        status=status,
        endpoint_state=endpoint_state,
        record_count=7680,
        prediction_row_count=prediction_row_count,
    )


__all__ = [
    "Phase4D4ProtocolBVerifierError",
    "Phase4D4ProtocolBVerifierSummary",
    "verify_phase4_d4_protocol_b",
    "verify_phase4_d4_protocol_b_from_inputs",
]
