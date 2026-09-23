from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpe.evaluation import SpectrumPairInput, evaluate_metric
from rpe.io.store import UnifiedDataset
from rpe.metrics import MSEMetric
from rpe.perturb import (
    P11GlobalWavenumberShift,
    P9GaussianWhiteNoise,
    PerturbationContext,
    PerturbationSweepConfig,
    load_perturbation_sweep_config,
)
from rpe.runner.phase1_config import (
    REPO_ROOT,
    code_snapshot_digest,
    code_snapshot_document,
    load_phase1_core_config,
    source_snapshot_digest,
)
from rpe.runner.phase1_selection import (
    Phase1Source,
    load_phase1_source,
    load_source_inventory,
    select_source_rows,
    selected_records_jsonl_bytes,
    source_subset_jsonl_bytes,
)


PREVIEW_SCHEMA_VERSION = "phase1-rruff-mse-response-preview-v1"
PREVIEW_PAYLOAD_FILES = (
    "figure1_mse_response_preview.png",
    "manifest.json",
    "response_curve.csv",
    "response_records.jsonl",
)
_FORMAL_CONFIG_PATH = Path("experiments/phase1/configs/rruff_raw_core10k_v1.json")
_SWEEP_PATH = Path("experiments/shared/raman_perturbation_sweep_v1.json")
_PERTURBATION_IDS = ("p09", "p11")
_CODE_PATHS = (
    "rpe/evaluation/contracts.py",
    "rpe/metrics/fidelity.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/contracts.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/perturb/sweep.py",
    "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_preview.py",
    "rpe/runner/phase1_selection.py",
)
_CSV_FIELDS = (
    "perturbation_id",
    "alpha",
    "alpha_float64_le_hex",
    "spectrum_count",
    "mean_mse",
    "median_mse",
    "q25_mse",
    "q75_mse",
    "min_mse",
    "max_mse",
    "nonzero_mse_count",
    "axis_changed_count",
    "intensity_changed_count",
)


class Phase1PreviewError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase1PreviewSummary:
    path: Path
    sample_count: int
    row_count: int
    curve_row_count: int
    checked_files: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase1PreviewError(path, "must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise Phase1PreviewError(path, "must be a finite real number")
    return converted


def _validate_sources(sources: Sequence[Phase1Source]) -> tuple[Phase1Source, ...]:
    if not isinstance(sources, Sequence) or not sources:
        raise Phase1PreviewError("sources", "must be a nonempty sequence")
    parsed = tuple(sources)
    if any(not isinstance(source, Phase1Source) for source in parsed):
        raise Phase1PreviewError("sources", "must contain only Phase1Source values")
    ranks = tuple(source.selection.selection_rank for source in parsed)
    if ranks != tuple(range(len(parsed))):
        raise Phase1PreviewError(
            "sources.selection_rank", "must be consecutive from zero"
        )
    record_ids = tuple(source.selection.record_id for source in parsed)
    if record_ids != tuple(sorted(record_ids)) or len(set(record_ids)) != len(record_ids):
        raise Phase1PreviewError("sources.record_id", "must be sorted and unique")
    return parsed


def _run_rows(
    sources: tuple[Phase1Source, ...],
    sweep: PerturbationSweepConfig,
) -> list[dict[str, object]]:
    context = PerturbationContext(
        sweep_id=sweep.sweep_id,
        sweep_config_sha256=sweep.sha256,
        global_seed=sweep.global_seed,
    )
    metric = MSEMetric()
    rows: list[dict[str, object]] = []
    for source in sources:
        spectrum = source.spectrum
        operators = (
            ("p09", P9GaussianWhiteNoise(sweep)),
            ("p11", P11GlobalWavenumberShift(sweep)),
        )
        for perturbation_id, operator in operators:
            state = operator.prepare(spectrum, context)
            source_rows: list[dict[str, object]] = []
            for alpha in sweep.alpha_grid:
                result = operator.apply(spectrum, alpha, state)
                metric_result = evaluate_metric(
                    metric,
                    SpectrumPairInput(reference=spectrum, candidate=result.output),
                )
                mse = _finite_float("mse", metric_result.outputs[0].value)
                source_rows.append(
                    {
                        "alpha": float(alpha),
                        "alpha_float64_le_hex": struct.pack("<d", alpha).hex(),
                        "axis_changed": result.axis_changed,
                        "class_label": source.selection.class_label,
                        "intensity_changed": result.intensity_changed,
                        "mineral_name": source.selection.mineral_name,
                        "mse": mse,
                        "noise_mean_square": (
                            result.diagnostics["noise_mean_square"]
                            if perturbation_id == "p09"
                            else None
                        ),
                        "output_spectrum_id": result.output.spectrum_id,
                        "perturbation_id": perturbation_id,
                        "realized_max_abs_offset_cm1": (
                            result.diagnostics["realized_max_abs_offset_cm1"]
                            if perturbation_id == "p11"
                            else None
                        ),
                        "sample_id": source.selection.sample_id,
                        "selection_rank": source.selection.selection_rank,
                        "sigma": (
                            result.diagnostics["sigma"]
                            if perturbation_id == "p09"
                            else None
                        ),
                        "source_record_id": source.selection.record_id,
                        "source_spectrum_id": spectrum.spectrum_id,
                        "state_digest": state.state_digest,
                    }
                )
            source_mse = [float(row["mse"]) for row in source_rows]
            if source_mse[0] != 0.0:
                raise Phase1PreviewError(
                    f"science.{source.selection.record_id}.{perturbation_id}.alpha_zero",
                    "MSE must equal zero",
                )
            if perturbation_id == "p09":
                if source_mse != sorted(source_mse):
                    raise Phase1PreviewError(
                        f"science.{source.selection.record_id}.p09.monotonic",
                        "MSE must be nondecreasing over alpha",
                    )
                positive_sigmas = [float(row["sigma"]) for row in source_rows[1:]]
                if any(sigma > 0.0 and mse <= 0.0 for sigma, mse in zip(
                    positive_sigmas, source_mse[1:], strict=True
                )):
                    raise Phase1PreviewError(
                        f"science.{source.selection.record_id}.p09.positive",
                        "positive sigma must have positive MSE",
                    )
            elif any(value != 0.0 for value in source_mse):
                raise Phase1PreviewError(
                    f"science.{source.selection.record_id}.p11.blindness",
                    "index-aligned intensity MSE must remain zero",
                )
            rows.extend(source_rows)
    return rows


def _curve_rows(
    rows: Sequence[Mapping[str, object]],
    sweep: PerturbationSweepConfig,
) -> list[dict[str, object]]:
    curves: list[dict[str, object]] = []
    for perturbation_id in _PERTURBATION_IDS:
        for alpha in sweep.alpha_grid:
            selected = [
                row
                for row in rows
                if row["perturbation_id"] == perturbation_id
                and row["alpha"] == alpha
            ]
            values = np.asarray([row["mse"] for row in selected], dtype="<f8")
            curves.append(
                {
                    "alpha": float(alpha),
                    "alpha_float64_le_hex": struct.pack("<d", alpha).hex(),
                    "axis_changed_count": sum(
                        bool(row["axis_changed"]) for row in selected
                    ),
                    "intensity_changed_count": sum(
                        bool(row["intensity_changed"]) for row in selected
                    ),
                    "max_mse": float(np.max(values)),
                    "mean_mse": float(np.mean(values)),
                    "median_mse": float(np.median(values)),
                    "min_mse": float(np.min(values)),
                    "nonzero_mse_count": int(np.count_nonzero(values)),
                    "perturbation_id": perturbation_id,
                    "q25_mse": float(np.quantile(values, 0.25)),
                    "q75_mse": float(np.quantile(values, 0.75)),
                    "spectrum_count": len(selected),
                }
            )
    return curves


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical_json_bytes(row) for row in rows))


def _csv_value(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _write_curve_csv(path: Path, curves: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for curve in curves:
            writer.writerow({name: _csv_value(curve[name]) for name in _CSV_FIELDS})


def _write_figure(path: Path, curves: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "savefig.dpi": 150,
        }
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.8), dpi=150)
        for perturbation_id, label, color in (
            ("p09", "P9 Gaussian white noise", "#0072B2"),
            ("p11", "P11 wavenumber shift", "#D55E00"),
        ):
            selected = [
                row for row in curves if row["perturbation_id"] == perturbation_id
            ]
            axis.plot(
                [row["alpha"] for row in selected],
                [row["mean_mse"] for row in selected],
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                color=color,
                label=label,
            )
        axis.set_title("Exploratory Figure 1 prototype: MSE response")
        axis.set_xlabel("Perturbation alpha (frozen grid)")
        axis.set_ylabel("Mean index-aligned MSE")
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.legend(frameon=False)
        figure.text(
            0.5,
            0.01,
            "P11 shifts the physical axis but leaves index-aligned MSE at zero; "
            "P7 not run (missing explicit baseline).",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
        figure.savefig(
            path,
            format="png",
            dpi=150,
            metadata={"Software": "raman-preproc-eval"},
        )
        plt.close(figure)


def _code_snapshot() -> tuple[dict[str, dict[str, int | str]], str]:
    document = code_snapshot_document(REPO_ROOT, _CODE_PATHS)
    return (
        {path: dict(identity) for path, identity in document.items()},
        code_snapshot_digest(REPO_ROOT, _CODE_PATHS),
    )


def build_mse_preview_from_sources(
    output_dir: Path,
    *,
    sources: Sequence[Phase1Source],
    sweep: PerturbationSweepConfig,
    provenance: Mapping[str, object],
) -> Phase1PreviewSummary:
    output = Path(output_dir)
    parsed_sources = _validate_sources(sources)
    if not isinstance(sweep, PerturbationSweepConfig):
        raise Phase1PreviewError("sweep", "must be PerturbationSweepConfig")
    if not isinstance(provenance, Mapping):
        raise Phase1PreviewError("provenance", "must be a mapping")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise Phase1PreviewError("output_dir", str(error)) from error

    rows = _run_rows(parsed_sources, sweep)
    curves = _curve_rows(rows, sweep)
    _write_jsonl(output / "response_records.jsonl", rows)
    _write_curve_csv(output / "response_curve.csv", curves)
    code_document, code_digest = _code_snapshot()
    selected_rows = tuple(source.selection for source in parsed_sources)
    manifest = {
        "claim_boundary": "exploratory_preview_not_final_paper_evidence",
        "code_snapshot": code_document,
        "code_snapshot_sha256": code_digest,
        "environment": {
            "matplotlib": __import__("matplotlib").__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "metric": {
            "axis_policy": "index_aligned",
            "metric_id": "mse",
            "unit": "intensity_squared",
        },
        "mse_interpretation": {
            "p09": "intensity_noise_response_expected_to_increase_with_alpha",
            "p11": "index_aligned_intensity_mse_is_zero_despite_physical_axis_shift",
        },
        "p07": {
            "dependency": "phase2_semisynthetic_or_separately_approved_physical_baseline",
            "reason": "missing_explicit_baseline",
            "status": "not_run_missing_explicit_baseline",
        },
        "payload_files": list(PREVIEW_PAYLOAD_FILES),
        "perturbation_ids": list(_PERTURBATION_IDS),
        "preprocessing": {"interpolation": "none", "normalization": "none"},
        "preview_sample": {
            "class_count": len(
                {source.selection.class_label for source in parsed_sources}
            ),
            "first_record_id": parsed_sources[0].selection.record_id,
            "last_record_id": parsed_sources[-1].selection.record_id,
            "record_count": len(parsed_sources),
            "sample_id_count": len(
                {source.selection.sample_id for source in parsed_sources}
            ),
            "selected_records_sha256": hashlib.sha256(
                selected_records_jsonl_bytes(selected_rows)
            ).hexdigest(),
            "source_subset_sha256": hashlib.sha256(
                source_subset_jsonl_bytes(parsed_sources)
            ).hexdigest(),
            "statistical_role": (
                "deterministic_pipeline_smoke_test_not_stratified_"
                "representative_or_inferential"
            ),
        },
        "provenance": dict(provenance),
        "schema_version": PREVIEW_SCHEMA_VERSION,
        "sweep": {
            "alpha_grid": list(sweep.alpha_grid),
            "bit_generator": sweep.bit_generator,
            "byte_count": sweep.byte_count,
            "config_path": _SWEEP_PATH.as_posix(),
            "global_seed": sweep.global_seed,
            "sha256": sweep.sha256,
            "sweep_id": sweep.sweep_id,
        },
    }
    (output / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    _write_figure(output / "figure1_mse_response_preview.png", curves)
    checksum_lines = "".join(
        f"{_sha256_file(output / name)}  {name}\n"
        for name in PREVIEW_PAYLOAD_FILES
    )
    (output / "SHA256SUMS").write_text(
        checksum_lines, encoding="utf-8", newline="\n"
    )
    return Phase1PreviewSummary(
        path=output,
        sample_count=len(parsed_sources),
        row_count=len(rows),
        curve_row_count=len(curves),
        checked_files=PREVIEW_PAYLOAD_FILES,
    )


def build_rruff_mse_preview(
    output_dir: Path,
    *,
    sample_count: int = 300,
) -> Phase1PreviewSummary:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise Phase1PreviewError("sample_count", "must be a positive integer")
    config = load_phase1_core_config(REPO_ROOT / _FORMAL_CONFIG_PATH)
    if sample_count > config.subset_size:
        raise Phase1PreviewError(
            "sample_count", "must not exceed the formal subset size"
        )
    sweep = load_perturbation_sweep_config(REPO_ROOT / _SWEEP_PATH)
    inventory = load_source_inventory(config)
    formal_selection = select_source_rows(
        inventory,
        global_seed=sweep.global_seed,
        subset_size=config.subset_size,
    )
    preview_rows = formal_selection[:sample_count]
    with UnifiedDataset.open(
        REPO_ROOT / config.source_dataset_path,
        verify_checksums=False,
    ) as dataset:
        sources = tuple(load_phase1_source(dataset, row) for row in preview_rows)
    provenance = {
        "distribution_status": config.distribution_status,
        "experiment_id": config.experiment_id,
        "formal_config_byte_count": config.file_byte_count,
        "formal_config_path": _FORMAL_CONFIG_PATH.as_posix(),
        "formal_config_sha256": config.file_sha256,
        "formal_selected_records_sha256": hashlib.sha256(
            selected_records_jsonl_bytes(formal_selection)
        ).hexdigest(),
        "formal_selected_source_count": len(formal_selection),
        "scientific_config_byte_count": config.scientific_config_byte_count,
        "scientific_config_sha256": config.scientific_config_sha256,
        "selection_algorithm": config.selection_algorithm,
        "selection_rule": "formal_core10k_selection_rank_prefix",
        "source_dataset_id": config.source_dataset_id,
        "source_snapshot_sha256": source_snapshot_digest(config),
    }
    return build_mse_preview_from_sources(
        output_dir,
        sources=sources,
        sweep=sweep,
        provenance=provenance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the exploratory Phase 1 RRUFF MSE-response preview."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = build_rruff_mse_preview(
        arguments.output_dir,
        sample_count=arguments.sample_count,
    )
    print(
        json.dumps(
            {
                "curve_row_count": summary.curve_row_count,
                "path": str(summary.path),
                "row_count": summary.row_count,
                "sample_count": summary.sample_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREVIEW_PAYLOAD_FILES",
    "PREVIEW_SCHEMA_VERSION",
    "Phase1PreviewError",
    "Phase1PreviewSummary",
    "build_mse_preview_from_sources",
    "build_rruff_mse_preview",
    "main",
]
