"""End-to-end walkthrough of the evaluation framework on synthetic spectra.

The script needs no downloaded data. It builds a small classification task,
applies the five perturbation families used in the paper (P08--P12), computes
metric harm for all thirteen spectral quality measures, computes task harm for a
Fixed PCA/logistic classifier, and then reports AG, OC, and their paired
MSE-relative contrasts with cluster-bootstrap intervals, sign-flip tests, and
Holm adjustment.

Run from the repository root:

    python examples/quickstart_synthetic.py

The synthetic numbers illustrate the API only; they are not results from the
paper. Replace ``make_dataset`` and ``fit_task`` with your own spectra and
downstream analysis to evaluate a new measure.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.alignment import (  # noqa: E402
    AlignmentObservation,
    alignment_gap,
    compare_alignment,
    cross_perturbation_accuracy,
    holm_step_down,
    paired_cluster_bootstrap,
    paired_contribution_sign_flip,
)
from rpe.evaluation import (  # noqa: E402
    PeakPairInput,
    PreferredDirection,
    SingleSpectrumInput,
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.methods import load_classical_catalog  # noqa: E402
from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system  # noqa: E402
from rpe.metrics import (  # noqa: E402
    ISLikeStructureToNoiseMetric,
    MAEMetric,
    MSEMetric,
    NMSEMetric,
    PeakDetectionCurvesMetric,
    PearsonRMetric,
    RMSEMetric,
    SAMMetric,
    Wasserstein1Metric,
)
from rpe.perturb import PerturbationContext, load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_perturbations import operator_for  # noqa: E402

# Paper settings: P08 baseline, P09 independent noise, P10 correlated noise,
# P11 global shift, P12 quadratic warp; eight positive strengths.
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80)
CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
PEAK_TOLERANCE_CM1 = 2.0

PAIR_METRICS = {
    "mse": MSEMetric, "rmse": RMSEMetric, "mae": MAEMetric, "nmse": NMSEMetric,
    "sam": SAMMetric, "pearson_r": PearsonRMetric, "wasserstein_1_cm1": Wasserstein1Metric,
}
SINGLE_METRICS = {"is_like_structure_to_noise": ISLikeStructureToNoiseMetric}
PEAK_METRICS = ("precision", "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio")
METRIC_IDS = (*PAIR_METRICS, *SINGLE_METRICS, *PEAK_METRICS)

# Small sizes keep the walkthrough to well under a minute. The paper uses
# 2,000 bootstrap draws and 100,000 sign flips.
N_CLASSES, N_TRAIN, N_TEST = 12, 30, 6
BOOTSTRAP_DRAWS, SIGN_FLIPS = 200, 5000


def make_dataset(rng: np.random.Generator):
    """Shared strong peaks plus one weak, narrow class-specific band."""
    axis = np.arange(400.0, 1800.0 + 1e-9, 2.0)

    def band(center, width, height):
        return height / (1.0 + ((axis - center) / width) ** 2)

    shared = sum(band(c, 8.0, h) for c, h in ((620, 1.0), (1001, 1.6), (1450, 0.8), (1660, 1.2)))
    class_centers = np.linspace(700.0, 1350.0, N_CLASSES)

    def draw(label, count):
        rows = []
        for _ in range(count):
            spectrum = shared * rng.uniform(0.9, 1.1)
            spectrum = spectrum + band(class_centers[label] + rng.normal(0.0, 1.0), 4.0, 0.35)
            rows.append(spectrum + rng.normal(0.0, 0.01, axis.size) + 0.2)
        return rows

    train = [(label, s) for label in range(N_CLASSES) for s in draw(label, N_TRAIN)]
    test = [(label, s) for label in range(N_CLASSES) for s in draw(label, N_TEST)]
    return axis, train, test


def fit_task(train, task_axis, native_axis):
    """Fixed protocol: fit PCA/logistic regression on unperturbed spectra."""
    x = np.stack([np.interp(task_axis, native_axis, s) for _, s in train])
    y = np.array([label for label, _ in train])
    model = make_pipeline(PCA(n_components=10), LogisticRegression(C=1.0, max_iter=1000))
    return model.fit(x, y)


def scalar(result, output_id):
    return next(float(o.value) for o in result.outputs if o.output_id == output_id)


def detect_peaks(system, spectrum):
    result = run_peak_detection_system(system, spectrum)
    if result.status not in (PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING):
        raise RuntimeError(f"peak detector status {result.status.value}")
    return tuple(peak.to_peak1d() for peak in result.peaks)


def metric_values(reference, candidate, reference_peaks, candidate_peaks):
    values = {}
    pair = SpectrumPairInput(reference, candidate)
    for metric_id, metric in PAIR_METRICS.items():
        values[metric_id] = scalar(evaluate_metric(metric(), pair), metric_id)
    for metric_id, metric in SINGLE_METRICS.items():
        values[metric_id] = scalar(evaluate_metric(metric(), SingleSpectrumInput(candidate)), metric_id)
    peaks = evaluate_metric(
        PeakDetectionCurvesMetric(),
        PeakPairInput(reference_peaks, candidate_peaks, PEAK_TOLERANCE_CM1, (0.0,)),
    )
    values.update({metric_id: scalar(peaks, metric_id) for metric_id in PEAK_METRICS})
    return values


def orient(metric_id, value, reference_value):
    """Metric harm: positive means worse spectral quality (paper Eq. 2)."""
    higher_is_better = {"pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1"}
    return reference_value - value if metric_id in higher_is_better else value - reference_value


def main() -> None:
    rng = np.random.default_rng(7)
    native_axis, train, test = make_dataset(rng)
    # A fixed support inside every shifted or warped axis avoids extrapolation.
    task_axis = np.arange(420.0, 1780.0 + 1e-9, 2.0)
    model = fit_task(train, task_axis, native_axis)

    sweep = load_perturbation_sweep_config(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json")
    context = PerturbationContext(sweep.sweep_id, sweep.sha256, sweep.global_seed)
    operators = {pid: operator_for(pid, sweep) for pid in PERTURBATIONS}
    catalog = load_classical_catalog(ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json")
    cwt = next(system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID)

    # Per-cluster sums of metric harm and correctness for each condition.
    harm_sum = defaultdict(float)
    correct = defaultdict(int)
    counts = defaultdict(int)
    for index, (label, intensity) in enumerate(test):
        cluster = f"class_{label:02d}"
        source = Spectrum1D(f"test-{index:04d}", cluster, native_axis, intensity)
        source_peaks = detect_peaks(cwt, source)
        reference_values = metric_values(source, source, source_peaks, source_peaks)
        clean_prediction = model.predict(np.interp(task_axis, native_axis, intensity)[None, :])[0]
        correct[cluster, "clean"] += int(clean_prediction == label)
        counts[cluster] += 1
        for pid, operator in operators.items():
            state = operator.prepare(source, context)
            for alpha in ALPHAS:
                output = operator.apply(source, alpha, state).output
                values = metric_values(source, output, source_peaks, detect_peaks(cwt, output))
                for metric_id in METRIC_IDS:
                    harm_sum[cluster, pid, alpha, metric_id] += orient(
                        metric_id, values[metric_id], reference_values[metric_id]
                    )
                task_input = np.interp(task_axis, output.axis_cm1, output.intensity)[None, :]
                correct[cluster, pid, alpha] += int(model.predict(task_input)[0] == label)

    # One (x, y) observation per cluster, perturbation type, and strength.
    observations = {metric_id: [] for metric_id in METRIC_IDS}
    for cluster, n in counts.items():
        clean_accuracy = correct[cluster, "clean"] / n
        for pid in PERTURBATIONS:
            for alpha in ALPHAS:
                task_harm = clean_accuracy - correct[cluster, pid, alpha] / n
                for metric_id in METRIC_IDS:
                    observations[metric_id].append(AlignmentObservation(
                        cluster, pid, alpha, harm_sum[cluster, pid, alpha, metric_id] / n, task_harm,
                    ))

    reference = observations["mse"]
    print(f"{len(counts)} clusters x {len(PERTURBATIONS)} perturbations x {len(ALPHAS)} strengths")
    print(f"MSE: AG={alignment_gap(reference).alignment_gap:.4f}  "
          f"OC={cross_perturbation_accuracy(reference).accuracy:.4f}\n")
    print(f"{'measure':<28}{'AG':>8}{'OC':>8}{'dAG':>9}{'dAG 95% CI':>20}{'dOC':>9}{'dOC 95% CI':>20}")
    p_values, rows = {}, {}
    for metric_id in METRIC_IDS[1:]:
        comparison = compare_alignment(reference, observations[metric_id])
        interval = paired_cluster_bootstrap(reference, observations[metric_id], resamples=BOOTSTRAP_DRAWS)
        ag_test = paired_contribution_sign_flip(
            [c.value for c in comparison.ag_contribution_differences], aggregation="sum", resamples=SIGN_FLIPS)
        oc_test = paired_contribution_sign_flip(
            [c.value for c in comparison.acc_contribution_differences], aggregation="mean", resamples=SIGN_FLIPS)
        p_values[f"{metric_id}:ag"], p_values[f"{metric_id}:oc"] = ag_test.p_value, oc_test.p_value
        rows[metric_id] = (comparison, interval)
    adjusted = {result.hypothesis_id: result.adjusted_p_value for result in holm_step_down(p_values)}
    for metric_id, (comparison, interval) in rows.items():
        mark_ag = "*" if adjusted[f"{metric_id}:ag"] < 0.05 else " "
        mark_oc = "*" if adjusted[f"{metric_id}:oc"] < 0.05 else " "
        ag_ci = "[{:+.3f}, {:+.3f}]".format(*interval.d_ag_interval)
        oc_ci = "[{:+.3f}, {:+.3f}]".format(*interval.d_acc_interval)
        print(f"{metric_id:<28}{comparison.candidate_gap.alignment_gap:>8.4f}"
              f"{comparison.candidate_accuracy.accuracy:>8.4f}{comparison.d_ag:>+8.4f}{mark_ag}{ag_ci:>20}"
              f"{comparison.d_acc:>+8.4f}{mark_oc}{oc_ci:>20}")
    print("\nPositive dAG = AG(MSE) - AG(measure) and dOC = OC(measure) - OC(MSE) favor the measure.")
    print(f"* Holm-adjusted p < 0.05 across the {len(p_values)} tests above (illustrative sizes).")
    assert all(math.isfinite(p) for p in adjusted.values())


if __name__ == "__main__":
    main()
