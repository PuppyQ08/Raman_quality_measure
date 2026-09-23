# Raman spectral quality measures: a task-based evaluation framework

Code, frozen experiment configurations, and aggregate results for

> Xiuyi Qin. *A Task-Based Framework for Evaluating Raman Spectral Quality
> Measures.* Preprint, 2026.

The framework asks whether a spectral quality measure (MSE, W1, peak F1, ...)
tracks what matters for a downstream analysis. Five controlled perturbation
families, each at eight strengths, produce paired changes in a measure
(**metric harm**, *x*) and in task performance (**task harm**, *y*). Two
summaries compare them:

- **Alignment gap (AG):** how much one monotone *x*-to-*y* curve improves when
  each perturbation type receives its own curve. Smaller is better.
- **Ordering concordance (OC):** how often the measure ranks two conditions from
  different perturbation types in the same order as their task harm. Larger is
  better.

The paper applies the framework to thirteen measures and three public datasets
(bacterial classification, sugar-mixture quantification, and mineral
identification), with downstream models fitted to unperturbed spectra
(**Fixed**) or to each perturbed condition (**Adapted**).

No third-party spectra are redistributed here. The repository contains every
aggregate number shown in the paper, the code that produced them, and scripts
that download and checksum-verify the source datasets.

## Contents

```text
Raman_quality_measure/
├── rpe/                   Python package (source layout; run from the repo root)
│   ├── metrics/           the thirteen spectral quality measures
│   ├── perturb/           perturbation operators (the paper uses P08–P12)
│   ├── alignment/         AG, OC, cluster bootstrap, sign-flip tests, Holm
│   ├── evaluation/        typed Spectrum1D / metric-input contracts
│   ├── io/                dataset adapters and the unified record store
│   ├── downstream/        task cohorts, cosine library matching (minerals)
│   ├── methods/           preprocessing catalog; provides the CWT peak detector
│   ├── runner/            executed experiment pipelines and their verifiers
│   └── stats/, semisynth/ auxiliary statistics and an unused semi-synthetic model
├── tools/                 command-line entry points for each pipeline stage
├── scripts/data/          download and checksum-verify the three source datasets
├── experiments/           frozen JSON configurations for every executed run
├── examples/              data-free walkthrough of the full framework
├── paper/                 aggregate inputs and builders for all paper figures/tables
├── reports/robustness/w1_axis/  aggregate results of the axis/representation controls
├── reports/phase2/, phase3/     design records the peak-detector catalog verifies by checksum
├── metadata/sources.json  official URLs, sizes, and SHA-256 of the source archives
├── docs/DATA.md           source download, verification, and conversion guide
├── tests/                 unit, contract, and fixture-based tests
├── env/                   pinned dependency locks
└── LICENSE                MIT
```

## Quick start

The executed study used Python 3.13.11 on Linux. The commands below were also
checked with Python 3.14 on macOS.

```bash
python -m venv .venv
./.venv/bin/pip install -r env/requirements.lock -r env/phase3-requirements.lock
```

Run all commands from the repository root; the package is not installed as a
wheel.

**1. See the framework end to end (no downloads, about 30 s).**

```bash
./.venv/bin/python examples/quickstart_synthetic.py
```

The script builds a small synthetic classification task, applies P08–P12 at the
paper's eight strengths, computes all thirteen measures, fits a Fixed
PCA/logistic-regression classifier, and prints AG, OC, and MSE-relative
contrasts with bootstrap intervals, sign-flip tests, and Holm adjustment. Its
numbers illustrate the API and are not results from the paper.

**2. Rebuild every figure and table in the paper from the aggregate CSVs.**

```bash
cd paper && make assets PYTHON=../.venv/bin/python
```

See [paper/README.md](paper/README.md). The regenerated `tables/v4/*.tex` are
byte-identical to the committed versions.

**3. Run the data-free test suite.**

```bash
./.venv/bin/python -m unittest \
  tests.test_phase1_contracts tests.test_phase1_fidelity tests.test_phase1_transport \
  tests.test_phase1_snr tests.test_phase1_peak tests.test_phase1_reference_free \
  tests.test_phase1_analytical tests.test_phase1_consistency \
  tests.test_phase1_p1_p5 tests.test_phase1_p6_p7 tests.test_phase1_p8 \
  tests.test_phase1_p9 tests.test_phase1_p10 tests.test_phase1_p11_p12 \
  tests.test_phase3_classical_catalog tests.test_phase4_alignment
```

These tests cover the measures, perturbation operators, the peak-detector
catalog, and the AG/OC statistics. Altogether, 29 of the 85 test modules pass in
a fresh checkout; the other 13 are `test_phase1_formal`,
`test_phase1_perturbed_store`, `test_phase1_preview`, `test_phase1_runner_*`
(4 modules), `test_phase05_d4_runner`, `test_phase05_d5_matcher`,
`test_phase6_submission_bundle`, `test_source_audit`,
`test_unified_validate_cli`, and `test_w1_axis_statistics`. The remaining
modules exercise the experiment runners. They need the source datasets,
intermediate artifacts, or internal design records from the original
workspace.

## How to read this repository

Read the code in the order the paper introduces its ideas.

1. **Spectral quality measures** (paper §1, SI §S1) – `rpe/metrics/`

   | Measure | Output ID in code and CSVs | Module |
   |---|---|---|
   | MSE, RMSE, MAE, NMSE, SAM, Pearson | `mse`, `rmse`, `mae`, `nmse`, `sam`, `pearson_r` | `fidelity.py` |
   | Wasserstein distance W1 | `wasserstein_1_cm1` | `transport.py` |
   | Structure-to-noise ratio S/N | `is_like_structure_to_noise` | `reference_free.py` |
   | Peak precision, recall, F1, artifact ratio, missing ratio | `precision`, `recall`, `f1`, `artifact_peak_ratio`, `missing_peak_ratio` | `peak.py` |

   Peaks are detected with SciPy's continuous-wavelet detector wrapped in
   `rpe/methods/classical/peaks.py`. The paper's setting is catalog system
   `6579e821…` in `experiments/phase3/configs/classical_system_catalog_v1.json`
   (widths 1, 2, 4, 8 cm⁻¹; minimum SNR 2; noise percentile 10). Other
   measures in `rpe/metrics/` (for example, LOD/LOQ and alternative S/N
   definitions) are implemented but not evaluated in the paper.

2. **Controlled perturbations** (paper §3.1, Table 2) – `rpe/perturb/`

   | Paper | Code ID | Module |
   |---|---|---|
   | Baseline distortion | `p08` | `baseline_distortion.py` |
   | Independent (Gaussian) noise | `p09` | `gaussian_noise.py` |
   | Correlated noise | `p10` | `correlated_noise.py` |
   | Global shift | `p11` | `axis_transform.py` |
   | Quadratic warp | `p12` | `axis_transform.py` |

   Strengths, the global seed (20260817), and per-spectrum seed derivation are
   in `experiments/shared/raman_perturbation_sweep_v1.json` and
   `rpe/perturb/sweep.py`. Each operator has `prepare(spectrum, context)` and
   `apply(spectrum, alpha, state)`; the same state is reused across strengths.
   Operators P01–P07 (peak and baseline-residual families) are implemented but
   not used in the paper.

3. **AG, OC, and inference** (paper §3.4–3.5, SI §S2) – `rpe/alignment/`

   - `core.py`: `alignment_gap`, `cross_perturbation_accuracy` (OC),
     `compare_alignment` (paired MSE-relative contrasts and per-cluster
     contributions), `orient_harm`.
   - `inference.py`: `paired_cluster_bootstrap` (2,000 draws in the paper),
     `paired_contribution_sign_flip` (100,000 draws), `holm_step_down`.
   - `bulk.py`: the vectorized bootstrap used by the large runs.

4. **Datasets and downstream tasks** (paper §2, Table 1)

   | Paper | Code prefix | Adapter (`rpe/io/`) | Downstream analysis |
   |---|---|---|---|
   | Bacteria, full data | `d1` | `bacteria_id.py` | PCA + logistic regression (`rpe/runner/d1_bacteria_id.py`, `phase4_d1_*`) |
   | Bacteria, 5/10/20 shot | `d2_5`, `d2_10`, `d2_20` | `bacteria_id.py` | same model; shot selection in `rpe/runner/d2_selection.py` |
   | Sugar mixtures | `d4` | `sugar_mixtures*.py` | multi-output PLS (`rpe/runner/d4_sugar.py`, `phase4_d4_*`) |
   | Minerals (RRUFF) | `d5` | `rruff*.py` | L2-normalized cosine matching (`rpe/downstream/rruff_matching.py`) |

5. **Fixed and Adapted experiments** (paper §3.2, §4) – `rpe/runner/phase4_*`

   Protocol A in code is **Fixed**; protocol B is **Adapted**. For each task,
   `phase4_<task>_protocol_<a|b>.py` perturbs the test spectra, computes the
   thirteen measures, obtains downstream predictions, and computes AG, OC,
   intervals, and tests. The `*_eligibility.py` modules check that every
   cluster and condition is evaluable before the outcome run; `*_verifier.py`
   modules independently recheck a finished run; `*_authority.py` modules bind
   parent artifacts by checksum. `phase4_protocol_ab_contrast.py` computes the
   protocol effects *G_p* and metric interactions, and
   `phase6_publication_core.py` writes the 143-row summary used in the paper.

6. **Axis and representation controls** (paper §3.5, Fig. 4, SI §S5)

   `rpe/runner/w1_axis_common_grid.py` recomputes all measures on a common
   wavenumber grid, `w1_axis_robustness.py` removes the axis families and
   builds the four analyses, `w1_axis_statistics.py` runs the shared bootstrap
   and the 66/726-test Holm families. The configuration is
   `experiments/robustness/w1_axis_v1.json`.

7. **Paper displays** – `paper/`

   Builders turn the aggregate CSVs into the four figures and all tables.

### Naming conventions

The code keeps the identifiers used when the experiments were executed.

| In the paper | In code and CSV files |
|---|---|
| Fixed / Adapted | protocol `a` / `b` (panel suffix `_a`, `_b`) |
| Ordering concordance (OC) | `acc_cross`; its contrast is `d_acc` |
| AG contrast ΔAG = AG(MSE) − AG(m) | `d_ag` |
| Task harm | `downstream_harm` |
| Panel, e.g. 10-shot bacteria, Adapted | `d2_10_b` |
| Native / common-grid representation | `native` / `common_grid` |
| All five / no-axis perturbation sets | `all5` / `no_axis` |

Module names also carry the project stage in which they were written. Phase 1
built the measures and perturbations, Phase 0.5 (`phase05`) the dataset loaders
and downstream baselines, Phase 3 the preprocessing catalog (the paper uses only
its peak detector), Phase 4 the Fixed/Adapted experiments, and Phase 6 the
aggregate publication tables. `w1_axis` modules implement the robustness
controls. Phase 2 (a semi-synthetic background model) and the Phase 3/6
preprocessing-method evidence runners are retained for completeness; they do
not contribute results to the paper.

## Where each paper result comes from

| Paper item | Aggregate data | Builder |
|---|---|---|
| Fig. 1 (framework schematic) | none (illustrative) | `paper/build_v4_assets.py` |
| Fig. 2, SI §S3 (143-row matrix), SI Table S4 (native W1 contrasts) | `paper/data/table_s1_full_alignment.csv` | `build_assets.py` → `build_v4_assets.py` |
| SI Table S2 (representative contrasts) | `paper/data/table_s1_full_alignment.csv` | `build_profiles.py` |
| Fig. 3 (task-harm responses) | `paper/data/figure1_response_data.csv` | `build_v4_assets.py` |
| SI Table S3 (protocol effects *G_p*) | `paper/data/figure1_protocol_effect_data.csv` | `build_assets.py` |
| Fig. 4c,d (W1 interactions) | `paper/data/table_s2_protocol_interactions.csv` | `build_v4_assets.py` |
| Fig. 4a,b, SI Table S5 (33 W1 control contrasts) | `reports/robustness/w1_axis/alignment_summary.csv` (572 rows) | `build_w1_robustness.py` |
| SI Table S6 (AG family contributions) | `reports/robustness/w1_axis/ag_family_decomposition.csv` | `build_w1_robustness.py` |
| OC pair-group decomposition (SI §S5) | `reports/robustness/w1_axis/oc_group_summary.csv` | text |
| SI Table S7 (axis harm shares) | `reports/robustness/w1_axis/task_harm_shares.csv` | `build_w1_robustness.py` |
| MAE/MSE pair diagnostic (SI §S2) | `paper/data/v4/mae_oc_pair_diagnostic.csv` | `build_v4_assets.py` |

`paper/data/w1_robustness/` holds byte-identical copies of the robustness CSVs
used by the builders. Column meanings, intervals, and adjusted *p*-values are
described in [paper/README.md](paper/README.md). Unavailable results are empty
fields, never zeros.

`reports/robustness/w1_axis/` holds the aggregate tables and figures of the
executed robustness run; `SHA256SUMS` in that directory lists the digest of
every file (`shasum -a 256 -c SHA256SUMS`).

## Evaluating a new spectral quality measure

1. Implement the measure. A class following `rpe.evaluation.Metric`
   (see `rpe/metrics/fidelity.py`) works with `evaluate_metric`; any function
   returning a scalar also works.
2. Choose the task, the coordinate convention (native or common grid), and the
   Fixed or Adapted fitting protocol. Keep the downstream predictions identical
   for all measures you compare.
3. For every spectrum, apply each perturbation at each strength, and record the
   measure's change from its unperturbed value, oriented so that larger means
   worse (`orient_harm`). Average within your statistical unit (class, well,
   ...). Compute task harm for the same unit and condition.
4. Build one `AlignmentObservation(cluster_id, perturbation_id, alpha,
   metric_harm, downstream_harm)` per unit and condition, then call
   `compare_alignment` against MSE, `paired_cluster_bootstrap`,
   `paired_contribution_sign_flip`, and `holm_step_down`.

`examples/quickstart_synthetic.py` implements exactly these steps; replace its
`make_dataset` and `fit_task` functions with your own data and analysis.

## Reproducing the experiments from source data

The study used about 50 GB of source and derived data, which are not
redistributed. Reproduction has three levels:

| Level | Needs | What you can do |
|---|---|---|
| 1 | this repository | run the quickstart, data-free tests, and display builders |
| 2 | + source archives | download and verify sources, build unified stores ([docs/DATA.md](docs/DATA.md)) |
| 3 | + the executed intermediate artifacts | rerun or independently verify individual pipeline stages |

The executed pipeline ran in this order. Each stage has an entry point in
`tools/` and a frozen configuration in `experiments/`:

1. Build unified stores: `tools/build_bacteria_id_unified.py`,
   `tools/build_rruff_unified.py`; Sugar is read through `raman-data`
   (`tools/audit_ramanbench.py`).
2. Define task cohorts, splits, and downstream baselines:
   `tools/run_phase05*.py`, `tools/aggregate_phase05*.py`,
   `tools/freeze_d2_selection.py` (`experiments/phase05/`).
3. Check coverage, then run the Fixed and Adapted experiments for each task:
   `tools/run_phase4_<task>_eligibility.py` (D2, D4, D5),
   `tools/run_phase4_<task>_protocol_a.py`,
   `tools/run_phase4_<task>_protocol_b_eligibility.py`,
   `tools/run_phase4_<task>_protocol_b.py` (`experiments/phase4/`).
4. Protocol contrasts and aggregate tables:
   `tools/run_phase4_protocol_ab_contrast.py`,
   `tools/run_phase4_final_figures.py`,
   `tools/run_phase6_publication_core.py`.
5. Robustness controls: `tools/run_w1_axis_robustness.py --stage all`
   (`--artifact-root` points to the workspace holding the source data and the
   results of stages 3 and 4).
6. Displays: `paper/Makefile`.

The formal stages have `build` and `verify` subcommands; `verify --run-path
<run>` independently rechecks a finished run. The runners are the exact
executed implementation, not a single-command pipeline: each formal stage
checks the SHA-256 of its configuration, its parent artifacts, and several
design records from the original workspace. Some of those records are internal
planning documents that are not distributed here, so an end-to-end rerun from
source data requires adapting these identity checks.

## Data availability

- **Bacteria-ID** (Ho et al., 2019): downloaded from the authors' official
  link; no explicit dataset license was located, so it is not mirrored.
- **RRUFF** (Lafuente et al., 2015): eight Raman ZIP archives from the official
  site; redistribution permission was not established, so they are not mirrored.
- **Sugar mixtures** (Zenodo 10.5281/zenodo.10779223): CC BY 4.0 at the
  source; downloaded from Zenodo.

`metadata/sources.json` records each archive's URL, size, and SHA-256.
[docs/DATA.md](docs/DATA.md) gives the download, verification, and conversion
commands. Source discovery and loading checks used RamanBench and
`raman-data` v1.2.6. The experiments use the study's own cohorts, splits, and
downstream runners.

## Citation

Please cite the paper if you use this code or its results (see also
[CITATION.cff](CITATION.cff)):

```bibtex
@misc{qin2026raman,
  author = {Qin, Xiuyi},
  title  = {A Task-Based Framework for Evaluating Raman Spectral Quality Measures},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/PuppyQ08/Raman_quality_measure}
}
```

## License

This repository is released under the [MIT License](LICENSE). Third-party
datasets are not included and remain subject to their providers' terms.
