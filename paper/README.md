# Paper figures, tables, and aggregate data

This directory regenerates every figure and table in the manuscript and its
Supporting Information (SI) from aggregate CSV files. No spectra are read and no
experiment is rerun. The manuscript source itself is distributed with the
preprint.

## Rebuild

```bash
cd paper
make assets PYTHON=../.venv/bin/python
```

The five builders need NumPy and Matplotlib and run in a few seconds:

| Builder | Reads | Writes |
|---|---|---|
| `build_assets.py` | `data/table_s1_full_alignment.csv`, `data/figure1_*.csv`, `data/table_s2_protocol_interactions.csv` | `tables/full_alignment.tex`, `tables/w1_intervals.tex`, `tables/protocol_effects.tex` |
| `build_profiles.py` | `data/table_s1_full_alignment.csv` | `tables/representative_contrasts.tex`, `data/metric_profiles_v2.csv` |
| `build_w1_robustness.py` | `../reports/robustness/w1_axis/*.csv`, checked against its `SHA256SUMS` and copied to `data/w1_robustness/` | `tables/w1_robustness.tex`, `tables/w1_axis_harm_shares.tex`, `tables/w1_ag_family_example.tex` |
| `build_v4_assets.py` | the files above | `figures/v4/*.pdf`, `figures/v4/*.png`, `tables/v4/*.tex`, `provenance/v4_assets.json` |
| `build_v5_figures.py` | synthetic teaching data only | `figures/v5/*.pdf`, `figures/v5/*.png`, `provenance/v5_figures.json` |

The first three builders write row tables with the identifiers used during
execution (protocol A/B, `Acc`). `build_v4_assets.py` converts them to the
paper's names (Fixed/Adapted, OC) without changing any number. A rebuild
reproduces `tables/v4/*.tex`, `data/`, and `provenance/v4_assets.json` byte for
byte; figure files can differ at the byte level across Matplotlib versions.
Side outputs of the first three builders (legacy figures and receipts) are
ignored by Git, and `figures/v4/framework.pdf` is superseded by
`figures/v5/framework.pdf`.

| Output | Paper item |
|---|---|
| `figures/v5/framework.pdf` | Figure 1 (schematic) |
| `figures/v5/perturbations.pdf` | Figure 2 (synthetic teaching examples) |
| `figures/v5/ag_oc.pdf` | Figure 3 (synthetic teaching examples) |
| `figures/v4/alignment.pdf` | Figure 4 |
| `figures/v4/responses.pdf` | Figure 5 |
| `figures/v4/w1_comparison.pdf` | Figure 6 |
| `tables/v4/representative_contrasts.tex` | SI Table S2 |
| `tables/v4/full_alignment.tex` | SI Section S3 (all 143 rows) |
| `tables/v4/protocol_effects.tex` | SI Table S3 |
| `tables/v4/w1_intervals.tex` | SI Table S4 |
| `tables/v4/w1_robustness.tex` | SI Table S5 |
| `tables/v4/w1_ag_family_example.tex` | SI Table S6 |
| `tables/v4/w1_axis_harm_shares.tex` | SI Table S7 |

## Data dictionary

Common identifiers:

- `panel_id`: task and protocol, e.g. `d1_a` (full-data bacteria, Fixed),
  `d2_10_b` (10-shot bacteria, Adapted), `d4_a` (sugar, Fixed), `d5_b`
  (minerals, Adapted). `cell_id` omits the protocol suffix.
- `protocol_id`: `a` = Fixed, `b` = Adapted.
- `metric_output_id`: `mse`, `rmse`, `mae`, `nmse`, `sam`, `pearson_r`,
  `wasserstein_1_cm1` (W1), `is_like_structure_to_noise` (S/N), `precision`,
  `recall`, `f1`, `artifact_peak_ratio`, `missing_peak_ratio`.
- `perturbation_id`: `p08` baseline, `p09` independent noise, `p10`
  correlated noise, `p11` shift, `p12` warp.
- `acc_cross` / `oc`: ordering concordance (OC), not classification accuracy.
- `*_lower`, `*_upper`: pointwise 95% cluster-bootstrap percentile interval.
- `*_adjusted_p_value`, `adjusted_p_*`: Holm-adjusted two-sided Monte Carlo
  sign-flip *p*-value within the family named by `*_family_id`.

### Native five-family results

`data/table_s1_full_alignment.csv` (143 rows = 13 measures × 11 panels). `ag`
and `acc_cross` are each measure's AG and OC. For candidates, `d_ag` =
AG(MSE) − AG(candidate) and `d_acc` = OC(candidate) − OC(MSE); positive values
favor the candidate. Each task/protocol forms a 24-test Holm family.

The fields `d_ag_rejected` and `d_acc_rejected` record *favorable* rejections
only, so an adverse contrast can have `rejected = False` despite an adjusted
*p*-value below 0.05. The paper marks two-sided significance by
`adjusted_p_value < 0.05` and takes the direction from the sign of the
contrast.

`data/metric_profiles_v2.csv` records the two-sided decision for each
candidate/panel pair (favorable, adverse, or not rejected) and the joint
profile over AG and OC.

### Fixed versus Adapted

- `data/figure1_response_data.csv` (440 rows): mean MSE metric harm
  (`metric_x`) and task harm (`downstream_harm`) for each cell, protocol,
  perturbation, and strength (Figure 5).
- `data/figure1_protocol_effect_data.csv` (225 rows): protocol effects
  `g_harm` = mean(task harm Fixed − task harm Adapted). Rows with
  `summary_type = integrated` average over strengths (SI Table S3); the others
  are strength-specific.
- `data/table_s2_protocol_interactions.csv` (65 rows): `delta_ag` and
  `delta_acc` are each measure's own change in AG and OC from Fixed to Adapted
  (Adapted minus Fixed). For candidates, `i_ag` and `i_acc` are the
  interactions I = Δ(Adapted) − Δ(Fixed) of the MSE-relative contrasts
  (Figure 6c,d). Each paired endpoint forms a 29-test Holm family.

### Axis and representation controls

The robustness package is `../reports/robustness/w1_axis/`; `data/w1_robustness/`
holds byte-identical copies of the files the builders read.

- `alignment_summary.csv` (572 rows = 13 measures × 11 panels × 4 analyses).
  `analysis_id` is `native_all5` (the original analysis), `native_no_axis`,
  `common_grid_all5`, or `common_grid_no_axis`. `delta_ag` and `delta_oc` are
  MSE-relative contrasts. The `native_all5` rows reproduce the 143 original
  estimates, whose tests are those in `table_s1_full_alignment.csv`. In the
  three control analyses, W1 rows use the 66-test Holm family and the other
  candidates the 726-test family. Four common-grid mineral W1 rows are marked
  `unavailable` in `ag_state`/`oc_state` and have empty estimates.
- `ag_family_decomposition.csv`: per-family AG contributions `d_p`, which sum
  to AG (SI Table S6).
- `oc_group_summary.csv`: OC within non-axis/non-axis, mixed, and axis/axis
  pair groups.
- `task_harm_shares.csv`: axis-family share of positive and absolute task harm
  (SI Table S7).
- `protocol_interactions.csv`: W1 Fixed-to-Adapted interactions under each
  control.
- `bridge_comparisons.csv`: descriptive native-W1 versus common-grid-MSE
  comparisons.

`data/v4/mae_oc_pair_diagnostic.csv` holds the 20 native Fixed 10-shot MAE/MSE
pair-group rows used in SI Section S2.
