# Phase 2 Step 4 — Fallback Decision and Phase Closure

**Date:** 2026-08-19 UTC  
**Phase status:** `CLOSED UNDER AUTHORIZED FALLBACK`  
**Scientific status:** `PRE-GENERATION MODEL-ADEQUACY FAILURE`  
**Generation status:** `NOT AUTHORIZED; UNITS 3–5 RETIRED FOR THIS PROTOCOL`

## 1. Decision

Phase 2 is closed under the parent plan's fallback discipline. The active
benchmark will not create `semisynth_v2`, will not generate a semi-synthetic
dataset, and will not tune the degree, thresholds, strata, exclusions, or
candidate family after observing the `semisynth_v1` result. The preserved v1
failure becomes a methods result, and baseline-correction evaluation reverts to
the predeclared three-part indirect evidence system:

1. downstream-task validation;
2. reference-free evaluation; and
3. half-split replicate consistency.

This decision follows the parent plan's rules that a failed Phase 2 gate affects
only the baseline-correction line, that unsuccessful criteria should trigger a
fallback instead of being repaired until they pass, and that denoising and peak
detection continue independently.

## 2. Exact claim boundary

The parent plan's originally named terminal fallback is a method-ranking flip
across three extraction-derived semi-synthetic datasets. That experiment was
**not run**: the protocol failed an earlier, stricter pre-generation gate.
Therefore the project must not claim any of the following:

- that airPLS-, arPLS-, and morphology-derived method rankings flipped;
- that any pairwise Kendall tau-b was observed;
- that a generated semi-synthetic dataset was faithful or unfaithful;
- that Sugar fidelity or MMD failed; or
- that RRUFF processed spectra are physical clean ground truth.

The defensible result is narrower and precedes all those claims:

> A frozen degree-6 Legendre parameter generator, although operationally valid
> on all 28,092 extractor/pair attempts, failed the preregistered upper-tail
> reconstruction-adequacy criterion in 7 of 12 extractor-by-excitation cells.
> Consequently this protocol was not authorized to generate candidate ground
> truth, and ranking stability was not evaluable.

This is evidence that generator adequacy itself must be gated before a
semi-synthetic benchmark can be treated as ground truth. It is not evidence
that every possible semi-synthetic protocol is invalid.

## 3. Evidence supporting closure rather than selective repair

The authoritative failed run is retained at:

```text
results/phase2/background_fit/
  phase2-background-fit-6f8dc92c8b3edbefe2a422b805214b6687990c3ac13a34fbaed3faaa3fad6cdd/
```

Its four scientific payload checksums verify, `failed.json` is present,
`complete.json` is absent, and a complete rerun reproduced all six files
byte-for-byte. The run used all 9,364 stratified extraction-fit pairs and
recorded 28,092/28,092 valid attempts. Five model cells passed; seven failed
only the frozen `p95 reconstruction NRMSE <= 0.25` criterion.

The failure is not attributable to one malformed file family or a removable
slice artifact:

| Diagnostic | Observed evidence |
|---|---:|
| Unique pairs exceeding 0.25 for at least one extractor | 1,100 / 9,364 |
| Affected in exactly one extractor | 722 |
| Affected in exactly two extractors | 264 |
| Affected in all three extractors | 114 |
| Affected green-514 pairs | 279 / 3,647 (7.65%) |
| Affected green-532 pairs | 522 / 2,853 (18.30%) |
| Affected nir-780 pairs | 192 / 2,027 (9.47%) |
| Affected nir-785 pairs | 107 / 837 (12.78%) |
| airPLS rows above 0.25 | 412 / 9,364 (4.40%) |
| arPLS rows above 0.25 | 509 / 9,364 (5.44%) |
| morphology rows above 0.25 | 671 / 9,364 (7.17%) |

Failures occur across all four strata, multiple archives, and both dominant
exact-axis relations. Correlation between reconstruction NRMSE and the RMS
magnitude of Legendre coefficients 4–6 is approximately 0.725 for airPLS,
0.727 for arPLS, and 0.783 for morphology. The corresponding correlation with
log point count is approximately 0.416, 0.258, and 0.113. These diagnostics
support broad upper-tail shape complexity as the primary mechanism; they do
not justify deleting a small outlier subset.

## 4. Alternatives considered

### 4.1 Selected — execute the parent fallback

This preserves the prospective nature of the protocol, turns the failed
model-adequacy gate into a publishable warning about semi-synthetic GT
construction, and directs effort to evidence paths that do not require a
claimed clean baseline target. It also follows the parent plan's explicit
preference for fallback over repair after a success criterion fails.

### 4.2 Rejected for the active benchmark — one-shot versioned v2

A defensible independent follow-up could preregister a fixed menu such as
Legendre degrees 8 and 10 plus one fixed-knot cubic B-spline family, choose by a
group-safe extraction-fit-only inner split, keep all original thresholds, and
require all 12 cells to pass in one attempt. It would have to leave
`signal_template` and `real_holdout` untouched during selection and permanently
fallback after any failure.

That route is not selected here. Because the candidate menu would be proposed
after observing exactly where v1 failed, it adds outcome-tuning risk and delays
the parent study. Any future v2 must be a separate prospective protocol and may
not retroactively convert v1 into a pass.

### 4.3 Prohibited — local rescue edits

The following are explicitly rejected: raising degree only in failed cells,
relaxing the 0.25 threshold, excluding green-532 or morphology, deleting
high-NRMSE minerals/archives, choosing a model after inspecting Sugar fidelity
or downstream rankings, or repurposing the planned GP sensitivity as an
unregistered rescue.

## 5. Consequences for the remainder of the benchmark

### Baseline correction

- Phase 3 still builds the classical method pool, including the named
  pybaselines systems and frozen hyperparameter variants.
- Phase 4 must not report semi-synthetic baseline RMSE/NMSE as validated GT.
- Baseline-correction claims must be supported jointly by downstream tasks,
  reference-free metrics, and half-split consistency. No one component alone is
  promoted to physical truth.
- Physical anchors such as SERDS and substrate blanks remain separately labeled
  descriptive/sanity analyses; they cannot be pooled into a general clean-GT
  population.

### Denoising and peak detection

These lines continue under the existing Phase 1/3/4 protocols. The Phase 2
fallback does not weaken their available reference targets, perturbation
experiments, or downstream validation.

### Publication artifacts

- The planned Figure 3 rank-stability plot cannot be produced and must not be
  fabricated. Phase 6 should replace it with a pre-generation adequacy figure or
  table showing the 12 model-cell median/p95 results and gate status.
- The manuscript must include the failed v1 protocol in Methods, report the
  5/12 versus 7/12 result in Results, state that ranking/fidelity were not
  evaluable, and explain the indirect-evidence fallback in Discussion and
  Limitations.
- No semi-synthetic dataset is a release contribution under this outcome. The
  code, frozen config, split ledger, attempts, receipts, and failed artifact
  remain reproducibility evidence.

## 6. Prompt-to-artifact closure checklist

| Phase 2 requirement | Concrete evidence | Closure ruling |
|---|---|---|
| No circular reuse of one extracted instance | Parameter distributions only; exact fit/template/holdout roles in `step01`–`step03` | Implemented for the attempted protocol |
| Frozen source/config/split identities | `semisynth_v1.json`, lock, split ledger SHA, run manifest | Satisfied |
| Three extraction methods | 28,092 attempts and 12 receipts in the failed artifact | Executed |
| Pre-generation adequacy | `gate.json`; 5/12 pass, 7/12 p95 fail | Failed honestly |
| Three generated datasets | `data/semisynth/` absent | Retired by fallback |
| Sugar fidelity and GP sensitivity | No generated input; no result exists | Not evaluable; no pass claim |
| At least 80 systems / 10 families and three tau-b values | Phase 3 pool absent; no generated datasets | Not evaluable; no rank-instability claim |
| Failure preservation | `failed.json`, no `complete.json`, payload checksums valid | Satisfied |
| No post-result rescue | No `semisynth_v2.json`; no v2 code/data | Satisfied |
| Parent fallback | This report freezes indirect-evidence route | Triggered |

## 7. Verification and next boundary

No production code, config, scientific artifact, or test was changed in this
decision step. The most recent implementation verification remains the Step 3
warning-as-error discovery (`966/966`), and this closure relies on direct
readback of the failed marker, absent complete marker, four valid payload
checksums, 12 model-cell gates, and 28,092 attempt rows.

Phase 2 is now closed under fallback. The next scientific unit is Phase 3 Step
1: freeze the classical method-pool inventory and system/config schema while
explicitly tagging which systems serve denoising, baseline correction, and peak
detection. No Phase 3 implementation is included here.
