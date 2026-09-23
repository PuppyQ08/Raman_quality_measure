# Phase 3 Step 1 — Classical System Registry and Pool Design

**Date:** 2026-08-19 UTC  
**Status:** `DESIGN READY FOR USER REVIEW — NO PHASE 3 REGISTRY IMPLEMENTED`  
**Parent requirement:** `raman_preproc_benchmark_plan_v2.md` Phase 3

## 1. Goal and completion boundary

Phase 3 must supply the real preprocessing systems used in the main table and
the Phase 5 observational meta-evaluation. A *system* is one fully specified
preprocessing configuration, not a method name, downstream training seed,
reported literature number, or partially available implementation.

This step freezes the registry architecture, the classical candidate pools,
the statistical counting rules, dependencies, applicability contracts, and
implementation sequence. It does not:

- create `experiments/phase3/` or `results/phase3/`;
- install Optuna or PyWavelets;
- implement a generic baseline, denoising, or peak-detection wrapper;
- run any method on the 10k evaluation subset or full datasets;
- audit or download deep-learning repositories; or
- produce a main-table or Phase 5 number.

The workspace is not a Git repository, so future code identity will use the
same path-bound SHA-256 snapshots already used by Phase 1/2 rather than an
invented commit identity.

## 2. Parent-plan conflict and ruling

The parent plan simultaneously requires:

1. denoising, baseline correction, and peak detection to remain separate
   because their targets and outputs differ; and
2. more than 200 systems for Phase 5 statistical power.

It is invalid to satisfy the second requirement by pooling the three task lines
into one correlation sample. Their score tensors do not share a common target,
metric eligibility, output type, or downstream interpretation. It is also
invalid to multiply K by downstream training seeds: those are repeated
measurements of one preprocessing system.

**Ruling:** use one canonical catalog envelope with three immutable task-line
views. Every view has its own raw system count K, method-family count,
replicate count, complete-case coverage, and effective cluster count. Phase 5
power is judged per comparable matrix. The baseline-correction view alone is
designed to reach K=210 because the Phase 2 fallback makes natural
between-system variation especially important for that line. Denoising and peak
detection remain valid method pools but are `not_powered_for_phase5` until a
future prospective expansion independently reaches its gate.

## 3. Alternatives and selected design

| Option | Baseline | Denoising | Peak detection | Ruling |
|---|---:|---:|---:|---|
| A — lean | 14 families × 8 = 112 | about 40 | about 24 | Rejected: fast, but baseline K misses the parent power target |
| **B — separated balanced** | **14 × 15 = 210** | **5 × 12 = 60** | **3 × 12 = 36** | **Selected** |
| C — combination expansion | >200 via many multi-operation pipelines | variable | variable | Rejected: inflates K with composition confounding |

Option B produces 306 planned classical systems without treating them as one
306-row inferential sample. Atomic single-operation systems come first. A
pipeline such as `airPLS + SG` is a separate ordered-composition family and may
be added only by a later versioned protocol; its effect cannot be attributed to
either atomic operation.

## 4. Statistical units

The registry distinguishes four non-interchangeable counts:

- `system_count`: distinct scientific preprocessing configurations;
- `family_count`: independent method families used as the Phase 5 clustering
  unit;
- `replicate_count`: method-training or downstream-evaluation repeats nested
  within one system; and
- `effective_cluster_count`: the family-cluster information reported by the
  Phase 5 analysis.

For deterministic classical methods, `method_seed=null`. Optuna's GridSampler
seed and downstream Protocol A/B training seeds are orchestration/replicate
identities, not new systems. A genuinely stochastic trained preprocessor may
include its own training seed in the system descriptor, but all such systems
remain clustered under the same family.

Reported-only methods never enter `system_count`, K, the main table, or metric
matrices. They live in a separate reproducibility-audit collection.

### Phase 5 power gate per view

The baseline-correction view is `powered_for_phase5` only if the formal run has:

```text
eligible system_count             >= 200
eligible family_count             >= 10
eligible configurations/family   >= 8 for every counted family
common-record coverage            >= 0.95
per-system successful coverage    >= 0.99
family-cluster bootstrap           required
raw K and effective clusters       both reported
```

Failure does not trigger parameter replacement or selective new grids. The view
becomes `not_powered_for_phase5`, remains in the reproducibility audit/main
table where scientifically eligible, and cannot support a powered correlation
claim.

## 5. Canonical catalog envelope

The later implementation will create
`experiments/phase3/configs/classical_system_catalog_v1.json` using canonical
finite JSON and the schema `phase3-classical-system-catalog-v1`. The document
contains exact keys:

```text
schema_version
catalog_id
global_seed
dependency_lock
identity_domains
views
systems
availability_audit
phase2_fallback_binding
```

`catalog_id` is derived from the path-free scientific document. The ordered
systems are sorted by UTF-8 `system_id`; duplicate IDs or scientific
descriptors are rejected.

### 5.1 System identity

```text
system_id = SHA256(
  "rpe-phase3-system-v1\0"
  + canonical_json({
      task_line, family_id, method_id, backend,
      hyperparameters, input_contract, output_contract,
      ordered_composition, method_seed
    })
)
```

Availability, local file paths, runtime duration, warning counts, and downstream
training seeds are excluded from `system_id`; changing the scientific operation
or its inputs changes the ID. The catalog separately binds package, code, config,
and data identities for the actual execution.

### 5.2 Required system fields

Each system stores:

```text
system_id
task_line                 # baseline_correction | denoising | peak_detection
family_id
method_id
backend                   # public callable and package
hyperparameters           # complete canonical finite mapping
input_contract
output_contract
ordered_composition       # exactly one atomic operation in v1
method_seed               # null for deterministic classical systems
metric_eligibility
downstream_eligibility
protocol_eligibility      # A, B, Phase 5, main table
availability
evidence
```

### 5.3 Input and output contracts

All atomic wrappers accept immutable finite little-endian float64 vectors and a
strictly increasing physical wavenumber axis. They do not silently sort, reverse,
interpolate, resample, normalize, clip, or add offsets. Task-required alignment
such as D5's common grid is an external downstream representation applied
identically to control and all systems and is recorded outside the system ID.

- baseline correction returns both `baseline_estimate` and
  `corrected = input - baseline_estimate`; it never clips the correction;
- denoising returns one same-axis intensity vector; and
- peak detection returns immutable peaks with index, physical position, height,
  prominence, width, and area when defined.

Index-coordinate methods explicitly declare that limitation. Dataset-trained
PCA/SVD systems require a common axis and fit only on the downstream training
partition; they never fit on validation/test or on all spectra before splitting.

## 6. Availability and execution states

Registry availability is fail-closed and evidence-backed:

```text
runnable
dependency_missing
implementation_missing
code_missing
weights_missing
data_missing
reported_only
excluded
```

Only `runnable` systems with verified code/dependency identities may execute or
count toward K. The per-record execution taxonomy is separate:

```text
complete
complete_with_warning
not_applicable
failed_domain
failed_convergence
failed_runtime
```

Warnings and tolerance histories are retained. A nonconverged iterative method
is not silently relabeled complete, and no wrapper falls back to another method
or parameter set. Formal matrices preserve all failure rows and apply the frozen
coverage gates rather than imputing a favorable output.

## 7. Baseline-correction pool — 210 systems

All 14 named families use public `pybaselines==1.2.1` calls. Parameters omitted
below remain fixed at the stated package defaults in the catalog, not left
implicit. `weights=None`; `tol=1e-3` unless listed otherwise. Iterative methods
record `tol_history` and convergence.

Define:

```text
L5  = [1e3, 1e4, 1e5, 1e6, 1e7]
L15 = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5,
       1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9]
```

| Family ID | 15-system grid | Fixed values |
|---|---|---|
| `asls` | `L5 × p=[0.001,0.01,0.1]` | `diff_order=2,max_iter=50` |
| `iasls` | `L5 × p=[0.001,0.01,0.1]` | `lam_1=1e-4,diff_order=2,max_iter=50` |
| `airpls` | `lam=L15` | `diff_order=2,max_iter=50,normalize_weights=false` |
| `arpls` | `lam=L15` | `diff_order=2,max_iter=50` |
| `drpls` | `L5 × eta=[0.1,0.5,0.9]` | `diff_order=2,max_iter=50` |
| `iarpls` | `lam=L15` | `diff_order=2,max_iter=50` |
| `aspls` | `L5 × asymmetric_coef=[0.1,0.5,0.9]` | `diff_order=2,max_iter=100,alpha=None` |
| `psalsa` | `L5 × p=[0.01,0.1,0.5]` | `k=None,diff_order=2,max_iter=50` |
| `modpoly` | `poly_order=[2,3,4,5,6] × three variants` | variants `(use_original,mask_initial_peaks)=(F,F),(F,T),(T,F)`; `max_iter=250` |
| `imodpoly` | `poly_order=[2,3,4,5,6] × num_std=[0.5,1,2]` | `use_original=false,mask_initial_peaks=true,max_iter=250` |
| `penalized_poly` | `poly_order=[2,3,4,5,6] × three costs` | costs `asymmetric_truncated_quadratic, asymmetric_huber, asymmetric_indec`; `alpha_factor=0.99,max_iter=250` |
| `snip` | physical half-window `[10,20,40,80,160] cm^-1 × filter_order=[2,4,6]` | `decreasing=false,smooth_half_window=None,pad_kwargs=None` |
| `morphological` | physical half-window `[8,12,16,24,32,48,64,80,96,128,160,192,256,320,400] cm^-1` | backend `Baseline.mor`, `window_kwargs=None` |
| `beads` | `freq_cutoff=[0.0025,0.005,0.01,0.02,0.04] × asymmetry=[2,6,10]` | `lam_0=lam_1=lam_2=1,filter_type=1,cost_function=2,max_iter=50,tol=0.01,fit_parabola=true` |

Physical half-windows convert to index half-windows by
`round(width_cm1 / median(diff(axis_cm1)))`. Values outside the public method's
valid domain are `not_applicable`; they are never clipped to another system.

Read-only API compatibility evidence: all 14×15 configurations returned finite
same-shape outputs on one common 513-point synthetic Raman oracle under
pybaselines 1.2.1; every family produced 15 distinct output hashes. Some extreme
arPLS/iarPLS/asPLS configurations emitted `ParameterWarning`, which proves why
warnings/convergence must be first-class evidence. This probe is not a
performance result and selected no configuration.

## 8. Denoising pool — 60 systems

| Family ID | 12-system grid | Contract |
|---|---|---|
| `savitzky_golay` | odd `window_length=[5,7,9,11,15,21] × polyorder=[2,3]` | `deriv=0,axis=-1,mode=interp`, explicit index-coordinate system |
| `wavelet` | wavelet `[db4,db6,sym8] × threshold_mode=[soft,hard] × strategy=[universal_mad,bayes_shrink]` | symmetric extension; per-spectrum finest-detail MAD; deterministic maximum feasible level capped at 5 |
| `pca_reconstruction` | `n_components=[2,3,4,6,8,12,16,24,32,48,64,96]` | centered, `svd_solver=full`, no whitening; train-partition fit on common axis |
| `svd_reconstruction` | same component grid | uncentered randomized SVD, `n_iter=5,random_state=20260819`; train-partition fit |
| `whittaker_smoothing` | `lambda=[1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]` | solve `(I + lambda D2^T D2)z=y`; index-coordinate second difference |

The already validated Phase 0.5 SG11/poly3 implementation is useful evidence
and may be adapted, but its old config identity is not a Phase 3 system ID. The
generic Phase 3 wrapper remains unimplemented.

## 9. Peak-detection pool — 36 systems

### `find_peaks` — 12

Use `scipy.signal.find_peaks` with prominence expressed as a fraction of
`q99(intensity)-q01(intensity)`:

```text
[0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50]
```

Other selection filters are null; prominence, physical FWHM, and area are
computed for every retained peak. A nonpositive robust range is not applicable.

### `find_peaks_cwt` — 12

Cross four fixed physical-width banks with `min_snr=[1,2,3]`:

```text
[1,2,3,4] cm^-1
[1,2,4,8] cm^-1
[2,4,6,8] cm^-1
[2,4,8,12] cm^-1
```

Widths convert by median native spacing, preserving unique positive point
widths; an unsupported bank is not applicable. Package defaults for wavelet,
max distances, gap threshold, minimum ridge length, noise percentile, and
window size are frozen explicitly in the future catalog receipt.

### `mspd` — 12 planned, implementation missing

Freeze four maximum physical scales `[4,8,12,16] cm^-1` crossed with
ridge-vote fractions `[0.25,0.50,0.75]`. The implementation must construct a
multi-scale local-maxima scalogram, retain peaks whose maxima ridge occurs in at
least the configured fraction of valid scales, and use deterministic
position-ascending tie resolution. It cannot count toward K until its algorithm
contract has a hand-derived scalogram test and real-axis focused verification.

Peak-detector systems never inherit an unregistered denoiser or baseline
corrector. Any composed detector pipeline belongs to a later catalog version.

## 10. Dependencies and orchestration

Current verified environment:

```text
pybaselines 1.2.1     present
scipy 1.18.0          present
scikit-learn 1.9.0    present
joblib 1.5.3          present
Optuna                 missing
PyWavelets             missing
MSPD implementation    missing
```

The accepted Phase 2 config binds the existing `env/requirements.lock` bytes,
so that file must not be overwritten. Phase 3 will create a separate
`env/phase3-requirements.lock`, prospectively pinning `optuna==4.9.0` and
`PyWavelets==1.9.0` plus the retained scientific stack.

Optuna uses `GridSampler(search_space, seed=20260819)` only to enumerate and
schedule the complete frozen grid. No pruning, best-trial promotion, adaptive
sampler, validation-objective replacement, or test/downstream-selected winner
is allowed. All 15 or 12 preregistered configurations remain independent
systems.

## 11. Phase 2 fallback and eligibility

Baseline correction has no validated semi-synthetic clean target after Phase 2.
Therefore its main-table and Phase 5 eligibility is limited to the joint
fallback evidence:

```text
downstream task outcomes
reference-free metrics
half-split replicate consistency
separately labelled physical-anchor descriptions
```

The catalog must explicitly mark baseline RMSE/NMSE against generated clean GT
as unavailable. RRUFF processed spectra cannot become clean GT. A system is not
allowed into a metric column merely because its output shape is compatible.

Denoising may use direct fidelity targets only on datasets with defensible
long-exposure/multi-acquisition targets. Peak detection may use direct peak GT
only on pure/known-phase spectra with accepted assignments. Metric eligibility
is dataset- and task-line-specific and is validated before execution.

Protocol A/B seeds belong to the downstream result matrix. The same system ID
must be reused across both protocols so their disagreement is attributable to
the downstream protocol rather than a changed preprocessing system.

## 12. Implementation and scientific gates

### Registry gate

- exact canonical config identity and dependency lock identity;
- exactly 210/60/36 planned systems in the three views;
- exactly 14/5/3 families and 15/12/12 systems per family;
- unique system IDs and descriptors;
- no reported-only record in a system view or K;
- no downstream seed in a classical system ID;
- complete input/output, metric, downstream, and availability contracts; and
- Phase 2 fallback columns encoded fail-closed.

### Wrapper gate

Each family requires a literal or analytic oracle, alpha-independent
determinism where relevant, immutable outputs, stable domain errors, warning and
convergence capture, no implicit processing, and one real-data focused check.
Missing dependency/implementation entries stay absent from runnable K.

### Execution gate

Exploration and canaries use the frozen `eval_subset_10k` identity. Final
paper-table values run once on the full eligible dataset. The subset/full method
ranking must have Kendall tau-b strictly greater than 0.95 or the subset is
expanded prospectively. Formal outputs are append-only, checksummed, and bind
catalog, code, data, environment, hardware, warnings, failures, and coverage.

## 13. Incremental implementation plan

The approved efficiency rules still apply: one scientific unit, one self-review,
focused tests inside the unit, one full discovery at step closure, and one report
per step. Proposed increments are:

1. catalog contracts, canonical config, dependency lock, and availability audit;
2. 14-family baseline wrapper and 210-system deterministic canary;
3. denoising wrappers and 60-system canary;
4. peak-detector wrappers and 36-system canary;
5. formal 10k execution, coverage gate, and subset/full ranking check; and
6. separate DL reproducibility audit, with runnable and reported-only entries
   kept out of the classical catalog until their own contracts pass.

## 14. Review checklist and next boundary

- K is never obtained by pooling incomparable task lines.
- Seeds and reported-only literature records do not inflate system count.
- Atomic systems precede composed pipelines.
- Every method family has an exact finite grid and a family cluster ID.
- Baseline fallback eligibility excludes unvalidated clean-GT metrics.
- Missing dependencies and implementations remain explicit and do not count.
- Existing Phase 0.5 results are evidence, not hidden hyperparameter tuning.
- No method is dropped because its output appears poor.

This design awaits user review. The next permitted increment after approval is
catalog contracts/config/lock only; no wrapper or 306-system execution belongs
in that same step.
