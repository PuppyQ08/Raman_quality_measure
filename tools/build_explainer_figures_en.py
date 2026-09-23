"""Build English explanatory figures from retained results and labelled toy examples.

Run from the repository with numpy and matplotlib installed. No experiments rerun.
The figures distinguish teaching examples from measured results.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "raman-explainer-mpl"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/methods_results_en"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"], "font.size": 12, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 180, "figure.facecolor": "white",
    "axes.titlepad": 14,
})
INK = "#193348"
BLUE = "#2378B5"
ORANGE = "#D98130"
GREEN = "#218A76"
RED = "#B85454"
GREY = "#697B88"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def canvas(title, subtitle, size=(14, 7.5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    fig.suptitle(title, x=.06, y=.98, ha="left", fontsize=19, color=INK, weight="bold")
    fig.text(.06, .89, subtitle, fontsize=11, color=GREY)
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=BLUE, fontsize=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                              facecolor=color + "10", edgecolor=color, linewidth=1.3))
    if body:
        ax.text(x+w/2, y+h*.72, title, ha="center", va="center", fontsize=fontsize,
                color=color, weight="bold")
        ax.text(x+w/2, y+h*.32, body, ha="center", va="center", fontsize=fontsize-1,
                color=INK, linespacing=1.7)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=fontsize,
                color=color, linespacing=1.7)


def arrow(ax, start, end, color=GREY):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.7,
                            "shrinkA": 3, "shrinkB": 3, "mutation_scale": 15})


# 1. Study flow. The scores never feed the downstream predictor.
fig, ax = canvas("Do spectral measures reflect downstream task performance?", "Framework overview: measure spectral changes and task consequences, then compare them.")
box(ax, 5, 70, 34, 21, "Reference spectrum", "Bacteria / sugar mixtures / minerals", BLUE)
box(ax, 49, 70, 46, 21, "Apply controlled changes", "5 perturbation types × 8 strengths", ORANGE)
arrow(ax, (40, 80), (48, 80))
box(ax, 5, 34, 40, 25, "Spectral-measure branch", "Measure the spectral change\nMetric harm x", BLUE)
box(ax, 55, 34, 40, 25, "Downstream-task branch", "Classify / quantify / identify\nTask harm y", GREEN)
ax.plot([72, 72, 25], [69, 65, 65], color=GREY, lw=1.7)
arrow(ax, (25, 65), (25, 60))
ax.plot([72, 75], [65, 65], color=GREY, lw=1.7)
arrow(ax, (75, 65), (75, 60))
box(ax, 22, 1, 56, 23, "Compare metric harm x with task harm y", "AG: does the relationship depend on type?\nOC: is the more harmful condition ranked correctly?", INK)
arrow(ax, (25, 33), (38, 25))
arrow(ax, (75, 33), (62, 25))
fig.text(.06, .025, "All candidate measures use the same task outcomes. Changing the measure does not change the predictions.", fontsize=11, color=GREY)
save(fig, "01_workflow")


# 2. Synthetic shapes. Coordinate effects deliberately enlarged for visibility.
x = np.linspace(400, 1800, 1401)
base = .04 + sum(a*np.exp(-.5*((x-mu)/sig)**2)
                 for a, mu, sig in [(1, 700, 14), (.55, 1020, 24), (.8, 1430, 18)])
rng = np.random.default_rng(20260920)
z = rng.normal(size=x.size)
kernel = np.exp(-.5*(np.arange(-80, 81)/20)**2)
corr = np.convolve(z, kernel/kernel.sum(), mode="same")
corr = (corr-corr.mean()) / corr.std() * .13
shift = 25.0
warped_x = x + 65*((x-x.min())/(x.max()-x.min()))**2
curves = [(x, base), (x, base+.30*((x-950)/850)**2-.13),
          (x, base+.09*z), (x, base+corr), (x+shift, base), (warped_x, base)]
titles = ["Reference spectrum", "Baseline distortion", "Independent noise", "Correlated noise", "Global wavenumber shift", "Nonlinear axis warp"]
captions = ["Synthetic spectrum with three peaks", "Add a smooth, curved background", "Add independent random fluctuations", "Add fluctuations shared by nearby points", "Move every coordinate by the same amount", "Move different coordinates by different amounts"]
fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.8))
fig.subplots_adjust(top=.83, bottom=.13, hspace=.70, wspace=.24)
fig.suptitle("Five controlled ways to change a spectrum", x=.06, y=.98, ha="left", fontsize=19, color=INK, weight="bold")
fig.text(.06, .89, "Synthetic teaching examples. Shifts and warps are enlarged for visibility; the experiment uses at most 3.2 cm⁻¹.", fontsize=11, color=GREY)
for i, (ax, (xx, yy), title, caption) in enumerate(zip(axs.flat, curves, titles, captions)):
    ax.plot(x, base, color="#A4AFB8", lw=1.7, label="Reference")
    ax.plot(xx, yy, color=BLUE if i == 0 else ORANGE, lw=1.25, label="Perturbed")
    ax.set(xlim=(390, 1870), ylim=(-.28, 1.3), xlabel="Wavenumber / cm⁻¹", ylabel="Intensity (arbitrary units)")
    ax.set_title(title, loc="left", fontsize=14, color=INK)
    ax.text(0, -.38, caption, transform=ax.transAxes, fontsize=10, color=GREY)
    ax.grid(alpha=.12)
axs.flat[1].legend(frameon=False, fontsize=10)
save(fig, "02_perturbations")


# 3. Analysis protocols.
fig, ax = canvas("Fixed and Adapted: change the data used to establish the analysis", "Both evaluate the same perturbed test spectra. Test samples are never used for fitting or selection.", (14, 9))
ax.text(0, 97, "Bacterial classification / sugar concentration prediction", fontsize=16, weight="bold", color=INK)
for y, label, train, fit, col in [
    (65, "Fixed", "Unperturbed training\nand validation spectra", "Fit and select once;\nkeep parameters fixed", BLUE),
    (35, "Adapted", "Training and validation\nspectra perturbed to match", "Fit and select separately\nfor each type and strength", ORANGE),
]:
    ax.text(0, y+25.5, label, fontsize=13, color=col, weight="bold")
    box(ax, 1, y, 30, 24, train, color=col)
    box(ax, 39, y, 27, 24, fit, color=col)
    box(ax, 74, y, 25, 24, "Evaluate perturbed tests\nObtain task error", color=GREEN)
    arrow(ax, (32, y+12), (38, y+12))
    arrow(ax, (67, y+12), (73, y+12))
ax.text(0, 26, "Mineral identification: match a query spectrum to a reference library", fontsize=14, weight="bold", color=INK)
box(ax, 1, 1, 46, 19, "Fixed: unchanged library ← perturbed queries", color=BLUE, fontsize=13)
box(ax, 53, 1, 46, 19, "Adapted: perturbed library ← same queries", color=ORANGE, fontsize=13)
fig.text(.06, .025, "Matching means the same type and strength, with record-specific random draws. Full-data bacteria uses Fixed only.", fontsize=11, color=GREY)
save(fig, "03_protocols")


# 4. AG intuition. At each x there are two observations; their mean is monotone,
# so the pooled isotonic fit is exactly the pointwise mean in these toy cases.
fig, axs = plt.subplots(1, 2, figsize=(13, 5.7))
fig.subplots_adjust(top=.78, bottom=.20, wspace=.27)
fig.suptitle("AG: how much does knowing the perturbation type help?", x=.06, y=.98, ha="left", fontsize=19, color=INK, weight="bold")
fig.text(.06, .87, "Two-type teaching example. The experiment compares one pooled curve with five type-specific curves.", fontsize=11, color=GREY)
toy_ags = []
for ax, gap, title in zip(axs, [.1, 4.0], ["Small AG: similar relationships across types", "Large AG: similar metric harm, different task harm"]):
    xx = np.arange(1, 5, dtype=float)
    y1, y2 = xx, xx+gap
    pooled = (y1+y2)/2
    yy = np.r_[y1, y2]
    sst = ((yy-yy.mean())**2).sum()
    ag = float((((y1-pooled)**2).sum()+((y2-pooled)**2).sum())/sst)
    toy_ags.append(ag)
    ax.plot(xx, y1, "o-", color=BLUE, lw=2, label="Noise: separate fit")
    ax.plot(xx, y2, "s-", color=ORANGE, lw=2, label="Shift: separate fit")
    ax.plot(xx, pooled, "--", color=INK, lw=1.8, label="All data: pooled fit")
    ax.set(xlim=(.7, 4.3), ylim=(0, 9), xlabel="Metric harm x", ylabel="Task harm y (arbitrary units)")
    ax.set_title(title, fontsize=13, color=INK)
    ax.text(.06, .9, f"Illustrative AG = {ag:.3f}", transform=ax.transAxes, fontsize=14, weight="bold", color=INK)
    ax.grid(alpha=.15)
axs[1].legend(frameon=False, fontsize=10, loc="lower right")
fig.text(.06, .04, "A small AG means little benefit from separate fits. Both fits can still be poor; a small gap alone does not establish usefulness.", fontsize=11, color=RED)
save(fig, "04_ag")


# 5. OC simple numerical examples.
fig, ax = canvas("OC: which spectral change is more harmful to the task?", "Three separate teaching examples. Larger metric harm means worse spectral quality.", (14, 6.5))
cases = [
    (1, GREEN, "Matching order", "Metric: noise 0.2 < shift 0.7", "Score: 1", "Both rank the shift as more harmful"),
    (35, RED, "Opposite order", "Metric: noise 0.7 > shift 0.2", "Score: 0", "The metric incorrectly ranks noise higher"),
    (69, GREY, "Exact metric tie", "Metric: noise 0.4 = shift 0.4", "Score: 0.5", "The metric cannot distinguish the pair"),
]
for x0, col, title, metric, score, meaning in cases:
    box(ax, x0, 17, 29, 66, "", color=col, fontsize=15)
    ax.text(x0+14.5, 74, title, ha="center", fontsize=15, color=col, weight="bold")
    ax.text(x0+14.5, 63, metric, ha="center", fontsize=11, color=INK)
    ax.text(x0+14.5, 49, "Observed accuracy loss\nNoise: 2 percentage points\nShift: 10 percentage points", ha="center", fontsize=10, color=INK, linespacing=1.8)
    ax.text(x0+14.5, 31, score, ha="center", fontsize=20, weight="bold", color=col)
    ax.text(x0+14.5, 21, meaning, ha="center", fontsize=10, color=GREY)
ax.text(50, 3, "Average pair scores within each class or well, then average across those units to obtain OC.", ha="center", fontsize=13, color=INK)
save(fig, "05_oc")


# 6. Complete factual profile, projected from retained estimates and adjusted tests.
SOURCE = ROOT / "paper/data/table_s1_full_alignment.csv"
with SOURCE.open() as f:
    rows = list(csv.DictReader(f))
ROBUST_SOURCE = ROOT / "paper/data/w1_robustness/alignment_summary.csv"
with ROBUST_SOURCE.open() as f:
    robustness_rows = list(csv.DictReader(f))
fixed_controls = [r for r in robustness_rows
                  if r["panel_id"] in ("d1_a", "d2_5_a", "d2_10_a", "d2_20_a")
                  and r["analysis_id"] in ("native_no_axis", "common_grid_all5", "common_grid_no_axis")
                  and r["metric_output_id"] == "wasserstein_1_cm1"]
assert len(fixed_controls) == 12
assert all(float(r["delta_" + outcome]) > 0 and float(r["adjusted_p_" + outcome]) < .05
           for r in fixed_controls for outcome in ("ag", "oc"))
lookup = {(r["panel_id"], r["metric_output_id"]): r for r in rows}
assert len(rows) == len(lookup) == 143
panels = ["d1_a", "d2_5_a", "d2_10_a", "d2_20_a", "d2_5_b", "d2_10_b", "d2_20_b", "d4_a", "d4_b", "d5_a", "d5_b"]
panel_names = ["Bacteria full / Fixed", "Bacteria 5-shot / Fixed", "Bacteria 10-shot / Fixed", "Bacteria 20-shot / Fixed", "Bacteria 5-shot / Adapted", "Bacteria 10-shot / Adapted", "Bacteria 20-shot / Adapted", "Sugar / Fixed", "Sugar / Adapted", "Mineral / Fixed", "Mineral / Adapted"]
metrics = ["rmse", "mae", "nmse", "sam", "pearson_r", "wasserstein_1_cm1", "is_like_structure_to_noise", "precision", "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio"]
metric_names = ["RMSE", "MAE", "NMSE", "SAM", "Pearson", "W1", "S/N", "Precision", "Recall", "Peak F1", "Artifact", "Missing"]
state_names = ["Unresolved", "Both +", "AG +", "OC +", "Both −", "AG −", "OC −", "Trade-off"]
palette = ["#F0F2F4", "#248E79", "#D9EEE6", "#DCEBFA", "#B75558", "#F3D9DB", "#FCE6D2", "#E1D5EE"]


def state(r):
    a, o = float(r["d_ag"]), float(r["d_acc"])
    sa, so = float(r["d_ag_adjusted_p_value"]) < .05, float(r["d_acc_adjusted_p_value"]) < .05
    if sa and so:
        if a > 0 and o > 0: return 1
        if a < 0 and o < 0: return 4
        return 7
    if sa: return 2 if a > 0 else 5
    if so: return 3 if o > 0 else 6
    return 0


matrix = np.array([[state(lookup[p, m]) for m in metrics] for p in panels])
fig, ax = plt.subplots(figsize=(18, 9))
fig.subplots_adjust(left=.23, right=.985, top=.79, bottom=.21)
fig.suptitle("Measured results: how does each candidate compare with MSE?", x=.05, y=.97, ha="left", fontsize=21, color=INK, weight="bold")
fig.text(.05, .895, "Native representation, all five perturbation types; Holm adjustment across 24 tests per task/protocol.", fontsize=12, color=GREY)
ax.imshow(matrix, cmap=ListedColormap(palette), norm=BoundaryNorm(np.arange(-.5, 8.5), 8), aspect="auto")
ax.set_xticks(np.arange(len(metrics)), metric_names, fontsize=11)
ax.xaxis.tick_top()
ax.set_yticks(np.arange(len(panels)), panel_names, fontsize=12)
ax.tick_params(axis="both", length=0, pad=10)
for i in range(len(panels)):
    for j in range(len(metrics)):
        s = matrix[i, j]
        ax.text(j, i, state_names[s], ha="center", va="center", fontsize=9,
                color="white" if s in (1, 4) else INK, weight="bold" if s in (1, 4) else "normal")
ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(panels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
for spine in ax.spines.values(): spine.set_visible(False)
fig.text(.05, .135, "Both + / Both −: significantly better / worse on both AG and OC. AG or OC alone: only that contrast is significant.", fontsize=12, color=INK)
fig.text(.05, .09, "Unresolved: neither difference is significant. A better measure tracks task harm; it does not improve the task predictions.", fontsize=12, color=INK)
fig.text(.05, .045, "Separate axis-removal and common-grid controls support the Fixed bacterial W1 advantage; see the guide for scope.", fontsize=12, color=RED)
save(fig, "06_result_matrix")


# 7. Why index correspondence differs from physical correspondence.
nu = np.linspace(995, 1007, 601)
intensity = np.exp(-.5*((nu-1000)/.6)**2)
delta = 2.0
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(top=.78, bottom=.23, wspace=.28)
fig.suptitle("Why can index-aligned MSE stay zero after an axis shift?", x=.06, y=.98, ha="left", fontsize=19, color=INK, weight="bold")
fig.text(.06, .87, "Synthetic example: the intensity array is unchanged, but every physical coordinate increases by 2 cm⁻¹.", fontsize=11, color=GREY)
axs[0].plot(np.arange(nu.size), intensity, color=BLUE, lw=3, label="Reference")
axs[0].plot(np.arange(nu.size), intensity, color=ORANGE, lw=2, ls="--", label="After shift")
axs[0].set(xlabel="Array position (index)", ylabel="Intensity", title="Compare indices: identical arrays, MSE = 0")
axs[1].plot(nu, intensity, color=BLUE, lw=2, label="Reference")
axs[1].plot(nu+delta, intensity, color=ORANGE, lw=2, ls="--", label="After shift")
axs[1].set(xlim=(997, 1005), xlabel="Physical wavenumber / cm⁻¹", ylabel="Intensity", title="Compare coordinates: the peak moves to 1002")
for ax in axs:
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=.13)
fig.text(.06, .085, "W1 uses physical coordinates and detects the shift; peak-position measures can also respond.", fontsize=12, color=INK)
fig.text(.06, .035, "After interpolation onto a common physical grid, MSE can detect the shift too. No peak registration is applied.", fontsize=12, color=RED)
save(fig, "07_axis_convention")


profiles_path = OUT / "result_profiles.csv"
with profiles_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["panel_id", "metric_output_id", "profile", "d_ag", "d_acc", "d_ag_adjusted_p_value", "d_acc_adjusted_p_value"], lineterminator="\n")
    writer.writeheader()
    for p in panels:
        for m in metrics:
            r = lookup[p, m]
            writer.writerow({**{k: r[k] for k in writer.fieldnames if k != "profile"}, "profile": state_names[state(r)]})
receipt = {
    "source": str(SOURCE.relative_to(ROOT)),
    "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "source_rows": len(rows),
    "robustness_source": str(ROBUST_SOURCE.relative_to(ROOT)),
    "robustness_sha256": hashlib.sha256(ROBUST_SOURCE.read_bytes()).hexdigest(),
    "fixed_bacterial_controls_favorable_on_both": len(fixed_controls),
    "illustrations": ["01_workflow", "02_perturbations", "03_protocols", "04_ag", "05_oc", "07_axis_convention"],
    "data_figure": "06_result_matrix",
    "toy_ag_values": toy_ags,
    "note": "No new experiments or inferential analyses; retained point estimates and adjusted tests only.",
    "font": FONT,
    "language": "en",
    "perturbation_labels": "Descriptive names; no implementation IDs displayed",
    "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.png"))},
}
(OUT / "source_receipts.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n")
assert matrix.shape == (11, 12)
assert all(state(lookup[p, "wasserstein_1_cm1"]) == 1 for p in ["d1_a", "d2_5_a", "d2_10_a", "d2_20_a"])
assert state(lookup["d4_b", "f1"]) == 1
print(f"Built seven figures and verified 132 candidate profiles in {OUT}")
