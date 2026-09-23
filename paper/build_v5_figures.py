"""Build the v5 explanatory figures: framework, perturbations, and AG/OC.

The designs follow the illustrated guide (../docs/methods_results_explained_en.md)
and reuse its synthetic teaching data. Titles and
notes move to the LaTeX captions. No experimental result is read or rerun.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures/v5"
FIG.mkdir(parents=True, exist_ok=True)
INK, BLUE, ORANGE, GREEN, RED, GREY = "#193348", "#2378B5", "#D98130", "#218A76", "#B85454", "#697B88"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def panel(ax, title):
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ax.text(0, 100, title, va="top", fontweight="bold", fontsize=9, color=INK)


def box(ax, x, y, w, h, title, body="", color=BLUE, size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor=color + "10", edgecolor=color, linewidth=0.9))
    if body:
        ax.text(x + w/2, y + h*.70, title, ha="center", va="center", fontsize=size,
                color=color, fontweight="bold")
        ax.text(x + w/2, y + h*.30, body, ha="center", va="center", fontsize=size - .5,
                color=INK, linespacing=1.4)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=size,
                color=INK, linespacing=1.4)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 1.0,
                            "shrinkA": 2, "shrinkB": 2, "mutation_scale": 9})


def framework():
    fig = plt.figure(figsize=(6.5, 4.4), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.0])
    a = fig.add_subplot(gs[0])
    panel(a, "(a) Pair each spectral change with its task consequence")
    box(a, 4, 69, 36, 18, "Reference spectrum", "Bacteria / sugar mixtures / minerals", BLUE)
    box(a, 52, 69, 44, 18, "Controlled perturbation", "5 types × 8 strengths", ORANGE)
    arrow(a, (41, 78), (51, 78))
    box(a, 4, 35, 40, 22, "Spectral-measure branch", "Measure the spectral change\nMetric harm x", BLUE)
    box(a, 56, 35, 40, 22, "Downstream-task branch", "Classify / quantify / identify\nTask harm y", GREEN)
    a.plot([74, 74, 24], [68, 63, 63], color=GREY, lw=1.0)
    arrow(a, (24, 63), (24, 58))
    arrow(a, (74, 63), (74, 58))
    box(a, 22, 1, 56, 22, "Compare metric harm x with task harm y",
        "AG: does the relationship depend on perturbation type?\n"
        "OC: is the more harmful condition ranked correctly?", INK)
    arrow(a, (24, 34), (38, 24))
    arrow(a, (76, 34), (62, 24))
    b = fig.add_subplot(gs[1])
    panel(b, "(b) Two ways to establish the downstream analysis")
    box(b, 2, 34, 45, 44, "Fixed", "Fit to unperturbed training spectra;\nkeep the reference library unchanged", BLUE)
    box(b, 53, 34, 45, 44, "Adapted", "Fit to each perturbed training condition;\nperturb the reference library to match", ORANGE)
    b.text(50, 4, "Evaluate the same perturbed test or query spectra", ha="center", va="bottom", fontsize=8, color=INK)
    arrow(b, (25, 32), (40, 15))
    arrow(b, (75, 32), (60, 15))
    save(fig, "framework")


def perturbations():
    x = np.linspace(400, 1800, 1401)
    base = .04 + sum(a * np.exp(-.5 * ((x - mu) / sig) ** 2)
                     for a, mu, sig in [(1, 700, 14), (.55, 1020, 24), (.8, 1430, 18)])
    rng = np.random.default_rng(20260920)
    z = rng.normal(size=x.size)
    kernel = np.exp(-.5 * (np.arange(-80, 81) / 20) ** 2)
    corr = np.convolve(z, kernel / kernel.sum(), mode="same")
    corr = (corr - corr.mean()) / corr.std() * .13
    warped_x = x + 65 * ((x - x.min()) / (x.max() - x.min())) ** 2
    curves = [(x, base), (x, base + .30 * ((x - 950) / 850) ** 2 - .13), (x, base + .09 * z),
              (x, base + corr), (x + 25.0, base), (warped_x, base)]
    titles = ["(a) Reference spectrum", "(b) Baseline distortion", "(c) Independent noise",
              "(d) Correlated noise", "(e) Global wavenumber shift", "(f) Nonlinear axis warp"]
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.6), sharex=True, sharey=True, layout="constrained")
    for i, (ax, (xx, yy), title) in enumerate(zip(axs.flat, curves, titles)):
        ax.plot(x, base, color="#A4AFB8", lw=1.1, label="Reference")
        ax.plot(xx, yy, color=BLUE if i == 0 else ORANGE, lw=.8, label="Perturbed")
        ax.set(xlim=(390, 1870), ylim=(-.28, 1.3))
        ax.set_title(title, loc="left", color=INK)
        ax.grid(alpha=.12)
    for ax in axs[1]:
        ax.set_xlabel("Wavenumber / cm$^{-1}$")
    for ax in axs[:, 0]:
        ax.set_ylabel("Intensity (a.u.)")
    axs.flat[1].legend(frameon=False, loc="upper right")
    save(fig, "perturbations")


def ag_oc():
    fig = plt.figure(figsize=(6.5, 4.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.45, 1.0])
    top.suptitle("(a) AG: how much does knowing the perturbation type help?", x=0, ha="left",
                 fontweight="bold", fontsize=9, color=INK)
    ags = []
    for j, (ax, gap, title) in enumerate(zip(top.subplots(1, 2), [.1, 4.0],
                                             ["Small AG: similar relationships across types",
                                              "Large AG: similar metric harm, different task harm"])):
        xx = np.arange(1, 5, dtype=float)
        y1, y2 = xx, xx + gap
        pooled = (y1 + y2) / 2  # pooled isotonic fit: pointwise mean, already monotone
        yy = np.r_[y1, y2]
        ag = float((((y1 - pooled) ** 2).sum() + ((y2 - pooled) ** 2).sum()) / ((yy - yy.mean()) ** 2).sum())
        ags.append(ag)
        ax.plot(xx, y1, "o-", color=BLUE, lw=1.5, ms=3.5, label="Noise: separate fit")
        ax.plot(xx, y2, "s-", color=ORANGE, lw=1.5, ms=3.5, label="Shift: separate fit")
        ax.plot(xx, pooled, "--", color=INK, lw=1.2, label="All data: pooled fit")
        ax.set(xlim=(.7, 4.3), ylim=(0, 9), xlabel="Metric harm x", ylabel="Task harm y (a.u.)" if j == 0 else "")
        ax.set_title(title, fontsize=8, color=INK)
        ax.text(.05, .9, f"Illustrative AG = {ag:.3f}", transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK)
        ax.grid(alpha=.15)
        if j == 1:
            ax.legend(frameon=False, loc="lower right")
    b = bottom.subplots()
    panel(b, "(b) OC: is the more harmful condition ranked correctly?")
    cases = [(1, GREEN, "Matching order", "noise 0.2 < shift 0.7", "Score 1"),
             (35, RED, "Opposite order", "noise 0.7 > shift 0.2", "Score 0"),
             (69, GREY, "Exact metric tie", "noise 0.4 = shift 0.4", "Score 1/2")]
    for x0, col, title, metric, score in cases:
        b.add_patch(FancyBboxPatch((x0, 4), 30, 78, boxstyle="round,pad=0.4,rounding_size=1.2",
                                   facecolor=col + "10", edgecolor=col, linewidth=0.9))
        b.text(x0 + 15, 73, title, ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")
        b.text(x0 + 15, 58, f"Metric harm\n{metric}", ha="center", va="center",
               fontsize=7.5, color=INK, linespacing=1.4)
        b.text(x0 + 15, 36, "Accuracy loss (points)\nnoise 2 < shift 10", ha="center", va="center",
               fontsize=7.5, color=INK, linespacing=1.4)
        b.text(x0 + 15, 14, score, ha="center", va="center", fontsize=10.5, color=col, fontweight="bold")
    save(fig, "ag_oc")
    return ags


if __name__ == "__main__":
    framework()
    perturbations()
    ags = ag_oc()
    receipt = {"figures": sorted(p.name for p in FIG.glob("*.pdf")),
               "design_source": "docs/methods_results_explained_en.md figures 1, 2, 5, 6",
               "synthetic_teaching_data": True, "experiments_rerun": False,
               "illustrative_ag": [round(v, 3) for v in ags],
               "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FIG.glob("*.png"))}}
    (ROOT / "provenance/v5_figures.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Built three v5 explanatory figures:", ", ".join(receipt["figures"]))
