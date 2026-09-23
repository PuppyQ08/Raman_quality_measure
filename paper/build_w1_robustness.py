"""Project the accepted W1 robustness results into manuscript displays.

Uses retained aggregate tables only; no spectra, fitting, or inference reruns.
The original result tables and their inference families remain unchanged.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "reports/robustness/w1_axis"
DATA = ROOT / "data/w1_robustness"
STATISTICS_RUN_ID = "w1-axis-statistics-230046cbc65dace8"
FILES = ("alignment_summary.csv", "ag_family_decomposition.csv", "oc_group_summary.csv",
         "task_harm_shares.csv", "protocol_interactions.csv", "bridge_comparisons.csv")
W1 = "wasserstein_1_cm1"
PANELS = ("d1_a", "d2_5_a", "d2_5_b", "d2_10_a", "d2_10_b", "d2_20_a",
          "d2_20_b", "d4_a", "d4_b", "d5_a", "d5_b")
LABELS = ("Bacteria full A", "Bacteria 5-shot A", "Bacteria 5-shot B", "Bacteria 10-shot A",
          "Bacteria 10-shot B", "Bacteria 20-shot A", "Bacteria 20-shot B",
          "Sugar A", "Sugar B", "Mineral A", "Mineral B")
ANALYSES = ("native_no_axis", "common_grid_all5", "common_grid_no_axis")
ANALYSIS_LABELS = ("Native, no axes", "Common grid, all five", "Common grid, no axes")


def rows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def estimate(r, field, digits=4):
    if r[field] == "":
        return "Unavailable"
    lo, hi = (r[f"bootstrap_{field}_{suffix}"] for suffix in ("lower", "upper"))
    return f"{float(r[field]):.{digits}f} [{float(lo):.{digits}f}, {float(hi):.{digits}f}]"


def pvalue(value):
    if not value:
        return "--"
    return "$<0.001$" if float(value) < .001 else f"{float(value):.3f}"


def build():
    DATA.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(PACKAGE / name, DATA / name)
    expected = dict(reversed(line.split(maxsplit=1))
                    for line in (PACKAGE / "SHA256SUMS").read_text().splitlines())
    hashes = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in FILES}
    assert all(hashes[name] == expected[name] for name in FILES)
    summary = rows("alignment_summary.csv")
    index = {(r["panel_id"], r["analysis_id"]): r for r in summary if r["metric_output_id"] == W1}
    assert len(index) == 44

    table = []
    for analysis, label in zip(ANALYSES, ANALYSIS_LABELS):
        table.append(r"\multicolumn{5}{@{}l}{\normalsize\textbf{" + label + r"}}\\[2pt]")
        for panel, name in zip(PANELS, LABELS):
            r = index[panel, analysis]
            table.append(" & ".join((name, estimate(r, "delta_ag"), estimate(r, "delta_oc"),
                                     pvalue(r["adjusted_p_ag"]), pvalue(r["adjusted_p_oc"]))) + r" \\")
        table.append(r"\midrule")
    (ROOT / "tables/w1_robustness.tex").write_text("\n".join(table[:-1]) + "\n")

    shares = {r["panel_id"]: r for r in rows("task_harm_shares.csv") if r["component"] == "axis_p11_p12"}
    table = []
    for panel, name in zip(PANELS, LABELS):
        r = shares[panel]
        cells = [name]
        for kind in ("positive", "absolute"):
            val, lo, hi = (100 * float(r[f"{kind}_harm_share{suffix}"]) for suffix in ("", "_lower", "_upper"))
            cells.append(f"{val:.3f} [{lo:.3f}, {hi:.3f}]")
        table.append(" & ".join(cells) + r" \\")
    (ROOT / "tables/w1_axis_harm_shares.tex").write_text("\n".join(table) + "\n")

    ag = {(r["metric_output_id"], r["family"]): r for r in rows("ag_family_decomposition.csv")
          if r["panel_id"] == "d2_10_a" and r["representation"] == "native" and r["perturbation_scope"] == "all5"}
    table = []
    for family, name in zip(("p08", "p09", "p10", "p11", "p12"),
                            ("Baseline", "Independent noise", "Correlated noise", "Global shift", "Axis warp")):
        mse, w1 = (float(ag[m, family]["d_p"]) for m in ("mse", W1))
        table.append(f"{name} & {mse:.6f} & {w1:.6f} & {mse-w1:.6f}" + r" \\")
    (ROOT / "tables/w1_ag_family_example.tex").write_text("\n".join(table) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 5.1), layout="constrained", sharey=True)
    for ax, outcome, title in zip(axs, ("ag", "oc"), (r"(a) $\Delta$AG", r"(b) $\Delta$OC")):
        values = np.asarray([[float(index[p, a]["delta_" + outcome]) if index[p, a]["delta_" + outcome] else np.nan
                              for a in ANALYSES] for p in PANELS])
        limit = float(np.nanmax(np.abs(values)))
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#ededed")
        im = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.set_xticks(range(3), ("Native\nno axes", "Common grid\nall five", "Common grid\nno axes"), fontsize=8)
        ax.set_yticks(range(11), LABELS, fontsize=8)
        ax.tick_params(axis="both", length=0, pad=6)
        for i, panel in enumerate(PANELS):
            for j, analysis in enumerate(ANALYSES):
                value = values[i, j]
                if np.isnan(value):
                    ax.text(j, i, "NA", ha="center", va="center", color="#555555", fontsize=8)
                else:
                    p = float(index[panel, analysis]["adjusted_p_" + outcome])
                    label = f"{value:+.3f}" + ("*" if p < .05 else "")
                    ax.text(j, i, label, ha="center", va="center", fontsize=7.7,
                            color="white" if abs(value) > .57*limit else "#222222")
        for boundary in (6.5, 8.5):
            ax.axhline(boundary, color="white", lw=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.06, pad=.045, shrink=.9)
        bar.ax.tick_params(labelsize=8)
        bar.set_label("MSE favored  <  0  >  W1 favored", fontsize=8)
    fig.savefig(ROOT / "figures/w1_robustness.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(ROOT / "figures/w1_robustness.png", dpi=190)
    plt.close(fig)
    receipt = {"statistics_run_id": STATISTICS_RUN_ID, "input_sha256": hashes,
               "w1_rows": 44, "new_w1_rows_displayed": 33, "harm_share_rows": 66,
               "new_jointly_favorable_fixed_bacteria_cells": sum(
                   float(index[p, a]["delta_ag"]) > 0 and float(index[p, a]["delta_oc"]) > 0
                   and float(index[p, a]["adjusted_p_ag"]) < .05 and float(index[p, a]["adjusted_p_oc"]) < .05
                   for p in ("d1_a", "d2_5_a", "d2_10_a", "d2_20_a") for a in ANALYSES),
               "experiment_rerun": False}
    (ROOT / "provenance/w1_robustness_assets.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    build()
