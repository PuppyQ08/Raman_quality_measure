"""Build v4-only figures and tables from retained aggregates; leave v3 intact."""
from pathlib import Path
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures/v4"
TAB = ROOT / "tables/v4"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
def read(path):
    with (ROOT / path).open() as f:
        return list(csv.DictReader(f))
alignment = read("data/table_s1_full_alignment.csv")
robust = read("data/w1_robustness/alignment_summary.csv")
responses = read("data/figure1_response_data.csv")
interactions = read("data/table_s2_protocol_interactions.csv")
PANELS = ["d1_a","d2_5_a","d2_5_b","d2_10_a","d2_10_b","d2_20_a","d2_20_b","d4_a","d4_b","d5_a","d5_b"]
LABELS = ["Bacteria full / Fixed","Bacteria 5 / Fixed","Bacteria 5 / Adapted","Bacteria 10 / Fixed","Bacteria 10 / Adapted","Bacteria 20 / Fixed","Bacteria 20 / Adapted","Sugar / Fixed","Sugar / Adapted","Mineral / Fixed","Mineral / Adapted"]
METRICS = ["rmse","mae","nmse","sam","pearson_r","wasserstein_1_cm1","is_like_structure_to_noise","precision","recall","f1","artifact_peak_ratio","missing_peak_ratio"]
MLABELS = ["RMSE","MAE","NMSE","SAM","Pearson","W1","S/N","Precision","Recall","F1","Artifact","Missing"]
INDEX = {(r["panel_id"], r["metric_output_id"]): r for r in alignment}
W1 = "wasserstein_1_cm1"
RIDX = {(r["panel_id"], r["analysis_id"]): r for r in robust if r["metric_output_id"] == W1}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,
                     "axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
                     "pdf.fonttype":42,"savefig.dpi":200,"axes.spines.top":False,
                     "axes.spines.right":False})

def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"), metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(FIG / (name + ".png"), dpi=200)
    plt.close(fig)


def framework():
    fig = plt.figure(figsize=(6.5, 6.0), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.55])
    blue, orange, ink = "#16618a", "#a65229", "#293846"
    def panel(ax, title):
        ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
        ax.text(0,1,title,va="top",fontweight="bold",fontsize=9,color=ink)
    def box(ax, x,y,w,h, text, color=blue, size=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012",ec=color,fc=color+"0d",lw=.8))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,color=ink)
    def arrow(ax,a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=.8,color="#667784"))
    a=fig.add_subplot(gs[0,:]); panel(a,"(a) Pair a spectral change with its task consequence")
    box(a,.015,.55,.25,.25,"Reference spectrum")
    box(a,.38,.55,.25,.25,"Perturbation\ntype + strength")
    box(a,.745,.55,.24,.25,"Perturbed spectrum")
    arrow(a,(.28,.675),(.36,.675)); arrow(a,(.65,.675),(.725,.675))
    box(a,.05,.05,.40,.30,"Metric harm x\nChange in spectral quality measure")
    box(a,.55,.05,.40,.30,"Task harm y\nChange in classification / quantification /\nidentification loss",orange,7.7)
    arrow(a,(.50,.54),(.25,.37)); arrow(a,(.67,.54),(.75,.37))
    b=fig.add_subplot(gs[1,:]); panel(b,"(b) Two ways to establish the downstream analysis")
    box(b,.015,.33,.46,.47,"Fixed\nFit to unperturbed training spectra;\nkeep reference library unchanged")
    box(b,.525,.33,.46,.47,"Adapted\nFit to each perturbed training condition;\nperturb reference library to match",orange)
    b.text(.5,.04,"Evaluate the same perturbed test or query spectra",ha="center",va="bottom",fontsize=8)
    arrow(b,(.24,.31),(.38,.16)); arrow(b,(.76,.31),(.62,.16))
    c=fig.add_subplot(gs[2,0]); panel(c,"(c) AG: benefit of separate curves")
    for pos, large, title in [(.06,True,"Larger AG"),(.57,False,"Smaller AG")]:
        ax=c.inset_axes([pos,.28,.38,.48])
        x=np.array([.12,.35,.60,.85])
        low=.10+.4*x; high=(.60+.4*x) if large else low
        ax.plot(x,low,color=blue,lw=2,marker="o",ms=2.6)
        ax.plot(x,high,color=orange,lw=2,marker="o",ms=2.6,alpha=.8)
        ax.plot(x,(low+high)/2,color=ink,ls="--",lw=1.5)
        ax.set(xlim=(0,1),ylim=(0,1),xticks=[],yticks=[],xlabel="x")
        if large: ax.set_ylabel("y",labelpad=2)
        ax.set_title(title,fontsize=8,pad=5)
    c.text(.05,.13,"- -  One pooled curve",fontsize=7.6,color=ink)
    c.text(.05,.035,"Colored curves: one per perturbation type",fontsize=7.3,color=ink)
    d=fig.add_subplot(gs[2,1]); panel(d,"(d) OC: agreement in ordering")
    d.text(.02,.83,"Example task harm: baseline > noise",fontsize=8)
    for y, text, score, color in [(.58,"Metric: baseline > noise","1",blue),(.36,"Metric: baseline < noise","0",orange),(.14,"Exact tie in x or y","1/2","#666666")]:
        box(d,.025,y,.77,.16,text,color,7.8)
        d.text(.92,y+.08,score,ha="center",va="center",fontsize=10,color=color,fontweight="bold")
    save(fig,"framework")


def profile_matrix():
    fig,axes=plt.subplots(2,1,figsize=(6.5,6.4),layout="constrained",sharex=True)
    for ax,field,title in zip(axes,("d_ag","d_acc"),(r"(a) $\Delta$AG",r"(b) $\Delta$OC")):
        arr=np.array([[float(INDEX[p,m][field]) for m in METRICS] for p in PANELS])
        limit=np.max(np.abs(arr))
        im=ax.imshow(arr,cmap="RdBu",norm=TwoSlopeNorm(0,-limit,limit),aspect="auto")
        for i,p in enumerate(PANELS):
            for j,m in enumerate(METRICS):
                if float(INDEX[p,m][field+"_adjusted_p_value"])<.05:
                    ax.text(j,i,"*",ha="center",va="center",fontsize=8,
                            color="white" if abs(arr[i,j])>.5*limit else "#222222")
        ax.set_yticks(range(11),LABELS,fontsize=7.3); ax.tick_params(length=0,pad=4)
        ax.set_title(title,loc="left",fontweight="bold")
        for x in (2.5,4.5,5.5): ax.axvline(x,color="white",lw=1.6)
        for y in (6.5,8.5): ax.axhline(y,color="white",lw=1.6)
        for spine in ax.spines.values(): spine.set_visible(False)
        bar=fig.colorbar(im,ax=ax,orientation="vertical",fraction=.035,pad=.02)
        bar.ax.tick_params(labelsize=7)
    axes[1].set_xticks(range(12),MLABELS,rotation=45,ha="right",fontsize=8)
    save(fig,"alignment")


def task_responses():
    colors=["#0072B2","#D55E00","#009E73","#CC79A7","#B57900"]
    names=["Baseline","Independent noise","Correlated noise","Shift","Warp"]
    fig,axs=plt.subplots(3,2,figsize=(6.5,6.0),layout="constrained")
    for i,(cell,title) in enumerate([("d2_10","Bacteria 10-shot"),("d4","Sugar"),("d5","Mineral")]):
        for j,(proto,label) in enumerate([("a","Fixed"),("b","Adapted")]):
            ax=axs[i,j]
            for pid,col,name in zip(["p08","p09","p10","p11","p12"],colors,names):
                rr=sorted([r for r in responses if r["cell_id"]==cell and r["protocol_id"]==proto and r["perturbation_id"]==pid],key=lambda r:float(r["alpha"]))
                assert len(rr)==8
                ax.plot([float(r["alpha"]) for r in rr],[float(r["downstream_harm"]) for r in rr],color=col,marker="o",ms=2,lw=1.2,label=name)
            ax.axhline(0,color="#999999",lw=.5); ax.grid(alpha=.12)
            ax.set_title(f"({chr(97+2*i+j)}) {title} / {label}",loc="left",fontsize=8.5)
            ax.set_xlabel(r"Perturbation strength $\alpha$")
            ax.set_ylabel("Task harm (accuracy loss)" if cell!="d4" else "Task harm (normalized loss)")
            if cell=="d4":
                ax.set_yscale("symlog",linthresh=.005); ax.set_ylim(-.05,100)
                ax.set_yticks([-.01,0,.01,.1,1,10,100],["-.01","0",".01",".1","1","10","100"])
        if cell!="d4":
            lo=min(ax.get_ylim()[0] for ax in axs[i]); hi=max(ax.get_ylim()[1] for ax in axs[i])
            for ax in axs[i]: ax.set_ylim(lo,hi)
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False,fontsize=8)
    save(fig,"responses")


def w1_comparison():
    fig=plt.figure(figsize=(6.5,6.4),layout="constrained")
    gs=fig.add_gridspec(2,2,height_ratios=[2.5,1])
    analyses=["native_no_axis","common_grid_all5","common_grid_no_axis"]
    for j,outcome in enumerate(("ag","oc")):
        ax=fig.add_subplot(gs[0,j])
        arr=np.array([[float(RIDX[p,a]["delta_"+outcome]) if RIDX[p,a]["delta_"+outcome] else np.nan for a in analyses] for p in PANELS])
        limit=np.nanmax(np.abs(arr)); cmap=plt.get_cmap("RdBu").copy(); cmap.set_bad("#ededed")
        im=ax.imshow(arr,cmap=cmap,vmin=-limit,vmax=limit,aspect="auto")
        for i,p in enumerate(PANELS):
            for k,a in enumerate(analyses):
                v=arr[i,k]
                text="NA" if np.isnan(v) else f"{v:+.3f}"+("*" if float(RIDX[p,a]["adjusted_p_"+outcome])<.05 else "")
                ax.text(k,i,text,ha="center",va="center",fontsize=7.4,color="white" if abs(v)>.57*limit else "#222222")
        ax.set_xticks(range(3),["Native\nno axes","Grid\nall five","Grid\nno axes"],fontsize=7.4)
        ax.set_yticks(range(11),LABELS if j==0 else [""]*11,fontsize=7)
        ax.tick_params(length=0,pad=4); ax.set_title(f"({chr(97+j)}) "+r"$\Delta$"+outcome.upper()+": controls",loc="left",fontweight="bold")
        for y in (6.5,8.5): ax.axhline(y,color="white",lw=1.6)
        for sp in ax.spines.values(): sp.set_visible(False)
        fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.045,pad=.035)
    for j,(key,outcome) in enumerate((("i_ag","AG"),("i_acc","OC"))):
        ax=fig.add_subplot(gs[1,j])
        for i,cell in enumerate(("d2_5","d2_10","d2_20","d4","d5")):
            r=next(r for r in interactions if r["cell_id"]==cell and r["metric_output_id"]==W1)
            v,lo,hi=[float(r[k]) for k in (key,key+"_lower",key+"_upper")]
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color="#a54831",ms=3.5,capsize=2)
        ax.set_yticks(range(5),["Bacteria 5","Bacteria 10","Bacteria 20","Sugar","Mineral"] if j==0 else [""]*5,fontsize=7.5)
        ax.invert_yaxis(); ax.axvline(0,color="#555555",ls="--",lw=.7); ax.grid(axis="x",alpha=.15)
        ax.set_title(f"({chr(99+j)}) {outcome} interaction: native / five",loc="left",fontsize=8.3,fontweight="bold")
        ax.set_xlabel("Advantage: Adapted minus Fixed",fontsize=7.3)
    save(fig,"w1_comparison")


def tables():
    files=("full_alignment.tex","representative_contrasts.tex","w1_intervals.tex","protocol_effects.tex",
           "w1_robustness.tex","w1_axis_harm_shares.tex","w1_ag_family_example.tex")
    for name in files:
        s=(ROOT/"tables"/name).read_text()
        s=s.replace(r"Acc_{\times}","OC").replace(r"\Delta Acc",r"\Delta OC").replace("p_{H,Acc}","p_{H,OC}")
        s=s.replace(" / A"," / Fixed").replace(" / B"," / Adapted")
        s=s.replace("RRUFF", "Mineral")
        s=s.replace("Gaussian noise", "Independent noise")
        if name == "full_alignment.tex":
            s=s.replace("\\clearpage\n", "")
        s=re.sub(r"(?<=\S) A(?= &)"," / Fixed",s); s=re.sub(r"(?<=\S) B(?= &)"," / Adapted",s)
        (TAB/name).write_text(s)
    # Point estimates are unchanged; names and notation are display-only edits.
    assert len(INDEX)==143 and len(RIDX)==44


def mae_diagnostic():
    source = ROOT.parent / "reports/robustness/w1_axis/oc_pair_summary.csv"
    dest = ROOT / "data/v4/mae_oc_pair_diagnostic.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        with source.open() as f:
            rows = list(csv.DictReader(f))
        selected = [r for r in rows if r["panel_id"] == "d2_10_a"
                    and r["representation"] == "native"
                    and r["metric_output_id"] in ("mse", "mae")]
        assert len(selected) == 20
        with dest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=selected[0].keys(), lineterminator="\n")
            writer.writeheader(); writer.writerows(selected)
    if not dest.exists():
        raise FileNotFoundError("Missing retained MAE pair diagnostic")


if __name__=="__main__":
    framework(); profile_matrix(); task_responses(); w1_comparison(); tables(); mae_diagnostic()
    sources=["data/table_s1_full_alignment.csv","data/table_s2_protocol_interactions.csv",
             "data/figure1_response_data.csv","data/figure1_protocol_effect_data.csv",
             "data/w1_robustness/alignment_summary.csv", "data/v4/mae_oc_pair_diagnostic.csv"]
    receipt={"inputs":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
             "alignment_rows":143,"w1_robustness_rows":44,"experiments_rerun":False,
             "framework_curves":"Schematic illustrations, not measured fits",
             "protocol_display_mapping":{"a":"Fixed","b":"Adapted"},
             "metric_group_order":METRICS}
    (ROOT/"provenance/v4_assets.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Built four v4 figures and seven tables without changing v3 assets.")
