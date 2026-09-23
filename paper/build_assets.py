"""Build publication displays by projecting retained results; no experiment rerun."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'raman-paper-mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
TAB = HERE / 'tables'
DATA = HERE / 'data'
for d in (FIG, TAB, DATA):
    d.mkdir(exist_ok=True)
paths = {
    'alignment': ROOT / 'release/phase6-publication-core/table_s1_full_alignment.csv',
    'interactions': ROOT / 'release/phase6-publication-core/table_s2_protocol_interactions.csv',
    'response': ROOT / 'release/phase4-final-figures/figure1_response_data.csv',
    'effects': ROOT / 'release/phase4-final-figures/figure1_protocol_effect_data.csv',
}
source_paths = dict(paths)
paths = {k: p if p.exists() else DATA / p.name for k,p in paths.items()}
rows = {k: list(csv.DictReader(p.open())) for k, p in paths.items()}
for p in paths.values():
    if p.resolve() != (DATA / p.name).resolve():
        shutil.copy2(p, DATA / p.name)
receipts = {k: {'source': str(source_paths[k].relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'rows': len(rows[k])} for k, p in paths.items()}
(HERE / 'source_receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
assert {k: len(v) for k, v in rows.items()} == {'alignment':143,'interactions':65,'response':440,'effects':225}
panels = ['d1_a','d2_5_a','d2_5_b','d2_10_a','d2_10_b','d2_20_a','d2_20_b','d4_a','d4_b','d5_a','d5_b']
names = {'d1_a':'Bacteria full / A', 'd2_5_a':'Bacteria 5-shot / A','d2_5_b':'Bacteria 5-shot / B',
         'd2_10_a':'Bacteria 10-shot / A','d2_10_b':'Bacteria 10-shot / B',
         'd2_20_a':'Bacteria 20-shot / A','d2_20_b':'Bacteria 20-shot / B',
         'd4_a':'Sugar / A','d4_b':'Sugar / B','d5_a':'RRUFF / A','d5_b':'RRUFF / B'}
metrics = ['mse','rmse','mae','sam','pearson_r','nmse','wasserstein_1_cm1',
           'is_like_structure_to_noise','precision','recall','f1','artifact_peak_ratio','missing_peak_ratio']
mn = dict(zip(metrics, ['MSE','RMSE','MAE','SAM','Pearson r','NMSE','W1','S/N','Precision','Recall','F1','Artifact ratio','Missing ratio']))
lookup = {(r['panel_id'],r['metric_output_id']):r for r in rows['alignment']}
assert len(lookup)==143
for p in panels:
    assert all((p,m) in lookup for m in metrics)
    for m in metrics[1:]:
        r=lookup[p,m]; base=lookup[p,'mse']
        assert abs(float(r['d_ag'])-(float(base['ag'])-float(r['ag'])))<1e-12
        assert abs(float(r['d_acc'])-(float(r['acc_cross'])-float(base['acc_cross'])))<1e-12

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
pids=['p08','p09','p10','p11','p12']
plabels=['Baseline distortion','Gaussian noise','Correlated noise','Axis shift','Axis warp']

# Response figures use actual alpha values and preserve the original harm units.
fig, axs=plt.subplots(3,2,figsize=(7.1,7.5),layout='constrained')
for i,(cell,title) in enumerate([('d2_10','Bacteria 10-shot'),('d4','Sugar quantification'),('d5','RRUFF retrieval')]):
    for j,proto in enumerate(['a','b']):
        ax=axs[i,j]
        for pid,col,lab in zip(pids,colors,plabels):
            rr=sorted([r for r in rows['response'] if r['cell_id']==cell and r['protocol_id']==proto and r['perturbation_id']==pid],key=lambda r:float(r['alpha']))
            assert len(rr)==8
            ax.plot([float(r['alpha']) for r in rr],[float(r['downstream_harm']) for r in rr],color=col,marker='o',ms=2.8,lw=1.4,label=lab)
        ax.axhline(0,color='#999999',lw=.65,zorder=0)
        ax.set_title(f'{chr(97+2*i+j)}  {title} / {proto.upper()}',loc='left')
        ax.set_xlabel(r'Perturbation severity $\alpha$')
        ax.set_ylabel('Normalized squared-loss increase' if cell=='d4' else 'Accuracy loss')
        ax.grid(alpha=.13)
        if cell=='d4':
            ax.set_yscale('symlog',linthresh=.005)
            ax.set_ylim(-.05,100)
            ax.set_yticks([-.01,0,.01,.1,1,10,100],['-0.01','0','0.01','0.1','1','10','100'])
            ax.set_ylabel('Normalized squared-loss increase')
    if cell!='d4':
        lo=min(a.get_ylim()[0] for a in axs[i]); hi=max(a.get_ylim()[1] for a in axs[i])
        for a in axs[i]: a.set_ylim(lo,hi)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.savefig(FIG/'responses.pdf');fig.savefig(FIG/'responses.png');plt.close(fig)

# Every candidate and every completed endpoint/protocol appears, with no ranking selection.
fig,axs=plt.subplots(1,2,figsize=(7.2,5.3),layout='constrained',sharey=True)
for ax,field,title in zip(axs,['d_ag','d_acc'],[r'a  $\Delta AG = AG_{MSE}-AG_m$',r'b  $\Delta Acc = Acc_m-Acc_{MSE}$']):
    arr=np.array([[float(lookup[p,m][field]) for m in metrics[1:]] for p in panels])
    vmax=float(np.max(np.abs(arr)))
    im=ax.imshow(arr,cmap='RdBu',norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),aspect='auto')
    for y,p in enumerate(panels):
        for x,m in enumerate(metrics[1:]):
            r=lookup[p,m]
            if float(r[field+'_adjusted_p_value']) < .05:
                ax.text(x,y,'*',ha='center',va='center',fontsize=10,color='white' if abs(arr[y,x])>vmax*.48 else '#222222')
    ax.set_xticks(range(12),[mn[m] for m in metrics[1:]],rotation=62,ha='right',fontsize=7)
    ax.set_yticks(range(11),[names[p] for p in panels],fontsize=8)
    ax.set_title(title,loc='left')
    ax.spines[['left','bottom']].set_visible(False)
    fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.9,pad=.01,label='Favorable contrast toward blue')
fig.savefig(FIG/'alignment.pdf');fig.savefig(FIG/'alignment.png');plt.close(fig)

# Matched protocol differences of W1-vs-MSE effects, from original interaction inference.
fig,axs=plt.subplots(1,2,figsize=(7.1,2.65),layout='constrained',sharey=True)
cells=['d2_5','d2_10','d2_20','d4','d5']
cn=['Bacteria 5-shot','Bacteria 10-shot','Bacteria 20-shot','Sugar','RRUFF']
for ax,k,title in zip(axs,['i_ag','i_acc'],[r'a  $I_{AG}=\Delta AG_B-\Delta AG_A$',r'b  $I_{Acc}=\Delta Acc_B-\Delta Acc_A$']):
    for y,c in enumerate(cells):
        r=next(r for r in rows['interactions'] if r['cell_id']==c and r['metric_output_id']=='wasserstein_1_cm1')
        v,lo,hi=[float(r[q]) for q in [k,k+'_lower',k+'_upper']]
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color='#A64732',capsize=3,ms=5)
        assert r[k+'_rejected']=='True' and r[k+'_direction']=='adverse'
    ax.axvline(0,color='#555555',lw=.8,ls='--');ax.set_title(title,loc='left',fontsize=9)
    ax.set_yticks(range(5),cn);ax.grid(axis='x',alpha=.15);ax.set_xlabel('W1 relative advantage change (B minus A)')
    if k=='i_acc':
        ax.set_xticks([-.12,-.08,-.04,0],['-0.12','-0.08','-0.04','0.00'])
axs[0].invert_yaxis()
fig.savefig(FIG/'interactions.pdf');fig.savefig(FIG/'interactions.png');plt.close(fig)

def f(x):
    v=float(x)
    return f'{0.0 if abs(v)<.00005 else v:.4f}'
def pv(x):
    v=float(x)
    return r'$<0.001$' if v<.001 else f'{v:.3f}'
def star(r,k):return r'$^{*}$' if float(r[k+'_adjusted_p_value']) < .05 else ''
def ci(r,k):return f"{f(r[k])} [{f(r[k+'_lower'])}, {f(r[k+'_upper'])}]"

with (TAB/'overview.tex').open('w') as out:
    for p in panels:
        b,w=lookup[p,'mse'],lookup[p,'wasserstein_1_cm1']
        out.write(f"{names[p]} & {f(b['ag'])} & {f(w['ag'])} & {f(b['acc_cross'])} & {f(w['acc_cross'])} & {f(w['d_ag'])}{star(w,'d_ag')} & {f(w['d_acc'])}{star(w,'d_acc')} \\\\\n")

with (TAB/'full_alignment.tex').open('w') as out:
    for ix,p in enumerate(panels):
        if ix in [2,5,8]:out.write('\\clearpage\n')
        out.write(r'\subsection*{'+names[p]+ '}\n'+r'\noindent{\small\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}\toprule'+'\n')
        out.write(r'Metric & $AG$ & $Acc_{\times}$ & $\Delta AG$ & $p_{H,AG}$ & $\Delta Acc$ & $p_{H,Acc}$ \\ \midrule'+'\n')
        for m in metrics:
            r=lookup[p,m]
            vals=[mn[m],f(r['ag']),f(r['acc_cross'])]
            vals += ['--']*4 if m=='mse' else [f(r['d_ag']),pv(r['d_ag_adjusted_p_value']),f(r['d_acc']),pv(r['d_acc_adjusted_p_value'])]
            out.write(' & '.join(vals)+r' \\'+'\n')
        out.write(r'\bottomrule\end{tabular*}\par}'+('\n' if ix==len(panels)-1 else '\n\n'))

with (TAB/'w1_intervals.tex').open('w') as out:
    for p in panels:
        r=lookup[p,'wasserstein_1_cm1']
        out.write(' & '.join([names[p],ci(r,'d_ag'),ci(r,'d_acc')])+r' \\'+'\n')

with (TAB/'protocol_effects.tex').open('w') as out:
    for cell,title in zip(cells,cn):
        rr=[r for r in rows['effects'] if r['cell_id']==cell and r['summary_type']=='integrated']
        for i,r in enumerate(rr):
            typ=plabels[pids.index(r['perturbation_id'])]
            out.write(' & '.join([title if i==0 else '',typ,ci(r,'g_harm'),pv(r['adjusted_p_value'])])+r' \\'+'\n')
        if cell!=cells[-1]:out.write(r'\addlinespace'+'\n')

# Transparent, human-readable review of projection invariants.
checks={'retained_csv_counts_verified':True,'all_143_metric_rows_included':True,
        'contrast_signs_match_original_estimates':True,'w1_interactions_verified_in_all_five_cells':True,
        'recomputed_experimental_statistics':False,
        'd1_cluster_label':'class_label (execution config/code override publication display typo patient)',
        'figure_transforms':'response vs alpha; sugar symlog disclosed; existing intervals and p-values preserved',
        'significance_glyphs':'adjusted_p_value < 0.05; historical within-protocol rejected flag marks favorable rejections only'}
(HERE/'asset_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print('Built 3 vector figures and complete result tables from checksummed aggregate CSVs.')
