"""
================================================================================
PHASE 2 — FIGURE AND TABLE GENERATOR
================================================================================
Generates all figures and tables for the Phase 2 paper.

Run: python phase2_figures.py
Outputs: Results/Phase 2 Rerun Clean/figures/

Confirmed results locked in RESULTS_DATA — verified against eval CSVs.
JS values corrected from paper (V3 P0vP1=0.111, BWD P0vP1=0.075, BWD P0vP2=0.056).

Author: Indramuthu Sundaram — Phase 2, North Carolina A&T State University
================================================================================
"""
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from collections import Counter

BASE    = Path(r"C:\Users\gsind\North Carolina A&T State University\Marwan Bikdash - Indra\Indra - Research Folder")
RESULTS = BASE / "Results" / "Phase 2 Rerun Clean"
FIGS    = RESULTS / "figures"
FIGS.mkdir(exist_ok=True)

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
except ImportError:
    HAS_MPL = False; print("WARNING: matplotlib not found")

# ── Locked results ────────────────────────────────────────────────────────────
PERSONA_NAMES = ["P0: Safety", "P1: Scenic", "P2: Efficient"]
P_COLORS      = ["#2E86AB", "#A23B72", "#F18F01"]

RESULTS_DATA = {
    'V1s': {
        'P0':{'success':0.5,  'walk':97,'transit':0, 'access':0, 'car':1},
        'P1':{'success':0.0,  'walk':96,'transit':0, 'access':0, 'car':1},
        'P2':{'success':0.0,  'walk':95,'transit':0, 'access':0, 'car':1},
        'JS':{(0,1):0.0027,(0,2):0.0033,(1,2):0.0001},
        'JS_CI':{(0,1):(0.0000,0.0082),(0,2):(0.0004,0.0093),(1,2):(0.0000,0.0040)},
        'JS_sig':{(0,1):'*',(0,2):'*',(1,2):'ns'},
    },
    'V2Oracle': {
        'P0':{'success':4.5,  'walk':3, 'transit':2, 'access':90,'car':5},
        'P1':{'success':2.0,  'walk':98,'transit':0, 'access':0, 'car':1},
        'P2':{'success':100.0,'walk':17,'transit':29,'access':49,'car':5},
        'JS':{(0,1):0.6086,(0,2):0.1310,(1,2):0.4315},
        'JS_CI':{(0,1):(0.5968,0.6205),(0,2):(0.1217,0.1418),(1,2):(0.4132,0.4509)},
        'JS_sig':{(0,1):'***',(0,2):'***',(1,2):'***'},
    },
    'V3': {
        'P0':{'success':100.0,'walk':33,'transit':20,'access':46,'car':0},
        'P1':{'success':100.0,'walk':34,'transit':46,'access':19,'car':0},
        'P2':{'success':100.0,'walk':18,'transit':55,'access':26,'car':0},
        'JS':{(0,1):0.1114,(0,2):0.1290,(1,2):0.0179},
        'JS_CI':{(0,1):(0.0570,0.2102),(0,2):(0.0574,0.2517),(1,2):(0.0049,0.0620)},
        'JS_sig':{(0,1):'***',(0,2):'***',(1,2):'ns'},
    },
    'BWD': {
        'P0':{'success':100.0,'walk':21,'transit':15,'access':64,'car':0},
        'P1':{'success':100.0,'walk':16,'transit':46,'access':38,'car':0},
        'P2':{'success':100.0,'walk':19,'transit':0, 'access':80,'car':0},
        'JS':{(0,1):0.0746,(0,2):0.0560,(1,2):0.1961},
        'JS_CI':{(0,1):(0.0646,0.0873),(0,2):(0.0484,0.0659),(1,2):(0.1851,0.2091)},
        'JS_sig':{(0,1):'***',(0,2):'***',(1,2):'***'},
    },
}

BWD_RL_DELTA = {
    'V2Oracle':{(0,1):(0.534,0.516,0.551,'***'),(0,2):(0.075,0.063,0.086,'***'),(1,2):(0.235,0.212,0.258,'***')},
    'V3':      {(0,1):(0.037,-0.016,0.138,'ns'),(0,2):(0.073,0.007,0.185,'*'),(1,2):(-0.178,-0.196,-0.143,'ns')},
}

VARIANT_LABELS = {
    'V1s':      'V1s\nPure PPO\n(small graph)',
    'V2Oracle': 'V2-Oracle\nBC-PPO+Oracle\n(full graph)',
    'V3':       'V3\nRoute Selection\n(full graph)',
    'BWD':      'BWD\nBehavior-Weighted\nDijkstra',
}

def wilson_ci(pct, n=200):
    p=pct/100; z=1.96; d=1+z**2/n; c=(p+z**2/(2*n))/d
    m=z*sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return round(max(0,(c-m)*100),1), round(min(100,(c+m)*100),1)

def js_interp(js):
    if js<0.05: return 'negligible'
    if js<0.10: return 'weak'
    if js<0.30: return 'moderate'
    return 'strong'

SAVE_KW = dict(dpi=150, bbox_inches='tight')

# ── TABLE 1 ───────────────────────────────────────────────────────────────────
def make_table1():
    rows=[]
    for var,vdata in RESULTS_DATA.items():
        for i,(pid,pname) in enumerate(zip(['P0','P1','P2'],PERSONA_NAMES)):
            d=vdata[pid]; s=d['success']; ci=wilson_ci(s)
            js01=vdata['JS'].get((0,1),''); js12=vdata['JS'].get((1,2),'')
            sig01=vdata['JS_sig'].get((0,1),''); sig12=vdata['JS_sig'].get((1,2),'')
            rows.append({'Variant':var,'Persona':pname,
                'Success (%)':f"{s:.1f}",'95% CI':f"[{ci[0]},{ci[1]}]",
                'Walk (%)':d['walk'],'Transit (%)':d['transit'],
                'Access (%)':d['access'],'Car (%)':d['car'],
                'JS P0vP1':f"{js01:.4f}{sig01}" if isinstance(js01,float) else '',
                'JS P1vP2':f"{js12:.4f}{sig12}" if isinstance(js12,float) else '',
            })
    df=pd.DataFrame(rows)
    df.to_csv(FIGS/"table1_phase2_results.csv",index=False)
    print(f"Saved: table1_phase2_results.csv")

# ── TABLE 2 ───────────────────────────────────────────────────────────────────
def make_table2():
    rows=[]
    for var,vdata in RESULTS_DATA.items():
        row={'Variant':var}
        for i,j in [(0,1),(0,2),(1,2)]:
            js=vdata['JS'][(i,j)]; lo,hi=vdata['JS_CI'][(i,j)]
            sig=vdata['JS_sig'][(i,j)]
            row[f'JS P{i}vP{j}']=f"{js:.4f} [{lo:.4f},{hi:.4f}]"
            row[f'P{i}vP{j} Sig']=f"{js_interp(js)} {sig}"
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(FIGS/"table2_js_divergence.csv",index=False)
    print(f"Saved: table2_js_divergence.csv")

# ── TABLE 3 ───────────────────────────────────────────────────────────────────
def make_table3():
    rows=[]
    for var in ['V2Oracle','V3']:
        for i,j in [(0,1),(0,2),(1,2)]:
            d,lo,hi,sig=BWD_RL_DELTA[var][(i,j)]
            rows.append({'Comparison':f"{var} vs BWD P{i}vP{j}",
                'Delta JS':round(d,4),'95% CI':f"[{lo:+.4f},{hi:+.4f}]",
                'Sig':sig,'Interpretation':'RL wins' if d>0 and sig!='ns' else
                             'BWD wins' if d<0 and sig!='ns' else 'no sig. diff.'})
    df=pd.DataFrame(rows)
    df.to_csv(FIGS/"table3_bwd_vs_rl.csv",index=False)
    print(f"Saved: table3_bwd_vs_rl.csv")

# ── FIGURE 1: Success Rate ────────────────────────────────────────────────────
def make_figure1():
    if not HAS_MPL: return
    variants=list(RESULTS_DATA.keys())
    x=np.arange(len(variants)); w=0.22
    fig,ax=plt.subplots(figsize=(7,4.5))
    for i,(pid,pname) in enumerate(zip(['P0','P1','P2'],PERSONA_NAMES)):
        vals=[RESULTS_DATA[v][pid]['success'] for v in variants]
        cis=[wilson_ci(s) for s in vals]
        errs=[[s-ci[0] for s,ci in zip(vals,cis)],[ci[1]-s for s,ci in zip(vals,cis)]]
        ax.bar(x+(i-1)*w,vals,w,label=pname,color=P_COLORS[i],alpha=0.85,edgecolor='white',lw=0.5)
        ax.errorbar(x+(i-1)*w,vals,yerr=errs,fmt='none',color='#333',capsize=3,lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v].split('\n')[0] for v in variants])
    ax.set_ylabel('Strict Success Rate (%)')
    ax.set_ylim(0,115)
    ax.set_title('Figure 1. Navigation Success Rate by Variant and Persona\n(N=200, Wilson 95% CI)',pad=8)
    ax.legend(loc='upper left',framealpha=0.9)
    ax.axhline(100,color='gray',lw=0.5,ls=':')
    plt.tight_layout()
    plt.savefig(FIGS/"fig1_success_rate.png",**SAVE_KW); plt.close()
    print("Saved: fig1_success_rate.png")

# ── FIGURE 2: Mode Share  (2×2, 6.5×6.5 in — Word page width) ────────────────
def make_figure2():
    if not HAS_MPL: return
    variants=list(RESULTS_DATA.keys())
    MODE_COLORS={'walk':'#4CAF50','transit':'#2196F3','access':'#90CAF9','car':'#FF5722'}
    fig,axes=plt.subplots(2,2,figsize=(6.5,6.5),sharey=True)
    axes=axes.flatten()
    for ax,var in zip(axes,variants):
        for pi_,pname in enumerate(['P0','P1','P2']):
            d=RESULTS_DATA[var][pname]; bot=0
            for mode,color in MODE_COLORS.items():
                ax.bar(pi_,d.get(mode,0),bottom=bot,color=color,width=0.55,edgecolor='white',lw=0.5)
                bot+=d.get(mode,0)
        ax.set_title(f"{var} — {VARIANT_LABELS[var].split(chr(10))[1]}",fontsize=11,fontweight='bold',pad=6)
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['P0 Safety','P1 Scenic','P2 Efficient'],fontsize=10)
        ax.set_ylabel('Mode Share (%)',fontsize=11)
        ax.tick_params(axis='both',labelsize=10)
        ax.set_ylim(0,115); ax.axhline(100,color='gray',lw=0.8,ls=':')
    patches=[mpatches.Patch(color=c,label=m) for m,c in MODE_COLORS.items()]
    fig.legend(handles=patches,loc='lower center',ncol=4,bbox_to_anchor=(0.5,0.0),fontsize=10)
    fig.suptitle('Figure 2. Mode Share by Variant and Persona',fontsize=12,fontweight='bold',y=0.99)
    plt.tight_layout(rect=[0,0.07,1,0.97])
    plt.savefig(FIGS/"fig2_mode_share.png",**SAVE_KW); plt.close()
    print("Saved: fig2_mode_share.png")

# ── FIGURE 3: JS Heatmap  (2×2, 6.5×6.5 in) ─────────────────────────────────
def make_figure3():
    if not HAS_MPL: return
    variants=list(RESULTS_DATA.keys())
    labels=['P0','P1','P2']
    fig,axes=plt.subplots(2,2,figsize=(6.5,6.5))
    axes=axes.flatten()
    for ax,var in zip(axes,variants):
        mat=np.zeros((3,3))
        for i,j in [(0,1),(0,2),(1,2)]:
            v=RESULTS_DATA[var]['JS'][(i,j)]; mat[i][j]=v; mat[j][i]=v
        im=ax.imshow(mat,cmap='YlOrRd',vmin=0,vmax=0.65,aspect='auto')
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(labels,fontsize=11); ax.set_yticklabels(labels,fontsize=11)
        ax.tick_params(axis='both',labelsize=10)
        ax.set_title(f"{var} — {VARIANT_LABELS[var].split(chr(10))[1]}",fontsize=11,fontweight='bold',pad=6)
        for i in range(3):
            for j in range(3):
                sig=RESULTS_DATA[var]['JS_sig'].get((min(i,j),max(i,j)),'')
                txt=f"{mat[i,j]:.3f}{sig}" if mat[i,j]>0 else "0.000"
                ax.text(j,i,txt,ha='center',va='center',fontsize=10,
                        color='white' if mat[i,j]>0.35 else 'black',fontweight='bold')
        cbar=plt.colorbar(im,ax=ax,shrink=0.85)
        cbar.set_label('JS',fontsize=10); cbar.ax.tick_params(labelsize=9)
    fig.suptitle('Figure 3. Pairwise JS Divergence\n(stars = permutation significance)',fontsize=12,fontweight='bold',y=0.99)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(FIGS/"fig3_js_divergence.png",**SAVE_KW); plt.close()
    print("Saved: fig3_js_divergence.png")

# ── FIGURE 4: JS Summary  (1×3 landscape, 9×4.5 in) ─────────────────────────
def make_figure4():
    if not HAS_MPL: return
    pairs=[(0,1,'P0 vs P1'),(1,2,'P1 vs P2'),(0,2,'P0 vs P2')]
    variants=list(RESULTS_DATA.keys())
    var_colors={'V1s':'#888888','V2Oracle':'#E91E8C','V3':'#00BCD4','BWD':'#FF9800'}
    fig,axes=plt.subplots(1,3,figsize=(9,4.5))
    xs=np.arange(len(variants))
    for ax,(i,j,plabel) in zip(axes,pairs):
        for vi,var in enumerate(variants):
            js=RESULTS_DATA[var]['JS'][(i,j)]
            lo,hi=RESULTS_DATA[var]['JS_CI'][(i,j)]
            sig=RESULTS_DATA[var]['JS_sig'][(i,j)]
            ax.bar(vi,js,color=var_colors[var],alpha=0.85,edgecolor='white',lw=0.5,width=0.6)
            ax.errorbar(vi,js,yerr=[[js-lo],[hi-js]],fmt='none',color='#333',capsize=4,lw=1.5)
            if sig!='ns': ax.text(vi,hi+0.015,sig,ha='center',va='bottom',fontsize=13,color='#333',fontweight='bold')
        ax.set_title(plabel,fontsize=13,fontweight='bold',pad=6)
        ax.set_xticks(xs)
        ax.set_xticklabels(variants,fontsize=11,rotation=20,ha='right')
        ax.tick_params(axis='y',labelsize=11)
        ax.set_ylim(0,max(0.80,ax.get_ylim()[1]))
        ax.axhline(0.10,color='gray',lw=1.0,ls='--',alpha=0.6)
        ax.axhline(0.30,color='gray',lw=1.0,ls=':',alpha=0.6)
        ax.set_ylabel('JS Divergence',fontsize=12)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles=[mpatches.Patch(color=c,label=v) for v,c in var_colors.items()]
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(0.5,-0.01),fontsize=11)
    fig.suptitle('Figure 4. JS Divergence by Persona Pair and Variant\n(95% CI; stars = permutation test)',fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(rect=[0,0.08,1,1.0])
    plt.savefig(FIGS/"fig4_js_summary.png",**SAVE_KW); plt.close()
    print("Saved: fig4_js_summary.png")

# ── FIGURE 5: Behavioral Fingerprint  (2×2, 6.5×6.5 in) ─────────────────────
def make_figure5():
    if not HAS_MPL: return
    variants=list(RESULTS_DATA.keys())
    markers={'P0':'o','P1':'s','P2':'^'}
    fig,axes=plt.subplots(2,2,figsize=(6.5,6.5),sharex=True,sharey=True)
    axes=axes.flatten()
    for ax,var in zip(axes,variants):
        for pi_,pid in enumerate(['P0','P1','P2']):
            d=RESULTS_DATA[var][pid]
            walk=d['walk']; transit=d['transit']+d['access']
            ax.scatter(walk,transit,c=P_COLORS[pi_],marker=markers[pid],s=120,zorder=5,edgecolors='white',lw=1.5)
            ax.text(walk+2,transit+2,pid,fontsize=11,color=P_COLORS[pi_],fontweight='bold')
        ax.set_title(f"{var} — {VARIANT_LABELS[var].split(chr(10))[1]}",fontsize=11,fontweight='bold',pad=6)
        ax.set_xlabel('Walk Share (%)',fontsize=11)
        ax.set_ylabel('Transit Share (%)',fontsize=11)
        ax.tick_params(axis='both',labelsize=10)
        ax.set_xlim(-5,105); ax.set_ylim(-5,105)
        ax.plot(np.linspace(0,100,50),100-np.linspace(0,100,50),color='lightgray',lw=1,ls='--',alpha=0.5)
        ax.grid(alpha=0.2)
    patches=[mpatches.Patch(color=c,label=n) for c,n in zip(P_COLORS,PERSONA_NAMES)]
    fig.legend(handles=patches,loc='lower center',ncol=3,bbox_to_anchor=(0.5,0.0),fontsize=11)
    fig.suptitle('Figure 5. Behavioral Fingerprint: Walk vs Transit',fontsize=12,fontweight='bold',y=0.99)
    plt.tight_layout(rect=[0,0.07,1,0.97])
    plt.savefig(FIGS/"fig5_behavioral_fingerprint.png",**SAVE_KW); plt.close()
    print("Saved: fig5_behavioral_fingerprint.png")

# ── FIGURE 6: BWD vs RL ───────────────────────────────────────────────────────
def make_figure6():
    if not HAS_MPL: return
    pairs=[(0,1,'P0 vs P1'),(0,2,'P0 vs P2'),(1,2,'P1 vs P2')]
    x=np.arange(len(pairs))
    var_colors={'V2Oracle':'#E91E8C','V3':'#00BCD4'}
    fig,axes=plt.subplots(1,2,figsize=(7,4.5))
    for ax,(var,label) in zip(axes,var_colors.items()):
        deltas=[BWD_RL_DELTA[var][(i,j)][0] for i,j,_ in pairs]
        los   =[BWD_RL_DELTA[var][(i,j)][1] for i,j,_ in pairs]
        his   =[BWD_RL_DELTA[var][(i,j)][2] for i,j,_ in pairs]
        sigs  =[BWD_RL_DELTA[var][(i,j)][3] for i,j,_ in pairs]
        ax.bar(x,deltas,color=[var_colors[var] if d>0 else '#aaa' for d in deltas],alpha=0.85,edgecolor='white',width=0.5)
        ax.errorbar(x,deltas,yerr=[[d-lo for d,lo in zip(deltas,los)],[hi-d for d,hi in zip(deltas,his)]],
                    fmt='none',color='#333',capsize=4,lw=1.5)
        for xi,sig in enumerate(sigs):
            y=his[xi]+0.01 if deltas[xi]>=0 else los[xi]-0.02
            ax.text(xi,y,sig,ha='center',va='bottom',fontsize=12,color='#333',fontweight='bold')
        ax.axhline(0,color='black',lw=1)
        ax.set_xticks(x); ax.set_xticklabels([p for _,_,p in pairs])
        ax.set_title(f"{var} vs BWD",fontsize=12,fontweight='bold')
        ax.set_ylabel('Δ JS (RL − BWD)')
        ax.text(0.5,0.97,'+ = RL more differentiated',transform=ax.transAxes,
                ha='center',va='top',fontsize=9,style='italic',color='gray')
    fig.suptitle('Figure 6. Paired Bootstrap: RL vs BWD\n(95% CI; stars = significance)',fontsize=12,fontweight='bold',y=1.0)
    plt.tight_layout()
    plt.savefig(FIGS/"fig6_bwd_vs_rl.png",**SAVE_KW); plt.close()
    print("Saved: fig6_bwd_vs_rl.png")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating Phase 2 figures and tables...")
    print(f"Output: {FIGS}\n")
    make_table1(); make_table2(); make_table3()
    make_figure1(); make_figure2(); make_figure3()
    make_figure4(); make_figure5(); make_figure6()
    print(f"\nDone. All outputs in: {FIGS}")
