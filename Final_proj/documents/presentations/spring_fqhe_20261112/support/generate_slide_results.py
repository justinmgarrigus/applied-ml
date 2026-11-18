"""Large-label scientific graphics sized for the compact Beamer result frames.

Preserves the original figure set. No training or diagonalization.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/spring-slide-assets")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[3]
OUT = HERE / "figures"
INPUTS = {}
COLORS = ["#234968", "#D88A17", "#078774"]
plt.rcParams.update({
    "font.family":"DejaVu Sans", "font.size":19,
    "axes.labelsize":20, "axes.labelpad":3,
    "xtick.labelsize":17, "ytick.labelsize":17,
    "legend.fontsize":15, "axes.spines.top":False,
    "axes.spines.right":False, "pdf.fonttype":42,
    "axes.linewidth":1.1, "xtick.major.pad":3, "ytick.major.pad":3,
})


def read(relative):
    path = FINAL / relative
    INPUTS[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(fig,name):
    for ax in fig.axes:
        ax.grid(alpha=.15,axis="y")
        ax.set_axisbelow(True)
    fig.savefig(OUT/f"{name}.pdf",bbox_inches="tight",pad_inches=.035)
    fig.savefig(OUT/f"{name}.png",bbox_inches="tight",pad_inches=.035,dpi=200)
    plt.close(fig)


path=FINAL/"SPRING_Ising_test/results/spring_n6/training.csv"
INPUTS[str(path.relative_to(FINAL))]=hashlib.sha256(path.read_bytes()).hexdigest()
with path.open() as f:
    rows=list(csv.DictReader(f))
x=np.array([int(r["step"]) for r in rows])
y=np.array([float(r["energy_error"]) for r in rows])
fig,ax=plt.subplots(figsize=(7,3.1),layout="constrained")
ax.semilogy(x,y,color=COLORS[0],lw=2.7)
ax.scatter(x[-1],y[-1],color=COLORS[0],s=40,zorder=4)
ax.set(xlabel="SPRING updates",ylabel=r"$E-E_0$ ($J$)",xlim=(0,255))
ax.set_xticks([0,100,200,250])
ax.set_yticks([1,1e-2,1e-4])
save(fig,"slide_ising")

depth=read("SPRING_Haldane_FQHE/results/ne8_depth_ablation_20261103/summary.json")
fig,ax=plt.subplots(1,2,figsize=(9,2.35),layout="constrained")
for i,layers in enumerate((1,2,6)):
    group=[r for r in depth["endpoints"] if r["depth"]==layers]
    for panel,field,factor in ((0,"energy_V1",1),(1,"fidelity",100)):
        vals=np.array([r[field] for r in group])*factor
        ax[panel].bar(i,vals.mean(),width=.55,color=COLORS[i],alpha=.42)
        ax[panel].scatter([i-.08,i+.08],vals,color=COLORS[i],s=38,zorder=4)
for a in ax:
    a.set_xticks(range(3),["1","2","6"])
    a.set_xlabel("Hidden layers")
ax[0].set(ylabel=r"$(E-E_0)/V_1$",ylim=(0,2.5),yticks=[0,1,2])
ax[1].set(ylabel="Fidelity (%)",ylim=(0,6),yticks=[0,2,4,6])
save(fig,"slide_n8_depth")

probe=read("SPRING_Haldane_FQHE/results/ne7_gg_explore_20261111/frozen_probe.json")
fig,ax=plt.subplots(1,2,figsize=(9,2.35),layout="constrained")
x=np.arange(3)
seeds=[s["seed"] for s in probe["seeds"]]
for i,beta in enumerate((1.,.5)):
    y=[100*s["ideal_proposals"][str(beta)]["independent_pool_rare_hit_probability"] for s in probe["seeds"]]
    ax[0].bar(x+(-.18 if i==0 else .18),y,width=.34,color=COLORS[i],label="Born" if i==0 else "Broader")
ax[1].bar(x,[100*s["rare_variance_fraction"] for s in probe["seeds"]],color=COLORS[2],alpha=.78)
for a in ax:
    a.set_xticks(x,[str(s) for s in seeds])
    a.set(xlabel="Saved 8k state (seed)",ylim=(0,105),yticks=[0,50,100])
ax[0].set_ylabel("Pool hit (%)")
ax[1].set_ylabel("Variance share (%)",fontsize=18)
ax[0].legend(frameon=False,loc="upper center",bbox_to_anchor=(.53,1.02),ncol=2,
             handlelength=1.1,columnspacing=.75,handletextpad=.3,fontsize=15)
save(fig,"slide_rare")

explore=read("SPRING_Haldane_FQHE/results/ne7_gg_explore_20261111/summary.json")
fig,ax=plt.subplots(1,2,figsize=(9,2.7))
fig.subplots_adjust(left=.095,right=.995,bottom=.29,top=.80,wspace=.39)
for method,color,label in zip(("gg_original","gg_tempered","gg_sequential"),COLORS,
                              ("GG (Born)","Broader","Broader + seq.")):
    rows=[r for r in explore["trajectories"] if r["method"]==method]
    for seed in (71,137,211):
        group=sorted([r for r in rows if r["seed"]==seed],key=lambda r:r["step"])
        ax[0].plot([r["step"] for r in group],[r["energy_error_V1"] for r in group],color=color,alpha=.20,lw=.9)
        ax[1].plot([r["training_seconds"] for r in group],[100*r["fidelity"] for r in group],color=color,alpha=.20,lw=.9)
    steps=sorted({r["step"] for r in rows})
    means=lambda field:np.array([np.mean([r[field] for r in rows if r["step"]==s]) for s in steps])
    ax[0].plot(steps,means("energy_error_V1"),lw=2.4,marker="o",ms=3.7,color=color,label=label)
    ax[1].plot(means("training_seconds"),100*means("fidelity"),lw=2.4,marker="o",ms=3.7,color=color)
ax[0].set(xlabel="Total updates",ylabel=r"$(E-E_0)/V_1$",xticks=[8000,9000,10000],yticks=[.05,.10,.15,.20])
ax[0].xaxis.set_major_formatter(FuncFormatter(lambda value,_:f"{value/1000:g}k"))
ax[1].set(xlabel="Active time (s)",ylabel="Fidelity (%)",ylim=(85,100),xticks=[0,30,60],yticks=[85,90,95,100])
handles,labels=ax[0].get_legend_handles_labels()
fig.legend(handles,labels,frameon=False,loc="upper center",bbox_to_anchor=(.54,1.015),
           fontsize=14,ncol=3,handlelength=1.2,handletextpad=.35,columnspacing=1.0)
save(fig,"slide_broad")
(OUT/"slide_figure_inputs.json").write_text(json.dumps({"input_sha256":INPUTS,
    "no_new_training":True,"no_new_ED":True,"source":"support/generate_slide_results.py",
    "presentation_sizing":"Large labels 20pt, ticks17pt, legends14-15pt at source; compact two-panel aspect."},indent=2)+"\n")
print("Rendered four new large-label slide figure pairs.")
