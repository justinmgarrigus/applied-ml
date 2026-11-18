"""Render presentation figures from retained, independently evaluated results.

No training, ED, or source-result mutation. Re-run with .venv-fqh-gpu/bin/python.
"""
from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spring-slides-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[3]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INPUTS = {}


def read(relative):
    path = FINAL / relative
    INPUTS[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42})
COLORS = ["#234968", "#D88A17", "#078774"]


def save(fig, name):
    for ax in fig.axes:
        ax.grid(alpha=.16, axis="y")
        ax.set_axisbelow(True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


ising = read("SPRING_Ising_test/results/spring_n6/metrics.json")
path = FINAL / "SPRING_Ising_test/results/spring_n6/training.csv"
INPUTS[str(path.relative_to(FINAL))] = hashlib.sha256(path.read_bytes()).hexdigest()
with path.open() as stream:
    rows = list(csv.DictReader(stream))
steps = np.array([int(r["step"]) for r in rows])
err = np.array([float(r["energy_error"]) for r in rows])
infid = np.array([1-float(r["ground_state_fidelity"]) for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(10.9, 3.45), constrained_layout=True)
ax[0].semilogy(steps, err, color=COLORS[0], lw=2.2)
ax[0].scatter(steps[-1], err[-1], color=COLORS[0], s=35)
ax[0].set(xlabel="SPRING updates", ylabel=r"Exact energy error $E-E_0$", title="Energy: all 64 configurations evaluated")
ax[1].semilogy(steps, infid, color=COLORS[2], lw=2.2)
ax[1].scatter(steps[-1], infid[-1], color=COLORS[2], s=35)
ax[1].set(xlabel="SPRING updates", ylabel=r"Exact infidelity $1-F$", title=r"$F=|\langle\psi_{\rm ED}|\psi_\theta\rangle|^2$")
save(fig, "ising_exact_convergence")
fig, a = plt.subplots(figsize=(7.6, 3.25), constrained_layout=True)
a.semilogy(steps, err, color=COLORS[0], lw=2.4)
a.scatter(steps[-1], err[-1], color=COLORS[0], s=40)
a.set(xlabel="SPRING updates", ylabel=r"Exact energy error $E-E_0$")
save(fig, "ising_exact_energy_single")

depth = read("SPRING_Haldane_FQHE/results/ne8_depth_ablation_20261103/summary.json")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.45), constrained_layout=True)
for i, layer in enumerate((1,2,6)):
    group = [r for r in depth["endpoints"] if r["depth"] == layer]
    for panel, field, factor in ((0,"energy_V1",1),(1,"fidelity",100),(2,"effective_occupation_support",1)):
        vals = np.array([r[field] for r in group]) * factor
        if panel < 2:
            ax[panel].bar(i, vals.mean(), color=COLORS[i], alpha=.45, width=.55)
        ax[panel].scatter(np.array([i-.08,i+.08]), vals, color=COLORS[i], s=35, zorder=4)
    ax[2].scatter(i, np.mean([r["effective_occupation_support"] for r in group]),
                  marker="_", s=500, linewidths=2, color=COLORS[i])
for a in ax:
    a.set_xticks(range(3), ["1", "2", "6"])
    a.set_xlabel("Hidden layers (width 64)")
ax[0].set(ylabel=r"Exact $(E-E_0)/V_1$", title="Energy after 5,000 updates")
ax[1].set(ylabel="ED fidelity (%)", title="Greater depth did not fix accuracy")
ax[2].set_yscale("log")
ax[2].axhline(depth["ed_reference_effective_probability_support"], color="#52616B", ls="--", lw=1.4)
ax[2].text(2.35, depth["ed_reference_effective_probability_support"]*1.05, "ED: 641", ha="right", va="bottom", fontsize=10)
ax[2].set(ylabel=r"Effective support $1/\sum_n p_n^2$", title="Learned probability concentration", ylim=(.8,1800))
save(fig, "fqhe_n8_depth_and_support")
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.4), constrained_layout=True)
for i, layer in enumerate((1, 2, 6)):
    group = [r for r in depth["endpoints"] if r["depth"] == layer]
    for panel, field, factor in ((0,"energy_V1",1),(1,"fidelity",100)):
        vals = np.array([r[field] for r in group]) * factor
        ax[panel].bar(i, vals.mean(), color=COLORS[i], alpha=.45, width=.55)
        ax[panel].scatter(np.array([i-.08,i+.08]), vals, color=COLORS[i], s=35, zorder=4)
for a in ax:
    a.set_xticks(range(3), ["1", "2", "6"])
    a.set_xlabel("Hidden layers (width 64)")
ax[0].set(ylabel=r"Exact $(E-E_0)/V_1$", title="Energy after 5,000 updates")
ax[1].set(ylabel="ED fidelity (%)", title="Bars: means; dots: two matched seed pairs")
save(fig, "fqhe_n8_depth_energy_fidelity")

geometry = read("SPRING_Haldane_FQHE/results/ne7_gg_spring_20261111/summary.json")
fig, ax = plt.subplots(1,2,figsize=(10.9,3.45),constrained_layout=True)
for method, color, label in zip(("spring","pool_random","gg_spring"),COLORS,
    ("SPRING: batch 32","Pool 128, random 32","GG-SPRING: pool 128 → 32")):
    trajectory = [r for r in geometry["trajectories"] if r["method"]==method]
    for seed in (71,137,211):
        r=sorted([r for r in trajectory if r["seed"]==seed],key=lambda r:r["step"])
        ax[0].semilogy([x["step"] for x in r],[x["energy_error_V1"] for x in r],color=color,alpha=.23,lw=1)
        ax[1].plot([x["step"] for x in r],[100*x["fidelity"] for x in r],color=color,alpha=.23,lw=1)
    common=sorted({r["step"] for r in trajectory})
    means=lambda field:[np.mean([r[field] for r in trajectory if r["step"]==step]) for step in common]
    ax[0].semilogy(common,means("energy_error_V1"),color=color,lw=2.2,marker="o",ms=3,label=label)
    ax[1].plot(common,np.array(means("fidelity"))*100,color=color,lw=2.2,marker="o",ms=3)
ax[0].set(xlabel="Optimizer updates",ylabel=r"Exact $(E-E_0)/V_1$",title=r"$N_e=7$, pure $V_1$, no wall")
ax[1].set(xlabel="Optimizer updates",ylabel="ED fidelity (%)",title="Three paired sampling trajectories")
ax[0].legend(frameon=False,loc="upper right")
save(fig,"fqhe_n7_geometry_no_wall")

explore = read("SPRING_Haldane_FQHE/results/ne7_gg_explore_20261111/summary.json")
fig, ax = plt.subplots(1,2,figsize=(10.9,3.45),constrained_layout=True)
labels=("Original GG-SPRING","Broader pool, β=0.5","Broader + sequential geometry")
for method,color,label in zip(("gg_original","gg_tempered","gg_sequential"),COLORS,labels):
    trajectory=[r for r in explore["trajectories"] if r["method"]==method]
    for seed in (71,137,211):
        r=sorted([r for r in trajectory if r["seed"]==seed],key=lambda r:r["step"])
        ax[0].plot([x["step"] for x in r],[x["energy_error_V1"] for x in r],color=color,alpha=.24,lw=1)
        ax[1].plot([x["training_seconds"] for x in r],[100*x["fidelity"] for x in r],color=color,alpha=.24,lw=1)
    common=sorted({r["step"] for r in trajectory})
    means=lambda field:[np.mean([r[field] for r in trajectory if r["step"]==step]) for step in common]
    ax[0].plot(common,means("energy_error_V1"),color=color,lw=2.2,marker="o",ms=3,label=label)
    ax[1].plot(means("training_seconds"),np.array(means("fidelity"))*100,color=color,lw=2.2,marker="o",ms=3)
ax[0].set(xlabel="Total optimizer updates",ylabel=r"Exact $(E-E_0)/V_1$",title="Continue the same full 8k states")
ax[1].set(xlabel="Additional active training time (s)",ylabel="ED fidelity (%)",title="Accuracy gains and added cost",ylim=(85,100))
ax[0].legend(frameon=False,loc="upper right",fontsize=9)
save(fig,"fqhe_n7_broader_pool_continuation")
fig, ax = plt.subplots(1, 2, figsize=(10.9, 3.45), constrained_layout=True)
for method,color,label in zip(("gg_original","gg_tempered","gg_sequential"),COLORS,labels):
    trajectory=[r for r in explore["trajectories"] if r["method"]==method]
    for seed in (71,137,211):
        r=sorted([r for r in trajectory if r["seed"]==seed],key=lambda r:r["step"])
        for a,field,factor in ((ax[0],"energy_error_V1",1),(ax[1],"fidelity",100)):
            a.plot([x["step"] for x in r],[factor*x[field] for x in r],color=color,alpha=.24,lw=1)
    common=sorted({r["step"] for r in trajectory})
    means=lambda field:[np.mean([r[field] for r in trajectory if r["step"]==step]) for step in common]
    ax[0].plot(common,means("energy_error_V1"),color=color,lw=2.2,marker="o",ms=3,label=label)
    ax[1].plot(common,np.array(means("fidelity"))*100,color=color,lw=2.2,marker="o",ms=3)
ax[0].set(xlabel="Total optimizer updates",ylabel=r"Exact $(E-E_0)/V_1$",title="Continue the same full 8k states")
ax[1].set(xlabel="Total optimizer updates",ylabel="ED fidelity (%)",title="Thin: seeds; thick: three-seed means",ylim=(85,100))
ax[0].legend(frameon=False,loc="upper right",fontsize=9)
save(fig,"fqhe_n7_broader_pool_updates")

probe = read("SPRING_Haldane_FQHE/results/ne7_gg_explore_20261111/frozen_probe.json")
fig,ax=plt.subplots(1,2,figsize=(10.9,3.45),constrained_layout=True)
seeds=[s["seed"] for s in probe["seeds"]]
x=np.arange(3)
for i,beta in enumerate((1.,.5)):
    y=[100*s["ideal_proposals"][str(beta)]["independent_pool_rare_hit_probability"] for s in probe["seeds"]]
    ax[0].bar(x+(-.18 if i==0 else .18),y,width=.34,color=COLORS[i],label="Born pool" if i==0 else "Broader pool")
ax[0].set_xticks(x,[str(s) for s in seeds]);ax[0].set(xlabel="Saved 8k state / sampling seed",ylabel="Chance of ≥1 rare candidate (%)",title="128 independent proposal candidates",ylim=(0,100));ax[0].legend(frameon=False)
ax[1].bar(x,[100*s["rare_variance_fraction"] for s in probe["seeds"]],color=COLORS[2],alpha=.8)
ax[1].set_xticks(x,[str(s) for s in seeds]);ax[1].set(xlabel="Saved 8k state / sampling seed",ylabel="Share of exact residual variance (%)",title=r"Rare means $p_\theta(n)<10^{-6}$",ylim=(0,105))
for i,s in enumerate(probe["seeds"]):
    ax[1].text(i,7,r"Born mass"+"\n"+f"{s['rare_born_probability']:.1e}",ha="center",color="white",fontsize=10)
save(fig,"fqhe_n7_rare_tail_coverage")

(OUT / "figure_inputs.json").write_text(json.dumps({"input_sha256":INPUTS,
    "no_new_training":True,"no_new_ED":True,
    "scope":"Curves and exact endpoint metrics read from retained JSON/CSV; seed ranges descriptive only; timing active-only."},indent=2)+"\n")
print("Rendered 8 PDF/PNG figure pairs and input SHA256 receipt:",OUT)
