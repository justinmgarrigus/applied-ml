"""Render the verified Born-pool GG-SPRING training history for slide 16.

Joins the complete 0--8k wall study to its exact full-state 8k--10k
continuation. No training, diagonalization, or mutation of source results.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spring-born-slide")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[3]
RESULTS = FINAL / "SPRING_Haldane_FQHE/results"
OLD = RESULTS / "ne7_gg_edge_20261111"
NEW = RESULTS / "ne7_gg_explore_20261111"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
INPUTS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    INPUTS[str(path.relative_to(FINAL))] = digest(path)
    return json.loads(path.read_text())


old = read(OLD / "summary.json")
new = read(NEW / "summary.json")
assert old["status"] == new["status"] == "complete"
fixed = ("ne", "flux", "two_lz", "width", "layers", "readout", "batch", "pool",
         "sweeps", "learning_rate", "damping", "mu", "max_norm", "refresh_every",
         "diagonal_floor", "geometry_ridge", "selection_mix", "novelty_floor",
         "confinement_strength", "wall_x", "seed", "init_seed")
fields = ("energy_V1", "energy_error_V1", "fidelity", "variance_V1_squared")
all_rows, joins = [], []
for seed in (71, 137, 211):
    before = sorted([r for r in old["trajectories"] if r["strength_V1"] == 10 and r["seed"] == seed], key=lambda r:r["step"])
    after = sorted([r for r in new["trajectories"] if r["method"] == "gg_original" and r["seed"] == seed], key=lambda r:r["step"])
    assert before[0]["step"] == 0 and before[-1]["step"] == after[0]["step"] == 8000
    assert after[-1]["step"] == 10000
    original_run = OLD / "runs" / f"wall10_seed{seed}"
    continued_run = NEW / "runs" / f"gg_original_seed{seed}"
    original = read(original_run / "run_config.json")
    continued = read(continued_run / "run_config.json")
    for name in fixed:
        assert original["configuration"][name] == continued["configuration"][name], name
    assert continued["configuration"]["beta"] == 1
    parent = original_run / "checkpoints/step_000000008000.npz"
    copied = continued_run / "checkpoints/step_000000008000.npz"
    checksum = digest(parent)
    assert checksum == digest(copied) == before[-1]["checkpoint_sha256"] == after[0]["checkpoint_sha256"]
    assert checksum == continued["parent_checkpoint_sha256"]
    assert continued["parent_config_sha256"] == original["config_sha256"]
    assert continued["restored_optimizer_counters"] == {"spring":8000, "learning_rate":8000}
    for name in ("reference_manifest_sha256", "orbital_profile_sha256", "ground_energy_V1"):
        assert before[-1][name] == after[0][name], name
    for field in fields:
        assert before[-1][field] == after[0][field], field
    old_eval = read(OLD / before[-1]["evaluation_file"])
    new_eval = read(NEW / after[0]["evaluation_file"])
    for name in ("checkpoint_sha256", "reference_manifest_sha256", "coefficient_file_sha256"):
        assert old_eval[name] == new_eval[name], name
    joins.append(dict(seed=seed, parent_checkpoint_sha256=checksum,
        parent_config_sha256=original["config_sha256"],
        reference_manifest_sha256=after[0]["reference_manifest_sha256"],
        source_and_continuation_evaluations_identical=True,
        numeric_join_max_difference=max(abs(before[-1][f]-after[0][f]) for f in fields),
        restored_optimizer_counters=continued["restored_optimizer_counters"]))
    all_rows.extend(before + after[1:])

ground = float(all_rows[0]["ground_energy_V1"])
assert all(r["ground_energy_V1"] == ground for r in all_rows)
steps = sorted({r["step"] for r in all_rows})
means = {field:np.array([np.mean([r[field] for r in all_rows if r["step"] == step]) for step in steps]) for field in fields}
initial = {field:float(means[field][0]) for field in fields}
final = {field:float(means[field][-1]) for field in fields}
ranges = {field:[min(r[field] for r in all_rows if r["step"]==10000),
                 max(r[field] for r in all_rows if r["step"]==10000)] for field in fields}
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":19,
    "axes.labelsize":20, "axes.labelpad":3, "xtick.labelsize":17,
    "ytick.labelsize":17, "axes.spines.top":False, "axes.spines.right":False,
    "pdf.fonttype":42, "axes.linewidth":1.1, "xtick.major.pad":3, "ytick.major.pad":3})
blue, teal = "#234968", "#078774"
fig, axes = plt.subplots(1, 2, figsize=(9, 2.7))
fig.subplots_adjust(left=.10, right=.995, bottom=.29, top=.80, wspace=.38)
for seed in (71, 137, 211):
    rows = sorted([r for r in all_rows if r["seed"]==seed], key=lambda r:r["step"])
    axes[0].plot([r["step"] for r in rows], [r["energy_V1"] for r in rows], color=blue, alpha=.22, lw=1)
    axes[1].plot([r["step"] for r in rows], [100*r["fidelity"] for r in rows], color=blue, alpha=.22, lw=1)
axes[0].plot(steps, means["energy_V1"], color=blue, lw=2.5, marker="o", ms=3.3)
axes[1].plot(steps, 100*means["fidelity"], color=blue, lw=2.5, marker="o", ms=3.3)
axes[0].axhline(ground, color=teal, ls="--", lw=1.8)
axes[1].axhline(100, color=teal, ls="--", lw=1.8)
axes[0].set(ylabel=r"Exact $E/V_1$", ylim=(2.0,15.5), yticks=[3,8,13])
axes[1].set(ylabel="Fidelity (%)", ylim=(-2,105), yticks=[0,50,100])
for ax in axes:
    ax.set(xlabel="Total updates", xlim=(-150,10150), xticks=[0,5000,10000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:"0" if x==0 else f"{x/1000:g}k"))
    ax.grid(alpha=.15, axis="y")
    ax.set_axisbelow(True)
handles = [Line2D([],[],color=blue,lw=2.5,marker="o",ms=3.3),
           Line2D([],[],color=teal,lw=1.8,ls="--")]
fig.legend(handles,["GG-SPRING, Born pool", "ED target"], frameon=False,
    loc="upper center",bbox_to_anchor=(.55,1.02),ncol=2,fontsize=14,handlelength=1.8,columnspacing=1.7)
fig.savefig(OUT / "slide_born_baseline.pdf", bbox_inches="tight", pad_inches=.035)
fig.savefig(OUT / "slide_born_baseline.png", bbox_inches="tight", pad_inches=.035, dpi=200)
plt.close(fig)
evidence = dict(status="passed", source="support/generate_born_baseline_figure.py",
    no_new_training=True, no_new_ED=True, method="GG-SPRING with original Born candidate pool beta=1 and static geometric selection",
    physical_target="N7 F18 Lz0 full19orbital LLL, V1 plus projected U10 wall at x=.88; fixed-sector ED target",
    input_sha256=INPUTS, joins=joins, seeds=[71,137,211],
    total_updates=10000, initial_means=initial, final_means=final, final_ranges=ranges,
    energy_excess_reduction_fraction=1-final["energy_error_V1"]/initial["energy_error_V1"],
    ground_energy_V1=ground, merged_points_per_seed=len(steps),
    figure_convention="Thin lines are three individual sampling trajectories; thick line is their arithmetic mean; no confidence interval or seed selection.",
    caveat="Substantial energy/fidelity progress without broadening; positive remaining energy excess and variance do not establish exact ground-state convergence.")
(OUT / "born_baseline_evidence.json").write_text(json.dumps(evidence, indent=2)+"\n")
print(json.dumps(dict(initial=initial, final=final, ranges=ranges,
                     energy_excess_reduction_fraction=evidence["energy_excess_reduction_fraction"],
                     points_per_seed=len(steps)), indent=2))
