"""Render compact vector figures from audited saved measurements only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spring_constraint_figures_20261116")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

SUPPORT = Path(__file__).resolve().parent
MANIFEST = SUPPORT / "constraint_comparison_20261116.json"
FIGURES = SUPPORT / "figures"
METADATA = SUPPORT / "constraint_figures_metadata_20261116.json"
NAVY, TEAL, ORANGE = "#18334D", "#158371", "#DF9008"
MUTED, PALE = "#536473", "#E9F5F1"
SIZE_CM = (14., 4.3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.,
        "text.color": NAVY, "axes.labelcolor": NAVY,
        "axes.edgecolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.titlesize": 7.2,
        "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
        "axes.linewidth": .55, "grid.color": "#DCE4E9",
        "grid.linewidth": .45, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.facecolor": "white", "legend.frameon": False,
    })


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=.5, length=2.4, pad=2)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem):
    result = {}
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{stem}.{suffix}"
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(path, dpi=300, metadata=metadata)
        result[str(path)] = sha(path)
    plt.close(fig)
    return result


def matched_short(data):
    arms = data["matched_1k"]["arms"]
    fig, axes = plt.subplots(1, 3, figsize=tuple(v / 2.54 for v in SIZE_CM))
    fig.subplots_adjust(left=.061, right=.988, bottom=.21, top=.76, wspace=.36)
    columns = [("energy_error_V1", 1., r"Physical $(E-E_0)/V_1$"),
               ("ground_subspace_fidelity", 100., "Full ED fidelity (%)"),
               ("physical_L0_probability", 100., r"$P(L=0)$ (%)")]
    for name, color in (("kappa_0_seed71", NAVY), ("kappa_0p1_seed71", TEAL)):
        records = arms[name]["records"]
        xs = [row["step"] for row in records]
        for ax, (key, scale, title) in zip(axes, columns, strict=True):
            ax.plot(xs, [scale * row[key] for row in records],
                    color=color, lw=1.25, marker="o", ms=2., mew=0, zorder=3)
            ax.set_title(title, pad=5)
    for ax in axes:
        style(ax)
        ax.set_xlim(-30, 1100)
        ax.set_xticks([0, 500, 1000], ["0", "500", "1,000"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].set_ylim(.90, 1.38)
    axes[0].set_yticks([1., 1.15, 1.30], ["1.00", "1.15", "1.30"])
    axes[1].set_ylim(1., 3.4)
    axes[1].set_yticks([1., 2., 3.])
    axes[2].set_ylim(1.3, 4.55)
    axes[2].set_yticks([2., 3., 4.])
    for ax, key, scale in zip(axes, [c[0] for c in columns], [c[1] for c in columns], strict=True):
        for name, color, offset in (("kappa_0_seed71", NAVY, -9),
                                    ("kappa_0p1_seed71", TEAL, 4)):
            value = arms[name]["last"][key] * scale
            ax.annotate(f"{value:.2f}", (1000, value), xytext=(-3, offset),
                        textcoords="offset points", ha="right", va="bottom",
                        color=color, fontsize=6.5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=.1, alpha=.9))
    handles = [Line2D([], [], color=NAVY, lw=1.5, marker="o", ms=2.7,
                      label=r"No coherent $L^2$: $\kappa=0$"),
               Line2D([], [], color=TEAL, lw=1.5, marker="o", ms=2.7,
                      label=r"Augmented: $\kappa=0.1$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.035),
               ncol=2, fontsize=7., columnspacing=1.7, handlelength=1.8,
               handletextpad=.5)
    fig.supxlabel("Added attempts: same source, batch 32, one seed", y=.015, fontsize=7.)
    return save(fig, "constraint_matched_short")


def long_training(data):
    records = [row for row in data["long_B128_L2"]["records"]
               if row["absolute_step"] >= 19000]
    assert len(records) == 71
    assert records[0]["absolute_step"] == 19000
    assert records[-1]["absolute_step"] == 54000
    xs = np.asarray([row["absolute_step"] for row in records]) / 1000.
    fig, axes = plt.subplots(1, 2, figsize=tuple(v / 2.54 for v in SIZE_CM))
    fig.subplots_adjust(left=.068, right=.985, bottom=.22, top=.73, wspace=.28)
    columns = [("energy_error_V1", 1., NAVY, r"Physical $(E-E_0)/V_1$"),
               ("ground_subspace_fidelity", 100., TEAL, "Full ED fidelity (%)")]
    for ax, (key, scale, color, title) in zip(axes, columns, strict=True):
        ys = np.asarray([row[key] for row in records]) * scale
        ax.axvspan(44., 54.8, color=PALE, zorder=0)
        ax.plot(xs, ys, color=color, lw=1.25, marker="o", ms=1.7, mew=0, zorder=3)
        ax.axvline(43., color=ORANGE, ls=(0, (3, 2)), lw=1.05, zorder=4)
        ax.set_title(title, pad=5)
        style(ax)
        ax.set_xlim(18.4, 55.)
        ax.set_xticks([19., 30., 40., 54.], ["19", "30", "40", "54"])
        ax.annotate(f"{ys[-1]:.3f}" if scale == 1 else f"{ys[-1]:.1f}%",
                    (54., ys[-1]), xytext=(-2, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.7, color=color,
                    bbox=dict(facecolor="white", edgecolor="none", pad=.2, alpha=.9))
    axes[0].set_ylim(.08, .98)
    axes[0].set_yticks([.25, .50, .75], ["0.25", "0.50", "0.75"])
    axes[1].set_ylim(0., 86.)
    axes[1].set_yticks([0., 20., 40., 60., 80.])
    fig.text(.5, .975, r"Fixed batch 128, $\kappa=0.1$; optimizer and sampler state continued",
             ha="center", va="top", fontsize=7.)
    fig.text(.5, .87, "Orange: CG limit 384 → 1024 at 43k; shaded: latest 10k",
             ha="center", va="top", fontsize=6.6, color=MUTED)
    fig.supxlabel("Absolute training iteration (thousands); retained checkpoints", y=.015, fontsize=7.)
    return save(fig, "constraint_long_training")


def main():
    setup()
    data = json.loads(MANIFEST.read_text())
    assert data["status"] == "passed"
    verified, unavailable = {}, []
    for name, digest in data["input_sha256"].items():
        path = Path(name)
        if path.exists():
            assert sha(path) == digest, "Source input changed: " + name
            verified[name] = digest
        else:
            unavailable.append(name)
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = matched_short(data) | long_training(data)
    receipt = dict(
        status="passed" if not unavailable else "passed_with_unavailable_source_artifacts",
        generator=str(Path(__file__).resolve()), generator_sha256=sha(__file__),
        manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST),
        no_training_or_ED_or_wavefunction_evaluation_performed=True,
        figure_size_cm=list(SIZE_CM), png_dpi=300, vector_pdf=True,
        source_inputs_present_verified=verified,
        source_inputs_present_verified_count=len(verified),
        source_inputs_unavailable=unavailable,
        source_inputs_unavailable_count=len(unavailable),
        selections={
            "constraint_matched_short": dict(
                arms=["kappa_0_seed71", "kappa_0p1_seed71"],
                added_attempts=list(range(0, 1001, 100)),
                metrics=["energy_error_V1", "ground_subspace_fidelity", "physical_L0_probability"],
                interpretation="Matched one-seed short test: sector/fidelity improve but physical energy is worse"),
            "constraint_long_training": dict(
                absolute_start=19000, absolute_end=54000, checkpoints=71,
                absolute_steps=list(range(19000, 54001, 500)),
                metrics=["energy_error_V1", "ground_subspace_fidelity"],
                fixed_batch=128, fixed_kappa=.1, candidate_pool=512,
                CG_limit_change_absolute_update=43000, latest_segment_start=44000,
                discarded_work_included=False,
                interpretation="Continued retained trajectory; not an equal-budget no-L2 ablation"),
        },
        file_sha256=outputs,
    )
    METADATA.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(status=receipt["status"], metadata=str(METADATA),
                          verified_inputs=len(verified), unavailable_inputs=len(unavailable),
                          outputs=[str(Path(p).name) for p in outputs]), indent=2))


if __name__ == "__main__":
    main()
