"""Aggregate and plot the Cartesian MinSR FQHE study (Ne4 pilot, Ne8 main)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLOURS = {"cartesian": "#1b6d8a", "exponential": "#bf4d24"}
LABELS = {"cartesian": r"Cartesian  $u=e^{\ell_0}(1+\ell-\ell_0)$", "exponential": r"Exponential  $\psi=e^{\ell}$"}
TARGET = 0.01


def read_json(path):
    return json.loads(path.read_text())


def load(folder):
    records = []
    for path in sorted(folder.glob("*/metrics.json")):
        record = read_json(path)
        record["name"] = path.parent.name
        record["curve"] = np.genfromtxt(path.parent / "curve.csv", delimiter=",", names=True)
        records.append(record)
    return records


def med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def summary_rows(ne, maps):
    rows = []
    for amplitude_map, records in maps.items():
        if not records:
            continue
        rows.append(dict(
            ne=ne, amplitude_map=amplitude_map, runs=len(records),
            steps=records[0]["config"]["steps"], batch=records[0]["config"]["batch"],
            learning_rate=records[0]["config"]["learning_rate"], damping=records[0]["config"]["damping"],
            max_norm=records[0]["config"]["max_norm"],
            target_hits=sum(r["target_step"] is not None for r in records),
            first_hit_step_median=med([r["target_step"] for r in records]),
            final_energy_median=med([r["final"]["energy"] for r in records]),
            final_energy_min=min(r["final"]["energy"] for r in records),
            final_energy_max=max(r["final"]["energy"] for r in records),
            final_error_per_electron_median=med([r["final"]["error_per_electron"] for r in records]),
            final_gap_ratio_median=med([r["final"]["gap_ratio"] for r in records]),
            final_fidelity_median=med([r["final"]["fidelity"] for r in records]),
            final_fidelity_min=min(r["final"]["fidelity"] for r in records),
            final_fidelity_max=max(r["final"]["fidelity"] for r in records),
            final_L0_probability_median=med([r["final_L0_probability"] for r in records]),
            final_effective_support_median=med([r["final"]["effective_support"] for r in records]),
            final_variance_median=med([r["final"]["variance"] for r in records]),
            training_seconds_median=med([r["training_seconds"] for r in records]),
            compile_seconds_median=med([r["compilation_seconds"] for r in records]),
            peak_rss_mib_median=med([r["peak_rss_mib"] for r in records]),
            device=records[0]["device"], e0=records[0]["e0"], gap=records[0]["gap"]))
    return rows


def plot(ne, maps, output, dimension, tag):
    figure, all_axes = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True)
    time_axis, axes = all_axes[1], [all_axes[0], all_axes[2], all_axes[3]]
    panels = (("gap_ratio", r"$(E-E_0)/\Delta$", True), ("fidelity", "Ground-state fidelity", False),
              ("effective_support", "Effective support  $1/\\sum|\\psi|^4$", True))
    for amplitude_map, records in maps.items():
        if not records:
            continue
        x = records[0]["curve"]["step"]
        for axis, (column, label, log) in zip(axes, panels):
            y = np.array([r["curve"][column] for r in records])
            axis.plot(x, np.median(y, axis=0), color=COLOURS[amplitude_map], label=LABELS[amplitude_map])
            axis.fill_between(x, y.min(axis=0), y.max(axis=0), color=COLOURS[amplitude_map], alpha=0.15)
            axis.set_ylabel(label)
            axis.set_xlabel("MinSR updates")
            if log:
                axis.set_yscale("log")
    for amplitude_map, records in maps.items():
        if not records:
            continue
        # Same updates, plotted against the median training time (excludes compilation and evaluation).
        seconds = np.median([r["curve"]["training_seconds"] for r in records], axis=0)
        y = np.array([r["curve"]["gap_ratio"] for r in records])
        time_axis.plot(seconds, np.median(y, axis=0), color=COLOURS[amplitude_map])
        time_axis.fill_between(seconds, y.min(axis=0), y.max(axis=0), color=COLOURS[amplitude_map], alpha=0.15)
    time_axis.set_yscale("log")
    time_axis.set_xlabel(f"Training seconds ({records[0]['device']})")
    time_axis.set_ylabel(r"$(E-E_0)/\Delta$")
    for axis in (axes[0], time_axis):
        axis.axhline(TARGET, color="#9a9a9a", linestyle=":", linewidth=1)
    axes[2].axhline(dimension, color="#9a9a9a", linestyle="--", linewidth=1)
    axes[2].text(0.02, 0.93, f"sector dimension {dimension}", transform=axes[2].transAxes, fontsize=8, color="#666")
    axes[1].legend(fontsize=8, loc="best")
    figure.suptitle(f"MinSR on FQHE Ne{ne} (pure $V_1$, $L_z=0$): median and range over "
                    f"{len(next(iter(v for v in maps.values() if v)))} paired seeds")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{tag}_minsr_cartesian_vs_exponential.{extension}", dpi=180)
    plt.close(figure)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ne, dimension, study in ((4, 18, "production"), (4, 18, "extended"), (8, 8512, "production")):
        root = HERE / "runs" / f"ne{ne}"
        maps = {m: load(root / f"{study}_{m}") for m in ("cartesian", "exponential")}
        if not any(maps.values()):
            continue
        tag = f"ne{ne}" if study == "production" else f"ne{ne}_{study}"
        all_rows += [dict(study=study, **row) for row in summary_rows(ne, maps)]
        plot(ne, maps, output, dimension, tag)
        with (output / f"{tag}_runs.csv").open("w", newline="") as stream:
            writer = None
            for m, records in maps.items():
                for r in records:
                    row = dict(ne=ne, amplitude_map=m, name=r["name"], init_seed=r["config"]["init_seed"],
                               seed=r["config"]["seed"], target_step=r["target_step"],
                               final_energy=r["final"]["energy"], final_gap_ratio=r["final"]["gap_ratio"],
                               final_fidelity=r["final"]["fidelity"], final_L0_probability=r["final_L0_probability"],
                               final_effective_support=r["final"]["effective_support"],
                               best_energy=r["best_energy"], training_seconds=r["training_seconds"],
                               device=r["device"])
                    writer = writer or csv.DictWriter(stream, fieldnames=row.keys(), lineterminator="\n")
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
        tuning = root / "tuning_table.json"
        if tuning.exists():
            table = read_json(tuning)
            with (output / f"ne{ne}_tuning.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[k for k in table[0] if k != "final_energies"],
                                        lineterminator="\n", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(table)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    for row in all_rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=HERE / "analysis")
    main(cli.parse_args().output.resolve())
