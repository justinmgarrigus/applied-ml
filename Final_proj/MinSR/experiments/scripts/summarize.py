"""Verify, aggregate, and plot the completed local experiments."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
COLOURS = {"MinSR": "#1b6d8a", "SPRING": "#bf4d24", "SGD": "#646979"}
TARGET = 0.01


def read_json(path):
    return json.loads(path.read_text())


def keyed(items, key):
    grouped = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return grouped


def median(numbers):
    return float(np.median(numbers)) if len(numbers) else None


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path, data):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def load_ising(folder, expected_count, expected_steps):
    records = []
    for path in sorted(folder.rglob("metrics.json")):
        record = read_json(path)
        run_dir = path.parent
        record["path"] = str(run_dir.relative_to(ROOT))
        record["group"] = run_dir.parent.name
        curve = np.genfromtxt(run_dir / "curve.csv", delimiter=",", names=True)
        assert record["n_steps"] == expected_steps == curve["step"][-1]
        assert np.allclose(np.diff(curve["step"]), 20)
        assert np.all(np.diff(curve["wall_seconds"]) > 0)
        assert np.max(curve["spin_flip_residual"]) < 1e-12
        assert np.isclose(curve["gap_ratio"][-1], record["final"]["gap_ratio"], rtol=1e-12, atol=1e-14)
        assert digest(run_dir / "parameters.npz") == record["final_file_sha256"]
        hit = np.flatnonzero(curve["gap_ratio"] <= TARGET)
        assert record["target_step"] == (curve["step"][hit[0]] if len(hit) else None)
        record["curve"] = curve
        records.append(record)
    assert len(records) == expected_count, (folder, len(records), expected_count)
    return records


def target_counts(items):
    return dict(
        runs=len(items), first_hit_count=sum(item["target_step"] is not None for item in items),
        final_success_count=sum(item["final"]["gap_ratio"] <= TARGET for item in items),
        first_hit_step_median=median([item["target_step"] for item in items
                                     if item["target_step"] is not None]))


def ising_outcomes(items):
    ratios = [item["final"]["gap_ratio"] for item in items]
    return dict(**target_counts(items),
                final_gap_ratio_median=median(ratios), final_gap_ratio_min=min(ratios),
                final_gap_ratio_max=max(ratios),
                final_error_median_J=median([item["final"]["energy_error"] for item in items]),
                final_error_per_spin_median_J=median([item["final"]["error_per_spin"] for item in items]),
                final_variance_median_J2=median([item["final"]["variance"] for item in items]),
                final_fidelity_median=median([item["final"]["fidelity"] for item in items]),
                final_success_at_1e_4=sum(value <= 0.0001 for value in ratios),
                step_cap_fraction_median=median([item["capped_fraction"] for item in items]))


def last_at(record, column, cutoff):
    curve = record["curve"]
    return float(curve["gap_ratio"][np.searchsorted(curve[column], cutoff, side="right") - 1])


def summarize_ising(output):
    protocol = read_json(ROOT / "ising/protocol.json")
    assert protocol["harness_hash"] == digest(ROOT / "ising_study.py")
    assert protocol["source_hash"] == digest(ROOT / "source/Final_proj/shared/ising_backend.py")
    assert not set(protocol["pilot_seeds"]) & set(protocol["production_seeds"])
    pilot = load_ising(ROOT / "ising/pilot", 96, 1000)
    production = load_ising(ROOT / "ising/production", 180, 3000)
    sampled = load_ising(ROOT / "ising/memory", 30, 3000)
    exact = load_ising(ROOT / "ising/exact", 15, 3000)
    controls = load_ising(ROOT / "controls/ablation", 40, 3000)
    paired = keyed(production, lambda item: (item["n"], item["field"], item["batch"], item["seed"]))
    selected = read_json(ROOT / "ising/frozen_settings.json")
    for (_, _, batch, _), items in paired.items():
        assert {item["method"] for item in items} == set(COLOURS)
        assert len({item["initial_weights_sha256"] for item in items}) == 1
        for item in items:
            assert all(item[key] == value for key, value in selected[str(batch)][item["method"]].items())
    paired_rows = []
    for (n, field, batch, seed), items in sorted(paired.items()):
        methods = {item["method"]: item for item in items}
        minsr = methods["MinSR"]
        for method in ("SPRING", "SGD"):
            other = methods[method]
            paired_rows.append(dict(n=n, field=field, batch=batch, seed=seed, versus=method,
                                    minsr_minus_comparator_final_gap_ratio=(
                                        minsr["final"]["gap_ratio"] - other["final"]["gap_ratio"]),
                                    minsr_first_hit=minsr["target_step"], comparator_first_hit=other["target_step"]))
    write_csv(output / "ising_paired_differences.csv", paired_rows)

    rows = []
    for (n, field, batch), items in sorted(keyed(production, lambda x: (x["n"], x["field"], x["batch"])).items()):
        common_wall = min(item["final"]["wall_seconds"] for item in items)
        for method, group in sorted(keyed(items, lambda x: x["method"]).items()):
            rows.append(dict(n=n, field=field, batch=batch, method=method, **ising_outcomes(group),
                             first_hit_seconds_median=median([x["target_seconds"] for x in group if x["target_seconds"] is not None]),
                             wall_seconds_median=median([x["final"]["wall_seconds"] for x in group]),
                             training_seconds_median=median([x["final"]["training_seconds"] for x in group]),
                             evaluation_seconds_median=median([x["final"]["evaluation_seconds"] for x in group]),
                             compile_warmup_seconds_median=median([x["compile_warmup_seconds"] for x in group]),
                             common_wall_seconds=common_wall,
                             common_wall_gap_ratio_median=median([last_at(x, "wall_seconds", common_wall) for x in group]),
                             matched_draws=32000,
                             matched_draw_gap_ratio_median=median([last_at(x, "step", 32000/batch) for x in group])))
    write_csv(output / "ising_results.csv", rows)
    write_csv(output / "ising_all_runs.csv", [
        {key: item[key] for key in ("phase", "path", "n", "field", "batch", "method", "seed",
                                   "learning_rate", "damping", "radius", "mu", "n_steps", "target_step", "target_seconds")}
        | {f"final_{key}": item["final"][key]
           for key in ("energy", "energy_error", "error_per_spin", "gap_ratio", "variance", "fidelity", "wall_seconds")}
        for item in pilot + production + sampled + exact + controls])

    memory = []
    groups = keyed(sampled + exact, lambda x: (x["batch"] or 0, x["mu"]))
    for (batch, mu), group in sorted(groups.items()):
        memory.append(dict(batch=batch or None, weighted_exact=batch == 0, mu=mu, **ising_outcomes(group)))
    write_csv(output / "ising_memory_ablation.csv", memory)
    ablation = []
    baseline = [item | {"phase": "baseline"} for item in sampled if item["mu"] == 0]
    for (batch, change), group in sorted(keyed(baseline + controls, lambda x: (x["batch"], x["phase"])).items()):
        item = group[0]
        ablation.append(dict(batch=batch, change=change, learning_rate=item["learning_rate"],
                             damping=item["damping"], radius=item["radius"], **ising_outcomes(group)))
    write_csv(output / "ising_damping_radius.csv", ablation)

    profiles = load_ising(ROOT / "controls/profiles", 12, 3000)
    measurements = []
    for item in profiles:
        cost, rss = (ROOT / item["path"]).parent.joinpath("process_time_rss.txt").read_text().split()
        baseline_run = next(result for result in paired[item["n"], 1.0, item["batch"], item["seed"]]
                            if result["method"] == item["method"])
        assert np.isclose(item["final"]["gap_ratio"], baseline_run["final"]["gap_ratio"], atol=1e-12, rtol=1e-8)
        measurements.append(dict(n=item["n"], batch=item["batch"], method=item["method"], seed=item["seed"],
                                 whole_process_seconds=float(cost), peak_rss_mib=float(rss)/1024,
                                 training_plus_evaluation_seconds=item["final"]["wall_seconds"]))
    write_csv(output / "ising_memory_profiles.csv", measurements)
    return production, groups


def summarize_fqhe(output):
    protocol = read_json(ROOT / "fqhe/protocol.json")
    assert protocol["harness_sha256"] == digest(ROOT / "fqhe_study.py")
    pilot = sorted((ROOT / "fqhe/pilot").rglob("metrics.json"))
    records = []
    for path in sorted((ROOT / "fqhe/production").rglob("metrics.json")):
        record = read_json(path)
        config = record["config"]
        directory = path.parent
        method = directory.name.split("-")[0]
        frozen = read_json(ROOT / "fqhe/frozen_settings.json")
        assert all(config[key] == value for key, value in frozen[method].items())
        summary = read_json(directory / "train/training_summary.json")
        assert summary["completed_steps"] == summary["target_steps"] == 1000
        assert digest(directory / "train/checkpoint_latest.npz") == record["final"]["checkpoint_sha256"]
        with (directory / "curve.csv").open(newline="") as stream:
            curve = list(csv.DictReader(stream))
        assert [int(row["step"]) for row in curve] == list(range(0, 1001, 100))
        assert len(list((directory / "train/checkpoints").glob("*.npz"))) == len(curve)
        assert all(digest(directory / "train/checkpoints" / f"step_{int(row['step']):012d}.npz") ==
                   row["checkpoint_sha256"] for row in curve)
        record.update(method=method, path=str(directory.relative_to(ROOT)), curve=curve)
        records.append(record)
    assert len(records) == 9 and len(pilot) == 12
    for seed_pair, runs in keyed(records, lambda x: (x["config"]["init_seed"], x["config"]["seed"])).items():
        assert seed_pair != tuple(protocol["pilot_pair"])
        assert {x["method"] for x in runs} == set(COLOURS)
        assert len({x["initial_parameters_sha256"] for x in runs}) == 1
        shared = {"ne", "flux", "two_lz", "v3", "width", "layers", "variant", "readout",
                  "source", "batch", "sweeps", "burn_sweeps", "refresh_every", "max_norm", "pair_chunk", "steps"}
        assert all(all(x["config"][key] == runs[0]["config"][key] for key in shared) for x in runs)
    write_csv(output / "fqhe_runs.csv", [
        dict(method=record["method"], path=record["path"],
             **{key: record["config"][key] for key in ("init_seed", "seed", "learning_rate", "damping", "max_norm", "mu")},
             target_step=record["target_step"], **record["final"],
             training_total_seconds=record["training_seconds"], compilation_seconds=record["compilation_seconds"],
             whole_process_seconds=record["total_process_seconds"], peak_rss_mib=record["peak_rss_mib"])
        for record in records])
    columns = ("energy", "energy_error", "error_per_electron", "gap_ratio", "variance",
               "fidelity", "L0_probability", "local_acceptance", "distinct_configurations")
    rows = []
    for method, items in sorted(keyed(records, lambda x: x["method"]).items()):
        rows.append(dict(method=method, **target_counts(items),
                         **{f"final_{key}_median": median([x["final"][key] for x in items]) for key in columns},
                         **{f"{key}_median": median([x[key] for x in items])
                            for key in ("training_seconds", "compilation_seconds", "total_process_seconds", "peak_rss_mib")}))
    write_csv(output / "fqhe_results.csv", rows)
    return records


def save_figure(figure, output, name):
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def plot_ising(output, production, groups):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for n, row in zip((6, 10), axes):
        for axis, column in zip(row, ("step", "wall_seconds")):
            runs = [item for item in production if item["n"] == n and item["field"] == 1.0]
            for (method, batch), items in sorted(keyed(runs, lambda x: (x["method"], x["batch"])).items()):
                first = max(item["curve"][column][1] for item in items)
                end = min(item["curve"][column][-1] for item in items)
                x = np.geomspace(first, end, 180)
                y = [median([last_at(item, column, point) for item in items]) for point in x]
                axis.loglog(x, np.maximum(y, 1e-13), linestyle="-" if batch == 16 else "--",
                            color=COLOURS[method])
            axis.axhline(TARGET, color="#9a9a9a", linestyle=":", linewidth=1)
            axis.set(title=f"{n} spins, h/J = 1",
                     xlabel="Optimizer updates" if column == "step" else "Training and evaluation (s)",
                     ylabel=r"$(E-E_0)/\Delta_{\mathrm{even}}$")
            axis.grid(alpha=0.2)
    figure.legend(handles=[Line2D([0], [0], color=c, label=m) for m, c in COLOURS.items()]
                  + [Line2D([0], [0], color="black", label=f"B = {b}", linestyle=style)
                     for b, style in ((16, "-"), (64, "--"))], loc="outside upper center", ncols=5)
    save_figure(figure, output, "ising_learning_curves")

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout="constrained")
    for axis, batch in zip(axes, (16, 64, 0)):
        for mu, colour in zip((0.0, 0.5, 0.9), ("#1b6d8a", "#986d20", "#bf4d24")):
            items = groups[batch, mu]
            x = items[0]["curve"]["step"]
            y = np.median([item["curve"]["gap_ratio"] for item in items], axis=0)
            axis.loglog(x[1:], np.maximum(y[1:], 1e-13), color=colour, label=fr"$\mu={mu:g}$")
        axis.axhline(TARGET, color="#9a9a9a", linestyle=":", linewidth=1)
        axis.set(title=f"Sampled, B = {batch}" if batch else "Exact Born weights",
                 xlabel="Optimizer updates")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$(E-E_0)/\Delta_{\mathrm{even}}$")
    axes[-1].legend()
    save_figure(figure, output, "ising_memory_ablation")


def plot_fqhe(output, records):
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for axis, (metric, label) in zip(axes, (("gap_ratio", r"$(E-E_0)/\Delta_{L_z=0}$"),
                                            ("fidelity", "Ground-state fidelity"), ("L0_probability", r"$P(L=0)$"))):
        for method, items in sorted(keyed(records, lambda x: x["method"]).items()):
            x = [int(row["step"]) for row in items[0]["curve"]]
            y = np.array([[float(row[metric]) for row in item["curve"]] for item in items])
            axis.plot(x, np.median(y, axis=0), label=method, color=COLOURS[method])
            axis.fill_between(x, y.min(axis=0), y.max(axis=0), color=COLOURS[method], alpha=0.14)
        if metric == "gap_ratio":
            axis.set_yscale("log")
            axis.axhline(TARGET, color="#9a9a9a", linestyle=":", linewidth=1)
        else:
            axis.set_ylim(0, 1.04)
        axis.set(xlabel="Optimizer updates", ylabel=label)
        axis.grid(alpha=0.2)
    axes[-1].legend()
    save_figure(figure, output, "fqhe_ne4_learning_curves")


def summarize_fqhe_exact(output, records):
    sampled = {x["config"]["init_seed"]: x for x in records if x["method"] == "MinSR"}
    files = sorted((ROOT / "fqhe_exact").rglob("metrics.json"))
    assert len(files) == len(sampled) == 3
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for path, colour in zip(files, ("#1b6d8a", "#bf4d24", "#646979")):
        exact = read_json(path)
        seed = exact["init_seed"]
        comparison = sampled[seed]
        assert np.isclose(exact["initial"]["energy"], comparison["initial"]["energy"], atol=1e-12, rtol=0)
        assert np.isclose(exact["initial"]["fidelity"], comparison["initial"]["fidelity"], atol=1e-12, rtol=0)
        assert exact["full_sector_weighted_autodiff_gradient_max_error"] < 1e-11
        with path.with_name("curve.csv").open(newline="") as stream:
            curve = list(csv.DictReader(stream))
        assert [int(row["step"]) for row in curve] == list(range(0, 1001, 100))
        for axis, metric in zip(axes, ("gap_ratio", "fidelity")):
            for points, style in ((comparison["curve"], "--"), (curve, "-")):
                axis.plot([int(point["step"]) for point in points],
                          [float(point[metric]) for point in points],
                          color=colour, linestyle=style)
        rows.append(dict(init_seed=seed, sampled_final_energy_V1=comparison["final"]["energy"],
                         exact_final_energy_V1=exact["final"]["energy"],
                         sampled_final_fidelity=comparison["final"]["fidelity"],
                         exact_final_fidelity=exact["final"]["fidelity"],
                         exact_final_gap_ratio=exact["final"]["gap_ratio"],
                         exact_final_L0_probability=exact["final"]["L0_probability"],
                         exact_target_step=exact["target_step"],
                         exact_gradient_error=exact["full_sector_weighted_autodiff_gradient_max_error"]))
    axes[0].set(ylabel=r"$(E-E_0)/\Delta_{L_z=0}$", yscale="log")
    axes[0].axhline(TARGET, color="#9a9a9a", linestyle=":", linewidth=1)
    axes[1].set(ylabel="Ground-state fidelity", ylim=(0, 1.04))
    for axis in axes:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.2)
    handles = [Line2D([0], [0], color=colour, label=f"Seed {row['init_seed']}")
               for row, colour in zip(rows, ("#1b6d8a", "#bf4d24", "#646979"))]
    handles += [Line2D([0], [0], color="black", linestyle=style, label=label)
                for style, label in (("-", "Exact Born weights"), ("--", "MCMC"))]
    figure.legend(handles=handles, loc="outside upper center", ncols=5)
    save_figure(figure, output, "fqhe_minsr_sampling_control")
    write_csv(output / "fqhe_minsr_sampling_control.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ising-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_ising(args.output, *summarize_ising(args.output))
    if not args.ising_only:
        fqhe = summarize_fqhe(args.output)
        plot_fqhe(args.output, fqhe)
        summarize_fqhe_exact(args.output, fqhe)
