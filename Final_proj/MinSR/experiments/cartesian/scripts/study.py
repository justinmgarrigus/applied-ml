"""Cartesian MinSR FQHE study: Ne8 main study (GPU) and Ne4 pilot (CPU).

Protocol (mirrors the paired Ising study):
  1. tune: every grid setting on two tuning seed pairs, disjoint from production;
  2. freeze: the setting with the lowest median final exact energy;
  3. produce: five new paired seeds with the frozen setting, exact ED every `every` updates;
  4. control: the same production seeds with the exponential amplitude map
     (Cartesian vs exponential is the only change).
Jobs that already have metrics.json are skipped, so the script can be resumed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path
import statistics
import threading

from harness import SOURCE_COMMIT, Evaluator, file_hash, make_config, read_json, train, write_json

TUNING_PAIRS = ((7, 71), (11, 111))
PRODUCTION_PAIRS = ((17, 171), (29, 291), (43, 431), (59, 591), (83, 831))

SPECS = {
    8: dict(device="gpu", workers=3, cpu_sets=("0,2", "4,6", "8,10"),
            tune_steps=1000, tune_every=250, prod_steps=5000, prod_every=100,
            grid=[dict(batch=128, learning_rate=lr, damping=d, max_norm=r)
                  for lr, d, r in product((0.01, 0.05), (0.001, 0.01, 0.1), (0.03, 0.1))]),
    4: dict(device="cpu", workers=3, cpu_sets=("12,13", "14,15", "16,17"),
            tune_steps=250, tune_every=250, prod_steps=1000, prod_every=100,
            extend_steps=5000, extend_every=250,
            grid=[dict(batch=32, learning_rate=lr, damping=d, max_norm=0.03)
                  for lr, d in product((0.01, 0.04), (0.003, 0.03))]),
}
# Ne4 exponential control reuses the setting frozen in the first (exponential) Ne4 study.
NE4_EXPONENTIAL = dict(batch=32, learning_rate=0.04, damping=0.03, max_norm=0.03)


def run_jobs(jobs, spec, evaluator, lock):
    """jobs: list of (directory, config). Trains in parallel, evaluates serially."""
    slots = list(spec["cpu_sets"])

    def worker(index):
        for job, config in jobs[index::spec["workers"]]:
            if (job / "metrics.json").exists():
                continue
            if job.exists():
                raise RuntimeError(f"Incomplete job directory {job}; remove it to rerun")
            train(job, config, device=spec["device"], cpus=slots[index])
            with lock:
                result = evaluator.run(job)
            print(f"{job.relative_to(job.parents[2])}: E={result['final']['energy']:.4f} "
                  f"F={result['final']['fidelity']:.4f} r={result['final']['gap_ratio']:.3f} "
                  f"train={result['training_seconds']:.0f}s", flush=True)

    with ThreadPoolExecutor(spec["workers"]) as executor:
        list(executor.map(worker, range(spec["workers"])))
    return [read_json(job / "metrics.json") for job, _ in jobs]


def main(ne, output, phases):
    spec = SPECS[ne]
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(source_commit=SOURCE_COMMIT, harness_sha256=file_hash(Path(__file__).with_name("harness.py")),
                    study_sha256=file_hash(__file__), ne=ne, tuning_pairs=TUNING_PAIRS,
                    production_pairs=PRODUCTION_PAIRS, selection="lowest median final exact energy over tuning pairs",
                    target="(E-E0)/gap <= 0.01", **{k: v for k, v in spec.items() if k != "cpu_sets"})
    if not (output / "protocol.json").exists():
        write_json(output / "protocol.json", protocol)
    evaluator, lock = Evaluator(ne), threading.Lock()

    if "tune" in phases:
        jobs = [(output / "tune" / f"grid{g:02d}" / f"i{i}-s{s}",
                 make_config(ne, i, s, spec["tune_steps"], spec["tune_every"], settings, backend=spec["device"]))
                for g, settings in enumerate(spec["grid"]) for i, s in TUNING_PAIRS]
        results = run_jobs(jobs, spec, evaluator, lock)
        table = []
        for g, settings in enumerate(spec["grid"]):
            finals = [r["final"]["energy"] for r in results if r["config"] | settings == r["config"]]
            table.append(dict(grid=g, **settings, median_final_energy=statistics.median(finals),
                              final_energies=finals))
        best = min(table, key=lambda row: row["median_final_energy"])
        write_json(output / "tuning_table.json", table)
        write_json(output / "frozen_settings.json", {k: best[k] for k in spec["grid"][0]})
        print("Frozen:", read_json(output / "frozen_settings.json"), flush=True)

    if "extend" in phases:
        # Same paired seeds and frozen settings, longer budget; runs are deterministic, so the
        # first prod_steps updates reproduce the production runs.
        frozen = read_json(output / "frozen_settings.json")
        jobs = []
        for amplitude_map in ("cartesian", "exponential"):
            settings = NE4_EXPONENTIAL if (ne == 4 and amplitude_map == "exponential") else frozen
            for i, s in PRODUCTION_PAIRS:
                jobs.append((output / f"extended_{amplitude_map}" / f"MinSR-i{i}-s{s}",
                             make_config(ne, i, s, spec["extend_steps"], spec["extend_every"], settings,
                                         amplitude_map=amplitude_map, backend=spec["device"])))
        run_jobs(jobs, spec, evaluator, lock)

    if phases & {"produce", "control"}:
        frozen = read_json(output / "frozen_settings.json")
        maps = (["cartesian"] if "produce" in phases else []) + (["exponential"] if "control" in phases else [])
        jobs = []
        for amplitude_map in maps:
            settings = NE4_EXPONENTIAL if (ne == 4 and amplitude_map == "exponential") else frozen
            for i, s in PRODUCTION_PAIRS:
                jobs.append((output / f"production_{amplitude_map}" / f"MinSR-i{i}-s{s}",
                             make_config(ne, i, s, spec["prod_steps"], spec["prod_every"], settings,
                                         amplitude_map=amplitude_map, backend=spec["device"])))
        run_jobs(jobs, spec, evaluator, lock)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--ne", type=int, choices=sorted(SPECS), required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--phases", nargs="+", default=["tune", "produce", "control"],
                     choices=["tune", "produce", "control", "extend"])
    args = cli.parse_args()
    main(args.ne, args.output.resolve(), set(args.phases))
