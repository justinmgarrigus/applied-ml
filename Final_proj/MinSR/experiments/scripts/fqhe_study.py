"""Run and independently evaluate the portable Ne4 FQHE optimizers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "source/Final_proj"
METHODS = {"MinSR": ("MinSR", "minsr"), "SPRING": ("SPRING", "spring"),
           "SGD": ("VMC", "vmc_sgd")}
PILOT_PAIR = (7, 71)
PRODUCTION_PAIRS = ((17, 171), (29, 291), (43, 431))
PILOT_STEPS, PRODUCTION_STEPS = 250, 1000
TARGET, RADIUS = 0.01, 0.03
REFERENCE = PROJECT / "benchmarks/fqhe/references/ne4"
CPU_GROUPS = ((8, 10), (12, 13), (14, 15), (16, 17))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def candidates(method):
    mu = 0.9 if method == "SPRING" else 0.0
    if method == "SGD":
        return [dict(learning_rate=eta, damping=0.01, mu=mu)
                for eta in (0.005, 0.02, 0.05, 0.10)]
    return [dict(learning_rate=eta, damping=damping, mu=mu)
            for eta, damping in product((0.01, 0.04), (0.003, 0.03))]


def train(job_dir, method, parameters, pair, phase, cpus):
    job_dir.mkdir(parents=True, exist_ok=False)
    model_dir, method_id = METHODS[method]
    entry = PROJECT / model_dir / "FQHE/training/run.py"
    steps = PILOT_STEPS if phase == "pilot" else PRODUCTION_STEPS
    every = steps if phase == "pilot" else 100
    config = dict(method=method_id, ne=4, flux=9, two_lz=0, v3=0.0,
                  width=64, layers=6, variant="mlp", readout="legacy_additive",
                  source=None, init_seed=pair[0], seed=pair[1],
                  steps=steps, batch=32, sweeps=2, burn_sweeps=50, refresh_every=25,
                  max_norm=RADIUS, pair_chunk=6, checkpoint_every=every,
                  log_every=every, keep_checkpoints=steps // every + 1,
                  native_threads=1, cpus=len(cpus), backend="cpu", **parameters)
    write_json(job_dir / "input.json", config)
    affinity = ",".join(map(str, cpus))
    command = ["/usr/bin/time", "-f", "%e %M", "-o", str(job_dir / "process_time_rss.txt"),
               "taskset", "-c", affinity, sys.executable, str(entry),
               "--config", str(job_dir / "input.json"), "--output", str(job_dir / "train")]
    with (job_dir / "train.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"FQHE runner exited {result.returncode}. See {job_dir / 'train.log'}")
    assert read_json(job_dir / "train/training_summary.json")["completed_steps"] == steps


def make_evaluator():
    sys.path.insert(0, str(PROJECT / "shared"))
    import jax
    import jax.numpy as jnp
    from fqh_spring import model
    from fqh_spring.evaluate import read_parameters, state_diagnostics
    from fqh_spring.references import load_reference, sha256 as file_hash

    jax.config.update("jax_enable_x64", True)
    data = load_reference(REFERENCE)
    values = np.sort(data["eigenvalues"])
    e0, gap = float(values[0]), float(values[1] - values[0])
    occupations = jnp.asarray(data["occupations"])
    forward = jax.jit(model.batch_log_psi)

    def evaluate(job_dir):
        job_dir = Path(job_dir)
        trained = job_dir / "train"
        config = read_json(job_dir / "input.json")
        assert all(config[key] == data["metadata"][key] for key in ("ne", "flux", "two_lz", "v3"))
        training_summary = read_json(trained / "training_summary.json")
        with (trained / "metrics.csv").open(newline="") as stream:
            mc_rows = {int(row["step"]): row for row in csv.DictReader(stream)}
        checkpoints = sorted((trained / "checkpoints").glob("step_*.npz"))
        assert [int(path.stem.split("_")[1]) for path in checkpoints] == [0, *mc_rows]
        initial = read_parameters(checkpoints[0])
        digest = sha256()
        for name, array in sorted(initial.items()):
            digest.update(name.encode())
            digest.update(array.tobytes())
        jax.block_until_ready(forward(initial, occupations))
        rows = []
        accumulated_eval_seconds = 0.0
        for checkpoint in checkpoints:
            started = perf_counter()
            checkpoint_hash = file_hash(checkpoint)
            parameters = read_parameters(checkpoint)
            logs = np.asarray(forward(parameters, occupations))
            psi = np.exp(logs - logs.real.max())
            diagnostics = state_diagnostics(psi, data, angular=True)
            assert file_hash(checkpoint) == checkpoint_hash
            accumulated_eval_seconds += perf_counter() - started
            step = int(checkpoint.stem.split("_")[1])
            training_row = mc_rows.get(step)
            seconds = float(training_row["training_seconds"]) if training_row else 0.0
            energy = diagnostics["exact_variational_energy_V1"]
            error = energy - e0
            ratio = error / gap
            assert error >= -1e-9 and diagnostics["ground_state_fidelity"] + 1e-10 >= 1 - ratio
            rows.append(dict(step=step, training_seconds=seconds,
                             evaluation_seconds=accumulated_eval_seconds,
                             posthoc_combined_seconds=seconds + accumulated_eval_seconds,
                             energy=energy, energy_error=error, error_per_electron=error/config["ne"],
                             gap_ratio=ratio, variance=diagnostics["exact_residual_variance_V1_squared"],
                             fidelity=diagnostics["ground_state_fidelity"],
                             L0_probability=diagnostics["exact_L0_probability"],
                             local_acceptance=float(training_row["acceptance"]) if training_row else None,
                             distinct_configurations=float(training_row["distinct_configurations"]) if training_row else None,
                             checkpoint_sha256=checkpoint_hash))
        assert all(file_hash(REFERENCE / name) == expected
                   for name, expected in data["manifest"].items())
        write_json(job_dir / "endpoint_diagnostics.json", diagnostics)
        with (job_dir / "curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        first_hit = next((row for row in rows if row["gap_ratio"] <= TARGET), None)
        elapsed, rss = map(float, (job_dir / "process_time_rss.txt").read_text().split())
        result = dict(config=config, e0=e0, gap=gap, target_step=first_hit["step"] if first_hit else None,
                      target_posthoc_seconds=first_hit["posthoc_combined_seconds"] if first_hit else None,
                      training_seconds=training_summary["training_seconds"],
                      compilation_seconds=training_summary["current_launch_compile_seconds"],
                      total_process_seconds=elapsed, peak_rss_mib=rss/1024,
                      initial_parameters_sha256=digest.hexdigest(),
                      reference_manifest_sha256=file_hash(REFERENCE / "manifest.json"),
                      initial=rows[0], final=rows[-1])
        write_json(job_dir / "metrics.json", result)
        return result

    return evaluate


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", dict(
        source_commit="fb726d230b5f9efe59a5f4b5f994aaff8c9af0a0",
        harness_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        pilot_pair=PILOT_PAIR, production_pairs=PRODUCTION_PAIRS,
        production_cpu_affinity=CPU_GROUPS[0], pilot_steps=PILOT_STEPS,
        production_steps=PRODUCTION_STEPS, target=TARGET,
        pilot_selection="least exact endpoint energy on the disjoint pilot pair",
        search={name: candidates(name) for name in METHODS}, run_order_seed=901,
        evaluation="frozen full-sector ED plus angular-momentum diagnostics every 100 updates"))
    jobs = [(output / "pilot" / method / f"grid{index}", method, parameters)
            for method in METHODS for index, parameters in enumerate(candidates(method))]
    # Each pilot worker owns a distinct CPU pair. Pilot runtimes are excluded from comparisons.
    def run_group(index):
        for directory, method, settings in jobs[index::len(CPU_GROUPS)]:
            train(directory, method, settings, PILOT_PAIR, "pilot", CPU_GROUPS[index])
            print(f"Pilot {method} {directory.name} trained", flush=True)

    with ThreadPoolExecutor(len(CPU_GROUPS)) as executor:
        list(executor.map(run_group, range(len(CPU_GROUPS))))
    os.sched_setaffinity(0, CPU_GROUPS[0])
    evaluate = make_evaluator()
    scores = {directory: evaluate(directory)["final"]["energy"] for directory, _, _ in jobs}
    selected = {}
    for method in METHODS:
        directory, _, settings = min((job for job in jobs if job[1] == method),
                                     key=lambda job: scores[job[0]])
        selected[method] = settings
        print(f"Selected {method} {directory.name}: E={scores[directory]:.6f}", flush=True)
    write_json(output / "frozen_settings.json", selected)
    order = list(product(METHODS, PRODUCTION_PAIRS))
    random.Random(901).shuffle(order)
    for method, pair in order:
        directory = output / "production" / f"{method}-i{pair[0]}-s{pair[1]}"
        train(directory, method, selected[method], pair, "production", CPU_GROUPS[0])
        result = evaluate(directory)
        print(f"Production {directory.name}: E={result['final']['energy']:.6f}, "
              f"F={result['final']['fidelity']:.6f}", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    main(cli.parse_args().output.resolve())
