"""Replay the local FQHE controls on CUDA and check Ne6 feasibility."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from fqhe_study import METHODS, PRODUCTION_PAIRS, PROJECT, ROOT, make_evaluator, write_json


ALLOWED_CPUS = sorted(os.sched_getaffinity(0))
CPU_IDS = (8, 10) if {8, 10} <= set(ALLOWED_CPUS) else tuple(ALLOWED_CPUS[:2])
CPU_MASK = ",".join(map(str, CPU_IDS))


def read_json(path):
    return json.loads(path.read_text())


def gpu_training(job, method, config):
    job.mkdir(parents=True)
    model_dir, _ = METHODS[method]
    write_json(job / "input.json", config | {"cpus": len(CPU_IDS)})
    entry = PROJECT / model_dir / "FQHE/training/run.py"
    command = ["/usr/bin/time", "-f", "%e %M", "-o", str(job / "process_time_rss.txt"),
               "taskset", "-c", CPU_MASK, sys.executable, str(entry),
               "--config", str(job / "input.json"), "--output", str(job / "train")]
    # JAX 0.9.1 expands its generic gpu alias to CUDA and ROCm on this host.
    env = dict(os.environ, JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    with (job / "train.log").open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)
    summary = read_json(job / "train/training_summary.json")
    runtime = read_json(job / "train/run_config.json")
    assert summary["status"] == "completed" and summary["completed_steps"] == config["steps"]
    assert "cuda" in runtime["device"].lower() or "gpu" in runtime["device"].lower(), runtime["device"]
    return runtime["device"], summary


def matched_ne4(output):
    evaluate = make_evaluator()
    rows = []
    for init_seed, sample_seed in PRODUCTION_PAIRS:
        for method in METHODS:
            name = f"{method}-i{init_seed}-s{sample_seed}"
            cpu_path = ROOT / "fqhe/production" / name
            cpu = read_json(cpu_path / "metrics.json")
            job = output / "ne4" / name
            config = cpu["config"] | {"backend": "auto",
                                      "source": str(cpu_path / "train/initial_parameters.npz")}
            device, _ = gpu_training(job, method, config)
            gpu = evaluate(job)
            assert cpu["initial_parameters_sha256"] == gpu["initial_parameters_sha256"]
            np.testing.assert_allclose(cpu["initial"]["energy"], gpu["initial"]["energy"], rtol=0, atol=1e-12)
            checkpoint = "train/checkpoints/step_000000000000.npz"
            with (np.load(cpu_path / checkpoint, allow_pickle=False) as cpu_initial,
                  np.load(job / checkpoint, allow_pickle=False) as gpu_initial):
                np.testing.assert_array_equal(cpu_initial["walkers"], gpu_initial["walkers"])
            row = dict(method=method, init_seed=init_seed, sample_seed=sample_seed, device=device,
                       steps=config["steps"], batch=config["batch"],
                       identical_initial_parameters=True,
                       cpu_training_seconds=cpu["training_seconds"],
                       gpu_training_seconds=gpu["training_seconds"],
                       cpu_process_seconds=cpu["total_process_seconds"],
                       gpu_process_seconds=gpu["total_process_seconds"],
                       gpu_compilation_seconds=gpu["compilation_seconds"],
                       cpu_final_energy=cpu["final"]["energy"], gpu_final_energy=gpu["final"]["energy"],
                       cpu_final_fidelity=cpu["final"]["fidelity"], gpu_final_fidelity=gpu["final"]["fidelity"],
                       gpu_target_step=gpu["target_step"])
            rows.append(row)
            with (output / "ne4_cpu_gpu.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            print(f"{name}: {device}, E={row['gpu_final_energy']:.6f}, "
                  f"F={row['gpu_final_fidelity']:.6f}, training={row['gpu_training_seconds']:.2f}s",
                  flush=True)


def ne6_pilot(output):
    init_seed, sample_seed = PRODUCTION_PAIRS[0]
    name = f"MinSR-i{init_seed}-s{sample_seed}"
    cpu_path = ROOT / "fqhe/production" / name
    config = read_json(cpu_path / "input.json") | {"backend": "auto", "source": None,
                                                   "ne": 6, "flux": 15, "batch": 64}
    job = output / "ne6" / name
    device, summary = gpu_training(job, "MinSR", config)
    evaluation = job / "exact_evaluation.json"
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "shared"), JAX_PLATFORMS="cpu")
    with (job / "evaluation.log").open("w") as stream:
        subprocess.run(["taskset", "-c", CPU_MASK, sys.executable, "-m", "fqh_spring.evaluate",
                        "--checkpoint", str(job / "train/checkpoint_latest.npz"),
                        "--reference", str(PROJECT / "benchmarks/fqhe/references/ne6"),
                        "--angular", "--output", str(evaluation)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)
    exact = read_json(evaluation)
    write_json(output / "ne6_pilot.json", dict(
        device=device, steps=config["steps"], batch=config["batch"],
        seed=sample_seed, init_seed=init_seed,
        training_seconds=summary["training_seconds"],
        compilation_seconds=summary["current_launch_compile_seconds"],
        basis_dimension=exact["basis_dimension"], energy=exact["exact_variational_energy_V1"],
        fidelity=exact["ground_state_fidelity"], L0_probability=exact["exact_L0_probability"]))
    print(f"Ne6 {device}: {config['steps']} steps, E={exact['exact_variational_energy_V1']:.6f}, "
          f"F={exact['ground_state_fidelity']:.6f}, training={summary['training_seconds']:.2f}s",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    matched_ne4(output)
    ne6_pilot(output)
