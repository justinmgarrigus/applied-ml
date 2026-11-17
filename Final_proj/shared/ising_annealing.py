"""Gradient-free weight-space simulated annealing for the shared Ising RBM.

Acceptance uses the exact full-basis Rayleigh energy, never ED labels. This
small-system baseline deliberately removes Monte Carlo loss noise. ED runs
only after optimization, for independent endpoint diagnostics. No exact resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

import ising_backend as backend
import jax
import jax.numpy as jnp
import numpy as np


def accept(delta: float, temperature: float, rng) -> bool:
    """Metropolis acceptance for a symmetric weight proposal; log form is stable."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    if not np.isfinite(delta):
        return False
    if delta <= 0:
        return True
    draw = rng.random()
    return draw == 0 or np.log(draw) < -delta / temperature


def anneal(initial, objective, *, steps, temperature, final_temperature,
           proposal_scale, final_proposal_scale, seed):
    """One objective call per proposal; keep current and best states separately.

    proposal_scale is the expected RMS Euclidean proposal norm, not the standard
    deviation of each weight. Proposals are symmetric, isotropic Gaussians.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    for value in (temperature, final_temperature, proposal_scale, final_proposal_scale):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Temperatures and proposal scales must be positive and finite")
    if final_temperature > temperature:
        raise ValueError("final_temperature must not exceed temperature")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    current = np.array(initial, dtype=np.float64, copy=True)
    if current.size == 0 or not np.isfinite(current).all():
        raise ValueError("Initial weights must be finite and nonempty")
    energy = float(objective(current))
    if not np.isfinite(energy):
        raise ValueError("Initial energy must be finite")
    initial_energy = best_energy = energy
    best = current.copy()
    rng = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    for index in range(steps):
        fraction = index / max(steps - 1, 1)
        temp = float(np.exp((1 - fraction) * np.log(temperature) + fraction * np.log(final_temperature)))
        scale = float(np.exp((1 - fraction) * np.log(proposal_scale) + fraction * np.log(final_proposal_scale)))
        perturbation = scale / np.sqrt(current.size) * rng.standard_normal(current.shape)
        proposal = current + perturbation
        candidate = float(objective(proposal))
        delta = candidate - energy
        accepted = accept(delta, temp, rng)
        if accepted:
            current, energy = proposal, candidate
            if energy < best_energy:
                best, best_energy = current.copy(), energy
        history.append(dict(step=index + 1, temperature=temp, proposal_scale=scale,
                            proposal_norm=float(np.linalg.norm(perturbation)),
                            candidate_energy=candidate if np.isfinite(candidate) else None,
                            delta_energy=delta if np.isfinite(delta) else None,
                            accepted=int(accepted), current_energy=energy, best_energy=best_energy,
                            objective_evaluations=index + 2,
                            elapsed_seconds=time.perf_counter() - started))
    return dict(current=current, best=best, initial_energy=initial_energy,
                current_energy=energy, best_energy=best_energy, history=history)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--n", type=int, default=6)
    cli.add_argument("--hidden", type=int, default=12)
    cli.add_argument("--J", type=float, default=1.)
    cli.add_argument("--h", type=float, default=1.)
    cli.add_argument("--steps", type=int, default=1000)
    cli.add_argument("--temperature", type=float, default=.1)
    cli.add_argument("--final-temperature", type=float, default=.0001)
    cli.add_argument("--proposal-scale", type=float, default=.05)
    cli.add_argument("--final-proposal-scale", type=float, default=.005)
    cli.add_argument("--seed", type=int, default=7, help="Shared RBM initialization seed")
    cli.add_argument("--search-seed", type=int, default=8, help="Independent proposal/acceptance RNG seed")
    cli.add_argument("--output", type=Path)
    return cli


def run(args, results_root):
    if not 3 <= args.n <= 10 or args.hidden < 1:
        raise ValueError("Require 3 <= n <= 10 and hidden >= 1")
    if not np.isfinite(args.J) or not np.isfinite(args.h) or args.h <= 0 or args.seed < 0:
        raise ValueError("Require finite J, finite positive h, and nonnegative initialization seed")
    # Validate search settings before creating files or compiling the objective.
    if args.steps < 1 or args.search_seed < 0:
        raise ValueError("Require positive steps and nonnegative search seed")
    values = (args.temperature, args.final_temperature, args.proposal_scale, args.final_proposal_scale)
    if any(not np.isfinite(v) or v <= 0 for v in values) or args.final_temperature > args.temperature:
        raise ValueError("Require positive finite scales/temperatures and a nonincreasing temperature")
    output = (args.output or Path(results_root) / "runs" / f"n{args.n}_{uuid.uuid4().hex[:12]}").resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    states = backend.enumerate_spins(args.n)
    matrix = backend.dense_hamiltonian(args.n, args.J, args.h)
    initial = backend.init_params(args.seed, args.n, args.hidden)
    H = jnp.asarray(matrix)

    @jax.jit
    def objective(weights):
        return backend.exact_energy({"W": weights, "c": initial["c"]}, states, H)

    # Warm compilation separately; include this real evaluation in the budget.
    compilation_start = time.perf_counter()
    objective(initial["W"]).block_until_ready()
    compilation_seconds = time.perf_counter() - compilation_start
    started = time.perf_counter()
    result = anneal(np.asarray(initial["W"]), objective, steps=args.steps,
                    temperature=args.temperature, final_temperature=args.final_temperature,
                    proposal_scale=args.proposal_scale, final_proposal_scale=args.final_proposal_scale,
                    seed=args.search_seed)
    training_seconds = time.perf_counter() - started

    # Diagonalization and target overlap are strictly post-training diagnostics.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    def diagnostics(weights):
        obs = backend.exact_observables({"W": jnp.asarray(weights), "c": initial["c"]}, states, H)
        return dict(energy=float(obs["energy"]), energy_error=float(obs["energy"] - eigenvalues[0]),
                    variance=float(obs["variance"]),
                    fidelity=float(abs(eigenvectors[:, 0] @ np.asarray(obs["psi"])) ** 2))

    output.mkdir(parents=True, exist_ok=False)
    for name, weights in (("initial_parameters", initial["W"]),
                          ("trained_parameters", result["current"]),
                          ("best_parameters", result["best"])):
        np.savez(output / f"{name}.npz", W=np.asarray(weights), c=np.asarray(initial["c"]))
    with (output / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["history"][0]))
        writer.writeheader()
        writer.writerows(result["history"])
    configuration = vars(args) | {"output": str(output)}
    metrics = dict(configuration=configuration, optimizer="simulated_annealing",
                   objective="exact full-basis Rayleigh energy; no sampled loss",
                   proposal="symmetric Gaussian on W; per-weight std=proposal_scale/sqrt(W.size)",
                   schedule="geometric endpoints, first attempt uses initial temperature and scale",
                   normalization_parameter="c fixed at shared initialization",
                   parameter_count=args.n * args.hidden + 1, searched_parameters=args.n * args.hidden,
                   exact_ground_energy=float(eigenvalues[0]),
                   initial=diagnostics(initial["W"]), final=diagnostics(result["current"]),
                   best=diagnostics(result["best"]),
                   accepted_proposals=sum(row["accepted"] for row in result["history"]),
                   acceptance_rate=float(np.mean([row["accepted"] for row in result["history"]])),
                   search_objective_evaluations=args.steps + 1,
                   compilation_objective_evaluations=1, endpoint_diagnostic_evaluations=3,
                   training_seconds=training_seconds, compilation_seconds=compilation_seconds,
                   no_gradient_used=True, no_ed_labels_in_training=True,
                   exact_resume_supported=False,
                   versions={name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy")},
                   code_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), Path(backend.__file__))},
                   parameter_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                     for name in ("initial_parameters.npz", "trained_parameters.npz", "best_parameters.npz")})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "initial": metrics["initial"],
                      "final": metrics["final"], "best": metrics["best"],
                      "acceptance_rate": metrics["acceptance_rate"]}, indent=2))
    return metrics


def run_algorithm(results_root, argv=None):
    return run(parser().parse_args(argv), results_root)
