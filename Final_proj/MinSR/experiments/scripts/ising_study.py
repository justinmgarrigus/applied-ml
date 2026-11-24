"""Paired Ising optimizer experiments using the repository's JAX backend."""

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source/Final_proj/shared"))

import ising_backend as model
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax

SOURCE_COMMIT = "fb726d230b5f9efe59a5f4b5f994aaff8c9af0a0"
METHODS = ("MinSR", "SPRING", "SGD")
CASES = ((4, 1.0), (6, 0.5), (6, 1.0), (6, 2.0), (8, 1.0), (10, 1.0))
BATCHES = (16, 64)
PILOT_SEEDS = (7, 11)
PRODUCTION_SEEDS = (17, 29, 43, 59, 83)
TARGET = 0.01
PILOT_STEPS = 1000
PRODUCTION_STEPS = 3000
EVALUATE_EVERY = 20


@dataclass(frozen=True)
class Settings:
    learning_rate: float
    damping: float
    radius: float
    mu: float


def candidates(method):
    if method == "SGD":
        return [Settings(lr, 0.0, radius, 0.0)
                for lr, radius in product((0.01, 0.03, 0.06, 0.12), (0.03, 0.1))]
    mu = 0.9 if method == "SPRING" else 0.0
    return [Settings(lr, damping, radius, mu)
            for lr, damping, radius in product((0.015, 0.05), (0.001, 0.01), (0.03, 0.1))]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def optimizer(method, lr, damping, radius, mu):
    if method == "SGD":
        return model.make_sgd(lr, radius)
    return model.make_spring(lr, damping, mu, radius)


@lru_cache
def sampled_kernel(n, batch, method):
    states = model.enumerate_spins(n)

    def one_step(carry, strength, settings):
        params, state, key = carry
        lr, damping, radius, mu = settings
        key, sample_key = jax.random.split(key)
        logits = 2 * model.batch_log_psi(params, states)
        indexes = jax.random.categorical(sample_key, logits, shape=(batch,))
        samples = states[indexes]
        gradient = model.energy_covariance_gradient(params, samples, h=strength)
        transform = optimizer(method, lr, damping, radius, mu)
        if method == "SGD":
            updates, state = transform.update(gradient, state, params)
        else:
            force = jax.tree.map(lambda value: 0.5 * value, gradient)
            updates, state = transform.update(force, state, params, data=samples)
        size = optax.global_norm(updates)
        return (optax.apply_updates(params, updates), state, key), size

    @jax.jit
    def block(carry, strength, settings):
        return jax.lax.scan(lambda state, _: one_step(state, strength, settings),
                            carry, xs=None, length=EVALUATE_EVERY)

    return block, one_step


@lru_cache
def exact_kernel(n):
    states = model.enumerate_spins(n)

    def one_step(carry, strength, settings):
        params, previous, key = carry
        lr, damping, radius, mu = settings
        weights = jax.nn.softmax(2 * model.batch_log_psi(params, states))
        scores = model.score_matrix(params, states)
        energy = model.local_energies(params, states, h=strength)
        centered_scores = scores - weights @ scores
        centered_energy = energy - weights @ energy
        metric = (centered_scores.T * weights) @ centered_scores
        force = centered_scores.T @ (weights * centered_energy)
        raw = jnp.linalg.solve(metric + damping * jnp.eye(metric.shape[0]),
                               -force + damping * mu * previous)
        size = jnp.linalg.norm(raw)
        update = jnp.minimum(lr, radius / (size + 1e-8)) * raw
        _, unravel = ravel_pytree(params)
        return (optax.apply_updates(params, unravel(update)), raw, key), jnp.linalg.norm(update)

    @jax.jit
    def block(carry, strength, settings):
        return jax.lax.scan(lambda state, _: one_step(state, strength, settings),
                            carry, xs=None, length=EVALUATE_EVERY)

    return block


@lru_cache
def reference(n, strength):
    states = model.enumerate_spins(n)
    matrix = model.dense_hamiltonian(n, h=strength)
    values, vectors = np.linalg.eigh(matrix)
    # Spin-flip complementary bit strings form an orthonormal parity-even basis.
    half = 2 ** (n - 1)
    even = (np.eye(2 * half)[:, :half] + np.eye(2 * half)[::-1, :half]) / np.sqrt(2)
    parity_values = np.linalg.eigvalsh(even.T @ matrix @ even)
    if not np.isclose(parity_values[0], values[0], atol=1e-12, rtol=0):
        raise ArithmeticError("The ground state did not match the even sector")
    gap = float(parity_values[1] - parity_values[0])
    matrix, ground = jnp.asarray(matrix), jnp.asarray(vectors[:, 0])

    @jax.jit
    def observe(params):
        result = model.exact_observables(params, states, matrix)
        psi = result["psi"]
        return jnp.asarray([result["energy"], result["variance"],
                            jnp.abs(ground @ psi)**2, jnp.linalg.norm(psi - psi[::-1])])

    return float(values[0]), gap, observe


def seed_state(n, seed, method, mode):
    params = model.init_params(seed, n, 2 * n)
    state = (jnp.zeros_like(ravel_pytree(params)[0]) if mode == "exact"
             else optimizer(method, 0.05, 0.001, 0.1, 0.0).init(params))
    return params, state, jax.random.PRNGKey(seed + 1)


def run(destination, phase, n, strength, batch, method, seed, settings, steps, mode="sampled"):
    destination.mkdir(parents=True)
    assert steps % EVALUATE_EVERY == 0
    carry = seed_state(n, seed, method, mode)
    e0, gap, observe = reference(n, strength)
    flat_settings = jnp.asarray(list(asdict(settings).values()), dtype=jnp.float64)
    field = jnp.asarray(strength, dtype=jnp.float64)
    weights_before = file_hash_value(carry[0]["W"])
    config = dict(phase=phase, n=n, field=strength, hidden=2*n, batch=batch, method=method,
                  seed=seed, sampling_seed=seed + 1 if mode == "sampled" else None,
                  mode=mode, steps=steps, **asdict(settings))
    save_json(destination / "config.json", config)
    block = exact_kernel(n) if mode == "exact" else sampled_kernel(n, batch, method)[0]
    # Report JIT/cache lookup separately, then start from the untouched initial state.
    start = perf_counter()
    run_block = block.lower(carry, field, flat_settings).compile()
    observe_compiled = observe.lower(carry[0]).compile()
    jax.block_until_ready(run_block(carry, field, flat_settings))
    jax.block_until_ready(observe_compiled(carry[0]))
    compile_warmup = perf_counter() - start
    total_train_seconds = 0.0
    total_evaluation_seconds = 0.0
    hit = None
    reached_halfway = None
    capped = 0
    rows = []
    clock = perf_counter()
    cap = settings.radius
    for step in range(0, steps + 1, EVALUATE_EVERY):
        if step:
            start = perf_counter()
            carry, norms = run_block(carry, field, flat_settings)
            applied = np.asarray(norms)
            total_train_seconds += perf_counter() - start
            capped += np.count_nonzero(applied >= cap * (1 - 1e-6))
        start = perf_counter()
        energy, variance, fidelity, parity = map(float, observe_compiled(carry[0]))
        total_evaluation_seconds += perf_counter() - start
        elapsed = perf_counter() - clock
        error = energy - e0
        if (not np.isfinite([energy, variance, fidelity, parity]).all()
                or error < -1e-9 or parity > 1e-12
                or fidelity + 1e-10 < 1 - error / gap):
            raise ArithmeticError(f"Invalid exact diagnostic at {destination}, step {step}")
        row = dict(step=step, wall_seconds=elapsed, training_seconds=total_train_seconds,
                   evaluation_seconds=total_evaluation_seconds, energy=energy,
                   energy_error=error, error_per_spin=error/n, gap_ratio=error/gap,
                   variance=variance, fidelity=fidelity, spin_flip_residual=parity)
        rows.append(row)
        if hit is None and row["gap_ratio"] <= TARGET:
            hit = row
        if step == steps // 2:
            reached_halfway = row
    final = rows[-1]
    with (destination / "curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(destination / "parameters.npz", **carry[0])
    result = dict(**config, ground_energy=e0, even_gap=gap, target=TARGET,
                  target_step=hit["step"] if hit else None,
                  target_seconds=hit["wall_seconds"] if hit else None,
                  stayed_below_target_from_first_hit=(all(row["gap_ratio"] <= TARGET
                      for row in rows if row["step"] >= hit["step"]) if hit else False),
                  first=rows[0], halfway=reached_halfway, final=final,
                  capped_fraction=float(capped / steps),
                  compile_warmup_seconds=compile_warmup, n_steps=steps,
                  process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  initial_weights_sha256=weights_before,
                  final_file_sha256=file_hash(destination / "parameters.npz"))
    save_json(destination / "metrics.json", result)
    return result


def file_hash_value(array):
    return sha256(np.asarray(array).tobytes()).hexdigest()


def validate(output):
    # Freeze repeated rows so the sampled Gram is rank deficient by construction.
    states = model.enumerate_spins(6)[jnp.asarray([0, 3, 7, 13, 13, 21, 25, 33, 37, 55])]
    params = model.init_params(7, 6, 12)
    scores = np.asarray(model.score_matrix(params, states))
    energy = np.asarray(model.local_energies(params, states))
    A = (scores - scores.mean(axis=0)) / np.sqrt(len(energy))
    b = (energy - energy.mean()) / np.sqrt(len(energy))
    force = A.T @ b
    physical_gradient = model.energy_covariance_gradient(params, states)
    results = {}
    for damping in (0.001, 0.01):
        transform = optimizer("MinSR", 0.05, damping, 0.1, 0.0)
        updates, state = transform.update(
            jax.tree.map(lambda value: value / 2, physical_gradient), transform.init(params),
            params, data=states)
        direct = -np.linalg.solve(A.T @ A + damping * np.eye(A.shape[1]), force)
        dual = -A.T @ np.linalg.solve(A @ A.T + damping * np.eye(A.shape[0]), b)
        raw = -np.asarray(state[0].prev_delta)
        expected_update = np.minimum(0.05, 0.1 / (np.linalg.norm(direct) + 1e-8)) * direct
        actual_update = np.asarray(ravel_pytree(updates)[0])
        np.testing.assert_allclose(direct, dual, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(direct, raw, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(expected_update, actual_update, atol=1e-9, rtol=1e-9)
        results[str(damping)] = dict(primal_dual_error=float(max(abs(direct-dual))),
                                     primal_vendor_error=float(max(abs(direct-raw))),
                                     applied_update_error=float(max(abs(expected_update-actual_update))))
    smoke = ROOT / "validation/ising_minsr/training.csv"
    if smoke.exists():
        _, single_step = sampled_kernel(6, 16, "MinSR")
        state = seed_state(6, 7, "MinSR", "sampled")
        e0, _, observe = reference(6, 1.0)
        settings = jnp.asarray([0.05, 0.001, 0.1, 0.0])
        for _ in range(5):
            state, _ = jax.jit(single_step)(state, jnp.asarray(1.0), settings)
        with smoke.open(newline="") as stream:
            last = list(csv.DictReader(stream))[-1]
        error = abs(float(observe(state[0])[0]) - e0 - float(last["energy_error"]))
        assert error < 1e-11
        results["stock_runner_final_energy_error_difference"] = error
    save_json(output / "frozen_batch_validation.json", results)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(source_commit=SOURCE_COMMIT, upstream_commit=model.UPSTREAM_COMMIT,
                    versions=model.package_versions(), platform=platform.platform(),
                    python=platform.python_version(), affinity=sorted(os.sched_getaffinity(0)),
                    source_hash=file_hash(Path(model.__file__)), harness_hash=file_hash(Path(__file__)),
                    cases=CASES, batches=BATCHES, pilot_seeds=PILOT_SEEDS,
                    production_seeds=PRODUCTION_SEEDS, methods=METHODS,
                    grid={method: [asdict(value) for value in candidates(method)] for method in METHODS},
                    target=TARGET, pilot_steps=PILOT_STEPS, production_steps=PRODUCTION_STEPS,
                    evaluate_every=EVALUATE_EVERY, pilot_selection="lowest median final gap ratio",
                    run_order_seed=351, dtype="float64", J=1.0)
    save_json(output / "protocol.json", metadata)
    validate(output)
    selected = {}
    for batch in BATCHES:
        for method in METHODS:
            scores = []
            for idx, settings in enumerate(candidates(method)):
                results = [run(output / "pilot" / f"b{batch}" / method / f"grid{idx}-s{seed}",
                               "pilot", 6, 1.0, batch, method, seed, settings, PILOT_STEPS)
                           for seed in PILOT_SEEDS]
                scores.append(float(np.median([result["final"]["gap_ratio"] for result in results])))
            winner = int(np.argmin(scores))
            selected[batch, method] = candidates(method)[winner]
            print(f"Pilot B={batch} {method}: grid {winner}, median ratio {scores[winner]:.5g}", flush=True)
    save_json(output / "frozen_settings.json",
              {str(batch): {method: asdict(selected[batch, method]) for method in METHODS} for batch in BATCHES})
    shuffle = random.Random(351)
    for n, strength in CASES:
        for batch in BATCHES:
            jobs = list(product(METHODS, PRODUCTION_SEEDS))
            shuffle.shuffle(jobs)
            case = f"n{n}-h{strength:g}-b{batch}"
            for method, seed in jobs:
                run(output / "production" / case / f"{method}-s{seed}", "production", n,
                    strength, batch, method, seed, selected[batch, method], PRODUCTION_STEPS)
            print(f"Production {case}: {len(jobs)} runs complete", flush=True)
    for batch in BATCHES:
        for mu, seed in product((0.0, 0.5, 0.9), PRODUCTION_SEEDS):
            settings = replace(selected[batch, "MinSR"], mu=mu)
            method = "MinSR" if mu == 0 else "SPRING"
            run(output / "memory" / f"b{batch}-mu{mu:g}-s{seed}", "memory", 6, 1.0, batch,
                method, seed, settings, PRODUCTION_STEPS)
        print(f"Memory ablation B={batch} complete", flush=True)
    for mu, seed in product((0.0, 0.5, 0.9), PRODUCTION_SEEDS):
        settings = replace(selected[16, "MinSR"], mu=mu)
        method = "MinSR" if mu == 0 else "SPRING"
        run(output / "exact" / f"mu{mu:g}-s{seed}", "exact", 6, 1.0, None, method, seed,
            settings, PRODUCTION_STEPS, mode="exact")
    print("Exact-weighted control complete", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    main(cli.parse_args().output.resolve())
