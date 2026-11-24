"""Full-sector Born-weighted MinSR as a Ne4 diagnostic without sampling noise."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter

from fqhe_study import PRODUCTION_PAIRS, PRODUCTION_STEPS, PROJECT, REFERENCE, ROOT, write_json

sys.path.insert(0, str(PROJECT / "shared"))

from fqh_spring import model
from fqh_spring.evaluate import state_diagnostics
from fqh_spring.physics import SphereOperator
from fqh_spring.references import load_reference
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)


def build_physics():
    operator = SphereOperator(ne=4, flux=9, v3=0.0, two_lz=0)
    occupied = [indices for indices in combinations(range(operator.norb), operator.ne)
                if sum(indices) == operator.target_sum]
    assert len(occupied) == operator.dimension() == 18
    occupations = np.zeros((len(occupied), operator.norb), dtype=np.int8)
    for row, indices in zip(occupations, occupied):
        row[list(indices)] = 1
    rows = {tuple(row): index for index, row in enumerate(occupations)}
    matrix = np.zeros((len(occupied), len(occupied)))
    for index, row in enumerate(occupations):
        neighbors, coefficients = operator.connections(row)
        for neighbor, coefficient in zip(neighbors, coefficients):
            matrix[index, rows[tuple(neighbor)]] = coefficient
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-13, rtol=0)
    return jnp.asarray(occupations, dtype=jnp.float64), jnp.asarray(matrix)


def make_kernels(occupations, matrix, *, learning_rate, damping, radius):
    def coefficients(params):
        logs = model.batch_log_psi(params, occupations)
        psi = jnp.exp(logs - logs.real.max())
        return psi / jnp.linalg.norm(psi), jnp.exp(-1j * logs.imag)

    def rayleigh(params):
        psi, _ = coefficients(params)
        return jnp.vdot(psi, matrix @ psi).real

    def weighted_problem(params):
        psi, reverse_phase = coefficients(params)
        magnitude = jnp.abs(psi)
        applied = matrix @ psi
        energy = jnp.vdot(psi, applied).real
        scores = model.score_tree(params, occupations)
        scores = jax.vmap(lambda tree: ravel_pytree(tree)[0])(scores)
        centered = scores - (magnitude**2) @ scores
        rows = magnitude[:, None] * centered
        residual = reverse_phase * (applied - energy * psi)
        return jnp.concatenate((rows.real, rows.imag)), jnp.concatenate((residual.real, residual.imag))

    def step(params, _):
        rows, residual = weighted_problem(params)
        raw = -rows.T @ jnp.linalg.solve(rows @ rows.T + damping * jnp.eye(rows.shape[0]), residual)
        update = jnp.minimum(learning_rate, radius/(jnp.linalg.norm(raw) + 1e-8)) * raw
        _, unravel = ravel_pytree(params)
        return optax.apply_updates(params, unravel(update)), None

    @jax.jit
    def block(params):
        return jax.lax.scan(step, params, xs=None, length=100)[0]

    return block, jax.jit(coefficients), rayleigh, weighted_problem


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((ROOT / "fqhe/frozen_settings.json").read_text())["MinSR"]
    assert settings["mu"] == 0
    occupations, matrix = build_physics()
    block, coefficients, rayleigh, weighted = make_kernels(
        occupations, matrix, learning_rate=settings["learning_rate"],
        damping=settings["damping"], radius=0.03)
    pending = []
    for init_seed, sample_seed in PRODUCTION_PAIRS:
        run_dir = output / f"i{init_seed}"
        run_dir.mkdir()
        params = model.initialize(10, width=64, layers=6, kind="mlp",
                                  seed=init_seed, readout="legacy_additive")
        original = ROOT / f"fqhe/production/MinSR-i{init_seed}-s{sample_seed}/train/initial_parameters.npz"
        if original.exists():
            with np.load(original, allow_pickle=False) as train_initial:
                assert set(train_initial.files) == set(params)
                for name in params:
                    np.testing.assert_array_equal(params[name], train_initial[name])
        A, b = weighted(params)
        exact_gradient, _ = ravel_pytree(jax.grad(rayleigh)(params))
        error = float(jnp.max(jnp.abs(2 * (A.T @ b) - exact_gradient)))
        assert error < 1e-11
        start = perf_counter()
        kernel = block.lower(params).compile()
        compile_seconds = perf_counter() - start
        states = [params]
        timings = [0.0]
        start = perf_counter()
        for _ in range(0, PRODUCTION_STEPS, 100):
            params = kernel(params)
            jax.block_until_ready(params)
            timings.append(perf_counter() - start)
            states.append(params)
        np.savez_compressed(run_dir / "parameters.npz", **params)
        pending.append((run_dir, init_seed, error, compile_seconds, states, timings))
        print(f"Exact-weighted MinSR seed {init_seed} trained", flush=True)

    # Load the independently computed ED basis and eigenvectors after optimization.
    data = load_reference(REFERENCE)
    indexes = {tuple(row): index for index, row in enumerate(np.asarray(occupations))}
    reorder = np.array([indexes[tuple(row)] for row in data["occupations"]])
    np.testing.assert_allclose(np.asarray(matrix)[np.ix_(reorder, reorder)], data["H"].toarray(),
                               atol=1e-12, rtol=0)
    e0, e1 = sorted(data["eigenvalues"])[:2]
    gap = e1 - e0
    for run_dir, seed, error, compile_seconds, states, timings in pending:
        rows = []
        for index, (params, elapsed) in enumerate(zip(states, timings, strict=True)):
            psi = np.asarray(coefficients(params)[0])[reorder]
            exact = state_diagnostics(psi, data, angular=True)
            energy = exact["exact_variational_energy_V1"]
            rows.append(dict(step=100*index, training_seconds=elapsed, energy=energy,
                             gap_ratio=float((energy-e0)/gap),
                             fidelity=exact["ground_state_fidelity"],
                             L0_probability=exact["exact_L0_probability"],
                             variance=exact["exact_residual_variance_V1_squared"]))
        with (run_dir / "curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        hit = next((row["step"] for row in rows if row["gap_ratio"] <= 0.01), None)
        write_json(run_dir / "metrics.json", dict(
            init_seed=seed, settings=settings | {"radius": 0.03},
            full_sector_weighted_autodiff_gradient_max_error=error,
            compilation_seconds=compile_seconds, target_step=hit,
            initial=rows[0], final=rows[-1]))
        print(f"Exact-weighted MinSR seed {seed}: E={rows[-1]['energy']:.8f}, "
              f"F={rows[-1]['fidelity']:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve())
