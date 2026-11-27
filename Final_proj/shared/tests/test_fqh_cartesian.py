"""Check Cartesian fits, optimizer conventions, sampled rows, and checkpoints."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from fqh_spring import model
from fqh_spring.cartesian import make_amplitude, make_fit, make_proposal
from fqh_spring.evaluate import evaluate, read_parameters
from fqh_spring.local_energy import make_hamiltonian_action
from fqh_spring.optimizer import make_optimizer
from fqh_spring.physics import SphereOperator
from fqh_spring.references import build_reference, sparse_operators


jax.config.update("jax_enable_x64", True)


def occupations_from_bits(bits, norb):
    occupations = (bits[:, None] >> np.arange(norb)) & 1
    return jnp.asarray(occupations, dtype=jnp.float64)


def test_raw_hamiltonian_rows_agree_with_exact_action_at_a_node():
    operator = SphereOperator(4, v3=0.2)
    bits, hamiltonian, raising, _ = sparse_operators(operator)
    _, first_channel, _, _ = sparse_operators(SphereOperator(4))
    occupations = occupations_from_bits(bits, operator.norb)

    rng = np.random.default_rng(59)
    amplitudes = rng.normal(size=len(bits)) + 1j * rng.normal(size=len(bits))
    amplitudes[0] = 0.0
    by_mask = np.zeros(1 << operator.norb, dtype=np.complex128)
    by_mask[bits] = amplitudes
    powers = 1 << jnp.arange(operator.norb)

    def amplitude(table, occupation):
        return table[occupation.astype(powers.dtype) @ powers]

    expected = {
        "amplitude": amplitudes,
        "energy": hamiltonian @ amplitudes,
        "C1": first_channel @ amplitudes,
        "C3": ((hamiltonian - first_channel) @ amplitudes) / operator.v3,
        "L2": raising.T @ (raising @ amplitudes),
    }
    assert abs(expected["energy"][0]) > 0.1
    raw_action = make_hamiltonian_action(operator, amplitude, pair_chunk_size=4)
    actual = raw_action(jnp.asarray(by_mask), occupations)
    for channel, values in expected.items():
        np.testing.assert_allclose(actual[channel], values, atol=2e-12, rtol=2e-12)


def test_born_fit_matches_exact_metric_and_optimizer_steps():
    operator = SphereOperator(4)
    bits, hamiltonian, _, _ = sparse_operators(operator)
    occupations = occupations_from_bits(bits, operator.norb)
    anchor = {
        "W": jnp.zeros((operator.norb, 1)).at[0, 0].set(np.arctanh(0.5) / 2),
        "b": jnp.asarray([np.arctanh(0.5) / 2]),
        "final_out": jnp.zeros((1, 2)),
        "bias": jnp.zeros(2),
    }
    params = dict(anchor, final_out=jnp.asarray([[0.0, 2.0 * np.sqrt(2.0)]]))
    amplitude = make_amplitude(anchor)
    batch_amplitude = jax.vmap(amplitude, in_axes=(None, 0))
    np.testing.assert_allclose(batch_amplitude(anchor, occupations),
                               jnp.exp(model.batch_log_psi(anchor, occupations)), atol=1e-14)

    u = 1.0 + 1j * np.sqrt(2.0) * np.asarray(occupations[:, 0])
    np.testing.assert_allclose(batch_amplitude(params, occupations), u, atol=2e-15)
    acted = hamiltonian @ u

    proposal = jax.vmap(make_proposal(amplitude), in_axes=(None, 0))
    # These integer counts realize the Born distribution over every coordinate.
    counts = np.where(np.asarray(occupations[:, 0]) == 0, 1, 3)
    np.testing.assert_allclose(jnp.exp(2 * proposal(params, occupations)), counts, atol=2e-15)
    sample = np.repeat(np.arange(len(u)), counts)
    fit = make_fit(anchor)
    force, rows, diagnostics = fit(params, occupations[sample], jnp.asarray(u[sample]),
                                   jnp.asarray(acted[sample]))

    flat, unravel = ravel_pytree(params)
    anchor_output = model.batch_log_psi(anchor, occupations)

    def exact_amplitudes(vector):
        output = model.batch_log_psi(unravel(vector), occupations)
        return jnp.exp(anchor_output) * (1 + output - anchor_output)

    derivative = np.asarray(jax.jacfwd(exact_amplitudes)(flat))
    norm = np.linalg.norm(u)
    psi = u / norm
    projector = np.eye(len(u)) - np.outer(psi, psi.conj())
    jacobian = projector @ derivative / norm
    target = projector @ acted / norm
    expected_gram = (jacobian.conj().T @ jacobian).real
    expected_force = (jacobian.conj().T @ target).real
    optimizer_rows = np.asarray(rows)
    np.testing.assert_allclose(optimizer_rows.mean(axis=0), 0.0, atol=2e-13)
    sampled_gram = (optimizer_rows.conj().T @ optimizer_rows).real / len(sample)
    np.testing.assert_allclose(sampled_gram, expected_gram, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(ravel_pytree(force)[0], expected_force, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(diagnostics["energy_estimate"], np.vdot(psi, hamiltonian @ psi).real,
                               atol=2e-12)
    np.testing.assert_allclose(diagnostics["local_energy_variance"], np.vdot(target, target).real,
                               atol=2e-12)

    damping, rate = 0.1, 0.02
    expected_direction = np.linalg.solve(expected_gram + damping * np.eye(len(flat)), expected_force)
    for method in ("spring", "minsr"):
        optimizer = make_optimizer(method, learning_rate=rate, damping=damping, max_norm=10.0)
        updates, state = optimizer.update(force, optimizer.init(params), params, data=rows)
        np.testing.assert_allclose(state[0].prev_delta, expected_direction, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(ravel_pytree(updates)[0], -rate * expected_direction, atol=2e-12)
        if method == "minsr":
            repeated, _ = optimizer.update(force, state, params, data=rows)
            np.testing.assert_allclose(ravel_pytree(repeated)[0], ravel_pytree(updates)[0], atol=2e-12)

    gradient_fit = make_fit(anchor, metric_rows=False)
    gradient_force, unused_rows, gradient_diagnostics = gradient_fit(
        params, occupations[sample], jnp.asarray(u[sample]), jnp.asarray(acted[sample]),
    )
    assert unused_rows is None
    np.testing.assert_allclose(ravel_pytree(gradient_force)[0], expected_force, atol=2e-12, rtol=2e-12)
    for key in diagnostics:
        np.testing.assert_allclose(gradient_diagnostics[key], diagnostics[key], atol=2e-12)

    def exact_energy(vector):
        amplitudes = exact_amplitudes(vector)
        return (jnp.vdot(amplitudes, jnp.asarray(hamiltonian.toarray()) @ amplitudes)
                / jnp.vdot(amplitudes, amplitudes)).real

    gradient = np.asarray(jax.grad(exact_energy)(flat))
    np.testing.assert_allclose(gradient, 2 * expected_force, atol=2e-12, rtol=2e-12)
    physical_gradient = jax.tree.map(lambda value: 2 * value, gradient_force)
    radius = 0.001
    optimizer = make_optimizer("vmc_sgd", learning_rate=rate, max_norm=radius)
    updates, _ = optimizer.update(physical_gradient, optimizer.init(params), params)
    expected_update = -gradient * min(rate, radius / np.linalg.norm(gradient))
    np.testing.assert_allclose(ravel_pytree(updates)[0], expected_update, atol=2e-12, rtol=2e-12)
    assert float(jnp.linalg.norm(ravel_pytree(updates)[0])) <= radius + 1e-12


def test_born_fit_discards_zero_probability_rows():
    operator = SphereOperator(4)
    bits, hamiltonian, _, _ = sparse_operators(operator)
    occupations = occupations_from_bits(bits, operator.norb)
    anchor = {
        "W": jnp.zeros((operator.norb, 1)).at[0, 0].set(np.arctanh(0.5) / 2),
        "b": jnp.asarray([np.arctanh(0.5) / 2]),
        "final_out": jnp.zeros((1, 2)),
        "bias": jnp.zeros(2),
    }
    params = dict(anchor, final_out=jnp.asarray([[2.0, 2.0]]), bias=jnp.asarray([-1.0, 0.0]))
    amplitude = make_amplitude(anchor)
    u = np.asarray(occupations[:, 0]) * (1.0 + 1.0j)
    acted = hamiltonian @ u
    valid = np.flatnonzero(u != 0)
    node = np.flatnonzero(u == 0)[0]
    proposal = make_proposal(amplitude)
    assert np.isneginf(float(proposal(params, occupations[node])))
    fit = make_fit(anchor)

    def solve(indices):
        return fit(params, occupations[indices], jnp.asarray(u[indices]), jnp.asarray(acted[indices]))

    force, rows, diagnostics = solve(valid)
    padded = np.append(valid, node)
    padded_force, padded_rows, padded_diagnostics = solve(padded)
    np.testing.assert_allclose(ravel_pytree(padded_force)[0], ravel_pytree(force)[0], atol=2e-12)
    gram = (np.asarray(rows).conj().T @ np.asarray(rows)).real / len(valid)
    padded_gram = (np.asarray(padded_rows).conj().T @ np.asarray(padded_rows)).real / len(padded)
    np.testing.assert_allclose(padded_gram, gram, atol=2e-12)
    for key in diagnostics:
        np.testing.assert_allclose(padded_diagnostics[key], diagnostics[key], atol=2e-12)
    assert padded_diagnostics["valid_born_rows"] == len(valid)


@pytest.mark.parametrize("algorithm,method", (("SPRING", "spring"), ("MinSR", "minsr"), ("VMC", "vmc_sgd")))
def test_cartesian_cli_resume_and_numbered_checkpoint_evaluation(tmp_path, algorithm, method):
    root = Path(__file__).resolve().parents[2]
    entry = root / algorithm / "FQHE/training/run.py"
    config = {
        "method": method, "ne": 4, "width": 4, "layers": 2, "amplitude_map": "cartesian",
        "batch": 4, "sweeps": 1, "burn_sweeps": 2, "refresh_every": 1,
        "checkpoint_every": 1, "log_every": 1, "keep_checkpoints": 4, "pair_chunk": 2,
        "backend": "cpu", "cpus": 2,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")

    def launch(output, steps, resume=False, success=True):
        command = [sys.executable, str(entry), "--config", str(path), "--output", str(output),
                   "--steps", str(steps)]
        if resume:
            command.append("--resume")
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        assert (result.returncode == 0) == success, result.stdout + "\n" + result.stderr
        return result

    uninterrupted, split = tmp_path / "uninterrupted", tmp_path / "split"
    launch(uninterrupted, 3)
    launch(split, 1)
    launch(split, 3, resume=True)
    latest = split / "checkpoint_latest.npz"
    with np.load(uninterrupted / latest.name, allow_pickle=False) as left, \
            np.load(latest, allow_pickle=False) as right:
        assert set(left.files) == set(right.files)
        for name in left.files:
            if name != "metadata":
                np.testing.assert_allclose(left[name], right[name], atol=1e-12, rtol=1e-12, err_msg=name)
        a, b = (json.loads(archive["metadata"].item()) for archive in (left, right))
        assert a["step"] == b["step"] == 3
        assert a["rng_state"] == b["rng_state"]
        assert a["config_sha256"] == b["config_sha256"]

    with (split / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == [1, 2, 3]
    assert all(0 < float(row["valid_born_rows"]) <= config["batch"] for row in rows)
    run_config = split / "run_config.json"
    provenance = run_config.read_bytes()
    metadata = json.loads(provenance)
    assert metadata["configuration"]["method"] == method
    assert metadata["configuration"]["amplitude_map"] == "cartesian"
    assert "sampling_floor" not in metadata["configuration"]
    anchor_path = split / "initial_parameters.npz"
    assert metadata["frozen_anchor_sha256"] == hashlib.sha256(anchor_path.read_bytes()).hexdigest()

    reference_path = tmp_path / "reference"
    reference = build_reference(4, reference_path, eigenpairs=4)
    anchor = read_parameters(anchor_path)
    occupations = jnp.asarray(reference["occupations"])
    anchor_output = model.batch_log_psi(anchor, occupations)
    numbered = split / "checkpoints/step_000000000001.npz"
    for index, checkpoint in enumerate((latest, numbered)):
        parameters = read_parameters(checkpoint)
        output = model.batch_log_psi(parameters, occupations)
        u = np.asarray(jnp.exp(anchor_output) * (1 + output - anchor_output))
        psi = u / np.linalg.norm(u)
        acted = reference["H"] @ psi
        energy = np.vdot(psi, acted).real
        variance = np.linalg.norm(acted - energy * psi)**2
        fidelity = abs(np.vdot(reference["eigenvectors"][:, 0], psi))**2
        result = evaluate(checkpoint, reference_path, tmp_path / f"evaluation_{index}.json")
        fields = ("exact_variational_energy_V1", "exact_residual_variance_V1_squared",
                  "ground_state_fidelity")
        np.testing.assert_allclose([result[field] for field in fields], [energy, variance, fidelity],
                                   atol=1e-11, rtol=1e-11)
        assert result["amplitude_map"] == "cartesian"
        assert result["frozen_anchor_sha256"] == metadata["frozen_anchor_sha256"]

    config["damping"] = 0.02
    path.write_text(json.dumps(config))
    failed = launch(split, 4, resume=True, success=False)
    assert "settings changed" in failed.stderr
    assert run_config.read_bytes() == provenance

    # A changed anchor must never be used to reinterpret a saved Cartesian checkpoint.
    original = read_parameters(anchor_path)
    np.savez(anchor_path, **dict(original, bias=original["bias"] + 0.1))
    with pytest.raises(ValueError, match="anchor"):
        evaluate(numbered, reference_path, tmp_path / "invalid_evaluation.json")


@pytest.mark.parametrize("algorithm,method", (("SPRING", "spring"), ("MinSR", "minsr"), ("VMC", "vmc_sgd")))
def test_default_cli_uses_cartesian(tmp_path, algorithm, method):
    entry = Path(__file__).resolve().parents[2] / algorithm / "FQHE/training/run.py"
    output = tmp_path / "default"
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(entry), "--steps", "1", "--output", str(output)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    metadata = json.loads((output / "run_config.json").read_text())
    config = metadata["configuration"]
    assert config["method"] == method
    assert config["ne"] == 4
    assert config["amplitude_map"] == "cartesian"
    assert "sampling_floor" not in config
    resumed = subprocess.run(
        [sys.executable, str(entry), "--resume", "--steps", "2", "--output", str(output)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert resumed.returncode == 0, resumed.stdout + "\n" + resumed.stderr
    summary = json.loads((output / "training_summary.json").read_text())
    assert summary["completed_steps"] == 2
