"""Test actual upstream SPRING against independent parameter-space solves."""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
import pytest

from ising_backend import (energy_covariance_gradient, enumerate_spins, init_params,
                          make_sgd, make_spring)


def analytic_score(params, spins):
    """Derivative of sum(logcosh(s W)), computed with NumPy, without autodiff."""
    spins = np.asarray(spins)
    hidden = np.tanh(spins @ np.asarray(params["W"]))
    weights = (spins[:, :, None] * hidden[:, None, :]).reshape(len(spins), -1)
    # JAX sorts dict leaves: W followed by c.
    return np.column_stack((weights, np.ones(len(spins))))


@pytest.mark.parametrize("mu", [0.0, 0.9])
def test_upstream_spring_matches_dense_parameter_space_solve(mu):
    params = init_params(23, 4, 3)
    states = enumerate_spins(4)
    eta, damping = 0.037, 0.013
    optimizer = make_spring(eta, damping, mu, max_norm=None)
    state = optimizer.init(params)
    # A nonzero previous direction tests memory in the underdetermined system.
    previous = np.linspace(-0.21, 0.17, ravel_pytree(params)[0].size)
    state = (state[0]._replace(prev_delta=jnp.asarray(previous)), *state[1:])
    for indices in ([0, 1, 3, 7, 12, 12, 15], [2, 4, 5, 6, 9, 11, 13]):
        samples = states[jnp.asarray(indices)]
        force = jax.tree.map(lambda g: 0.5 * g, energy_covariance_gradient(params, samples))
        force_flat = np.asarray(ravel_pytree(force)[0])
        score = analytic_score(params, samples)
        centered = score - score.mean(axis=0)
        covariance = centered.T @ centered / len(samples)
        expected = np.linalg.solve(covariance + damping * np.eye(len(previous)),
                                   force_flat + damping * mu * previous)
        updates, state = optimizer.update(force, state, params, data=samples)
        np.testing.assert_allclose(state[0].prev_delta, expected, atol=2e-11, rtol=2e-10)
        np.testing.assert_allclose(ravel_pytree(updates)[0], -eta * expected,
                                   atol=2e-12, rtol=2e-10)
        params = optax.apply_updates(params, updates)
        previous = expected


def test_norm_clipping_preserves_raw_spring_memory():
    params = init_params(13, 4, 3)
    samples = enumerate_spins(4)[jnp.asarray([0, 1, 2, 4, 5, 11, 14, 15])]
    eta, cap = 2.0, 0.01
    optimizer = make_spring(eta, 0.001, 0.9, max_norm=cap)
    force = jax.tree.map(lambda g: 0.5 * g, energy_covariance_gradient(params, samples))
    updates, state = optimizer.update(force, optimizer.init(params), params, data=samples)
    raw = np.asarray(state[0].prev_delta)
    actual = np.asarray(ravel_pytree(updates)[0])
    assert eta * np.linalg.norm(raw) > 10 * cap  # clipping is actually active
    expected = -min(eta, cap / (np.linalg.norm(raw) + 1e-8)) * raw
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    assert np.linalg.norm(actual) <= cap
    assert not np.allclose(raw, -actual)


def test_sgd_caps_the_applied_full_gradient_update():
    params = init_params(19, 4, 3)
    samples = enumerate_spins(4)[jnp.asarray([0, 1, 2, 4, 5, 11, 14, 15])]
    gradient = energy_covariance_gradient(params, samples)
    full_gradient = np.asarray(ravel_pytree(gradient)[0])
    eta, cap = 2.0, 0.003
    assert eta * np.linalg.norm(full_gradient) > 10 * cap
    optimizer = make_sgd(eta, cap)
    updates, _ = optimizer.update(gradient, optimizer.init(params), params)
    actual = np.asarray(ravel_pytree(updates)[0])
    # Independent radial projection of the physical-gradient SGD step.
    expected = -full_gradient * min(eta, cap / np.linalg.norm(full_gradient))
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
    np.testing.assert_allclose(np.linalg.norm(actual), cap, atol=2e-14)

    uncapped = make_sgd(eta, 100.0)
    updates, _ = uncapped.update(gradient, uncapped.init(params), params)
    np.testing.assert_allclose(ravel_pytree(updates)[0], -eta * full_gradient,
                               atol=2e-14, rtol=2e-12)


@pytest.mark.parametrize("name", ["adam", "rmsprop"])
def test_adaptive_optimizers_use_physical_gradient_and_cap(name):
    from ising_backend import FIRST_ORDER
    params = init_params(5, 4, 3)
    samples = enumerate_spins(4)[jnp.asarray([0, 1, 2, 4, 5, 11, 14, 15])]
    gradient = energy_covariance_gradient(params, samples)
    cap = 0.01
    optimizer = FIRST_ORDER[name](0.5, cap)
    updates, _ = optimizer.update(gradient, optimizer.init(params), params)
    flat_update = np.asarray(ravel_pytree(updates)[0])
    flat_gradient = np.asarray(ravel_pytree(gradient)[0])
    assert np.linalg.norm(flat_update) <= cap * (1 + 1e-9)
    # Descent direction: the first adaptive step has the sign of -gradient.
    nonzero = np.abs(flat_gradient) > 1e-12
    assert np.all(np.sign(flat_update[nonzero]) == -np.sign(flat_gradient[nonzero]))
