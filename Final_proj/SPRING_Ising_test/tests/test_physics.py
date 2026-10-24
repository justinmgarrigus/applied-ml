"""Independent physical and derivative checks for the small TFIM example.

The references use the dense Rayleigh quotient rather than differentiating the
VMC covariance formula.  These are deterministic checks over the complete spin
basis, so discrepancies cannot be explained by Monte Carlo uncertainty.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from ising_spring import (
    batch_log_psi,
    dense_hamiltonian,
    energy_covariance_gradient,
    enumerate_spins,
    exact_energy,
    exact_observables,
    init_params,
    local_energies,
)


@pytest.fixture
def small_system():
    """Use a nonstationary, nonsingular state with a normalization parameter."""
    n, hidden = 4, 3
    params = dict(init_params(seed=17, n=n, hidden=hidden))
    params["W"] = jnp.asarray(
        [
            [0.21, -0.09, 0.13],
            [-0.17, 0.24, 0.08],
            [0.11, 0.06, -0.19],
            [0.07, -0.15, 0.22],
        ],
        dtype=jnp.float64,
    )
    params["c"] = jnp.asarray(0.17, dtype=jnp.float64)
    states = enumerate_spins(n)
    hamiltonian = dense_hamiltonian(n, J=1.0, h=1.0)
    return params, states, hamiltonian


def _rayleigh_reference(params, states, hamiltonian):
    """Dense linear algebra, independent of local-energy estimators."""
    log_amplitudes = np.asarray(batch_log_psi(params, states))
    psi = np.exp(log_amplitudes - np.max(log_amplitudes))
    norm = psi @ psi
    hpsi = np.asarray(hamiltonian) @ psi
    energy = float(psi @ hpsi / norm)
    variance = float(hpsi @ hpsi / norm - energy**2)
    return psi, psi**2 / norm, hpsi, energy, variance


def test_spin_basis_and_pauli_hamiltonian_convention():
    states = np.asarray(enumerate_spins(4))
    assert states.shape == (16, 4)
    assert set(np.unique(states)) == {-1, 1}
    assert np.unique(states, axis=0).shape[0] == 16

    hamiltonian = np.asarray(dense_hamiltonian(4, J=1.0, h=1.0))
    np.testing.assert_allclose(hamiltonian, hamiltonian.T, atol=0.0)
    # Independently evaluated periodic four-site Pauli TFIM, J=h=1.
    # This catches spin-1/2 versus Pauli conventions and boundary bond errors.
    np.testing.assert_allclose(
        np.linalg.eigvalsh(hamiltonian)[0],
        -5.226251859505508,
        rtol=0.0,
        atol=2e-12,
    )


def test_local_energy_equals_dense_hpsi_ratio(small_system):
    params, states, _ = small_system
    # Nonunit and unequal couplings also catch omitted or interchanged factors.
    coupling, field = 0.73, 1.21
    hamiltonian = dense_hamiltonian(4, J=coupling, h=field)
    psi, _, hpsi, _, _ = _rayleigh_reference(params, states, hamiltonian)
    actual = local_energies(params, states, J=coupling, h=field)
    np.testing.assert_allclose(actual, hpsi / psi, rtol=2e-12, atol=2e-12)


def test_exact_observables_equal_dense_rayleigh_reference(small_system):
    params, states, hamiltonian = small_system
    _, probabilities, _, energy, variance = _rayleigh_reference(
        params, states, hamiltonian
    )
    observables = exact_observables(params, states, hamiltonian)
    np.testing.assert_allclose(
        observables["probabilities"], probabilities, rtol=2e-12, atol=2e-14
    )
    np.testing.assert_allclose(observables["energy"], energy, atol=2e-12)
    np.testing.assert_allclose(observables["variance"], variance, atol=2e-11)
    np.testing.assert_allclose(np.sum(probabilities), 1.0, atol=2e-15)
    # This state is intentionally not an eigenvector: a zero gradient/variance
    # would make the derivative checks below uninformative.
    assert variance > 1e-3


def test_covariance_gradient_matches_differentiated_rayleigh_quotient(small_system):
    params, states, hamiltonian = small_system
    _, probabilities, _, _, _ = _rayleigh_reference(params, states, hamiltonian)
    covariance = energy_covariance_gradient(
        params, states, weights=jnp.asarray(probabilities)
    )
    direct = jax.grad(exact_energy)(params, states, jnp.asarray(hamiltonian))
    covariance_flat, _ = ravel_pytree(covariance)
    direct_flat, _ = ravel_pytree(direct)
    assert np.linalg.norm(np.asarray(direct_flat)) > 1e-3
    # In particular this tests the physical energy-gradient factor of two.
    np.testing.assert_allclose(
        covariance_flat, direct_flat, rtol=2e-10, atol=2e-11
    )
    np.testing.assert_allclose(covariance["c"], 0.0, atol=2e-12)


def test_covariance_gradient_matches_directional_finite_differences(small_system):
    params, states, hamiltonian = small_system
    _, probabilities, _, _, _ = _rayleigh_reference(params, states, hamiltonian)
    covariance = energy_covariance_gradient(
        params, states, weights=jnp.asarray(probabilities)
    )
    theta, unravel = ravel_pytree(params)
    gradient, _ = ravel_pytree(covariance)
    theta = np.asarray(theta)
    gradient = np.asarray(gradient)
    rng = np.random.default_rng(20261020)
    # More than one direction catches accidental agreement along a special
    # symmetry direction; recomputing the quotient updates normalization too.
    for _ in range(3):
        direction = rng.normal(size=theta.shape)
        direction /= np.linalg.norm(direction)
        predicted = float(gradient @ direction)
        numerical = []
        for step in (1e-4, 1e-5, 1e-6):
            plus = exact_energy(
                unravel(jnp.asarray(theta + step * direction)),
                states,
                jnp.asarray(hamiltonian),
            )
            minus = exact_energy(
                unravel(jnp.asarray(theta - step * direction)),
                states,
                jnp.asarray(hamiltonian),
            )
            numerical.append(float((plus - minus) / (2.0 * step)))
        best_error = min(abs(value - predicted) for value in numerical)
        assert best_error < 5e-8 + 2e-6 * abs(predicted)


def test_global_log_amplitude_shift_is_unphysical(small_system):
    params, states, hamiltonian = small_system
    shifted = {**params, "c": params["c"] + 2.7}
    before = exact_observables(params, states, hamiltonian)
    after = exact_observables(shifted, states, hamiltonian)
    for name in ("energy", "variance", "probabilities"):
        np.testing.assert_allclose(before[name], after[name], rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        local_energies(params, states),
        local_energies(shifted, states),
        rtol=2e-12,
        atol=2e-12,
    )
    first = energy_covariance_gradient(
        params, states, weights=before["probabilities"]
    )
    second = energy_covariance_gradient(
        shifted, states, weights=after["probabilities"]
    )
    first_flat, _ = ravel_pytree(first)
    second_flat, _ = ravel_pytree(second)
    np.testing.assert_allclose(first_flat, second_flat, rtol=2e-11, atol=2e-11)


def test_state_and_local_energy_have_global_spin_flip_symmetry(small_system):
    params, states, _ = small_system
    np.testing.assert_allclose(
        batch_log_psi(params, states),
        batch_log_psi(params, -states),
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        local_energies(params, states),
        local_energies(params, -states),
        rtol=2e-12,
        atol=2e-12,
    )
