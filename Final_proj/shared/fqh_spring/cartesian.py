"""Anchored complex amplitudes and the full-support SPRING fit."""
import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from . import model


def make_amplitude(anchor):
    """Return a raw coefficient with the exponential amplitude and tangent at the anchor."""
    def amplitude(params, occupation):
        ell0 = model.log_psi(anchor, occupation)
        ell = model.log_psi(params, occupation)
        return jnp.exp(ell0) * (1.0 + (ell - ell0))

    return amplitude


def _check_floor(floor):
    if isinstance(floor, bool) or not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("The Cartesian sampling floor must be positive and finite")


def make_proposal(amplitude, floor):
    """Adapt the sampler's log-coefficient interface to the full-support target."""
    _check_floor(floor)

    def log_proposal(params, occupation):
        return 0.5 * jnp.log(jnp.abs(amplitude(params, occupation))**2 + floor)

    return log_proposal


def make_fit(anchor, floor):
    """Build the importance-weighted projected amplitude fit for sampled rows.

    The input `acted` is the physical Hamiltonian action on the raw amplitude.
    Return the half-energy-gradient force, metric rows in JaQMC's convention,
    and self-normalized Monte Carlo diagnostics.
    """
    _check_floor(floor)

    def fit(params, occupations, amplitudes, acted):
        density = jnp.abs(amplitudes)**2
        inverse_target = 1.0 / (density + floor)
        norm_contributions = inverse_target * density
        norm = jnp.sum(norm_contributions)
        weights = inverse_target / norm
        root_weights = jnp.sqrt(weights)

        anchor_amplitudes = jnp.exp(model.batch_log_psi(anchor, occupations))
        derivatives = model.score_tree(params, occupations)
        rows = jax.vmap(lambda row: ravel_pytree(row)[0])(derivatives)
        jacobian = anchor_amplitudes[:, None] * rows

        bra = weights * amplitudes.conj()
        energy = bra @ acted
        tangent = root_weights[:, None] * (jacobian - amplitudes[:, None] * (bra @ jacobian))
        residual = root_weights * (acted - energy * amplitudes)
        force = (residual @ tangent.conj()).real

        # Rotate the weighted state to a real unit vector, then reflect that
        # vector to the negative constant row direction. JaQMC's row centering
        # then preserves the projected metric. Its 1/B cancels the row scale.
        tangent = jnp.exp(-1j * jnp.angle(amplitudes))[:, None] * tangent
        batch_size = amplitudes.shape[0]
        reflection = root_weights * jnp.abs(amplitudes) + batch_size**-0.5
        correction = 2.0 * (reflection @ tangent) / (reflection @ reflection)
        optimizer_rows = batch_size**0.5 * (tangent - reflection[:, None] * correction)

        _, unravel = ravel_pytree(params)
        diagnostics = dict(
            energy_estimate=energy.real,
            energy_imaginary_estimate=energy.imag,
            local_energy_variance=jnp.vdot(residual, residual).real,
            importance_ess=norm**2 / jnp.sum(norm_contributions**2),
        )
        return unravel(force), optimizer_rows, diagnostics

    return fit
