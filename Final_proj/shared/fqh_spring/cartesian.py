"""Anchored complex amplitudes and projective fitting with Born samples."""
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


def make_proposal(amplitude):
    """Return a log coefficient whose squared modulus defines the Born target."""
    def log_proposal(params, occupation):
        return jnp.log(jnp.abs(amplitude(params, occupation)))

    return log_proposal


def make_fit(anchor, *, metric_rows=True):
    """Build the projected amplitude fit for ordinary Born samples.

    The input `acted` is the physical Hamiltonian action on the raw amplitude.
    Return the half-energy-gradient force, metric rows in JaQMC's convention,
    and Monte Carlo diagnostics. VMC-SGD sets metric_rows=False
    to compute its force without preparing a metric solve.
    """
    def fit(params, occupations, amplitudes, acted):
        density = jnp.abs(amplitudes)**2
        valid = density > 0.0
        count = jnp.sum(valid)
        # Born samples give equal estimator weight to nonzero amplitudes.
        # A node has zero probability and contributes zero if present in a batch.
        inverse_density = jnp.where(valid, 1.0 / jnp.where(valid, density, 1.0), 0.0)
        weights = inverse_density / count
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

        optimizer_rows = None
        if metric_rows:
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
            valid_born_rows=count,
        )
        return unravel(force), optimizer_rows, diagnostics

    return fit
