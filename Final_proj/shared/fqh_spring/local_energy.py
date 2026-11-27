"""Exact bounded-memory pair hopping for occupation coefficients.

The network is a callable ``log_fn(params, occupation)`` returning a complex
log coefficient in the canonical LLL Slater basis.  All allowed occupied
source pairs and same-sum destination pairs are included.  Only a fixed
number of source pairs is processed at once; neither a many-body matrix nor
an enumerated many-body basis is captured by this kernel.

Invalid hops are replaced by their valid source occupation *before* network
evaluation.  Their contribution is then masked.  Diagonal coefficient
ratios are set to exactly one.  Fermion signs and the L2 subtraction follow
the independently checked ``multilayer_local_energy`` implementation.
"""
from __future__ import annotations

from itertools import combinations
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .physics import SphereOperator


def make_local_energy(operator: SphereOperator, log_fn, pair_chunk_size: int = 1):
    """Return jitted ``evaluate(params, occupations) -> energy,C1,C3,L2``.

    Occupations must be binary [batch,norb] rows in the operator's fixed
    particle-number and Lz sector.  The supplied network must be finite on
    valid configurations.  No cutoff, importance tempering, or hop sampling
    is used in this evaluation.
    """
    return _make_pair_evaluator(operator, log_fn, pair_chunk_size, raw_amplitudes=False)


def make_hamiltonian_action(operator: SphereOperator, amplitude_fn, pair_chunk_size: int = 1):
    """Return the raw coefficient and operator actions at each occupation.

    The `amplitude_fn(params, occupation)` callable returns a complex
    coefficient. The output keys are `amplitude`, `energy` (the physical
    Hamiltonian action), `C1`, `C3`, and `L2`. Each action is evaluated on
    the raw wavefunction without dividing by the sampled coefficient.
    """
    return _make_pair_evaluator(operator, amplitude_fn, pair_chunk_size, raw_amplitudes=True)


def _make_pair_evaluator(operator, coefficient_fn, pair_chunk_size, *, raw_amplitudes):
    if isinstance(pair_chunk_size, bool) or not isinstance(pair_chunk_size, Integral):
        raise TypeError("pair_chunk_size must be a positive integer")
    chunk_size = int(pair_chunk_size)
    if chunk_size < 1:
        raise ValueError("pair_chunk_size must be a positive integer")

    packed = np.zeros((*operator.pair_coefficients.shape, 3), dtype=np.float64)
    for source, (c, d) in enumerate(operator.pairs):
        pair_sum = int(c + d)
        pairs = operator.pairs_by_sum[pair_sum]
        column = int(np.flatnonzero(np.all(pairs == (c, d), axis=1))[0])
        size = len(pairs)
        packed[source, :size, 0] = operator.channel_blocks[1][pair_sum][:, column]
        packed[source, :size, 1] = operator.channel_blocks[3][pair_sum][:, column]
        packed[source, :size, 2] = operator.pair_l2_blocks[pair_sum][:, column]

    positions = np.asarray(list(combinations(range(operator.ne), 2)), dtype=np.int32)
    pair_count = len(positions)
    padded_count = ((pair_count + chunk_size - 1) // chunk_size) * chunk_size
    padded_positions = np.tile(positions[:1], (padded_count, 1))
    padded_positions[:pair_count] = positions
    position_chunks = jnp.asarray(padded_positions.reshape(-1, chunk_size, 2))
    active_chunks = jnp.asarray((np.arange(padded_count) < pair_count).reshape(-1, chunk_size))
    destinations = jnp.asarray(operator.pair_destinations)
    coefficients = jnp.asarray(packed)
    valid_padding = jnp.asarray(operator.pair_valid)
    pair_index = jnp.asarray(operator.pair_index)
    orbitals = jnp.arange(operator.norb)
    ne, v3 = operator.ne, operator.v3
    q = operator.flux / 2.0
    l2_shift = ne * (ne - 2) * q * (q + 1.0)
    batch_coefficients = jax.vmap(coefficient_fn, in_axes=(None, 0))

    @jax.jit
    def evaluate(params, occupations):
        occupations = occupations.astype(jnp.int32)
        batch_size = occupations.shape[0]
        occupied = jax.vmap(lambda n: jnp.nonzero(n, size=ne, fill_value=0)[0])(occupations)
        prefix = jnp.cumsum(occupations, axis=1) - occupations
        base = batch_coefficients(params, occupations)
        sample = jnp.arange(batch_size)[:, None, None]
        sample_pair = jnp.arange(batch_size)[:, None]

        def accumulate(values, source_chunk):
            source_positions, active = source_chunk
            c = occupied[:, source_positions[:, 0]]
            d = occupied[:, source_positions[:, 1]]
            row = pair_index[c, d]
            a, b = destinations[row, :, 0], destinations[row, :, 1]
            c3, d3 = c[..., None], d[..., None]
            vacant_a = occupations[sample, a] - (a == c3) - (a == d3) == 0
            vacant_b = occupations[sample, b] - (b == c3) - (b == d3) == 0
            valid = valid_padding[row] & vacant_a & vacant_b & active[None, :, None]
            parity = (prefix[sample_pair, c][..., None]
                      + prefix[sample_pair, d][..., None] - 1
                      + prefix[sample, a] + prefix[sample, b]
                      - (c3 < a) - (d3 < a) - (c3 < b) - (d3 < b))
            sign = 1 - 2 * (parity % 2)
            # Boolean updates are expressed as integer indicator sums because
            # source and destination orbitals may overlap on a valid hop.
            delta = ((orbitals == a[..., None]).astype(jnp.int32)
                     + (orbitals == b[..., None]).astype(jnp.int32)
                     - (orbitals == c3[..., None]).astype(jnp.int32)
                     - (orbitals == d3[..., None]).astype(jnp.int32))
            neighbors = occupations[:, None, None, :] + delta
            neighbors = jnp.where(valid[..., None], neighbors,
                                  occupations[:, None, None, :])
            coefficients_at_neighbors = batch_coefficients(params, neighbors.reshape(-1, operator.norb))
            coefficients_at_neighbors = coefficients_at_neighbors.reshape(valid.shape)
            diagonal = (a == c3) & (b == d3)
            if raw_amplitudes:
                values_at_neighbors = jnp.where(diagonal, base[:, None, None], coefficients_at_neighbors)
            else:
                log_ratio = jnp.where(valid & ~diagonal,
                                      coefficients_at_neighbors - base[:, None, None], 0.0 + 0.0j)
                values_at_neighbors = jnp.exp(log_ratio)
            signed_values = jnp.where(valid, sign * values_at_neighbors, 0.0j)
            increment = jnp.sum(coefficients[row] * signed_values[..., None], axis=(1, 2))
            return values + increment, None

        values, _ = jax.lax.scan(accumulate,
            jnp.zeros((batch_size, 3), dtype=jnp.result_type(base, 1j)),
            (position_chunks, active_chunks))
        first, third = values[:, 0], values[:, 1]
        if raw_amplitudes:
            return {"amplitude": base, "energy": first + v3 * third, "C1": first, "C3": third,
                    "L2": values[:, 2] - l2_shift * base}
        return {"energy": first + v3 * third, "C1": first, "C3": third, "L2": values[:, 2] - l2_shift}

    return evaluate
