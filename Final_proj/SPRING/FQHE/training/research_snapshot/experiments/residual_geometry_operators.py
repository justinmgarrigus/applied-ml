"""Local operator data for explicitly selected occupation roots.

The caller draws a candidate pool and chooses root *indices*. This helper
keeps every chosen root row, including repeated indices/configurations. Only
network evaluation inputs are deduplicated. Selection probabilities and their
inverse weights remain the caller's responsibility; they must not be absorbed
into or confused with the Hamiltonian-connection importance coefficients.

No complete occupation basis, dense operator, projector or ED state is read.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from learned_sector_operators import (
    RootConnectionCache,
    _padded_rows,
    _positive_integer,
    _sample_h_connections,
    occupation_bits,
    occupations_from_bits,
)


def operator_batch_from_roots(operator, roots, rng: np.random.Generator,
                              cache: RootConnectionCache | None = None, *,
                              h_samples_per_root: int | None = 4,
                              union_pad_to: int | None = None,
                              h_pad_to: int | None = None,
                              k_pad_to: int | None = 43) -> dict[str, Any]:
    """Build exact local K and exact/importance-sampled H rows for given roots.

    Roots must be a nonempty binary ``[batch, norb]`` array in the operator's
    fixed particle-number and angular-projection sector. The supplied RNG is
    used only for H-connection draws. Repeated roots get independent H draws.
    Thus callers can use separate RNG streams for pool generation, randomized
    selection and operator sampling.

    With ``h_samples_per_root=None`` H is exact. Otherwise H draws are with
    replacement, proportional to ``abs(H_nm)``, with the original unbiased
    ``H_nm/(draw_count * p_nm)`` coefficients. K is always exact and must be
    assembled before a squared angular residual is formed. Insufficient
    padding raises rather than truncating a local row.
    """
    started = time.perf_counter()
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a NumPy Generator')
    roots = np.asarray(roots)
    if (roots.ndim != 2 or roots.shape[0] < 1 or
            roots.shape[1] != operator.norb or
            not np.all((roots == 0) | (roots == 1))):
        raise ValueError('roots must be a nonempty binary [batch, norb] array')
    if not np.all(roots.sum(axis=1) == operator.ne):
        raise ValueError('a root has the wrong particle count')
    orbital_sums = roots @ np.arange(operator.norb)
    if not np.all(2 * orbital_sums - operator.ne * operator.flux == operator.two_lz):
        raise ValueError('a root has the wrong angular projection')
    roots = np.asarray(roots, dtype=np.int8)
    if cache is None:
        cache = RootConnectionCache(operator)
    if cache.operator is not operator:
        raise ValueError('The cache belongs to a different operator object')
    if h_samples_per_root is not None:
        h_samples_per_root = _positive_integer(h_samples_per_root, 'h_samples_per_root')

    root_bits = [occupation_bits(root) for root in roots]
    h_rows, k_rows = [], []
    exact_h_counts, exact_k_counts = [], []
    previous_hits, previous_misses = cache.hits, cache.misses
    for root in roots:
        h_bits, h_values, k_bits, k_values = cache.connections(root)
        exact_h_counts.append(len(h_bits))
        exact_k_counts.append(len(k_bits))
        if h_samples_per_root is not None:
            h_bits, h_values = _sample_h_connections(
                h_bits, h_values, rng, h_samples_per_root)
        h_rows.append((h_bits, h_values))
        k_rows.append((k_bits, k_values))

    union_bits = list(dict.fromkeys(root_bits))
    union_index = {bits: index for index, bits in enumerate(union_bits)}
    for bits, _ in h_rows + k_rows:
        for key in bits:
            if key not in union_index:
                union_index[key] = len(union_bits)
                union_bits.append(key)
    unique_count = len(union_bits)
    capacity = (unique_count if union_pad_to is None else
                _positive_integer(union_pad_to, 'union_pad_to'))
    if capacity < unique_count:
        raise ValueError(f'union_pad_to={capacity} is smaller than required '
                         f'{unique_count}; no truncation is allowed')
    cache.visited_amplitude_configurations.update(union_bits)
    padded_bits = union_bits + [union_bits[0]] * (capacity - unique_count)
    h_indices, h_coeff, h_valid = _padded_rows(h_rows, union_index, h_pad_to, 'h_pad_to')
    k_indices, k_coeff, k_valid = _padded_rows(k_rows, union_index, k_pad_to, 'k_pad_to')
    dimension = int(operator.dimension())
    coverage = dict(
        root_rows=len(roots), unique_root_configurations=len(set(root_bits)),
        unique_H_neighbor_configurations=len(set(key for bits, _ in h_rows for key in bits)),
        unique_K_neighbor_configurations=len(set(key for bits, _ in k_rows for key in bits)),
        unique_amplitude_configurations=unique_count, NN_input_rows_including_padding=capacity,
        padding_rows=capacity - unique_count,
        exact_H_connections_per_root=exact_h_counts,
        exact_K_connections_per_root=exact_k_counts,
        returned_H_connections_per_root=[len(row[0]) for row in h_rows],
        returned_K_connections_per_root=[len(row[0]) for row in k_rows],
        sector_dimension_from_integer_DP=dimension,
        minibatch_unique_fraction_of_sector=unique_count / dimension,
        cache_hits_this_batch=cache.hits - previous_hits,
        cache_misses_this_batch=cache.misses - previous_misses,
        **cache.statistics())
    return dict(
        roots=roots,
        occupations=occupations_from_bits(padded_bits, operator.norb),
        occupation_mask=np.arange(capacity) < unique_count,
        root_indices=np.asarray([union_index[key] for key in root_bits], dtype=np.int32),
        H_indices=h_indices, H_coefficients=h_coeff, H_mask=h_valid,
        K_indices=k_indices, K_coefficients=k_coeff, K_mask=k_valid,
        coverage=coverage, preprocessing_seconds=time.perf_counter() - started,
        H_estimator=('exact' if h_samples_per_root is None else
                     'unbiased_abs_weighted_connections'),
        H_samples_per_root=h_samples_per_root,
        K_estimator='exact_combined_Lminus_Lplus_plus_M_Mplus1',
        root_distribution='explicit_selected_roots; caller supplies selection correction',
        explicit_many_body_basis_enumeration=False, reference_data_loaded=False)
