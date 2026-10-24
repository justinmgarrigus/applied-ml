"""Independent V1 convention checks using angular lowering and full Fock space.

Production constructs pair projectors from two-particle L² eigenvectors and
applies individual scattering connections.  Here the R=1 pair amplitudes come
from lowering the analytic highest-weight Slater state |q-1,q>, and the
many-body interaction is built from explicit fermion annihilation matrices.
This checks the pair normalization, orbital order, and exchange signs without
using the production pair diagonalization or scattering implementation.
"""

from itertools import combinations

import numpy as np
import pytest
from scipy import sparse

from fqh_spring.physics import SphereOperator
from fqh_spring.references import sparse_operators


def _lowered_r1_pairs(flux):
    """Return normalized Slater amplitudes of |L=q-1,M> for every pair M."""
    angular = flux - 1
    amplitudes = {(flux - 1, flux): 1.0}
    blocks = {}
    for magnetization in range(angular, -angular - 1, -1):
        pairs = [pair for pair in combinations(range(flux + 1), 2)
                 if sum(pair) == magnetization + flux]
        vector = np.array([amplitudes.get(pair, 0.0) for pair in pairs])
        blocks[magnetization + flux] = (pairs, vector)
        if magnetization == -angular:
            break
        lowered = {}
        for (a, b), value in amplitudes.items():
            if a > 0:
                pair = (a - 1, b)
                lowered[pair] = (lowered.get(pair, 0.0)
                                 + value * np.sqrt(a * (flux - a + 1)))
            if b - 1 > a:
                pair = (a, b - 1)
                lowered[pair] = (lowered.get(pair, 0.0)
                                 + value * np.sqrt(b * (flux - b + 1)))
        normalization = np.sqrt((angular + magnetization)
                                * (angular - magnetization + 1))
        amplitudes = {pair: value / normalization
                      for pair, value in lowered.items()}
    return blocks


def _full_fock_annihilation(norb, orbital):
    """c_a in the complete 2**norb occupation basis, with ascending creators."""
    dimension = 1 << norb
    columns = np.array([bits for bits in range(dimension)
                        if bits & (1 << orbital)])
    rows = columns ^ (1 << orbital)
    coefficients = np.array([
        -1.0 if (int(bits) & ((1 << orbital) - 1)).bit_count() % 2 else 1.0
        for bits in columns
    ])
    return sparse.csr_matrix((coefficients, (rows, columns)),
                             shape=(dimension, dimension))


def independent_v1_validation(ne):
    """Compute measured errors; also reused to generate a validation record."""
    flux = 3 * (ne - 1)
    norb = flux + 1
    dimension = 1 << norb
    blocks = _lowered_r1_pairs(flux)
    annihilation = [_full_fock_annihilation(norb, a) for a in range(norb)]
    full_hamiltonian = sparse.csr_matrix((dimension, dimension), dtype=float)
    pair_norm_error = 0.0
    for pairs, amplitudes in blocks.values():
        pair_norm_error = max(pair_norm_error, abs(amplitudes @ amplitudes - 1))
        # B_M = sum_(a<b) u_ab(M) c_b c_a, so H_V1=sum_M B_M† B_M.
        pair_annihilation = sparse.csr_matrix((dimension, dimension), dtype=float)
        for (a, b), coefficient in zip(pairs, amplitudes):
            pair_annihilation += coefficient * (annihilation[b] @ annihilation[a])
        full_hamiltonian += pair_annihilation.T @ pair_annihilation

    # Enumerate this basis independently of the production completion counter.
    bits = np.array([
        state for state in range(dimension)
        if state.bit_count() == ne
        and 2 * sum(a for a in range(norb) if state & (1 << a)) == ne * flux
    ], dtype=np.int64)
    independent_hamiltonian = full_hamiltonian[bits][:, bits]
    operator = SphereOperator(ne)
    production_bits, hamiltonian, _, _ = sparse_operators(operator)
    np.testing.assert_array_equal(bits, production_bits)
    pair_error = 0.0
    for pair_sum, pairs in operator.pairs_by_sum.items():
        independent_pairs, amplitudes = blocks[pair_sum]
        assert independent_pairs == list(map(tuple, pairs))
        pair_error = max(pair_error, float(np.max(abs(
            np.outer(amplitudes, amplitudes) - operator.channel_blocks[1][pair_sum]))))
    difference = independent_hamiltonian - hamiltonian
    matrix_error = float(np.max(abs(difference.data), initial=0.0))
    eigenvalues = np.linalg.eigvalsh(independent_hamiltonian.toarray())
    return {
        "ne": ne,
        "flux": flux,
        "complete_fixed_M0_dimension": len(bits),
        "full_occupation_Fock_dimension": dimension,
        "maximum_pair_vector_normalization_error": float(pair_norm_error),
        "maximum_pair_projector_error": pair_error,
        "maximum_manybody_matrix_error": matrix_error,
        "independent_ground_energy_V1": float(eigenvalues[0]),
        "independent_neutral_gap_V1": float(eigenvalues[1] - eigenvalues[0]),
    }


@pytest.mark.parametrize("ne", [2, 3, 4])
def test_independent_pair_projector_and_fermion_embedding(ne):
    result = independent_v1_validation(ne)
    assert result["maximum_pair_vector_normalization_error"] < 2e-13
    assert result["maximum_pair_projector_error"] < 2e-13
    assert result["maximum_manybody_matrix_error"] < 2e-13
    assert abs(result["independent_ground_energy_V1"]) < 2e-13
    assert result["independent_neutral_gap_V1"] > 0
    if ne == 2:
        # A normalized two-electron R=1 projector has its nonzero eigenvalue 1.
        assert result["independent_neutral_gap_V1"] == pytest.approx(1, abs=2e-13)
