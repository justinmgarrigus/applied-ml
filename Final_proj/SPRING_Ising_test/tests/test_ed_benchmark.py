"""Independent complete-basis checks for the NumPy/SciPy ED benchmark."""

from pathlib import Path

import numpy as np
import pytest

from ed_benchmark import compare_checkpoint, rbm_wavefunction, solve_case, sparse_hamiltonian, spin_basis


def test_hamiltonian_matches_independent_tensor_product_construction():
    # Site zero is the least significant bit, hence the rightmost Kronecker factor.
    n, J, h = 4, 0.73, 1.21
    identity = np.eye(2)
    z, x = np.diag([-1.0, 1.0]), np.array([[0.0, 1.0], [1.0, 0.0]])

    def tensor(operators):
        result = np.ones((1, 1))
        for operator in reversed(operators):
            result = np.kron(result, operator)
        return result

    reference = np.zeros((2**n, 2**n))
    for site in range(n):
        zz, flip = [identity] * n, [identity] * n
        zz[site] = zz[(site + 1) % n] = z
        flip[site] = x
        reference -= J * tensor(zz) + h * tensor(flip)
    np.testing.assert_array_equal(sparse_hamiltonian(n, J, h).toarray(), reference)
    np.testing.assert_array_equal(spin_basis(4)[1], [1, -1, -1, -1])


@pytest.mark.parametrize("n", [3, 4, 6])
def test_critical_finite_size_energies_and_parity_gaps(n):
    report, arrays = solve_case(n, 1.0, 1.0)
    np.testing.assert_allclose(report["ground_energy"], -2 / np.sin(np.pi / (2 * n)), atol=3e-12)
    np.testing.assert_allclose(report["full_gap"], 2 * np.tan(np.pi / (4 * n)), atol=3e-12)
    np.testing.assert_allclose(report["even_parity_gap"], 8 * np.sin(np.pi / (2 * n)), atol=3e-12)
    assert report["ground_spin_flip_parity"] == 1
    assert report["low_state_spin_flip_parities"][1] == -1
    assert report["ground_degeneracy"] == 1
    np.testing.assert_allclose(np.linalg.eigvalsh(arrays["H"].toarray()), arrays["eigenvalues"], atol=3e-12)


def test_analytic_classical_and_transverse_limits():
    classical, _ = solve_case(4, 1.3, 0.0)
    assert classical["ground_degeneracy"] == 2
    np.testing.assert_allclose(classical["ground_energy"], -4 * 1.3, atol=2e-12)
    np.testing.assert_allclose(classical["full_gap"], 4 * 1.3, atol=2e-12)
    np.testing.assert_allclose(classical["ground_observables"]["nearest_neighbor_zz_per_bond"], 1.0)
    transverse, _ = solve_case(4, 0.0, 0.7)
    np.testing.assert_allclose(transverse["ground_energy"], -4 * 0.7, atol=2e-12)
    np.testing.assert_allclose(transverse["full_gap"], 2 * 0.7, atol=2e-12)
    np.testing.assert_allclose(transverse["ground_observables"]["transverse_magnetization_per_site"], 1.0)
    np.testing.assert_allclose(transverse["ground_observables"]["nearest_neighbor_zz_per_bond"], 0.0, atol=2e-12)


def test_retained_spring_checkpoint_independent_numpy_regression():
    parameters = Path(__file__).resolve().parents[1] / "results/spring_n6/trained_parameters.npz"
    report, arrays = solve_case(6, 1.0, 1.0)
    result, psi = compare_checkpoint(parameters, report, arrays)
    np.testing.assert_allclose(result["energy"], -7.727390852046698, atol=2e-12)
    np.testing.assert_allclose(result["variance"], 0.00020536374883783966, atol=2e-14)
    np.testing.assert_allclose(result["fidelity"], 0.9999985691163893, atol=2e-13)
    np.testing.assert_allclose(psi, psi[::-1], atol=2e-15)
    assert 1 - result["fidelity"] <= result["infidelity_upper_bound_from_even_gap"] + 1e-13
    np.testing.assert_array_equal(psi, rbm_wavefunction(parameters, 6))


def test_invalid_or_excessively_large_models_are_rejected():
    for n in (2, 11, 3.5, True):
        with pytest.raises(ValueError):
            sparse_hamiltonian(n)
    for J, h in ((np.nan, 1.0), (1.0, np.inf), (-1.0, 1.0), (1.0, -1.0)):
        with pytest.raises(ValueError):
            sparse_hamiltonian(4, J, h)
