"""Analytic pair-projector and complete-sector ED benchmark checks."""
import numpy as np
import pytest

from fqh_spring.ed_benchmark import benchmark_one, diagonalize
from fqh_spring.physics import SphereOperator
from fqh_spring.references import sparse_operators


def test_two_electron_normalized_projector_and_angular_spectrum():
    bits, H, raising, _ = sparse_operators(SphereOperator(2, flux=3))
    # The antisymmetric M=0 pair sector contains exactly R=3,L=0 and R=1,L=2.
    assert len(bits) == 2
    assert np.max(abs((H @ H - H).toarray())) < 1e-12
    result = diagonalize(H, raising, count=2)
    np.testing.assert_allclose(result["values"], [0, 1], atol=1e-12)
    assert [state["L"] for state in result["states"]] == [0, 2]
    assert result["neutral_gap_V1"] == pytest.approx(1, abs=1e-12)
    assert result["validation"]["all_reported_L_resolved"]


def test_four_electron_fresh_benchmark_is_zero_uniform_and_portable(tmp_path):
    root = tmp_path / "portable_project"
    result = benchmark_one(4, root, root / "results", count=8)
    assert result["dimension"] == 18
    assert abs(result["ground_energy_V1"]) < 1e-12
    assert result["neutral_gap_V1"] > 0
    assert result["zero_modes_in_saved_spectrum"] == 1
    assert result["ground"]["L2"] < 1e-20
    assert result["ground"]["maximum_occupation_error_from_uniform"] < 1e-12
    assert result["runtime_reconstruction"]["basis_identical"]
    assert result["runtime_reconstruction"]["Hamiltonian_max_error"] == 0
    assert result["reference"] == "references/ne4"
    assert not result["ed_used_for_training"]
