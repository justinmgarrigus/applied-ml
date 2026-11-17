"""Check acceptance, state ownership, reproducibility and independent diagnostics."""
import json

import numpy as np
import pytest

import ising_annealing as sa


class FixedDraw:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_metropolis_boundary_and_nonfinite():
    assert sa.accept(-1., .1, FixedDraw(.99))
    assert sa.accept(0., .1, FixedDraw(.99))
    assert sa.accept(1., 1., FixedDraw(.3))
    assert not sa.accept(1., 1., FixedDraw(.4))
    assert not sa.accept(1000., .001, FixedDraw(.5))
    assert not sa.accept(float("nan"), 1., FixedDraw(.1))
    assert not sa.accept(float("inf"), 1., FixedDraw(.1))


def settings(**overrides):
    return dict(steps=20, temperature=1., final_temperature=.01,
                proposal_scale=.5, final_proposal_scale=.05, seed=9) | overrides


def test_rejection_keeps_current_and_best_and_counts_calls():
    calls = []

    def objective(weights):
        calls.append(weights.copy())
        return 0. if len(calls) == 1 else float("inf")

    initial = np.zeros((2, 3))
    result = sa.anneal(initial, objective, **settings())
    np.testing.assert_array_equal(result["current"], initial)
    np.testing.assert_array_equal(result["best"], initial)
    assert len(calls) == 21
    assert all(row["accepted"] == 0 for row in result["history"])
    assert result["history"][-1]["objective_evaluations"] == 21


def test_best_survives_accepted_uphill_and_schedules_reproduce():
    def search():
        energies = iter([3., 1., 2.])
        return sa.anneal(np.zeros(2), lambda _: next(energies),
                         **settings(steps=2, temperature=1e100, final_temperature=1e99))

    first, second = search(), search()
    assert first["current_energy"] == 2.
    assert first["best_energy"] == 1.
    assert not np.array_equal(first["current"], first["best"])
    np.testing.assert_array_equal(first["current"], second["current"])
    assert first["history"][0]["temperature"] == pytest.approx(1e100)
    assert first["history"][-1]["temperature"] == pytest.approx(1e99)
    assert first["history"][-1]["proposal_scale"] == pytest.approx(.05)


@pytest.mark.parametrize("overrides", [dict(steps=0), dict(temperature=0),
    dict(final_temperature=2), dict(proposal_scale=float("nan")), dict(seed=-1)])
def test_invalid_search_never_evaluates_objective(overrides):
    with pytest.raises(ValueError):
        sa.anneal(np.zeros(2), lambda _: pytest.fail("must validate first"), **settings(**overrides))


def test_shared_rbm_endpoints_and_artifacts_without_gradients(tmp_path, monkeypatch):
    monkeypatch.setattr(sa.backend, "energy_covariance_gradient",
                        lambda *args, **kwargs: pytest.fail("SA must not compute a gradient"))
    output = tmp_path / "run"
    args = sa.parser().parse_args(["--n", "4", "--hidden", "3", "--steps", "8",
                                  "--output", str(output)])
    result = sa.run(args, tmp_path)
    assert result["search_objective_evaluations"] == 9
    assert result["searched_parameters"] == 12
    assert result["best"]["energy"] <= result["initial"]["energy"]
    assert result["best"]["energy"] <= result["final"]["energy"] + 1e-12
    assert result["best"]["energy_error"] >= -1e-12
    assert 0 <= result["best"]["fidelity"] <= 1 + 1e-12
    assert json.loads((output / "metrics.json").read_text())["no_gradient_used"]
    initial = sa.backend.init_params(7, 4, 3)
    with np.load(output / "initial_parameters.npz") as saved:
        np.testing.assert_array_equal(saved["W"], initial["W"])
    with np.load(output / "best_parameters.npz") as saved:
        assert saved["c"] == 0.
        states = sa.backend.enumerate_spins(4)
        H = sa.backend.dense_hamiltonian(4)
        assert float(sa.backend.exact_energy(dict(saved), states, H)) == pytest.approx(result["best"]["energy"])
    with pytest.raises(FileExistsError):
        sa.run(args, tmp_path)
