"""Portable modules must reproduce the retained N=8 checkpoint results."""
from pathlib import Path

import pytest

from fqh_spring.evaluate import evaluate


@pytest.mark.parametrize("filename,expected_energy,expected_fidelity", [
    ("source_ne8.npz", 0.5045306910453273, 0.35298627610467526),
    ("best_dense_ne8.npz", 0.4908449813352472, 0.36569881386865055),
])
def test_retained_checkpoint_numerical_regression(tmp_path, filename, expected_energy, expected_fidelity):
    root = Path(__file__).resolve().parents[1]
    result = evaluate(root / "checkpoints" / filename, root / "references/ne8",
                      tmp_path / "evaluation.json", angular=expected_fidelity is not None)
    assert abs(result["exact_variational_energy_V1"] - expected_energy) < 1e-11
    if expected_fidelity is not None:
        assert abs(result["ground_state_fidelity"] - expected_fidelity) < 1e-11
        assert result["exact_L0_probability"] > result["ground_state_fidelity"]
    assert result["inputs_unchanged"]
    assert not result["ed_used_for_training"]
