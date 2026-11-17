"""Keep optimizer labels, memory controls and output ownership explicit."""
from pathlib import Path

import pytest

import ising_backend as backend


@pytest.mark.parametrize("optimizer,mu", [("spring", 0.9), ("sgd", 0.9), ("damped-minsr", 0.0)])
def test_algorithm_entrypoint_keeps_fixed_label_and_fresh_destination(monkeypatch, tmp_path, optimizer, mu):
    monkeypatch.setattr(backend, "run", lambda args: vars(args))
    first = backend.run_algorithm(optimizer, tmp_path, [])
    second = backend.run_algorithm(optimizer, tmp_path, [])
    assert first["optimizer"] == optimizer
    assert first["mu"] == mu
    assert Path(first["output"]).parent == tmp_path / "runs"
    assert first["output"] != second["output"]


def test_memory_free_control_rejects_nonzero_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "run", lambda args: pytest.fail("training should not start"))
    with pytest.raises(SystemExit) as exc:
        backend.run_algorithm("damped-minsr", tmp_path, ["--mu", "0.9"])
    assert exc.value.code == 2


def test_fixed_algorithm_rejects_another_optimizer(monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "run", lambda args: pytest.fail("training should not start"))
    with pytest.raises(SystemExit) as exc:
        backend.run_algorithm("spring", tmp_path, ["--optimizer", "sgd"])
    assert exc.value.code == 2


def test_existing_output_is_rejected_before_training(tmp_path):
    marker = tmp_path / "preserve.txt"
    marker.write_text("existing result\n")
    args = backend.parser().parse_args(["--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        backend.run(args)
    assert marker.read_text() == "existing result\n"
