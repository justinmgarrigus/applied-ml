"""A split launch must reproduce uninterrupted SPRING, walkers and RNGs."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('method',['spring','vmc_sgd','minsr'])
def test_full_state_resume_matches_uninterrupted_updates(tmp_path,method):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "vendor/jaqmc_spring/src"),
                                          str(root), env.get("PYTHONPATH", "")))
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               JAX_PLATFORMS="cpu")
    config = dict(method=method, ne=4, width=8, layers=2, variant="mlp", source=None,
                  steps=5, batch=4, sweeps=1, burn_sweeps=2, refresh_every=2,
                  pair_chunk=2, cpus=2, backend="cpu", seed=71,
                  checkpoint_every=1, log_every=1, keep_checkpoints=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    def launch(output, steps, resume=False, expected_success=True):
        command = [sys.executable, "-m", "fqh_spring.train", "--config", str(path),
                   "--output", str(output), "--steps", str(steps)]
        if resume:
            command.append("--resume")
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        if expected_success:
            assert result.returncode == 0, result.stdout + "\n" + result.stderr
        else:
            assert result.returncode != 0, "Changed invariant settings must be rejected"
        return result

    uninterrupted, split = tmp_path / "uninterrupted", tmp_path / "split"
    launch(uninterrupted, 5)
    launch(split, 2)
    launch(split, 5, resume=True)
    with np.load(uninterrupted / "checkpoint_latest.npz", allow_pickle=False) as a, \
            np.load(split / "checkpoint_latest.npz", allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            if name != "metadata":
                np.testing.assert_allclose(a[name], b[name], atol=1e-12, rtol=1e-12, err_msg=name)
        ma, mb = (json.loads(archive["metadata"].item()) for archive in (a, b))
        assert ma["step"] == mb["step"] == 5
        assert ma["rng_state"] == mb["rng_state"]
        assert ma["config_sha256"] == mb["config_sha256"]
    recorded = json.loads((split / 'run_config.json').read_text())
    assert recorded['architecture']['readout'] == 'final'
    assert recorded['architecture']['base_layers'] == 2
    from fqh_spring.evaluate import read_parameters
    full, plain = (read_parameters(split / name) for name in
                   ("checkpoint_latest.npz", "trained_parameters.npz"))
    assert set(full) == set(plain)
    for name in full:
        np.testing.assert_array_equal(full[name], plain[name])
    with (split / "metrics.csv").open(newline="") as stream:
        assert [int(row["step"]) for row in csv.DictReader(stream)] == [1, 2, 3, 4, 5]
    assert len(list((split / "checkpoints").glob("step_*.npz"))) == 2
    # Resume must fail before replacing the original run provenance.
    original_config = (split / "run_config.json").read_bytes()
    config["damping"] = 0.02
    path.write_text(json.dumps(config))
    failed = launch(split, 6, resume=True, expected_success=False)
    assert "settings changed" in failed.stderr
    assert (split / "run_config.json").read_bytes() == original_config
