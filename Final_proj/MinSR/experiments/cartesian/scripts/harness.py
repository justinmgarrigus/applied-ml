"""Shared launcher and exact evaluator for the Cartesian MinSR FQHE study.

Training uses the pinned portable MinSR entry point from main 73b18a8
(anchored Cartesian amplitudes, ordinary Born sampling). Evaluation rebuilds
u_theta(n) on the complete fixed-Lz sector and compares it with ED.
"""
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "source/Final_proj"
SOURCE_COMMIT = "73b18a8bbaa4501842818bd42e0595027b6b1fc5"
ENTRY = PROJECT / "MinSR/FQHE/training/run.py"
WORKSPACE = HERE.parent
PYTHON = {"gpu": WORKSPACE / ".venv-gpu/bin/python", "cpu": WORKSPACE / ".venv/bin/python"}
TARGET = 0.01
FLUX = {4: 9, 6: 15, 8: 21}

# Fixed physics/model settings shared with SPRING's ne8_cartesian.json.
BASE = dict(method="minsr", two_lz=0, v3=0.0, variant="mlp", readout="legacy_additive",
            amplitude_map="cartesian", width=64, layers=6, source=None,
            sweeps=2, burn_sweeps=50, refresh_every=25, mu=0.0, pair_chunk=2,
            native_threads=1, max_runtime_seconds=None)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def make_config(ne, init_seed, seed, steps, every, settings, amplitude_map="cartesian", backend="gpu"):
    config = dict(BASE, ne=ne, flux=FLUX[ne], init_seed=init_seed, seed=seed, steps=steps,
                  checkpoint_every=every, log_every=every, keep_checkpoints=steps // every + 1,
                  cpus=2, backend="auto" if backend == "gpu" else "cpu", amplitude_map=amplitude_map)
    config.update(settings)
    return config


def train(job, config, device="gpu", cpus="8,10"):
    """Run one training job in a fresh directory; never overwrite."""
    job.mkdir(parents=True, exist_ok=False)
    write_json(job / "input.json", config)
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    env["JAX_PLATFORMS"] = "cuda" if device == "gpu" else "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    command = ["/usr/bin/time", "-f", "%e %M", "-o", str(job / "process_time_rss.txt"),
               "taskset", "-c", cpus, str(PYTHON[device]), str(ENTRY),
               "--config", str(job / "input.json"), "--output", str(job / "train")]
    with (job / "train.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"MinSR runner exited {result.returncode}; see {job / 'train.log'}")
    summary = read_json(job / "train/training_summary.json")
    runtime = read_json(job / "train/run_config.json")
    assert summary["status"] == "completed" and summary["completed_steps"] == config["steps"], summary
    if device == "gpu":
        assert "cuda" in runtime["device"].lower(), runtime["device"]
    return summary, runtime["device"]


class Evaluator:
    """Load one ED reference and evaluate many frozen checkpoints of that sector."""

    def __init__(self, ne):
        sys.path.insert(0, str(PROJECT / "shared"))
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from fqh_spring import model
        from fqh_spring.cartesian import make_amplitude
        from fqh_spring.evaluate import read_parameters, state_diagnostics
        from fqh_spring.references import load_reference
        self.jax, self.jnp, self.model = jax, jnp, model
        self.make_amplitude, self.read_parameters = make_amplitude, read_parameters
        self.state_diagnostics = state_diagnostics
        self.reference = PROJECT / f"benchmarks/fqhe/references/ne{ne}"
        self.data = load_reference(self.reference)
        values = np.sort(self.data["eigenvalues"])
        self.e0, self.gap = float(values[0]), float(values[1] - values[0])
        self.occupations = jnp.asarray(self.data["occupations"])
        self.ne = ne

    def psi(self, params, anchor, amplitude_map):
        jnp = self.jnp
        params = {k: jnp.asarray(v) for k, v in params.items()}
        if amplitude_map == "cartesian":
            anchor = {k: jnp.asarray(v) for k, v in anchor.items()}
            amplitude = self.jax.vmap(self.make_amplitude(anchor), in_axes=(None, 0))
            values = np.asarray(amplitude(params, self.occupations))
        else:
            logs = np.asarray(self.model.batch_log_psi(params, self.occupations))
            values = np.exp(logs - logs.real.max())
        assert np.isfinite(values).all()
        return values / np.linalg.norm(values)

    def run(self, job, angular_final=True):
        """Evaluate every saved checkpoint of a finished job; write curve.csv and metrics.json."""
        job = Path(job)
        trained = job / "train"
        config = read_json(job / "input.json")
        meta = self.data["metadata"]
        assert all(config[key] == meta[key] for key in ("ne", "flux", "two_lz", "v3"))
        run_config = read_json(trained / "run_config.json")
        amplitude_map = run_config["configuration"].get("amplitude_map", "exponential")
        assert amplitude_map == config["amplitude_map"]
        anchor_path = trained / "initial_parameters.npz"
        anchor = self.read_parameters(anchor_path)
        if amplitude_map == "cartesian":
            assert file_hash(anchor_path) == run_config["frozen_anchor_sha256"]
        with (trained / "metrics.csv").open(newline="") as stream:
            mc_rows = {int(row["step"]): row for row in csv.DictReader(stream)}
        checkpoints = sorted((trained / "checkpoints").glob("step_*.npz"))
        rows, eval_seconds = [], 0.0
        H = self.data["H"]
        gs = self.data["eigenvectors"][:, np.argsort(self.data["eigenvalues"])[0]]
        for checkpoint in checkpoints:
            started = perf_counter()
            step = int(checkpoint.stem.split("_")[1])
            psi = self.psi(self.read_parameters(checkpoint), anchor, amplitude_map)
            energy = float(np.vdot(psi, H @ psi).real)
            fidelity = float(abs(np.vdot(gs, psi)) ** 2)
            prob = np.abs(psi) ** 2
            support = float(1.0 / np.sum(prob ** 2))
            residual = H @ psi - energy * psi
            eval_seconds += perf_counter() - started
            mc = mc_rows.get(step, {})
            error = energy - self.e0
            rows.append(dict(step=step,
                             training_seconds=float(mc["training_seconds"]) if mc else 0.0,
                             energy=energy, energy_error=error, error_per_electron=error / self.ne,
                             gap_ratio=error / self.gap, fidelity=fidelity,
                             variance=float(np.vdot(residual, residual).real),
                             effective_support=support,
                             mc_energy=float(mc["energy_estimate"]) if mc else None,
                             acceptance=float(mc["acceptance"]) if mc and mc.get("acceptance") else None,
                             cap_active=float(mc["cap_active"]) if mc and mc.get("cap_active") else None,
                             distinct_configurations=float(mc["distinct_configurations"]) if mc else None))
        final_psi = self.psi(self.read_parameters(checkpoints[-1]), anchor, amplitude_map)
        diagnostics = self.state_diagnostics(final_psi, self.data, angular=angular_final)
        diagnostics.pop("orbital_occupations", None)
        # Cross-check the per-checkpoint metrics against the repository's own diagnostics.
        assert abs(diagnostics["ground_state_fidelity"] - rows[-1]["fidelity"]) < 1e-9
        assert abs(diagnostics["exact_variational_energy_V1"] - rows[-1]["energy"]) < 1e-9
        write_json(job / "endpoint_diagnostics.json", diagnostics)
        with (job / "curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary = read_json(trained / "training_summary.json")
        elapsed, rss = map(float, (job / "process_time_rss.txt").read_text().split())
        hit = next((row for row in rows if row["gap_ratio"] <= TARGET), None)
        best = min(rows, key=lambda row: row["energy"])
        result = dict(config=config, device=run_config.get("device"), e0=self.e0, gap=self.gap,
                      target=TARGET, target_step=hit["step"] if hit else None,
                      target_training_seconds=hit["training_seconds"] if hit else None,
                      training_seconds=summary["training_seconds"],
                      compilation_seconds=summary["current_launch_compile_seconds"],
                      total_process_seconds=elapsed, peak_rss_mib=rss / 1024,
                      evaluation_seconds=eval_seconds,
                      initial_parameters_sha256=file_hash(anchor_path),
                      final_L0_probability=diagnostics.get("exact_L0_probability"),
                      final_max_fidelity_fixed_magnitudes=diagnostics.get("maximum_fidelity_at_fixed_magnitudes"),
                      best_step=best["step"], best_energy=best["energy"], best_fidelity=best["fidelity"],
                      initial=rows[0], final=rows[-1])
        write_json(job / "metrics.json", result)
        return result
