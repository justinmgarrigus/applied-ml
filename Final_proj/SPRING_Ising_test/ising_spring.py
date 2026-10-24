"""Small quantum Ising VMC experiment using the actual JaQMC SPRING optimizer.

Run with this folder's .venv/bin/python. See README.md for physics/conventions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import time

# A reproducible CPU experiment; set before importing JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache" / "matplotlib"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
from jaqmc.optimizer.sr.jasr import robust_sr


UPSTREAM_COMMIT = "71f94158f2aca88ded904451f5314f71cdff5666"


def enumerate_spins(n: int) -> jax.Array:
    """Bit order: site zero is the least significant bit; bit 0 means spin -1."""
    ids = np.arange(2**n, dtype=np.int64)
    return jnp.asarray(2 * ((ids[:, None] >> np.arange(n)) & 1) - 1, dtype=jnp.float64)


def dense_hamiltonian(n: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:
    """H = -J sum Z_i Z_(i+1) - h sum X_i, periodic, Pauli eigenvalues +/-1."""
    if n < 3:
        raise ValueError("Use n >= 3 to avoid the two-site periodic-bond ambiguity.")
    spins = np.asarray(enumerate_spins(n))
    matrix = np.diag(-J * np.sum(spins * np.roll(spins, -1, axis=1), axis=1))
    ids = np.arange(2**n)
    for site in range(n):
        matrix[ids, ids ^ (1 << site)] -= h
    return matrix


def init_params(seed: int, n: int, hidden: int) -> dict:
    key = jax.random.PRNGKey(seed)
    return {"W": 0.15 * jax.random.normal(key, (n, hidden), dtype=jnp.float64),
            "c": jnp.array(0.0, dtype=jnp.float64)}


def log_psi(params: dict, spin: jax.Array) -> jax.Array:
    """Positive RBM amplitude with hidden biases zero: f=c+sum log cosh(s W).

    logcosh is evaluated stably. This enforces global spin-flip symmetry.
    c is a normalization gauge; its exact physical gradient should vanish.
    """
    activation = spin @ params["W"]
    return params["c"] + jnp.sum(jnp.logaddexp(activation, -activation) - jnp.log(2.0))


batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0))
score_fn = jax.vmap(jax.grad(log_psi), in_axes=(None, 0))


def score_matrix(params: dict, spins: jax.Array) -> jax.Array:
    return jax.vmap(lambda tree: ravel_pytree(tree)[0])(score_fn(params, spins))


def local_energies(params: dict, spins: jax.Array, J: float = 1.0,
                   h: float = 1.0) -> jax.Array:
    n = spins.shape[1]
    flipped = spins[:, None, :] * (1.0 - 2.0 * jnp.eye(n))[None, :, :]
    log_values = batch_log_psi(params, spins)
    log_flipped = batch_log_psi(params, flipped.reshape(-1, n)).reshape(-1, n)
    diagonal = -J * jnp.sum(spins * jnp.roll(spins, -1, axis=1), axis=1)
    return diagonal - h * jnp.sum(jnp.exp(log_flipped - log_values[:, None]), axis=1)


def exact_energy(params: dict, states: jax.Array, H: jax.Array) -> jax.Array:
    log_values = batch_log_psi(params, states)
    psi = jnp.exp(log_values - jnp.max(log_values))
    psi = psi / jnp.linalg.norm(psi)
    return psi @ H @ psi


def exact_observables(params: dict, states: jax.Array, H: jax.Array) -> dict:
    log_values = batch_log_psi(params, states)
    psi = jnp.exp(log_values - jnp.max(log_values))
    psi = psi / jnp.linalg.norm(psi)
    energy = psi @ H @ psi
    residual = H @ psi - energy * psi
    return {"energy": energy, "variance": residual @ residual,
            "probabilities": psi**2, "psi": psi}


def energy_covariance_gradient(params: dict, spins: jax.Array,
                               weights: jax.Array | None = None,
                               J: float = 1.0, h: float = 1.0) -> dict:
    """Physical gradient: 2 Cov(O,E_loc), using exact or empirical weights.

    For a finite sampled batch this uses the empirical mean (1/B, not 1/(B-1)),
    consistently with JaQMC's score covariance. It has the usual finite-B bias.
    """
    energies = local_energies(params, spins, J, h)
    weights = (jnp.ones(spins.shape[0]) / spins.shape[0] if weights is None
               else weights / jnp.sum(weights))
    residual = energies - jnp.sum(weights * energies)
    return jax.tree.map(lambda score: 2.0 * jnp.tensordot(weights * residual,
                                                       score, axes=(0, 0)),
                        score_fn(params, spins))


def make_spring(learning_rate: float = 0.05, damping: float = 0.001,
                mu: float = 0.9, max_norm: float | None = 0.1):
    """Call upstream JaQMC, with its robust-SR/MARCH extras explicitly disabled."""
    return robust_sr(log_psi, learning_rate=learning_rate, damping=damping,
                     spring_mu=mu, max_norm=max_norm, max_cond_num=None,
                     robust_gamma=None, march_beta=None, mixed_precision=True,
                     score_chunk_size=None, score_norm_clip=None,
                     gram_num_chunks=None, gram_dot_prec="F64", prune_inactive=False)


def gradient_diagnostics(params: dict, states: jax.Array, H: jax.Array,
                         J: float, h: float) -> tuple[dict, dict]:
    obs = exact_observables(params, states, H)
    cov = energy_covariance_gradient(params, states, obs["probabilities"], J, h)
    auto = jax.grad(exact_energy)(params, states, H)
    flat, unravel = ravel_pytree(params)
    g_cov, _ = ravel_pytree(cov)
    g_auto, _ = ravel_pytree(auto)
    # Full central finite differences are affordable for this tiny model.
    eps = 1e-5
    basis = np.eye(flat.size)
    finite = np.array([(float(exact_energy(unravel(flat + eps * v), states, H))
                        - float(exact_energy(unravel(flat - eps * v), states, H)))
                       / (2 * eps) for v in basis])
    report = {
        "parameter_count": int(flat.size), "finite_difference_step": eps,
        "covariance_vs_autodiff_max_abs": float(jnp.max(jnp.abs(g_cov - g_auto))),
        "covariance_vs_finite_difference_max_abs": float(np.max(np.abs(g_cov - finite))),
        "exact_physical_gradient_norm": float(jnp.linalg.norm(g_auto)),
        "normalization_gauge_gradient": float(cov["c"]),
    }
    arrays = {"parameters": np.asarray(flat), "covariance_gradient": np.asarray(g_cov),
              "autodiff_gradient": np.asarray(g_auto), "finite_difference_gradient": finite}
    return report, arrays


def run(args: argparse.Namespace) -> dict:
    if not 3 <= args.n <= 10:
        raise ValueError("This exact-enumeration demo supports 3 <= n <= 10.")
    if args.h <= 0 or args.hidden < 1 or args.steps < 1 or args.batch_size < 2:
        raise ValueError("Require h > 0, hidden >= 1, steps >= 1, batch-size >= 2.")
    if args.damping <= 0 or args.learning_rate <= 0 or not 0 <= args.mu < 1:
        raise ValueError("Require positive damping/learning-rate and 0 <= mu < 1.")
    if args.log_every < 1 or args.max_norm <= 0:
        raise ValueError("Require log-every >= 1 and max-norm > 0.")
    if args.output is None:
        args.output = str(Path(__file__).parent / "results" / f"{args.optimizer}_n{args.n}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    states = enumerate_spins(args.n)
    H_np = dense_hamiltonian(args.n, args.J, args.h)
    eigenvalues, eigenvectors = np.linalg.eigh(H_np)
    H = jnp.asarray(H_np)
    ground = jnp.asarray(eigenvectors[:, 0])
    params = init_params(args.seed, args.n, args.hidden)
    gradient_report, gradient_arrays = gradient_diagnostics(params, states, H, args.J, args.h)
    np.savez(out / "gradient_check.npz", **gradient_arrays)
    (out / "gradient_check.json").write_text(json.dumps(gradient_report, indent=2) + "\n")

    optimizer = (make_spring(args.learning_rate, args.damping, args.mu, args.max_norm)
                 if args.optimizer == "spring" else optax.sgd(args.learning_rate))
    optimizer_state = optimizer.init(params)
    key = jax.random.PRNGKey(args.seed + 1)

    @jax.jit
    def step(params, optimizer_state, key):
        key, sample_key = jax.random.split(key)
        # Exact categorical sampling is possible only for this small benchmark.
        # A larger VMC problem would replace enumeration with a Markov chain.
        logits = 2.0 * batch_log_psi(params, states)
        indices = jax.random.categorical(sample_key, logits, shape=(args.batch_size,))
        samples = states[indices]
        physical_gradient = energy_covariance_gradient(params, samples, J=args.J, h=args.h)
        if args.optimizer == "spring":
            # Original SPRING convention: input f = g_physical/2.
            force = jax.tree.map(lambda g: 0.5 * g, physical_gradient)
            updates, new_state = optimizer.update(force, optimizer_state, params, data=samples)
            raw_norm = jnp.linalg.norm(new_state[0].prev_delta)
        else:
            updates, new_state = optimizer.update(physical_gradient, optimizer_state, params)
            raw_norm = jnp.linalg.norm(ravel_pytree(physical_gradient)[0])
        diagnostics = {"sampled_energy_before_update": jnp.mean(local_energies(params, samples, args.J, args.h)),
                       "sampled_physical_gradient_norm": jnp.linalg.norm(ravel_pytree(physical_gradient)[0]),
                       "raw_direction_norm": raw_norm,
                       "parameter_update_norm": jnp.linalg.norm(ravel_pytree(updates)[0])}
        return optax.apply_updates(params, updates), new_state, key, diagnostics

    history = []

    def record(index, diagnostics):
        obs = exact_observables(params, states, H)
        row = {"step": index, "exact_energy": float(obs["energy"]),
               "energy_error": float(obs["energy"] - eigenvalues[0]),
               "exact_variance": float(obs["variance"]),
               "ground_state_fidelity": float(jnp.abs(ground @ obs["psi"])**2)}
        row.update({k: float(v) for k, v in diagnostics.items()})
        history.append(row)

    record(0, {})
    start = time.perf_counter()
    for index in range(1, args.steps + 1):
        params, optimizer_state, key, diagnostics = step(params, optimizer_state, key)
        if index == 1 or index % args.log_every == 0 or index == args.steps:
            record(index, diagnostics)
            if not np.all(np.isfinite(list(history[-1].values()))):
                raise FloatingPointError("Non-finite training diagnostics; reduce the learning rate.")
            if index == 1 or index % 50 == 0 or index == args.steps:
                print(f"step={index:4d} E={history[-1]['exact_energy']:.10f} "
                      f"E-E0={history[-1]['energy_error']:.3e}", flush=True)
    elapsed = time.perf_counter() - start
    fields = list(history[-1])
    with (out / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    np.savez(out / "trained_parameters.npz", **{k: np.asarray(v) for k, v in params.items()})
    result = {
        "configuration": vars(args), "upstream_commit": UPSTREAM_COMMIT,
        "backend": str(jax.devices()), "dtype": str(params["W"].dtype),
        "versions": {p: importlib.metadata.version(p) for p in ("jaqmc", "jax", "jaxlib", "optax", "numpy")},
        "optimizer_module": robust_sr.__module__, "parameter_count": int(ravel_pytree(params)[0].size),
        "exact_ground_energy": float(eigenvalues[0]), "initial": history[0], "final": history[-1],
        "gradient_check": gradient_report,
        "training_seconds_including_jit": elapsed,
        "sampling": "iid categorical draws from enumerated |psi|^2; exact expectations used for evaluation only",
        "spring_force_convention": "physical_gradient/2; SGD uses physical_gradient",
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    make_plots(history, gradient_arrays, out, args.optimizer)
    print(json.dumps({"results": str(out), "E0": result["exact_ground_energy"],
                      "final_energy": history[-1]["exact_energy"],
                      "gradient_check": gradient_report}, indent=2))
    return result


def make_plots(history, gradients, out, optimizer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    steps = [row["step"] for row in history]
    axes[0].semilogy(steps, np.maximum([row["energy_error"] for row in history], 1e-15))
    axes[0].set(xlabel=f"{optimizer.upper()} step", ylabel=r"Exact $E_\theta-E_0$", title="Energy error")
    axes[1].semilogy(steps, np.maximum([row["exact_variance"] for row in history], 1e-15))
    axes[1].set(xlabel=f"{optimizer.upper()} step", ylabel=r"$\langle(H-E)^2\rangle$", title="Energy variance")
    auto = gradients["autodiff_gradient"]
    axes[2].scatter(auto, gradients["finite_difference_gradient"], s=12)
    limits = [auto.min(), auto.max()]
    axes[2].plot(limits, limits, "k--", lw=1)
    axes[2].set(xlabel="Exact autodiff gradient", ylabel="Finite difference gradient", title="Gradient validation")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out / "diagnostics.png", dpi=180)
    fig.savefig(out / "diagnostics.pdf")
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--n", type=int, default=6)
    cli.add_argument("--hidden", type=int, default=12)
    cli.add_argument("--J", type=float, default=1.0)
    cli.add_argument("--h", type=float, default=1.0)
    cli.add_argument("--steps", type=int, default=250)
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--learning-rate", type=float, default=0.05)
    cli.add_argument("--damping", type=float, default=0.001)
    cli.add_argument("--mu", type=float, default=0.9)
    cli.add_argument("--max-norm", type=float, default=0.1)
    cli.add_argument("--seed", type=int, default=7)
    cli.add_argument("--log-every", type=int, default=5)
    cli.add_argument("--optimizer", choices=("spring", "sgd"), default="spring")
    cli.add_argument("--output", default=None)
    return cli


if __name__ == "__main__":
    run(parser().parse_args())
