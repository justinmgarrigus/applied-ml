"""Genuine upstream JaQMC SPRING with frozen Born score rows.

The force is Re Cov(O*, E_local), half the physical energy gradient.
The sampled metric is Re[centered(O)^H centered(O)]/B. No temperature,
importance reweighting, or target state enters this adapter. One process
uses one JAX device; independent Slurm tasks may run independent seeds.
"""
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from jaqmc.optimizer.sr.jasr import robust_sr


def score_data(params, scores):
    rows = jax.vmap(lambda row: ravel_pytree(row)[0])(scores)
    if rows.shape[1] != ravel_pytree(params)[0].size:
        raise ValueError('Score columns must match parameter count')
    return jax.lax.stop_gradient(rows - jnp.mean(rows, axis=0))


def make_spring(*, learning_rate=.01, damping=.01, mu=.9, max_norm=.03):
    def metric_log(params, row):
        return jnp.dot(jax.lax.stop_gradient(row), ravel_pytree(params)[0])
    return robust_sr(metric_log, learning_rate=learning_rate, damping=damping,
                     spring_mu=mu, max_norm=max_norm, max_cond_num=None,
                     robust_gamma=None, march_beta=None, mixed_precision=True,
                     score_in_axes=0, score_chunk_size=None, score_norm_clip=None,
                     gram_num_chunks=None, gram_dot_prec='F64', axis_name=None,
                     prune_inactive=False)


def make_minsr(*, learning_rate=.01, damping=.01, max_norm=.03):
    """Memory-free, finite-damping sample-space SR (MinSR convention).

    With centered complex scores realified as A=[Re O; Im O]/sqrt(B),
    and f=A.T @ epsilon, the raw direction is
    A.T @ solve(A @ A.T + damping*I, epsilon).
    The upstream Woodbury implementation is algebraically identical to
    solve(A.T @ A + damping*I, f). No spectral cutoff/pseudoinverse is used.
    """
    def metric_log(params, row):
        return jnp.dot(jax.lax.stop_gradient(row), ravel_pytree(params)[0])
    return robust_sr(metric_log, learning_rate=learning_rate, damping=damping,
                     spring_mu=None, max_norm=max_norm, max_cond_num=None,
                     robust_gamma=None, march_beta=None, mixed_precision=True,
                     score_in_axes=0, score_chunk_size=None, score_norm_clip=None,
                     gram_num_chunks=None, gram_dot_prec='F64', axis_name=None,
                     prune_inactive=False)


def make_optimizer(method, *, learning_rate=.01, damping=.01, mu=.9, max_norm=.03):
    """Explicit optimizer selection; VMC-SGD consumes the FULL energy gradient."""
    if method == 'spring':
        return make_spring(learning_rate=learning_rate, damping=damping, mu=mu, max_norm=max_norm)
    if method == 'minsr':
        return make_minsr(learning_rate=learning_rate, damping=damping, max_norm=max_norm)
    if method == 'vmc_sgd':
        # Clip the applied SGD update to the same parameter-step radius.
        return optax.chain(optax.sgd(learning_rate), optax.clip_by_global_norm(max_norm))
    raise ValueError('Unknown method; implemented: spring, vmc_sgd, minsr')
