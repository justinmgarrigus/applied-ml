"""Historical one-layer N=6 wavefunction and reweighted SPRING kernels.

This opt-in implementation restores the October N=6 protocol. Its artificial
sampling temperature changes the proposal target, not the physical state.
The current Born-sampling workflows and their checkpoint signatures are left
unchanged. Enable JAX float64 before constructing arrays.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from . import model
from .optimizer import robust_sr
from .sampler import global_refresh as born_refresh, make_sampler as make_born_sampler


def initialize(seed=7, norb=16, width=64):
    """Original random initializer; no ED coefficients enter training."""
    kw, kb, ko = jax.random.split(jax.random.PRNGKey(seed), 3)
    return dict(
        W=.8 / np.sqrt(norb) * jax.random.normal(kw, (norb, width), dtype=jnp.float64),
        b=.1 * jax.random.normal(kb, (width,), dtype=jnp.float64),
        out=jax.random.normal(ko, (width, 2), dtype=jnp.float64)
            * jnp.array([.15, .8])[None, :] / np.sqrt(width),
        bias=jnp.zeros(2, dtype=jnp.float64),
    )


def sampling_functions(ne, flux, temperature=2.):
    """Same pair moves as Born sampling, targeting |psi|**(2/T)."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Sampling temperature must be positive and finite')

    def tempered_log(params, occupation):
        return model.log_psi(params, occupation) / temperature

    sampler = make_born_sampler(tempered_log, ne, flux)

    def refresh(params, walkers, proposals, key):
        return born_refresh(tempered_log, params, walkers, proposals, key)

    return sampler, refresh


def importance_weights(params, walkers, temperature=2.):
    """Self-normalized p/q weights, with finite-batch ratio-estimator bias."""
    logs = 2. * (1. - 1. / temperature) * model.batch_log_psi(params, walkers).real
    return jax.nn.softmax(logs)


def metric_data(params, walkers, weights):
    """Original signed paired-quadrature rows and weighted centering."""
    weights = jax.lax.stop_gradient(weights / jnp.sum(weights))
    mean = jax.tree.map(lambda score: jnp.tensordot(weights, score, axes=(0, 0)),
                        model.score_tree(params, walkers))
    mean = jax.lax.stop_gradient(ravel_pytree(mean)[0])
    scales = jnp.sqrt(walkers.shape[0] * weights)
    return dict(occupation=jnp.concatenate((walkers, walkers)),
                scale=jnp.concatenate((scales, -scales)),
                mean_score=jnp.broadcast_to(mean, (2 * walkers.shape[0], mean.size)))


def make_spring(*, learning_rate=.03, damping=.001, mu=.9, max_norm=.1):
    """Historical JaQMC adapter; max_norm is sqrt(C), the update radius."""
    def metric_log(params, row):
        return row['scale'] * (model.log_psi(params, row['occupation'])
                               - row['mean_score'] @ ravel_pytree(params)[0])

    return robust_sr(metric_log, learning_rate=learning_rate, damping=damping,
                     spring_mu=mu, max_norm=max_norm, max_cond_num=None,
                     robust_gamma=None, march_beta=None, mixed_precision=True,
                     score_norm_clip=None, gram_num_chunks=None, gram_dot_prec='F64',
                     prune_inactive=False)


def weighted_force(params, walkers, local_values, weights):
    """Re Cov(O*, E_local), half the physical energy gradient."""
    residual = local_values - weights @ local_values
    return jax.tree.map(lambda score: jnp.tensordot(
        weights * residual, score.conj(), axes=(0, 0)).real,
        model.score_tree(params, walkers))


def build_step(config, sampler, local, optimizer):
    """A counted update; retain the original unused-pair RNG split."""
    import optax

    @jax.jit
    def step(params, state, walkers, key):
        walkers, key, acceptance = sampler(params, walkers, key, config['sweeps'])
        # The old builder split off an unused random_pairs key. The split
        # affects subsequent walker draws and is part of the old RNG stream.
        key, _unused_pair_key = jax.random.split(key)
        weights = importance_weights(params, walkers, config['temperature'])
        values = local(params, walkers)['energy']
        force = weighted_force(params, walkers, values, weights)
        updates, state = optimizer.update(force, state, params,
                                          data=metric_data(params, walkers, weights))
        direction = state[0].prev_delta
        norm = jnp.linalg.norm(direction)
        scale = jnp.minimum(config['learning_rate'], config['max_norm'] / (norm + 1e-8))
        update = ravel_pytree(updates)[0]
        mean_energy = weights @ values
        diagnostics = dict(energy_estimate=mean_energy.real,
                           energy_imaginary_estimate=mean_energy.imag,
                           importance_ess=1. / jnp.sum(weights**2), acceptance=acceptance,
                           force_norm=jnp.linalg.norm(ravel_pytree(force)[0]),
                           raw_direction_norm=norm, effective_step_scale=scale,
                           cap_active=config['max_norm'] / (norm + 1e-8) < config['learning_rate'],
                           update_norm=jnp.linalg.norm(update),
                           update_scale_consistency_error=jnp.linalg.norm(update + scale * direction))
        return optax.apply_updates(params, updates), state, walkers, key, diagnostics

    return step
