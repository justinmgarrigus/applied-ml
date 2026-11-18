"""Isolated N=7 Geometry-Guided SPRING pilot; no ED data enter training.

Compare ordinary Born-sampled SPRING, uniform subsampling of a Born candidate
pool, and diagonal-QGT-whitened novelty selection from the same-size pool.
The latter uses full-support probabilities and self-normalized inverse
selection weights, retaining finite-sample ratio/covariance bias. Persistent
MCMC walkers are never replaced by the adaptively selected configurations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def run(config):
    from .train import _set_runtime, digest
    _set_runtime(config)
    if config['backend'] == 'gpu':
        os.environ['JAX_PLATFORMS'] = 'cuda'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    import numpy as np
    import optax
    from . import model, ne6_baseline
    from .geometry_sampling import (geometry_novelty, selection_probabilities,
                                    select_candidates, weighted_score_data)
    from .physics import SphereOperator
    from .sampler import make_sampler, global_refresh
    from .local_energy import make_local_energy
    from .optimizer import make_spring
    from .checkpoint import atomic_json, atomic_npz, save_checkpoint

    output = Path(config['output'])
    if output.exists():
        raise FileExistsError(f'Preserving existing run: {output}')
    method, batch = config['method'], config['batch']
    if method not in ('spring', 'pool_random', 'gg_spring'):
        raise ValueError('Unknown method')
    if config['steps'] < 1 or batch < 2 or config['pool'] < batch:
        raise ValueError('Require positive steps and pool >= batch >= 2')
    operator = SphereOperator(7, flux=18, v3=0., two_lz=0)
    pool_size = batch if method == 'spring' else config['pool']
    params = model.prepare(ne6_baseline.initialize(config['init_seed'], norb=19, width=64),
                           norb=19, width=64, layers=6, kind='mlp',
                           seed=config['init_seed'], readout='legacy_additive')
    flat, unravel = ravel_pytree(params)
    if flat.size != 22850 or operator.dimension() != 1656:
        raise AssertionError('Unexpected N=7 model/sector')
    optimizer = make_spring(learning_rate=config['learning_rate'], damping=config['damping'],
                            mu=config['mu'], max_norm=config['max_norm'])
    state = optimizer.init(params)
    sampler = make_sampler(model.log_psi, 7, 18)
    local = make_local_energy(operator, model.log_psi, pair_chunk_size=2)

    def score_rows(parameters, occupations):
        tree = model.score_tree(parameters, occupations)
        return jax.vmap(lambda row: ravel_pytree(row)[0])(tree)

    @jax.jit
    def step(parameters, opt_state, walkers, anchor, mc_key, selection_key):
        walkers, mc_key, acceptance = sampler(parameters, walkers, mc_key, config['sweeps'])
        selection_key, draw_key = jax.random.split(selection_key)
        novelty = jnp.zeros(pool_size, dtype=jnp.float64)
        if method == 'gg_spring':
            combined = score_rows(parameters, jnp.concatenate((walkers, anchor)))
            pool_scores, anchor_scores = combined[:pool_size], combined[pool_size:]
            novelty = geometry_novelty(pool_scores, anchor_scores,
                                       diagonal_floor=config['diagonal_floor'],
                                       ridge=config['geometry_ridge'])
            probabilities = selection_probabilities(novelty, mix=config['selection_mix'],
                                                    novelty_floor=config['novelty_floor'])
            selected = select_candidates(draw_key, probabilities, batch)
            indices, weights = selected['indices'], selected['weights']
            occupations, scores = walkers[indices], pool_scores[indices]
        elif method == 'pool_random':
            probabilities = jnp.full(pool_size, 1. / pool_size)
            selected = select_candidates(draw_key, probabilities, batch)
            indices, weights = selected['indices'], selected['weights']
            occupations = walkers[indices]
            scores = score_rows(parameters, occupations)
        else:
            probabilities = jnp.full(pool_size, 1. / pool_size)
            indices, weights = jnp.arange(batch), jnp.full(batch, 1. / batch)
            occupations = walkers
            scores = score_rows(parameters, occupations)

        energies = local(parameters, occupations)['energy']
        mean_energy = weights @ energies
        residual = energies - mean_energy
        force_flat = jnp.real((weights * residual) @ scores.conj())
        updates, opt_state = optimizer.update(unravel(force_flat), opt_state, parameters,
                                              data=weighted_score_data(scores, weights))
        direction = opt_state[0].prev_delta
        norm = jnp.linalg.norm(direction)
        scale = jnp.minimum(config['learning_rate'], config['max_norm'] / (norm + 1e-8))
        flat_update = ravel_pytree(updates)[0]
        same_index = indices[:, None] == indices[None, :]
        same_configuration = jnp.all(occupations[:, None, :] == occupations[None, :, :], axis=-1)
        row = dict(energy_estimate=mean_energy.real, energy_imaginary_estimate=mean_energy.imag,
                   local_energy_variance=weights @ jnp.abs(residual)**2,
                   acceptance=acceptance, importance_ess=1. / jnp.sum(weights**2),
                   novelty_mean=jnp.mean(novelty), selected_novelty_mean=jnp.mean(novelty[indices]),
                   probability_min=jnp.min(probabilities), probability_max=jnp.max(probabilities),
                   selection_probability_ess=1. / jnp.sum(probabilities**2),
                   selected_distinct_indices=jnp.sum(1. / jnp.sum(same_index, axis=1)),
                   selected_distinct_configurations=jnp.sum(1. / jnp.sum(same_configuration, axis=1)),
                   force_norm=jnp.linalg.norm(force_flat), raw_direction_norm=norm,
                   effective_step_scale=scale, cap_active=(scale < config['learning_rate']),
                   update_norm=jnp.linalg.norm(flat_update),
                   update_scale_consistency_error=jnp.linalg.norm(flat_update + scale * direction))
        return (optax.apply_updates(parameters, updates), opt_state, walkers, occupations,
                mc_key, selection_key, row)

    rng = np.random.default_rng(config['seed'])
    walkers = jnp.asarray(operator.sample_uniform(rng, pool_size), dtype=jnp.float64)
    mc_key = jax.random.PRNGKey(config['seed'])
    selection_key = jax.random.PRNGKey(config['seed'] + 100000)
    burn_start = time.perf_counter()
    walkers, mc_key, acceptance = sampler(params, walkers, mc_key, config['burn_sweeps'])
    jax.block_until_ready((walkers, mc_key, acceptance))
    burn_seconds = time.perf_counter() - burn_start
    anchor = walkers[:batch]
    compile_start = time.perf_counter()
    compiled = step.lower(params, state, walkers, anchor, mc_key, selection_key).compile()
    # Execute the compiled warm-up without advancing the counted trajectory.
    warm = compiled(params, state, walkers, anchor, mc_key, selection_key)
    jax.block_until_ready(warm)
    del warm
    compile_seconds = time.perf_counter() - compile_start
    source_names = ('train_geometry', 'geometry_sampling', 'ne6_baseline', 'model',
                    'legacy_model', 'optimizer', 'physics', 'local_energy', 'sampler', 'checkpoint')
    hashes = {name: digest(Path(__file__).parent / f'{name}.py') for name in source_names}
    import inspect
    from .optimizer import robust_sr
    hashes['jaqmc/optimizer/sr/jasr.py'] = digest(inspect.getfile(robust_sr))
    signature = hashlib.sha256(json.dumps(dict(config=config, sources=hashes),
                                         sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    atomic_npz(output / 'initial_parameters.npz', **params)
    atomic_json(output / 'run_config.json', dict(
        configuration=config, config_sha256=signature, implementation_sha256=hashes,
        architecture=model.architecture(params), sector_dimension=1656,
        physics='N=7, flux18, Lz=0, LLL sphere, pure normalized V1=1',
        sampling='Born MCMC pool; independent selection RNG; persistent pool not resampled by selection',
        selection_weights='self-normalized inverse selection probabilities; finite-batch ratio/covariance bias',
        geometry='diagonal-QGT whitening and ridge-complement novelty against previous selected batch at current theta',
        metric='weighted covariance encoded by signed duplicate rows in upstream JaQMC SPRING',
        no_ed_data_in_training=True, source=None, device=str(jax.devices()[0]),
        versions={n: importlib.metadata.version(n) for n in ('jax', 'jaxlib', 'numpy', 'scipy', 'optax')},
        burn_seconds=burn_seconds, compile_seconds=compile_seconds))
    started = time.perf_counter()
    row = {}
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = None
        for update in range(1, config['steps'] + 1):
            if config['refresh_every'] and update % config['refresh_every'] == 0:
                proposals = jnp.asarray(operator.sample_uniform(rng, pool_size), dtype=jnp.float64)
                walkers, mc_key, _ = global_refresh(model.log_psi, params, walkers, proposals, mc_key)
            params, state, walkers, anchor, mc_key, selection_key, diagnostics = compiled(
                params, state, walkers, anchor, mc_key, selection_key)
            if update == 1 or update % config['log_every'] == 0 or update == config['steps']:
                values = jax.device_get(diagnostics)
                row = dict(step=update, training_seconds=time.perf_counter() - started,
                           **{name: float(value) for name, value in values.items()})
                if not all(np.isfinite(value) for value in row.values()):
                    raise ArithmeticError(f'Nonfinite diagnostics at update {update}')
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
            if update % config['checkpoint_every'] == 0 or update == config['steps']:
                jax.block_until_ready((params, state))
                elapsed = time.perf_counter() - started
                checkpoint = output / 'checkpoints' / f'step_{update:012d}.npz'
                # Anchor and separate selection key are saved as extra state;
                # these experimental checkpoints are not production resumes.
                save_checkpoint(checkpoint, params=params,
                                state=(state, anchor, selection_key), walkers=walkers,
                                key=mc_key, rng_state=rng.bit_generator.state,
                                step=update, training_seconds=elapsed, config_sha256=signature)
                print(f'{method} seed={config["seed"]} step={update} '
                      f'MC_E={row.get("energy_estimate", float("nan")):.8g} '
                      f'ESS={row.get("importance_ess", 0):.2f} t={elapsed:.2f}s', flush=True)
    jax.block_until_ready((params, state))
    atomic_json(output / 'training_summary.json', dict(
        status='completed', completed_steps=config['steps'], method=method,
        seed=config['seed'], init_seed=config['init_seed'], parameter_count=int(flat.size),
        training_seconds=time.perf_counter() - started, compile_seconds=compile_seconds,
        burn_seconds=burn_seconds, last_metrics=row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
