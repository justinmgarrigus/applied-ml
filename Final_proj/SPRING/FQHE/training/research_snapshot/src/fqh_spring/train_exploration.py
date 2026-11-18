"""Full-state N=7 GG-SPRING branches from immutable wall-study checkpoints.

The physical energy objective and six-layer all-readout model are unchanged.
Only candidate sampling and selection differ. Hot pools use q_beta proportional
to |psi|**(2*beta) and both proposal and selection importance corrections.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time


def make_components(config, params):
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    import optax
    from . import model
    from .physics import SphereOperator
    from .sampler import make_sampler
    from .local_energy import make_local_energy
    from .north_droplet import profile_a_epsilon, make_confinement_local_energy
    from .optimizer import make_spring
    from .geometry_sampling import (geometry_novelty, selection_probabilities,
                                    select_candidates, weighted_score_data)
    from .exploration_sampling import select_static_tempered, select_sequential

    method, batch, pool_size, beta = (config['method'], config['batch'],
                                     config['pool'], config['beta'])
    _, unravel = ravel_pytree(params)
    optimizer = make_spring(learning_rate=config['learning_rate'], damping=config['damping'],
                            mu=config['mu'], max_norm=config['max_norm'])
    log_fn = model.log_psi if beta == 1. else lambda p, n: beta * model.log_psi(p, n)
    sampler = make_sampler(log_fn, 7, 18)
    operator = SphereOperator(7, flux=18, v3=0., two_lz=0)
    epsilon = profile_a_epsilon(18, config['confinement_strength'], config['wall_x'])
    local = make_confinement_local_energy(
        make_local_energy(operator, model.log_psi, pair_chunk_size=2), epsilon)

    def score_rows(parameters, occupations):
        return jax.vmap(lambda row: ravel_pytree(row)[0])(
            model.score_tree(parameters, occupations))

    @jax.jit
    def step(parameters, opt_state, walkers, anchor, mc_key, selection_key):
        walkers, mc_key, acceptance = sampler(parameters, walkers, mc_key, config['sweeps'])
        selection_key, draw_key = jax.random.split(selection_key)
        combined = score_rows(parameters, jnp.concatenate((walkers, anchor)))
        pool_scores, anchor_scores = combined[:pool_size], combined[pool_size:]
        log_amplitudes = model.batch_log_psi(parameters, walkers).real
        if method == 'gg_original':
            novelty = geometry_novelty(pool_scores, anchor_scores,
                                       diagonal_floor=config['diagonal_floor'],
                                       ridge=config['geometry_ridge'])
            probabilities = selection_probabilities(novelty, mix=config['selection_mix'],
                                                    novelty_floor=config['novelty_floor'])
            selected = select_candidates(draw_key, probabilities, batch)
            indices, weights = selected['indices'], selected['weights']
            novelty_mean = jnp.mean(novelty)
            selected_novelty = jnp.mean(novelty[indices])
            probability_min, probability_max = jnp.min(probabilities), jnp.max(probabilities)
        else:
            selector = select_sequential if method == 'gg_sequential' else select_static_tempered
            selected = selector(draw_key, pool_scores, anchor_scores, batch,
                                beta=beta, log_amplitudes=log_amplitudes,
                                mix=config['selection_mix'], floor=config['novelty_floor'],
                                diagfloor=config['diagonal_floor'], ridge=config['geometry_ridge'])
            indices, weights = selected['indices'], selected['weights']
            novelty_mean, selected_novelty = selected['novelty_mean'], selected['selected_novelty_mean']
            probability_min, probability_max = selected['probability_min'], selected['probability_max']
        occupations, scores = walkers[indices], pool_scores[indices]
        local_values = local(parameters, occupations)
        energies = local_values['energy']
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
        pool_weights = jax.nn.softmax(2 * (1 - beta) * log_amplitudes)
        row = dict(energy_estimate=mean_energy.real, energy_imaginary_estimate=mean_energy.imag,
                   local_energy_variance=weights @ jnp.abs(residual)**2,
                   interaction_energy_estimate=(weights @ local_values['interaction_energy']).real,
                   wall_energy_estimate=(weights @ local_values['confinement_energy']).real,
                   acceptance=acceptance, importance_ess=1. / jnp.sum(weights**2),
                   pool_importance_ess=1. / jnp.sum(pool_weights**2),
                   novelty_mean=novelty_mean, selected_novelty_mean=selected_novelty,
                   probability_min=probability_min, probability_max=probability_max,
                   selected_distinct_indices=jnp.sum(1. / jnp.sum(same_index, axis=1)),
                   selected_distinct_configurations=jnp.sum(1. / jnp.sum(same_configuration, axis=1)),
                   force_norm=jnp.linalg.norm(force_flat), raw_direction_norm=norm,
                   effective_step_scale=scale, cap_active=(scale < config['learning_rate']),
                   update_norm=jnp.linalg.norm(flat_update),
                   update_scale_consistency_error=jnp.linalg.norm(flat_update + scale * direction))
        return (optax.apply_updates(parameters, updates), opt_state, walkers, occupations,
                mc_key, selection_key, row)

    return dict(step=step, optimizer=optimizer, sampler=sampler, log_fn=log_fn,
                operator=operator, local=local, unravel=unravel, score_rows=score_rows,
                legacy_scope=locals())


def read_parent(config):
    """Verify the original full-state scientific signature before branching."""
    from .train import digest
    path = Path(config['parent_checkpoint']).resolve()
    if digest(path) != config['parent_checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    metadata = json.loads((path.parent.parent / 'run_config.json').read_text())
    parent_config, sources = metadata['configuration'], metadata['implementation_sha256']
    expected = hashlib.sha256(json.dumps(dict(config=parent_config, sources=sources),
                                         sort_keys=True).encode()).hexdigest()
    if expected != metadata['config_sha256']:
        raise ValueError('Parent scientific signature is invalid')
    project = Path(__file__).resolve().parents[2]
    for name, checksum in sources.items():
        source = (project / 'vendor/jaqmc_spring/src' / name if name.startswith('jaqmc/')
                  else Path(__file__).parent / f'{name}.py')
        if digest(source) != checksum:
            raise ValueError(f'Parent implementation changed: {name}')
    for name in ('ne', 'flux', 'two_lz', 'v3', 'width', 'layers', 'readout', 'batch', 'pool',
                 'sweeps', 'learning_rate', 'damping', 'mu', 'max_norm', 'refresh_every',
                 'diagonal_floor', 'geometry_ridge', 'selection_mix', 'novelty_floor',
                 'confinement_strength', 'wall_x', 'seed'):
        if config[name] != parent_config[name]:
            raise ValueError(f'Continuation changes fixed setting {name}')
    return path, metadata


def run(config):
    from .train import _set_runtime, digest
    _set_runtime(config)
    if config['backend'] == 'gpu':
        os.environ['JAX_PLATFORMS'] = 'cuda'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from . import model
    from .evaluate import read_parameters
    from .checkpoint import load_checkpoint, save_checkpoint, atomic_json, atomic_npz
    from .sampler import global_refresh

    if config['method'] not in ('gg_original', 'gg_tempered', 'gg_sequential'):
        raise ValueError('Unknown exploration method')
    expected_beta = 1. if config['method'] == 'gg_original' else .5
    if config['beta'] != expected_beta:
        raise ValueError('Pilot beta was predeclared as 1 or .5 by method')
    parent_path, parent = read_parent(config)
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError(f'Preserving existing branch {output}')
    params = {name: jnp.asarray(value) for name, value in read_parameters(parent_path).items()}
    components = make_components(config, params)
    batch = config['batch']
    template = (components['optimizer'].init(params),
                jnp.zeros((batch, 19), dtype=jnp.float64), jax.random.PRNGKey(0))
    saved = load_checkpoint(parent_path, template, expected_config_sha256=parent['config_sha256'])
    if saved['step'] != config['first_step'] or config['target_step'] <= saved['step']:
        raise ValueError('Unexpected starting or target step')
    params, (state, anchor, selection_key), walkers, mc_key = (
        saved['params'], saved['state'], saved['walkers'], saved['key'])
    restored_counts = dict(spring=int(np.asarray(state[0].counter)),
                           learning_rate=int(np.asarray(state[1].count)))
    if any(value != saved['step'] for value in restored_counts.values()):
        raise ValueError('Optimizer counters do not match the saved parent step')
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['rng_state']
    # Only the sampling pool and MC RNG evolve during the frozen hot warmup.
    warmup_start = time.perf_counter()
    if config['beta'] != 1.:
        walkers, mc_key, _ = components['sampler'](params, walkers, mc_key, config['hot_warmup_sweeps'])
        jax.block_until_ready((walkers, mc_key))
    warmup_seconds = time.perf_counter() - warmup_start
    compile_start = time.perf_counter()
    compiled = components['step'].lower(params, state, walkers, anchor, mc_key, selection_key).compile()
    warm = compiled(params, state, walkers, anchor, mc_key, selection_key)
    jax.block_until_ready(warm)
    del warm
    compile_seconds = time.perf_counter() - compile_start
    source_names = ('train_exploration', 'exploration_sampling', 'geometry_sampling', 'model',
                    'legacy_model', 'optimizer', 'physics', 'local_energy', 'north_droplet',
                    'sampler', 'checkpoint')
    sources = {name: digest(Path(__file__).parent / f'{name}.py') for name in source_names}
    from .optimizer import robust_sr
    sources['jaqmc/optimizer/sr/jasr.py'] = digest(inspect.getfile(robust_sr))
    signature = hashlib.sha256(json.dumps(dict(config=config, sources=sources), sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    (output / 'checkpoints').mkdir()
    shutil.copyfile(parent_path, output / 'checkpoints' / f'step_{saved["step"]:012d}.npz')
    atomic_npz(output / 'initial_parameters.npz', **params)
    atomic_json(output / 'run_config.json', dict(
        configuration=config, config_sha256=signature, implementation_sha256=sources,
        architecture=model.architecture(params), sector_dimension=1656,
        orbital_profile_sha256=parent['orbital_profile_sha256'],
        parent_checkpoint=str(parent_path), parent_checkpoint_sha256=digest(parent_path),
        parent_config_sha256=parent['config_sha256'], parent_step=saved['step'],
        restored_optimizer_counters=restored_counts,
        parent_training_seconds=saved['training_seconds'],
        restored_state='Parameters, SPRING direction/memory/counters, anchors, pool and both JAX RNGs plus NumPy RNG',
        sampling='q_beta proportional to |psi|^(2 beta); hot pool warmed at fixed parent parameters',
        weighting='Both p/q_beta and conditional inverse-index selection weights; self-normalized finite-batch ratio/covariance bias remains',
        geometry='Importance-centered diagonal Born metric; sequential arm projects out newly selected directions',
        no_ed_data_in_training=True, compile_seconds=compile_seconds, warmup_seconds=warmup_seconds,
        burn_seconds=warmup_seconds, device=str(jax.devices()[0])))
    start = time.perf_counter()
    last_row = {}
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = None
        for update in range(saved['step'] + 1, config['target_step'] + 1):
            if config['refresh_every'] and update % config['refresh_every'] == 0:
                proposals = jnp.asarray(components['operator'].sample_uniform(rng, config['pool']), dtype=jnp.float64)
                walkers, mc_key, _ = global_refresh(components['log_fn'], params, walkers, proposals, mc_key)
            params, state, walkers, anchor, mc_key, selection_key, values = compiled(
                params, state, walkers, anchor, mc_key, selection_key)
            if update == saved['step'] + 1 or update % config['log_every'] == 0 or update == config['target_step']:
                data = jax.device_get(values)
                last_row = dict(step=update, training_seconds=time.perf_counter() - start,
                                **{name: float(value) for name, value in data.items()})
                if not all(np.isfinite(value) for value in last_row.values()):
                    raise ArithmeticError('Nonfinite continuation diagnostics')
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(last_row))
                    writer.writeheader()
                writer.writerow(last_row)
                stream.flush()
            if update % config['checkpoint_every'] == 0 or update == config['target_step']:
                jax.block_until_ready((params, state))
                elapsed = time.perf_counter() - start
                save_checkpoint(output / 'checkpoints' / f'step_{update:012d}.npz',
                                params=params, state=(state, anchor, selection_key), walkers=walkers,
                                key=mc_key, rng_state=rng.bit_generator.state, step=update,
                                training_seconds=saved['training_seconds'] + elapsed, config_sha256=signature)
                print(f'{config["method"]} seed={config["seed"]} total_step={update} '
                      f'MC_E={last_row["energy_estimate"]:.8g} ESS={last_row["importance_ess"]:.2f} '
                      f'new_seconds={elapsed:.2f}', flush=True)
    jax.block_until_ready((params, state))
    atomic_json(output / 'training_summary.json', dict(
        status='completed', completed_steps=config['target_step'], first_step=saved['step'],
        additional_steps=config['target_step'] - saved['step'], method=config['method'], seed=config['seed'],
        training_seconds=time.perf_counter() - start, compile_seconds=compile_seconds,
        warmup_seconds=warmup_seconds, burn_seconds=warmup_seconds, parameter_count=22850,
        parent_checkpoint_sha256=digest(parent_path), last_metrics=last_row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
