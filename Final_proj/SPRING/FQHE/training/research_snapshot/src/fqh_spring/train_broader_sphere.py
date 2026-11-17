"""Matrix-free sphere SPRING with a broader proposal and no GG selection.

Every walker enters the update. q_beta is proportional to |psi|**(2*beta),
while p/q importance weights retain the physical Born energy objective.
Self-normalized covariance estimates retain finite-batch bias. No ED matrix,
target coefficients, candidate scoring, geometric selection or anchor is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import time

DEFAULTS = dict(ne=12, flux=33, two_lz=0, v3=0., width=64, layers=6,
    variant='mlp', readout='legacy_additive', source=None, beta=.5,
    confinement_strength=10., wall_x=.9345454545454546,
    batch=128, sweeps=2, burn_sweeps=500, refresh_every=25,
    learning_rate=.01, damping=.01, mu=.9, max_norm=.03,
    seed=71, init_seed=7, steps=10000, max_runtime_seconds=300.,
    pair_chunk=2, log_every=50, checkpoint_every=500,
    backend='gpu', cpus=4, native_threads=1,
    output='runs/ne12_broader_seed71', wall_scaling=None, description=None)
OPERATIONAL = {'output', 'steps', 'max_runtime_seconds', 'log_every',
               'checkpoint_every', 'backend', 'cpus', 'native_threads'}


def validate_config(supplied):
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    config = DEFAULTS | supplied
    for key in ('ne', 'flux', 'width', 'layers', 'batch', 'sweeps', 'steps',
                'pair_chunk', 'log_every', 'checkpoint_every', 'native_threads'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'init_seed', 'burn_sweeps', 'refresh_every'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if config['ne'] < 4 or config['ne'] > config['flux'] + 1:
        raise ValueError('Require 4 <= ne <= flux+1')
    if not isinstance(config['two_lz'], int) or isinstance(config['two_lz'], bool):
        raise ValueError('two_lz must be an integer')
    for key in ('learning_rate', 'damping', 'max_norm'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not 0 < config['beta'] <= 1 or not 0 <= config['mu'] < 1:
        raise ValueError('Require 0 < beta <= 1 and 0 <= mu < 1')
    if config['variant'] != 'mlp' or config['readout'] != 'legacy_additive' or config['source'] is not None:
        raise ValueError('This experiment requires a fresh additive-readout MLP (source=null)')
    if config['backend'] not in ('cpu', 'gpu', 'auto'):
        raise ValueError('backend must be cpu, gpu or auto')
    if config['cpus'] is not None and (not isinstance(config['cpus'], int) or config['cpus'] < 1):
        raise ValueError('cpus must be positive or null')
    if not math.isfinite(config['v3']):
        raise ValueError('v3 must be finite')
    maximum = config['max_runtime_seconds']
    if maximum is not None and (not math.isfinite(maximum) or maximum <= 0):
        raise ValueError('max_runtime_seconds must be positive or null')
    from .north_droplet import validate_cap_wall
    validate_cap_wall(config['confinement_strength'], config['wall_x'])
    return config


def load_config(path):
    path = Path(path).resolve()
    config = validate_config(json.loads(path.read_text()))
    output = Path(config['output']).expanduser()
    config['output'] = str(output if output.is_absolute() else (path.parent / output).resolve())
    return config


def importance_weights(log_amplitudes, beta):
    """Stable normalized p/q_beta factors for equal-weight q_beta draws."""
    import jax
    import jax.numpy as jnp
    if not 0 < beta <= 1:
        raise ValueError('beta must be in (0,1]')
    amplitudes = jnp.asarray(log_amplitudes)
    if amplitudes.ndim != 1 or amplitudes.size < 1:
        raise ValueError('log_amplitudes must be a nonempty vector')
    return jax.nn.softmax(2. * (1. - beta) * amplitudes)


def weighted_statistics(scores, energies, weights):
    """Physical half-gradient from physical log scores, never beta*score."""
    import jax.numpy as jnp
    mean = weights @ energies
    residual = energies - mean
    return dict(energy=mean, residual=residual,
                variance=weights @ jnp.abs(residual)**2,
                force=jnp.real((weights * residual) @ scores.conj()))


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
    from .geometry_sampling import weighted_score_data

    operator = SphereOperator(config['ne'], flux=config['flux'], v3=config['v3'], two_lz=config['two_lz'])
    epsilon = profile_a_epsilon(config['flux'], config['confinement_strength'], config['wall_x'])
    local = make_confinement_local_energy(make_local_energy(operator, model.log_psi,
                                        pair_chunk_size=config['pair_chunk']), epsilon)
    log_fn = lambda p, n: config['beta'] * model.log_psi(p, n)
    sampler = make_sampler(log_fn, config['ne'], config['flux'])
    optimizer = make_spring(learning_rate=config['learning_rate'], damping=config['damping'],
                            mu=config['mu'], max_norm=config['max_norm'])
    _, unravel = ravel_pytree(params)

    def score_rows(parameters, occupations):
        return jax.vmap(lambda row: ravel_pytree(row)[0])(model.score_tree(parameters, occupations))

    @jax.jit
    def step(parameters, state, walkers, key):
        walkers, key, acceptance = sampler(parameters, walkers, key, config['sweeps'])
        scores = score_rows(parameters, walkers)
        weights = importance_weights(model.batch_log_psi(parameters, walkers).real, config['beta'])
        values = local(parameters, walkers)
        statistics = weighted_statistics(scores, values['energy'], weights)
        force = statistics['force']
        updates, state = optimizer.update(unravel(force), state, parameters,
                                          data=weighted_score_data(scores, weights))
        direction = state[0].prev_delta
        raw_norm = jnp.linalg.norm(direction)
        scale = jnp.minimum(config['learning_rate'], config['max_norm'] / (raw_norm + 1e-8))
        flat_update = ravel_pytree(updates)[0]
        same = jnp.all(walkers[:, None, :] == walkers[None, :, :], axis=-1)
        wall = values.get('confinement_energy', jnp.zeros_like(values['energy']))
        interaction = values.get('interaction_energy', values['energy'])
        metrics = dict(energy_estimate=statistics['energy'].real,
            energy_imaginary_estimate=statistics['energy'].imag,
            local_energy_variance=statistics['variance'],
            interaction_energy_estimate=(weights @ interaction).real,
            wall_energy_estimate=(weights @ wall).real,
            L2_estimate=(weights @ values['L2']).real,
            acceptance=acceptance, importance_ess=1. / jnp.sum(weights**2),
            maximum_importance_weight=jnp.max(weights),
            distinct_configurations=jnp.sum(1. / jnp.sum(same, axis=1)),
            force_norm=jnp.linalg.norm(force), raw_direction_norm=raw_norm,
            effective_step_scale=scale, cap_active=scale < config['learning_rate'],
            update_norm=jnp.linalg.norm(flat_update),
            update_scale_consistency_error=jnp.linalg.norm(flat_update + scale * direction))
        return optax.apply_updates(parameters, updates), state, walkers, key, metrics

    return dict(operator=operator, epsilon=epsilon, local=local, log_fn=log_fn,
                sampler=sampler, optimizer=optimizer, score_rows=score_rows, step=step)


def scientific_signature(config):
    from .train import digest
    from .optimizer import robust_sr
    names = ('train_broader_sphere', 'train', 'model', 'base_model', 'legacy_model',
             'structured_model', 'ne6_baseline', 'physics', 'sampler', 'local_energy',
             'north_droplet', 'geometry_sampling', 'optimizer', 'checkpoint')
    hashes = {f'fqh_spring/{name}.py': digest(Path(__file__).parent / f'{name}.py') for name in names}
    hashes['jaqmc/optimizer/sr/jasr.py'] = digest(inspect.getfile(robust_sr))
    invariant = {key: value for key, value in config.items() if key not in OPERATIONAL}
    signature = hashlib.sha256(json.dumps(dict(configuration=invariant, implementation_sha256=hashes),
                                         sort_keys=True).encode()).hexdigest()
    return signature, hashes


def run(config, *, resume=False):
    config = validate_config(config)
    from .train import _set_runtime, _trim_csv
    _set_runtime(config)
    if config['backend'] == 'gpu':
        os.environ['JAX_PLATFORMS'] = 'cuda'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from . import model
    from .checkpoint import atomic_json, atomic_npz, save_checkpoint, load_checkpoint
    from .sampler import global_refresh
    from .north_droplet import profile_sha256
    output = Path(config['output'])
    if resume:
        if not (output / 'checkpoint_latest.npz').is_file():
            raise FileNotFoundError(f'No full resume checkpoint in {output}')
    elif output.exists():
        raise FileExistsError(f'Preserving existing output {output}; choose a new path or --resume')
    signature, hashes = scientific_signature(config)
    initial = model.initialize(config['flux'] + 1, width=config['width'], layers=config['layers'],
                               kind=config['variant'], seed=config['init_seed'], readout=config['readout'])
    components = make_components(config, initial)
    state = components['optimizer'].init(initial)
    rng = np.random.default_rng(config['seed'])
    params, first, previous_seconds = initial, 0, 0.
    burn_seconds = 0.
    if resume:
        old = json.loads((output / 'run_config.json').read_text())
        if old['config_sha256'] != signature:
            raise ValueError('Resume changed a scientific setting or implementation source')
        saved = load_checkpoint(output / 'checkpoint_latest.npz', state, expected_config_sha256=signature)
        params, state, walkers, key = saved['params'], saved['state'], saved['walkers'], saved['key']
        rng.bit_generator.state = saved['rng_state']
        first, previous_seconds = saved['step'], saved['training_seconds']
        if int(np.asarray(state[0].counter)) != first or int(np.asarray(state[1].count)) != first:
            raise ValueError('Restored optimizer counters do not match checkpoint step')
        if (output / 'metrics.csv').is_file() and (output / 'metrics.csv').stat().st_size:
            _trim_csv(output / 'metrics.csv', first)
    else:
        walkers = jnp.asarray(components['operator'].sample_uniform(rng, config['batch']), dtype=jnp.float64)
        key = jax.random.PRNGKey(config['seed'])
        started = time.perf_counter()
        walkers, key, _ = components['sampler'](params, walkers, key, config['burn_sweeps'])
        jax.block_until_ready((walkers, key))
        burn_seconds = time.perf_counter() - started
    occupations = np.asarray(walkers)
    if (occupations.shape != (config['batch'], config['flux'] + 1)
            or not np.all((occupations == 0) | (occupations == 1))
            or not np.all(occupations.sum(axis=1) == config['ne'])
            or not np.all(occupations @ np.arange(config['flux'] + 1) == components['operator'].target_sum)):
        raise ValueError('Walkers violate the configured fixed-N,Lz sector')
    compile_start = time.perf_counter()
    compiled = components['step'].lower(params, state, walkers, key).compile()
    warm = compiled(params, state, walkers, key)
    jax.block_until_ready(warm)
    del warm
    compile_seconds = time.perf_counter() - compile_start
    if not resume:
        output.mkdir(parents=True)
        (output / 'checkpoints').mkdir()
        atomic_npz(output / 'initial_parameters.npz', **initial)
        atomic_npz(output / 'wall_profile.npz', epsilon_a=components['epsilon'],
                   orbital_m=np.arange(config['flux'] + 1) - config['flux'] / 2.)
        atomic_json(output / 'run_config.json', dict(configuration=config, config_sha256=signature,
            implementation_sha256=hashes, architecture=model.architecture(initial),
            sector_dimension=components['operator'].dimension(), orbital_profile_sha256=profile_sha256(components['epsilon']),
            device=str(jax.devices()[0]), no_ed_data_in_training=True, no_geometry_guidance=True,
            sampling='q_beta proportional to |psi|^(2 beta), all walkers enter update',
            weighting='p/q_beta self-normalized weights; finite-batch covariance/ratio bias remains',
            source=None, burn_seconds=burn_seconds, compile_seconds=compile_seconds))
    stop = {'requested': False}
    previous_handlers = {}
    for number in (signal.SIGTERM, signal.SIGUSR1):
        previous_handlers[number] = signal.signal(number, lambda *_: stop.update(requested=True))
    start = time.perf_counter()
    current, last_values, reason = first, {}, 'target_steps'
    last_logged_step = first

    def checkpoint(step):
        jax.block_until_ready((params, state, walkers, key))
        seconds = previous_seconds + time.perf_counter() - start
        kwargs = dict(params=params, state=state, walkers=walkers, key=key,
                      rng_state=rng.bit_generator.state, step=step, training_seconds=seconds,
                      config_sha256=signature)
        save_checkpoint(output / 'checkpoints' / f'step_{step:012d}.npz', **kwargs)
        save_checkpoint(output / 'checkpoint_latest.npz', **kwargs)
        return seconds

    try:
        if not resume:
            checkpoint(0)
        metrics_path = output / 'metrics.csv'
        with metrics_path.open('a' if resume else 'w', newline='') as stream:
            writer = None
            for current in range(first + 1, config['steps'] + 1):
                if config['max_runtime_seconds'] is not None and previous_seconds + time.perf_counter() - start >= config['max_runtime_seconds']:
                    current -= 1
                    reason = 'runtime_limit'
                    break
                refresh_acceptance = None
                if config['refresh_every'] and current % config['refresh_every'] == 0:
                    proposals = jnp.asarray(components['operator'].sample_uniform(rng, config['batch']), dtype=jnp.float64)
                    walkers, key, refresh_acceptance = global_refresh(components['log_fn'], params, walkers, proposals, key)
                params, state, walkers, key, values = compiled(params, state, walkers, key)
                if current == first + 1 or current % config['log_every'] == 0 or current == config['steps'] or stop['requested']:
                    data = jax.device_get(values)
                    last_values = {name: float(value) for name, value in data.items()}
                    if not all(math.isfinite(value) for value in last_values.values()):
                        raise ArithmeticError('Nonfinite broader-sphere training metric')
                    row = dict(step=current, training_seconds=previous_seconds + time.perf_counter() - start,
                               global_refresh_acceptance='' if refresh_acceptance is None else float(refresh_acceptance), **last_values)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if not resume or metrics_path.stat().st_size == 0:
                            writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    last_logged_step = current
                if current % config['checkpoint_every'] == 0:
                    seconds = checkpoint(current)
                    print(f'N={config["ne"]} beta={config["beta"]} step={current} '
                          f'MC_E={last_values.get("energy_estimate", float("nan")):.8g} '
                          f'ESS={last_values.get("importance_ess", float("nan")):.2f} seconds={seconds:.2f}', flush=True)
                if stop['requested']:
                    reason = 'signal'
                    break
            if current > first and last_logged_step != current:
                last_values = {name: float(value) for name, value in jax.device_get(values).items()}
                if not all(math.isfinite(value) for value in last_values.values()):
                    raise ArithmeticError('Nonfinite final broader-sphere training metric')
                row = dict(step=current, training_seconds=previous_seconds + time.perf_counter() - start,
                           global_refresh_acceptance='' if refresh_acceptance is None else float(refresh_acceptance),
                           **last_values)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not resume or metrics_path.stat().st_size == 0:
                        writer.writeheader()
                writer.writerow(row)
                stream.flush()
        seconds = checkpoint(current)
        summary = dict(status='completed' if current == config['steps'] else 'paused',
            stop_reason=reason, completed_steps=current, requested_steps=config['steps'],
            training_seconds=seconds, additional_training_seconds=seconds - previous_seconds,
            first_step=first, compile_seconds=compile_seconds, burn_seconds=burn_seconds,
            parameter_count=model.architecture(params)['parameter_count'], last_metrics=last_values,
            last_metrics_step=current if current > first else None,
            config_sha256=signature, no_geometry_guidance=True, beta=config['beta'])
        atomic_json(output / 'training_summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(load_config(args.config), resume=args.resume)


if __name__ == '__main__':
    main()
