"""N=10 north-centered droplet with a projected real-space cap wall.

At extra_flux=2 and delta_k=0, F=29, K=135 and M=10 (two_lz=20).
Use a fresh 30-input, width-64, one-hidden-layer historical tanh network with
2114 real parameters. The finite wall is U*Theta(x-wall_x), x=sin(theta/2)^2,
projected exactly into orbital energies using beta survival probabilities.
The wall cutoff .88 is retained from the N=8 experiment, rather than moving it
to the N=10 outermost compact orbital mean x=28/31. All flux+1 orbitals remain present. Training uses alpha=0, historical T2
importance sampling and JaQMC SPRING, without ED labels or reference matrices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

from . import train as portable
from .north_droplet import north_sector, validate_cap_wall, profile_a_epsilon, profile_sha256

DEFAULTS = portable.DEFAULTS | dict(
    ne=10, flux=None, two_lz=None, extra_flux=2, delta_k=0,
    width=64, layers=1, variant='mlp', readout='legacy_additive',
    source=None, seed=71, init_seed=7, steps=2_000, batch=256,
    learning_rate=.03, damping=.001, max_norm=.001, burn_sweeps=500,
    log_every=100, checkpoint_every=500, keep_checkpoints=6,
    backend='gpu', temperature=2., l2_penalty=0., confinement_strength=10.,
    wall_x=.88, profile='cap_step', max_runtime_seconds=7200.,
    output='runs/ne10_north_droplet_seed71',
)
OPERATIONAL = portable.OPERATIONAL


def load_config(path, overrides=None):
    path = Path(path).resolve()
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict):
        raise ValueError('Configuration must be a JSON object')
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    config = DEFAULTS | supplied
    config.update({key: value for key, value in (overrides or {}).items() if value is not None})
    fixed = dict(ne=10, v3=0., width=64, layers=1, variant='mlp', readout='legacy_additive')
    if any(config[key] != value for key, value in fixed.items()):
        raise ValueError('This runner requires the N=10 one-layer64 historical pure-V1 interaction model')
    if isinstance(config['l2_penalty'], bool) or config['l2_penalty'] != 0:
        raise ValueError('This droplet experiment requires l2_penalty=0')
    sector = north_sector(config['ne'], config['extra_flux'], config['delta_k'])
    for key in ('flux', 'two_lz'):
        if config[key] is not None and (isinstance(config[key], bool) or config[key] != sector[key]):
            raise ValueError(f'{key} disagrees with extra_flux/delta_k derived sector: {sector[key]}')
        config[key] = sector[key]
    validate_cap_wall(config['confinement_strength'], config['wall_x'], config['profile'])
    config['confinement_strength'] = float(config['confinement_strength'])
    config['wall_x'] = float(config['wall_x'])
    for key in ('steps', 'batch', 'sweeps', 'pair_chunk', 'checkpoint_every', 'log_every',
                'keep_checkpoints', 'native_threads'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('burn_sweeps', 'refresh_every', 'seed', 'init_seed'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('learning_rate', 'damping', 'max_norm', 'temperature'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not 0 <= config['mu'] < 1 or config['backend'] not in ('cpu', 'gpu', 'auto'):
        raise ValueError('Require 0 <= mu < 1 and backend cpu/gpu/auto')
    if config['cpus'] is not None and (isinstance(config['cpus'], bool)
                                      or not isinstance(config['cpus'], int) or config['cpus'] < 1):
        raise ValueError('cpus must be a positive integer')
    limit = config['max_runtime_seconds']
    if limit is not None and (not math.isfinite(limit) or limit <= 0):
        raise ValueError('max_runtime_seconds must be positive and finite')
    root = path.parent.parent if path.parent.name == 'configs' else path.parent
    for key in ('source', 'output'):
        if config[key] is not None:
            value = Path(config[key]).expanduser()
            config[key] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    return config


def build_step(config, sampler, local, optimizer):
    """Original weighted SPRING force and metric for the total physical energy."""
    from . import ne6_baseline as historical
    if config['confinement_strength'] == 0:
        # Leave the original kernel and arithmetic intact; the host logger
        # fills the identically zero confinement diagnostic for this arm.
        return historical.build_step(config, sampler, local, optimizer)
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    import optax

    @jax.jit
    def step(params, state, walkers, key):
        walkers, key, acceptance = sampler(params, walkers, key, config['sweeps'])
        key, _unused_pair_key = jax.random.split(key)
        weights = historical.importance_weights(params, walkers, config['temperature'])
        values = local(params, walkers)
        force = historical.weighted_force(params, walkers, values['energy'], weights)
        updates, state = optimizer.update(force, state, params,
                                          data=historical.metric_data(params, walkers, weights))
        direction = state[0].prev_delta
        norm = jnp.linalg.norm(direction)
        scale = jnp.minimum(config['learning_rate'], config['max_norm'] / (norm + 1e-8))
        update = ravel_pytree(updates)[0]
        mean_energy = weights @ values['energy']
        diagnostics = dict(energy_estimate=mean_energy.real, energy_imaginary_estimate=mean_energy.imag,
                           interaction_energy_estimate=(weights @ values['interaction_energy']).real,
                           confinement_energy_estimate=weights @ values['confinement_energy'],
                           importance_ess=1. / jnp.sum(weights**2), acceptance=acceptance,
                           force_norm=jnp.linalg.norm(ravel_pytree(force)[0]),
                           raw_direction_norm=norm, effective_step_scale=scale,
                           cap_active=config['max_norm'] / (norm + 1e-8) < config['learning_rate'],
                           update_norm=jnp.linalg.norm(update),
                           update_scale_consistency_error=jnp.linalg.norm(update + scale * direction))
        return optax.apply_updates(params, updates), state, walkers, key, diagnostics

    return step


def run(config, *, resume=False):
    portable._set_runtime(config)
    if config['backend'] == 'gpu':
        # The generic gpu alias can expand to unavailable ROCm backends.
        os.environ['JAX_PLATFORMS'] = 'cuda'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from . import model, ne6_baseline as historical
    from .physics import SphereOperator
    from .local_energy import make_local_energy
    from .north_droplet import make_confinement_local_energy
    from .checkpoint import atomic_json, atomic_npz, save_checkpoint, load_checkpoint

    output = Path(config['output'])
    if resume:
        if not (output / 'checkpoint_latest.npz').is_file():
            raise FileNotFoundError(f'No complete resume checkpoint in {output}')
    elif output.exists():
        raise FileExistsError(f'Output exists; use --resume or choose another directory: {output}')
    source_hash = portable.digest(config['source']) if config['source'] is not None else None
    names = ('train_ne10_north_droplet', 'north_droplet', 'ne6_baseline', 'train', 'physics', 'local_energy',
             'model', 'legacy_model', 'base_model', 'structured_model', 'optimizer', 'sampler', 'checkpoint')
    hashes = {name: portable.digest(Path(__file__).parent / f'{name}.py') for name in names}
    import inspect
    vendor_root = Path(inspect.getfile(historical.robust_sr)).parents[2]
    for name in ('__init__.py', 'optimizer/__init__.py', 'optimizer/sr/__init__.py',
                 'optimizer/sr/jasr.py', 'utils/__init__.py', 'utils/chunked_vmap.py', 'array_types.py'):
        hashes[f'jaqmc/{name}'] = portable.digest(vendor_root / name)
    epsilon_a = profile_a_epsilon(config['flux'], config['confinement_strength'], config['wall_x'], config['profile'])
    epsilon_hash = profile_sha256(epsilon_a)
    sector = north_sector(config['ne'], config['extra_flux'], config['delta_k'])
    invariant = {key: value for key, value in config.items() if key not in OPERATIONAL and key != 'source'}
    invariant.update(source_sha256=source_hash, implementation_sha256=hashes, profile_sha256=epsilon_hash)
    signature = hashlib.sha256(json.dumps(invariant, sort_keys=True).encode()).hexdigest()
    if resume:
        old = json.loads((output / 'run_config.json').read_text())
        if old['config_sha256'] != signature:
            raise ValueError('Resume settings changed; start a new run for intentional changes')
        with np.load(output / 'initial_parameters.npz', allow_pickle=False) as archive:
            initial = {key: jnp.asarray(archive[key]) for key in archive.files}
    elif config['source'] is not None:
        with np.load(config['source'], allow_pickle=False) as archive:
            if 'metadata' in archive.files:
                raise ValueError('source must contain plain parameters; --resume requires a runner checkpoint')
            initial = model.prepare({key: archive[key] for key in archive.files}, norb=config['flux'] + 1, width=64,
                                    layers=1, kind='mlp', readout='legacy_additive')
    else:
        initial = historical.initialize(config['init_seed'], norb=config['flux'] + 1, width=64)
    architecture = model.architecture(initial)
    if architecture['parameter_count'] != 64 * (config['flux'] + 1) + 194 or any(not np.isfinite(np.asarray(v)).all() for v in initial.values()):
        raise ValueError('Expected finite one-layer N=10 parameters with 64*(flux+1)+194 real entries')
    operator = SphereOperator(10, flux=config['flux'], v3=0., two_lz=config['two_lz'])
    sampler, refresh = historical.sampling_functions(10, config['flux'], config['temperature'])
    local = make_local_energy(operator, model.log_psi, pair_chunk_size=config['pair_chunk'])
    local = make_confinement_local_energy(local, epsilon_a)
    optimizer = historical.make_spring(learning_rate=config['learning_rate'], damping=config['damping'],
                                       mu=config['mu'], max_norm=config['max_norm'])
    template = optimizer.init(initial)
    step = build_step(config, sampler, local, optimizer)
    rng = np.random.default_rng(config['seed'])
    previous_seconds, first_step = 0., 0
    if resume:
        restored = load_checkpoint(output / 'checkpoint_latest.npz', template,
                                    expected_config_sha256=signature)
        params, state, walkers, key = (restored[name] for name in ('params', 'state', 'walkers', 'key'))
        rng.bit_generator.state = restored['rng_state']
        first_step, previous_seconds = restored['step'], restored['training_seconds']
        if first_step > config['steps']:
            raise ValueError('Target steps is smaller than the completed checkpoint step')
        portable._trim_csv(output / 'metrics.csv', first_step)
    else:
        params, state = initial, template
        walkers = jnp.asarray(operator.sample_uniform(rng, config['batch']), dtype=jnp.float64)
        key = jax.random.PRNGKey(config['seed'] + 1)
        if config['burn_sweeps']:
            walkers, key, _ = sampler(params, walkers, key, config['burn_sweeps'])
            jax.block_until_ready((walkers, key))
        output.mkdir(parents=True, exist_ok=False)
        atomic_npz(output / 'initial_parameters.npz', **initial)
    w = np.asarray(walkers)
    if (w.shape != (config['batch'], operator.norb) or not np.isin(w, [0, 1]).all()
            or not (w.sum(1) == operator.ne).all() or not (w @ np.arange(operator.norb) == operator.target_sum).all()):
        raise ValueError('Invalid saved walker shape or sector')
    metadata = dict(configuration=config, config_sha256=signature, invariant_configuration=invariant,
                    architecture=architecture, sector_dimension=operator.dimension(), orbitals=operator.norb,
                    flux=operator.flux, two_lz=operator.two_lz, parameter_count=architecture['parameter_count'],
                    sector=sector, outward_K=sector['outward_K'],
                    sampling='q_T proportional to |psi|^(2/T), self-normalized p/q weights; finite-batch ratio bias',
                    physical_temperature='Pure-state variational optimization; sampling T is artificial',
                    objective='Expectation of H_V1 + sum_a epsilon_a*n_a; l2_penalty=0; V1=1',
                    physical_hamiltonian='Normalized sphere V1 interaction plus LLL-projected finite cap step wall',
                    physics='LLL sphere, canonical antisymmetric Slater basis; all flux+1 orbitals retained',
                    potential_definition='U*Theta(x-wall_x), x=sin(theta/2)^2; epsilon_k=U*betaincc(k+1,F-k+1,wall_x)',
                    potential_profile=dict(profile=config['profile'], confinement_strength=config['confinement_strength'],
                                           wall_x=config['wall_x'], epsilon_a=epsilon_a.tolist(),
                                           epsilon_k=epsilon_a[::-1].tolist(), profile_sha256=epsilon_hash,
                                           hash_convention='SHA256 raw little-endian float64 epsilon_a bytes', units='V1'),
                    profile_sha256=epsilon_hash,
                    rotational_symmetry=('SO(3) preserved by the interaction; fixed M sector' if config['confinement_strength'] == 0 else
                                         'SO(3) broken by cap wall; axial rotations and total Lz preserved'),
                    coordinate_convention='a=m+F/2; k=F-a; north pole selects k=0,a=F; outward K=N*F/2-M',
                    optimizer='Actual JaQMC SPRING, original signed paired-quadrature weighted score adapter',
                    optimizer_memory_reset=not resume, force_convention='Re Cov(O*, E_local), half physical gradient',
                    norm_cap='max_norm is sqrt(C)', no_ed_data_in_training=True, many_body_matrix_constructed=False,
                    initialization=(f"Fresh historical initializer, seed {config['init_seed']}, norb={operator.norb}; no padding or embedding of old N=8 weights"
                                    if source_hash is None else 'Validated same-dimension plain parameters; fresh SPRING memory on new run'),
                    source_sha256=source_hash, device=str(jax.devices()[0]),
                    numerical_trajectory='Historical weighted SPRING protocol; new flux, sector and finite projected wall',
                    versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy', 'scipy', 'optax')})
    atomic_json(output / 'run_config.json', metadata)

    def save(index, seconds):
        save_checkpoint(output / 'checkpoint_latest.npz', params=params, state=state, walkers=walkers, key=key,
                        rng_state=rng.bit_generator.state, step=index, training_seconds=seconds, config_sha256=signature)
        atomic_npz(output / 'trained_parameters.npz', **params)
        history = output / 'checkpoints'
        history.mkdir(exist_ok=True)
        destination = history / f'step_{index:012d}.npz'
        shutil.copyfile(output / 'checkpoint_latest.npz', destination)
        for old in sorted(history.glob('step_*.npz'))[:-config['keep_checkpoints']]:
            old.unlink()

    if not resume:
        save(0, 0.)
    atomic_json(output / 'status.json', dict(status='compiling', step=first_step, target_steps=config['steps']))
    compile_begin = time.perf_counter()
    compiled = step.lower(params, state, walkers, key).compile()
    compile_seconds = time.perf_counter() - compile_begin
    # Query diagnostic names from the compiled disposable update, leaving
    # parameters, walkers, RNGs and optimizer memory at their saved state.
    disposable = compiled(params, state, walkers, key)
    jax.block_until_ready(disposable)
    names = sorted(set(disposable[-1]) | {'interaction_energy_estimate', 'confinement_energy_estimate'})
    atomic_json(output / 'status.json', dict(status='running', step=first_step, target_steps=config['steps']))
    stop = {'signal': None}
    handlers = {}

    def request_stop(number, _frame):
        stop['signal'] = signal.Signals(number).name

    for name in ('SIGTERM', 'SIGUSR1'):
        if hasattr(signal, name):
            number = getattr(signal, name)
            handlers[number] = signal.signal(number, request_stop)
    counted_start, index, last_row = time.perf_counter(), first_step, {}
    try:
        with (output / 'metrics.csv').open('a', newline='', buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=['step', 'training_seconds', 'global_refresh_acceptance'] + names)
            if stream.tell() == 0:
                writer.writeheader()
            for index in range(first_step + 1, config['steps'] + 1):
                if stop['signal'] is not None:
                    index -= 1
                    break
                params, state, walkers, key, row = compiled(params, state, walkers, key)
                refresh_acceptance = None
                if config['refresh_every'] and index % config['refresh_every'] == 0:
                    proposals = jnp.asarray(operator.sample_uniform(rng, config['batch']), dtype=jnp.float64)
                    walkers, key, refresh_acceptance = refresh(params, walkers, proposals, key)
                jax.block_until_ready((params, state, walkers, key, row))
                # Start the diagnostic transfers together; converting device
                # scalars one at a time serializes host/device round trips.
                # Values, finite checks and every optimizer update are unchanged.
                last_row = {key: float(value) for key, value in jax.device_get(row).items()}
                if config['confinement_strength'] == 0:
                    last_row.update(interaction_energy_estimate=last_row['energy_estimate'], confinement_energy_estimate=0.)
                if not all(math.isfinite(value) for value in last_row.values()):
                    raise FloatingPointError('Nonfinite diagnostics; latest prior checkpoint retained')
                if last_row['update_scale_consistency_error'] > 1e-9:
                    raise RuntimeError('SPRING update cap consistency check failed')
                seconds = previous_seconds + time.perf_counter() - counted_start
                timed_out = config['max_runtime_seconds'] is not None and time.perf_counter() - counted_start >= config['max_runtime_seconds']
                stopping = stop['signal'] is not None or timed_out
                if index % config['log_every'] == 0 or index == config['steps'] or stopping:
                    writer.writerow(dict(step=index, training_seconds=seconds,
                                         global_refresh_acceptance='' if refresh_acceptance is None else float(refresh_acceptance), **last_row))
                    print(f"N10 north-droplet update {index}: E_MC={last_row['energy_estimate']:.8f}, acceptance={last_row['acceptance']:.3f}", flush=True)
                if index % config['checkpoint_every'] == 0 or index == config['steps'] or stopping:
                    stream.flush()
                    os.fsync(stream.fileno())
                    save(index, seconds)
                    atomic_json(output / 'status.json', dict(status='running', step=index, target_steps=config['steps']))
                if stopping:
                    break
        seconds = previous_seconds + time.perf_counter() - counted_start
        save(index, seconds)
        completed = index == config['steps']
        summary = dict(status='completed' if completed else 'paused', completed_steps=index,
                       target_steps=config['steps'], training_seconds=seconds, compile_seconds=compile_seconds,
                       architecture=architecture, last_logged_diagnostics=last_row,
                       metrics_scope='Training MC estimates; frozen ED evaluation is separate',
                       reason=None if completed else (stop['signal'] or 'max_runtime_seconds'))
        atomic_json(output / 'training_summary.json', summary)
        atomic_json(output / 'status.json', summary)
        return summary
    except BaseException as error:
        atomic_json(output / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/ne10_north_cap.json'))
    parser.add_argument('--steps', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--backend', choices=('cpu', 'gpu', 'auto'))
    parser.add_argument('--l2-penalty', type=float, help='alpha in H + alpha L^2, with hbar=V1=1')
    parser.add_argument('--confinement-strength', type=float, help='U in the projected cap step wall, V1 units')
    parser.add_argument('--wall-x', type=float, help='wall cutoff in x=sin(theta/2)^2')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    overrides = {name: getattr(args, name) for name in ('steps', 'cpus', 'backend', 'l2_penalty', 'confinement_strength', 'wall_x')}
    if args.output is not None:
        overrides['output'] = str(args.output.resolve())
    return run(load_config(args.config, overrides), resume=args.resume)


if __name__ == '__main__':
    main()
