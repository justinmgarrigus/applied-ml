"""Portable Born-sampling SPRING / VMC-SGD / damped-MinSR training; steps are optimizer updates.

Run ``python -m fqh_spring.train --config configs/ne8_mlp_200k.json``.
There are no fixed-dataset epochs. A resume restores all parameters, SPRING
memory, walkers, and both RNGs. The
requested steps are a total target, so increasing --steps extends a saved run.
SIGTERM/SIGUSR1 request an atomic checkpoint and clean exit after an update.
The Hamiltonian and model are evaluated without ED labels or a many-body matrix.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

DEFAULTS = dict(method='spring', ne=8, flux=None, v3=0., two_lz=0, width=64, layers=6,
                variant='mlp', readout='final', amplitude_map='exponential',
                sampling_floor=0., source=None, seed=71, init_seed=1729,
                steps=200000, batch=32, sweeps=2, learning_rate=.01,
                damping=.01, mu=.9, max_norm=.03, refresh_every=25,
                burn_sweeps=500, pair_chunk=2, checkpoint_every=1000,
                log_every=100, keep_checkpoints=3, native_threads=1,
                cpus=None, backend='auto', max_runtime_seconds=None,
                output='runs/ne8_final_mlp_seed71')
# Administrative changes do not change the mathematical continuation state.
OPERATIONAL = {'steps', 'output', 'checkpoint_every', 'log_every', 'keep_checkpoints',
               'native_threads', 'cpus', 'backend', 'max_runtime_seconds'}
# Last published pre-Cartesian implementation. Its exponential/Born update is
# preserved, so existing signed checkpoints can continue on that same path.
LEGACY_BORN_SOURCES = {
    'fqh_spring/train.py': '67e1d255eb7dc811943c9524b0363041e497f2721cb3484539ae9e154bf5b39d',
    'fqh_spring/local_energy.py': 'd5bdfe1735ca2f4e05ce1577e2302d426b89f6fdb00ef60512052c4ec441d729',
}
PUBLISHED_CARTESIAN_SOURCES = {
    'fqh_spring/train.py': 'b84819c082f6aaa16f422c58a63b2545ac8030c1a77cc75627c61d9854625aad',
    'fqh_spring/cartesian.py': '915e9a63dadfb71cf9099d93ab342f0adee159a3a0fb67595a96eb12b02ab0d0',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path, overrides=None):
    path = Path(path).resolve()
    supplied = json.loads(path.read_text())
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    config = DEFAULTS | supplied
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    config['flux'] = 3 * (config['ne'] - 1) if config['flux'] is None else config['flux']
    for name in ('ne', 'flux', 'width', 'layers', 'steps', 'batch', 'sweeps', 'pair_chunk',
                 'checkpoint_every', 'log_every', 'keep_checkpoints', 'native_threads'):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('burn_sweeps', 'refresh_every', 'seed', 'init_seed'):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    for name in ('learning_rate', 'damping', 'max_norm'):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not math.isfinite(config['v3']) or not 0 <= config['mu'] < 1:
        raise ValueError('Require finite v3 and 0 <= mu < 1')
    if config['method'] not in ('spring', 'vmc_sgd', 'minsr'):
        raise ValueError('Implemented methods are spring, vmc_sgd, minsr')
    mode, floor = config['amplitude_map'], config['sampling_floor']
    if mode not in ('exponential', 'cartesian'):
        raise ValueError('amplitude_map must be exponential or cartesian')
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor):
        raise ValueError('sampling_floor must be a finite number')
    if mode == 'cartesian' and floor <= 0:
        raise ValueError('Cartesian amplitudes require sampling_floor > 0')
    if mode == 'exponential' and floor != 0:
        raise ValueError('Exponential/Born mode requires sampling_floor=0')
    config['sampling_floor'] = float(floor)
    if config['variant'] not in ('mlp', 'dense', 'attention') or config['backend'] not in ('auto', 'cpu', 'gpu'):
        raise ValueError('Unknown architecture or JAX backend')
    if config['readout'] not in ('final', 'legacy_additive'):
        raise ValueError('Unknown readout architecture')
    if config['readout'] == 'final' and config['variant'] != 'mlp':
        raise ValueError('Final readout requires variant="mlp"; dense/attention corrections are legacy models')
    if config['cpus'] is not None and (not isinstance(config['cpus'], int) or config['cpus'] < 1):
        raise ValueError('cpus must be a positive integer')
    maximum = config['max_runtime_seconds']
    if maximum is not None and (not math.isfinite(maximum) or maximum <= 0):
        raise ValueError('max_runtime_seconds must be positive and finite')
    # Shipped configs are in PROJECT/configs; custom configs use the same rule.
    root = path.parent.parent if path.parent.name == 'configs' else path.parent
    for name in ('source', 'output'):
        if config[name] is not None:
            value = Path(config[name]).expanduser()
            config[name] = str(value.resolve() if value.is_absolute() else (root / value).resolve())
    return config


def _set_runtime(config):
    """Respect the allocation's allowed CPU IDs, including nonzero Slurm IDs."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = str(config['native_threads'])
    if config['backend'] != 'auto':
        os.environ['JAX_PLATFORMS'] = config['backend']
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        limit = config['cpus']
        if limit is None and os.environ.get('SLURM_CPUS_PER_TASK'):
            limit = int(os.environ['SLURM_CPUS_PER_TASK'])
        if limit:
            os.sched_setaffinity(0, allowed[:min(limit, len(allowed))])


def _trim_csv(path, step):
    """Discard rows newer than the last atomic checkpoint after interruption."""
    if not path.exists():
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row['step']) <= step]
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _scientific_signature(invariant):
    return hashlib.sha256(json.dumps(invariant, sort_keys=True).encode()).hexdigest()


def _legacy_born_invariant(invariant):
    legacy = {key: value for key, value in invariant.items()
              if key not in ('amplitude_map', 'sampling_floor')}
    legacy['implementation_sha256'] = invariant['implementation_sha256'] | LEGACY_BORN_SOURCES
    return legacy


def _published_invariant(invariant):
    previous = dict(invariant)
    sources = dict(invariant['implementation_sha256'])
    for name, fingerprint in PUBLISHED_CARTESIAN_SOURCES.items():
        if name in sources:
            sources[name] = fingerprint
    previous['implementation_sha256'] = sources
    return previous


def _resume_invariant(invariant, recorded):
    """Accept an exact match to one of the two audited prior implementations."""
    candidates = [_published_invariant(invariant)]
    if invariant['amplitude_map'] == 'exponential':
        candidates.append(_legacy_born_invariant(invariant))
    allowed_runtime_sources = [invariant['implementation_sha256'],
                               candidates[0]['implementation_sha256']]
    for candidate in candidates:
        signature = _scientific_signature(candidate)
        runtime_sources = recorded.get('runtime_implementation_sha256')
        if (recorded['config_sha256'] == signature
                and recorded.get('invariant_configuration') == candidate
                and (runtime_sources is None or runtime_sources in allowed_runtime_sources)):
            return candidate, signature
    return invariant, _scientific_signature(invariant)


def run(config, *, resume=False):
    config = {'amplitude_map': 'exponential', 'sampling_floor': 0., **config}
    _set_runtime(config)
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    import numpy as np
    import optax
    from . import model
    from .physics import SphereOperator
    from .sampler import make_sampler, global_refresh
    from .local_energy import make_local_energy
    from .optimizer import make_optimizer, score_data, robust_sr
    from .checkpoint import atomic_json, atomic_npz, save_checkpoint, load_checkpoint

    output = Path(config['output'])
    anchor_path = output / 'initial_parameters.npz'
    cartesian = config['amplitude_map'] == 'cartesian'
    if resume:
        if not (output / 'checkpoint_latest.npz').exists():
            raise FileNotFoundError(f'No resume checkpoint in {output}')
    elif output.exists():
        raise FileExistsError(f'Output already exists; use --resume or choose a new directory: {output}')
    operator = SphereOperator(config['ne'], flux=config['flux'], v3=config['v3'], two_lz=config['two_lz'])
    source_hash = digest(config['source']) if config['source'] is not None else None
    invariant = {name: value for name, value in config.items() if name not in OPERATIONAL and name != 'source'}
    invariant['source_sha256'] = source_hash
    runtime_names = ('__init__', 'physics', 'base_model', 'legacy_model', 'structured_model', 'model',
                     'sampler', 'local_energy', 'optimizer', 'checkpoint', 'train')
    if cartesian:
        runtime_names += ('cartesian',)
    source_hashes = {f'fqh_spring/{name}.py': digest(Path(__file__).parent / f'{name}.py')
                     for name in runtime_names}
    vendor_root = Path(importlib.util.find_spec('jaqmc').origin).parent
    for name in ('__init__.py', 'optimizer/__init__.py', 'optimizer/sr/__init__.py',
                 'optimizer/sr/jasr.py', 'utils/__init__.py', 'utils/chunked_vmap.py', 'array_types.py'):
        source_hashes[f'jaqmc/{name}'] = digest(vendor_root / name)
    invariant['implementation_sha256'] = source_hashes
    if resume:
        old = json.loads((output / 'run_config.json').read_text())
        if cartesian:
            invariant['frozen_anchor_sha256'] = digest(anchor_path)
            if old.get('frozen_anchor_sha256') != invariant['frozen_anchor_sha256']:
                raise ValueError('Saved Cartesian anchor hash differs from the run configuration')
        signature = _scientific_signature(invariant)
        if old['config_sha256'] != signature:
            invariant, signature = _resume_invariant(invariant, old)
        if old['config_sha256'] != signature:
            raise ValueError('Resume source code, physics, model, seed, sampling or optimizer settings changed; start a new run for intentional changes')
        with np.load(anchor_path, allow_pickle=False) as archive:
            initial = {name: jnp.asarray(archive[name]) for name in archive.files}
    elif config['source'] is not None:
        with np.load(config['source'], allow_pickle=False) as archive:
            if 'metadata' in archive.files:
                raise ValueError('source must be a plain parameter NPZ; use --resume for full checkpoints')
            initial = model.prepare({name: archive[name] for name in archive.files}, norb=operator.norb,
                                    width=config['width'], layers=config['layers'], kind=config['variant'],
                                    seed=config['init_seed'], readout=config['readout'])
    else:
        initial = model.initialize(operator.norb, width=config['width'], layers=config['layers'],
                                   kind=config['variant'], seed=config['init_seed'], readout=config['readout'])
    if any(not np.isfinite(np.asarray(value)).all() for value in initial.values()):
        raise ValueError('Nonfinite source parameters')
    architecture = model.architecture(initial)
    optimizer = make_optimizer(config['method'], learning_rate=config['learning_rate'], damping=config['damping'],
                            mu=config['mu'], max_norm=config['max_norm'])
    if cartesian:
        from .cartesian import make_amplitude, make_fit, make_proposal
        from .local_energy import make_hamiltonian_action

        amplitude = make_amplitude(initial)
        log_target = make_proposal(amplitude, config['sampling_floor'])
        local = make_hamiltonian_action(operator, amplitude, pair_chunk_size=config['pair_chunk'])
        fit = make_fit(initial, config['sampling_floor'], metric_rows=config['method'] != 'vmc_sgd')
    else:
        log_target = model.log_psi
        local = make_local_energy(operator, log_target, pair_chunk_size=config['pair_chunk'])
    sampler = make_sampler(log_target, operator.ne, operator.flux)
    prefix = 'dense_' if config['variant'] == 'dense' else 'attn_'
    correction_keys = tuple(name for name in initial if name.startswith(prefix)) if config['variant'] != 'mlp' else ()
    readout_keys = tuple(name for name in correction_keys if name.startswith(prefix + 'head_'))
    hidden_keys = tuple(name for name in correction_keys if name not in readout_keys)
    radius, eta = config['max_norm'], config['learning_rate']

    @jax.jit
    def train_step(params, state, walkers, key):
        walkers, key, acceptance = sampler(params, walkers, key, config['sweeps'])
        evaluated = local(params, walkers)
        if cartesian:
            force, rows, estimates = fit(params, walkers, evaluated['amplitude'], evaluated['energy'])
        else:
            energies = evaluated['energy']
            residual = energies - jnp.mean(energies)
            scores = model.score_tree(params, walkers)
            force = jax.tree.map(lambda score: jnp.mean(residual.reshape((-1,) + (1,) * (score.ndim - 1)) * score.conj(), axis=0).real, scores)
            if config['method'] != 'vmc_sgd':
                rows = score_data(params, scores)
            estimates = dict(energy_estimate=jnp.mean(energies).real,
                             energy_imaginary_estimate=jnp.mean(energies).imag,
                             local_energy_variance=jnp.mean(jnp.abs(residual)**2))
        if config['method'] == 'vmc_sgd':
            physical_gradient = jax.tree.map(lambda value: 2.*value, force)
            direction = ravel_pytree(physical_gradient)[0]
            updates, state = optimizer.update(physical_gradient, state, params)
        else:
            updates, state = optimizer.update(force, state, params, data=rows)
            direction = state[0].prev_delta
        raw_norm = jnp.linalg.norm(direction)
        flat_update = ravel_pytree(updates)[0]
        if config['method'] == 'vmc_sgd':
            scale = jnp.where(raw_norm > 0, jnp.linalg.norm(flat_update)/jnp.maximum(raw_norm,1e-300), 0.)
        else:
            scale = jnp.minimum(eta, radius / (raw_norm + 1e-8))
        next_params = optax.apply_updates(params, updates)
        same = jnp.all(walkers[:, None, :] == walkers[None, :, :], axis=-1)
        multiplicity = jnp.sum(same, axis=1)
        row = dict(**estimates, acceptance=acceptance,
                   distinct_configurations=jnp.sum(1. / multiplicity),
                   configuration_mass_ess=walkers.shape[0]**2 / jnp.sum(same),
                   maximum_configuration_multiplicity=jnp.max(multiplicity),
                   force_norm=jnp.linalg.norm(ravel_pytree(force)[0]), physical_gradient_norm=2*jnp.linalg.norm(ravel_pytree(force)[0]), raw_direction_norm=raw_norm,
                   effective_step_scale=scale, cap_active=(raw_norm > 0) & (scale < eta),
                   update_norm=jnp.linalg.norm(flat_update),
                   update_scale_consistency_error=jnp.linalg.norm(flat_update + scale * direction))
        for label, names in (('readout', readout_keys), ('hidden', hidden_keys)):
            row['correction_' + label + '_update_norm'] = jnp.sqrt(sum(jnp.sum(updates[name]**2) for name in names))
            row['correction_' + label + '_force_norm'] = jnp.sqrt(sum(jnp.sum(force[name]**2) for name in names))
        return next_params, state, walkers, key, row

    rng = np.random.default_rng(config['seed'])
    initial_state = optimizer.init(initial)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        atomic_npz(anchor_path, **initial)
        if cartesian:
            invariant['frozen_anchor_sha256'] = digest(anchor_path)
        signature = _scientific_signature(invariant)
    metadata = dict(configuration=config, config_sha256=signature, invariant_configuration=invariant,
                    architecture=architecture, sector_dimension=operator.dimension(),
                    flux=operator.flux, orbitals=operator.norb, parameter_count=ravel_pytree(initial)[0].size,
                    sampling='Direct Born |psi|^2; uniform 1/B row weights; no artificial temperature',
                    physics='LLL Haldane sphere; canonical antisymmetric Fock basis; fixed N and Lz',
                    hamiltonian='sum P_R1 + v3 sum P_R3; normalized pair projectors; V1=1',
                    force_convention='Re Cov(O*, E_local), half the physical energy gradient',
                    norm_cap='max_norm is radius sqrt(C), not C',
                    many_body_matrix_constructed=False, no_ed_data_in_training=True,
                    method=config['method'],
                    optimizer='optax.sgd + update-radius clipping' if config['method']=='vmc_sgd' else robust_sr.__module__,
                    optimizer_source_sha256=digest(Path(__import__('inspect').getfile(optax.sgd if config['method']=='vmc_sgd' else robust_sr))),
                    method_convention='VMC-SGD uses full physical gradient=2force; SPRING/MinSR use half-gradient force; MinSR has no carried direction',
                    minsr_regularization='positive damping; no eigenvalue cutoff or Moore-Penrose pseudoinverse',
                    versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy', 'scipy', 'optax')},
                    device=str(jax.devices()[0]), devices=[str(device) for device in jax.devices()],
                    cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)
    if cartesian:
        metadata.update(
            frozen_anchor_sha256=invariant['frozen_anchor_sha256'],
            sampling='Metropolis target |u|^2 + sampling_floor, with inverse target weights',
            force_convention='Real projected amplitude Jacobian adjoint times weighted residual, half the physical energy gradient',
        )
    if invariant['implementation_sha256'] != source_hashes:
        metadata['runtime_implementation_sha256'] = source_hashes
    atomic_json(output / 'run_config.json', metadata)
    stop = {'signal': None}
    previous_handlers = {}
    def request_stop(number, _frame):
        stop['signal'] = signal.Signals(number).name
    for name in ('SIGTERM', 'SIGUSR1'):
        if hasattr(signal, name):
            number = getattr(signal, name)
            previous_handlers[number] = signal.signal(number, request_stop)
    begin_wall = time.perf_counter()
    last_row, log_stream = {}, None
    first_step, previous_seconds = 0, 0.
    try:
        if resume:
            restored = load_checkpoint(output / 'checkpoint_latest.npz', initial_state, expected_config_sha256=signature)
            params, state, walkers, key = (restored[name] for name in ('params', 'state', 'walkers', 'key'))
            rng.bit_generator.state = restored['rng_state']
            first_step, previous_seconds = restored['step'], restored['training_seconds']
            _trim_csv(output / 'metrics.csv', first_step)
        else:
            params, state = initial, initial_state
            walkers = jnp.asarray(operator.sample_uniform(rng, config['batch']), dtype=jnp.float64)
            key = jax.random.PRNGKey(config['seed'])
            if config['burn_sweeps']:
                walkers, key, burn_acceptance = sampler(params, walkers, key, config['burn_sweeps'])
                jax.block_until_ready((walkers, key))
                atomic_json(output / 'initialization.json', dict(burn_sweeps=config['burn_sweeps'], burn_acceptance=float(burn_acceptance), source_sha256=source_hash))
        expected_shape = (config['batch'], operator.norb)
        w = np.asarray(walkers)
        if w.shape != expected_shape or not np.all((w == 0) | (w == 1)) or not np.all(w.sum(axis=1) == operator.ne) or not np.all(w @ np.arange(operator.norb) == operator.target_sum):
            raise ValueError('Invalid saved walker sector or batch size')
        if first_step > config['steps']:
            raise ValueError('Target --steps is smaller than completed checkpoint step')
        if model.architecture(params) != architecture:
            raise ValueError('Checkpoint architecture differs from initial model')
        def save(index, seconds):
            save_checkpoint(output / 'checkpoint_latest.npz', params=params, state=state, walkers=walkers, key=key,
                            rng_state=rng.bit_generator.state, step=index, training_seconds=seconds, config_sha256=signature)
            history = output / 'checkpoints'
            history.mkdir(exist_ok=True)
            destination = history / f'step_{index:012d}.npz'
            temporary = destination.with_suffix('.tmp')
            shutil.copyfile(output / 'checkpoint_latest.npz', temporary)
            os.replace(temporary, destination)
            for obsolete in sorted(history.glob('step_*.npz'))[:-config['keep_checkpoints']]:
                obsolete.unlink()
            atomic_npz(output / 'trained_parameters.npz', **params)
        if not resume:
            save(0, 0.)
        atomic_json(output / 'status.json', dict(status='compiling', step=first_step, target_steps=config['steps']))
        print(f"Compiling {config['variant']} {config['method']} on {jax.devices()[0]}; {first_step}/{config['steps']} updates", flush=True)
        compile_begin = time.perf_counter()
        compiled_step = train_step.lower(params, state, walkers, key).compile()
        compile_seconds = time.perf_counter() - compile_begin
        log_path = output / 'metrics.csv'
        log_stream = log_path.open('a', newline='', buffering=1)
        fields = ['step', 'training_seconds', 'global_refresh_acceptance'] + sorted(train_step(params, state, walkers, key)[-1])
        writer = csv.DictWriter(log_stream, fieldnames=fields)
        if log_path.stat().st_size == 0:
            writer.writeheader()
        atomic_json(output / 'status.json', dict(status='running', step=first_step, target_steps=config['steps']))
        counted_start = time.perf_counter()
        index = first_step
        for index in range(first_step + 1, config['steps'] + 1):
            if stop['signal'] is not None:
                index -= 1
                break
            params, state, walkers, key, row = compiled_step(params, state, walkers, key)
            refresh_acceptance = None
            if config['refresh_every'] and index % config['refresh_every'] == 0:
                proposals = jnp.asarray(operator.sample_uniform(rng, config['batch']), dtype=jnp.float64)
                walkers, key, refresh_acceptance = global_refresh(log_target, params, walkers, proposals, key)
            jax.block_until_ready((params, state, walkers, key, row))
            last_row = {name: float(value) for name, value in row.items()}
            if not all(math.isfinite(value) for value in last_row.values()):
                raise FloatingPointError(f'Nonfinite training diagnostics at update {index}; prior checkpoint retained')
            if last_row['update_scale_consistency_error'] > 1e-9:
                raise RuntimeError(f'Optimizer cap consistency failure at update {index}')
            seconds = previous_seconds + time.perf_counter() - counted_start
            timed_out = config['max_runtime_seconds'] is not None and time.perf_counter() - begin_wall >= config['max_runtime_seconds']
            stopping = stop['signal'] is not None or timed_out
            if index % config['log_every'] == 0 or index == config['steps'] or stopping:
                writer.writerow(dict(step=index, training_seconds=seconds,
                                     global_refresh_acceptance='' if refresh_acceptance is None else float(refresh_acceptance), **last_row))
                log_stream.flush()
                print(f"update {index}: E_MC={last_row['energy_estimate']:.8f}, accept={last_row['acceptance']:.3f}, distinct={last_row['distinct_configurations']:.1f}/{config['batch']}", flush=True)
            if index % config['checkpoint_every'] == 0 or index == config['steps'] or stopping:
                os.fsync(log_stream.fileno())
                save(index, seconds)
                atomic_json(output / 'status.json', dict(status='running', step=index, target_steps=config['steps']))
            if stopping:
                break
        seconds = previous_seconds + time.perf_counter() - counted_start
        save(index, seconds)
        completed = index == config['steps']
        status = 'completed' if completed else 'paused'
        reason = None if completed else (stop['signal'] or 'max_runtime_seconds')
        summary = dict(status=status, reason=reason, completed_steps=index, target_steps=config['steps'],
                       training_seconds=seconds, current_launch_compile_seconds=compile_seconds,
                       architecture=architecture, last_logged_diagnostics=last_row,
                       metrics_scope='MC estimates during training; use separate evaluate command for exact small-system energy',
                       walker_observations=index * config['batch'], independent_samples_claimed=False)
        atomic_json(output / 'training_summary.json', summary)
        atomic_json(output / 'status.json', dict(status=status, reason=reason, step=index, target_steps=config['steps']))
        print(f'{status}: {index}/{config["steps"]}; checkpoint {output / "checkpoint_latest.npz"}', flush=True)
        return summary
    except BaseException as error:
        atomic_json(output / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}', latest_safe_checkpoint=str(output / 'checkpoint_latest.npz')))
        raise
    finally:
        if log_stream is not None:
            log_stream.close()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cpus', type=int, help='Limit to first N CPU IDs already allowed by Slurm/cpuset')
    parser.add_argument('--backend', choices=('auto', 'cpu', 'gpu'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in ('steps', 'seed', 'cpus', 'backend')}
    if args.output is not None:
        overrides['output'] = str(args.output.resolve())
    config = load_config(args.config, overrides)
    run(config, resume=args.resume)


if __name__ == '__main__':
    main()
