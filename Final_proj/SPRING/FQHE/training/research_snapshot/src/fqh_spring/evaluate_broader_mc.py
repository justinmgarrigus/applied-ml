"""Held-out matrix-free Monte Carlo diagnostics for a frozen sphere network.

No ED basis, matrix or labels are loaded. Fresh independent chains target
q_beta proportional to |psi|**(2*beta); p/q importance ratios estimate the
physical Born expectations. Time-correlated packs are grouped by independent
chains, rather than treated as IID observations for the reported MCSE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def importance_weights(log_amplitudes, beta):
    """Stable globally normalized p/q weights; input can be [packs,chains]."""
    values = np.asarray(log_amplitudes, dtype=np.float64)
    if not 0 < beta <= 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Require finite nonempty amplitudes and 0 < beta <= 1')
    logs = 2. * (1. - beta) * values
    raw = np.exp(logs - logs.max())
    return raw / raw.sum()


def cluster_ratio_statistics(weights, values, groups=16):
    """SN ratio and delta-method MCSE clustered over independent chain groups.

    All packs from a chain stay in the same cluster. Clusters contain equal
    numbers of chains, assigned by chain-index modulo groups. This preserves
    within-chain temporal dependence. The variance estimate is
    G/(G-1) sum_g |N_g-R*D_g|^2 / (sum_g D_g)^2, separately for real/imaginary
    components. It is asymptotic ratio inference, not a finite-sample bound.
    """
    weights, values = np.asarray(weights, dtype=float), np.asarray(values)
    if (weights.ndim != 2 or values.shape[:2] != weights.shape
            or not np.isfinite(weights).all() or not np.isfinite(values).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Expected finite weights and observables with matching [packs,chains]')
    if isinstance(groups, bool) or int(groups) != groups or not 2 <= groups <= weights.shape[1]:
        raise ValueError('Require at least two independent chain groups')
    groups = int(groups)
    if weights.shape[1] % groups:
        raise ValueError('Chain count must be divisible by the number of groups')
    extra = (None,) * (values.ndim - 2)
    total = weights.sum()
    estimate = np.sum(weights[(...,) + extra] * values, axis=(0, 1)) / total
    numerators, denominators = [], []
    for group in range(groups):
        w = weights[:, group::groups]
        v = values[:, group::groups]
        numerators.append(np.sum(w[(...,) + extra] * v, axis=(0, 1)))
        denominators.append(w.sum())
    numerators, denominators = np.asarray(numerators), np.asarray(denominators)
    if np.any(denominators <= 0):
        raise ArithmeticError('An independent-chain group has zero importance mass; coverage is insufficient')
    residual = numerators - denominators[(...,) + extra] * estimate
    factor = groups / (groups - 1.) / total**2
    real_se = np.sqrt(factor * np.sum(residual.real**2, axis=0))
    imag_se = np.sqrt(factor * np.sum(residual.imag**2, axis=0))
    return dict(estimate=estimate, mcse_real=real_se, mcse_imag=imag_se,
                group_denominators=denominators,
                group_ratios=numerators / denominators[(...,) + extra])


def basic_split_rhat(values):
    """Basic split-chain Rhat for a scalar q-chain observable, not rank Rhat."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 4 or not np.isfinite(values).all():
        raise ValueError('Rhat needs finite [packs,chains] with at least four packs')
    length = values.shape[0] // 2
    split = np.concatenate((values[:length], values[-length:]), axis=1)
    within = float(np.mean(np.var(split, axis=0, ddof=1)))
    between = float(length * np.var(np.mean(split, axis=0), ddof=1))
    if within == 0:
        return 1. if between == 0 else None
    return float(np.sqrt(((length - 1.) / length * within + between / length) / within))


def find_configuration(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        path = directory / 'run_config.json'
        if path.is_file():
            metadata = json.loads(path.read_text())
            return path, metadata, metadata.get('configuration', metadata)
    raise FileNotFoundError(f'No run_config.json next to {checkpoint}')


def checkpoint_identity(checkpoint, run_metadata):
    with np.load(checkpoint, allow_pickle=False) as archive:
        if 'metadata' not in archive.files:
            return dict(kind='parameter_only', step=None, scientific_signature_checked=False)
        raw = archive['metadata']
        if raw.shape != () or raw.dtype.kind not in 'US':
            raise ValueError('Invalid checkpoint metadata scalar')
        record = json.loads(raw.item())
    expected = run_metadata.get('config_sha256')
    actual = record.get('config_sha256')
    if expected is not None and actual != expected:
        raise ValueError('Checkpoint signature differs from run metadata')
    return dict(kind='full_checkpoint', step=record.get('step'),
                config_sha256=actual, scientific_signature_checked=expected is not None)


def validate_sector(occupations, ne, flux, two_lz):
    occupations = np.asarray(occupations)
    target = (ne * flux + two_lz) // 2
    if (occupations.ndim != 2 or occupations.shape[1] != flux + 1
            or np.any((occupations != 0) & (occupations != 1))
            or np.any(occupations.sum(axis=1) != ne)
            or np.any(occupations @ np.arange(flux + 1) != target)):
        raise ArithmeticError('Held-out configurations left the fixed-N,Lz sector')


def evaluate(checkpoint, output, *, seed=29071, chains=256, packs=32, sweeps=10,
             burn_sweeps=1000, groups=16, beta=None, backend='gpu', cpus=4,
             pair_chunk_size=2, refresh_every=4):
    from .train import _set_runtime
    _set_runtime(dict(native_threads=1, backend=backend, cpus=cpus))
    if backend == 'gpu':
        import os
        os.environ['JAX_PLATFORMS'] = 'cuda'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from scipy.special import comb
    from scipy.stats import t as student_t
    from . import model
    from .evaluate import read_parameters
    from .physics import SphereOperator
    from .sampler import make_sampler, global_refresh
    from .local_energy import make_local_energy
    from .north_droplet import profile_a_epsilon, profile_sha256, make_confinement_local_energy
    from .checkpoint import atomic_json, atomic_npz

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    samples_path = output.with_name(output.stem + '.samples.npz')
    if output.exists() or samples_path.exists():
        raise FileExistsError(f'Preserving existing evaluation {output}')
    for name, value, minimum in (('chains', chains, 2), ('packs', packs, 4),
                                 ('sweeps', sweeps, 1), ('burn_sweeps', burn_sweeps, 0),
                                 ('groups', groups, 2), ('seed', seed, 0),
                                 ('refresh_every', refresh_every, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'Invalid {name}')
    if groups > chains or chains % groups:
        raise ValueError('Require chain count divisible by independent-chain groups')
    run_path, run, config = find_configuration(checkpoint)
    ne = int(config['ne'])
    flux = int(config.get('flux') if config.get('flux') is not None else 3 * (ne - 1))
    two_lz, v3 = int(config.get('two_lz', 0)), float(config.get('v3', 0.))
    strength, wall_x = float(config.get('confinement_strength', 0.)), float(config.get('wall_x', .88))
    beta = float(config.get('beta', .5) if beta is None else beta)
    if not 0 < beta <= 1:
        raise ValueError('Require 0 < evaluation beta <= 1')
    if seed == config.get('seed'):
        raise ValueError('Choose an evaluation seed distinct from the training sampling seed')
    before_checkpoint, before_metadata = digest(checkpoint), digest(run_path)
    identity = checkpoint_identity(checkpoint, run)
    params = {name: jnp.asarray(value) for name, value in read_parameters(checkpoint).items()}
    architecture = model.architecture(params)
    if architecture['orbitals'] != flux + 1:
        raise ValueError('Network input size differs from the configured orbital count')
    epsilon = profile_a_epsilon(flux, strength, wall_x)
    profile_hash = profile_sha256(epsilon)
    saved_profile = run.get('orbital_profile_sha256', config.get('orbital_profile_sha256'))
    if saved_profile is not None and saved_profile != profile_hash:
        raise ValueError('Configured edge potential differs from saved run profile')
    operator = SphereOperator(ne, flux=flux, two_lz=two_lz, v3=v3)
    log_fn = model.log_psi if beta == 1. else lambda p, n: beta * model.log_psi(p, n)
    sampler = make_sampler(log_fn, ne, flux)
    interaction = make_local_energy(operator, model.log_psi, pair_chunk_size=pair_chunk_size)
    local = make_confinement_local_energy(interaction, epsilon)

    @jax.jit
    def measurements(parameters, walkers):
        values = local(parameters, walkers)
        wall = walkers @ jnp.asarray(epsilon)
        return (model.batch_log_psi(parameters, walkers).real, values['energy'],
                values['energy'] - wall, wall)

    rng = np.random.default_rng(seed)
    key_seed = int(seed ^ 0x6DCE1AB3)
    key = jax.random.PRNGKey(key_seed)
    walkers = jnp.asarray(operator.sample_uniform(rng, chains), dtype=jnp.float64)
    validate_sector(np.asarray(walkers), ne, flux, two_lz)
    device = str(jax.devices()[0])
    if backend == 'gpu' and not device.lower().startswith('cuda:'):
        raise RuntimeError(f'GPU evaluation requested but device is {device}')
    started = time.perf_counter()
    burn_started = time.perf_counter()
    burn_acceptance = []
    remaining = burn_sweeps
    while remaining:
        count = min(100, remaining)
        walkers, key, acceptance = sampler(params, walkers, key, count)
        proposals = jnp.asarray(operator.sample_uniform(rng, chains), dtype=jnp.float64)
        walkers, key, _ = global_refresh(log_fn, params, walkers, proposals, key)
        burn_acceptance.append(float(acceptance))
        remaining -= count
    jax.block_until_ready((walkers, key))
    burn_seconds = time.perf_counter() - burn_started
    compile_started = time.perf_counter()
    measured = measurements.lower(params, walkers).compile()
    warm = measured(params, walkers)
    jax.block_until_ready(warm)
    compile_seconds = time.perf_counter() - compile_started
    samples, amplitudes, energies, interactions, walls, acceptances, refreshes = [], [], [], [], [], [], []
    production_started = time.perf_counter()
    for pack in range(packs):
        if refresh_every and pack % refresh_every == 0:
            proposals = jnp.asarray(operator.sample_uniform(rng, chains), dtype=jnp.float64)
            walkers, key, accepted = global_refresh(log_fn, params, walkers, proposals, key)
            refreshes.append(float(accepted))
        walkers, key, acceptance = sampler(params, walkers, key, sweeps)
        values = jax.device_get(measured(params, walkers))
        occupations = np.asarray(walkers, dtype=np.uint8)
        validate_sector(occupations, ne, flux, two_lz)
        if not all(np.isfinite(value).all() for value in values):
            raise ArithmeticError('Nonfinite held-out local energies or amplitudes')
        samples.append(occupations)
        amplitudes.append(values[0]); energies.append(values[1])
        interactions.append(values[2]); walls.append(values[3])
        acceptances.append(float(acceptance))
        print(f'held-out pack {pack + 1}/{packs}; acceptance={acceptances[-1]:.4f}', flush=True)
    production_seconds = time.perf_counter() - production_started
    samples, amplitudes, energies, interactions, walls = map(np.asarray,
        (samples, amplitudes, energies, interactions, walls))
    weights = importance_weights(amplitudes, beta)
    energy_stats = cluster_ratio_statistics(weights, energies, groups)
    interaction_stats = cluster_ratio_statistics(weights, interactions, groups)
    wall_stats = cluster_ratio_statistics(weights, walls, groups)
    occupation_stats = cluster_ratio_statistics(weights, samples, groups)
    mean = complex(energy_stats['estimate'])
    second = float(np.sum(weights * abs(energies)**2))
    # These centered forms equal Q-(Re E)^2 and Q-|E|^2, respectively,
    # without subtracting nearly equal large numbers at an eigenstate.
    physical_variance_influence = abs(energies - mean.real)**2
    centered_influence = abs(energies - mean)**2
    physical_variance = float(np.sum(weights * physical_variance_influence))
    centered_variance = float(np.sum(weights * centered_influence))
    physical_variance_stats = cluster_ratio_statistics(weights, physical_variance_influence, groups)
    centered_variance_stats = cluster_ratio_statistics(weights, centered_influence, groups)
    occupations = np.asarray(occupation_stats['estimate'], dtype=float)
    integral, angular = float(occupations.sum()), float(occupations @ (np.arange(flux + 1) - flux / 2))
    if abs(integral - ne) > 1e-9 or abs(angular - two_lz / 2) > 1e-9:
        raise ArithmeticError('Held-out weighted density violates fixed N or Lz')
    theta = np.linspace(0., np.pi, 181)
    a = np.arange(flux + 1)
    p = np.cos(theta[:, None] / 2)**2
    kernels = comb(flux, a)[None, :] * p**a * (1 - p)**(flux - a)
    density = (flux + 1) / ne * (kernels @ occupations)
    critical = float(student_t.ppf(.975, groups - 1))
    pack_results = []
    for pack in range(packs):
        w = weights[pack] / weights[pack].sum()
        estimate = complex(w @ energies[pack])
        pack_results.append(dict(pack=pack, energy_real_V1=estimate.real,
            energy_imaginary_V1=estimate.imag, importance_ess=float(1. / np.sum(w**2)),
            acceptance=acceptances[pack], interaction_real_V1=float((w @ interactions[pack]).real),
            wall_V1=float(w @ walls[pack])))
    if digest(checkpoint) != before_checkpoint or digest(run_path) != before_metadata:
        raise RuntimeError('Frozen checkpoint or run metadata changed during evaluation')
    atomic_npz(samples_path, occupations=samples, log_amplitudes=amplitudes,
               local_energies=energies, interaction_local_energies=interactions,
               wall_local_energies=walls, normalized_importance_weights=weights)
    output_result = dict(status='completed', checkpoint=str(checkpoint),
        checkpoint_sha256=before_checkpoint, checkpoint_identity=identity,
        run_config=str(run_path), run_config_sha256=before_metadata,
        ne=ne, flux=flux, two_lz=two_lz, v3=v3, sector_dimension=operator.dimension(),
        orbitals=flux + 1, architecture=architecture, configuration=config,
        confinement_strength_V1=strength, wall_x=wall_x,
        epsilon_a_V1=epsilon.tolist(), orbital_profile_sha256=profile_hash,
        saved_profile_verified=saved_profile is not None,
        proposal='q_beta proportional to |psi|^(2 beta)', beta=beta,
        seed=seed, numpy_seed=seed, jax_seed=key_seed,
        independent_of_training_walkers_and_rng=True,
        chains=chains, packs=packs, samples=int(chains * packs),
        sweeps_between_packs=sweeps, burn_sweeps=burn_sweeps,
        uniform_sector_refresh_every_packs=refresh_every,
        chain_groups=groups, chains_per_group=chains // groups,
        energy_real_V1=mean.real, energy_imaginary_V1=mean.imag,
        energy_mcse_V1=float(energy_stats['mcse_real']),
        imaginary_energy_mcse_V1=float(energy_stats['mcse_imag']),
        approximate_energy_95_interval_V1=[mean.real - critical * float(energy_stats['mcse_real']),
                                           mean.real + critical * float(energy_stats['mcse_real'])],
        ci_degrees_of_freedom=groups - 1,
        second_energy_moment_V1_squared=second,
        physical_variance_V1_squared=float(physical_variance),
        physical_variance_mcse_V1_squared=float(physical_variance_stats['mcse_real']),
        physical_variance_definition='SN E_q[w |Eloc|^2] - (Re SN E_q[w Eloc])^2',
        variance_formula_roundoff_difference_V1_squared=float(abs(physical_variance - (second - mean.real**2))),
        centered_complex_residual_variance_V1_squared=float(max(0., centered_variance)),
        centered_variance_mcse_V1_squared=float(centered_variance_stats['mcse_real']),
        interaction_energy_V1=float(interaction_stats['estimate'].real),
        interaction_energy_mcse_V1=float(interaction_stats['mcse_real']),
        wall_energy_V1=float(wall_stats['estimate'].real),
        wall_energy_mcse_V1=float(wall_stats['mcse_real']),
        energy_decomposition_error_V1=float(abs(interaction_stats['estimate'] + wall_stats['estimate'] - mean)),
        importance_ess=float(1. / np.sum(weights**2)),
        importance_ess_scope='Weight concentration only; not autocorrelation-adjusted sample count',
        mean_acceptance=float(np.mean(acceptances)),
        burn_mean_acceptance=float(np.mean(burn_acceptance)) if burn_acceptance else None,
        uniform_refresh_mean_acceptance=float(np.mean(refreshes)) if refreshes else None,
        basic_split_rhat_q_observables=dict(
            log_amplitude=basic_split_rhat(amplitudes),
            real_local_energy=basic_split_rhat(energies.real),
            wall_local_energy=basic_split_rhat(walls)),
        rhat_scope='Basic split Rhat on short q-chain traces, not rank-normalized or proof of equilibration',
        orbital_occupations=occupations.tolist(),
        orbital_occupation_mcse=np.asarray(occupation_stats['mcse_real']).tolist(),
        density_particle_number=integral, density_Lz=angular,
        theta_degrees=np.rad2deg(theta).tolist(), density_relative_to_uniform=density.tolist(),
        density_convention='a=m+flux/2; north selects a=flux; density divided by Ne/(4 pi)',
        per_pack=pack_results,
        group_energy_real_V1=np.asarray(energy_stats['group_ratios']).real.tolist(),
        group_energy_imaginary_V1=np.asarray(energy_stats['group_ratios']).imag.tolist(),
        group_normalization_weights=np.asarray(energy_stats['group_denominators']).tolist(),
        samples_file=samples_path.name, samples_sha256=digest(samples_path),
        evaluator_sha256=digest(__file__), actual_device=device,
        burn_seconds_including_sampler_compile=burn_seconds,
        local_energy_compile_seconds=compile_seconds,
        production_seconds=production_seconds, total_seconds=time.perf_counter() - started,
        no_ED_used=True, inputs_unchanged=True,
        uncertainty_scope='Delta-method SN ratio MCSE across independent chain groups, keeping all temporal packs of each chain together. Assumes independently initialized stationary chains; burn bias, unseen tails and finite-SN-ratio bias are not covered.',
        accuracy_scope='Matrix-free frozen-state MC energy and variance; no ED ground energy, fidelity or convergence certification')
    if v3 == 0. and flux == 3 * (ne - 1) and two_lz == 0:
        trial = ne * strength * (1 - wall_x)
        output_result.update(laughlin_trial_energy_V1=trial,
            energy_minus_laughlin_trial_V1=mean.real - trial,
            trial_reference_scope='Analytic uniform-density Laughlin trial: interaction zero and wall expectation Ne*U*(1-xc). Variational upper bound to this sector ground energy, not its exact value or training target.')
    # Enforce strict numeric JSON before the shared atomic writer.
    json.dumps(output_result, allow_nan=False)
    atomic_json(output, output_result)
    print(f'Held-out E/V1={mean.real:.9g} +/- {float(energy_stats["mcse_real"]):.3g}; '
          f'ImE={mean.imag:.3g}; variance={physical_variance:.6g}; '
          f'weightESS={output_result["importance_ess"]:.1f}/{chains*packs}', flush=True)
    return output_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=29071)
    parser.add_argument('--chains', type=int, default=256)
    parser.add_argument('--packs', type=int, default=32)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--burn-sweeps', type=int, default=1000)
    parser.add_argument('--groups', type=int, default=16)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--backend', choices=('cpu', 'gpu', 'auto'), default='gpu')
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--pair-chunk-size', type=int, default=2)
    parser.add_argument('--refresh-every', type=int, default=4)
    args = vars(parser.parse_args())
    evaluate(**args)


if __name__ == '__main__':
    main()
