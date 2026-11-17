"""Profile-checked exact diagnostics for the fixed-geometry GG-SPRING wall pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .north_droplet import profile_a_epsilon, profile_sha256
from .references import load_reference, sha256

PROJECT = Path(__file__).resolve().parents[2]


def find_run_metadata(checkpoint):
    """A milestone checkpoint lives one level below the run metadata."""
    checkpoint = Path(checkpoint)
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        path = directory / 'run_config.json'
        if path.is_file():
            metadata = json.loads(path.read_text())
            return path, metadata, metadata.get('configuration', metadata)
    raise FileNotFoundError(f'No run_config.json adjacent to {checkpoint}')


def check_profile(checkpoint, reference):
    """Reject a physically different reference before any NN evaluation."""
    path, run, config = find_run_metadata(checkpoint)
    metadata = reference['metadata']
    for key, expected in dict(ne=7, flux=18, two_lz=0, v3=0.).items():
        if config.get(key) != expected or metadata.get(key) != expected:
            raise ValueError(f'Run/reference disagree with the fixed N=7 model: {key}')
    strength = float(config['confinement_strength'])
    wall_x = float(config['wall_x'])
    epsilon = profile_a_epsilon(18, strength, wall_x)
    expected = profile_sha256(epsilon)
    saved = run.get('orbital_profile_sha256', config.get('orbital_profile_sha256'))
    if saved != expected:
        raise ValueError('Run orbital-profile hash disagrees with its configuration')
    target_strength = float(metadata.get('confinement_strength_V1', 0.))
    if strength != target_strength:
        raise ValueError('Checkpoint wall strength differs from the ED Hamiltonian')
    if strength:
        if (float(metadata.get('wall_x', -1.)) != wall_x
                or metadata.get('orbital_profile_sha256', metadata.get('profile_sha256')) != expected
                or not np.allclose(metadata.get('eps_a', []), epsilon, atol=0., rtol=0.)):
            raise ValueError('Checkpoint projected-wall profile differs from ED')
    elif np.any(epsilon):
        raise ArithmeticError('Zero-wall profile is nonzero')
    return dict(configuration=config, run_metadata=run, run_config_path=path,
                epsilon_a=epsilon, orbital_profile_sha256=expected,
                confinement_strength_V1=strength, wall_x=wall_x)


def evaluate(checkpoint, reference, output, *, base_reference=None, batch=256):
    from .evaluate import evaluate as standard_evaluate
    from .checkpoint import atomic_json

    checkpoint, reference, output = map(Path, (checkpoint, reference, output))
    base_reference = PROJECT / 'references/ne7' if base_reference is None else Path(base_reference)
    data, base = load_reference(reference), load_reference(base_reference)
    profile = check_profile(checkpoint, data)
    if (not np.array_equal(data['bits'], base['bits'])
            or any(base['metadata'].get(key) != value for key, value in
                   dict(ne=7, flux=18, two_lz=0, v3=0.).items())):
        raise ValueError('Base interaction reference has a different sector or basis')
    if float(base['metadata'].get('confinement_strength_V1', 0.)) != 0.:
        raise ValueError('Base interaction reference already contains a wall')
    diagonal = base['occupations'] @ profile['epsilon_a']
    from scipy import sparse
    delta = data['H'] - base['H'] - sparse.diags(diagonal, format='csr')
    difference = float(np.max(abs(delta.data), initial=0.))
    if difference > 1e-11:
        raise ArithmeticError('Reference Hamiltonian does not equal interaction plus configured wall')
    before_run_hash = sha256(profile['run_config_path'])
    result = standard_evaluate(checkpoint, reference, output, batch=batch, save_psi=True)
    coefficient_file = output.with_name(result['coefficient_file'])
    psi = np.load(coefficient_file, allow_pickle=False)
    probability = abs(psi)**2
    interaction = float(np.vdot(psi, base['H'] @ psi).real)
    wall = float(probability @ diagonal)
    total = result['exact_variational_energy_V1']
    decomposition_error = abs(interaction + wall - total)
    if decomposition_error > 1e-10:
        raise ArithmeticError('Frozen interaction plus wall energy does not match total')
    eigenvalues = np.sort(data['eigenvalues'])
    ground_energy = float(eigenvalues[0])
    gap = float(eigenvalues[1] - ground_energy)
    if gap <= 1e-8:
        raise ArithmeticError('ED ground is not resolved as unique in this sector')
    ground = data['eigenvectors'][:, int(np.argmin(data['eigenvalues']))]
    base_ground = base['eigenvectors'][:, int(np.argmin(base['eigenvalues']))]
    result.update(interaction_energy_V1=interaction, wall_energy_V1=wall,
                  confinement_energy_V1=wall, energy_decomposition_error_V1=decomposition_error,
                  ground_energy_V1=ground_energy, energy_error_V1=total - ground_energy,
                  fixed_sector_gap_V1=gap,
                  confinement_strength_V1=profile['confinement_strength_V1'], wall_x=profile['wall_x'],
                  orbital_profile_sha256=profile['orbital_profile_sha256'],
                  profile_sha256=profile['orbital_profile_sha256'],
                  targetchange_ground_state_fidelity=float(abs(np.vdot(base_ground, ground))**2),
                  original_no_wall_ground_state_fidelity=float(abs(np.vdot(base_ground, psi))**2),
                  base_reference=str(base_reference), base_reference_manifest_sha256=sha256(base_reference / 'manifest.json'),
                  run_config_sha256=before_run_hash, profile_checked_against_run_config=True,
                  hamiltonian_profile_difference_max_error=difference,
                  geometry_scope='Original N=7,F=18,Lz=0 sphere retained; no extra flux or orbital cutoff',
                  target_scope='Each fidelity and energy error uses the ED ground state of its own Hamiltonian')
    if sha256(profile['run_config_path']) != before_run_hash:
        raise RuntimeError('Run metadata changed while evaluating its frozen state')
    for directory, loaded in ((reference, data), (base_reference, base)):
        if any(sha256(directory / name) != expected for name, expected in loaded['manifest'].items()):
            raise RuntimeError('Reference inputs changed during frozen evaluation')
    atomic_json(output, result)
    print(f"wall={profile['confinement_strength_V1']:g}: E-E0={result['energy_error_V1']:.10g}; "
          f"Eint={interaction:.10g}; Ewall={wall:.10g}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-reference', type=Path, default=PROJECT / 'references/ne7')
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.reference, args.output,
             base_reference=args.base_reference, batch=args.batch)


if __name__ == '__main__':
    main()
