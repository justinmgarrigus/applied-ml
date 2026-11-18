"""Build the note's compact figures and freeze its local numerical evidence.

Run with the course GPU Python environment; this script uses only CPU NumPy,
SciPy and Matplotlib. It neither trains nor changes any experiment artifacts.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqh-edge-note-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaincc

HERE = Path(__file__).resolve().parent
COURSE = HERE.parents[4]
PROJECT = COURSE / 'Final_proj/SPRING_Haldane_FQHE'
RESULTS = PROJECT / 'results'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {
        'pilot': RESULTS / 'ne8_north_cap_local_20261109/summary.json',
        'long': RESULTS / 'ne8_north_cap_continue_20261109/summary.json',
        'short': RESULTS / 'ne8_north_cap_short_step_20261109/summary.json',
        'fine': RESULTS / 'ne8_north_cap_radius0p001_20261109/summary.json',
        'batch': RESULTS / 'ne8_north_cap_batch256_20261109/summary.json',
        'batch512': RESULTS / 'ne8_north_cap_batch512_20261109/summary.json',
        'diagnosis': RESULTS / 'ne8_north_cap_short_step_20261109/diagnosis.json',
        'reference': RESULTS / 'ne8_north_cap_local_20261109/reference_record.json',
        'previous_no_edge': RESULTS / 'ne8_l2_alpha1_local_20261109/summary.json',
        'earlier_no_edge': RESULTS / 'ne8_l2_local_20261109/summary.json',
    }
    data = {key: read(path) for key, path in paths.items()}
    phases = ('pilot', 'long', 'short', 'fine', 'batch', 'batch512')
    for phase in phases:
        assert data[phase]['status'] == 'complete', phase
        assert not data[phase].get('warnings', []), phase
    for phase in ('short', 'fine', 'batch'):
        assert all(data[phase]['matched_branch_checks'].values()), phase
    provenance = {key: {'path': str(path.relative_to(COURSE)), 'sha256': digest(path)}
                  for key, path in paths.items()}
    arms = []
    exact_fields = {
        'step': 'checkpoint_step',
        'energy_V1': 'exact_variational_energy_V1',
        'target_fidelity': 'ground_state_fidelity',
        'energy_excess_V1': 'energy_excess_V1',
        'variance_V1_squared': 'exact_residual_variance_V1_squared',
    }
    for phase in phases:
        for arm in data[phase]['arms']:
            point = arm['endpoint']
            evaluation = paths[phase].parent / point['evaluation_file']
            exact = read(evaluation)
            assert digest(Path(exact['checkpoint'])) == exact['checkpoint_sha256'] == point['checkpoint_sha256']
            assert all(point[key] == exact[field] for key, field in exact_fields.items()), (phase, arm['arm'])
            arms.append(dict(phase=phase, arm=arm['arm'], config=arm['configuration'],
                             endpoint=point, evaluation=str(evaluation.relative_to(COURSE)),
                             evaluation_sha256=digest(evaluation)))
    metadata = dict(status='verified', source_records=provenance, arms=arms,
                    physical_wall='10 V1 Theta(x-.88), x=sin(theta/2)^2',
                    comparison_scopes={
                        'fine': data['fine']['initialization_scope'],
                        'batch': data['batch']['initialization_scope'],
                        'batch512': data['batch512']['comparison_scope'],
                    },
                    hypotheses='Wall sharpness is untested. The matched 128/256 comparison supports batch-size influence in one continuation; it does not establish gradient-noise variance or a general convergence cause. The 512 result is a single additional continuation, without a new matched 256 control.',
                    training_launched=False)
    (HERE / 'note_evidence.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'pdf.fonttype': 42})
    flux, strength, cutoff = 23, 10., .88
    k = np.arange(flux + 1)
    mean = (k + 1) / (flux + 2)
    epsilon = strength * betaincc(k + 1, flux - k + 1, cutoff)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), layout='constrained')
    ax = axes[0]
    ax.plot([0, cutoff, cutoff, 1], [0, 0, strength, strength], color='#b52c39', lw=1.6,
            label=r'Real-space $U(x)/V_1$')
    ax.plot(mean, epsilon, 'o-', ms=3, lw=1.2, color='#245f9d', label=r'Projected $\epsilon_k/V_1$ at $\langle x\rangle_k$')
    ax.axvspan(cutoff, 1, color='#b52c39', alpha=.08)
    ax.set(xlim=(0, 1), ylim=(-.3, 11), xlabel=r'$x=\sin^2(\theta/2)$', ylabel=r'Energy / $V_1$', title='Sharp spatial wall; smooth orbital energies')
    ax.legend(loc='upper left'); ax.grid(alpha=.2)
    ax = axes[1]
    selected = np.arange(17, 24)
    ax.plot(selected, epsilon[selected], 'o-', ms=4, lw=1.4, color='#245f9d')
    for index in (21, 22, 23):
        ax.annotate(f'{epsilon[index]:.2f}', (index, epsilon[index]), xytext=(-5, 7), textcoords='offset points', ha='right', fontsize=8)
    ax.set(xlim=(16.7, 23.5), ylim=(-.3, 11), xlabel=r'Outward orbital index $k$', ylabel=r'$\epsilon_k/V_1$', title=r'Finite tails near $\langle x\rangle_{21}=0.88$')
    ax.set_xticks(selected); ax.grid(alpha=.2)
    for extension in ('pdf', 'png'):
        fig.savefig(HERE / f'wall_profile.{extension}', dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), layout='constrained')
    colors = {.01: '#245f9d', .1: '#ca6c19'}
    for arm in data['short']['arms']:
        radius = arm['max_norm']; points = arm['trajectory']
        updates = np.array([point['additional_updates'] for point in points]) / 1000
        label = f'Radius {radius:g}'
        axes[0].plot(updates, [point['energy_excess_V1'] for point in points], 'o-', ms=3, lw=1.4, color=colors[radius], label=label)
        axes[1].plot(updates, [100 * point['target_fidelity'] for point in points], 'o-', ms=3, lw=1.4, color=colors[radius], label=label)
    axes[0].set(ylabel=r'Exact $(E-E_0)/V_1$', ylim=(.088, .143), title='Energy error after the shared checkpoint')
    axes[1].set(ylabel='Exact own-target fidelity (%)', ylim=(88., 93.), title='Fidelity at frozen checkpoints')
    for ax in axes:
        ax.set_xlabel('Additional updates (thousands)'); ax.set_xticks([0, 5, 10, 15, 20]); ax.grid(alpha=.2)
    axes[0].legend(loc='upper right')
    for extension in ('pdf', 'png'):
        fig.savefig(HERE / f'short_step_comparison.{extension}', dpi=220)
    plt.close(fig)
    print(json.dumps({'status': 'verified', 'source_records': len(paths), 'frozen_endpoints': len(arms), 'figures': 2}))


if __name__ == '__main__':
    main()
