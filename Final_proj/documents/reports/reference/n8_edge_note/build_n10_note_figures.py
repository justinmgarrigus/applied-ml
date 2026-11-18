"""Build the N=10 additions from immutable complete-sector evaluations.

Run with the course .venv/bin/python. This script uses CPU plotting and file
validation only; it neither samples nor optimizes a wavefunction.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fqh-n10-edge-note-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COURSE = HERE.parents[4]
PROJECT = COURSE / 'Final_proj/SPRING_Haldane_FQHE'
RESULTS = PROJECT / 'results'
ROOTS = {
    'batch256': RESULTS / 'ne10_north_cap_120k_20261110',
    'batch300': RESULTS / 'ne10_north_cap_batch300_lr3em5_20261110',
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    records = {key: read(root / 'summary.json') for key, root in ROOTS.items()}
    evidence = {'status': 'verified', 'training_launched': False,
                'scope': 'Exact normalized frozen coefficients over all 473259 states in fixed N10 F29 K135 M10 sector',
                'comparison': 'One continuation, batch and learning rate change together at 120000; no matched control and no established convergence speedup',
                'inputs': {}, 'points': {}, 'reference_files': {}, 'training_sources': {}}
    fields = ('checkpoint_step', 'exact_variational_energy_V1', 'energy_excess_V1',
              'ground_state_fidelity', 'exact_residual_variance_V1_squared',
              'checkpoint_sha256', 'coefficient_file_sha256', 'training_seconds')
    for key, summary in records.items():
        assert summary['status'] == 'completed' and not summary['warnings']
        assert summary['ne'] == 10 and summary['flux'] == 29
        assert summary['sector_K'] == 135 and summary['two_lz'] == 20
        assert summary['alpha'] == 0 and summary['max_norm'] == .001
        assert summary['sector_dimension'] == 473259
        assert summary['parameter_count'] == 2114
        assert summary['matched_control_run'] is False
        evidence['inputs'][key] = {'path': str(ROOTS[key] / 'summary.json'),
                                  'sha256': digest(ROOTS[key] / 'summary.json')}
        verified = []
        for point in summary['points']:
            evaluation = Path(point['evaluation_file'])
            exact = read(evaluation)
            assert all(point[field] == exact[field] for field in fields)
            checkpoint = Path(exact['checkpoint'])
            coefficients = evaluation.parent / exact['coefficient_file']
            assert digest(checkpoint) == exact['checkpoint_sha256']
            assert digest(coefficients) == exact['coefficient_file_sha256']
            assert exact['ed_used_for_training'] is False
            assert exact['basis_dimension'] == 473259
            verified.append({field: exact[field] for field in fields} |
                            {'evaluation_file': str(evaluation),
                             'evaluation_sha256': digest(evaluation),
                             'checkpoint': str(checkpoint),
                             'coefficient_file': str(coefficients)})
        evidence['points'][key] = verified

    old, new = records['batch256'], records['batch300']
    assert [p['checkpoint_step'] for p in old['points']] == list(range(0, 120001, 2000))
    assert [p['checkpoint_step'] for p in new['points']] == list(range(120000, 130001, 2000))
    assert all(old['endpoint'][f] == new['before'][f] for f in fields if f != 'checkpoint_sha256')
    # The branch checkpoint includes new walkers and metadata but the same state.
    assert new['authorized_scientific_changes'] == {
        'batch': {'old': 256, 'new': 300},
        'learning_rate': {'old': .03, 'new': 3e-5},
    }
    reference = Path(old['reference'])
    assert old['reference'] == new['reference']
    assert digest(reference / 'manifest.json') == old['reference_manifest_sha256'] == new['reference_manifest_sha256']
    for filename, expected in read(reference / 'manifest.json').items():
        assert digest(reference / filename) == expected
        evidence['reference_files'][filename] = expected
    source_manifest = ROOTS['batch300'] / 'validation/source_hashes.json'
    for filename, expected in read(source_manifest).items():
        assert digest(PROJECT / filename) == expected
        evidence['training_sources'][filename] = expected
    evidence['inputs']['source_manifest'] = {'path': str(source_manifest), 'sha256': digest(source_manifest)}

    by_step = {p['checkpoint_step']: p for p in old['points']}
    by_step[130000] = new['endpoint']
    selected = (0, 2000, 10000, 20000, 40000, 80000, 120000, 130000)
    evidence['table'] = []
    lines = [r'\begin{tabular}{@{}rrrrrrrr@{}}', r'\toprule',
             r'Total updates & $B$ & $\eta$ & $E/\V$ & $\de/\V$ & $\fid$ (\%) & Variance$/\V^2$ & Active min.\\',
             r'\midrule']
    for step in selected:
        point = by_step[step]
        batch, rate = (256, .03) if step <= 120000 else (300, 3e-5)
        row = {'step': step, 'batch': batch, 'learning_rate': rate,
               'energy_V1': point['exact_variational_energy_V1'],
               'energy_excess_V1': point['energy_excess_V1'],
               'fidelity_percent': 100 * point['ground_state_fidelity'],
               'variance_V1_squared': point['exact_residual_variance_V1_squared'],
               'cumulative_active_minutes': point['training_seconds'] / 60}
        evidence['table'].append(row)
        fidelity = (r'$3.47\!\times\!10^{-5}$' if step == 0 else
                    r'$8.39\!\times\!10^{-5}$' if step == 2000 else
                    f"{row['fidelity_percent']:.5f}")
        cells = [f'{step:,}', str(batch), '$0.03$' if rate == .03 else r'$3\!\times\!10^{-5}$',
                 f"{row['energy_V1']:.6f}", f"{row['energy_excess_V1']:.6f}", fidelity,
                 f"{row['variance_V1_squared']:.6f}", f"{row['cumulative_active_minutes']:.2f}"]
        if step == 130000:
            cells = [r'\textbf{' + value + '}' if '$' not in value else r'{\boldmath ' + value + '}' for value in cells]
        lines.append(' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (HERE / 'n10_results_table.tex').write_text('\n'.join(lines) + '\n')

    plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 7.2, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 1.9), layout='constrained')
    for key, color, label in (
        ('batch256', '#245f9d', r'$B=256,\ \eta=0.03$'),
        ('batch300', '#c26920', r'$B=300,\ \eta=3\times10^{-5}$'),
    ):
        points = records[key]['points']
        steps = np.array([p['checkpoint_step'] for p in points]) / 1000
        axes[0].plot(steps, [p['energy_excess_V1'] for p in points], color=color,
                     lw=1.6, marker='o', ms=2.3 if key == 'batch300' else 1.5, label=label)
        axes[1].plot(steps, [100 * p['ground_state_fidelity'] for p in points],
                     color=color, lw=1.6, marker='o', ms=2.3 if key == 'batch300' else 1.5)
    axes[0].set(yscale='log', ylabel=r'Exact $(E-E_0)/V_1$', ylim=(.42, 24))
    axes[1].set(ylabel='Exact fidelity (%)', ylim=(-1.5, 64))
    axes[0].legend(loc='upper right', framealpha=.95, borderpad=.3, labelspacing=.25)
    axes[1].annotate('57.07%', xy=(130, 57.07036528807754), xytext=(86, 59.5),
                     fontsize=8, color='#9a4a12', arrowprops={'arrowstyle': '-', 'color': '#9a4a12', 'lw': .7})
    for ax in axes:
        ax.axvspan(120, 130, color='#c26920', alpha=.08, zorder=-1)
        ax.axvline(120, color='#777777', linestyle='--', lw=.7)
        ax.set(xlabel='Total optimizer updates (thousands)', xlim=(-2, 134))
        ax.set_xticks([0, 40, 80, 120])
        ax.grid(alpha=.18, which='major')
    for extension in ('pdf', 'png'):
        fig.savefig(HERE / f'n10_convergence.{extension}', dpi=220)
    plt.close(fig)
    evidence['outputs'] = {name: digest(HERE / name) for name in (
        'build_n10_note_figures.py', 'n10_results_table.tex', 'n10_convergence.pdf', 'n10_convergence.png')}
    (HERE / 'n10_note_evidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': 'verified', 'frozen_points': sum(map(len, evidence['points'].values())),
                      'table_rows': len(selected), 'reference_files': len(evidence['reference_files']),
                      'training_sources': len(evidence['training_sources'])}))


if __name__ == '__main__':
    main()
