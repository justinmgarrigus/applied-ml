# MinSR on FQHE with Cartesian vs exponential amplitudes

This study tests the anchored Cartesian amplitude map that `main` adopted in
PR #6 ([`73b18a8`](https://github.com/justinmgarrigus/applied-ml/commit/73b18a8bbaa4501842818bd42e0595027b6b1fc5))
for damped MinSR. The network output $`\ell_\theta(n)`$ is either used as
$`\psi=e^{\ell}`$ (exponential) or as
$`u=e^{\ell_0}(1+\ell-\ell_0)`$ (Cartesian, anchored at the frozen initial
output $`\ell_0`$). Everything else is shared: pure $`V_1`$, lowest Landau
level, $`L_z=0`$, the six-layer 64-wide all-layer-readout MLP, Born-sampled
MCMC, and MinSR with $`\mu=0`$.

## Why it was tested

In the earlier Ne4 study, every MinSR, SPRING and SGD run stalled near
$`E\approx0.87V_1`$ and fidelity 0.39. The full 18-state analysis of those
checkpoints showed that the hidden tanh units were not saturated. Instead,
the exponential output collapsed onto about three configurations. The largest
ground-state component was driven to $`|\psi|^2\sim10^{-5}`$. With
$`\partial\psi/\partial\theta=\psi\,\partial\ell/\partial\theta`$, the
gradient for a small amplitude shrinks with that amplitude. The exact
Born-weighted control collapsed the same way, so sampling noise was not the
cause. The Cartesian derivative keeps the fixed scale $`e^{\ell_0}`$ and can
pass through zero.

## Protocol

Same structure as the paired Ising study. Each grid setting runs on two tuning
seed pairs, (7,71) and (11,111). The setting with the lowest median final exact
energy is frozen. Production then uses five new paired seeds, 17, 29, 43, 59
and 83, with sampling seeds `10*i+1`. The exponential control reuses the
production seeds and the setting frozen for the Cartesian map. Both maps start
from identical initial parameters for every seed (verified in `all_runs.csv`).
The target is $`(E-E_0)/\Delta\le0.01`$, with $`\Delta`$ the first gap in the
complete $`L_z=0`$ sector.

| | Ne4 pilot | Ne8 main study |
|---|---|---|
| Flux / sector dimension | 9 / 18 | 21 / 8,512 |
| Tuning grid | lr {0.01, 0.04} x damping {0.003, 0.03}, radius 0.03, B=32, 250 updates | lr {0.01, 0.05} x damping {0.001, 0.01, 0.1} x radius {0.03, 0.1}, B=128, 1,000 updates |
| Frozen | lr 0.04, damping 0.03, radius 0.03 | lr 0.05, damping 0.1, radius 0.1 |
| Production | 1,000 updates, and an extended 5,000-update repeat | 5,000 updates |
| Exact ED evaluation | every 100 (250 for extended) updates | every 100 updates |
| Hardware | CPU (i9-13900H), two cores per run | RTX 4060 Laptop GPU, three runs sharing it |

The Ne4 Cartesian tuning selected the same setting as the earlier exponential
study, so the Ne4 comparison changes only the amplitude map. The exponential
Ne4 control reproduces the earlier endpoints exactly (seed 17: $`E=0.8715`$,
$`F=0.3922`$).

## Results

Medians over five paired seeds. Energies are in units of $`V_1`$; $`E_0=0`$.

| Case | Map | Final $`E`$ (range) | Fidelity | Effective support | Training time | At target |
|---|---|---:|---:|---:|---:|---:|
| Ne4, 1,000 updates | Cartesian | 0.51 (0.36–0.58) | 0.42 | 3.8 / 18 | 128 s CPU | 0/5 |
| | Exponential | 0.87 (0.55–0.96) | 0.39 | 3.0 / 18 | 48 s CPU | 0/5 |
| Ne4, 5,000 updates | Cartesian | 0.25 (0.23–0.39) | 0.69 | 6.2 / 18 | 638 s CPU | 0/5 |
| | Exponential | 0.49 (0.24–0.86) | 0.63 | 4.6 / 18 | 238 s CPU | 0/5 |
| Ne8, 5,000 updates | Cartesian | 2.31 (2.04–2.65) | 0.027 | 4.9 / 8,512 | 1,043 s GPU | 0/5 |
| | Exponential | 1.31 (1.11–1.35) | 0.058 | 5.9 / 8,512 | 579 s GPU | 0/5 |

- **Ne4.** Cartesian has a lower median energy at every checkpoint. It wins
  5/5 seed pairs at 1,000 updates and 4/5 at 5,000. The exponential runs
  leave their plateau only in sudden steps. The Cartesian update costs about
  2.7 times as much on the CPU. At equal training time (about 240 s), the two
  maps are tied; Cartesian pulls ahead because it keeps improving.
- **Ne8.** Exponential wins all five seed pairs on energy and fidelity and
  needs about half the GPU time. The frozen setting was tuned for the
  Cartesian map, so this result does not favor the exponential control.
- **Interpretation.** The Ne8 ground state has 3,220 exact zeros among 8,512
  coefficients. Its nonzero $`|c|^2`$ values span about 10 orders of
  magnitude. At Ne4 the range is about 2 orders. Exact zeros favor the
  Cartesian map, while a 10-decade dynamic range is a shift of about 23 in
  $`\mathrm{Re}\,\ell`$ for the exponential map. The linear Cartesian map
  would need very large output changes for the same range. In both maps the
  Ne8 state collapses to about five configurations, compared with an
  effective support of 641 in the exact ground state, and Metropolis
  acceptance falls to a few percent.
- No FQHE run reached the target. These are fixed-budget comparisons.

Limitations: the Ne8 frozen setting lies at the corner of the tuning grid
(largest learning rate, damping and radius), so the grid should be widened.
Concurrent GPU jobs share one device, so their times are comparable to each
other but not to an isolated run. The last exponential seed ran alone in 247 s.

## Files

- `scripts/harness.py`: training launcher and complete-sector evaluator for
  both maps. The final metrics are cross-checked against `fqh_spring.evaluate`.
- `scripts/study.py`: tuning, freezing, production, exponential control and
  the extended Ne4 phase. Completed jobs are skipped on resume.
- `scripts/summarize.py`, `scripts/timing.py`: tables, figures and the
  CPU/CUDA timing check.
- `results/summary.csv`, `results/ne*_runs.csv`, `results/all_runs.csv`:
  aggregate, seed-level and all 62 runs.
- `results/curves.csv.gz`: every exact-evaluation trajectory.
- `results/ne*_minsr_cartesian_vs_exponential.{pdf,png}`: energy against
  updates and training time, fidelity and effective support.
- `results/ne*_{protocol,frozen_settings,tuning_table}.json`,
  `results/ne8_gpu_cpu_timing.json`, `results/environment_cuda12.txt`.

The CPU/CUDA timing run gave identical Ne8 energies on both devices
(difference about $`10^{-11}`$). It measured 0.029, 0.079 and 0.45 s per update on CUDA at
B = 32, 128 and 512, and 0.067 and 0.27 s on CPU at B = 32 and 128.

## Reproduction

The scripts expect the `main` source at `73b18a8` as `source/Final_proj`
next to them and the two environments described in `../README.md`.

```bash
git worktree add /tmp/src-73b18a8 73b18a8
mkdir run && cp Final_proj/MinSR/experiments/cartesian/scripts/*.py run/
ln -s /tmp/src-73b18a8 run/source
cd run
python study.py --ne 4 --output "$PWD/runs/ne4"
python study.py --ne 4 --output "$PWD/runs/ne4" --phases extend
python study.py --ne 8 --output "$PWD/runs/ne8"   # trains on CUDA
python summarize.py
```

`harness.py` looks for the CPU and CUDA interpreters at `../.venv` and
`../.venv-gpu` relative to the run folder; edit `PYTHON` there for another layout.
