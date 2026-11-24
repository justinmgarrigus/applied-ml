# Paired MinSR optimizer experiments

The study compares the existing damped MinSR, SPRING, and VMC-SGD updates on a
common variational Monte Carlo (VMC) objective. It measures accuracy and
computational cost across paired seeds. MinSR reached the proposal's accuracy
target in all 60 Ising runs. The Ne4 FQHE study reports fixed-budget accuracy and
cost because every sampled run ended above the target.

## Experimental protocol

We use \(r=(E-E_0)/\Delta_{\rm sector}\) with a target of \(r\leq 0.01\). ED
provides the reference energy and sector gap. At each evaluation, we measure
energy, residual variance, fidelity, and model-specific symmetry diagnostics.

- **Ising.** Six periodic transverse-field Ising settings, batches of 16 or 64
  independent Born draws, positive even-parity RBM with \(2N\) hidden units, and
  an even-sector gap. Each method and batch size had eight tuning settings and
  two tuning seeds at \(N=6,h/J=1\). Each tuning run used 1,000 updates. We
  froze the settings that minimized median final \(r\). Production used five
  new paired seeds, 3,000 updates, and exact evaluations every 20 updates.
- **FQHE.** Four electrons on the sphere in the lowest Landau level, flux 9,
  \(L_z=0\), pure \(V_1\), and a common six-layer, 64-wide complex MLP. Each
  method had four tuning settings, one tuning seed pair, and 250 updates per
  tuning run. Production used three new paired seeds, 1,000 updates,
  \(B=32\), two sweeps per update, and exact checkpoint evaluation every
  100 updates.

The comparisons use the same starting parameters, Hamiltonian, network,
sampler, and evaluation protocol within each case. MinSR uses the existing
JaQMC sample-space damped SR update with zero previous-direction memory.
MinSR and SPRING receive the half-gradient force, and SGD receives the full
physical gradient. K-FAC is outside the implemented comparison. The learning
curves and time-to-target values report the first observed crossing on the
recorded evaluation grid.

## Ising accuracy and cost

All 180 production runs crossed the target and finished below it. The
following results are medians across five paired seeds at \(h/J=1\).

| Spins | Batch | First target update MinSR / SPRING / SGD | Final energy error MinSR / SPRING / SGD (J) |
|---:|---:|---:|---:|
| 6 | 16 | 40 / 100 / 40 | 1.90e-4 / 6.77e-5 / 1.76e-4 |
| 6 | 64 | 60 / 40 / 40 | 2.21e-6 / near floating-point resolution / 1.45e-4 |
| 10 | 16 | 180 / 180 / 280 | 2.12e-3 / 1.23e-3 / 4.63e-3 |
| 10 | 64 | 100 / 60 / 100 | 5.46e-4 / 5.28e-4 / 1.29e-3 |

At \(N=10,B=16\), MinSR beat SGD on the final energy in all five seed pairs.
At \(N=10,B=64\), the two geometric methods had close median final errors
and each won at least two seed pairs. For MinSR at \(N=10\), increasing the
batch reduced the final error and the number of updates to target. It increased
the observed median time to target from about 55 to 80 ms. These measurements
include training and scheduled evaluation on the local CPU and separate JIT
warm-up. The complete tables also report comparisons at common wall time and
32,000 Born draws.

In the \(N=6,h/J=1,B=16\) memory ablation, \(\mu=0,0.5,0.9\) gave final
median \(r\) of \(9.17\times 10^{-5},4.79\times 10^{-5},1.81\times 10^{-5}\).
All three first crossed the target at a median of 40 updates. The exact
Born-weighted counterparts reached floating-point resolution by the final
update. The radius and damping ablations hold the other tuned settings fixed.

[All Ising results](results/ising_results.csv) ·
[Learning curves](results/ising_learning_curves.pdf) ·
[Seed-paired differences](results/ising_paired_differences.csv) ·
[Memory ablation](results/ising_memory_ablation.pdf) ·
[Damping and radius](results/ising_damping_radius.csv) ·
[Peak RSS](results/ising_memory_profiles.csv)

## FQHE accuracy and cost

| Method | Median final energy (\(V_1\)) | Median final fidelity | Median training time | Successes |
|---|---:|---:|---:|---:|
| MinSR | 0.872 | 0.392 | 20.6 s | 0/3 |
| SPRING | 0.865 | 0.396 | 21.1 s | 0/3 |
| SGD | 0.973 | 0.220 | 3.85 s | 0/3 |

The ground energy is zero to numerical accuracy and the target error is
approximately \(0.00681V_1\). The separately evaluated full-sector states
show that the geometric optimizers gave lower median errors and higher median
fidelity than SGD at this update budget. SGD used less time and peak resident
memory. The three-seed pilot has limited resolution for distinguishing MinSR
and SPRING. The exact Born-weighted MinSR control improved on all three
paired MCMC endpoints, but finished at \(0.461,0.852,0.852V_1\). Removing the
sampling noise was insufficient to reach the target within the tested budget.

[All FQHE results](results/fqhe_results.csv) ·
[Seed-level results](results/fqhe_runs.csv) ·
[Learning curves](results/fqhe_ne4_learning_curves.pdf) ·
[Exact Born-weighted control](results/fqhe_minsr_sampling_control.pdf)

## CPU and CUDA replay

The Ne4 runs were repeated on an RTX 4060 Laptop GPU. All nine CPU/CUDA pairs
had identical initial parameter arrays and post-burn-in walkers. The largest
absolute difference in endpoint fidelity was \(3.4\times10^{-12}\). The
medians below cover 1,000 updates and exclude JIT compilation.

| Method | CPU training | CUDA training |
|---|---:|---:|
| MinSR | 20.58 s | 7.85 s |
| SPRING | 21.12 s | 7.98 s |
| SGD | 3.85 s | 4.88 s |

The Ne6 CUDA feasibility run uses flux 15, 338 allowed configurations,
\(B=64\), one MinSR seed, and the Ne4-selected hyperparameters. It completed
1,000 updates in 17.4 s and ended at fidelity 0.099. It provides evidence
that the larger configuration runs on this device.

[Ne4 seed-level replay](results/gpu/ne4_cpu_gpu.csv) ·
[CUDA summary](results/gpu/summary.json) ·
[Ne6 configuration and endpoint](results/gpu/ne6_configuration.json)

## Initialization checks across protocols

The single-seed \(N=6,B=128\), 250-update replay reproduced the archived
SPRING endpoint. Changing only \(\mu\) from 0.9 to 0 gave errors of
\(1.58\times10^{-5} J\) for SPRING and \(1.71\times10^{-5} J\) for MinSR.
The published SA initial parameter arrays on the teammate branch match
the stock seed-7 RBM. SA uses the exact full-basis Rayleigh objective, and
the three-method production comparison uses sampled gradients. A paired
exact-objective SA comparison would need its own settings and budget.

[Replay curves](results/ising_archived_protocol_replay.csv) ·
[Paired initialization audit](validation/paired_initialization.json) ·
[Teammate SA results](https://github.com/justinmgarrigus/applied-ml/blob/3dd777c59bf0d499aa68ef587e7f96d3a5694aa3/Final_proj/SA/Ising/results/README.md)

## Reproduction and retained data

Run on Linux in the [shared Python 3.12 CPU environment](../../shared/README.md).
Pass a new output directory outside the repository. On the measured machine,
Ising used CPU IDs 0, 2, 4, and 6. FQHE production used IDs 8 and 10. The
launcher chooses available CPU IDs on other machines. Runtime and process RSS
measurements depend on the hardware and the CPU allocation.

```bash
bash Final_proj/MinSR/experiments/reproduce.sh /path/to/new-output
```

For the CUDA replay, use a separate Python 3.12 environment with the same
shared dependencies and vendored JaQMC. Install `jax[cuda12]==0.9.1` in that
environment. The launcher above copies the replay script alongside the CPU
artifacts, which supply the exact initial parameters. Run the following from
the populated output directory with the CUDA environment's Python.

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH JAX_PLATFORMS=cpu \
    python gpu_trial.py --output gpu_trials
```

Training subprocesses select CUDA explicitly and the evaluator uses CPU.
The [CUDA environment record](validation/environment_cuda12.txt) lists the
captured package versions.

The [results folder](results/) retains the frozen settings, aggregate tables,
Ising run index, compressed evaluation trajectories, and PDF figures. The
launcher recreates the per-run files, including fresh checkpoints, in the
designated output directory. The recorded source snapshot is
[`fb726d23`](https://github.com/justinmgarrigus/applied-ml/commit/fb726d230b5f9efe59a5f4b5f994aaff8c9af0a0).
The [validation folder](validation/) records the passing 44-test suite and the
frozen-batch MinSR direction check. The Ising study and Ne4 control scripts
combine the existing models and optimizer interfaces with experimental
measurement and analysis code.
