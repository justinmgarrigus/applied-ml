# Haldane FQHE with SPRING

A portable implementation of Born-sampled SPRING training. The default experiment
continues the best retained N=8 dense network for **200,000 optimizer updates**.
This is a training budget, not a claim that the ground state has converged.
No long run or remote job was launched during this cleanup.

## Physics and learning problem

The retained calculation is the fully spin-polarized Haldane sphere, projected
strictly into the lowest Landau level (LLL). At filling 1/3 the flux is
`N_phi = 3*(N-1)`. Orbital `a=0,...,N_phi` has `m=a-N_phi/2`; the calculation
fixes particle number and total `L_z=0`. The input is its binary occupation
vector. For N=8 this has 22 columns, and the sector contains 8,512 Slater states.
Fermionic antisymmetry is carried by creation operators in ascending orbital
order and by the corresponding signs in every Hamiltonian connection.

The default Hamiltonian is `H/V1 = sum_(i<j) P_(R=1)(i,j)`, with normalized pair
projectors and no energy shift. It has a zero-energy Laughlin ground state at
this flux. The code also supports an optional `v3 * sum P_(R=3)` term; the
shipped configurations set `v3=0`. Higher Landau levels, LL mixing, a Laughlin
factor, and an `L=0` projection are absent. Adding a hidden layer changes the
variational coefficients, not this Hamiltonian or basis.

The network outputs the log magnitude and phase of a complex occupation-basis
coefficient. Its objective is the variational energy

```
E(theta) = <psi_theta|H|psi_theta> / <psi_theta|psi_theta>
E_local(n) = sum_n' H[n,n'] psi_theta(n') / psi_theta(n)
p_theta(n) = |psi_theta(n)|^2 / sum_n' |psi_theta(n')|^2
```

Sector-preserving Metropolis pair moves and uniform-sector independence
refreshes target this Born distribution. There is no artificial temperature.
Uniform-sector draws initialize or propose moves; they are not a replacement
for the Born-weighted energy average. A batch of 32 contains 32 current walker
rows and may include duplicates or correlated samples.

Optimization uses the upstream JaQMC implementation of SPRING, not ordinary
SGD. Centered log-derivative rows define the sampled metric, and the force is
`Re Cov(O*, E_local)`, half the physical energy gradient. Defaults are learning
rate 0.01, damping 0.01 and SPRING memory coefficient 0.9. `max_norm=0.03` is
the **update radius sqrt(C)** in `dtheta = phi*min(eta, sqrt(C)/||phi||)`;
therefore `C=0.0009`. It is not a per-parameter clipping threshold.

## Retained code and results

| Path | Contents |
|---|---|
| `src/fqh_spring/` | One physics, model, Born sampler, SPRING, checkpoint and evaluation workflow |
| `configs/` | Four configurations with a 200,000-update target |
| `checkpoints/` | Common six-layer source and best dense pilot parameters, with hashes |
| `references/ne4` through `ne8` | Verified small-system sparse Hamiltonians and eigenpairs, for evaluation only |
| `results/ne8_architecture_pilot/` | All six retained endpoints, exact diagnostics and comparison figure |
| `results/ed_benchmarks/` | Fresh N=4–8 ED spectra, angular momenta, neutral gaps and checkpoint comparisons |
| `results/validation/` | Release tests and numerical reproduction checks |
| `tests/` | Physics, evaluator and full-state resume checks |
| `vendor/jaqmc_spring/` | Minimal pinned SPRING subset, original license and source hashes |

The models share a six-layer width-64 MLP. The `dense` variant adds two small
residual dense blocks; `attention` adds two orbital-attention blocks. New
correction readouts start at zero so both branches can begin at the same
wavefunction. Attention learns Fock-space coefficients; this is not a
real-space FermiNet implementation.

The matched 1,000-update pilot used batch 32 and sampling seeds 71 and 137:

| Model | Mean exact energy / V1 | Mean Laughlin fidelity |
|---|---:|---:|
| MLP | 0.49973545 | 35.4091% |
| Dense correction | 0.49158096 | 36.2938% |
| Attention correction | 0.50427475 | 34.6992% |

These means are descriptive results from two sampling seeds, with one neural
initialization. No model reached the ground state. The best individual dense
endpoint has energy `0.490844981335 V1`, fidelity `0.365698814` and
`<L^2>=30.0167563`. Attention did not improve this short pilot and cost about
five times as much. See the [pilot record](results/ne8_architecture_pilot/README.md)
for comparison details and limitations.

## Exact-diagonalization benchmarks

The [executed ED suite](results/ed_benchmarks/README.md) covers N=4,5,6,7,8 in
the complete fixed-M=0 LLL sectors. It saves low eigenpairs, L labels, neutral
gaps, uniform-density checks and comparisons with all six neural pilot states.
Each ED ground energy is zero within roundoff. At N=8 the first neutral gap is
`0.531576794705 V1` with L=5; the lowest positive L=0 level in the retained
spectrum is `1.056888888541 V1`. These are excitation energies above the zero
mode, not ground-state energies.

After installation, regenerate the benchmarks from this directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m fqh_spring.ed_benchmark --root . --sizes 4 5 6 7 8
```

The ED code is a separate diagnostic; training does not call it or read its
eigenvectors. Existing N=6–8 references and all training configurations retain
their previous values. The cross-model comparison with Ising is in the course
project's `Final_proj/benchmarks/` folder.

## Install and check

Use Python 3.12 and install on the target machine. Virtual environments are
excluded from Git; the commands below create one. The sibling Ising
project is not required. From the Git repository root:

```bash
cd Final_proj/SPRING_Haldane_FQHE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps ./vendor/jaqmc_spring
python -m pip install --no-deps -e .
python -m pip check
python -m fqh_spring.train --config configs/ne8_dense_200k.json \
  --steps 20 --output runs/smoke --cpus 4 --backend cpu
python -m fqh_spring.evaluate --checkpoint runs/smoke/checkpoint_latest.npz \
  --reference references/ne8 --output runs/smoke/exact_evaluation.json --angular
```

For the tests, run `python -m pip install 'pytest>=9,<10'`, then
`python -m pytest`. Release validation covers
fermionic signs, pair-projector normalization, all three models, exact density
normalization, and resumed versus uninterrupted updates. All six retained
endpoints reproduce their earlier diagnostics within `3e-14`.

## Long runs and continuation

After the local installation above, choose the starting wavefunction explicitly:

| Configuration | Starting wavefunction |
|---|---|
| `ne8_dense_200k.json` | Best trained dense pilot, preserving its learned correction |
| `ne8_dense_common_source_200k.json` | Common MLP source plus a new zero-readout dense correction |
| `ne8_attention_200k.json` | Same common source plus a new zero-readout attention correction |
| `ne8_mlp_200k.json` | Common MLP source without a correction |

Use the common-source dense and attention configurations for a matched
architecture comparison. The best-dense continuation answers the practical
question of whether the best retained state improves with a longer budget.
Parameter-only pilot sources start with new optimizer memory and walkers;
they do not claim to resume the old stochastic trajectory.

```bash
python -m fqh_spring.train --config configs/ne8_dense_200k.json \
  --steps 200000 --output runs/ne8_dense_seed71 --cpus 4 --backend cpu
# Continue an interrupted run, using the same configuration and directory:
python -m fqh_spring.train --config configs/ne8_dense_200k.json \
  --steps 200000 --output runs/ne8_dense_seed71 --cpus 4 --backend cpu --resume
```

`--steps` is the target total, not additional updates. Full checkpoints restore
parameters, SPRING memory, walkers, both random generators and update count.
Runtime source hashes and invariant settings must match for resume; moving the
whole project is allowed. Periodic checkpoints are atomic, only three historical
snapshots are retained, and metrics are written incrementally. SIGUSR1/SIGTERM
request a checkpoint after the current update. Each run uses one JAX device;
independent sampling seeds run separately.

Check exact energy, residual variance, ground-state overlap, angular momentum,
acceptance and agreement between seeds. A low sampled energy or a long run
alone does not establish convergence. ED references are never read by training.
For larger systems the trainer remains matrix-free; this release provides
full-sector reference diagnostics only for the retained small systems.

## Source and recovery

FQHE project paper: [arXiv:2412.00618](https://arxiv.org/abs/2412.00618).
Upstream implementation: [ByteDance JaQMC](https://github.com/bytedance/jaqmc),
pinned commit `71f94158f2aca88ded904451f5314f71cdff5666`.
The bundled subset records unchanged upstream files and its minimal import
wrapper in `vendor/jaqmc_spring/PROVENANCE.json`.

Exploratory scripts, sweeps and unselected results were moved intact to
`_project_archive/20261023_fqhe_cleanup/Haldane_FQHE` at the course repository
root. They are not required by this package. The archive includes a recovery
guide and file hashes; no earlier experiment was permanently deleted.
