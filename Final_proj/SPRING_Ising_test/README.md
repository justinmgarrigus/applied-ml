# Small Ising gradient test with SPRING

This is an executable, CPU-only variational Monte Carlo (VMC) test using the
**actual JaQMC SPRING implementation**, rather than a replacement optimizer.
It uses a six-spin quantum Ising chain so that every derivative and final energy
can be checked against exact calculations. Your main course environment is unchanged.

## Run

For a fresh Git clone, install Python 3.12 and `uv`, then create the isolated
environment from this directory. Virtual environments are excluded from Git:

```bash
bash setup.sh
```

After setup, run from this directory:

```bash
.venv/bin/python ising_spring.py
.venv/bin/python -m pytest -q
```

From the course directory:

```bash
Final_proj/SPRING_Ising_test/.venv/bin/python Final_proj/SPRING_Ising_test/ising_spring.py
```

To rebuild the environment later, run `bash setup.sh` again.
The complete installed dependency versions are in `requirements-lock.txt`.
The first training step includes JAX compilation; no GPU is required.

Other small experiments:

```bash
# A four-site system; writes to results/spring_n4.
.venv/bin/python ising_spring.py --n 4 --hidden 8 --steps 150

# Remove SPRING memory to obtain the damped MinSR/SR control.
.venv/bin/python ising_spring.py --mu 0 --output results/minsr_n6

# Optional conventional SGD control. Its input is the full physical gradient.
.venv/bin/python ising_spring.py --optimizer sgd --learning-rate 0.01 --output results/sgd_n6
```

The SGD control needs its own learning-rate study for a fair performance comparison.
This project does not claim that SPRING universally beats SGD.

## Physics model and ML mapping

We use the **transverse-field quantum Ising model**, with periodic boundaries:

\[
H=-J\sum_{i=0}^{N-1}Z_iZ_{i+1}-h\sum_{i=0}^{N-1}X_i,
\qquad Z_N=Z_0.
\]

Here \(X,Z\) are Pauli matrices, with \(Z\) eigenvalues \(\pm1\), not spin
operators with eigenvalues \(\pm1/2\). Defaults are \(N=6,J=h=1\).
The transverse term flips individual spins and makes this a quantum ground-state
problem. There are only \(2^6=64\) configurations.

- **Input:** a spin configuration \(s=(s_0,\ldots,s_{N-1})\), each \(s_i=\pm1\).
- **Output:** a scalar log wavefunction amplitude \(f_\theta(s)=\log\psi_\theta(s)\).
- **Network:** an RBM amplitude with 12 hidden units and zero hidden/visible biases,
  \(f_\theta(s)=c+\sum_a\log\cosh(\sum_i s_iW_{ia})\).
  This is a nonlinear network with 72 trainable weights and one normalization
  parameter \(c\). The weights encode correlations among spins.
- **Symmetry:** the architecture enforces global spin-flip invariance,
  \(f(s)=f(-s)\). Translation symmetry is not explicitly enforced.
- **Probability:** \(p_\theta(s)=|\psi_\theta(s)|^2/\sum_{s'}|\psi_\theta(s')|^2\).
- **Loss:** the variational energy \(E_\theta=\langle\psi_\theta|H|\psi_\theta\rangle/
  \langle\psi_\theta|\psi_\theta\rangle\), not a supervised classification loss.

For \(h>0\), the finite-chain ground state can be chosen positive and real, so
this positive amplitude ansatz is physically appropriate. It cannot represent
arbitrary complex states such as general fractional quantum Hall wavefunctions.
The additive constant \(c\) changes normalization only: its physical gradient is zero.

The local-energy estimator uses all single-spin flips \(s^{(i)}\):

\[
E_{\rm loc}(s)=-J\sum_i s_i s_{i+1}
-h\sum_i\exp[f_\theta(s^{(i)})-f_\theta(s)].
\]

## Gradient and SPRING update

Let \(O_s=\nabla_\theta\log\psi_\theta(s)\). The **physical energy gradient** is

\[
g=2\langle(O-\langle O\rangle)(E_{\rm loc}-E_\theta)\rangle_{p_\theta}.
\]

The script saves and tests this full gradient. During SPRING optimization it
passes **\(f=g/2\)** to JaQMC to match the original SPRING energy-residual convention;
JaQMC adds no factor of two internally. The optional SGD control uses \(g\).

For an equally weighted sampled batch of size \(B\), define
\(A=(O-\bar O)/\sqrt B\), \(b=(E_{\rm loc}-\bar E)/\sqrt B\), so \(f=A^Tb\).
With damping \(\lambda\) and memory coefficient \(\mu\), SPRING computes

\[
d_k=\mu d_{k-1}+A^T(AA^T+\lambda I+P)^{-1}(b-\mu A d_{k-1}),
\quad P=\mathbf1\mathbf1^T/B.
\]

The mean-mode lift \(P\) does not change the ideal centered update. Equivalently,
\((A^TA+\lambda I)d_k=f+\lambda\mu d_{k-1}\). This parameter-space equation
provides an independent check of the upstream sample-space implementation.

JaQMC applies \(\theta_{k+1}=\theta_k-\alpha_k d_k\), with
\(\alpha_k=\min(\eta,C/(\|d_k\|+10^{-8}))\).
Here \(C\) bounds the **norm** of the parameter update, rather than its squared norm.
The stored memory is the raw direction \(d_k\), before learning-rate scaling or clipping.
Defaults: \(\eta=0.05,\lambda=10^{-3},\mu=0.9,C=0.1\), batch size 128, 250 steps, seed 7.

The package call is `jaqmc.optimizer.sr.jasr.robust_sr(...)`, configured with
`robust_gamma=None`, `march_beta=None`, `max_cond_num=None`, and no score clipping.
This selects damped SPRING without JaQMC's additional robust-SR/MARCH modifications.

## Sampling and validation

Each step draws independent configurations **with replacement** from the current
enumerated \(|\psi_\theta|^2\) distribution. This avoids Markov-chain mixing as a
confounding factor in a tiny gradient test. The batch covariance uses empirical
means with denominator \(B\), consistently with JaQMC; it has the usual finite-batch
bias. Exact expectations use normalized probability weights over all configurations.
Enumerating each state once with equal weights would give the wrong physical gradient.

Exact diagonalization and exact full-basis expectations are evaluation tools;
the training force is estimated from sampled configurations. Enumeration and dense
diagonalization scale exponentially and are limited here to \(3\le N\le10\).
This demonstrates correctness on a small system, rather than a speed advantage
over exact diagonalization. Larger VMC work would need a sampler such as Metropolis.

The tests check the Hamiltonian convention, local energy against \(H\psi/\psi\),
the covariance gradient against the differentiated Rayleigh quotient and finite
differences, normalization and spin-flip invariance, and the actual upstream SPRING
step against a dense parameter-space solve. The optimizer tests exercise nonzero
memory over changing batches, the \(\mu=0\) control, and active norm clipping.

## Saved output

The independent [ED benchmark suite](benchmarks/ed/README.md) contains complete
spectra for N=4,6,8,10 at J=h=1 and N=6 at h=0.5 and 2. It saves Hamiltonians,
bases, low eigenvectors, parity labels, observables and exact comparisons with
the retained N=6 SPRING checkpoint. Run `.venv/bin/python ed_benchmark.py` from
this directory to regenerate it. This NumPy/SciPy-only command performs no
training and reports both the full gap and the even-parity sector gap.

`results/spring_n6/` contains the executed default run:

- `metrics.json`: settings, dependency versions, exact reference, initial/final results.
- `gradient_check.json` and `gradient_check.npz`: all 73 physical gradient components
  from covariance, autodiff, and central finite differences at initialization.
- `training.csv`: exact energy/error/variance/fidelity after updates; sampled gradient,
  direction, update norm, and sampled energy from the batch immediately before each update.
- `trained_parameters.npz`: trained `W` and `c`.
- `diagnostics.png` and `diagnostics.pdf`: convergence and gradient checks.

Results from the executed seed-7 default run:

| Quantity | Value |
|---|---:|
| Exact ground energy | -7.727406610312546 |
| SPRING energy after 250 steps | -7.727390852046698 |
| Absolute energy error | 1.57583e-5 |
| Maximum gradient difference: covariance vs autodiff | 1.94e-15 |
| Maximum gradient difference: covariance vs finite differences | 2.87e-10 |

Read a saved gradient or trained weights with `.venv/bin/python`:

```python
import numpy as np
data = np.load("results/spring_n6/gradient_check.npz")
print(data["covariance_gradient"])
trained = dict(np.load("results/spring_n6/trained_parameters.npz"))
```

## Sources and provenance

- [SPRING paper: Goldshlager, Abrahamsen, and Lin (2024)](https://arxiv.org/abs/2401.10190).
- [Pinned JaQMC optimizer source](https://github.com/bytedance/jaqmc/blob/71f94158f2aca88ded904451f5314f71cdff5666/src/jaqmc/optimizer/sr/jasr.py).
- [VMCNet original SPRING implementation](https://github.com/jeffminlin/vmcnet/blob/bd23bb3f6171e8b741a6bfd47a45ebf550060cd9/vmcnet/updates/spring.py).
- [Your broader FQH project paper, arXiv:2412.00618](https://arxiv.org/abs/2412.00618).

The unmodified JaQMC source archive is stored in `vendor/jaqmc`, commit
`71f94158f2aca88ded904451f5314f71cdff5666` (Apache-2.0; see its LICENSE).
The isolated environment installs that source as a normal package. No upstream
optimizer code was modified. `PROVENANCE.json` records the archive checksum.
