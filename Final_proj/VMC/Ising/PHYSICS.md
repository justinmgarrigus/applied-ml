# Periodic quantum Ising model

This example minimizes the energy of a one-dimensional transverse-field Ising
ring. It uses Pauli matrices, whose spin eigenvalues are **−1 and +1**:

\[
H=-J\sum_{i=0}^{N-1}Z_iZ_{(i+1)\bmod N}-h\sum_{i=0}^{N-1}X_i.
\]

The default is `N=6`, `J=h=1`. The code supports `3 <= N <= 10`,
with `h>0`; complete enumeration makes it a small correctness experiment.
It does not demonstrate a sampling or memory advantage over ED at these sizes.

## Input, output, and trainable state

One input is an array `s` of shape `(N,)`, with entries `-1` or `+1`.
Site 0 is the least-significant bit of the basis index; bit 0 means spin −1.
For example, the six-site complete input table has shape `(64, 6)`.

The network returns one **log wavefunction amplitude**, not an energy:

\[
f_\theta(\mathbf s)=\log\psi_\theta(\mathbf s)
=c+\sum_{a=1}^{N_h}\log\cosh\!\left(\sum_i s_iW_{ia}\right).
\]

The parameters are `W[N, hidden]` and scalar `c`, so their count is
`P=N*hidden+1`. This positive real RBM sums over a single set of
hidden units. It is not the multilayer complex network used for FQHE.
Zero biases enforce global spin-flip symmetry, `psi(-s)=psi(s)`.
The scalar `c` only changes normalization; its physical gradient vanishes.
The positive ansatz is appropriate to this finite `h>0` Hamiltonian, whose
ground state can be chosen positive. It cannot represent arbitrary sign or
complex structures. Translation symmetry is not imposed.

## Energy loss and sampling

The scalar loss is the normalized Rayleigh quotient:

\[
E(\theta)=\frac{\langle\psi_\theta|H|\psi_\theta\rangle}
{\langle\psi_\theta|\psi_\theta\rangle}
=\mathbb E_{p_\theta}[E_{\rm loc}(\mathbf s)],\qquad
p_\theta(\mathbf s)\propto|\psi_\theta(\mathbf s)|^2.
\]

All off-diagonal terms are evaluated exactly for each sampled input. If
`s^(i)` flips only site `i`, then

\[
E_{\rm loc}(\mathbf s)=-J\sum_i s_is_{i+1}
-h\sum_i\exp[f_\theta(\mathbf s^{(i)})-f_\theta(\mathbf s)].
\]

Each update draws `batch_size` independent categorical samples **with
replacement** from the fully enumerated Born distribution. There is no Markov
chain or temperature. A batch may contain duplicate configurations, even when
its size exceeds `2**N`.

With `O_j = d f_theta / d theta_j`, the physical gradient is

\[
g_j=2\,\operatorname{Cov}_{p_\theta}(O_j,E_{\rm loc}).
\]

The code estimates covariance using empirical means with divisor `B`. This
finite-batch estimate has the usual covariance-estimator bias; exact gradient
checks use the complete Born distribution.

## VMC with SGD update

VMC names the energy-minimization framework; SPRING and MinSR also operate
within VMC. This folder selects plain **SGD on the full physical gradient**:

```math
g_k=2\operatorname{Cov}_{\rm batch}(O,E_{\rm loc}),\qquad
\theta_{k+1}=\theta_k-\alpha_k g_k,\qquad
\alpha_k=\min\left(\eta,\frac{r}{\|g_k\|}\right).
```

The implementation chains `optax.sgd` with `optax.clip_by_global_norm`:
first form the full-gradient update, then cap its norm at `r = max_norm`.
A zero gradient gives a zero update. There is no momentum or curvature solve.
`--damping` and `--mu` are accepted shared-parser fields but are **inactive for
SGD**. `--max-norm` is active and bounds the applied parameter update.
`raw_direction_norm` in its CSV is the norm of the full physical gradient. Equal numerical learning rates across SGD and the
SR-family methods do not imply equal parameter steps.

## Evaluation and shared ED data

The small runner diagonalizes the Hamiltonian and computes complete-basis
energy, residual variance `||(H-E)psi||^2`, and ground-state fidelity for
evaluation. Neither ED eigenvectors nor target coefficients enter the sampled
gradient or optimizer update. Gradient checks compare the covariance formula
with the differentiated Rayleigh quotient and finite differences.

The [shared benchmark index](../../benchmarks/README.md) links the independent
NumPy/SciPy ED database and its conventions. No benchmark directory is nested
inside this physics folder. [Stored results](results/README.md) distinguish the
execution checks from any future convergence study.

The implementation source is [shared/ising_backend.py](../../shared/ising_backend.py).
JaQMC's [license](../../shared/vendor/jaqmc_spring/LICENSE),
[notice](../../shared/vendor/jaqmc_spring/NOTICE), and
[provenance](../../shared/vendor/jaqmc_spring/PROVENANCE.json) are retained.
