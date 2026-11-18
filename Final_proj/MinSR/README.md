# Minimum-step stochastic reconfiguration

MinSR fits the change produced by imaginary-time evolution using the
wavefunction changes the network can make in a small parameter step. The
method solves a linear system in sample space, whose size depends on the
batch rather than the parameter count. This project's MinSR control uses
positive diagonal damping and recomputes the direction from the current
batch. [Chen and Heyl, MinSR](https://arxiv.org/abs/2302.01941)

| Model | Physics | Run and tune | Saved checks |
|---|---|---|---|
| Ising | [Spin basis and real RBM](Ising/PHYSICS.md) | [Run training](Ising/RUN_TRAINING.md) · [Parameters](Ising/training/PARAMETERS.md) | [Ising results](Ising/results/README.md) |
| FQHE | [Occupation basis and complex NN](FQHE/PHYSICS.md) | [Run training](FQHE/RUN_TRAINING.md) · [Parameters](FQHE/training/PARAMETERS.md) | [FQHE results](FQHE/results/README.md) |

Both models use the variational energy and the same model-specific Hamiltonian,
network and sampling procedure as the other optimizers. Install the shared
environment before running either wrapper. [Shared setup](../shared/README.md)

## Imaginary-time evolution and the network tangent space

Let $x$ be a spin or occupation configuration and $\theta\in\mathbb R^P$
the network parameters. Write the normalized state as
$|\Psi\rangle=|\psi_\theta\rangle/\|\psi_\theta\|$. The energy and the
probability of a configuration are

```math
E=\langle\Psi|H|\Psi\rangle,\qquad
p_\theta(x)=|\Psi(x)|^2,\qquad
E_{\rm loc}(x)=\frac{(H\psi_\theta)(x)}{\psi_\theta(x)}.
```

Expand the evolution operator $e^{-\Delta\tau H}$ and renormalize its output.
The normalization contributes $+\Delta\tau E|\Psi\rangle$, giving

```math
\frac{e^{-\Delta\tau H}|\Psi\rangle}
{\|e^{-\Delta\tau H}|\Psi\rangle\|}
=|\Psi\rangle-\Delta\tau(H-E)|\Psi\rangle+O(\Delta\tau^2).
```

The relative amplitude change at configuration $x$ is therefore
$-\Delta\tau[E_{\rm loc}(x)-E]$ to first order. A parameter step
$-\Delta\tau\phi$ changes the log wavefunction through the derivatives
$O_j(x)=\partial_{\theta_j}\log\psi_\theta(x)$. Projecting its tangent
vectors with $Q=I-|\Psi\rangle\langle\Psi|$ removes the direction that only
changes overall normalization or phase:

```math
\langle x|Q\partial_{\theta_j}\Psi\rangle
=\Psi(x)\big[O_j(x)-\mathbb E_{p_\theta}O_j\big].
```

Squaring the difference between the two projected state changes and dividing
by $\Delta\tau^2$ gives an expectation under the Born distribution:

```math
\mathbb E_{p_\theta}\left[
\left|
\sum_j\big(O_j(x)-\mathbb E_{p_\theta}O_j\big)\phi_j
-\big(E_{\rm loc}(x)-E\big)
\right|^2\right].
```

The fit computes a local direction for an energy update. Its target is
constructed from the current network and $H$. ED supplies a reference for
evaluating the trained state. [Shared ED benchmarks](../benchmarks/README.md)

## From configurations to the linear system

For a batch $x_1,\dots,x_B$, let $M_{ij}=O_j(x_i)-B^{-1}\sum_k O_j(x_k)$
and $e_i=E_{\rm loc}(x_i)-B^{-1}\sum_k E_{\rm loc}(x_k)$. For real
wavefunctions the score matrix and residual vector are
$A=M/\sqrt{B}$ and $b=e/\sqrt{B}$.
The FQHE network gives complex log amplitudes with real parameters. It uses
both quadratures in one real regression:

```math
A=\frac{1}{\sqrt B}
\begin{pmatrix}\operatorname{Re}M\\\operatorname{Im}M\end{pmatrix},
\qquad
b=\frac{1}{\sqrt B}
\begin{pmatrix}\operatorname{Re}e\\\operatorname{Im}e\end{pmatrix}.
```

For real $\phi$, this stacking preserves the sum of squared complex
residual magnitudes. With $S=A^\top A$ and $f=A^\top b$,
$S=\operatorname{Re}(M^\dagger M)/B$ is the sampled metric and
$f=\operatorname{Re}(M^\dagger e)/B$ is the covariance force.
Differentiating the normalized energy gives
$\partial_{\theta_j}E=2\operatorname{Re}\langle Q\partial_{\theta_j}\Psi|(H-E)|\Psi\rangle$.
Thus the sampled physical gradient is $2f$. The SR equation uses $f$
because it fits the imaginary-time state change directly.

The regularized least-squares problem and its normal equation are

```math
\phi_\delta=\underset{d\in\mathbb R^P}{\operatorname{argmin}}
\left(\|Ad-b\|^2+\delta\|d\|^2\right),\qquad
(A^\top A+\delta I_P)\phi_\delta=A^\top b.
```

Here $\delta>0$ is the `damping` parameter. To solve in sample space,
use $(A^\top A+\delta I_P)A^\top=A^\top(AA^\top+\delta I_m)$,
where $m$ is the row count of $A$. Factoring the $m\times m$ system
and multiplying back by $A^\top$ gives the same direction:

```math
(AA^\top+\delta I_m)z=b,\qquad \phi_\delta=A^\top z.
```

If $A=U\Sigma V^\top$, singular value $\sigma_i>0$ contributes the
direction $v_i(u_i^\top b)\sigma_i/(\sigma_i^2+\delta)$. The
zero-damping limit is the Moore–Penrose solution $A^+b$. Among all
vectors minimizing the residual, it has the smallest Euclidean norm because
its component in $\ker A$ is zero. Positive damping smoothly reduces
amplification from small singular values. The paper's numerical experiments
use a spectrally regularized pseudoinverse. Our configurations use
diagonal damping and a Cholesky solve. [MinSR paper, Methods](https://arxiv.org/pdf/2302.01941)

## One training iteration

1. Draw a batch from the current state's Born distribution. Ising makes
   independent draws with replacement from its enumerated basis. FQHE
   advances its configuration walkers using Metropolis–Hastings moves that
   preserve particle number and $L_z$, with periodic uniform-sector refresh
   proposals. [Ising sampling](Ising/PHYSICS.md) · [FQHE sampling](../shared/fqh_spring/sampler.py)
2. Evaluate the connected Hamiltonian terms in $E_{\rm loc}$ and use
   automatic differentiation to compute $O_j(x_i)$. Center the scores
   and local energies over the batch and compute $f$.
3. Solve the damped SR equation. The Ising wrapper passes $g/2$ to
   JaQMC's SR solver with `spring_mu=0`. The FQHE trainer computes the
   covariance force directly and passes scores through `score_data` to
   `make_minsr`, which uses `spring_mu=None`. Both disable the contribution
   of the previous direction. JaQMC retains `prev_delta` in its state so
   the trainers can report the current raw direction's norm.
   [Ising backend](../shared/ising_backend.py) · [FQHE trainer](../shared/fqh_spring/train.py) · [FQHE optimizer](../shared/fqh_spring/optimizer.py)
4. Apply the capped update
   $\theta\leftarrow\theta-\alpha\phi_\delta$, where
   $\alpha=\min(\eta,r/(\|\phi_\delta\|+10^{-8}))$.
   `learning_rate` sets $\eta$ and `max_norm` sets the parameter-update
   radius $r$. The next update draws a batch at the new parameters.

The $z$ equation above derives the direction from $b$. The vendor receives
$f$ and evaluates the equivalent Woodbury form for the supplied damping
values:

```math
\phi_\delta
=\frac{1}{\delta}\left[f-A^\top(AA^\top+\delta I_m)^{-1}Af\right].
```

Its factorization lifts the constant sample mode in each centered quadrature
and uses a $10^{-8}$ floor in the reciprocal damping factor. The constant
modes are orthogonal to $Af$ because the score rows are centered.
[Pinned JaQMC implementation](../shared/vendor/jaqmc_spring/src/jaqmc/optimizer/sr/jasr.py)

## Matrix sizes and comparison

The $m$ rows comprise $B$ real samples for Ising or $2B$ quadrature
rows for FQHE. Centering gives rank at most $\min(P,B-1)$ or
$\min(P,2B-2)$, respectively. Repeated configurations can reduce the
rank further. Damping makes the Gram system invertible even in those cases.

| Dense calculation | Gram storage | Gram formation | Factorization |
|---|---:|---:|---:|
| Parameter space, $A^\top A$ | $O(P^2)$ | $O(mP^2)$ | $O(P^3)$ |
| Sample space, $AA^\top$ | $O(m^2)$ | $O(m^2P)$ | $O(m^3)$ |

The score matrix itself takes $O(mP)$ storage. These costs describe only
the linear algebra. Sampling, Hamiltonian connections and differentiation
also contribute to each iteration. The default six-site Ising model has
73 parameters and 128 sample rows. The Ne4 FQHE check has 22,274 parameters
and 64 realified rows, where the sample-space system is much smaller.

The stored CPU checks validate the implemented directions with independent
primal and dual solves. Both the sample count and the singular spectrum
affect conditioning and the information captured by one batch. For an
optimizer comparison, match the Hamiltonian, ansatz, initialization and
sampling protocol, and compare frozen energy and fidelity at matched update
counts and wall times. [Ising optimizer tests](../shared/tests/test_ising_optimizer.py) · [FQHE optimizer tests](../shared/tests/test_fqh_algorithms.py)
