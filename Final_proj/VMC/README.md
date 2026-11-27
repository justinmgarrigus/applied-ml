# VMC with ordinary SGD

Variational Monte Carlo (VMC) minimizes the energy of a parameterized
wavefunction. The Hamiltonian determines the objective and the wavefunction
determines which configurations to sample. This folder implements ordinary
stochastic gradient descent (SGD). SPRING and MinSR use the same VMC energy
framework and precondition the gradient using wavefunction derivatives.
See the [shared implementation and environment](../shared/README.md).

| Model | Input and wavefunction output | Instructions |
|---|---|---|
| Ising | Spins in `{-1,+1}`. A positive real RBM returns a log amplitude. | [Physics](Ising/PHYSICS.md) · [Run](Ising/RUN_TRAINING.md) · [Parameters](Ising/training/PARAMETERS.md) · [Results](Ising/results/README.md) |
| FQHE | A binary LLL occupation vector at fixed particle number and `Lz`. A six-layer MLP uses all-layer readouts for an anchored complex Cartesian amplitude. | [Physics](FQHE/PHYSICS.md) · [Run](FQHE/RUN_TRAINING.md) · [Parameters](FQHE/training/PARAMETERS.md) · [Results](FQHE/results/README.md) |

## Cartesian FQHE default

The FQHE entry point defaults to the anchored Cartesian amplitude
$u_\theta=e^{\ell_0}(1+\ell_\theta-\ell_0)$ and target
$q(n)\propto |u_\theta(n)|^2+a$ with $a=1/3$. It computes the physical
energy gradient from the weighted raw amplitude derivatives and Hamiltonian
actions. SGD applies the full gradient $g=2f$. The exponential/Born
expressions below are an alternative form of that energy derivative at
nonzero amplitudes. [Cartesian parameters](FQHE/training/PARAMETERS.md)

## From the Rayleigh quotient to local energy

Let $x$ denote one whole basis configuration. With real trainable parameters
$\theta$, the model assigns a coefficient $\psi_\theta(x)$ to each $|x\rangle$.
The model can work with unnormalized coefficients because the energy is a
Rayleigh quotient.

```math
Z_\theta=\langle\psi_\theta|\psi_\theta\rangle,\qquad
E(\theta)=\frac{\langle\psi_\theta|H|\psi_\theta\rangle}{Z_\theta}.
```

For orthonormal energy eigenstates $|u_m\rangle$, write
$|\psi_\theta\rangle=\sum_m c_m|u_m\rangle$. Then
$E=\sum_m |c_m|^2E_m/\sum_m |c_m|^2$. This weighted average is bounded below
by the lowest eigenvalue in the chosen Hilbert space. The optimizer seeks a
low-energy state within the available neural ansatz.

In the configuration basis, rearrange each numerator term as
$\psi_\theta(x)^*(H\psi_\theta)(x)=|\psi_\theta(x)|^2E_{\rm loc}(x)$.
This gives the Born average

```math
E(\theta)=\sum_x p_\theta(x)E_{\rm loc}(x),\qquad
p_\theta(x)=\frac{|\psi_\theta(x)|^2}{Z_\theta},\qquad
E_{\rm loc}(x)=\sum_{x'}H_{xx'}
   \frac{\psi_\theta(x')}{\psi_\theta(x)}.
```

A common nonzero factor in all coefficients cancels from the Born weights and
amplitude ratios. The Ising row of $H$ connects a spin vector to itself and
its one-spin flips. The FQHE row connects occupations through signed fermionic
pair matrix elements. The model evaluates coefficients for these connected
configurations as part of the local-energy computation.
[Ising local energy](../shared/ising_backend.py) · [FQHE local energy](../shared/fqh_spring/local_energy.py)

## Derive the full energy gradient

Define $O_j(x)=\partial_{\theta_j}\log\psi_\theta(x)$, so that
$\partial_j\psi_\theta(x)=O_j(x)\psi_\theta(x)$. Hermiticity makes the two
numerator derivatives conjugates. The quotient rule includes the change in
normalization through the second term.

```math
\begin{aligned}
g_j=\partial_j E
  &=\frac{2\operatorname{Re}\langle\partial_j\psi|H|\psi\rangle}{Z_\theta}
    -E\frac{2\operatorname{Re}\langle\partial_j\psi|\psi\rangle}{Z_\theta}\\
  &=2\operatorname{Re}\,\mathbb E_{p_\theta}
    \big[O_j(x)^*(E_{\rm loc}(x)-E)\big].
\end{aligned}
```

Because the energy residual has zero Born mean, replacing $O_j$ by
$O_j-\mathbb E_{p_\theta}[O_j]$ leaves the gradient unchanged. A constant log
amplitude or phase therefore contributes zero gradient. For an exponential ansatz
$\log\psi=a+i\varphi$, both derivatives enter the real force. The real part
of $O_j^*(E_{\rm loc}-E)$ is
$(\partial_j a)\operatorname{Re}(E_{\rm loc}-E)+
(\partial_j\varphi)\operatorname{Im}(E_{\rm loc}-E)$.

For a batch $x_1,\ldots,x_B$, let $e_i=E_{\rm loc}(x_i)$ and
$\bar e=B^{-1}\sum_i e_i$. With $P$ trainable scalars, the scores have $B$ rows
and $P$ columns, indexed by $i$ and $j$ respectively. The implementation uses
the empirical estimator

```math
\widehat g_j=\frac{2}{B}\operatorname{Re}\sum_{i=1}^B
  O_j(x_i)^*(e_i-\bar e),\qquad \widehat E=\operatorname{Re}\bar e.
```

The sum of batch residuals is zero, so the trainer can use raw score rows and
center only the local energies in this product. The empirical mean retains
its imaginary part for that centering. The logged physical energy uses its
real part. The factor $2$ comes from differentiating the bra and the ket.
Here SGD consumes $\widehat g$. The SR-family adapters use the force
$f=\widehat g/2$. The divisor $B$ is the empirical covariance convention.
For IID draws its expectation at fixed parameters is $(1-1/B)g$, since the
batch mean is estimated from the same draws. Persistent FQHE walkers also
depend on mixing and the changing parameters.

## Apply one SGD update

Let $\eta$ be the nominal learning rate and $r$ the applied update norm radius
(`max_norm`). For a nonzero estimated gradient, the update is

```math
u=-\eta\widehat g,\qquad
\Delta\theta=u\min\left(1,\frac{r}{\|u\|_2}\right),\qquad
\theta_{k+1}=\theta_k+\Delta\theta.
```

A zero gradient produces a zero update. The norm is taken across all real
parameter arrays together. When clipping is active, the step retains the SGD
direction and its length is $r$. This update uses the Euclidean parameter
gradient and the current batch. The shared implementations chain
`optax.sgd` with `optax.clip_by_global_norm`.
[Ising `make_sgd`](../shared/ising_backend.py) · [FQHE `make_optimizer`](../shared/fqh_spring/optimizer.py)

```text
Initialize wavefunction parameters and sampling state
For each requested optimizer update:
    Draw or advance B configurations using the current sampling target
    Evaluate connected Hamiltonian actions and coefficient derivatives
    Form the physical energy half-gradient force f with the target weights
    g = 2 * f
    update = -learning_rate * g
    If norm(update) exceeds max_norm, scale update to that radius
    Add update to parameters
    Save diagnostics and parameter artifacts at the configured intervals
```

One step means one optimizer update. The sampling stage produces fresh draws
for Ising or advances the walkers for FQHE.

- **Ising.** The backend enumerates the $2^N$ spin configurations and uses
  independent categorical draws with replacement from logits
  $2\log\psi_\theta(x)$. `energy_covariance_gradient` forms the real full
  gradient from the sampled spins and single-flip local energies.
  [Ising trainer](../shared/ising_backend.py)
- **FQHE.** The trainer starts walkers from uniform configurations in the
  chosen sector and targets $q\propto |u|^2+a$ during burn-in, local pair
  moves, and periodic global refreshes. Local moves preserve particle number
  and $L_z$. Each update includes repeated walker rows and uses inverse
  target weights in the raw amplitude fit. `train_step` multiplies the
  half-gradient force by two before passing it to ordinary SGD.
  [Sampler](../shared/fqh_spring/sampler.py) · [FQHE trainer](../shared/fqh_spring/train.py)

## Check the gradient and evaluate a saved state

The small Ising runner records complete-basis energy and variance and compares
its state with the ED ground state at logging intervals. Its initial gradient
checks use direct differentiation and finite differences of the exact Rayleigh
quotient. The FQHE trainer records sampled energy and variance. Its stored Ne4
check uses a separate exact 18-state sector evaluation of frozen parameters.
The FQHE unit tests differentiate the exact Rayleigh quotient on a small sector
and verify the factor of two and the applied update cap. ED eigenvectors serve
as evaluation references. [Ising physics tests](../shared/tests/test_ising_physics.py) ·
[FQHE algorithm tests](../shared/tests/test_fqh_algorithms.py) · [Shared ED benchmarks](../benchmarks/README.md)

The stored short runs establish execution and numerical consistency. A
convergence comparison needs matched Hamiltonians, ansatzes, initialization,
sampling and evaluation, with update counts and wall time reported together.
