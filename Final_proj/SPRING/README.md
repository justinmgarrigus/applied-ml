# SPRING

SPRING fits the change in a neural wavefunction to an imaginary-time step
using a batch of configurations. When the current batch provides little
information about a parameter direction, the optimizer can carry information
in that direction from earlier batches.
[Goldshlager, Abrahamsen and Lin, §3](https://arxiv.org/html/2401.10190#S3)

## From the energy to a batch equation

For a configuration $`x`$, the network returns the log of its wavefunction
coefficient $`\psi_\theta(x)`$. Write
$`p_\theta(x)=|\psi_\theta(x)|^2/\|\psi_\theta\|^2`$ and
$`E_{\mathrm{loc}}(x)=(H\psi_\theta)(x)/\psi_\theta(x)`$. The energy is
$`E=\mathbb E_{p_\theta}[E_{\mathrm{loc}}]`$. For real parameters, define the
log derivatives $`O_j(x)=\partial_{\theta_j}\log\psi_\theta(x)`$. These are
derivatives of the network output with respect to its weights and biases.
[Energy and gradient derivation](../VMC/README.md)

Let $`\Psi_\theta=\psi_\theta/\|\psi_\theta\|`$. To first order in imaginary
time, the normalized target moves by $`-\Delta\tau(H-E)\Psi_\theta`$.
The projective tangent is the part of the network derivative orthogonal to
$`\Psi_\theta`$. Applying $`I-|\Psi_\theta\rangle\langle\Psi_\theta|`$ to
$`\partial_{\theta_j}\Psi_\theta`$ gives components
$`(O_j(x)-\mathbb E[O_j])\Psi_\theta(x)`$, removing the global normalization
and phase directions. With our convention $`\Delta\theta=-\Delta\tau\,\phi`$,
fitting the tangent to the target gives

```math
\min_{\phi\in\mathbb R^P}
\mathbb E_{p_\theta}\!\left[
 \left|\sum_{j=1}^P (O_j-\mathbb E[O_j])\phi_j
       -(E_{\mathrm{loc}}-E)\right|^2
\right].
```

For $`B`$ configurations, replace the expectations by sample means. Stack
the real and imaginary parts to obtain a real least-squares problem for the
$`P`$ real parameters.

```math
D_{ij}=\frac{O_j(x_i)-\bar O_j}{\sqrt B},\qquad
e_i=\frac{E_{\mathrm{loc}}(x_i)-\bar E_{\mathrm{loc}}}{\sqrt B},\qquad
A=\begin{bmatrix}\Re D\\\Im D\end{bmatrix},\qquad
b=\begin{bmatrix}\Re e\\\Im e\end{bmatrix}.
```

For real scores and energies, take $`A=D`$, $`b=e`$. The objective is
$`\|A\phi-b\|^2`$, the metric is $`S=A^\mathsf T A=\Re(D^\dagger D)`$,
and the force is $`f=A^\mathsf T b=\Re(D^\dagger e)`$. The sampled estimate
of the physical energy gradient is $`2f`$. We use a positive energy residual
in $`b`$ and subtract the resulting step. The paper uses a negative energy
residual and adds the step. [SPRING, §2–3](https://arxiv.org/html/2401.10190#S2)

| System | Configuration input | Score matrix in the solve | Rank bound |
|---|---|---|---|
| Ising | $`N`$ Pauli spins in $`\{-1,+1\}`$ | $`B\times P`$ | $`\min(P,B-1)`$ |
| FQHE | Binary orbital occupations at fixed $`N_e,L_z`$ | $`2B\times P`$ | $`\min(P,2(B-1))`$ |

For the default six-site Ising setting, $`B=128`$ and $`P=73`$. The full
spin basis has 64 configurations, and draws may repeat. For the default
four-electron FQHE check, $`B=32`$ and $`P=22{,}274`$, so $`A`$ has 64
rows. The Hamiltonian supplies $`b`$ through connected configurations.
Each score row comes from a sampled configuration. Repeated samples and
symmetries can reduce the rank below the bounds in the table.

## Deriving the projected increment

At update $`k`$, set $`m_k=\mu\phi_{k-1}`$ with $`\phi_0=0`$. Start at that
direction and use the current batch to correct its residual. For a consistent
batch equation, the closest solution to $`m_k`$ is

```math
\begin{aligned}
\phi_k
 &=\arg\min_{\phi:\,A_k\phi=b_k}\|\phi-m_k\|^2\\[0pt]
 &=m_k+A_k^+(b_k-A_km_k)
 =A_k^+b_k+(I-A_k^+A_k)m_k.
\end{aligned}
```

To see this, write $`\phi=m_k+s`$. The correction $`s`$ must solve
$`A_ks=b_k-A_km_k`$. Its minimum-norm solution is in the row space of
$`A_k`$. The orthogonal component of $`m_k`$ in $`\ker A_k`$ survives.
If the equation is inconsistent, the same pseudoinverse formula gives the
closest least-squares minimizer. For example, if $`A=[1\;\;0]`$, $`b=1`$
and $`m=(2,3)`$, SPRING returns $`(1,3)`$. Minimum-norm SR returns $`(1,0)`$.

For positive damping $`\delta`$, penalize large corrections to $`m_k`$.
Setting the derivative of this quadratic to zero gives

```math
\begin{aligned}
\phi_k
 &=\arg\min_\phi\left(\|A_k\phi-b_k\|^2+
                              \delta\|\phi-m_k\|^2\right),\\[0pt]
(A_k^\mathsf T A_k+\delta I)\phi_k
 &=f_k+\delta m_k .
\end{aligned}
```

Substitute $`\phi_k=m_k+s_k`$ into the normal equation, then use
$`(A^\mathsf T A+\delta I)A^\mathsf T=A^\mathsf T(AA^\mathsf T+\delta I)`$.
This gives the sample-space form

```math
\phi_k=m_k+A_k^\mathsf T
 (A_kA_k^\mathsf T+\delta I)^{-1}(b_k-A_km_k).
```

The inverse denotes a linear solve. With $`q=B`$ real rows for Ising or
$`q=2B`$ for complex FQHE scores, forming the dense Gram matrix costs
$`O(q^2P)`$, and factoring it costs $`O(q^3)`$. These costs are separate
from evaluating the log derivatives and local energies.

For $`A_k=U\Sigma V^\mathsf T`$, the component along a right singular
vector $`v_i`$ with singular value $`\sigma_i>0`$ is

```math
v_i^\mathsf T\phi_k=
\frac{\sigma_i}{\sigma_i^2+\delta}\,u_i^\mathsf T b_k
+\frac{\delta}{\sigma_i^2+\delta}\,v_i^\mathsf T m_k .
```

When $`\sigma_i^2\gg\delta`$, the current batch sets this component.
When $`\sigma_i^2\ll\delta`$, the old direction contributes more.
In a null direction, $`\phi_k`$ keeps exactly the component of $`m_k`$.
Setting $`\mu=0`$ gives the damped MinSR control. In training, $`A_k`$
and $`b_k`$ change with the state and sample. Classical Kaczmarz convergence
rates require a fixed linear system and their own sampling assumptions.
[Damped MinSR](../MinSR/README.md)

## Applying the update in the code

```text
initialize parameters, sampler and raw direction phi = 0
for each optimizer update:
    draw B configurations at the current parameters
    evaluate connected local energies and log derivatives
    form the centered score metric and the half-gradient force
    m = mu * phi
    solve the damped batch equation with prior direction m
    store the resulting raw direction phi
    alpha = min(learning_rate, max_norm / (norm(phi) + 1e-8))
    theta = theta - alpha * phi
    record diagnostics and save according to the run settings
```

- **Ising.** The small runner enumerates the spin basis to sample the Born
  distribution with replacement. It computes the full covariance gradient
  in `energy_covariance_gradient` and passes half to `make_spring`.
  Each local energy uses the diagonal $`ZZ`$ term and single-spin flips
  from $`X`$. [Backend](../shared/ising_backend.py)
- **FQHE.** The runner advances Metropolis walkers in the fixed particle
  number and $`L_z`$ sector. Local energy uses the sparse signed Hamiltonian
  connections. `train_step` forms $`f`$, and `score_data` packages the complex
  scores for the shared optimizer.
  [Training loop](../shared/fqh_spring/train.py) · [Optimizer adapter](../shared/fqh_spring/optimizer.py)
- **JaQMC solve.** The pinned `robust_sr` receives $`f`$ directly.
  For the damping values in the supplied configs, the Woodbury form is
  $`\phi=m+f/\delta-A^\mathsf T G^{-1}A(m+f/\delta)`$, where
  $`G=AA^\mathsf T+\delta I+P_0`$. The vendor projects out the constant
  row modes before adding the projector $`P_0`$ onto them. Centering gives
  $`A^\mathsf T P_0=0`$, so this lift has zero contribution to the parameter
  direction in exact arithmetic. Real scores have one such mode. Complex
  scores have one per quadrature.
  [Pinned solver source](../shared/vendor/jaqmc_spring/src/jaqmc/optimizer/sr/jasr.py)

`damping` is $`\delta`$ in the linear solve. `mu` sets the fraction of the
previous raw direction supplied to the next solve. `max_norm` is the
Euclidean step radius $`r=\sqrt C`$ in the paper's notation. The cap affects
the applied step through $`\alpha`$. JaQMC saves the raw direction in
`prev_delta`. The FQHE runner records the raw direction norm, effective step
scale and cap activity alongside the energy estimate.

## Run and inspect

| System | Physics and model | Run and tune | Results |
|---|---|---|---|
| Ising | [Physics](Ising/PHYSICS.md) | [Run](Ising/RUN_TRAINING.md) · [Parameters](Ising/training/PARAMETERS.md) | [Results](Ising/results/README.md) |
| FQHE | [Physics](FQHE/PHYSICS.md) | [Run](FQHE/RUN_TRAINING.md) · [Parameters](FQHE/training/PARAMETERS.md) | [Results](FQHE/results/README.md) |

The small-system ED references give independent energy, variance and fidelity
checks of trained states. The portable FQHE runs use the energy objective
above. The archived angular-aware runs document their own loss, metric and
numerical solver. [Shared setup](../shared/README.md) ·
[ED benchmarks](../benchmarks/README.md) ·
[Angular-aware source](FQHE/training/research_snapshot/README.md)
