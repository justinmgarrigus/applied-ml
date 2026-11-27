# FQHE model with VMC

The [common sphere basis, Hamiltonian and network](../../SPRING/FQHE/PHYSICS.md)
uses binary fixed-particle-number, fixed-$L_z=0$ Slater configurations in the
lowest Landau level, with $H/V_1=\sum P_1$. The default six-layer, width-64
network has all-layer readouts and the anchored Cartesian amplitude

```math
u_\theta(n)=e^{\ell_0(n)}[1+\ell_\theta(n)-\ell_0(n)].
```

The frozen initial output $\ell_0$ gives the same initial amplitudes and
Jacobian as the exponential map. Its fixed scale lets a coefficient respond
at first order as its current amplitude passes through zero. The sampler
targets the Born distribution $p_\theta(n)\propto |u_\theta(n)|^2$. The
fit uses equal empirical weights for the sampled rows and the raw Hamiltonian
action $(Hu)_n$ from the connected signed pair hops. Burn-in, local pair moves,
and periodic uniform-sector refreshes use that same target.

The loss is the physical energy expectation. ED supplies post-training
reference values and supplies no coefficients to training. The explicit
Ne4 and Ne6 baseline configurations use exponential amplitudes and Born
sampling.

Let $D$ be the sampled amplitude Jacobian projected orthogonal to
the state and let $b$ be the corresponding Hamiltonian residual.
The physical energy gradient is $g=2\operatorname{Re}(D^\dagger b)$.
Ordinary Optax SGD applies $-\eta g$ and clips the applied parameter update
to radius 0.03. The factor of two includes both the bra and ket derivatives.
[Fit and optimizer parameters](training/PARAMETERS.md)

Finite-batch estimates require separate frozen-state energy and fidelity
checks. The complete Ne4 reference sector has 18 states. The historical Ne12
angular-head experiments document their own architecture and objective.
