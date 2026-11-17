# VMC with ordinary SGD

Variational Monte Carlo (VMC) is the framework: a neural wavefunction defines
Born probabilities, sampling estimates its energy, and an optimizer updates
its parameters. SPRING and MinSR also operate within VMC. This folder uses
**ordinary SGD** as the comparison optimizer; it is not a different Hamiltonian
or a supervised regression problem.

Choose [Ising](Ising/PHYSICS.md) for the small positive RBM or
[FQHE](FQHE/PHYSICS.md) for the complex six-layer all-readout NN. Each folder has
PHYSICS.md, RUN_TRAINING.md, training/ with PARAMETERS.md, and results/.
[Shared code](../shared/README.md) owns both physical models and sampling;
[shared ED](../benchmarks/README.md) supplies independent small-system checks.

For real parameters and possibly complex coefficients,

```math
E=\mathbb E_{|\psi|^2}[E_{\mathrm{loc}}],\qquad
g_j=2\operatorname{Re}\langle
(O_j-\langle O_j\rangle)^*(E_{\mathrm{loc}}-\langle E_{\mathrm{loc}}\rangle)
\rangle,\quad O_j=\partial_{\theta_j}\log\psi.
```

SGD proposes `u=-eta*g`, caps its norm at radius r, and applies `theta+=u`.
It has no score-metric inverse, SPRING direction memory or optimizer damping.
The physical gradient is twice the force used by the SR-family adapters.
Do not compare numerical learning rates without accounting for that convention.

Input arrays and wavefunction coefficients have the same meanings as in the
SPRING folder. Energies, variance, parameter artifacts and run metadata are
saved in results/. ED eigenvectors never supply the gradient. Ising uses IID
draws from the enumerated Born distribution; FQHE uses fixed-sector Born MCMC.
The shared code keeps those differences explicit.

Current short runs and exact-gradient checks validate execution, signs and
scaling. They do not rank convergence against SPRING. A fair experiment matches
physics, initialization, architecture, sampling and evaluation and records both
updates and wall time. Adam and K-FAC are not implemented comparison arms.
