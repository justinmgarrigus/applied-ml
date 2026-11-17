# SPRING

The network maps a spin/occupation array to a wavefunction coefficient.
The Hamiltonian provides local energies; minimizing their Born-weighted mean
approximates the ground state without target coefficients from ED.

Choose [Ising](Ising/PHYSICS.md) for a positive real wavefunction or
[FQHE](FQHE/PHYSICS.md) for complex coefficients of antisymmetric occupation
states. Each problem has PHYSICS.md, RUN_TRAINING.md, training/ and results/.
Training contains entry points, configurations and PARAMETERS.md; results
separates historical evidence from execution checks. Code is shared in
[shared](../shared/README.md); [ED](../benchmarks/README.md) is common to all methods.

## Update and conventions

Let O(n)=d log(psi(n))/d theta. A contains centered score rows normalized by
sqrt(batch), and b contains centered local energies with the same scaling.
For complex states and real parameters, the metric is Re(O†O) and the force
is f=Re(O†b). Stacking real/imaginary rows expresses a real least-squares problem.
The physical energy gradient is g=2f.

Ignoring damping, the Kaczmarz-style SPRING correction is

```math
\phi_k=\mu\phi_{k-1}+A_k^+(b_k-\mu A_k\phi_{k-1}).
```

It preserves the previous direction's component undetermined by current rows.
The damped implementation uses a sample-space solve, with the corresponding
parameter-space equation

```math
(A_k^T A_k+\delta I)\phi_k=f_k+\delta\mu\phi_{k-1},
\qquad \theta_{k+1}=\theta_k-\alpha_k\phi_k.
```

Here alpha=min(eta, radius/(norm(phi)+1e-8)); radius is sqrt(C) in the paper's
squared-cap notation. Centered-row mean-mode stabilization in the vendor does
not add physical information.

The code reuses pinned JaQMC robust_sr with robust/clipping extensions disabled.
Damping controls conditioning; memory mu distinguishes SPRING from the
memory-free control. The samples and network change, so fixed-matrix Kaczmarz
analysis is not a nonlinear training convergence guarantee. Batch size limits
directions observed in one update, not all directions learned over time.

The portable energy-only baseline differs from the frozen angular-aware FQHE
research workflow. Read the FQHE result index before comparing energies.
Execution checks alone establish no optimizer ranking.

References: [SPRING](https://arxiv.org/abs/2401.10190),
[pinned JaQMC](https://github.com/bytedance/jaqmc/tree/71f94158f2aca88ded904451f5314f71cdff5666).
