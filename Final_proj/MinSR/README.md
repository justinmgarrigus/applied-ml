# Memory-free damped MinSR control

This folder implements a **finite-damping, memory-free sample-space SR**
control. It shares the neural models, energy objective and sampling with
SPRING. It does not claim an exact reproduction of the MinSR paper's spectral
cutoff or another library's default regularization.

Choose [Ising](Ising/PHYSICS.md) or [FQHE](FQHE/PHYSICS.md). Each has PHYSICS.md,
RUN_TRAINING.md, training/ with PARAMETERS.md, and results/.
[Shared code](../shared/README.md) owns physical operators and wavefunctions;
[shared ED](../benchmarks/README.md) is for independent evaluation only.

Let A be centered log-wavefunction derivatives divided by sqrt(batch), and b
the centered local-energy residual with the same normalization. For complex
coefficients and real parameters, stack real and imaginary parts. The direction
solves the regularized least-squares problem

```math
\phi=\underset{d}{\operatorname{argmin}}\left[
\|Ad-b\|^2+\delta\|d\|^2\right]
=A^T(AA^T+\delta I)^{-1}b,
\qquad \theta\leftarrow\theta-\min(\eta,r/(\|\phi\|+10^{-8}))\phi.
```

Equivalently `(A.T@A+delta*I)phi=f`, where `f=A.T@b` is half the physical
energy gradient. As damping tends to zero, the minimum-norm solution is `A+ b`
under the usual pseudoinverse limit; finite damping is what these runs use.
Solving in sample space can reduce linear-system size when rows are much fewer
than parameters, but dense solves and score storage still cost memory/time.

There is no previous-direction memory: Ising explicitly fixes mu=0; FQHE uses
the vendored no-memory mode. SPRING instead retains a component from earlier
directions. Tests compare the control with independently constructed primal
and dual solves, including complex-score realification and active norm caps.

The loss remains energy, not squared distance to an ED wavefunction. Short
execution checks do not establish a convergence advantage. Match initialization,
Hamiltonian, architecture, sampling, damping/caps and evaluation for a proper
optimizer comparison. Reference: [Chen and Heyl](https://arxiv.org/abs/2302.01941).
