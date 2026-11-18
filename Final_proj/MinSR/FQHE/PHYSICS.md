# FQHE model with MinSR

The [common sphere basis, Hamiltonian and NN description](../../SPRING/FQHE/PHYSICS.md)
applies unchanged: binary fixed-N/M=0 Slater configurations, strict LLL,
H/V1=sum P1, no wall, fresh complex-log six-layer 64-wide NN with all-layer
readouts. The loss is the physical energy expectation only. ED is a
post-training benchmark and supplies no labels or coefficients to training.

This method is finite-damping memory-free sample-space stochastic
reconfiguration. Center complex log scores and realify them as
`A=[Re O; Im O]/sqrt(B)` with realified centered local residual `epsilon`.
For real parameters, `f=A.T@epsilon=gradient(E)/2` and
`phi=A.T@solve(A@A.T+lambda*I,epsilon)`. This equals the damped primal solve.
It disables carried memory (`spring_mu=None`, config mu 0), uses lambda 0.01,
and clips the applied update to radius 0.03. There is no hard eigenvalue
cutoff or literal Moore-Penrose inverse; this convention is explicit.

A small finite-batch estimate is not exact energy or fidelity. Saved Ne4
checks use independent complete 18-state ED after parameters are frozen.
The historical Ne12 angular-head/auxiliary-loss experiment remains a separate
source archive under SPRING and is not reproduced by this baseline.
