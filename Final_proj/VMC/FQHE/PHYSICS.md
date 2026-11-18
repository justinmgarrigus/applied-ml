# FQHE model with VMC

The [common sphere basis, Hamiltonian and NN description](../../SPRING/FQHE/PHYSICS.md)
applies unchanged: binary fixed-N/M=0 Slater configurations, strict LLL,
H/V1=sum P1, no wall, fresh complex-log six-layer 64-wide NN with all-layer
readouts. The loss is the physical energy expectation only. ED is a
post-training benchmark and supplies no labels or coefficients to training.

This method uses ordinary Optax SGD on the FULL physical energy gradient
`g=2*Re Cov(O*,E_local)`. The applied update is `-eta*g`, clipped to parameter
radius 0.03. It uses no SR metric, damping or carried SPRING direction. Adam
and KFAC are not implemented by this entrypoint.

A small finite-batch estimate is not exact energy or fidelity. Saved Ne4
checks use independent complete 18-state ED after parameters are frozen.
The historical Ne12 angular-head/auxiliary-loss experiment remains a separate
source archive under SPRING and is not reproduced by this baseline.
