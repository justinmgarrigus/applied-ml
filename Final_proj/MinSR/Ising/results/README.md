# memory-free damped MinSR control: Ising results

`check_cpu_n4/` contains a fresh three-update CPU execution check with
`N=4`, hidden=3, batch=16 and seed=7. Its raw outputs, plots and a
[validation receipt](validation.json) document the check. This is not a convergence result,
and it establishes no accuracy or speed ranking against SPRING or other methods.

The check uses the same physical energy loss and initial model as the SPRING
execution check. This method's optimizer is explicitly `damped-minsr`.
MinSR memory is exactly zero; it is a finite-damping SR control, not an eigenvalue-truncated implementation.

New runs use unique `runs/` directories. No historical SPRING training result
has been copied or relabelled as a MinSR outcome. The independent ED
database remains in the [shared benchmark area](../../../benchmarks/README.md).
