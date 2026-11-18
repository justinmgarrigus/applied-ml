# VMC with SGD: Ising results

`check_cpu_n4/` contains a fresh three-update CPU execution check with
`N=4`, hidden=3, batch=16 and seed=7. Its raw outputs, plots and a
[validation receipt](validation.json) document the check. This is not a convergence result,
and it establishes no accuracy or speed ranking against SPRING or other methods.

The check uses the same physical energy loss and initial model as the SPRING
execution check. This method's optimizer is explicitly `sgd`.
VMC here means SGD on the full physical energy gradient, without momentum, with an applied-update norm cap.

New runs use unique `runs/` directories. No historical SPRING training result
has been copied or relabelled as a VMC outcome. The independent ED
database remains in the [shared benchmark area](../../../benchmarks/README.md).
