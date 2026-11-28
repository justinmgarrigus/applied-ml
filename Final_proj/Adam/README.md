# VMC with Adam

Same variational Monte Carlo problem as [VMC/SGD](../VMC/README.md); only the
parameter update differs. Adam keeps exponential moving averages of the gradient and its square, so the step is rescaled per parameter.
It is applied to the full physical energy gradient `g = 2 Cov(O, E_loc)`
(optax.adam with b1=0.9, b2=0.999, eps=1e-8), followed by the same applied-update norm cap (`--max-norm`, radius
not squared) as SGD. There is no score-metric inverse, SPRING memory or damping.

Only [Ising](Ising/PHYSICS.md) is implemented; FQHE is not supported here.
Physics, model, sampling and evaluation live in the
[shared backend](../shared/ising_backend.py), and ED is the shared
[benchmark database](../benchmarks/README.md), used for evaluation only.

The default learning rate is 0.01, not SGD's 0.05, because adaptive methods
move every parameter by roughly the learning rate per step. Learning rates are
not comparable across optimizers without tuning each with a comparable budget.
Execution checks and short runs establish no optimizer ranking.
