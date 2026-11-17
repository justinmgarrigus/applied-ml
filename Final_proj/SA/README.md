# Simulated annealing (SA)

This gradient-free baseline searches the weights of the shared neural model.
It does not search individual spin configurations. Currently only the small
[Ising experiment](Ising/RUN_TRAINING.md) is implemented; FQHE is not supported.

A symmetric Gaussian proposes new weights. Lower-energy proposals are accepted;
higher-energy proposals are accepted with probability `exp(-delta_energy/T)`.
Temperature and proposal scale follow geometric schedules. There is no local
gradient optimizer, backpropagation, or SPRING direction memory. The scalar
normalization parameter `c` is fixed because it cannot change the loss.

Acceptance uses exact full-basis energy, without Monte Carlo noise. Existing
SPRING/SGD/MinSR runners use sampled training statistics, so this is explicitly
an **exact-objective SA baseline**, not a matched sampling experiment. Compare
wall time, objective/sampling costs and independent endpoint metrics; equal
iteration counts are not equal work. A controlled exact-objective comparison
would also need exact-gradient/statistic variants of the gradient methods.
Tune every method with a comparable budget and use multiple initialization and
search seeds. A short execution check establishes no optimizer ranking.

The [shared backend](../shared/ising_backend.py) owns the RBM and Hamiltonian;
[ising_annealing.py](../shared/ising_annealing.py) owns the SA search. Exact
diagonalization is performed only after training for evaluation, with no target
coefficients or eigenvalues used by acceptance or the cooling schedule.
