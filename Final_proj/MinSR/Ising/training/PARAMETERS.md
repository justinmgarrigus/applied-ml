# memory-free damped MinSR control: Ising parameters

This entry point calls the same [shared Ising backend](../../../shared/ising_backend.py)
as the other methods. It changes the optimizer, not the Hamiltonian, network,
sampling distribution or energy loss. All parameter arrays use float64;
the default backend is CPU. Every invocation starts a new model.

| Flag | Default | Meaning |
|---|---:|---|
| `--n` | 6 | Periodic sites, supported 3–10; complete basis has `2**n` rows. |
| `--hidden` | 12 | RBM hidden units, not layer count. |
| `--J` | 1.0 | Pauli `ZZ` interaction coefficient. |
| `--h` | 1.0 | Positive Pauli `X` field. |
| `--steps` | 250 | Sampled parameter updates. |
| `--batch-size` | 128 | IID Born draws per update, with replacement. |
| `--learning-rate` | 0.05 | Nominal raw-direction multiplier; the norm cap may reduce it. |
| `--damping` | 0.001 | Positive regularizer in the score-covariance solve. |
| `--mu` | 0.0 | Must be zero; nonzero values are rejected. |
| `--max-norm` | 0.1 | Parameter-update norm radius, not squared radius. |
| `--seed` | 7 | Initialization; sampling starts with seed+1. |
| `--log-every` | 5 | Exact evaluation spacing; initial, first and final are also recorded. |
| `--optimizer` | damped-minsr | Fixed by this method's entry point; another value is rejected. |
| `--output` | fresh run | New directory; existing destinations are refused. |

## Optimizer convention

The solver receives `g/2`, with positive damping and `mu=0`.
The wrapper rejects nonzero `--mu`; it does not silently turn on SPRING memory.
This is a damped SR/MinSR control, not the original paper's eigenvalue filter.
The physical loss is the Rayleigh energy. See [PHYSICS.md](../PHYSICS.md)
for the update equation and its factor conventions. Initial full-basis gradient
checks do not use the sampled covariance as their independent reference.

## Output and evaluation

- `metrics.json`: configuration, method description, source hashes, versions,
  exact ED ground energy and exact initial/final energy, variance and fidelity.
- `training.csv`: logged exact evaluation plus sampled energy before the
  update, physical-gradient norm, raw-direction norm and applied-update norm.
- `trained_parameters.npz`: `W` `(n, hidden)` and scalar `c`, float64; no
  optimizer/RNG state and no exact resume support.
- `gradient_check.json`/`.npz`: initial complete-basis covariance, automatic
  differentiation and finite-difference gradient checks.
- `diagnostics.png`/`.pdf`: energy-error/variance curves and gradient comparison.

ED is an evaluation reference, not a set of supervised labels. Sampling is
IID categorical from the fully enumerated Born distribution, not MCMC or
finite-temperature sampling. The default complete input table is `(64, 6)`;
there are 73 parameters and a batch of 128 rows can repeat configurations.
The small dense evaluation prevents scaling this runner beyond the supported
sizes. The shared [benchmark index](../../../benchmarks/README.md) documents ED.
