# Ising training parameters

The entry point is `run.py`; numerical implementation is shared at
`Final_proj/shared/ising_backend.py`. All arrays use float64 and the default
backend is CPU. Every invocation initializes a new model; there is no resume
option in this small Ising runner.

| Flag | Default | Meaning |
|---|---:|---|
| `--n` | 6 | Sites in the periodic ring; supported range 3–10. Full basis size `2**n`. |
| `--hidden` | 12 | RBM hidden units, not hidden-layer count. `W` has shape `(n, hidden)`. |
| `--J` | 1.0 | Nearest-neighbor Pauli `ZZ` coupling. |
| `--h` | 1.0 | Transverse Pauli `X` field; must be positive. |
| `--steps` | 250 | Number of sampled optimizer updates. |
| `--batch-size` | 128 | IID categorical draws per update, with replacement. |
| `--learning-rate` | 0.05 | Nominal update multiplier `eta`; clipping can reduce it. |
| `--damping` | 0.001 | Positive regularizer `lambda` in the score-covariance solve. |
| `--mu` | 0.9 | Previous raw-direction memory weight, `0 <= mu < 1`. |
| `--max-norm` | 0.1 | Norm radius of a SPRING parameter update, not squared radius. |
| `--seed` | 7 | Initialization seed; categorical sampling starts with seed+1. |
| `--log-every` | 5 | Exact evaluation spacing; initial, first and final states are also saved. |
| `--optimizer` | spring | Fixed by this algorithm folder; other values are rejected. |
| `--output` | fresh run | New directory; existing directories are refused. |

There are `n*hidden+1` parameters: all `W` entries plus normalization gauge `c`.
For defaults: 73 parameters, full input table `(64, 6)`, full score matrix
`(64, 73)`, and sampled score matrix `(128, 73)`. Batch rows can repeat; a batch
of 128 does not mean 128 distinct six-site configurations.

## Loss and optimizer conventions

The loss is the energy expectation, estimated with exact connected local
energies. The covariance expression returns the **full physical gradient**
`g = 2 Cov(O, E_loc)`. JaQMC SPRING receives `g/2`, using its original force
convention. Its raw direction solves the regularized score-covariance system,
with memory. Additional robust-SR/MARCH options are disabled.

The final update is `-min(eta, max_norm/(norm(d)+1e-8))*d`. The memory stores
`d`, not the clipped update. A tiny update, a small sampled gradient, or several
flat recorded energies alone does not prove convergence. Use exact energy,
residual variance, fidelity, and update diagnostics together.

The shared backend also supports VMC with SGD on `g`, and a zero-memory damped
sample-space SR control (`mu=0`). The latter is labelled damped MinSR for this
project; it does not implement the MinSR paper's eigenvalue-truncation rule.
These controls use the same wavefunction, Hamiltonian, loss and sampling.

## Output files

| File | Contents |
|---|---|
| `metrics.json` | CLI configuration, exact ED ground energy, exact initial/final metrics, gradient checks, versions, source hashes and sampling description. |
| `training.csv` | Exact energy/error/variance/fidelity at logged states; sampled energy before the update, gradient norm, raw direction norm and parameter-update norm. The initial row has no update diagnostics. |
| `trained_parameters.npz` | `W` float64 `(n, hidden)` and scalar `c`; no optimizer/RNG state. |
| `gradient_check.json` | Initial full-basis covariance/autodiff/finite-difference agreement and gauge gradient. |
| `gradient_check.npz` | Flattened initial parameters and three physical-gradient references. |
| `diagnostics.png`, `.pdf` | Exact energy-error/variance trajectory and gradient comparison. |

ED data are for **evaluation**, not supervised training targets. The fixed
Hamiltonian is used directly in the energy loss. Exact basis enumeration and
dense diagonalization limit this runner to small systems. For the independent
benchmark database, use [the shared index](../../../benchmarks/README.md).
