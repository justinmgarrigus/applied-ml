# Portable SPRING parameters

| Parameter | Default Ne4 check | Meaning |
|---|---:|---|
| `method` | `spring` | Genuine pinned JaQMC SPRING; wrapper rejects another method |
| `ne`, `flux`, `two_lz` | 4,9,0 | Four electrons, ten LLL orbitals, total Lz=0 |
| `v3` | 0 | Pure normalized P1 interaction, V1=1 |
| `width`, `layers` | 64,6 | Tanh hidden architecture |
| `readout` | `legacy_additive` | Both complex-log channels read every layer |
| `source` | null | Fresh model; no historical/ED/Laughlin weights |
| `init_seed`, `seed` | 7,71 | Parameter and sampling seeds |
| `steps` | 20 | Total optimizer-update target, not fixed-dataset epochs |
| `batch` | 32 | Born-MC rows/walkers per update; duplicates are possible |
| `sweeps`, `burn_sweeps` | 2,50 | MCMC updates and initial burn-in |
| `refresh_every` | 25 | Uniform-sector independence refresh interval |
| `learning_rate` | 0.01 | Nominal coefficient applied to the raw direction |
| `damping` | 0.01 | Positive SR regularization, not a loss penalty |
| `mu` | 0.9 | Carried SPRING direction memory |
| `max_norm` | 0.03 | Applied parameter-step radius sqrt(C), so C=0.0009 |
| `pair_chunk` | 2 | Hamiltonian-connection evaluation chunk size |
| `checkpoint_every`, `log_every` | 10,5 | Saved/printed update intervals |
| `backend`, `cpus` | cpu,2 | Execution choices, not physical parameters |

The Ne6 config changes electron count/flux/output only (Ne6,flux15,16 inputs).
Finite-batch energy/variance estimates are not exact-sector measurements.
The selected historical Ne12 run instead used877454 parameters, angular heads,
an auxiliary head loss and coherent-L2 penalty. Its settings/results are
archival evidence and are not the defaults of this portable comparison.

The shared implementations have explicit conventions: SPRING and damped
MinSR consume `force=Re Cov(O*,Eloc)=gradient(E)/2`; VMC-SGD consumes the full
physical gradient `2*force`. MinSR disables carried memory and solves the
positive-damping sample-space SR equation. It implements no eigenvalue cutoff
or literal Moore-Penrose inverse. VMC uses ordinary Optax SGD with an applied
update-radius cap; Adam/KFAC are not implemented methods here.


For x=2n-1, h1=tanh(xW1+b1) and hl=tanh(h(l-1)Wl+bl).
The two real outputs obey `[a(n),phi(n)]=bias+sum_l hl(n)R_l`, and
`psi_theta(n)=exp(a(n)+i*phi(n))`. Each R_l has shape 64x2; W1 has
shape (n_orbitals,64), later W_l are 64x64. This is a coefficient for the
WHOLE occupation configuration n, not an individual orbital wavefunction.
The schema name `legacy_additive` preserves that all-layer readout meaning;
it is not the final-only readout schema. Newly added deeper readouts can
start at zero and acquire nonzero values during training.


Fresh wrapper outputs default to `results/runs/neN_METHOD_UTC_UUID/`, generated
uniquely on each launch; a config's static output field is not reused by the
wrapper when `--output` is omitted. Resume requires explicit `--output` to
identify the existing directory. The provided example results are protected.
