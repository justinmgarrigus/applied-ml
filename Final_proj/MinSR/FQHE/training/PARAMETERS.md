# Portable MinSR parameters

| Parameter | Default Ne4 check | Meaning |
|---|---:|---|
| `method` | `minsr` | Memory-free damped sample-space SR; wrapper rejects another method |
| `ne`, `flux`, `two_lz` | 4,9,0 | Four electrons, ten LLL orbitals, total Lz=0 |
| `v3` | 0 | Pure normalized P1 interaction, V1=1 |
| `width`, `layers` | 64,6 | Tanh hidden architecture |
| `readout` | `legacy_additive` | Both network output channels read every layer |
| `amplitude_map` | `cartesian` | Anchored complex amplitude readout |
| `source` | null | Fresh model; no historical/ED/Laughlin weights |
| `init_seed`, `seed` | 7,71 | Parameter and sampling seeds |
| `steps` | 5 | Total optimizer-update target, not fixed-dataset epochs |
| `batch` | 32 | MCMC rows per update, with possible repeated occupations |
| `sweeps`, `burn_sweeps` | 2,50 | MCMC updates and initial burn-in |
| `refresh_every` | 25 | Uniform-sector independence refresh interval |
| `learning_rate` | 0.01 | Nominal coefficient applied to the raw direction |
| `damping` | 0.01 | Positive SR regularization, not a loss penalty |
| `mu` | 0 | No carried update memory |
| `max_norm` | 0.03 | Applied parameter-step radius sqrt(C), so C=0.0009 |
| `pair_chunk` | 2 | Hamiltonian-connection evaluation chunk size |
| `checkpoint_every`, `log_every` | 10,5 | Saved/printed update intervals |
| `backend`, `cpus` | cpu,2 | Execution choices, not physical parameters |

The default uses `ne4_cartesian.json` with four electrons at flux 9.
The explicit Ne6 exponential config has six electrons at flux 15.
Finite-batch energy/variance estimates are not exact-sector measurements.
The selected historical Ne12 run instead used877454 parameters, angular heads,
an auxiliary head loss and coherent-L2 penalty. Its settings/results are
archival evidence and are not the defaults of this portable comparison.

SPRING and damped MinSR consume the half-gradient force. For the
exponential/Born path it is `Re Cov(O*, Eloc)`. The Cartesian path uses
the weighted projected amplitude derivative and raw Hamiltonian action.
VMC-SGD consumes the full physical gradient `2*force`. MinSR disables
carried memory and solves the
positive-damping sample-space SR equation. It implements no eigenvalue cutoff
or literal Moore-Penrose inverse. VMC uses ordinary Optax SGD with an applied
update-radius cap; Adam/KFAC are not implemented methods here.


## Cartesian amplitude fit

For the complex network output $\ell_\theta(n)$ and its frozen initial value
$\ell_0(n)$, the default amplitude is

```math
u_\theta(n)=e^{\ell_0(n)}[1+\ell_\theta(n)-\ell_0(n)].
```

The fixed factor $e^{\ell_0(n)}$ sets its derivative scale as the current
amplitude changes. The sampler targets the Born distribution
$p_\theta(n)\propto |u_\theta(n)|^2$. Each sampled row has equal empirical
weight in the physical energy and gradient estimators. The amplitude-space
fit expresses the projected derivative and Hamiltonian residual in that Born
expectation, with inverse density factors. The raw action $(Hu)_n$ uses the
connected signed Hamiltonian rows. Exact zero amplitudes have zero Born
sampling probability. Correlated walkers and finite-batch estimates require
independent checkpoint evaluation.

The explicit `ne4.json` and `ne6.json` configurations use exponential
amplitudes with the same Born sampling target. Resume with the original
configuration and frozen anchor. The shared evaluator reconstructs either
amplitude map from the run metadata.

For x=2n-1, h1=tanh(xW1+b1) and hl=tanh(h(l-1)Wl+bl).
The two real outputs combine as $\ell_\theta(n)$ and obey
`[Re ell(n), Im ell(n)]=bias+sum_l hl(n)R_l`. Each R_l has shape 64x2; W1 has
shape (n_orbitals,64), later W_l are 64x64. This is a coefficient for the
WHOLE occupation configuration n, not an individual orbital wavefunction.
The schema name `legacy_additive` preserves that all-layer readout meaning;
it is not the final-only readout schema. Newly added deeper readouts can
start at zero and acquire nonzero values during training.


Fresh wrapper outputs default to `results/runs/neN_METHOD_UTC_UUID/`, generated
uniquely on each launch; a config's static output field is not reused by the
wrapper when `--output` is omitted. Resume requires explicit `--output` to
identify the existing directory. The provided example results are protected.
