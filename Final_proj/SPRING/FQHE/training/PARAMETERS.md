# Portable SPRING parameters

| Parameter | Default Ne4 check | Meaning |
|---|---:|---|
| `method` | `spring` | Genuine pinned JaQMC SPRING; wrapper rejects another method |
| `ne`, `flux`, `two_lz` | 4,9,0 | Four electrons, ten LLL orbitals, total Lz=0 |
| `v3` | 0 | Pure normalized P1 interaction, V1=1 |
| `width`, `layers` | 64,6 | Tanh hidden architecture |
| `readout` | `legacy_additive` | The two output channels read every layer |
| `amplitude_map` | `cartesian` | Anchored amplitude fit for SPRING |
| `source` | null | Fresh model; no historical/ED/Laughlin weights |
| `init_seed`, `seed` | 7,71 | Parameter and sampling seeds |
| `steps` | 20 | Total optimizer-update target, not fixed-dataset epochs |
| `batch` | 32 | MCMC rows per update, with possible repeated occupations |
| `sweeps`, `burn_sweeps` | 2,50 | MCMC updates and initial burn-in |
| `refresh_every` | 25 | Uniform-sector independence refresh interval |
| `learning_rate` | 0.01 | Nominal coefficient applied to the raw direction |
| `damping` | 0.01 | Positive SR regularization, not a loss penalty |
| `mu` | 0.9 | Carried SPRING direction memory |
| `max_norm` | 0.03 | Applied parameter-step radius sqrt(C), so C=0.0009 |
| `pair_chunk` | 2 | Hamiltonian-connection evaluation chunk size |
| `checkpoint_every`, `log_every` | 10,5 | Saved/printed update intervals |
| `backend`, `cpus` | cpu,2 | Execution choices, not physical parameters |

The default uses `ne4_cartesian.json`. The explicit `ne4.json` and `ne6.json`
baseline configs use exponential amplitudes and Born sampling.
The optional `ne8_cartesian.json` config uses the anchored map with eight
electrons at flux 21.
Finite-batch energy/variance estimates are not exact-sector measurements.
The selected historical Ne12 run instead used877454 parameters, angular heads,
an auxiliary head loss and coherent-L2 penalty. Its settings/results are
archival evidence and are not the defaults of this portable comparison.

SPRING and damped MinSR use a half-gradient force. The exponential/Born
form estimates it as `Re Cov(O*, Eloc)`. Cartesian SPRING estimates it
in amplitude space using the weighted projected tangent below. VMC-SGD
consumes the full physical gradient. MinSR disables carried memory and solves the
positive-damping sample-space SR equation. It implements no eigenvalue cutoff
or literal Moore-Penrose inverse. VMC uses ordinary Optax SGD with an applied
update-radius cap; Adam/KFAC are not implemented methods here.

## Cartesian amplitude fit

Let $\ell_\theta(n)$ combine the two real network outputs and let $\ell_0(n)$
come from the frozen initial parameters. The anchored map uses

~~~math
u_\theta(n)=e^{\ell_0(n)}[1+\ell_\theta(n)-\ell_0(n)].
~~~

It gives the same initial state and derivative as the exponential map. The
derivative scale $e^{\ell_0(n)}$ stays fixed as the current amplitude changes.
The sampler targets the Born distribution
$p_\theta(n)\propto |u_\theta(n)|^2$ during burn-in, local pair moves, and
uniform-sector refreshes.

For sampled occupations, the fit uses equal empirical weights in the Born
expectation. In the amplitude-space calculation, inverse Born-density factors
express the projected wavefunction derivative and Hamiltonian residual in
that expectation. The raw action $(Hu)_n$ uses the signed pair hops at sampled
occupations. Exact zero amplitudes have zero Born sampling probability.
Correlated walkers and finite-batch estimates require separate frozen-state
energy, variance, and fidelity checks.

For `x=2n-1`, the hidden layers obey `h1=tanh(xW1+b1)` and
`hl=tanh(h(l-1)Wl+bl)`. The two real outputs forming $\ell_\theta(n)$ are
`bias+sum_l hl(n)R_l`. Each `R_l` has shape 64x2. The first weight matrix
has shape `(n_orbitals,64)` and later weight matrices are 64x64.
Each output describes a whole occupation configuration. The schema
`legacy_additive` reads every layer, and deeper readouts initialized to zero
can acquire nonzero values during training.


Fresh wrapper outputs default to `results/runs/neN_METHOD_UTC_UUID/`, generated
uniquely on each launch; a config's static output field is not reused by the
wrapper when `--output` is omitted. Resume requires explicit `--output` to
identify the existing directory. The provided example results are protected.
