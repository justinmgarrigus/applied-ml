# Verified N=8 architecture pilot

All six runs used the same initial wavefunction, ordinary Born sampling,
batch 32 and 1,000 additional SPRING updates. Energies, fidelities and variances
were evaluated over all 8,512 configurations after training.

| Model | Mean energy / V1 | Mean Laughlin fidelity | Mean training time |
|---|---:|---:|---:|
| Six-layer MLP continuation | 0.49973545 | 35.4091% | 28.09 s |
| Residual dense correction | 0.49158096 | 36.2938% | 29.62 s |
| Orbital-attention correction | 0.50427475 | 34.6992% | 148.74 s |

These are descriptive means over sampling seeds 71 and 137, with one common
neural initialization. The primary comparison was attention versus a dense
correction with nearly equal added parameter counts. The unchanged MLP was
added as an auxiliary control after the first dense endpoint was available.
No state reached the zero-energy ground state; mean L squared remains near30.
Attention did not help at this short budget. This does not determine what
happens after 200,000 updates or after architecture-specific tuning.

`summary.json`, `endpoints.csv` and `exact_trajectory.csv` retain the numerical
results. `comparison.png/pdf` shows the scientific comparison. `checkpoints/`
contains all six endpoint parameter dictionaries, normalized coefficients and
compact training summaries. The files under `provenance/` are the original
validation records; old source paths there identify the archived experiment,
not dependencies of the clean package. The active code is tested separately.

The physics is the strictly LLL Haldane sphere at N=8, flux21, M=0 and pure
positive V1 pseudopotential. Neither a Laughlin factor nor an L=0 constraint was
added. Antisymmetry is supplied by the canonical Slater basis. The attention
model learns occupation coefficients; it is not a real-space FermiNet.
