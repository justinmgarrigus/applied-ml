# Ising results

`reference_spring_n6/` is a byte-preserving copy of the retained six-site,
250-update SPRING result from `Final_proj/SPRING_Ising_test/results/spring_n6`.
Its [provenance manifest](reference_spring_n6/PROVENANCE.json) records each copied
file's source and SHA-256. The original absolute output path in `metrics.json`
is historical metadata and has not been rewritten.

| Exact complete-basis metric | Retained reference |
|---|---:|
| ED ground energy | −7.727406610312546 |
| Trained energy | −7.727390852046698 |
| Energy error | 1.57583 × 10⁻⁵ |
| Residual energy variance | 2.05364 × 10⁻⁴ |
| Ground-state fidelity | 0.9999985691 |

These are exact evaluations of the saved wavefunction, not noisy batch energy
estimates. The close result is a small-system demonstration, not a general
convergence guarantee. Its configuration is `N=6`, hidden=12, batch=128,
eta=0.05, damping=0.001, mu=0.9, max-norm=0.1, seed=7.

New training writes into unique `runs/` directories. Short reorganization
checks are stored in `check_cpu_n4/` and a validation receipt when available;
they validate code paths and do not replace the retained scientific result.
Independent ED data and the database introduction live in the
[shared benchmark area](../../../benchmarks/README.md).
