# Sources and third-party code

All portable SPRING examples use the JaQMC optimizer at commit
`71f94158f2aca88ded904451f5314f71cdff5666`. The shared runtime subset retains
the upstream [LICENSE](shared/vendor/jaqmc_spring/LICENSE),
[NOTICE](shared/vendor/jaqmc_spring/NOTICE) and
[provenance](shared/vendor/jaqmc_spring/PROVENANCE.json).
The optimizer files are upstream code; the small packaging wrapper is described
separately in that provenance. A second unmodified copy accompanies the frozen
research source snapshot where needed to preserve its imports.

The original full Ising JaQMC tree and earlier FQHE tree remain local archives.
The Git export uses the small licensed subset, rather than exporting unrelated
upstream examples, caches or installation artifacts. The original Ising source
attribution is retained in [its provenance](shared/ising_legacy_provenance.json).

Project code supplies Hamiltonians, encodings, wavefunction models, samplers,
training adapters, ED solvers, evaluations and experiments. SPRING is reused
software. New VMC-SGD and memory-free damped SR controls are execution/linear
algebra validated; a controlled convergence comparison remains future work.
The MinSR control is finite-damping SR, without claiming another implementation's
eigenvalue cutoff. K-FAC and Hubbard remain proposed extensions.

Scientific references:

- Goldshlager, Abrahamsen and Lin, [SPRING](https://arxiv.org/abs/2401.10190).
- [Original VMCNet SPRING implementation](https://github.com/jeffminlin/vmcnet/blob/bd23bb3f6171e8b741a6bfd47a45ebf550060cd9/vmcnet/updates/spring.py).
- Teng, Dai and Fu, [neural wavefunctions for FQHE](https://arxiv.org/abs/2412.00618).
- Carleo and Troyer, [neural quantum states](https://arxiv.org/abs/1606.02318).
- Chen and Heyl, [MinSR](https://arxiv.org/abs/2302.01941).
- Martens and Grosse, [K-FAC](https://arxiv.org/abs/1503.05671).
- Kingma and Ba, [Adam](https://arxiv.org/abs/1412.6980).

Upstream component licenses apply to their components. This submission does not
replace them with a new project-wide license. The proposal includes further
references. Historical PDFs preserve their original status and conclusions.
