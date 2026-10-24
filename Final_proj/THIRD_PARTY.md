# Sources and third-party code

## Reused optimizer implementation

Both demonstrations use the SPRING implementation from
[ByteDance JaQMC](https://github.com/bytedance/jaqmc), pinned to commit
`71f94158f2aca88ded904451f5314f71cdff5666`.

- Ising includes the upstream source archive in `SPRING_Ising_test/vendor/jaqmc`.
  Its [LICENSE](SPRING_Ising_test/vendor/jaqmc/LICENSE),
  [NOTICE](SPRING_Ising_test/vendor/jaqmc/NOTICE), and
  [provenance](SPRING_Ising_test/PROVENANCE.json) are retained.
- FQHE includes a minimal runtime subset in
  `SPRING_Haldane_FQHE/vendor/jaqmc_spring`, with its
  [LICENSE](SPRING_Haldane_FQHE/vendor/jaqmc_spring/LICENSE),
  [NOTICE](SPRING_Haldane_FQHE/vendor/jaqmc_spring/NOTICE), and
  [file hashes and provenance](SPRING_Haldane_FQHE/vendor/jaqmc_spring/PROVENANCE.json).
  The algorithm files match upstream; the provenance describes the small
  packaging wrapper separately.

The upstream notices include their own dependencies and acknowledgments.
Their licenses apply to those components; this submission does not replace
them with a new project-wide license.

## Scientific references

- Goldshlager, Abrahamsen and Lin, [SPRING](https://arxiv.org/abs/2401.10190).
- [Original VMCNet SPRING implementation](https://github.com/jeffminlin/vmcnet/blob/bd23bb3f6171e8b741a6bfd47a45ebf550060cd9/vmcnet/updates/spring.py), cited for method provenance.
- Teng, Dai and Fu, [neural wavefunctions for FQHE](https://arxiv.org/abs/2412.00618).
- Carleo and Troyer, [neural quantum states](https://arxiv.org/abs/1606.02318).
- Chen and Heyl, [MinSR](https://arxiv.org/abs/2302.01941).
- Martens and Grosse, [K-FAC](https://arxiv.org/abs/1503.05671).
- Kingma and Ba, [Adam](https://arxiv.org/abs/1412.6980).

The proposal contains further references and links.

## Project contribution and current scope

Project-specific code supplies the model encodings, NN parameterizations,
sampling and training wrappers, independent numerical checks, ED benchmark
records, evaluation tools and preliminary experiments. SPRING itself is reused
upstream software. The controlled multi-optimizer study and Hubbard extension
are proposed experiments, rather than completed results.
