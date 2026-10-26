# Applied ML final project

Contributor: [PassengerAn](https://github.com/PassengerAn).

Physics-guided neural-state learning with energy as the objective. Ising is
our entry-level benchmark; FQHE is an implemented advanced example, with
Hubbard proposed as an alternative. The project will compare SPRING,
SR/MinSR, K-FAC and Adam/SGD under controlled conditions.

## Read first

- [Main proposal — 2 pages](Final_proj/documents/reports/SPRING_Proposal.pdf)
- [Supporting reference documents](Final_proj/documents/reports/reference/README.md)
- [Course submission guide](Final_proj/SUBMISSION.md)
- [Project contents and current results](Final_proj/README.md)
- [ED benchmark database](Final_proj/benchmarks/README.md)
- [Reused code and references](Final_proj/THIRD_PARTY.md)

`SPRING_Proposal.pdf` is the primary course proposal. The `reference/` folder
contains extended reading copies, explanations, paper analysis and editable
sources for supplementary use.

The ED database covers 11 Hamiltonian instances and 12,085 configuration rows
counted per instance; its largest decoded input array is 8,512 × 22.
Current neural results are preliminary. The full optimizer comparison,
Hubbard experiments and planned 200,000-update runs remain future work.

## Reproduce

For the small Ising project, install Python 3.12 and uv, then:

```bash
cd Final_proj/SPRING_Ising_test
bash setup.sh
.venv/bin/python -m pytest -q
```

See the [Ising instructions](Final_proj/SPRING_Ising_test/README.md) and
[FQHE instructions](Final_proj/SPRING_Haldane_FQHE/README.md) for training,
benchmark regeneration and evaluation. Environments are created
locally from the supplied dependency specifications.

The [submission manifest](SUBMISSION_MANIFEST.json) records the packaged
file hashes. Upstream licenses and numerical provenance are retained.
