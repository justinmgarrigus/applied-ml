# ML final project

Code, benchmark data and preliminary results for comparing optimizers in
energy-based neural-state learning. Ising is the entry-level problem; FQHE is
an implemented advanced example, and Hubbard is a proposed extension. The full
matched comparison of SPRING, SR/MinSR, K-FAC and Adam/SGD remains project work.

Start with the [main proposal (two pages)](documents/reports/SPRING_Proposal.pdf) and
[submission guide](SUBMISSION.md). Contributor handle: **PassengerAn**.

| Folder | Use |
|---|---|
| [FQHE code](SPRING_Haldane_FQHE/README.md) | Born-sampled SPRING code, selected checkpoints, verified results and tests |
| [ED benchmarks](benchmarks/README.md) | FQHE N=4–8 and Ising N=4,6,8,10 exact references, spectra, gaps and neural comparisons |
| [Main proposal](documents/reports/SPRING_Proposal.pdf) | Two-page course proposal: Ising entry level, Hubbard/FQHE advanced |
| [Supporting references](documents/reports/reference/) | Extended reading copies, ML explanation, paper analysis, editable sources, figures and validation records |
| [Ising code](SPRING_Ising_test/README.md) | Small-system SPRING/SGD demonstration and gradient validation |
| [documents/papers](documents/papers/) | Source SPRING paper |
| [documents/reports](documents/reports/) | Main proposal and a guide to supporting references |
| [documents/course](documents/course/) | Course project specification |

The ED database covers 11 Hamiltonian instances and 12,085 configuration rows
counted per instance: 1,488 Ising and 10,597 FQHE rows. The largest decoded NN
input array is 8,512 × 22. Detailed counts, saved NN table shapes and source
hashes are in [the verified data record](documents/reports/reference/proposal_verified_data.json).

The selected N=8 architecture pilot retains all six endpoints and their exact
diagnostics. Its best dense energy is `0.490844981335 V1`; it has not reached
the zero-energy ground state. The 200,000-update configurations provide a
future training budget; those long runs have not been performed.

See [BUILD_REPORTS.txt](BUILD_REPORTS.txt) for document sources and
[third-party attribution](THIRD_PARTY.md) for reused code and papers.
