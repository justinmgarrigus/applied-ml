# ML final project

Code, benchmark data and preliminary results for comparing optimizers in
energy-based neural-state learning. Ising is the entry-level problem; FQHE is
an implemented advanced example, and Hubbard is a proposed extension. The full
matched comparison of SPRING, SR/MinSR, K-FAC and Adam/SGD remains project work.

Start with the [two-page outline](documents/reports/Project_Outline.pdf) and
[submission guide](SUBMISSION.md). Contributor handle: **PassengerAn**.

| Folder | Use |
|---|---|
| [FQHE code](SPRING_Haldane_FQHE/README.md) | Born-sampled SPRING code, selected checkpoints, verified results and tests |
| [TACC instructions](SPRING_Haldane_FQHE/TACC.md) | Installation, Slurm submission, monitoring and full-state resume |
| [TACC transfer bundle](Haldane_FQHE_TACC.tar.gz) | Standalone code, configurations, selected results and reference data |
| [ED benchmarks](benchmarks/README.md) | FQHE N=4–8 and Ising N=4,6,8,10 exact references, spectra, gaps and neural comparisons |
| [Revised proposal](documents/reports/SPRING_Multi_System_Optimizer_Proposal.pdf) | Strongly correlated systems: Ising entry level, Hubbard/FQHE advanced; two-page proposal plus appendices |
| [ML explanation attachment](documents/reports/ML_Explanation_Note.pdf) | Three-page guide to input/output arrays, a worked energy calculation, NN coefficients and sampled batches |
| [Proposal with attachment](documents/reports/SPRING_Proposal_With_ML_Explanation.pdf) | The proposal, workflow/results appendices and ML explanation in one PDF |
| [Ising code](SPRING_Ising_test/README.md) | Small-system SPRING/SGD demonstration and gradient validation |
| [documents/papers](documents/papers/) | Source SPRING paper |
| [documents/reports](documents/reports/) | Final proposal, detailed reading report, editable TeX and report figure |
| [documents/course](documents/course/) | Course project specification |

The ED database covers 11 Hamiltonian instances and 12,085 configuration rows
counted per instance: 1,488 Ising and 10,597 FQHE rows. The largest decoded NN
input array is 8,512 × 22. Detailed counts, saved NN table shapes and source
hashes are in [the verified data record](documents/reports/proposal_verified_data.json).

The selected N=8 architecture pilot retains all six endpoints and their exact
diagnostics. Its best dense energy is `0.490844981335 V1`; it has not reached
the zero-energy ground state. The 200,000-update configurations provide a
future training budget; those long runs have not been performed.

See [BUILD_REPORTS.txt](BUILD_REPORTS.txt) for document sources and
[third-party attribution](THIRD_PARTY.md) for reused code and papers.
