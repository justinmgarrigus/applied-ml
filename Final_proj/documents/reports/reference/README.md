# Supporting references

The main proposal is [../SPRING_Proposal.pdf](../SPRING_Proposal.pdf). The
documents here provide supplementary detail.

| Document | Purpose |
|---|---|
| [Proposal with appendices](SPRING_Multi_System_Optimizer_Proposal.pdf) | Four-page reading copy: the two-page proposal plus workflow and preliminary ED/NN benchmark appendices |
| [ML explanation](ML_Explanation_Note.pdf) | Three-page guide to input/output arrays, a worked energy calculation and sampled batches |
| [Combined reading packet](SPRING_Proposal_With_ML_Explanation.pdf) | Seven-page convenience copy combining the proposal with appendices and the ML explanation |
| [Paper reading report](2412_00618_reading_report.pdf) | Earlier analysis of arXiv:2412.00618 and possible follow-up research |

The TeX sources, generated numerical data and figures stay together here so the
documents can be rebuilt. `SPRING_Multi_System_Optimizer_Proposal.tex` also
generates the main two-page proposal when compiled with `\OutlineOnly`.
`make_optimizer_proposal_figures.py` regenerates proposal figures and numerical
rows from the saved benchmark records. `make_fqh_report_figure.py` generates the
reading report's analytical illustration.

`proposal_verified_data.json` records benchmark input hashes and counts.
`proposal_validation.json` and `ml_explanation_validation.json` retain the
document validation records. `SPRING_Multi_System_Optimizer_Proposal_sources.zip`
is a portable source bundle for the extended reading copy. Its proposal text
matches the current version, and its frozen figures allow compilation without
rerunning the numerical calculations. Its internal layout and figure-generator
notes predate this folder organization.

See [the build instructions](../../../BUILD_REPORTS.txt) for current commands.
