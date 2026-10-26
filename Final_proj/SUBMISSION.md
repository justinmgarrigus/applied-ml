# Submission guide

This folder contains the project outline, supporting explanations, reproducible
code, ED benchmarks and preliminary neural-network results. GitHub provides
supplementary materials; the course's final written report also goes to Canvas.

## Current milestone: project outline

The outline is due **October 26** and should be **2–3 pages**, covering the
problem, available data, possible approaches and selected references.

- [Main proposal — 2 pages](documents/reports/SPRING_Proposal.pdf): the primary course outline, previously named `Project_Outline.pdf`.
- [Supporting references](documents/reports/reference/README.md): extended reading copies, ML explanation and paper analysis.

Use `SPRING_Proposal.pdf` for the required document. The longer PDFs in
`documents/reports/reference/` supplement the same proposal if permitted.
The local course guidelines do not specify the
outline's upload destination. Check the assignment on Canvas.

## Present data and results

The ED database covers **11 Hamiltonian instances** and **12,085 configuration
rows**, counted separately for each instance: 1,488 Ising and 10,597 FQHE rows.
The largest decoded input array is **8,512 × 22**, or **187,264 input entries**.
These are enumerated configurations, rather than retained raw Monte Carlo
samples. The same six-spin Ising basis is reused at three field settings.

Existing NN results are preliminary. The full matched optimizer comparison
and Hubbard calculations remain proposed work. Planned long training runs
should be described as future experiments.

## Remaining course deliverables

- **Final report:** due **December 7 at midnight through Canvas**, one submission
  per group. Include the repository URL for code, data and supporting materials.
- **Presentation:** approximately **15–20 minutes per group**, scheduled for
  late November or early December.
- **Peer assessment:** each team member submits their own assessment by
  **December 7**.
- **Team details:** the course specifies teams of **3–5 students**. Confirm the
  assigned roster and add required names/identifiers before the course hand-in.
  The confirmed repository contributor handle is `PassengerAn`.

Check **Canvas → Modules → Projects** for the detailed presentation/report
rubric. Those additional instructions have not been verified here. The
[local course specification](<documents/course/Ghosh Term Project Applied ML.pdf>)
provides the requirements summarized above: outline and submission milestones
on page 1, data guidance on page 2, and reuse/peer-assessment guidance on page 3.

Retain upstream licenses and [source attribution](THIRD_PARTY.md). The final
report should identify reused code and the team's implementation, experiments
and analysis separately.
