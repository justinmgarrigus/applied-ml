# Submission guide

The primary proposal remains
[SPRING_Proposal.pdf](documents/project_report/SPRING_Proposal.pdf), with
[supporting references](https://github.com/justinmgarrigus/applied-ml/blob/6459402f696bfce70c560aaa26441455deeefa7a/Final_proj/documents/reports/reference/README.md).
The original 2–3 page outline milestone was October 26. This update reorganizes
the supplementary code; it does not replace the final Canvas submission.

## Current code and data

Start at [Final_proj/README.md](README.md), then choose SPRING, VMC-SGD or damped
MinSR and a physics problem. Each physics folder contains PHYSICS.md,
RUN_TRAINING.md, training/PARAMETERS.md, runnable code and results/. ED solvers,
inputs and outputs are shared in [benchmarks](benchmarks/README.md).

The frozen ED database has 11 Hamiltonian instances and 12,085 basis rows
counted per instance: 1,488 Ising and 10,597 FQHE. Its largest decoded input is
8,512 x 22. These are enumerated basis configurations, not a supervised dataset.

Retained results include small-system neural/ED comparisons and later FQHE
research summaries. The stopped Ne12 pure-V1 calculation has not reached its
analytical zero-energy ground state. The new code's execution checks are not a
matched optimizer ranking. A full controlled convergence study, K-FAC and
Hubbard remain future work. See each result index for actual settings and limits.

## Curated Git update

The export includes current algorithm folders, shared code/licensed vendor,
small ED reference data, selected scientific evidence and documents. Raw local
archives, large checkpoints, JAX caches, environments, SSH material and TACC
deployment bundles are excluded. See [migration](MIGRATION.md).

Preview and validate the prepared payload before applying it:

```bash
python Final_proj/tools/validate_reorganization.py
python Final_proj/tools/prepare_submission.py --check --refresh-root-readme
```

The export destination is the existing project repository
`.submission/applied-ml`. Do not initialize Git in the research container.
Publish a reviewable branch/PR and verify the remote head against the exact
commit. SUBMISSION_MANIFEST.json describes packaged files only; archive
provenance does not mean the raw checkpoint is included.

## Course hand-in

According to the retained [course specification](https://github.com/justinmgarrigus/applied-ml/blob/6459402f696bfce70c560aaa26441455deeefa7a/Final_proj/documents/course/Ghosh%20Term%20Project%20Applied%20ML.pdf):

- Final report: one group submission through Canvas, December 7 at midnight.
- Presentation: approximately 15–20 minutes, late November or early December.
- Individual peer assessment: December 7.
- Team size: 3–5; add the assigned roster and required identifiers to the hand-in.
  Confirmed repository contributor: PassengerAn.

Check Canvas for the current rubric and any changed deadlines; those are not
verified by this repository update. Identify reused software and the team's
implementation/experiments separately, and preserve [attribution](THIRD_PARTY.md).
