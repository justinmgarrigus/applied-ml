# Algorithm-first migration, 2026-11-17

The canonical layout groups methods first (SPRING, VMC, MinSR), then physics
(Ising, FQHE). Each physics folder has training/ and results/, plus physics and
execution notes. At the user's correction, ED remains under shared benchmarks/.

## Preservation

Local `SPRING_Ising_test/` and `SPRING_Haldane_FQHE/` remain in place as frozen
archives. Checkpoints/audits bind absolute paths, source hashes and relative
package roots. They are not the starting point for new work. The shared
backend is a separately owned portable snapshot, so edits do not alter an
archived experiment.

Small ED arrays are copied byte for byte to the common benchmark area.
benchmarks/migration_manifest.json records their destination hashes.
Historical results keep their original settings/checkpoint provenance.
The stopped Ne12 checkpoint stays local; its summary does not imply that a
portable baseline can resume its angular-head model.

## Git payload

Curated export includes the algorithm folders, common code/licensed vendor,
small ED references/solvers, selected evidence, documents and validation tools.
Raw archives, environments, caches, large checkpoints, deployment bundles and
SSH material stay local. Preview with tools/prepare_submission.py --check.
The generated submission manifest describes only included files; archival hash
records do not imply that every raw input is shipped.

The proposal is under documents/project_report/SPRING_Proposal.pdf. The
presentation folder is documents/presentation/. Earlier reports, slide
sources and other references remain available in the [archived document tree](https://github.com/justinmgarrigus/applied-ml/tree/6459402f696bfce70c560aaa26441455deeefa7a/Final_proj/documents).
