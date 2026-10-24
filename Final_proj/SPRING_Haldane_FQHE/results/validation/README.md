# Portable-release validation

`release_validation.json` summarizes the checked release on 23 October 2026.
The evidence includes eight passing focused tests, reproduction of all six
pilot endpoints within 3e-14, full-state resume equivalence within 1e-12,
an N=8 continuation after moving the package, and graceful signal checkpointing.

`pytest_results.xml`, `endpoint_reproduction.json` and
`packaging_validation.json` retain detailed evidence. Temporary local paths in
these records identify the validation environment; they are not runtime inputs.
The wheel checks used an isolated local installation, not a TACC compute node.
GPU execution and long training remain to be tested on the selected machine.

The 20/25-update relocation check is a software smoke test, not a new scientific
comparison or evidence of ground-state convergence. The retained scientific
comparison is in `../ne8_architecture_pilot/`.
