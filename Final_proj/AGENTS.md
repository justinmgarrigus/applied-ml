# Instructions for agents working in Final_proj

Read this file, the selected algorithm's README, then the physics folder's
PHYSICS.md, RUN_TRAINING.md and training/PARAMETERS.md before editing.

## Ownership and layout

- Canonical entry points: `SPRING|VMC|MinSR/{Ising,FQHE}/training/run.py`.
- Each physics folder has `training/` and `results/`; ED belongs in shared
  `benchmarks/`, not a third per-method copy.
- `shared/ising_backend.py` and `shared/fqh_spring/` own physics, models,
  sampling and energy estimation. Fix common physics there and check affected
  methods. Do not fork separate Hamiltonians for each optimizer.
- The VMC folder means energy-gradient SGD within VMC. SPRING and MinSR are also
  VMC optimizers. MinSR here is finite-damping, memory-free SR; do not claim an
  exact reproduction of another program's eigenvalue filtering.
- The old local physics-first research trees are frozen archives. Do not edit,
  move, delete or overwrite their code, checkpoints, measurements, manifests
  or absolute-path metadata. See MIGRATION.md.
- Archived numerical summaries describe original algorithms/settings. Never
  relabel them as results from newly organized code.

## Scientific checks

- ED labels/eigenvectors must not enter baseline training, initialization or
  batch selection. ED is evaluation only.
- Distinguish exact full-sector energy/fidelity from Monte Carlo estimates.
  Report units, uncertainty, seed, checkpoint hash, Hamiltonian and sampling.
- FQHE uses LLL occupations at fixed particle number and Lz. Lz=0 is not total
  L=0. Fermionic signs enter the Hamiltonian action.
- Use `readout=legacy_additive` for all-layer readouts; `final` is a different
  architecture. Layer count is not inferred from electron count.
- The portable baseline has no wall, V5, angular penalty or artificial
  temperature. Archived angular-aware experiments are a separate protocol.
- The cap argument is norm radius sqrt(C), not the paper's squared C.
- One training step is one optimizer update, not an epoch over a fixed dataset.
- Tiny execution checks do not prove convergence or a speed advantage.

## Execution and publication

Examples are not permission to launch long training. Checks use fresh output
directories. Resume only when asked and only with compatible code/settings,
restoring RNG and optimizer memory. Never silently load an incompatible
research checkpoint into the portable trainer.

Run layout and relevant physics/optimizer checks after editing.
Preview curated Git export with `tools/prepare_submission.py --check`.
Never stage raw archives, environments, JAX caches, SSH material, large
checkpoints or deployment bundles. Check the Git boundary per parent
instructions. The submission checkout is `.submission/applied-ml` in the
course workspace, not the research container.
