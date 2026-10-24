# Initialization checkpoints

- `source_ne8.npz`: common six-layer MLP source for controlled dense and
  attention experiments; exact E=0.504530691045 V1.
- `best_dense_ne8.npz`: lowest-energy dense endpoint from the two-seed pilot;
  exact E=0.490844981335 V1. This is a selected endpoint, not an unbiased
  estimate of a typical run.

`manifest.json` records hashes and exact diagnostics. These files contain
parameters only: starting a new run from one resets SPRING memory and burns
fresh walkers. Exact continuation uses the full `checkpoint_latest.npz` of a
clean-runtime run, including optimizer memory and both random states.
