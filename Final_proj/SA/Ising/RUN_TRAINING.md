# Run weight-space simulated annealing

Install the [shared environment](../../shared/README.md), then from the repository root:

```bash
python Final_proj/SA/Ising/training/run.py --n 4 --hidden 3 --steps 20
```

This creates a fresh directory under `SA/Ising/results/runs/`. Existing explicit
output destinations are refused. For a deliberately larger experiment:

```bash
python Final_proj/SA/Ising/training/run.py \
  --n 6 --hidden 12 --steps 1000 --seed 7 --search-seed 8 \
  --temperature 0.1 --final-temperature 0.0001 \
  --proposal-scale 0.05 --final-proposal-scale 0.005
```

These are starting hyperparameters, not tuned settings. Read
[PARAMETERS.md](training/PARAMETERS.md) before tuning. There is no exact resume
support. The default run starts from the same shared initialization as the
gradient runners for the same n, hidden and seed.

`training.csv` records every attempted proposal, including rejected attempts,
current/best energy, acceptance, temperature, proposal norm and elapsed time.
`metrics.json` records exact endpoint diagnostics, versions, source/parameter
hashes and evaluation counts. `trained_parameters.npz` is the last state;
`best_parameters.npz` is the lowest-loss visited state. They may differ because
uphill moves are permitted. Use a consistent last/best selection rule across
methods. All three parameter artifacts include W and c.

The search makes steps+1 objective evaluations, plus one separately recorded
JIT warm-up evaluation. Three post-training endpoint diagnostics and ED are
outside the reported training time. Compilation time is reported separately.
For wall-time comparisons include compilation/evaluation consistently for all
methods. An SA step is one proposed weight update, including rejection.

Run the shared tests with `python -m pytest -q Final_proj/shared/tests`.
The [results notes](results/README.md) describe the scope of execution checks.
