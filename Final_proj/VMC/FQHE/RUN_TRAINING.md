# Run VMC for the FQHE model

Install Python 3.12 dependencies once using the [shared setup](../../shared/README.md).
The wrapper selects the shared package/vendor explicitly and rejects a config
belonging to another algorithm. It runs on CPU by default.

```bash
python Final_proj/VMC/FQHE/training/run.py --backend cpu --cpus 2
python Final_proj/VMC/FQHE/training/run.py \
  --config Final_proj/VMC/FQHE/training/configs/ne6.json \
  --output /absolute/path/to/a/new/ne6_run --backend cpu --cpus 2
```

Fresh launches without `--output` create a unique timestamp/UUID directory
under `VMC/FQHE/results/runs/`. Existing output is never overwritten.
The default Ne4 check is 5 optimizer updates, not a convergence claim.
All three methods start the same fresh six-layer 64-wide all-layer-readout
model (init seed 7, sampling seed 71, batch 32). The short stored checks have
unequal budgets and are not a scientific speed/accuracy ranking.

Resume always requires an explicit existing run directory:

```bash
python Final_proj/VMC/FQHE/training/run.py --resume \
  --output /absolute/path/to/existing/run --steps 100 --backend cpu --cpus 2
```

`--steps 100` is a TOTAL target; parameters, optimizer state, walkers and both
RNGs are restored. Source/method/physics/model signatures must match. This
portable format differs from the archived Ne12 experiment checkpoints, whose
step budgets are additional attempts. No command here resumes the stopped
Ne12 research run. SIGUSR1/SIGTERM request an atomic safe checkpoint.

Independent small ED and frozen-state evaluation are implemented commands:

```bash
PYTHONPATH=Final_proj/shared python -m fqh_spring.references \
  --ne 4 --eigenpairs 4 --output /absolute/path/to/new/ne4_reference
PYTHONPATH=Final_proj/shared JAX_PLATFORMS=cpu python -m fqh_spring.evaluate \
  --checkpoint /absolute/path/to/run/checkpoint_latest.npz \
  --reference /absolute/path/to/ne4_reference --angular \
  --output /absolute/path/to/new/exact_evaluation.json
```

Shared ED tables/references stay under `Final_proj/benchmarks`; no per-method
ED copies are needed. See `PHYSICS.md` and `training/PARAMETERS.md` for the
Hamiltonian, gradient convention and active optimizer settings.
