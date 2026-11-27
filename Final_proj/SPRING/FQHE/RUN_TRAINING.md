# Run SPRING for the FQHE model

Install Python 3.12 dependencies once using the [shared setup](../../shared/README.md).
The wrapper uses [the Ne4 Cartesian config](training/configs/ne4_cartesian.json)
by default. It runs on CPU and selects the SPRING implementation from the
shared package.

```bash
python Final_proj/SPRING/FQHE/training/run.py --backend cpu --cpus 2
python Final_proj/SPRING/FQHE/training/run.py \
  --config Final_proj/SPRING/FQHE/training/configs/ne6.json \
  --output /absolute/path/to/a/new/ne6_run --backend cpu --cpus 2
```

Fresh launches without `--output` create a unique timestamp/UUID directory
under `SPRING/FQHE/results/runs/`. Existing output is never overwritten.
The default Ne4 check is 20 optimizer updates, not a convergence claim.
It uses a six-layer, width-64, all-layer-readout model with initialization
seed 7, sampling seed 71, and batch 32. The explicit
[Ne4](training/configs/ne4.json) and [Ne6](training/configs/ne6.json) configs
use the exponential readout and Born sampling for the baseline comparison.

## Cartesian SPRING with Born sampling

The default fits the physical residual at sampled occupations in the
pure-$V_1$, $L_z=0$ sector. Its Metropolis target is the current Born
distribution $p_\theta(n)\propto |u_\theta(n)|^2$. The same approach is
available for eight electrons.

~~~bash
python Final_proj/SPRING/FQHE/training/run.py \
  --config Final_proj/SPRING/FQHE/training/configs/ne8_cartesian.json \
  --output /absolute/path/to/a/new/ne8_cartesian_run --backend cpu --cpus 2
~~~

Keep `initial_parameters.npz` and `run_config.json` beside the latest checkpoint or
one directory above `checkpoints/step_*.npz`. They identify the frozen initial
amplitudes and the readout used to reconstruct the state. The evaluator accepts
`--run-dir` when a checkpoint is copied away from its original run directory.
The amplitude map and frozen anchor are part of the run signature.

## Resume and evaluate

Resume with the original training config and an explicit existing run directory.
This example resumes a run started with the current default.

```bash
python Final_proj/SPRING/FQHE/training/run.py --resume \
  --config Final_proj/SPRING/FQHE/training/configs/ne4_cartesian.json \
  --output /absolute/path/to/existing/run --steps 100 --backend cpu --cpus 2
```

When `--config` is omitted, the wrapper reads the recorded amplitude map
and selects the matching Ne4 config. Pass the original config explicitly for
Ne6, Ne8, or custom settings.

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
