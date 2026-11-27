# Run MinSR for the FQHE model

Install Python 3.12 dependencies once using the [shared setup](../../shared/README.md).
The wrapper uses [the Ne4 Cartesian config](training/configs/ne4_cartesian.json)
by default. It runs on CPU and selects its optimizer from the shared package.

```bash
python Final_proj/MinSR/FQHE/training/run.py --backend cpu --cpus 2
python Final_proj/MinSR/FQHE/training/run.py \
  --config Final_proj/MinSR/FQHE/training/configs/ne6.json \
  --output /absolute/path/to/a/new/ne6_run --backend cpu --cpus 2
```

Fresh launches without `--output` create a unique timestamp/UUID directory
under `MinSR/FQHE/results/runs/`. Existing output is never overwritten.
The default Ne4 check is 5 optimizer updates, not a convergence claim.
The example uses a fresh six-layer, width-64, all-layer-readout model with
initialization seed 7, sampling seed 71, and batch 32. It samples the target
$q(n)\propto |u_\theta(n)|^2+a$ with fixed $a=1/3$ and uses inverse target
weights to estimate the physical energy force. The explicit
[Ne4](training/configs/ne4.json) and [Ne6](training/configs/ne6.json) configs
use exponential amplitudes and Born sampling.

Resume with the original configuration and an explicit existing run directory.
This example resumes a run started with the current Cartesian default:

```bash
python Final_proj/MinSR/FQHE/training/run.py --resume \
  --config Final_proj/MinSR/FQHE/training/configs/ne4_cartesian.json \
  --output /absolute/path/to/existing/run --steps 100 --backend cpu --cpus 2
```

When `--config` is omitted, the wrapper reads the saved amplitude map and
selects the matching Ne4 configuration. Supply the original configuration
explicitly for Ne6 or custom settings. Keep `run_config.json` and
`initial_parameters.npz` with the saved checkpoint so the Cartesian state can
be reconstructed. The evaluator accepts `--run-dir` for copied checkpoints.

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
