# FQHE results

## Checked portable SPRING example

The fresh Ne4 six-layer64-wide all-readout NN completed20 CPU updates using
upstream JaQMC SPRING. All 44 shared CPU tests pass, including 27 FQHE physics,
optimizer, model, ED/evaluation and three-method split/resume checks.
Exact complete-sector energy changed3.27318809→2.84420344 V1; final ED fidelity
is0.0255033 and variance1.535645 V1². This is **not converged**. It establishes
runnable code, optimizer parity and independent evaluation, not a performance
ranking. See `SPRING_CHECK.json` and its bound run under `check_cpu_ne4/`.

## Separate archived research evidence

- [Ne7 research results](research_ne7/README.md): no-wall mixed Hamiltonian
  P1+0.8P3+0.6P5, latest54000-update state, E=6.40426894 versus
  ED6.24476826 V1 and fidelity0.778168791. Its Hamiltonian differs from the
  portable pure-V1 baseline; this is not a controlled size/method comparison.
- [Ne12 user-stopped results](research_ne12/README.md): pureV1 with angular
  heads/auxiliary losses, stopped at13721, frozen Born-MC E≈3.63448±0.02660 V1.
  No Ne12 ED fidelity is claimed. The final full resume checkpoint SHA is
  `54458312dae2fa4841c4053397b677018a09cd3e403fce54111ea7407c672a18`.
- [Advanced source-only snapshot](../training/research_snapshot/README.md):
  preserved angular-head/positive-L2/sampling/runtime code with explicit
  dependencies. The portable energy-only model does not silently reproduce it.

Copied research artifacts and their provenance remain byte-identical. Full
raw inputs/checkpoints stay in the legacy archive; original absolute metadata
is not rewritten. Nothing here authorizes resuming the stopped Ne12 run.
The short portable checks and historical systems are not causal evidence for
optimizer/depth/size superiority. Shared ED data stay in Final_proj/benchmarks.

The curated execution artifacts are in `check_cpu_ne4/`; their copied metadata
keeps the original run path. Fresh auto-named `runs/` outputs are local only.
