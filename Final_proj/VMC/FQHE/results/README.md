# VMC FQHE results

This folder stores short executable CPU checks of the shared energy-only
model, including exact Ne4 post-training ED diagnostics. Five updates are a
wiring/state-format check, not a convergence result or method ranking.

Physical/NN initialization is matched across methods. Optimizer conventions
and budgets are explicit in each run's `run_config.json`. Large historical
Ne12 research results remain separate under SPRING; they are not VMC
outputs. Shared ED references stay in `Final_proj/benchmarks`.

The checked fresh Ne4 run completed5 updates and has
exact final E/V1=2.933259201, fidelity=0.031152231,
variance=1.373131664 V1². These are deliberately short
nonconverged checks. `METHOD_CHECK.json` binds the actual sources/checkpoint.
Initial arrays match SPRING and the other method bit for bit; the checkpoint
split/resume test also passes. All 44 shared CPU tests passed.

The curated execution artifacts are in `check_cpu_ne4/`; their copied metadata
keeps the original run path. Fresh auto-named `runs/` outputs are local only.
