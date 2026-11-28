# VMC with Adam: Ising parameters

Flags are those of [VMC/SGD](../../../VMC/Ising/training/PARAMETERS.md) with these differences.
`--damping` and `--mu` are inactive shared-parser fields.

| Flag | Default | Meaning |
|---|---:|---|
| `--learning-rate` | 0.01 | Adam step size (optax.adam with b1=0.9, b2=0.999, eps=1e-8). |
| `--max-norm` | 0.1 | Active norm radius applied after the Adam transform. |
| `--optimizer` | adam | Fixed by this entry point. |
| `--adam-beta2` | 0.999 | Adam second-moment decay b2; other optimizers ignore it. |
