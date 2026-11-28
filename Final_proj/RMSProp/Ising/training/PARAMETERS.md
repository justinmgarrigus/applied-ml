# VMC with RMSProp: Ising parameters

Flags are those of [VMC/SGD](../../../VMC/Ising/training/PARAMETERS.md) with these differences.
`--damping` and `--mu` are inactive shared-parser fields.

| Flag | Default | Meaning |
|---|---:|---|
| `--learning-rate` | 0.01 | RMSProp step size (optax.rmsprop with decay=0.9, eps=1e-8, no momentum). |
| `--max-norm` | 0.1 | Active norm radius applied after the RMSProp transform. |
| `--optimizer` | rmsprop | Fixed by this entry point. |
| `--rmsprop-decay` | 0.9 | RMSProp squared-gradient decay; other optimizers ignore it. |
