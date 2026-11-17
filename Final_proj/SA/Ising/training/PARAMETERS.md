# SA Ising parameters

| Flag | Default | Meaning |
|---|---:|---|
| `--n` | 6 | Spin count, 3 through 10 |
| `--hidden` | 12 | Shared RBM hidden units |
| `--J`, `--h` | 1, 1 | Finite coupling and positive field |
| `--steps` | 1000 | Number of candidate weight proposals |
| `--seed` | 7 | Shared neural initialization seed |
| `--search-seed` | 8 | Proposal and acceptance RNG seed |
| `--temperature` | 0.1 | Initial temperature, in the same units as loss |
| `--final-temperature` | 0.0001 | Final positive temperature, <= initial |
| `--proposal-scale` | 0.05 | Initial root-mean-square Euclidean proposal norm |
| `--final-proposal-scale` | 0.005 | Final root-mean-square proposal norm |
| `--output` | fresh run | New output directory; existing paths are refused |

For P searched weights, each coordinate gets independent Gaussian noise with
standard deviation scale/sqrt(P), so E[||noise||^2] = scale^2. This is not a
hard norm cap. There are no weight bounds or gradient-based refinements.
Temperature and scale interpolate geometrically over the requested attempts;
one attempt uses the initial settings. Temperature must remain positive.
Lower-loss/equal-loss candidates always pass; uphill candidates pass with
probability exp(-delta/T), evaluated in log space. Nonfinite candidates are
rejected, and a nonfinite initial loss is an error. Rejected candidates do not
modify current weights. The best visited weights are retained separately.

Only W is perturbed, and c stays at its initialized value. No learning rate,
batch size, damping or memory coefficient applies. Cooling parameters must be
chosen without using target eigenvectors or ground-state labels.
