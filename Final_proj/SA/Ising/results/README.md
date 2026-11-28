# Ising simulated annealing results

CPU runs used the 6-spin transverse-field Ising ring (J = h = 1), a 12-unit
RBM, exact full-basis loss, initialization seed 7 and search seed 8. Each run
started at −6.192018 J and cooled geometrically from 0.1 to 0.0001, with proposal
scale decreasing from 0.05 to 0.005. The exact ground energy is −7.7274066103 J.

| Proposals | Best energy (J) | Final energy (J) | Best error (J) | Best fidelity | Search time (s) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | −7.509305 | −7.509305 | 0.218101 | 95.2803% | 0.056 |
| 10,000 | **−7.722031** | −7.721422 | **0.005375** | **99.9369%** | 0.482 |
| 1,000,000 | −7.512914 | −6.912098 | 0.214492 | 93.5259% | 52.633 |

The 10,000-proposal run gave the best result. Each budget stretches cooling
over its own run. Results use one seed pair and search times exclude compilation
and evaluation. Comparison with sampled SPRING requires a matched protocol.

Local runs under `runs/`, in table order, are `n6_d6e9c2f1190a`,
`n6_385880e46d20` and `n6_c7902e2e21cc`. Full trajectories and checkpoint hashes
are in those directories. The [100-proposal smoke check](check_cpu_n6/metrics.json)
is included in Git.

A tuned 10,000-proposal comparison against SPRING, SGD, Adam and RMSProp, with the random-search record for the schedule, is in [benchmarks/ising/optimizer_comparison](../../../benchmarks/ising/optimizer_comparison/README.md).
