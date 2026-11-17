# Ne12 pure V1: user-stopped partial continuation, safely checkpointed

Stopped at the user's request. The requested target was 17,000; actual global 10,000→13,721 retained work contains 3,721 of 7,000 requested optimizer attempts. The remaining 3,279 were not run. The final process saved its complete state and exited 0 after SIGUSR1. Physical E/V1 changes 3.8619223 ± 0.02654 → 3.6344834 ± 0.0266; endpoint−source = -0.22743891 ± 0.03758 V1. Analytical interaction E0=0. No fidelity or ED overlap is claimed.

## Failures, durable restarts and actual work

Three failed candidates (11215, 13005, 13162) were not applied. Durable states 11192, 13000 and 13157 define the retained boundaries. Discarded/replayed completed ranges are 11193–11214 (22), 13001–13004 (4) and 13158–13161 (4). These failed namespaces and all completed CSV rows remain preserved.

| Solver leg | Retained attempted range | Attempts | Accepted | Rejected |
|---|---|---:|---:|---:|
| CG1024 retained prefix | 10,001–11,192 | 1,192 | 1,169 | 23 |
| CG4096 retained prefix | 11,193–13,000 | 1,808 | 1,775 | 33 |
| Hybrid CG/PCG retained prefix | 13,001–13,157 | 157 | 154 | 3 |
| Original plain CG65536 | 13,158–13,721 | 564 | 555 | 9 |
| Canonical total | 10,001–13,721 | 3,721 | 3,653 | 68 |

Actual main work contains 3,751 completed optimizer attempts plus 3 failed candidates (3,754 recorded main candidates). Completed attempts include held-out rejections; they are not all accepted parameter updates. The 30 discarded/replayed completed attempts and all auxiliary solver-validation work are excluded from the actual retained partial path. The requested 7,000 attempts were not completed.

Each restart restores all scientific state exactly: NN parameters, nonzero direction memory, six RNGs, histories, probes, caches and the saved 256-root guard. This initial equality does not imply bitwise GPU replay. Natural replay differences and failed strict comparisons remain recorded; no favorable metric replacement is claimed.

## Same physical loss and SPRING linear equation

\[J=E_H+A_{\rm heads}+0.001\langle L^2/\hbar^2\rangle,\qquad (G_{\rm aug}+\lambda I)\phi=\tfrac12\nabla\widehat J+\lambda\mu\phi_{\rm prev},\quad \lambda=0.1,\ \mu=0.9.\]

The final leg uses the literal original plain-CG SPRING step, inherited without copying its implementation, with only its iteration ceiling raised to 65,536. It restores the full durable state at 13,157 without resetting memory or renewing the guard. Neither PCG nor GMRES is active in this final leg. The original matrix, RHS, warm start, damping, relative tolerance and caps stay fixed. The ordinary cg_iterations column records actual CG iterations.

Actual final-leg CG iterations min/median/max = 78/587.0/9,714, versus the configured ceiling 65,536. Maximum saved original-equation residual/tolerance = 0.999791. These are measured CG iteration counts, not budget estimates.

Acceptance requires the independently computed residual of the original equation to meet the unchanged test: ||Aφ−b|| ≤1.01 max(10⁻⁶||b||,10⁻¹²). A library success/info code alone is not convergence. Fixed-batch checks establish numerical adequacy for those systems, not an overall speedup or a general convergence guarantee.

The attempted GMRES proposal FAILED its overall validation and was never activated for scientific training. Its original failed receipt and 185.233s validation cost are preserved. On that SAME frozen system, the separate plain-CG reference passed in 8,615 iterations, with true residual 0.000329069 below original tolerance 0.000350517. The root activation gate approves ONLY that residual-checked plain-CG reference component; it never relabels the overall GMRES proposal as passed. A single fixed-system pass does not prove that every future batch will meet its budget.

The earlier PCG validation required 4,096 ordinary-CG iterations plus 1,088 additional PCG iterations and 32 probes, versus a 5,077-iteration plain-CG reference; 1,088 alone is not an end-to-end work comparison. All diagnostic reconstruction and solver comparisons remain separate from formal training.

## Frozen physical measurements

| Global attempted update | Physical E/V1 | Residual variance/V1² | Positive K=L²/hbar² |
|---|---:|---:|---:|---:|
| 10,000 | 3.8619223 ± 0.02654 | 1.6389739 ± 0.1286 | 370.34196 ± 5.769 |
| 13,721 | 3.6344834 ± 0.0266 | 2.1301881 ± 0.3602 | 348.98054 ± 6.846 |

K change = -21.361422 ± 8.952; variance change = +0.49121422 ± 0.3824 V1². The variance change is unresolved within two combined MCSE.

Observed endpoint max|Eloc| = 28.786 V1; one sample supplies 14.501% of the variance sum, and the largest chain variance mean is 10.9778 V1². All samples are retained without clipping or trimming. Unseen tails and incomplete equilibration are outside the MCSE guarantee.

Mean R=1 pair-channel probability g1=E/(66V1): 0.058513974 ± 0.0004022 → 0.05506793 ± 0.0004031; ground value 0. Final energy/electron = 0.30287361 ± 0.002217 V1. Relative error divided by E0 is undefined.

Physics is unchanged: Ne12, flux 33, M=0, strict LLL, H/V1=ΣP1, V3=V5=0, no wall, and no constant kinetic/background offset. Cubic Laughlin pair zeros and projector positivity establish interaction E0=0; K also annihilates Laughlin. No Laughlin factor or ED vector is embedded in this network.

The network has six shared 64-wide tanh layers with all-layer readouts and 132 private heads, each with two 16-wide hidden layers (877,454 real parameters). Kappa=0.001, gamma_aux=1, B256/pool1024/Lplus256/guard256 and 32 geometry anchors stay fixed. Coherent K is unscaled; only the head residual uses scale 3465. Nominal eta=0.001, damping, memory and parameter/state caps 0.015/0.01 remain fixed. The 512-row rematerialized components use a global amplitude shift without truncating the NN input union.

Only final step 13,721 MC seed 169071 is newly sampled; source step 10,000 seed 159071 is reused unchanged. Each endpoint uses 32 independent Born-MH chains × 64 packs = 2,048 roots, burn 500 sweeps, spacing 10 sweeps and 12 proposals per sweep per chain. The target is |ψ|² in the occupation basis, with no physical/artificial temperature. Full connected H/K actions precede residual squaring, and positive RB K is checked against signed Kloc.

Errors are one independent-chain-cluster MCSE, conditional on the fixed protocol. No Ne12 ED, complete angular projector, fidelity or P(L=0) measurement is available. Local residual variance equals quantum variance only when exact-node leakage is absent. Finite positive energy is not ground-state convergence, and it does not prove that later convergence is impossible. The head auxiliary may stay positive even at zero physical energy.

## Sampling, applied steps and closed process costs

| Global attempted update | Pair/global changed acceptance | Max basic split Rhat | Absolute ImE/MCSE | Absolute RB−signedK/paired MCSE |
|---|---:|---:|---:|---:|
| 10,000 | 0.12709/0.040231 | 1.00069 | 0.9358 | 0.6554 |
| 13,721 | 0.078498/0.0080795 | 1.00362 | 1.175 | 0.8741 |

Accepted applied eta min/median/max = 3.7294e-06/0.00010168/0.00031785, versus nominal 0.001. Accepted median parameter-step norm = 0.015; median training root importance ESS = 28.677 out of 256.

| Main process | Whole-process seconds |
|---|---:|
| CG1024 retained prefix | 2990.834 |
| CG4096 retained prefix | 5841.201 |
| Hybrid CG/PCG retained prefix | 772.809 |
| Original plain CG65536 | 2780.428 |
| All main processes, including failed/discarded/replayed work | 12385.272 |

| Auxiliary validation group (not formal epochs) | Proposal status | Approved component | Inclusive process seconds | Final reconstruction rows |
|---|---|---|---:|---:|
| Earlier PCG fixed-batch validation | passed | whole_proposal | 92.086 | 4 |
| Rejected GMRES proposal / passed plain-CG reference component | failed | plain_CG_reference_only | 185.233 | 4 |

Main+validation process cost = 12662.590s (3.517h). Inclusive validation costs retain failed launch/preparation work when present. All closed main processes include compilation, failed solves and discarded/replayed attempts; training before 10,000 and final MC are excluded. Validation replay/benchmarks are separately recorded and never counted as formal epochs.

## Historical context and artifacts

Frozen physical MC history is 0/1,000/2,000/3,000/10,000/13,721. Global 0→1,000 used kappa=0.1/B32; 1,000→2,000 jointly changed kappa/batch with direction reset and guard renewal. Physics/model/loss/sampler settings have been fixed since then. The current canonical path uses 10,001–11,192 from CG1024, 11,193–13,000 from CG4096, 13,001–13,157 from hybrid CG/PCG and 13,158–13,721 from the original plain-CG iteration with a 65,536 ceiling. The user stopped the run before the requested 17,000. Only numerical solvers change at these three recovery boundaries.

The restart boundaries mark numerical solver changes, not extra physical MC measurements. Lines connect frozen endpoints and do not claim an exact intermediate trajectory. Cumulative configuration visits are not simultaneous exact-sector quadrature or an ED basis.

Negative values in the bottom training curve are finite-batch weighted H8 row estimates. Positive-semidefinite H does not impose a per-batch variational bound on these noisy estimates. The frozen endpoint measurements instead use the full connected H action on independent Born samples; finite-MC uncertainty remains.

- All three failed namespaces are preserved and revalidated by hashes; this report writes only into recovery_cg65536. The rejected GMRES proposal and its failed validation receipt are preserved.
- `user_stopped_evaluation_comparison.json` contains all leg counts, discarded ranges, source-state restoration, validation, residual/budget metrics, source contrasts and closed costs.
- `pure_v1_user_stopped.png/.pdf` separates noisy retained training estimates from frozen physical MC estimates.
- Reporting performs no NN inference, GPU computation, MC sampling, ED, full Ne12 basis construction or parameter update.

## Dormant restart documentation

Nothing is launched by this report. If the user later authorizes finishing the original target, the saved full state can be resumed with the same frozen driver in a new output directory. The following is documentation only; the remaining budget is 3,279 attempts.

```bash
export SPRING_PYTHON=/home/leean/research/Applied_machine_learning_course/.venv-fqh-gpu/bin/python
export JAX_COMPILATION_CACHE_DIR=/tmp/ne12_explicitly_authorized_future_resume_cache
"$SPRING_PYTHON" -u experiments/train_ne12_pure_v1_batch256_cg65536.py \
  --resume /home/leean/research/Applied_machine_learning_course/Final_proj/SPRING_Haldane_FQHE/results/ne12_pure_v1_kappa001_batch256_continue7k_round2_20261117/recovery_cg65536/train_seed71/checkpoints/step_000013721.npz \
  --output /home/leean/research/Applied_machine_learning_course/Final_proj/SPRING_Haldane_FQHE/results/ne12_pure_v1_kappa001_batch256_continue7k_round2_20261117/recovery_cg65536/NOT_LAUNCHED_future_resume \
  --steps 3279 --save-every 1000 --checkpoint-seconds 60 --log-every 250 \
  --active-seconds 86400 --max-wall-seconds 90000 --require-gpu
```
