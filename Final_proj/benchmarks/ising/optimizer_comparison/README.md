# Ising optimizer comparison: SPRING, VMC-SGD, Adam, RMSProp, SA

Small matched experiment: periodic transverse-field Ising ring, `N=6`, `J=h=1`,
RBM with 12 hidden units. ED energy is used for evaluation only. Gradient methods
use 250 updates with 128 IID Born draws each. SA uses exact full-basis energy and
10,000 proposals (the best budget in the [SA results](../../../SA/Ising/results/README.md)),
so its work per run is not matched to the sampled methods.

## Protocol (every tuning trial is recorded)

Hyperparameters are chosen on tuning seeds 101-103 by lowest mean final error;
the chosen configuration is then run on separate seeds 7-11.

- **Stage 1** (`run_comparison.py`): learning rate only, update-norm cap fixed at 0.1.
  [stage1_tune_results.json](stage1_tune_results.json),
  [stage1_final_results.json](stage1_final_results.json), `stage1_final/`.
- **Stage 2** (`tune_and_compare.py`): wider search.
  [stage2_tune_results.json](stage2_tune_results.json) (588 trials, all configs and
  seeds), [stage2_selected.json](stage2_selected.json),
  [stage2_final_results.json](stage2_final_results.json), `stage2_final/` (per-run metrics).

| Method | Searched (tuning seeds only) | Selected |
|---|---|---|
| VMC-SGD | lr {0.03,0.1,0.3,1} x cap {0.03,0.1,0.3,none} | lr 0.1, cap 0.3 |
| Adam | lr {0.003,0.01,0.03,0.1} x cap x b2 {0.9,0.99,0.999} | lr 0.1, cap none, b2 0.999 |
| RMSProp | lr {0.001,0.003,0.01,0.03} x cap x decay {0.9,0.99,0.999} | lr 0.003, cap none, decay 0.999 |
| SPRING | lr {0.03,0.1,0.3} x cap x damping {1e-4,1e-3,1e-2}; mu fixed 0.9 | lr 0.03, cap 0.3, damping 1e-4 |
| SA | 48 random log-uniform draws of T0, Tf, scale0, scale_f | T0 2.4e-3, Tf 7.3e-6, scale 0.099 to 0.0127 |

"cap none" means `--max-norm 1000`. Adam b1, eps, RMSProp eps and momentum stay at optax defaults.

## Stage 2 results (seeds 7-11, final E-E0, J units)

| Method | Mean | Median | Min - max | Fidelity (mean) |
|---|---:|---:|---|---:|
| SPRING | 1.7e-07 | 1.4e-10 | 1.8e-12 - 8.3e-07 | 1.000000 |
| SA (10,000 proposals, best visited) | 4.5e-04 | 5.0e-04 | 3.4e-04 - 5.2e-04 | 0.999948 |
| VMC-SGD | 6.7e-04 | 6.2e-04 | 3.5e-04 - 1.1e-03 | 0.999864 |
| RMSProp | 1.1e-03 | 9.7e-04 | 7.0e-04 - 1.5e-03 | 0.999674 |
| Adam | 2.5e-01 | 1.0e-03 | 2.7e-04 - 1.3 | 0.923101 |

**Adam caveat.** The selected Adam config (lr 0.1, no cap) won the tuning seeds
(mean 2.8e-04, worst 3.4e-04) but diverged on evaluation seed 7 (error 1.3); the
other four seeds give 2.7e-04 to 3.0e-03. The mean is dominated by that one run, so
the median is the fairer summary. Choosing from three seeds rewarded an unstable
setting; the capped lr 0.03 config (stage 1) was robust. The result was not re-selected
after seeing the evaluation seeds.

Stage 1 (lr only, cap 0.1) gave SPRING 3.7e-06, Adam 5.0e-04, SGD 7.1e-04, RMSProp 1.5e-03.

## Limits

Five evaluation seeds and three tuning seeds, one system size, one batch size, a
coarse grid, fixed 250 steps and 10,000 SA proposals. Final energies are not converged.
SA uses exact energies while the others use sampled gradients; wall times (about 1 s) are
too small to compare. SGD, SA and Adam (excluding the diverged run) overlap in
range, so no ordering among them is established; the gap to SPRING is large relative to
seed spread. SPRING damping and other first-order parameters (b1, eps) were
not exhaustively tuned.
