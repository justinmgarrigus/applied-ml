# Saved-result audit for the commuting-constraint slide revision

No training, ED, or new wavefunction evaluation was performed. The extraction script reads saved scalar evaluations, checks checkpoint/state hashes, compares configurations, and verifies parameter equality at continuation joins.

## Reproducible source

Run from the course project root:

```bash
.venv-fqh-gpu/bin/python Final_proj/documents/presentations/spring_fqhe_20261112/support/extract_constraint_comparison_20261116.py
```

Output: `support/constraint_comparison_20261116.json`. The script passes with 269 hash-verified input artifacts and 81 distinct saved B128 records across absolute iterations 18,000–54,000. The JSON contains complete checkpoint/state provenance and hashes, exact endpoint and curve records, configurations, and clock fields.

## Common physical problem and neural architecture

All recommended comparison rows use seven electrons, flux 18, 19 strictly LLL orbitals, fixed Lz=0, the 1,656-dimensional antisymmetric Slater sector, zero edge potential, and

\[
H/V_1=\sum_{i<j}\left(P_1^{ij}+0.8P_3^{ij}+0.6P_5^{ij}\right).
\]

The independent ED ground energy is 6.2447682602150385 V1; the unique ground state in the fixed-Lz=0 sector has total L=0. Its saved eigensystem SHA is `bba98598e3d529b3fe458dbc5c91dfef30befc2f3fe2ce3ec6a90c486bcbe5fa`.

This mixed-pseudopotential ground state must not be called the exact pure-V1 Laughlin state. The latest NN has 77.8169% fidelity with the mixed-H ED ground state, but only 0.3770% with the original pure-V1 Laughlin reference. Filling/geometry alone does not identify the phase.

The no-coherent-L² controls and latest model share 294,464 real parameters, a six-layer 64-unit shared tanh trunk with features/readouts from all layers, and 42 private two-layer 16-unit tanh angular heads. The original 22,850-parameter single-state six-layer network is a different architecture.

**Naming:** `kappa=0` means no coherent `kappa <L²/hbar²>` penalty. The sector-head auxiliary loss is already present with gamma=gamma_aux=head_metric_weight=1 in both controls and the latest result. It is therefore inaccurate to label this control “energy-only original SPRING.” The current sampled objective includes physical energy, the head auxiliary loss, and the coherent penalty.

## The genuine matched 1,000-attempt comparison

Source checkpoint: absolute 16,000 / policy clock 15,000, SHA `1ad2d254d307f38c95fe5b7de99cf765d8bd1c3cc8f0eda507477c6b59026554`. Both branches start from the same parameter vector, six initial RNG states, probes, anchors, and matched reset memory/history/guards. The saved summary verifies identical candidate-pool and Lplus streams. The common configuration differs only in kappa after removing paths/provenance/arm labels.

Common settings: batch 32, pool 128, H8 connection sampling, eta=0.001, damping=0.1, memory=0.9, parameter radius=0.015, state radius=0.01, CG ceiling=384, tolerance=1e-6, validation every 50 attempts. Both arms reset raw optimizer memory and history to zero/empty at this objective-boundary test; this differs from later full-state continuations.

| Exact physical metric | Shared source | kappa=0 after 1k | kappa=0.1 after 1k |
|---|---:|---:|---:|
| E/V1 | 7.1992914243 | 7.3763606284 | 7.4643632897 |
| E-E0 / V1 | 0.9545231641 | 1.1315923682 | 1.2195950295 |
| Full mixed-H ED fidelity | 0.0206582664 | 0.0158222144 | 0.0309412248 |
| P(L=0) | 0.0245102422 | 0.0177989388 | 0.0420003405 |
| K=<L²>/hbar² | 102.087701890 | 136.019402942 | 73.705462128 |
| Physical variance/V1² | 0.4183613789 | 0.5473753448 | 0.6094522387 |
| Accounted seconds added | 0 | 112.062443186 | 110.089536064 |
| GPU-step seconds added | 0 | 84.631542609 | 83.034242636 |

This is evidence of better sector population and full fidelity relative to the control after this short budget. **It does not show improved physical energy:** both endpoints worsen from the source, and kappa=0.1 has higher E than kappa=0. Both sector and coherent-state behavior should be shown. One seed cannot establish a general improvement.

## Same-H historical and latest endpoints

| Exact physical metric | Historical no coherent L², absolute 16k | kappa=0 control, absolute 17k | Latest kappa=0.1, absolute 54k |
|---|---:|---:|---:|
| E/V1 | 7.1992914243 | 7.3763606284 | 6.4042689439 |
| E-E0 / V1 | 0.9545231641 | 1.1315923682 | 0.1595006837 |
| Full mixed-H ED fidelity | 2.06582664% | 1.58222144% | 77.81687909% |
| P(L=0) | 2.45102422% | 1.77989388% | 82.41060995% |
| K | 102.087701890 | 136.019402942 | 1.980676090 |
| Physical variance/V1² | 0.4183613789 | 0.5473753448 | 0.1361709099 |
| Batch / candidate pool | 32 / 128 | 32 / 128 | 128 / 512 |
| CG ceiling | 384 | 384 | 1024 |

This table demonstrates that the combined updated workflow reaches a much better saved physical state for the same Hamiltonian and neural parameterization. It is a historical, unequal-budget comparison, not a causal L²-only ablation or a speedup estimate. Compared with the 16k no-coherent-L² state, the final energy error is 83.29% smaller; compared with the 17k control endpoint it is 85.90% smaller. These ratios must retain the budget qualifier.

## Long B128 trajectory and changes to annotate

The JSON `long_B128_L2.records` joins saved records by `absolute_step` with exact parameter equality at each segment boundary. All use kappa=0.1 and batch 128/pool512.

| Study suffix | Absolute interval |
|---|---:|
| batch128_continue1k | 18,000–19,000 |
| batch128_continue5k | 19,000–24,000 |
| batch128_continue10k | 24,000–34,000 |
| batch128_second10k + recovery1000, retained path | 34,000–44,000 |
| batch128_third10k | 44,000–54,000 |

For the requested main plot, filter to absolute >=19,000 if using the clean 5k+10k+10k+10k continuation sequence. Mark CG384→1024 at absolute43,000. The original process failed at absolute43,219, resumed the last durable checkpoint43,000, discarded/recomputed218 successful unsaved attempts plus one failed candidate, and then continued. The retained path is complete; it is not an uninterrupted process. This history is preserved in the source recovery summary.

The first B128 transition at absolute18,000 changes more than batch size from its preceding branch: kappa returned from1 to0.1, the guard and batch grew, and anchors were fixed at32. Do not attribute that boundary improvement to one component alone. Later steps preserve full optimizer/sampler state except the documented solver-ceiling recovery.

The latest 10k segment alone runs 44,000→54,000 with all settings unchanged, 9,925 accepted and75 rejected attempts, all CG solves converged, 21 frozen states, 62.27-minute whole-process time. Source→final E6.4713074211→6.4042689439, full fidelity70.1833→77.8169%, and error reduction29.59%. This is the cleanest new continuation result, but still not a no-L² control.

## Figure recommendations

1. Main historical endpoint table or dumbbell chart: same physical H, no coherent L² historical checkpoint versus latest updated workflow; label absolute updates, batch/pool, and “unequal budgets.” Plot E-E0 rather than overlaying different Hamiltonian energies.
2. Small matched1k figure: kappa0 and0.1 curves for P(L=0) and full fidelity, with physical-energy panel or numeric endpoint annotation explicitly showing no early energy advantage.
3. Long learning curve: B128 continuation E-E0, full fidelity and P(L=0) versus absolute updates; annotate the CG recovery. The most recent segment can be colored separately.

The saved pure-V1 no-wall adaptive continuation (`ne7_no_wall_adaptive_continue10k_20261115`) has E≈1.633226V1 and full Laughlin fidelity≈4.1296% at11k for seed71, with22,850 parameters. It is useful as qualitative motivation for sector leakage in the pure-V1 problem, but has a different Hamiltonian, architecture and sampling method. It should be separate from the mixed-H numerical comparison, not an unlabeled “without L²” row.
