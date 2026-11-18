# Verified results for the SPRING/FQHE slides

All numerical values below were freshly read from the retained result JSON/CSV on 2026-11-12. No training or new diagonalization was run for this deck. Presentation assets live in `support/figures/`; `figure_inputs.json` records the input SHA256 values. The generator is `support/generate_results_figures.py`.

## Suggested result-slide sequence

1. Ising validates the energy/gradient/SPRING implementation on a tractable, positive-amplitude state.
2. ED database and historical near-ground-state N=6 FQHE example establish the benchmark and its actual precision.
3. Matched N=8 depth study shows low fidelity/probability concentration; greater depth alone did not fix this pilot.
4. N=7 GG-SPRING and the wall test show progress, with distinct physical targets clearly separated.
5. Rare configurations dominate the exact residual despite tiny Born mass; a broader candidate pool is a plausible intervention.
6. The matched 8k→10k continuation tests that intervention; the broader pool gives most of the mean benefit, while sequential geometry adds cost.

## Ising: actual retained training result

Files: `Final_proj/SPRING_Ising_test/results/spring_n6/{metrics.json,training.csv,gradient_check.json,validation.json}`. Hamiltonian and architecture: `SPRING_Ising_test/ising_spring.py`, functions `dense_hamiltonian`, `log_psi`, `local_energies`, `make_spring`, and `run`.

\[
H=-J\sum_{i=0}^{N-1}\sigma_i^z\sigma_{i+1}^z-h\sum_i\sigma_i^x,
\qquad J=h=1,\quad N=6,\quad\text{periodic},\quad \sigma^z=\pm1.
\]

Input: spin vector `s` of shape `(6,)`, each entry ±1; positive spin-flip-even RBM amplitude
\[
\log\psi_\theta(s)=c+\sum_{a=1}^{12}\log\cosh\left(\sum_i s_iW_{ia}\right).
\]
73 real parameters: 6×12 weights plus normalization gauge c. This small benchmark **enumerates all 64 configurations and draws IID categorical Born samples** for the stochastic update; it does not run MCMC. Batch128, 250updates, seed7, learning rate0.05, damping0.001, memory0.9, radius0.1 (paper C=0.01). JaQMC upstream commit `71f94158f2aca88ded904451f5314f71cdff5666`; actual `jaqmc.optimizer.sr.jasr.robust_sr`, not a renamed ordinary gradient optimizer.

| Quantity | Initial | Final |
|---|---:|---:|
| Exact energy | -6.1920178345 | -7.72739085205 |
| Exact energy error | 1.5353887758 | 1.5758265848e-5 |
| ED fidelity | .5174663206 | .999998569116 |
| Exact residual variance | 4.2712065235 | 2.0536374884e-4 |

ED E0=-7.727406610312546. Covariance-gradient versus autodiff max error1.9428902931e-15; versus central finite differences2.8707036748e-10. Saved validation:10tests passed and all numerical acceptance criteria passed. The saved CPU timer is0.47784s including JIT, **not a speed comparison with FQHE**.

Assets: `ising_exact_energy_single.pdf` is the single-panel slide version; `ising_exact_convergence.pdf` has energy error and infidelity side by side.

## Present ED database

Authoritative aggregate: `Final_proj/benchmarks/summary.json`; separate inputs `SPRING_Haldane_FQHE/results/ed_benchmarks/summary.json` and `SPRING_Ising_test/benchmarks/ed/summary.json`.

FQHE pure V1, sphere LLL, Nphi=3(Ne−1), Lz=0:

| Ne | Orbitals | Lz=0 dimension | Full fixed-N dimension |
|---:|---:|---:|---:|
|4|10|18|210|
|5|13|73|1,287|
|6|16|338|8,008|
|7|19|1,656|50,388|
|8|22|8,512|319,770|

Ground E0=0 within ~1e-16; unique zero mode in these sectors. Retained low eigenpairs:18 forNe4,20each forNe5–8; maximum saved eigenpair residual ≤1.07e-14V1. The largest fixed sector in this database (N8) remains ED tractable; do not call it beyond ED.

Ising ED cases: N4/6/8/10 atJ=h=1, plusN6 h=.5 and2, full dimensions16/64/256/1024 (additionalh cases64each). These provide the complete spectrum resolved into exact global spin-flip ± blocks. The **dataset consists of configuration-space bases and Hamiltonian/eigenstate benchmarks**, not supervised training labels for VMC. Count dimensions perHamiltonian rather than conflating all NN samples with independent examples.

## N=6 FQHE: historical selected near-ground state, not exact

`SPRING_Haldane_FQHE/results/ne6_baseline/{README.md,exact_evaluation.json,validation.json}`. CheckpointSHA256 `73a80dff758a0339fcc9f8cb41ef679977098f8b65d231510fc38b74074073f6`.

- Ne6, flux15,16LLLorbitals, Lz0dimension338, pure normalizedV1, no wall, no L² penalty, no analytic Laughlin factor, no ED training labels.
- One tanh hidden layer width64,1,218 realparameters; complex occupation-coefficient NN.
- E/V1=.03707402133193935; F=.9816744300748502; variance/V1²=.10599273973225239; L²=2.18318141724. Ground E0 numerically0.
- Batch128, eta.03, lambda.001, mu.9, radius.1; historical broader sampling q∝|psi| (auxiliary T2/beta.5), importance corrected in force and metric. Two sweeps/update, global uniform-sector refresh25.
- 2k-update pilot followed by a selected4k-update continuation; fresh optimizer memory for that historical continuation. Selected best from nine arms; second stochastic continuation E=.0728453,F=.962659.
- 39retainedvalidationtests passed; fresh and retained metrics agree within~1e-15. This is **not** an exact NN state or an unselected statistical mean.
- It is not a matched scaling comparison with the N8 study: initialization/protocol/sampling, architecture, and budget differ.

## N=8 matched depth pilot: accuracy and probability collapse

`SPRING_Haldane_FQHE/results/ne8_depth_ablation_20261103/{summary.json,validation.json,README.md}`. Ne8, flux21,22orbitals, Lz0dimension8512, pureV1, no wall. Width64, depths1/2/6;5kupdates, batch32, ordinary Born sampling. Two paired initialization/sampling seeds; equal initial complex log-wavefunctions across depths over all8512 configurations (maxdifference0); samewidth is not sameparametercount orcost.

| Layers | Parameters | Mean E/V1 | Mean F | Mean active s |
|---:|---:|---:|---:|---:|
|1|1,602|1.56252083036|.0354395487|15.56454|
|2|5,890|1.80107238882|.0226266779|27.60188|
|6|23,042|2.13634407384|.00982042185|114.03268|

ED E0≈1.58e-16. Learned effective support 1/sum(p²)=1.34–4.57, versus ED641.2955; logged acceptance~.7–2.7%. This is strong concentration evidence, **not proof the sampler targets the wrong distribution or that shallow networks suffice**. Saved validation:6/6completed,36full-sectorevaluations, all noEDtraining and unchangedinputs; sourcechecksums passed. Bars/dots in figures are descriptive means and the two seed values, not confidence intervals.

Assets: `fqhe_n8_depth_energy_fidelity.pdf` (2panels) and `fqhe_n8_depth_and_support.pdf` (3panels).

## N=7 GG-SPRING and wall results: separate physical targets

No-wall matched test: `results/ne7_gg_spring_20261111/summary.json`. Ne7/flux18/Lz0,19orbitals,D1656;6×64additivereadouts,P22850;5kupdates, updateB32. Same init7, three sampling restarts71/137/211 (not independent NN initializations). Pool128 controls random selection versus GG novelty; importance corrected. Means:

| Method | E/V1=energyerror | F | Active s |
|---|---:|---:|---:|
|SPRING batch32|1.24596887523|.0778395110|39.70459|
|Pool128, random32|1.12186232918|.0954128127|41.01279|
|GG pool128→32|.821566356999|.1639527598|70.62043|

Saved validation passed9/9arms and all54checkpoints. GG improves this early budget but remains far from the no-wall Laughlin groundstate. `fqhe_n7_geometry_no_wall.pdf` shows exact energy/fidelity curves.

Wall test: `results/ne7_gg_edge_20261111/summary.json`. GGonly8kupdates, matched init/seeds, samebasis/network, U0 versus projectedU10wall atx=.88. **Hamiltonian and exact target change**. No-wall meanerror.77749047,F.18719061. Wall meanE2.97732365, E0=2.82308466308447, error.1542389882,F.9085121345,variance.3800226481. Wall-ground overlap with no-wall Laughlin is only.0274625679; do not present the higherwallfidelity as a more accurate solver of the originalLaughlinproblem. EDwall effective support~3.54; originalno-wallED support~218.86. Validationpassed6/6arms,54exactevaluations.

## Accuracy limitation: rare tails are nearly absent from a narrow Born pool

`results/ne7_gg_explore_20261111/frozen_probe.json`, three frozen wall-trained8kstates. Define rareconfig individually by p(n)<1e-6. RaretotalBornmass is only1.92e-5/8.32e-6/2.35e-5, but carries94.10%/98.08%/89.91% of exact residualvariance. This is a concrete variance-tail diagnostic; lowprobability alone is not a general importance criterion.

128 IID Born candidates hit ≥1rarestate with probabilities.245%/.106%/.300%. For q_beta∝|psi|^(2beta), beta.5, probabilities rise to79.53%/50.14%/78.09%. These are ideal IID proposal coverage calculations, not persistent-MCMC convergence bounds. `fqhe_n7_rare_tail_coverage.pdf` illustrates these exact frozen diagnostics.

Frozen32-repeat force accuracy is mixed and normalizedweightESS drops (means~14original,4.8broaderstatic,6.3broadersequential). Rarecoverage alone does not prove a better finite-batch force. Cold diagnostic often entirely misses the tail: probabilityall32Bornpools miss is91–97%. Bias from self-normalized ratios/covariance means and MCMCautocorrelation remain.

## Broader-pool continuation: the actual solution-candidate test

`results/ne7_gg_explore_20261111/{summary.json,validation.json,README.md}`. Start from each saved8kU10wallcheckpoint, restore all NN/SPRINGmemory/counters/walkers/anchors/RNGstates; continue2kupdates. Allthreebrancheswithinseedsharethefullparent. Parentone-stepparityagainstoriginalcode passedbitexact for allthree seeds (counters8000→8001). Six64layers/allreadouts, eta.01,lambda.01,mu.9,radius.03, B32,pool128 remainfixed. Hotpoolswarm500sweepsatfrozenparams. TargetH/loss unchanged; beta auxiliary notphysicaltemperature.

Selection:20%uniform+80%geometricnovelty,floor.01,withreplacement/fullsupport. Staticbroaderpoolbeta.5;sequentialvariantupdatesresidualscorespan aftereachdraw. Weights proportional to |psi|^[2(1-beta)]/pi_t; selfnormalizedfinitebatchbiasremains. Geometry is diagonal-whitened/ridge-smoothed, not full-QGT invariant principalangles.

| Method | Mean error/V1 | Mean F | Mean var/V1² | Active additional s |
|---|---:|---:|---:|---:|
|Continued original GG|.151621707923|.912769015170|.377143939224|28.58781|
|Broader static pool|.0983630810768|.948448057384|.251466660893|29.48907|
|Broader +sequential|.0962747827377|.952182929324|.244293587629|57.09941|

- Broaderstatic:35.126% lowermeanerror vscontinuedoriginal,3.15%extraactivecost; lowererror/higherF in3/3matchedseeds.
- Sequentialvsbroaderstatic:meanerror2.12%lower,F+.3735percentagepoints,1.94×activecost; lowerenergy inonly1/3seeds. Extraaccuracyisseeddependent.
- Bestsequentialseed211:error.0164077451,F=.9876115775,var.05882793295. Hardseed137 remainsF~.892; no groundstateconvergenceclaim.
- Loggedmeanacceptanceorig2.22%,broader7.88%,sequential7.87%;selecteddistinctconfigs10.26→20.32→26.12. ImportanceweightESS13.92→5.75→6.49, not autocorrelation-adjustedESS. Bettercoverage costsweightconcentration.
- ValidationPASSED9/9CUDAarms,all45exactevaluations,0errors; independentcoefficientenergy/variance/fidelity/decomposition and checkpoint/source/parenthashchecks.
- Active timeexcludescompile,frozenwarmup,EDdiagnostics; localGPUtimings are not universal hardware scalingclaims. Threepairedlearnedstates are a smallpilot, not a significance test.

Assets: `fqhe_n7_broader_pool_updates.pdf` (energy/F versus updates; rootrequestedversion); `fqhe_n7_broader_pool_continuation.pdf` (energyvsupdates/Fvstime). Thinlinesindividualseeds;thicklinesmeans. A smallresultstablecanprovidetimecostbesideupdatesplot.
