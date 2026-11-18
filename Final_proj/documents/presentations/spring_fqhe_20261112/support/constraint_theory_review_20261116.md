# Symmetry-augmented SPRING: equations and claim audit

Reviewed 16 November 2026. This note supports the slide update only. No training,
ED, simulation-source change, or main-deck edit was performed by this reviewer.

## 1. Conventions that must be explicit

Use dimensionless physical energy `Hcal = H/V1` and `K = L²/hbar²` for the
present seven-electron, flux-18, fixed-`Lz=0`, strictly-LLL calculation. The
physical Hamiltonian is

\[
\mathcal H=\sum_{i<j}\big(P_1^{ij}+0.8P_3^{ij}+0.6P_5^{ij}\big),
\qquad [\mathcal H,K]=0.
\]

The rotationally invariant pair projectors commute with total angular momentum.
The unmodified physical ED target for this Hamiltonian is the unique `L=0`
ground state with `E0/V1 = 6.2447682602150385`. It is **not** the zero-energy
ground state of the earlier pure-`V1` parent Hamiltonian. Do not compare raw
energies across those two operators or reuse pure-`V1` fidelity as a control.

Let `κ` denote the symmetry-objective coefficient and `δ` denote SR damping.
The current values are `κ=0.1`, `δ=0.1`, `μ=0.9`, and nominal `η=0.001`.
Their numerical equality does not make them the same parameter. Reserve
`C_cap` for the original paper's squared parameter-step bound; a constraint
operator `C` is unrelated to that cap.

For a general physical `H + λ C`, `λ C` must have units of energy. In this
run the physical generator corresponding to the coherent part is
`H + κ V1 K`. Thus `τ_K/τ_H = κ` refers to the two **dimensionless** operators
`Hcal` and `K`, not two independent exact propagations in unmatched units.

## 2. Ideal single-state imaginary-time interpretation

For a normalized state `|ψθ>` and dimensionless Hermitian operators `Hcal,C`,
write `E=<Hcal>` and `c=<C>`. The desired normalized evolution is

\[
|\psi'\rangle=
\frac{e^{-\tau_H\mathcal H-\tau_C C}|\psi\rangle}
{\|e^{-\tau_H\mathcal H-\tau_C C}|\psi\rangle\|},
\qquad \kappa=\tau_C/\tau_H.
\]

Its projective first-order displacement is

\[
\delta|\psi\rangle=
-\tau_H\big[(\mathcal H-E)+\kappa(C-c)\big]|\psi\rangle+O(\tau_H^2).
\]

This derivation does not require commutation to first order. Commutation makes
the finite exponential factorization exact:
`exp[-τH Hcal - τC C] = exp[-τH Hcal] exp[-τC C]`.

For real NN parameters and complex coefficients, define

\[
O_j(n)=\partial_j\log\psi_\theta(n),\quad
\widetilde O_j=O_j-\langle O_j\rangle_p,\quad
E_{\rm loc}=\frac{\mathcal H\psi}{\psi},\quad
C_{\rm loc}=\frac{C\psi}{\psi},
\qquad p(n)=\frac{|\psi(n)|^2}{\|\psi\|^2}.
\]

The centered score removes the norm/global-phase direction. An occupation
array `n` is generally **not** a total-`L` eigenvector: `C_loc(n)` depends on
the entire current wavefunction and may be complex. It is not a label assigned
to an individual configuration.

For weighted quadrature `q_i` approximating `p`, define the complex rows

\[
A_{ij}=\sqrt{q_i}\widetilde O_j(n_i),\quad
(e_H)_i=\sqrt{q_i}(E_{{\rm loc},i}-E),\quad
(e_C)_i=\sqrt{q_i}(C_{{\rm loc},i}-c).
\]

Stack real and imaginary parts: `AR=[Re A; Im A]` and
`eR=[Re(eH+κeC); Im(eH+κeC)]`. Then

\[
S=A_R^T A_R,\qquad
f=A_R^Te_R=\tfrac12\nabla_\theta(E+\kappa c).
\]

With the code's **positive-force** direction `d` and negative parameter step,

\[
\boxed{d_k=\mu d_{k-1}+A_{R,k}^T
(A_{R,k}A_{R,k}^T+\delta I)^{-1}
\left[e_{R,k}-A_{R,k}\mu d_{k-1}\right]},
\qquad \Delta\theta_k=-\alpha_k d_k.
\]

Equivalently, `(S+δI)d = f+δ μ dprev`. In the original paper's descent
direction convention, set `φ=-d`, `ε=-(eH+κeC)` and use

\[
\boxed{\phi_k=\mu\phi_{k-1}+A_{R,k}^T
(A_{R,k}A_{R,k}^T+\delta I)^{-1}
\left[\epsilon_{R,k}-A_{R,k}\mu\phi_{k-1}\right]},
\quad \Delta\theta_k=+\alpha_k\phi_k.
\]

Here targets use unit time, and the learning rate is applied **once**, after
the direction is computed, as in the original SPRING Algorithm 1. Do not put
`τH` inside `ε` and multiply by `η=τH` again.

The original Euclidean cap is `α=min(η,sqrt(C_cap)/||φ||)`. Our current code
also enforces a projective-state bound and sampled/held-out backtracking:

\[
\alpha_k\leq\min\left(\eta,
\frac{r_\theta}{\|d_k\|},
\frac{r_\psi}{\sqrt{d_k^T G_kd_k}}\right),\qquad
r_\theta=0.015,\quad r_\psi=0.01.
\]

Therefore `τK/τH=κ` is an effective force-weight ratio. Caps, retained memory,
finite batches and ansatz projection mean a training step is not an exact
finite imaginary-time propagation, and no ground-state convergence guarantee
follows from the classical fixed-system Kaczmarz theorem.

## 3. What the saved training actually implements

The latest ansatz has 42 complex angular heads, private two-layer heads on
six shared width-64 tanh layers, and coherent mixing `ψ=Σs ws ψs`. Heads
are **learned approximate angular functions**, not exact angular projectors.
Their auxiliary residual targets are `ls=Ls(Ls+1)`.

The differentiable sampled objective is

\[
\boxed{\widehat J(\theta)=\widehat E_H(\theta)
+\gamma\widehat{\mathcal A}_{\rm heads}(\theta)
+\kappa\widehat{\langle K\rangle}(\theta)},\qquad
\gamma=1,\quad \kappa=0.1,
\]

with the ideal head-loss form

\[
\mathcal A_{\rm heads}=
\frac1{42}\sum_s
\frac{\|(K-l_s)\psi_s\|^2}{630^2\|\psi_s\|^2}.
\]

At `Lz=0`, `K=L_-L_+` (ladder operators in units of `hbar`), so

\[
\langle K\rangle=\frac{\|L_+\psi\|^2}{\|\psi\|^2}\geq0.
\]

The code samples 128 independent `Lz=1` output roots, sums **every** incoming
`Lz=0` parent coherently before squaring, and normalizes with the sampled
root norm. It uses the kinematic factor `(D1/D0)(B0/Bplus)`. This avoids
dividing the angular action by a possibly tiny individual coefficient.
It is a positive finite-batch ratio, not an unbiased exact expectation.

The unchanged current metric is the weighted sum of the coherent-state
projective metric and the mean of the 42 head projective metrics:

\[
G_{\rm aug}=G_\psi+\frac1{42}\sum_sG_{\psi_s}.
\]

The actual primal, matrix-free equation is therefore

\[
\boxed{(\widehat G_{\rm aug}+\delta I)d_k=
\tfrac12\nabla_\theta\widehat J+\delta\mu d_{k-1}},
\qquad \theta_{k+1}=\theta_k-\alpha_kd_k.
\]

This is the most accurate main-slide equation for the saved results.
Automatic differentiation includes all connected amplitudes, complex mixing,
and sampled normalization. Since the finite-batch physical Rayleigh quotient
uses sampled neighbor connections, its exact autodiff force need not equal
a naively assembled centered-score formula. The ideal `eH+κeC` view explains
the physics; the boxed primal equation defines the actual implementation.

The all-sector auxiliary loss and coherent `κ<K>` have different roles:
the first trains an angularly organized representation across many sectors;
the second **explicitly favors L=0**. No hard projection is imposed. Do not
say the latest run discovered `L=0` without any supplied sector preference.

The proposed Gauss–Newton angular-residual metric in the separate
`configuration_constraint_spring_derivation.md` was **not** used in these
latest runs. Do not attribute their improvement to added `BᵀB` curvature.

Code anchors:

- `experiments/l0_positive_penalty_runtime.py:80-86`: positive coherent `K` ratio.
- Same file `:97-130`: sampled `E`, head auxiliary term, coherent `κK`, full gradient.
- `experiments/residual_geometry_runtime.py:147-169`: augmented projective metric.
- Same file `:192-213`: memory-centered damped solve and two caps.
- Same file `:217-236`: guards, backtracking, rejection and raw-direction memory.
- `experiments/continue_ne7_angular_time01_batch128_third10k.py:102-110`: inherited
  settings and the mixed-pseudopotential Hamiltonian; CG ceiling 1024 is a
  numerical solver budget, not a Hamiltonian parameter.

## 4. What commutation does and does not preserve

For finite-dimensional Hermitian `H,C` with `[H,C]=0`, choose a simultaneous
eigenbasis `H|a>=Ea|a>`, `C|a>=ca|a>`. The augmented generator has the **same
joint eigenvectors** and eigenvalues `Ga=Ea+λca`. In a degenerate H subspace,
one may need to choose the basis that diagonalizes C; this does not mean
every arbitrary H eigenvector remains a C eigenvector.

The exact normalized imaginary-time weight ratio is

\[
\frac{p_a(\tau)}{p_b(\tau)}=
\frac{p_a(0)}{p_b(0)}
e^{-2\tau[(E_a-E_b)+\lambda(c_a-c_b)]}.
\]

This changes sector weights without rotating the joint eigenvectors. It
cannot create an exactly absent component. A nonzero initial overlap is
required for exact imaginary-time ground-state preparation.

**Within one C sector**, `ca=cb`, so energy ordering and all energy gaps are
unchanged. **Across sectors**, complete energy ordering is retained only if

\[
\boxed{E_b-E_a+\lambda(c_b-c_a)>0
\quad\text{for every pair with }E_a<E_b.}
\]

For `λ>=0`, a sufficient global condition is that `ca` is nondecreasing with
`Ea`. Otherwise the pairwise crossings impose upper bounds
`λ < (Eb-Ea)/(ca-cb)` whenever `Ea<Eb` but `ca>cb`.

Commutation alone is insufficient: `H=diag(0,1)`, `C=diag(2,0)`, `λ=1`
commute but have `H+λC=diag(2,1)`, which swaps the ground state.

**The useful weaker theorem for our case:** if `C>=0`, `C|ψ0>=0`, and
`λ>=0`, an original ground state `|ψ0>` of H remains a ground state, because

\[
(E_a-E_0)+\lambda c_a\geq0.
\]

If H has a unique gapped ground, it remains unique; if H is degenerate,
preserving the whole ground subspace requires C to annihilate that whole
subspace. This statement preserves the ground, not every excited-state
ordering. Example: `H=diag(0,1,2)`, `C=diag(0,2,0)`, `λ=1` gives
`H+C=diag(0,3,2)`: ground retained, excited levels reordered.

For our ED-verified `L=0` ground, `C=K>=0` and `K|ψ0>=0`. Hence any
nonnegative κ preserves that exact physical ground. In the exact joint
basis the gaps to unwanted angular sectors become

\[
\boxed{\Delta_{a,L}^{\rm aug}=
\Delta_{a,L}^{H}+\kappa L(L+1)}.
\]

Ratios of unwanted-sector weights to the `L=0` ground acquire an extra
suppression `exp[-2τ κ L(L+1)]`. Excited `L=0` states receive no additional
gap. This suggests a convergence benefit for sector contamination, not a
guarantee that finite-sample NN training converges faster. If the true ground
has `L>0`, a `K` penalty can select the wrong physical state.

Even exact imaginary-time evolution under `H+λC` does not generally make the
unpenalized energy monotonically decrease:

\[
\frac{d\langle H\rangle}{d\tau}=
-2\operatorname{Var}(H)-2\lambda\operatorname{Cov}(H,C).
\]

Commutation alone does not fix the covariance's sign. Decrease of the
augmented objective and decrease of physical energy are separate claims.

## 5. General target sectors

For a conserved observable Q and a desired eigenvalue q*, one can define a
nonnegative penalty `C=(Q-q*)²`, which commutes with H. If the desired
physical ground lies in q*, the preceding ground-preservation theorem applies.
If the target sector is not the global-ground sector, a sufficiently large
λ intentionally makes its sector minimum the augmented ground; it does not
preserve the original global energy ordering. In a finite joint spectrum,

\[
\lambda>\max_{a:q_a\ne q_*}
\frac{(E_*-E_a)_+}{(q_a-q_*)^2}
\]

is sufficient to make the target-sector minimum E* lower than every
outside-sector state (with the usual degeneracy qualifications).

Examples: particle number `(N-N*)²`, charge `(Q-Q*)²`, and spin-singlet
selection `S²`. For our zero-angular-momentum target, **K itself already is
nonnegative and vanishes only at L=0**; replacing it by K² would be a different,
stronger high-L penalty and is not the saved method.

Suggested main-slide conclusion:

> Add a commuting, nonnegative symmetry penalty that vanishes on the desired
> ground sector. This keeps that ground state and enlarges gaps to unwanted
> sectors. SPRING fits the combined tangent-space target; the physical energy
> and full-state ED fidelity remain the reported benchmarks.

## Primary sources checked

The ordering proofs and the extension above are direct finite-dimensional
derivations, not claims attributed to the SPRING paper.

- Goldshlager, Abrahamsen and Lin, *A Kaczmarz-inspired approach to accelerate
  the optimization of neural network wavefunctions*, JCP 516,113351 (2024),
  [arXiv:2401.10190v2](https://arxiv.org/html/2401.10190v2).
  Equations 27,30–33 and Algorithm 1 give projected memory, damping and the
  unit-time residual convention; equations 37–40 discuss the cap/stabilizer.
- Teng, Dai and Fu, *Solving and visualizing fractional quantum Hall
  wavefunctions with neural network*, PRB111,205117 (2025),
  [arXiv:2412.00618v2](https://arxiv.org/html/2412.00618v2).
  This is the FQH neural-wavefunction paper, **not** the SPRING optimizer paper.
  Its real-space, Landau-level-mixed problem is different from our occupation-
  basis, strictly-LLL sphere calculation.
- Kuroiwa and Nakagawa, *Penalty methods for a variational quantum
  eigensolver*, PRResearch3,013197 (2021),
  [publisher](https://doi.org/10.1103/PhysRevResearch.3.013197),
  [arXiv:2010.13951](https://arxiv.org/abs/2010.13951).
  This is related primary literature on symmetry-sector penalty objectives;
  the generic penalty concept is established, while our project adapts and
  tests it within the present SPRING implementation.

## First-ten-slide draft review

The new first ten slides were read against this note after root's draft was
written. No main-TeX edit was made by the reviewer. The signs, physical versus
dimensionless Hamiltonians, positive ladder identity, and the distinction
between fixed M and total L are correct. No blocking mathematical issue was
found.

Three clarity improvements were sent to root:

1. Slide 6: specify that normalized quadrature weights approximate Born
   probabilities. Normalized inverse-selection weights alone do not do that;
   the actual raw sampler weights and the ideal score-form probabilities
   should not be conflated.
2. Slide 9: add the explicit dimensionless mapping
   `e_H + λ e_C -> e_h + κ e_K`. For physical `C=K`, `λ=κ V1`; the current
   code operates on h and K. Applying the learning rate once remains correct.
3. Slide 10: replace the M=1 output occupation label `bold r` by `n_+` or
   `α`, so physics readers do not mistake occupation-space quadrature for
   real-space electron sampling.

There is no gauge mismatch in these slides: the new draft does not introduce
an explicit spinor gauge, and the canonical creation order remains consistent
with the fixed orbital ordering in the operators.

## Complete 25-frame draft review

The complete draft, particularly frames 11,18–23, was reviewed without editing
the main TeX. The implemented positive-force sign, augmented metric, explicit
L=0 preference, general commuting-operator theorem, mixed-H benchmark, and
controlled-versus-historical comparison limits are consistent with the source
and the derivations above. The examples correctly disprove unconditional
global energy ordering and distinguish ground preservation from excited-level
ordering.

Two precision corrections were sent to root:

- Frame 22: “monotone c as a function of E” must be **nondecreasing** for
  nonnegative λ. A monotone decreasing c is not a sufficient condition.
- Frames 11 and 23: explicitly label the quantities in the actual solve as
  sampled (`Jhat`, `Ghat`, and sampled component estimates), or define the
  convention in a sentence. Full-state projective identities on frame 23
  are implemented using weighted sampled counterparts. The distinction
  matters because the sampled autodiff force is not obtained by naively
  plugging a sampled local residual into an exact score-covariance identity.

Minor explanatory clarification: frame 18's displayed energy-error split uses
a normalized physical state Ψ. The numerical values are consistent with that
convention; saying “normalized Ψ” makes the identity explicit.
