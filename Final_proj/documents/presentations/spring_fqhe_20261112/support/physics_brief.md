# Verified physics brief for the SPRING/FQHE presentation

Checked 12 November 2026 against the current implementation and saved N=7 reference metadata. No training or reference files were changed.

## Suggested physics sequence: five slides

### 1. A controlled interacting-electron problem

Use fully spin-polarized electrons restricted to the lowest Landau level (LLL). The kinetic energy is constant within this level and is dropped. The current interaction is the **normalized V1 pseudopotential model**, not a full Coulomb calculation. Landau-level mixing, finite thickness, disorder and higher-LL competition are outside this benchmark.

The ν=1/3 Laughlin sequence on the sphere uses

\[
F=N_\phi=2Q=3(N_e-1),\qquad R_s=\sqrt{Q}\,\ell_B.
\]

This is a finite-size shift relation; do not identify N_e/(F+1) with exactly 1/3 at finite N. For example, N_e=7 gives F=18 and 19 orbitals.

Reference: [Haldane, PRL 51, 605 (1983)](https://doi.org/10.1103/PhysRevLett.51.605); [Laughlin, PRL 50, 1395 (1983)](https://doi.org/10.1103/PhysRevLett.50.1395).

### 2. Haldane sphere → a finite occupation array

The LLL has single-particle angular momentum l=Q and magnetic quantum numbers m=-Q,…,Q. Code orbital index a=0,…,F obeys m=a-Q. One possible spinor phase convention is u=cos(θ/2)e^{iφ/2}, v=sin(θ/2)e^{-iφ/2}, with orbital proportional to u^a v^{F-a}. Only the orbital-density convention is needed for the wall; coordinate gauge phases are not NN inputs.

The many-electron state is

\[
|\Psi_\vartheta\rangle=\sum_{\mathbf n\in\mathcal S_{N_e,M}}
\psi_\vartheta(\mathbf n)|\mathbf n\rangle,
\quad |\mathbf n\rangle=\prod_{a=0}^{F}(c_a^\dagger)^{n_a}|0\rangle,
\quad n_a\in\{0,1\}.
\]

Canonical creation-operator order is ascending a. Thus **antisymmetry is already supplied by the Slater/Fock basis**; the NN predicts a scalar coefficient of an entire configuration, not the nth orbital. Allowed inputs satisfy Σ_a n_a=N_e and Σ_a(a-Q)n_a=M. All 19 LLL orbitals remain available for the N=7 test; only the many-body sector is selected. Its M=Lz=0 dimension is 1,656.

Do not confuse fixed Lz=0 with total angular momentum L=0. The no-wall Laughlin ground state has L=0, but the basis contains other L sectors. The NN is not explicitly projected onto L²=0.

Code: `src/fqh_spring/physics.py:1–9,61–100,189–224`; reference `references/ne7/metadata.json`.

### 3. Interaction: penalize close angular-momentum pairs

Use relative pair angular momentum r (odd for spin-polarized fermions), and pair total angular momentum J=2Q-r=F-r:

\[
H_{\rm int}=\sum_{i<j}\sum_{r\;\mathrm{odd}}V_r P_{ij}^{(J=F-r)};
\qquad H_{\rm int}/V_1=\sum_{i<j}P_{ij}^{(r=1)}
\quad\text{in the reported tests}.
\]

The implementation also supports V3/V1, but the reported N=7 continuation fixes it to zero. Each pair projector is normalized: projector eigenvalues are 0 or 1. The no-wall model has the Laughlin state as a zero-energy ground state on the above flux sequence.

The corresponding ordered-pair second-quantized expression is

\[
H_{\rm int}=\sum_{a<b,\,c<d} V_{ab,cd}
c_a^\dagger c_b^\dagger c_d c_c,
\qquad a+b=c+d.
\]

There is **no extra 1/2 or 1/4** in this ordered-pair convention. V_ab,cd consists of normalized projectors in antisymmetric two-particle states. A Clebsch–Gordan expression is equivalent, but the actual code constructs L_pair²=L_-L_+ + M_pair(M_pair+1), diagonalizes each fixed-pair-M block, and forms the outer product of the eigenvector with J=F-r. Its arbitrary overall eigenvector sign cancels in the projector. Many-body hop signs are computed separately from canonical fermion parity.

Code: `physics.py:40–58,102–145,201–224`; `local_energy.py:77–115`. Historical finite-size sphere benchmark context: [Haldane and Rezayi, PRL 54, 237 (1985)](https://doi.org/10.1103/PhysRevLett.54.237).

### 4. A projected spatial wall, with all LLL orbitals retained

The sphere itself has no physical boundary. We introduce an axisymmetric finite potential to penalize the region beyond a polar cap:

\[
x=\sin^2(\theta/2),\quad V_{\rm wall}(\theta)=U\,\Theta(x-x_c),
\quad U=10V_1,\; x_c=0.88.
\]

Here θ is the real-space polar angle; use a different symbol, e.g. ϑ, for the NN parameters. x grows from north to south. In our orbital convention, k=F-a is the outward degree, and north-localized orbitals have large a (k≈0).

LLL projection makes the one-body term diagonal:

\[
H_{\rm wall}=\sum_a\epsilon_a n_a,\qquad
\epsilon_a=U\frac{\int_{x_c}^{1}x^{F-a}(1-x)^a\,dx}
{B(F-a+1,a+1)}
=U\,[1-I_{x_c}(F-a+1,a+1)].
\]

The code uses `scipy.special.betaincc(F-a+1,a+1,x_c)`. This is the exact orbital-averaged finite step within the LLL, not an orbital cutoff, not a penalty on L², and not a temperature term. Orbital energies fall smoothly with a: at F=18, epsilon_0≈9.1186046 V1 and epsilon_18≈3.19×10^-17 V1. A purely affine profile in k would be constant at fixed N,K, so this nonlinear profile matters.

Code: `north_droplet.py:1–7,33–59,72–91`.

### 5. The full Hamiltonian and what ED must benchmark

\[
H=H_{\rm int}+H_{\rm wall},\qquad
E_{\rm loc}(\mathbf n)=\sum_{\mathbf n'}H_{\mathbf n\mathbf n'}
\frac{\psi_\vartheta(\mathbf n')}{\psi_\vartheta(\mathbf n)}.
\]

The interaction connects occupied orbital pairs to allowed same-sum pairs; the diagonal wall adds Σ_a epsilon_a n_a. No many-body ED matrix is needed for training. Fermion signs enter H; coefficient ratios are evaluated by exponentiating differences of complex network log coefficients.

The wall preserves N and Lz, but breaks full rotation symmetry and does not commute with L² in general. The N=7 wall benchmark is **only the ground state within the complete fixed-N=7,F=18,Lz=0 sector**. It is the same full-sphere sector as the no-wall N=7 study, not the distinct extra-flux north-cap sector from some N=8/N=10 studies.

Saved wall ED values:

- E0=2.82308466308447 V1; same-sector gap=0.1857586203776065 V1.
- Ground-state energy is not expected to be zero after adding this wall.
- The wall ED ground state has squared overlap only 0.027462567925589612 with the original no-wall Laughlin ground state. Thus label fidelity against the **matching Hamiltonian's** ED state.
- The latest continuation preserves this same wall Hamiltonian, basis, and source states; comparing the candidate-sampling methods is controlled. No ED eigenvectors enter the optimizer.

Current reference inputs: `results/ne7_gg_edge_20261111/references/wall10/{metadata.json,eigenvalues.npy,eigenvectors.npy}`; final continuation check `results/ne7_gg_explore_20261111/validation.json`.

## Related neural-FQHE paper: do not identify its model with ours

[Teng, Dai and Fu, arXiv:2412.00618v2; PRB 111, 205117 (2025)](https://arxiv.org/abs/2412.00618) is an attention-based real-space fermionic NN on a disk, allowing Landau-level mixing. It is relevant motivation, not a replication of our sphere occupation-MLP model. Its comparison to LL-projected ED uses a smaller ED Hilbert space; energies below that projected benchmark do not violate the variational principle. Our strictly LLL state cannot describe higher-LL admixture.

## Compact figure suggestions

1. Sphere with monopole flux, north pole, the xc=.88 latitude, and a shaded penalized south cap. Label 'finite wall; all LLL orbitals retained'.
2. A row of 19 occupation boxes → NN coefficient ψϑ(n) → weighted sum of Slater basis states. State N=7 and Σa n_a=7, Σa(a−9)n_a=0.
3. A pair scattering diagram (c,d)→(a,b), a+b=c+d, with antisymmetry residing in the Slater basis and hop parity.
4. Projected epsilon_a/V1 versus a for F18,U10,xc.88; order is critical (largea means north).

## Citations suitable for slide footers

- F. D. M. Haldane, PRL **51**, 605 (1983), [doi:10.1103/PhysRevLett.51.605](https://doi.org/10.1103/PhysRevLett.51.605): spherical geometry and finite-size quantum Hall construction.
- R. B. Laughlin, PRL **50**, 1395 (1983), [doi:10.1103/PhysRevLett.50.1395](https://doi.org/10.1103/PhysRevLett.50.1395): variational Laughlin state.
- F. D. M. Haldane and E. H. Rezayi, PRL **54**, 237 (1985), [doi:10.1103/PhysRevLett.54.237](https://doi.org/10.1103/PhysRevLett.54.237): finite-size sphere studies.
- Y. Teng, D. D. Dai and L. Fu, PRB **111**, 205117 (2025), [arXiv:2412.00618](https://arxiv.org/abs/2412.00618): neural FQHE motivation, with distinct geometry and Hilbert-space assumptions.
