# SPRING and the implemented network: verified slide brief

Checked 2026-11-12 against the original papers and the current project source. No training or source changes were performed.

## Primary references and a compact literature summary

- G. Goldshlager, N. Abrahamsen, L. Lin, *A Kaczmarz-inspired approach to accelerate the optimization of neural network wavefunctions*, J. Comput. Phys. **516**, 113351 (2024): [arXiv 2401.10190v2](https://arxiv.org/html/2401.10190v2), [DOI](https://doi.org/10.1016/j.jcp.2024.113351).
- A. Chen and M. Heyl, *Empowering deep neural quantum states through efficient optimization*, Nature Physics **20**, 1476–1481 (2024): [arXiv 2302.01941](https://arxiv.org/abs/2302.01941), [DOI](https://doi.org/10.1038/s41567-024-02566-1). **2302.05029 is an unrelated astronomy paper.**

VMC supplies sampled energy and derivatives. SR uses the wavefunction metric to precondition the energy force. MinSR moves the solve from parameter space to the smaller sampled space. SPRING retains a decayed previous direction and corrects its disagreement with the current sampled equations. It therefore carries information between batches without storing old Jacobians. The original paper compares tuned optimizers for FermiNet atoms and molecules; its oxygen example reaches chemical accuracy with SPRING at 40k updates, while MinSR and KFAC do not by 100k. This is literature evidence, not our FQHE optimizer benchmark. The original FermiNet uses four equivariant layers, not one or two ordinary MLP layers. Classical fixed-system Kaczmarz convergence does not establish convergence for nonlinear, changing, damped VMC equations.

Paper locations: Secs. 2.1–2.3 for VMC/SR/MinSR; Sec. 3, Eqs. 27–34 for projected memory; Algorithm 1 and Eqs. 37–40 for cap/null-space stabilization; Fig. 2 and Sec. 4.1 for atoms; Appendix B Table 2 for FermiNet layers.

## Seven compact slide modules

### 1. VMC is the framework; the optimizer is a separate choice

For an occupation configuration \(\mathbf n\), the NN supplies a coefficient \(\psi_\theta(\mathbf n)\), not its energy. The known sparse Hamiltonian supplies

\[
E_{\rm loc}(\mathbf n)=\sum_{\mathbf n'}H_{\mathbf n\mathbf n'}\,
\frac{\psi_\theta(\mathbf n')}{\psi_\theta(\mathbf n)},\qquad
E(\theta)=\frac{\psi_\theta^\dagger H\psi_\theta}{\psi_\theta^\dagger\psi_\theta}
=\mathbb E_{p_\theta}E_{\rm loc},\quad p_\theta\propto|\psi_\theta|^2.
\]

Short ML wording: **unsupervised variational optimization; the Hamiltonian defines the loss; ED labels are not used for training.** SGD, Adam, SR, MinSR and SPRING are possible optimizers inside this same framework. Do not describe SPRING as replacing VMC.

### 2. From SGD to SR and MinSR

All trainable parameters are real. Define \(O_{ij}=\partial_{\theta_j}\log\psi_\theta(\mathbf n_i)\), normalized batch weights \(w_i\), and weighted centered quantities

\[
\widetilde O_i=O_i-\sum_r w_rO_r,\qquad
\widetilde e_i=E_{{\rm loc},i}-\sum_r w_rE_{{\rm loc},r},
\]
\[
A=\begin{bmatrix}\Re[\sqrt w\,\widetilde O]\\\Im[\sqrt w\,\widetilde O]\end{bmatrix},\quad
e=\begin{bmatrix}\Re[\sqrt w\,\widetilde e]\\\Im[\sqrt w\,\widetilde e]\end{bmatrix},
\quad f=A^Te=\Re\sum_i w_i\widetilde O_i^*\widetilde e_i=\tfrac12\nabla_\theta E.
\]

Here \(S=A^TA\) is the real-parameter quantum metric. With uniform Born samples, \(w_i=1/B\); the \(1/\sqrt B\) normalization is already in \(A,e\).

- SGD: \(\Delta\theta=-2\eta f\) if \(\eta\) multiplies the physical energy gradient.
- SR: \(\Delta\theta\propto-(A^TA+\lambda I)^{-1}f\): parameter-space solve, \(P\times P\).
- Damped MinSR: \(\phi_{\rm MinSR}=-A^T(AA^T+\lambda I)^{-1}e\): sample-space solve, at most \(2B\times2B\) in this complex, real-parameter formulation.

Short ML wording: **Jacobian features, covariance preconditioning, a smaller dual solve.** MinSR changes the way the direction is computed, not the physical loss. The original MinSR work uses a pseudoinverse cutoff; SPRING's reference implementation compares its Tikhonov-regularized variant.

### 3. SPRING retains the compatible part of the previous direction

Use a **descent-direction** convention, so \(\epsilon=-e\). The conceptual damped update is

\[
\phi_k=\mu\phi_{k-1}+A_k^T(A_kA_k^T+\lambda I)^{-1}
\left(\epsilon_k-A_k\mu\phi_{k-1}\right).
\]

Equivalently, it minimizes

\[
\|A_k\phi-\epsilon_k\|^2+\lambda\|\phi-\mu\phi_{k-1}\|^2.
\]

The correction reduces the current residual; the memory carries the previous direction outside the currently observed row space. This differs from adding ordinary momentum because its conflicting sampled component is corrected. The current code uses a **positive-force** internal direction \(\delta=-\phi\) and finally applies a minus sign through Optax. Mixing the paper's negative \(\epsilon\) with the code's positive force would give a wrong sign.

The implementation also lifts the known mean-null directions before Cholesky. That stabilizer leaves the update unchanged in exact arithmetic on centered score data. The equation above can be shown without that technical term on the main slide, with a small footnote that the code uses null-space stabilization.

### 4. What the two output heads mean

For the six-layer additive-readout model used in the current N=7 comparisons,

\[
x=2\mathbf n-1,\quad h^{(1)}=\tanh(xW^{(1)}+b^{(1)}),\quad
h^{(\ell)}=\tanh(h^{(\ell-1)}W^{(\ell)}+b^{(\ell)}),
\]
\[
(a_\theta,\varphi_\theta)=c+\sum_{\ell=1}^{6}h^{(\ell)}R^{(\ell)},\qquad
\psi_\theta(\mathbf n)=\exp[a_\theta(\mathbf n)+i\varphi_\theta(\mathbf n)].
\]

The two heads are **log magnitude and phase**, not raw real and imaginary coefficients:
\(\Re\psi=e^a\cos\varphi\), \(\Im\psi=e^a\sin\varphi\).
\(\theta\) means the collection of all network weights and biases; \(\mathbf n\) labels the whole occupation configuration, not an orbital or a parameter index. The coefficient multiplies one canonical antisymmetric Slater basis state.

Each \(R^{(\ell)}\) is a learned \(64\times2\) linear readout. The heads use hidden features at every depth, while each deeper hidden layer still depends on all preceding layers. This supplies multiple paths for gradients; it is a model choice, not a required consequence of electron number. Adding heads in log space multiplies their amplitude/phase factors in the coefficient. The energy loss stays exactly the same for one, two or six hidden layers.

Short ML wording: **nonlinear feature layers, two real heads, skip connections to the output.** Tanh is smooth and bounded, but can saturate; more depth does not guarantee easier optimization or a lower variational energy.

### 5. Actual parameter and sampling settings

| Quantity | Current N=7 continuation | Effect |
|---|---:|---|
| Input | 19 occupation entries | One LLL determinant configuration |
| Hidden layers | 6, width64, tanh | Trainable nonlinear features |
| Readouts | Every layer, each64×2 | Log magnitude and phase |
| Real parameter count | 22,850 | Every weight, bias and readout optimized |
| Candidate pool | M=128 | Configurations scored before selection |
| Update batch | B=32, replacement allowed | Rows contributing to force and metric |
| Local MCMC sweeps/update | 2 | Persistent walkers; temporal correlation remains |
| Uniform-sector MH refresh | Every25 updates | Additional opportunity to leave local regions |
| Learning rate | eta=.01 | Multiplies the raw direction |
| Damping | lambda=.01 | Stabilizes inverse; suppresses weak directions |
| Memory decay | mu=.9 | Forgets older directions as the linearization changes |
| Update radius | max_norm=.03 | Euclidean parameter-step cap |
| Source/final update | 8000→10000 | Full-state continuation, three paired seeds |

Parameter count independently reconstructed:
\[
P=(19\cdot64+64)+5(64^2+64)+6(64\cdot2)+2=22850.
\]

**Important cap convention:** this package's \(\mathtt{max\_norm}=r=.03\) is the radius, whereas the paper's \(C\) bounds the **squared** step norm. Thus \(C=r^2=.0009\), and
\[
\Delta\theta=\phi\min\left(\eta,\frac{r}{\|\phi\|+10^{-8}}\right),\qquad\|\Delta\theta\|\lesssim .03.
\]
Neither \(\lambda\) nor the cap is an extra Hamiltonian term, an \(L^2\) angular-momentum penalty, or weight decay in the physical energy loss.

**Matrix-size detail for backup/speaker notes:** raw selected score data are32×22850 complex. A minimal realification is64×22850. The historical weighted adapter first makes signed positive/negative copies of complex rows, then JaQMC splits real and imaginary channels; its actual Gram solve is128×128. Those copies preserve the same weighted metric and add no independent information. Rank is at most2(B−1)=62 before additional feature redundancy.

### 6. A full update: physics operation with a short ML label

1. Keep \(N_e,L_z\) fixed and evolve candidate configurations with MH: **minibatch generation**.
2. Evaluate \(a,\varphi\), then logarithmic derivatives for all parameters by automatic differentiation: **forward pass and Jacobian**.
3. Select B rows from M candidates, with strictly positive probabilities: **importance/diversity sampling**.
4. Apply the known sparse Hamiltonian via coefficient ratios to selected configurations: **compute the physics-defined loss and force**.
5. Center and weight scores, build the sample Gram matrix: **estimate curvature/covariance**.
6. Correct decayed direction memory using Cholesky, then scale/cap and apply: **preconditioned optimization update**.
7. Periodically evaluate the complete sector against ED: **independent validation**, not training labels.

The derivative \(O_{ij}\) includes gradients through **all** \(W^{(\ell)},b^{(\ell)},R^{(\ell)}\). Readouts are not the only parameters being trained.

### 7. Broader pool: change the proposal, retain the physical objective

The pilot uses \(q_\beta(\mathbf n)\propto|\psi_\theta(\mathbf n)|^{2\beta}\), with beta=.5. It broadens the proposal relative to the Born distribution. If conditional selection probability at draw t is \(\pi_t(i)>0\), selected weights satisfy

\[
w_i\propto\frac{|\psi_\theta(\mathbf n_i)|^{2(1-\beta)}}{M\pi_t(i)},\qquad\sum_iw_i=1.
\]

Beta is a sampling parameter, not a physical temperature, and no cooling schedule was run. The two importance factors correct proposal broadening and geometry-based selection. Sampling retains full support and replacement, including repeated configurations. Self-normalization and estimated covariance means retain finite-batch ratio/covariance bias; MCMC also has equilibration/autocorrelation errors. The geometric diversity approximation uses diagonal whitening and regularized score projections, not exact full-QGT principal angles. Low Born probability alone does not establish gradient importance.

## Audited source pointers (project-relative)

- `src/fqh_spring/legacy_model.py:74–98`: all-layer readout, complex log coefficient, real/imaginary score derivatives.
- `src/fqh_spring/model.py:21–39`: explicit final-vs-additive schema dispatch; `59–76`: fresh additive initialization; `42–56`: parameter count.
- `src/fqh_spring/optimizer.py:1–5`: half-gradient/metric convention; `21–29`: actual JaQMC settings and no MARCH.
- `src/fqh_spring/geometry_sampling.py:103–117`: weighted-centered signed complex rows preserve physical covariance.
- `src/fqh_spring/train_exploration.py:52–85`: actual candidate→selection→local energy→force→optimizer chain; `86–108`: raw direction, cap diagnostics and parameter update.
- `src/fqh_spring/exploration_sampling.py:36–53`: stable combined log importance weights; `56–80`: weighted Born geometry; `110–149`: sequential conditional selection and full support.
- `vendor/jaqmc_spring/src/jaqmc/optimizer/sr/jasr.py:201–205`: final negative Optax sign; `243–264`: documented kernel recurrence; `405–450`: centering/normalization/realification; `477–496`: actual damped solve and direction; `641–664`: null-space stabilization; `707–745`: cap uses the radius supplied as max_norm.
- `results/ne7_gg_explore_20261111/configs/gg_original_seed71.json`: actual fixed N=7 settings, all-layer readout, beta and parent SHA. Sibling configs differ only in declared method/beta and seed/path metadata.

## Claims to keep distinct on slides

- The original paper's favorable optimizer ranking is established for its tuned molecular experiments. Our recent N=7 matched comparison holds SPRING fixed and changes candidate sampling/selection; it is not an FQHE comparison of SPRING versus MinSR/KFAC.
- N=6, N=7 and N=8 results with different ansatzes/hamiltonians/budgets are progress examples, not a controlled particle-number scaling study.
- Small sampled local-energy variance or a quiet MC energy trace does not establish exact eigenstate accuracy. Show exact energy excess, residual variance and ED fidelity together where available.
- A larger candidate pool M, a larger update batch B, more updates and greater network depth are distinct interventions.
