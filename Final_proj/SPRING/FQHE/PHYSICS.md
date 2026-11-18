# FQHE model used by the portable SPRING comparison

The input is a binary occupation array `n` of length `flux+1`. Exactly `Ne`
entries are one, and `sum((2*a-flux)*n[a])=0` fixes total `Lz=0`. These arrays
label antisymmetric Slater determinants in ascending orbital order; the
Hamiltonian connections include the fermionic exchange signs.

We retain the lowest Landau level on the Haldane sphere, with
`flux=3*(Ne-1)` and normalized pair projectors:

\[
H/V_1=\sum_{i<j}P_{R=1}(i,j).
\]

The supplied comparison configs set `v3=0`, with no wall, no higher Landau
levels and no background/kinetic offset. The reusable operator also supports
an explicit `v3*P3` term. The pure-V1 Laughlin ground energy is analytically
zero; the NN receives neither Laughlin coefficients nor ED eigenvectors.

The network returns two real numbers, log magnitude `a(n)` and phase `phi(n)`,
so `psi(n)=exp(a(n)+i*phi(n))`. The default is six tanh layers of width64 with
an additive readout from **every** layer (`legacy_additive` is the parameter
schema name). All weights/readouts are trainable. This portable NN has no
private angular heads and no L-squared penalty. It is a fresh model, distinct
from the historical Ne12 angular-head research runs.

The objective and its sampled evaluation are

\[
 E_\theta=\frac{\psi^\dagger H\psi}{\psi^\dagger\psi},\qquad
 p_\theta(n)\propto|\psi_\theta(n)|^2,\qquad
 E_{\rm loc}(n)=\sum_{n'}H_{nn'}\psi(n')/\psi(n).
\]

The sampler uses fixed-sum pair moves and periodic uniform-sector refreshes.
There is no physical or artificial temperature in this portable protocol.
The covariance force is half the physical energy gradient. SPRING applies the
pinned upstream JaQMC optimizer with memory; ED remains post-training only.

Small shared ED references have M=0 dimensions18 (Ne4),73 (Ne5),338 (Ne6),
1656 (Ne7),8512 (Ne8). The shared benchmark remains at
[`Final_proj/benchmarks`](../../benchmarks/). No Ne12 ED or fidelity is claimed.
