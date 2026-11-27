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

The default network has six tanh layers of width 64 with additive readouts
from every layer (`legacy_additive`). Its two real outputs form a complex
number $\ell_\theta(n)$. The anchored Cartesian readout uses

\[
 u_\theta(n)=e^{\ell_0(n)}[1+\ell_\theta(n)-\ell_0(n)],
\]

where $\ell_0(n)$ comes from the frozen initial parameters. The amplitude
derivative inherits the fixed factor $e^{\ell_0(n)}$, so the current
amplitude can pass through zero. The explicit Ne4 and Ne6 baseline configs
use $\psi_\theta(n)=e^{\ell_\theta(n)}$ with Born sampling. All network
weights and readouts are trainable. The portable objective uses only the
physical Hamiltonian. The historical Ne12 angular-head runs have their own
training objective and architecture.

Both readouts use the physical energy objective

\[
 E_\theta=\frac{\psi^\dagger H\psi}{\psi^\dagger\psi}.
\]

The default sampler targets the Born distribution
$p_\theta(n)\propto |u_\theta(n)|^2$. SPRING uses equal empirical weights
for the sampled rows and constructs the projected amplitude tangent and
Hamiltonian residual from their coefficients and derivatives. The sampler uses
fixed-sum pair moves and periodic uniform-sector refresh proposals. The
force estimates half the physical energy gradient. The pinned JaQMC SPRING
optimizer carries memory across updates, and ED is used after training.
[Fit parameters](training/PARAMETERS.md)

Small shared ED references have M=0 dimensions18 (Ne4),73 (Ne5),338 (Ne6),
1656 (Ne7),8512 (Ne8). The shared benchmark remains at
[`Final_proj/benchmarks`](../../benchmarks/). No Ne12 ED or fidelity is claimed.
