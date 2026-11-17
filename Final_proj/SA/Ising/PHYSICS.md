# Shared Ising model, exact-objective SA

The Hamiltonian, positive bias-free RBM, spin-flip symmetry, initialization and
input convention are identical to the [shared Ising specification](../../VMC/Ising/PHYSICS.md).
For N=6, the network takes six spins as input and returns a scalar log amplitude.
Its 72 weights are searched; the normalization scalar c remains fixed at zero.

Unlike the sampled gradient runners, SA enumerates all 2^N configurations and
evaluates the normalized Rayleigh energy exactly for every candidate network.
This correctly weights configurations by the normalized squared amplitude;
it is not an unweighted average of local energies. There is no sampling batch.
The full Hamiltonian is reused from shared code, without training labels.

The artificial temperature belongs only to acceptance in weight space. It
does not change the Hamiltonian, the objective or the network probabilities.
The implementation supports 3 <= N <= 10 and h > 0, and is intended for small
systems. Enumeration and the dense Hamiltonian grow exponentially.

After training, diagonalization supplies ground energy and fidelity solely for
diagnostics. Residual variance and energy are computed independently for the
initial, last and best networks. Shared ED data remain in
[benchmarks](../../benchmarks/README.md); no new ED database is copied here.
