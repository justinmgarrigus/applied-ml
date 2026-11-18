"""Portable exact-diagonalization benchmarks for the periodic Pauli Ising chain.

This module intentionally imports neither JAX nor the training/optimizer module.
All 2**N basis configurations and both spin-flip parity sectors are retained.
The saved neural checkpoint is evaluated only after ED, never used to build H.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy import linalg, sparse


ROOT = Path(__file__).resolve().parent
BASIS_CONVENTION = (
    "Integer configurations 0..2**N-1; site i is bit i (least significant first); "
    "bit 0 has Z_i=-1 and bit 1 has Z_i=+1. X_i flips bit i."
)
HAMILTONIAN = "H = -J sum_i Z_i Z_(i+1 mod N) - h sum_i X_i; Pauli eigenvalues +/-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spin_basis(n: int) -> np.ndarray:
    """Return the complete basis, matching ising_spring.enumerate_spins."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 3 <= n <= 10:
        raise ValueError("Require an integer 3 <= N <= 10; dense ED is explicitly bounded.")
    indices = np.arange(2**n, dtype=np.int64)
    return (2 * ((indices[:, None] >> np.arange(n)) & 1) - 1).astype(np.int8)


def sparse_hamiltonian(n: int, J: float = 1.0, h: float = 1.0) -> sparse.csr_matrix:
    """Build H from diagonal Ising bonds and individual bit-flip matrix elements."""
    states = spin_basis(n)
    if not np.isfinite([J, h]).all() or J < 0 or h < 0:
        raise ValueError("This ferromagnetic benchmark requires finite J >= 0 and h >= 0.")
    indices = np.arange(2**n)
    rows = [indices]
    cols = [indices]
    data = [-J * np.sum(states * np.roll(states, -1, axis=1), axis=1)]
    for site in range(n):
        rows.append(indices)
        cols.append(indices ^ (1 << site))
        data.append(np.full(indices.size, -h))
    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(indices.size, indices.size), dtype=np.float64,
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def parity_diagonalization(H: sparse.csr_matrix, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact parity-block ED; return the COMPLETE spectrum in increasing energy.

    Representatives have the most significant bit zero. In each +/- sector the
    orthonormal basis is (|r> +/- |complement(r)>)/sqrt(2). This avoids arbitrary
    parity mixing when energies in the two sectors are degenerate.
    """
    dimension = 2**n
    if H.shape != (dimension, dimension):
        raise ValueError("H dimension does not match N.")
    representatives = np.arange(dimension // 2)
    complements = (dimension - 1) ^ representatives
    same = H[representatives][:, representatives].toarray()
    exchanged = H[representatives][:, complements].toarray()
    spectra, vectors, labels = [], [], []
    for parity in (1, -1):
        energies, sector_vectors = linalg.eigh(
            same + parity * exchanged, check_finite=False, driver="evd"
        )
        full_vectors = np.empty((dimension, dimension // 2))
        full_vectors[representatives] = sector_vectors / np.sqrt(2.0)
        full_vectors[complements] = parity * sector_vectors / np.sqrt(2.0)
        # Fix the otherwise arbitrary real global sign for saved eigenvectors.
        maxima = np.argmax(np.abs(full_vectors), axis=0)
        signs = np.sign(full_vectors[maxima, np.arange(dimension // 2)])
        full_vectors *= np.where(signs == 0, 1.0, signs)
        spectra.append(energies)
        vectors.append(full_vectors)
        labels.append(np.full(dimension // 2, parity, dtype=np.int8))
    energies = np.concatenate(spectra)
    eigenvectors = np.concatenate(vectors, axis=1)
    parities = np.concatenate(labels)
    order = np.argsort(energies, kind="stable")
    return energies[order], eigenvectors[:, order], parities[order]


def observables(psi: np.ndarray, n: int, H: sparse.csr_matrix) -> dict:
    """Full-basis expectations; zz and transverse magnetization are per site."""
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape != (2**n,) or not np.isfinite(psi).all() or np.linalg.norm(psi) == 0:
        raise ValueError("Expected a finite, nonzero full-basis real wavefunction.")
    psi = psi / np.linalg.norm(psi)
    states = spin_basis(n)
    indices = np.arange(2**n)
    probabilities = psi**2
    hpsi = H @ psi
    energy = float(psi @ hpsi)
    residual = hpsi - energy * psi
    return {
        "energy": energy,
        "variance": float(residual @ residual),
        "spin_flip_parity": float(psi @ psi[(2**n - 1) ^ indices]),
        "transverse_magnetization_per_site": float(sum(
            psi @ psi[indices ^ (1 << site)] for site in range(n)
        ) / n),
        "nearest_neighbor_zz_per_bond": float(probabilities @ np.mean(
            states * np.roll(states, -1, axis=1), axis=1
        )),
        "longitudinal_magnetization_per_site": float(probabilities @ np.mean(states, axis=1)),
        "longitudinal_magnetization_squared": float(probabilities @ np.mean(states, axis=1)**2),
    }


def rbm_wavefunction(parameters: Path, n: int) -> np.ndarray:
    """Independent NumPy implementation of c + sum_a log cosh(sum_i s_i W_ia)."""
    with np.load(parameters, allow_pickle=False) as data:
        if set(data.files) != {"W", "c"}:
            raise ValueError("Expected the original Ising RBM checkpoint with W and c only.")
        weights, constant = np.asarray(data["W"]), np.asarray(data["c"])
    if weights.ndim != 2 or weights.shape[0] != n or weights.shape[1] < 1:
        raise ValueError("Checkpoint weight shape does not match N.")
    if constant.shape != () or not np.isfinite(weights).all() or not np.isfinite(constant):
        raise ValueError("Checkpoint parameters must be finite, with scalar c.")
    activations = spin_basis(n) @ weights
    logs = float(constant) + np.sum(np.logaddexp(activations, -activations) - np.log(2.0), axis=1)
    psi = np.exp(logs - np.max(logs))
    return psi / np.linalg.norm(psi)


def solve_case(n: int, J: float, h: float, low_states: int = 8) -> tuple[dict, dict]:
    """Compute diagnostics plus arrays for one finite ring, without saving files."""
    if isinstance(low_states, bool) or not isinstance(low_states, int) or low_states < 2:
        raise ValueError("Require at least two saved low states.")
    H = sparse_hamiltonian(n, J, h)
    started = time.perf_counter()
    energies, vectors, parities = parity_diagonalization(H, n)
    ground_energy = float(energies[0])
    tolerance = 1e-10 * max(1.0, n * (J + h))
    ground_indices = np.flatnonzero(energies - energies[0] < tolerance)
    count = min(2**n, max(low_states, len(ground_indices)))
    eigen_residuals = np.linalg.norm(H @ vectors - vectors * energies[None, :], axis=0)
    normalization_error = float(np.max(np.abs(vectors.T @ vectors - np.eye(2**n))))
    complement = (2**n - 1) ^ np.arange(2**n)
    parity_residual = float(np.max(np.abs(vectors[complement] - vectors * parities[None, :])))
    ground_observables = observables(vectors[:, 0], n, H)
    positive_levels = energies[energies > energies[0] + tolerance]
    even_energies = energies[parities == 1]
    odd_energies = energies[parities == -1]
    even_excited = even_energies[even_energies > even_energies[0] + tolerance]
    gap = float(positive_levels[0] - energies[0]) if positive_levels.size else None
    even_gap = float(even_excited[0] - even_energies[0]) if even_excited.size else None
    energy_reconstruction = -J * n * ground_observables["nearest_neighbor_zz_per_bond"] - h * n * ground_observables["transverse_magnetization_per_site"]
    checks = {
        "maximum_eigenpair_residual": float(np.max(eigen_residuals)),
        "maximum_orthonormality_error": normalization_error,
        "maximum_parity_residual": parity_residual,
        "hamiltonian_hermiticity_error": float(np.max(np.abs((H - H.T).data), initial=0.0)),
        "ground_energy_from_observables_error": float(abs(energy_reconstruction - ground_energy)),
        "full_spectrum_trace_error": float(abs(np.sum(energies))),
    }
    analytic = None
    if J > 0 and np.isclose(J, h, rtol=0, atol=1e-15):
        # Critical periodic Pauli chain, from the even/odd Jordan-Wigner sums:
        # E_even=-2J*csc(pi/2N), E_odd=-2J*cot(pi/2N).
        # Pair excitation in the even sector has energy 8J*sin(pi/2N).
        analytic = {
            "ground_energy": float(-2 * J / np.sin(np.pi / (2 * n))),
            "full_gap": float(2 * J * np.tan(np.pi / (4 * n))),
            "even_parity_gap": float(8 * J * np.sin(np.pi / (2 * n))),
            "source": "Exact finite-ring Jordan-Wigner momentum sums, no thermodynamic extrapolation",
        }
        checks["critical_ground_energy_error"] = abs(ground_energy - analytic["ground_energy"])
        checks["critical_full_gap_error"] = abs(gap - analytic["full_gap"])
        checks["critical_even_parity_gap_error"] = abs(even_gap - analytic["even_parity_gap"])
    if max(checks.values()) > 5e-9:
        raise ArithmeticError(f"ED validation failed: {checks}")
    report = {
        "n": n, "J": float(J), "h": float(h), "dimension": 2**n,
        "parity_sector_dimensions": {"+1": 2**(n - 1), "-1": 2**(n - 1)},
        "ground_energy": ground_energy,
        "ground_degeneracy": int(len(ground_indices)),
        "full_gap": gap,
        "full_gap_definition": "First energy strictly above the complete ground-state eigenspace",
        "even_parity_gap": even_gap,
        "odd_sector_minimum": float(odd_energies[0]),
        "ground_spin_flip_parity": int(parities[0]),
        "ground_observables": ground_observables,
        "low_energies": [float(value) for value in energies[:count]],
        "low_state_spin_flip_parities": [int(value) for value in parities[:count]],
        "analytic_critical_reference": analytic,
        "validation": checks,
        "diagonalization_seconds": time.perf_counter() - started,
        "method": "Dense Hermitian ED in exact global spin-flip +/- blocks; merged COMPLETE spectrum",
    }
    arrays = {
        "H": H, "basis_bits": np.arange(2**n, dtype=np.int64), "basis_spins": spin_basis(n),
        "eigenvalues": energies, "eigenstate_parities": parities,
        "low_eigenvectors": vectors[:, :count], "ground_subspace": vectors[:, ground_indices],
    }
    return report, arrays


def compare_checkpoint(parameters: Path, case: dict, arrays: dict) -> tuple[dict, np.ndarray]:
    """Evaluate a retained variational state using exact ED labels only here."""
    n = case["n"]
    psi = rbm_wavefunction(parameters, n)
    obs = observables(psi, n, arrays["H"])
    error = obs["energy"] - case["ground_energy"]
    fidelity = float(np.sum(np.abs(arrays["ground_subspace"].T @ psi)**2))
    even_gap = case["even_parity_gap"]
    full_gap = case["full_gap"]
    with np.load(parameters, allow_pickle=False) as checkpoint:
        parameter_count = sum(checkpoint[name].size for name in checkpoint.files)
    if abs(obs["spin_flip_parity"] - 1.0) > 1e-12:
        raise ArithmeticError("RBM checkpoint violates its required even spin-flip symmetry.")
    comparison = {
        "model": "Positive zero-bias RBM with exact global spin-flip + symmetry",
        "source_checkpoint": str(parameters.relative_to(ROOT)) if parameters.is_relative_to(ROOT) else parameters.name,
        "checkpoint_sha256": sha256(parameters),
        "n": n, "J": case["J"], "h": case["h"],
        "parameter_count": int(parameter_count),
        "evaluation": "Complete-basis NumPy Rayleigh quotient; no Monte Carlo uncertainty",
        "energy": obs["energy"], "energy_error": error,
        "energy_error_per_site": error / n, "fidelity": fidelity,
        "variance": obs["variance"], "spin_flip_parity": obs["spin_flip_parity"],
        "observables": obs,
        "full_gap": full_gap,
        "even_parity_gap": even_gap,
        "energy_error_over_full_gap": float(max(0.0, error) / full_gap) if full_gap else None,
        "infidelity_upper_bound_from_even_gap": float(max(0.0, error) / even_gap) if even_gap else None,
        "fidelity_lower_bound_from_even_gap": float(max(0.0, 1.0 - max(0.0, error) / even_gap)) if even_gap else None,
        "gap_bound_note": "Numerical finite-size ED reference bound, using explicitly checked even parity; not an interval-arithmetic certificate",
        "used_for_training": False,
    }
    if error < -1e-10 or fidelity > 1 + 1e-12:
        raise ArithmeticError("Variational state violates an ED energy/fidelity bound.")
    return comparison, psi


def run(output: Path, sizes: list[int], include_field_sweep: bool = True, low_states: int = 8) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(set(sizes)) != len(sizes) or not sizes:
        raise ValueError("Provide a nonempty list of distinct sizes.")
    old_summary = output / "summary.json"
    if old_summary.exists():
        old_config = json.loads(old_summary.read_text())["configuration"]
        requested = {"critical_sizes": sizes, "include_n6_field_sweep": include_field_sweep,
                     "low_states_saved": low_states}
        if any(old_config.get(key) != value for key, value in requested.items()):
            raise ValueError("Output contains another benchmark configuration; choose a different --output directory.")
    cases = [(n, 1.0, 1.0) for n in sizes]
    if include_field_sweep:
        cases += [(6, 1.0, 0.5), (6, 1.0, 2.0)]
    records, neural = [], None
    for n, J, h in cases:
        report, arrays = solve_case(n, J, h, low_states)
        name = f"n{n}_J{J:g}_h{h:g}".replace(".", "p")
        folder = output / name
        folder.mkdir(exist_ok=True)
        report["artifact_directory"] = name
        sparse.save_npz(folder / "hamiltonian.npz", arrays["H"])
        for array_name in ("basis_bits", "basis_spins", "eigenvalues", "eigenstate_parities", "low_eigenvectors", "ground_subspace"):
            np.save(folder / f"{array_name}.npy", arrays[array_name], allow_pickle=False)
        (folder / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
        records.append(report)
        if n == 6 and J == h == 1:
            checkpoint = ROOT / "results" / "spring_n6" / "trained_parameters.npz"
            neural, psi = compare_checkpoint(checkpoint, report, arrays)
            np.save(output / "retained_spring_n6_wavefunction.npy", psi, allow_pickle=False)
        print(f"N={n:2d} J={J:g} h={h:g} E0={report['ground_energy']:.12f} gap={report['full_gap']:.12f}", flush=True)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physics": HAMILTONIAN, "basis_convention": BASIS_CONVENTION,
        "configuration": {"critical_sizes": sizes, "include_n6_field_sweep": include_field_sweep,
                          "low_states_saved": low_states, "precision": "float64",
                          "randomness": "None: deterministic dense eigensolvers; no sampling or training",
                          "source_sha256": sha256(Path(__file__))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "records": records, "neural_comparison": neural,
        "limitations": [
            "Finite periodic rings only; results are not thermodynamic-limit energies or gaps.",
            "The RBM enforces even spin-flip parity; its accessible excitation gap differs from the complete Hilbert-space gap.",
            "ED outputs are evaluation references and are not used as supervised labels or optimizer inputs.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["n", "J", "h", "dimension", "ground_energy", "full_gap", "even_parity_gap",
              "ground_spin_flip_parity", "transverse_magnetization_per_site",
              "nearest_neighbor_zz_per_bond", "longitudinal_magnetization_squared"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields if key in record}
            row.update({key: record["ground_observables"][key] for key in fields if key in record["ground_observables"]})
            writer.writerow(row)
    table = ["| N | J | h | Dimension | Ground energy | Full gap | Even-parity gap |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    table += [f"| {record['n']} | {record['J']:g} | {record['h']:g} | {record['dimension']} | "
              f"{record['ground_energy']:.12f} | {record['full_gap']:.12f} | {record['even_parity_gap']:.12f} |"
              for record in records]
    checkpoint_text = "No N=6, J=h=1 neural comparison was requested."
    if neural is not None:
        checkpoint_text = (
            "The retained N=6 SPRING checkpoint was reevaluated independently with NumPy over all 64 configurations. "
            f"Its energy is {neural['energy']:.12f}, energy error {neural['energy_error']:.9g}, "
            f"squared ground-state fidelity {neural['fidelity']:.12f}, and variance {neural['variance']:.9g}. "
            "This is a reevaluation of the existing 250-update run, not a new optimization. "
            f"Its numerical even-sector gap bound is F >= {neural['fidelity_lower_bound_from_even_gap']:.12f}."
        )
    readme = f"""# Exact periodic Ising benchmarks

The Hamiltonian is `{HAMILTONIAN}`. Z has eigenvalues +/-1, not +/-1/2.
{BASIS_CONVENTION}

{'\n'.join(table)}

These are complete finite-ring spectra, obtained by diagonalizing both exact global-spin-flip
parity blocks and merging their eigenvalues. The full gap is the first energy above the
entire ground eigenspace. The even-parity gap is the first excitation within parity +1.
The existing zero-bias RBM enforces parity +1 and cannot access the first odd-parity state.
All listed h>0 ground states are unique and even.

{checkpoint_text}

At criticality J=h>0 the finite-ring Jordan-Wigner sums give
`E_even=-2J*csc(pi/(2N))` and `E_odd=-2J*cot(pi/(2N))`.
Consequently `E0=-2J*csc(pi/(2N))`, `gap=2J*tan(pi/(4N))`, and
`even_gap=8J*sin(pi/(2N))`. These analytical values independently check the ED results;
they are not thermodynamic-limit approximations. Ground-state energy decomposes into
`-J*N*<ZZ> - h*N*<X>`, another saved consistency check.

Regenerate from the `spring_ising_test` directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python ed_benchmark.py
.venv/bin/python -m pytest -q tests/test_ed_benchmark.py
```

The ED program itself requires only NumPy and SciPy and never imports JAX or the optimizer.
It uses no random numbers. Different BLAS/LAPACK implementations can rotate eigenvectors
inside a degenerate same-parity space; energy, projectors, and observables are the invariant references.

`summary.json` records configuration, versions, all diagnostics, and the checkpoint comparison;
`summary.csv` is a flat table. Each case directory contains sparse `hamiltonian.npz`,
complete `basis_bits.npy`, `basis_spins.npy`, `eigenvalues.npy`, `eigenstate_parities.npy`,
the first {low_states} `low_eigenvectors.npy` (columns, or more if required to include the complete
ground eigenspace), `ground_subspace.npy`, and `diagnostics.json`.
The normalized saved neural state is `retained_spring_n6_wavefunction.npy` when N=6 is included.
`manifest.json` gives SHA256 hashes of all benchmark artifacts except itself.

Checks cover Hermiticity, all eigenpair residuals, complete orthonormality,
parity, trace, observable energy reconstruction, and the critical finite-N formulas.
These outputs are benchmarks only; no ED labels enter the SPRING force, model initialization,
or Born-sampling distribution. No new neural training is performed.
"""
    (output / "README.md").write_text(readme)
    manifest = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--output", type=Path, required=True, help="New directory only")
    parser.add_argument("--no-field-sweep", action="store_true")
    parser.add_argument("--low-states", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh directory")
    run(args.output, args.sizes, not args.no_field_sweep, args.low_states)


if __name__ == "__main__":
    main()
