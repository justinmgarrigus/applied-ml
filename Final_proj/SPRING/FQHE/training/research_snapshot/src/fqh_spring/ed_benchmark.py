"""Fresh, bounded exact-diagonalization benchmarks for the sphere V1 model.

The default N=4,...,8 calculations are complete fixed-Lz sectors within the LLL.
Only low eigenpairs are saved; sparse matrices stay in the portable references.
No training is performed and no ED eigenvectors enter SPRING optimization.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh

from .references import build_reference, load_reference, sha256, sparse_operators

ENERGY_TOL = 1e-9
DEGENERACY_TOL = 1e-8


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clusters(values, tolerance=DEGENERACY_TOL):
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[stop] - values[start] <= tolerance:
            stop += 1
        yield start, stop
        start = stop


def diagonalize(H, raising, *, count=20, seed=1729, two_lz=0):
    """Solve low states afresh, and resolve L within complete energy clusters.

    Extra eigenpairs guard the requested cutoff.  If the solver's outermost
    cluster touches that guard, it is omitted rather than falsely called a
    complete degenerate eigenspace.  L is reported only after a full-space L2
    residual check, so missing degenerate partners cannot produce a false L.
    """
    if not 2 <= count <= 128:
        raise ValueError("Request between 2 and 128 low states; this CLI is not intended for large full-spectrum ED")
    dimension = H.shape[0]
    if H.shape != (dimension, dimension) or raising.shape[1] != dimension:
        raise ValueError("Hamiltonian/raising dimensions disagree")
    difference = H - H.T.conj()
    hermiticity = float(np.max(abs(difference.data), initial=0.0))
    if hermiticity > 1e-10:
        raise ArithmeticError("Hamiltonian is not Hermitian")
    requested = min(int(count), dimension)
    solved = min(dimension, requested + 8)
    started = time.perf_counter()
    if dimension <= 512 or solved >= dimension - 1:
        values, vectors = linalg.eigh(H.toarray(), check_finite=True)
        solver, all_states = "scipy.linalg.eigh (full sector)", True
    else:
        values, vectors = eigsh(
            H, k=solved, which="SA", tol=1e-12, maxiter=20000,
            ncv=min(dimension, max(2 * solved + 1, 80)),
            v0=np.random.default_rng(seed).normal(size=dimension))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        solver, all_states = "scipy.sparse.linalg.eigsh (lowest algebraic)", False
    elapsed = time.perf_counter() - started
    # Complete the energy cluster crossing the requested cutoff.
    end = requested
    while end < len(values) and values[end] - values[requested - 1] <= DEGENERACY_TOL:
        end += 1
    if end == len(values) and not all_states:
        cut_start = next(a for a, b in _clusters(values) if b == end)
        end = cut_start
    if end < 2:
        raise ArithmeticError("Too few complete low-energy states were obtained")
    values, vectors = values[:end].copy(), vectors[:, :end].copy()
    magnetization = two_lz / 2

    def apply_l2(v):
        return raising.T.conj() @ (raising @ v) + magnetization * (magnetization + 1) * v

    for start, stop in _clusters(values):
        block = vectors[:, start:stop]
        projected = block.T.conj() @ apply_l2(block)
        _, rotation = linalg.eigh((projected + projected.T.conj()) / 2)
        vectors[:, start:stop] = block @ rotation
    # Fix deterministic real-vector signs for the artifacts.
    for column in range(vectors.shape[1]):
        pivot = np.argmax(abs(vectors[:, column]))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    applied, l2_applied = H @ vectors, apply_l2(vectors)
    values = np.einsum("ij,ij->j", vectors.conj(), applied).real
    residuals = np.linalg.norm(applied - vectors * values, axis=0)
    l2 = np.einsum("ij,ij->j", vectors.conj(), l2_applied).real
    l2_residuals = np.linalg.norm(l2_applied - vectors * l2, axis=0)
    inferred = (np.sqrt(np.maximum(0, 1 + 4 * l2)) - 1) / 2
    integer_l = np.rint(inferred).astype(int)
    integer_errors = abs(l2 - integer_l * (integer_l + 1))
    orthogonality = float(np.max(abs(vectors.T.conj() @ vectors - np.eye(end))))
    commutator = np.linalg.norm(H @ l2_applied - apply_l2(applied), axis=0)
    if residuals.max() > 1e-8 or orthogonality > 1e-9:
        raise ArithmeticError("Fresh ED eigenpair validation failed")
    if values.min() < -ENERGY_TOL:
        raise ArithmeticError("Positive-projector Hamiltonian has a negative eigenvalue")
    zero = abs(values) < ENERGY_TOL
    positive = values > ENERGY_TOL
    if zero.sum() != 1 or not positive.any():
        raise ArithmeticError("Expected one Laughlin zero mode and a positive neutral excitation")
    states = []
    for index in range(end):
        resolved = l2_residuals[index] < 1e-7 and integer_errors[index] < 1e-7
        states.append(dict(index=index, energy_V1=float(values[index]),
                           excitation_energy_V1=float(values[index] - values[0]),
                           L=int(integer_l[index]) if resolved else None,
                           L_resolved=bool(resolved), L2=float(l2[index]),
                           L2_eigenvector_residual=float(l2_residuals[index]),
                           eigenpair_residual_V1=float(residuals[index])))
    return dict(values=values, vectors=vectors, states=states,
                validation=dict(hermiticity_max_error=hermiticity,
                                orthogonality_max_error=orthogonality,
                                maximum_eigenpair_residual_V1=float(residuals.max()),
                                maximum_L2_eigenvector_residual=float(l2_residuals.max()),
                                maximum_H_L2_commutator_on_low_states=float(commutator.max()),
                                all_reported_L_resolved=all(s["L_resolved"] for s in states)),
                solver=solver, diagonalization_seconds=elapsed, ground_energy_V1=float(values[0]),
                neutral_gap_V1=float(np.min(values[positive]) - values[0]),
                zero_modes_in_saved_spectrum=int(zero.sum()),
                computed_full_spectrum=all_states, requested_eigenpairs=requested,
                saved_eigenpairs=end, cutoff_observed_energy_cluster_complete=True,
                cutoff_guard="Eight extra Ritz pairs; extend observed cutoff cluster and omit outer computed cluster if touched",
                seed=int(seed))


def benchmark_one(ne, root, output, *, count=20, seed=1729, rebuild_through=6):
    from .evaluate import state_diagnostics
    from .physics import SphereOperator

    root, output = Path(root).resolve(), Path(output).resolve()
    if not 2 <= ne <= 8:
        raise ValueError("Benchmarks are deliberately bounded to 2 <= N <= 8")
    reference_dir = root / "references" / f"ne{ne}"
    if not reference_dir.exists():
        build_reference(ne, reference_dir, eigenpairs=max(4, min(count, 8)))
    reference = load_reference(reference_dir)
    metadata = reference["metadata"]
    if metadata["flux"] != 3 * (ne - 1) or metadata["two_lz"] != 0 or metadata["v3"] != 0:
        raise ValueError("Benchmark requires pure V1 at Laughlin flux and M=0")
    result = diagonalize(reference["H"], reference["raising"], count=count, seed=seed)
    rebuild = dict(performed=False, scope="Rebuild omitted above configured small-N bound")
    if ne <= rebuild_through:
        started = time.perf_counter()
        bits, H, raising, _ = sparse_operators(SphereOperator(ne))
        difference, rdiff = H - reference["H"], raising - reference["raising"]
        matrix_error = float(np.max(abs(difference.data), initial=0.0))
        raising_error = float(np.max(abs(rdiff.data), initial=0.0))
        if not np.array_equal(bits, reference["bits"]) or matrix_error > 1e-12 or raising_error > 1e-12:
            raise ArithmeticError("Fresh runtime Hamiltonian reconstruction disagrees with reference")
        rebuild = dict(performed=True, basis_identical=True, Hamiltonian_max_error=matrix_error,
                       raising_max_error=raising_error, seconds=time.perf_counter() - started,
                       scope="Reassembled from runtime SphereOperator connections; pair-projector construction is shared")
    ground = state_diagnostics(result["vectors"][:, 0], reference)
    expected_occupations = ne / (int(metadata["flux"]) + 1)
    uniform_error = float(np.max(abs(np.asarray(ground["orbital_occupations"]) - expected_occupations)))
    if abs(ground["exact_L2"]) > 1e-8 or uniform_error > 1e-8:
        raise ArithmeticError("Laughlin ground-state rotation/uniform-density check failed")
    ne_output = output / f"ne{ne}"
    ne_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ne_output / "eigenpairs.npz", eigenvalues=result.pop("values"),
                        eigenvectors=result.pop("vectors"), basis_bits=reference["bits"])
    summary = dict(ne=ne, flux=int(metadata["flux"]), two_lz=0, v3=0.0,
                   dimension=len(reference["bits"]), orbitals=int(metadata["flux"]) + 1,
                   full_fixed_N_dimension=int(__import__("math").comb(int(metadata["flux"]) + 1, ne)),
                   Hamiltonian="sum_(i<j) P_R=1(ij), normalized sphere LLL projectors; V1=1",
                   basis="Complete fixed-N, fixed-M=0 LLL Slater basis, ascending sorted bitstrings",
                   reference=str(reference_dir.relative_to(root)),
                   reference_manifest_sha256=sha256(reference_dir / "manifest.json"),
                   eigenpairs_file="eigenpairs.npz", eigenpairs_sha256=sha256(ne_output / "eigenpairs.npz"),
                   **result, runtime_reconstruction=rebuild,
                   ground=dict(L2=ground["exact_L2"], residual_variance_V1_squared=ground["exact_residual_variance_V1_squared"],
                               fidelity_with_retained_ED_ground=ground["ground_state_fidelity"],
                               orbital_occupations=ground["orbital_occupations"],
                               maximum_occupation_error_from_uniform=uniform_error,
                               density_RMS_contrast=ground["density_RMS_contrast"],
                               density_particle_number_integral=ground["density_particle_number_integral"]),
                   ed_used_for_training=False,
                   gap_definition="First positive eigenenergy minus E0 at fixed N and flux; neutral finite-size gap, not charged gap")
    l0_positive = [s["energy_V1"] for s in summary["states"]
                   if s["L"] == 0 and s["energy_V1"] > ENERGY_TOL]
    l0_gap_resolved = (bool(l0_positive)
                       and all(s["L_resolved"] for s in summary["states"]
                               if s["energy_V1"] <= min(l0_positive)))
    summary["lowest_L0_excitation_gap_V1"] = (min(l0_positive) - summary["ground_energy_V1"]
                                                   if l0_gap_resolved else None)
    summary["L0_gap_status"] = ("Resolved within the saved low-energy spectrum, with all lower L labels resolved"
                                if l0_gap_resolved else
                                "No certified lowest positive L=0 state in saved levels; not determined")
    _json(ne_output / "summary.json", summary)
    _csv(ne_output / "spectrum.csv", summary["states"])
    print(f"N={ne}, dimension={summary['dimension']}: E0={summary['ground_energy_V1']:.5g}, "
          f"neutral gap={summary['neutral_gap_V1']:.10g}, low states={summary['saved_eigenpairs']}", flush=True)
    return summary


def compare_pilot(root, output, summary):
    """Reevaluate six actual neural checkpoints, then compare with fresh ED."""
    from .evaluate import evaluate

    root, output = Path(root).resolve(), Path(output).resolve()
    eigenpairs = np.load(output / "ne8/eigenpairs.npz", allow_pickle=False)
    ground = eigenpairs["eigenvectors"][:, 0]
    energy0, gap = summary["ground_energy_V1"], summary["neutral_gap_V1"]
    rows = []
    for variant in ("base", "dense", "attention"):
        for seed in (71, 137):
            arm = f"{variant}_seed{seed}"
            checkpoint = root / "results/ne8_architecture_pilot/checkpoints" / f"{arm}.npz"
            evaluation_file = output / "model_evaluations" / f"{arm}.json"
            result = evaluate(checkpoint, root / "references/ne8", evaluation_file,
                              save_psi=True, angular=True)
            psi = np.load(evaluation_file.with_name(evaluation_file.stem + "_coefficients.npy"), allow_pickle=False)
            fidelity = float(abs(np.vdot(ground, psi)) ** 2)
            energy = result["exact_variational_energy_V1"]
            error = float(energy - energy0)
            bound = float(max(0, 1 - error / gap))
            if fidelity + 1e-9 < bound:
                raise ArithmeticError("Gap/fidelity variational bound violated")
            # Keep every saved report transferable between machines.
            result["checkpoint"] = str(checkpoint.relative_to(root))
            result["reference"] = "references/ne8"
            result["fresh_ED_ground_state_fidelity"] = fidelity
            result["energy_error_from_fresh_ED_V1"] = error
            _json(evaluation_file, result)
            rows.append(dict(model=variant, seed=seed, parameter_count=result["parameter_count"],
                             energy_V1=energy, ED_ground_energy_V1=energy0,
                             energy_error_V1=error, neutral_gap_V1=gap,
                             energy_error_over_neutral_gap=error / gap,
                             ground_state_fidelity=fidelity,
                             infidelity=1 - fidelity,
                             gap_based_fidelity_lower_bound=bound,
                             residual_variance_V1_squared=result["exact_residual_variance_V1_squared"],
                             residual_norm_V1=float(np.sqrt(result["exact_residual_variance_V1_squared"])),
                             L2=result["exact_L2"], L0_probability=result["exact_L0_probability"],
                             checkpoint=str(checkpoint.relative_to(root)),
                             checkpoint_sha256=result["checkpoint_sha256"]))
    _csv(output / "checkpoint_comparison.csv", rows)
    _json(output / "ne8_neural_comparison.json", rows)
    return rows


def run_benchmarks(root, output, *, sizes=(4, 5, 6, 7, 8), count=20, seed=1729, compare_models=True):
    import scipy

    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sizes = tuple(sizes)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Supply at least one distinct system size")
    results = [benchmark_one(ne, root, output, count=count, seed=seed) for ne in sizes]
    rows = [dict(ne=r["ne"], flux=r["flux"], orbitals=r["orbitals"], dimension=r["dimension"],
                 full_fixed_N_dimension=r["full_fixed_N_dimension"], ground_energy_V1=r["ground_energy_V1"],
                 neutral_gap_V1=r["neutral_gap_V1"], zero_modes=r["zero_modes_in_saved_spectrum"],
                 first_excited_L=r["states"][1]["L"], ground_L2=r["ground"]["L2"],
                 maximum_eigenpair_residual_V1=r["validation"]["maximum_eigenpair_residual_V1"],
                 saved_eigenpairs=r["saved_eigenpairs"], lowest_L0_excitation_gap_V1=r["lowest_L0_excitation_gap_V1"],
                 diagonalization_seconds=r["diagonalization_seconds"])
            for r in results]
    _csv(output / "summary.csv", rows)
    comparisons = []
    if compare_models and 8 in sizes:
        comparisons = compare_pilot(root, output, next(r for r in results if r["ne"] == 8))
    report = dict(format_version=1, cases=rows, nncomparisons=comparisons, sizes=list(sizes), requested_eigenpairs=count,
                  seed=seed, eigenvalue_zero_tolerance_V1=ENERGY_TOL,
                  degeneracy_tolerance_V1=DEGENERACY_TOL, total_seconds=time.perf_counter() - started,
                  machine=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                               platform=platform.platform()),
                  source_sha256=sha256(Path(__file__)),
                  neural_checkpoint_comparisons=len(comparisons),
                  scope="Fresh complete-fixed-M-sector low-energy diagonalization, not a larger-system approximation; no ED used in training")
    _json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("results/ed_benchmarks"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--eigenpairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--skip-model-comparison", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    run_benchmarks(args.root, output, sizes=args.sizes, count=args.eigenpairs, seed=args.seed,
                   compare_models=not args.skip_model_comparison)


if __name__ == "__main__":
    main()
