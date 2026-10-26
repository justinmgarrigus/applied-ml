"""Rebuild the cross-model ED tables and scientific figure from saved results.

Run after each model's ED benchmark command. This script reads numerical
artifacts only; it neither trains a network nor runs an eigensolver.
Requires NumPy and Matplotlib (verified with Matplotlib 3.11.2).
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fqh_ed_matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path(__file__).resolve().parent
    project = output.parent
    inputs = {
        "fqhe": project / "SPRING_Haldane_FQHE/results/ed_benchmarks/summary.json",
        "ising": project / "SPRING_Ising_test/benchmarks/ed/summary.json",
    }
    fqhe, ising = (json.loads(inputs[key].read_text()) for key in ("fqhe", "ising"))
    fqhe_rows = sorted(fqhe["cases"], key=lambda row: row["ne"])
    ising_rows = sorted(ising["records"], key=lambda row: (row["n"], row["h"]))
    critical = [row for row in ising_rows if row["J"] == row["h"] == 1]
    best = min(fqhe["nncomparisons"], key=lambda row: row["energy_V1"])
    spin = ising["neural_comparison"]
    summary = {
        "physics": {
            "fqhe": "Sphere LLL, spin polarized, Nphi=3(N-1), M=0, normalized pure V1 pair projectors",
            "ising": "Periodic Pauli H=-J sum ZZ-h sum X, J=1; full Hilbert space resolved into spin-flip parity blocks",
        },
        "input_sha256": {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in inputs.values()},
        "fqhe": fqhe_rows,
        "ising": ising_rows,
        "neural_comparisons": {"best_retained_fqhe": best, "retained_ising": spin},
        "no_new_training": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    n = np.array([row["ne"] for row in fqhe_rows])
    axes[0].plot(n, [row["neutral_gap_V1"] for row in fqhe_rows], "o-",
                 color="#185f91", label="Neutral excitation gap")
    axes[0].plot(n, [row["ground_energy_V1"] for row in fqhe_rows], "s--",
                 color="#555555", label="Ground energy (numerical zero)")
    axes[0].set(xlabel="Electrons N", ylabel="Energy / V₁", xticks=n,
                title="FQHE: pure V₁ at Laughlin flux", ylim=(-0.04, 0.78))
    n = np.array([row["n"] for row in critical])
    axes[1].plot(n, [row["full_gap"] for row in critical], "o-",
                 color="#185f91", label="Full excitation gap")
    axes[1].plot(n, [row["even_parity_gap"] for row in critical], "s--",
                 color="#ad5624", label="Even spin-flip parity gap")
    axes[1].set(xlabel="Spins N", ylabel="Excitation energy / J", xticks=n,
                title="Ising: periodic chain, h = J = 1", ylim=(0, 3.3))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"ed_benchmarks.{extension}", dpi=220)
    plt.close(fig)

    lines = [
        "# Exact-diagonalization benchmarks", "",
        "These are executed small-system references for the implemented Hamiltonians. "
        "ED uses complete finite Hilbert sectors, with numerical eigenpair checks; "
        "there is no Monte Carlo sampling or new neural training in this benchmark.", "",
        "![Ground energies and excitation gaps](ed_benchmarks.png)", "",
        "## FQHE: Haldane sphere in the LLL", "",
        "The calculation retains the fully spin-polarized lowest Landau level, "
        "flux Nphi=3(N−1), fixed M=Lz=0, and H/V1=sum P_R=1 with normalized pair "
        "projectors. The canonical Slater basis includes fermionic antisymmetry. "
        "All energies below are total energies in V1 units, without an energy shift.", "",
        "| N | Orbitals | Full fixed-N dimension | M=0 dimension | ED E0/V1 | Neutral gap/V1 | First excited L |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fqhe_rows:
        lines.append(f"| {row['ne']} | {row['orbitals']} | {row['full_fixed_N_dimension']:,} | "
                     f"{row['dimension']:,} | {row['ground_energy_V1']:.2e} | "
                     f"{row['neutral_gap_V1']:.10f} | {row['first_excited_L']} |")
    lines += ["", "Every ground state is the unique zero mode to numerical accuracy, has L=0, "
              "and has uniform one-body density. The small negative or positive E0 values "
              "are floating-point roundoff. The positive neutral gap is E1−E0 at fixed N "
              "and flux; it does not indicate a nonzero ground energy. Since all total L "
              "are integer here, M=0 contains a representative of every rotational "
              "multiplet. This is not a quasiparticle/charge gap or an L=0-only gap.", "",
              "The code freshly solves each Hamiltonian. It retains 20 low eigenpairs "
              "(all 18 for N=4), checks degenerate-energy cutoffs, "
              "and verifies each reported L using full-space H and L² residuals. Existing N=6–8 reference files "
              "were not overwritten. Independent highest-weight ladder and fermion "
              "annihilation constructions validate the R=1 Hamiltonian for N=2–4.", "",
              "[FQHE numerical tables, low eigenvectors and diagnostics](../SPRING_Haldane_FQHE/results/ed_benchmarks/README.md)", "",
              "## Ising: periodic transverse-field chain", "",
              "The Hamiltonian is H=−J sum Z_i Z_(i+1)−h sum X_i, with Pauli "
              "eigenvalues ±1 and periodic boundaries. Site i is bit i of the basis "
              "integer; bit 0 has Z_i=−1. ED diagonalizes both spin-flip parity "
              "blocks and merges their complete spectra. All energies below use J=1.", "",
              "| N | h/J | Dimension | ED E0/J | Full gap/J | Even-parity gap/J |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in ising_rows:
        lines.append(f"| {row['n']} | {row['h']:g} | {row['dimension']:,} | "
                     f"{row['ground_energy']:.12f} | {row['full_gap']:.10f} | {row['even_parity_gap']:.10f} |")
    lines += ["", "The full gap includes the opposite-parity excitation. The current "
              "bias-free RBM has even spin-flip parity, so its symmetry-sector "
              "fidelity bound can use the larger even-parity gap. At h=J=1, the "
              "ED results also match the independent finite-ring expressions "
              "E0=−2 csc[π/(2N)], full gap=2 tan[π/(4N)], and even gap=8 sin[π/(2N)].", "",
              "[Ising spectra, observables and saved states](../SPRING_Ising_test/benchmarks/ed/README.md)", "",
              "## Existing neural states compared with ED", "",
              "These are comparisons of retained checkpoints with the appropriate "
              "exact ground state. They are different physical problems and different "
              "ansatzes, not a controlled speed comparison between algorithms. "
              "Variances use the square of the corresponding energy unit.", "",
              "| Retained state | ED E0 | Neural energy | Energy error | Ground-state fidelity | Energy variance |",
              "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| Ising N=6, SPRING step 250 (J units) | {next(r['ground_energy'] for r in critical if r['n']==6):.12f} | "
              f"{spin['energy']:.12f} | {spin['energy_error']:.6g} | {spin['fidelity']:.9f} | {spin['variance']:.6g} |",
              f"| FQHE N=8, selected dense seed {best['seed']} (V1 units) | ~0 | {best['energy_V1']:.12f} | "
              f"{best['energy_error_V1']:.6g} | {best['ground_state_fidelity']:.9f} | {best['residual_variance_V1_squared']:.6g} |", "",
              "The retained Ising state is close to the exact ground state. The FQHE "
              "state still has substantial excited-state weight and nonzero residual "
              "variance; an energy below the first excited eigenvalue does not make "
              "a variational state an eigenstate. All six FQHE pilot checkpoints are "
              "reevaluated in `checkpoint_comparison.csv`, not just the selected best one.", "",
              "For a unique ground state with gap Δ, F≥1−(E−E0)/Δ (up to numerical "
              "reference accuracy). The saved comparisons include this energy-based "
              "lower bound and the actual overlap. A weak bound does not replace "
              "the exact overlap measurement.", "",
              "## Reproduce and use the benchmarks", "",
              "The model-specific READMEs give the ED generation commands. After "
              "generating both suites, rebuild this report and figure from the course root:", "",
              "```bash", "Final_proj/SPRING_Ising_test/.venv/bin/python Final_proj/benchmarks/build_report.py", "```", "",
              "The report builder requires NumPy and Matplotlib; the numerical ED "
              "scripts specify their own requirements. `summary.json` records source "
              "data hashes and the cross-model tables. PDF and PNG versions of the "
              "figure are included. Future N=8 runs can be checked with the "
              "same frozen FQHE reference and evaluator. ED labels/eigenvectors do "
              "not enter SPRING training, and no 200,000-update run was performed here.", "",
              "[Validation record](validation.json) includes the full FQHE and Ising "
              "test-suite results. The independent Hamiltonian check is saved in "
              "[independent_fqhe_validation.json](independent_fqhe_validation.json).", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(f"Wrote {output / 'README.md'} and ED comparison figures")


if __name__ == "__main__":
    main()
