"""Generate proposal workflow and NN examples from verified local records.

Requires NumPy and Matplotlib. No training, ED solve or benchmark mutation.
Run from any directory; outputs are written beside this script.
"""
from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spring_proposal_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
FINAL = HERE.parents[2]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def inventory_database(ising, fqhe, rows_i, rows_f, endpoints, paths):
    """Count saved arrays and logs; never solve a Hamiltonian or train a model."""
    spin_cases, hall_cases = [], []
    for row in ising["records"]:
        folder = FINAL / "SPRING_Ising_test/benchmarks/ed" / row["artifact_directory"]
        basis_path, values_path = folder / "basis_spins.npy", folder / "eigenvalues.npy"
        shape = list(np.load(basis_path, mmap_mode="r").shape)
        eigenvalues = np.load(values_path, mmap_mode="r")
        assert shape == [row["dimension"], row["n"]]
        assert eigenvalues.shape == (row["dimension"],)
        paths[f"ising_basis_{folder.name}"] = basis_path
        paths[f"ising_spectrum_{folder.name}"] = values_path
        spin_cases.append({"n": row["n"], "J": row["J"], "h": row["h"],
                           "decoded_input_shape": shape,
                           "saved_eigenvalues": int(eigenvalues.size)})
    for row in fqhe["cases"]:
        path = FINAL / f"SPRING_Haldane_FQHE/results/ed_benchmarks/ne{row['ne']}/eigenpairs.npz"
        with np.load(path) as data:
            count = int(data["basis_bits"].size)
            eigenpairs = int(data["eigenvalues"].size)
            assert count == row["dimension"]
            assert data["eigenvectors"].shape == (count, eigenpairs)
        paths[f"fqhe_eigenpairs_ne{row['ne']}"] = path
        hall_cases.append({"n": row["ne"],
                           "decoded_input_shape": [count, row["flux"] + 1],
                           "saved_eigenpairs": eigenpairs})
    predictions = []
    for row in fqhe["nncomparisons"]:
        path = FINAL / ("SPRING_Haldane_FQHE/results/ed_benchmarks/model_evaluations/"
                        f"{row['model']}_seed{row['seed']}_coefficients.npy")
        shape = list(np.load(path, mmap_mode="r").shape)
        assert shape == [next(r["dimension"] for r in fqhe["cases"] if r["ne"] == 8)]
        paths[f"fqhe_prediction_{row['model']}_{row['seed']}"] = path
        predictions.append({"model": row["model"], "sampling_seed": row["seed"],
                            "coefficient_shape": shape})
    spin_rows = sum(r["decoded_input_shape"][0] for r in spin_cases)
    hall_rows = sum(r["decoded_input_shape"][0] for r in hall_cases)
    unique_spin_rows = sum({r["n"]: r["decoded_input_shape"][0] for r in spin_cases}.values())
    return {"hamiltonian_instances": len(spin_cases) + len(hall_cases),
            "ising": {"instances": len(spin_cases), "configuration_rows_per_instance_sum": spin_rows,
                      "unique_basis_rows_across_sizes": unique_spin_rows,
                      "saved_eigenvalues": sum(r["saved_eigenvalues"] for r in spin_cases),
                      "cases": spin_cases},
            "fqhe": {"instances": len(hall_cases), "configuration_rows_per_instance_sum": hall_rows,
                     "saved_low_eigenpairs": sum(r["saved_eigenpairs"] for r in hall_cases),
                     "cases": hall_cases, "neural_predictions": predictions},
            "total_configuration_rows_per_instance_sum": spin_rows + hall_rows,
            "decoded_input_entries_per_instance_sum": sum(np.prod(r["decoded_input_shape"]).item()
                                                          for r in spin_cases + hall_cases),
            "nn": {"ising_runs": 1, "fqhe_continuations": len(endpoints),
                   "ising_diagnostic_shape": [len(rows_i), len(rows_i[0])],
                   "fqhe_trajectory_shape": [len(rows_f), len(rows_f[0])],
                   "fqhe_endpoint_shape": [len(endpoints), len(endpoints[0])]},
            "counting_convention": "Configuration rows summed per Hamiltonian instance; the N=6 Ising basis is reused at three fields. FQHE bits are stored packed; shapes describe decoded NN inputs. Diagnostic rows are aggregate records, not saved raw Monte Carlo samples."}


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{extension}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def workflow():
    fig, axis = plt.subplots(figsize=(12, 7.5))
    axis.set(xlim=(0, 14), ylim=(0, 8.7))
    axis.axis("off")
    colors = {"physics": ("#e7eef6", "#24517e"), "network": ("#e5f1fa", "#17658c"),
              "sampling": ("#e3f3ef", "#287968"), "force": ("#fff1d9", "#9b6218"),
              "spring": ("#eee8f8", "#664795"), "ed": ("#f8e9e9", "#9d4545"),
              "plain": ("#f2f3f5", "#535c68")}

    def box(x, y, w, h, title, body, kind="plain", title_size=12, body_size=10):
        face, edge = colors[kind]
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                     fc=face, ec=edge, lw=1.5))
        axis.text(x+w/2, y+h-0.25, title, ha="center", va="top", fontsize=title_size,
                  fontweight="bold", color=edge)
        axis.text(x+w/2, y+(h-0.40)/2, body, ha="center", va="center", fontsize=body_size,
                  linespacing=1.5, color="#192c3b")

    def arrow(start, end, label=None, curve=0, dashed=False, color="#455362"):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                      lw=1.5, color=color, linestyle="--" if dashed else "-",
                                      connectionstyle=f"arc3,rad={curve}"))
        if label:
            middle = ((start[0]+end[0])/2, (start[1]+end[1])/2)
            axis.text(*middle, label, fontsize=9, ha="center", va="bottom", color=color,
                      bbox={"facecolor":"white", "edgecolor":"none", "pad":1.5})

    axis.text(0.15, 8.45, "ONE VARIATIONAL MONTE CARLO LOOP — ALTERNATIVE OPTIMIZERS",
              fontsize=13, weight="bold", color="#233c53")
    box(.15, 6.25, 3.00, 1.70, "1  Physics prior", "Hamiltonian H\nBasis and symmetries\nIsing; Hubbard or FQHE", "physics")
    box(3.65, 6.25, 3.00, 1.70, "2  Neural state", "$x \\mapsto \\log\\psi_\\theta(x)$\nLog amplitude / phase\nTrainable parameters", "network")
    box(7.15, 6.25, 2.85, 1.70, "3  Born sampling", "$p_\\theta(x)\\propto|\\psi_\\theta(x)|^2$\nB configurations\nMixing and sample checks", "sampling", body_size=9.8)
    box(10.50, 6.25, 3.30, 1.70, "4  Energy and scores", "$E_{\\mathrm{loc}}=H\\psi/\\psi$\n$O=\\partial_\\theta\\log\\psi$\nCentered score rows", "force")
    for start, end in [((3.18,7.1),(3.60,7.1)),((6.68,7.1),(7.10,7.1)),((10.03,7.1),(10.45,7.1))]:
        arrow(start,end)
    box(4.1, 2.45, 7.00, 2.65, "5  Optimizer comparison",
        "SGD / Adam      |      SR / MinSR      |      K-FAC\n\nSPRING: sampled geometric solve + previous direction\n"
        "$\\phi_k=\\mu\\phi_{k-1}+A_k^T(A_kA_k^T+\\lambda I)^{-1}(b_k-\\mu A_k\\phi_{k-1})$",
        "spring", body_size=10.3)
    arrow((12.15,6.18),(10.65,5.15), "gradient / geometry")
    box(11.65, 2.80, 2.15, 1.85, "Raw memory", "$\\phi_{k-1}$\nPrior raw direction\nNot MC walker state", "spring", title_size=10.8, body_size=9.8)
    arrow((11.60,3.75),(11.15,3.75))
    arrow((10.60,2.40),(12.65,2.75), "save raw direction", curve=.18, color="#664795")
    box(.15, 2.45, 3.15, 2.65, "6  Parameter update", "$\\theta\\leftarrow\\theta+\\alpha_k\\phi_k$\n"
        "$\\alpha_k=\\min(\\eta_k,\\sqrt{C}/\\|\\phi_k\\|)$\n\nCheckpoint and log", "network", body_size=10)
    arrow((4.05,3.8),(3.35,3.8))
    arrow((1.05,5.16),(4.05,6.20), "next iteration: resample current state", curve=-.22)
    box(.15, .10, 4.35, 1.45, "ED reference database", "Fixed H and low eigenstates\nVerified basis and symmetries", "ed", title_size=11.5)
    box(5.05, .10, 8.75, 1.45, "Independent frozen-state evaluation",
        "Energy error    |    variance    |    fidelity    |    symmetry\nED used only for diagnostics; no target coefficients in training", "ed", title_size=11.5)
    arrow((4.55,.82),(5.0,.82), dashed=True, color="#9d4545")
    arrow((1.75,2.40),(6.00,1.59), "saved neural state", dashed=True, curve=.06, color="#9d4545")
    save(fig,"spring_optimizer_workflow")


def main():
    paths = {
        "fqhe_ed": FINAL / "SPRING_Haldane_FQHE/results/ed_benchmarks/summary.json",
        "ising_ed": FINAL / "SPRING_Ising_test/benchmarks/ed/summary.json",
        "ising_metrics": FINAL / "SPRING_Ising_test/results/spring_n6/metrics.json",
        "ising_trajectory": FINAL / "SPRING_Ising_test/results/spring_n6/training.csv",
        "fqhe_trajectory": FINAL / "SPRING_Haldane_FQHE/results/ne8_architecture_pilot/exact_trajectory.csv",
        "fqhe_endpoints": FINAL / "SPRING_Haldane_FQHE/results/ne8_architecture_pilot/endpoints.csv",
        "validation": FINAL / "benchmarks/validation.json",
    }
    fqhe = json.loads(paths["fqhe_ed"].read_text())
    ising = json.loads(paths["ising_ed"].read_text())
    spin = ising["neural_comparison"]
    best = min(fqhe["nncomparisons"],key=lambda row:row["energy_V1"])
    rows_i, rows_f = read_csv(paths["ising_trajectory"]), read_csv(paths["fqhe_trajectory"])
    end = read_csv(paths["fqhe_endpoints"])
    database_size = inventory_database(ising, fqhe, rows_i, rows_f, end, paths)
    workflow()
    fig, axes = plt.subplots(1,2,figsize=(10.5,3.05),layout="constrained")
    axes[0].semilogy([int(r["step"]) for r in rows_i], [float(r["energy_error"]) for r in rows_i],
                      color="#17658c", lw=1.8)
    axes[0].set(xlabel="Optimizer updates", ylabel="Exact energy error / J",
                title="Ising N=6: retained SPRING run")
    styles={"base":("#68717a","MLP"),"dense":("#17658c","Dense correction"),"attention":("#9a6338","Attention correction")}
    for variant,(color,label) in styles.items():
        subset=[r for r in rows_f if r["variant"]==variant]
        times=sorted(set(int(r["steps"]) for r in subset))
        energies=np.array([[float(r["energy_V1"]) for r in subset if int(r["steps"])==t] for t in times])
        axes[1].plot(times,energies.mean(axis=1),"o-",color=color,lw=1.4,ms=3,label=label)
        axes[1].fill_between(times,energies.min(axis=1),energies.max(axis=1),color=color,alpha=.12)
    axes[1].set(xlabel="Additional optimizer updates", ylabel="Exact energy / V₁ (zoom)",
                title="FQHE N=8: common pretrained source",ylim=(.485,.515))
    axes[1].legend(frameon=False,fontsize=8,loc="upper right")
    axes[1].text(.03,.05,"ED ground energy = 0 (outside zoom)",transform=axes[1].transAxes,fontsize=8,color="#444444")
    for axis in axes:axis.grid(alpha=.2)
    save(fig,"nn_ed_proposal_examples")
    database_rows=[]
    for r in fqhe["cases"]:
        database_rows.append(f"{r['ne']} & ${r['dimension']:,}\\times{r['flux']+1}$ & {r['neutral_gap_V1']:.6f} \\\\")
    ising_rows=[]
    for r in ising["records"]:
        if r["J"]==r["h"]==1:
            ising_rows.append(f"{r['n']} & ${r['dimension']:,}\\times{r['n']}$ & {r['ground_energy']:.6f} & {r['full_gap']:.6f} \\\\")
    means=[]
    for variant,(_,label) in styles.items():
        group=[r for r in end if r["variant"]==variant]
        means.append(f"{label} & {int(group[0]['parameters']):,} & {np.mean([float(r['energy_V1']) for r in group]):.6f} & {np.mean([float(r['fidelity']) for r in group]):.6f} \\\\")
    tex = ["% Generated from verified benchmark records; do not edit numerical rows by hand.",
           "\\newcommand{\\FQHEDatabaseRows}{"+"\n".join(database_rows)+"}",
           "\\newcommand{\\IsingDatabaseRows}{"+"\n".join(ising_rows)+"}",
           "\\newcommand{\\ArchitectureRows}{"+"\n".join(means)+"}",
           "\\newcommand{\\IsingEDZero}{-7.727406610313}",
           f"\\newcommand{{\\FQHEGap}}{{{next(r['neutral_gap_V1'] for r in fqhe['cases'] if r['ne']==8):.9f}}}",
           f"\\newcommand{{\\IsingFidelity}}{{{spin['fidelity']:.9f}}}",
           f"\\newcommand{{\\FQHEFidelity}}{{{best['ground_state_fidelity']:.9f}}}",
           f"\\newcommand{{\\FQHEEnergy}}{{{best['energy_V1']:.9f}}}",
           f"\\newcommand{{\\IsingConfigRows}}{{{database_size['ising']['configuration_rows_per_instance_sum']:,}}}",
           f"\\newcommand{{\\FQHEConfigRows}}{{{database_size['fqhe']['configuration_rows_per_instance_sum']:,}}}",
           f"\\newcommand{{\\TotalConfigRows}}{{{database_size['total_configuration_rows_per_instance_sum']:,}}}",
           f"\\newcommand{{\\IsingLogShape}}{{{len(rows_i)}\\times{len(rows_i[0])}}}",
           f"\\newcommand{{\\FQHELogShape}}{{{len(rows_f)}\\times{len(rows_f[0])}}}",
           f"\\newcommand{{\\FQHEEndpointShape}}{{{len(end)}\\times{len(end[0])}}}"]
    (HERE/"proposal_data.tex").write_text("\n".join(tex)+"\n")
    payload={"inputs":{str(path.relative_to(FINAL)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
             "ising_example":spin,"fqhe_selected_example":best,"database_size":database_size,
             "fqhe_pilot_condition":"1000 additional updates from one shared pretrained state; batch32; sampling seeds71,137",
             "new_training_performed":False,"figure_generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/"proposal_verified_data.json").write_text(json.dumps(payload,indent=2)+"\n")
    print("Wrote workflow, NN examples, numerical TeX rows and source-data hashes.")


if __name__=="__main__":main()
