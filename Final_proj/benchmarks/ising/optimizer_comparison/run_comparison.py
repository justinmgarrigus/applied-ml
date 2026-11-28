"""STAGE 1 (learning rate only; outputs are stage1_*). Matched Ising comparison of SPRING, VMC-SGD, Adam and RMSProp.

Phase `tune` sweeps learning rates on tuning seeds; phase `final` runs the
selected rates on separate evaluation seeds. Same Hamiltonian, ansatz, batch,
step budget, norm cap and initialization seeds for every method. Exact ED
energy is used for evaluation only. Output directories must not already exist.
"""
import argparse, json, multiprocessing as mp, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared"))
OUT = Path(__file__).resolve().parent
GRID = {"spring": [0.01, 0.03, 0.1, 0.3], "sgd": [0.01, 0.03, 0.1, 0.3],
        "adam": [0.003, 0.01, 0.03, 0.1], "rmsprop": [0.001, 0.003, 0.01, 0.03]}
COMMON = ["--n", "6", "--hidden", "12", "--steps", "250", "--batch-size", "128",
          "--max-norm", "0.1", "--log-every", "25"]


def job(spec):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import io, contextlib, ising_backend as b
    method, lr, seed, phase = spec
    out = OUT / f"stage1_{phase}" / f"{method}_lr{lr}_seed{seed}"
    args = b.parser().parse_args(COMMON + ["--optimizer", method, "--learning-rate", str(lr),
                                           "--seed", str(seed), "--output", str(out)] +
                                 (["--mu", "0.9"] if method == "spring" else []))
    with contextlib.redirect_stdout(io.StringIO()):
        r = b.run(args)
    return {"method": method, "lr": lr, "seed": seed, "phase": phase,
            "final_error": r["final"]["energy_error"], "final_fidelity": r["final"]["ground_state_fidelity"],
            "seconds": r["training_seconds_including_jit"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("phase", choices=("tune", "final"))
    cli.add_argument("--seeds", type=int, nargs="+", required=True)
    cli.add_argument("--rates", help="JSON {method: lr} for the final phase")
    a = cli.parse_args()
    if a.phase == "tune":
        specs = [(m, lr, s, "tune") for m, g in GRID.items() for lr in g for s in a.seeds
                 if not (OUT / "stage1_tune" / f"{m}_lr{lr}_seed{s}").exists()]  # extend a sweep without overwriting
    else:
        rates = json.loads(a.rates)
        specs = [(m, lr, s, "final") for m, lr in rates.items() for s in a.seeds]
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(job, specs)
    target = OUT / f"stage1_{a.phase}_results.json"
    if target.exists() and a.phase == "tune":  # merge sweep extensions
        rows = json.loads(target.read_text()) + rows
    (target).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(rows)} runs written to {target}")
