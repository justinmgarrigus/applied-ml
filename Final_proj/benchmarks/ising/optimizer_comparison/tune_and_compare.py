"""STAGE 2: wider tuning (learning rate, update-norm cap, Adam b2, RMSProp decay,
SPRING damping, SA temperature/proposal schedule) and the final comparison.

`tune`: tuning seeds only (default 101 102 103); every trial is recorded in
stage2_tune_results.json and the lowest-mean-error config per method is written
to stage2_selected.json. `final`: runs those configs on separate seeds.
Raw run directories go to --workdir (outside the repository). ED is used only
for post-run evaluation. All methods: N=6, hidden=12, J=h=1.
"""
import argparse, itertools, json, multiprocessing as mp, os, sys, io, contextlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared"))
HERE = Path(__file__).resolve().parent
NO_CAP = 1e3          # effectively no update-norm cap
SA_STEPS = 10_000     # SA budget: best budget in SA/Ising/results/README.md
CAPS = [0.03, 0.1, 0.3, NO_CAP]


def first_order_grid():
    g = []
    g += [("sgd", dict(lr=lr, max_norm=c)) for lr in (0.03, 0.1, 0.3, 1.0) for c in CAPS]
    g += [("adam", dict(lr=lr, max_norm=c, beta2=b)) for lr in (0.003, 0.01, 0.03, 0.1)
          for c in CAPS for b in (0.9, 0.99, 0.999)]
    g += [("rmsprop", dict(lr=lr, max_norm=c, decay=d)) for lr in (0.001, 0.003, 0.01, 0.03)
          for c in CAPS for d in (0.9, 0.99, 0.999)]
    g += [("spring", dict(lr=lr, max_norm=c, damping=dm)) for lr in (0.03, 0.1, 0.3)
          for c in CAPS for dm in (1e-4, 1e-3, 1e-2)]
    return g


def sa_grid(count=48, seed=0):
    rng = np.random.default_rng(seed)  # random search, log-uniform
    g = []
    for _ in range(count):
        t0 = 10 ** rng.uniform(-3, 0)
        tf = min(10 ** rng.uniform(-6, -3), t0)
        s0 = 10 ** rng.uniform(-2, -0.3)
        sf = s0 * 10 ** rng.uniform(-2, 0)
        g.append(("sa", dict(temperature=round(t0, 6), final_temperature=round(tf, 8),
                             proposal_scale=round(s0, 5), final_proposal_scale=round(sf, 6))))
    return g


def job(spec):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    method, params, seed, phase, workdir = spec
    tag = "_".join(f"{k}{v:g}" for k, v in params.items())
    out = Path(workdir) / phase / f"{method}_{tag}_seed{seed}"
    with contextlib.redirect_stdout(io.StringIO()):
        if method == "sa":
            import ising_annealing as sa
            a = sa.parser().parse_args(
                ["--n", "6", "--hidden", "12", "--steps", str(SA_STEPS), "--seed", str(seed),
                 "--search-seed", str(seed + 1), "--output", str(out)] +
                sum([[f"--{k.replace('_', '-')}", str(v)] for k, v in params.items()], []))
            r = sa.run(a, out.parent)
            row = dict(final_error=r["final"]["energy_error"], best_error=r["best"]["energy_error"],
                       final_fidelity=r["best"]["fidelity"], seconds=r["training_seconds"])
        else:
            import ising_backend as b
            flags = ["--n", "6", "--hidden", "12", "--steps", "250", "--batch-size", "128",
                     "--log-every", "25", "--optimizer", method, "--learning-rate", str(params["lr"]),
                     "--max-norm", str(params["max_norm"]), "--seed", str(seed), "--output", str(out)]
            if method == "spring":
                flags += ["--mu", "0.9", "--damping", str(params["damping"])]
            if method == "adam":
                flags += ["--adam-beta2", str(params["beta2"])]
            if method == "rmsprop":
                flags += ["--rmsprop-decay", str(params["decay"])]
            r = b.run(b.parser().parse_args(flags))
            row = dict(final_error=r["final"]["energy_error"], best_error=r["final"]["energy_error"],
                       final_fidelity=r["final"]["ground_state_fidelity"],
                       seconds=r["training_seconds_including_jit"])
    if phase != "final":  # keep only the metrics for tuning trials
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    return dict(method=method, params=params, seed=seed, phase=phase, **row)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("phase", choices=("tune", "final"))
    cli.add_argument("--seeds", type=int, nargs="+", required=True)
    cli.add_argument("--workdir", required=True)
    a = cli.parse_args()
    if a.phase == "tune":
        specs = [(m, p, s, "tune", a.workdir) for m, p in first_order_grid() + sa_grid() for s in a.seeds]
    else:
        sel = json.loads((HERE / "stage2_selected.json").read_text())
        specs = [(m, sel[m]["params"], s, "final", a.workdir) for m in sel for s in a.seeds]
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(job, specs, chunksize=1)
    (HERE / f"stage2_{a.phase}_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    if a.phase == "tune":
        score = {}
        for r in rows:
            score.setdefault((r["method"], json.dumps(r["params"], sort_keys=True)), []).append(r["best_error"])
        sel = {}
        for (m, p), v in score.items():
            if m not in sel or np.mean(v) < sel[m]["mean_best_error"]:
                sel[m] = dict(params=json.loads(p), mean_best_error=float(np.mean(v)), tuning_seeds=a.seeds)
        (HERE / "stage2_selected.json").write_text(json.dumps(sel, indent=2) + "\n")
        print(json.dumps(sel, indent=2))
    print(f"{len(rows)} runs")
