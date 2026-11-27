"""Measure Ne8 Cartesian MinSR cost per update on CUDA and CPU."""
import argparse
from pathlib import Path

from harness import Evaluator, make_config, train, write_json

SETTINGS = dict(learning_rate=0.01, damping=0.01, max_norm=0.03)

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    output = cli.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evaluate = Evaluator(8)
    rows = []
    for device, batch in (("gpu", 32), ("gpu", 128), ("gpu", 512), ("cpu", 32), ("cpu", 128)):
        job = output / f"{device}_B{batch}"
        config = make_config(8, 7, 71, 200, 100, dict(SETTINGS, batch=batch), backend=device)
        summary, dev = train(job, config, device=device)
        result = evaluate.run(job, angular_final=False)
        row = dict(device=dev, batch=batch, steps=200,
                   seconds_per_update=summary["training_seconds"] / 200,
                   compile_seconds=summary["current_launch_compile_seconds"],
                   final_energy=result["final"]["energy"], final_fidelity=result["final"]["fidelity"])
        rows.append(row)
        print(row, flush=True)
    write_json(output / "timing.json", rows)
