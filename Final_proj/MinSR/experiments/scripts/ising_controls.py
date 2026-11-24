"""MinSR damping/radius ablations and isolated Ising process memory profiles."""

import argparse
from dataclasses import asdict, replace
from itertools import product
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from ising_study import (BATCHES, METHODS, PRODUCTION_SEEDS, PRODUCTION_STEPS,
                         Settings, file_hash, run, save_json)


def settings_at(frozen, batch, method):
    return Settings(**frozen[str(batch)][method])


def main(output, frozen_path):
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(frozen_path.read_text())
    save_json(output / "protocol.json", dict(
        parent_frozen_settings_sha256=file_hash(frozen_path),
        harness_sha256=file_hash(Path(__file__)),
        seeds=PRODUCTION_SEEDS, batches=BATCHES, steps=PRODUCTION_STEPS,
        cases={"ablation": "Ising N=6 J=h=1; one setting changed from tuned MinSR",
               "memory_profile": "Ising N=6 and N=10 J=h=1; seed 17, fresh process each"},
        run_order_seed=685, affinity=sorted(os.sched_getaffinity(0))))
    for batch in BATCHES:
        baseline = settings_at(frozen, batch, "MinSR")
        changes = {"damping_low": replace(baseline, damping=baseline.damping/10),
                   "damping_high": replace(baseline, damping=baseline.damping*10),
                   "radius_low": replace(baseline, radius=baseline.radius/3),
                   "radius_high": replace(baseline, radius=baseline.radius*3)}
        jobs = list(product(changes, PRODUCTION_SEEDS))
        random.Random(685 + batch).shuffle(jobs)
        for name, seed in jobs:
            run(output / "ablation" / f"b{batch}" / name / f"s{seed}", name, 6, 1.0,
                batch, "MinSR", seed, changes[name], PRODUCTION_STEPS)
        print(f"MinSR B={batch}: damping and radius ablations complete", flush=True)

    for n, batch, method in product((6, 10), BATCHES, METHODS):
        destination = output / "profiles" / f"n{n}-b{batch}-{method}"
        destination.mkdir(parents=True, exist_ok=False)
        settings = settings_at(frozen, batch, method)
        command = ["/usr/bin/time", "-f", "%e %M", "-o", str(destination / "process_time_rss.txt"),
                   sys.executable, str(Path(__file__).resolve()), "--output", str(destination / "run"),
                   "--profile", str(n), str(batch), method, json.dumps(asdict(settings))]
        with (destination / "process.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("Isolated Ising memory profiles complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-settings", type=Path)
    parser.add_argument("--profile", nargs=4, metavar=("N", "BATCH", "METHOD", "SETTINGS"))
    args = parser.parse_args()
    if args.profile:
        n, batch, method, settings = args.profile
        run(args.output.resolve(), "profile", int(n), 1.0, int(batch), method,
            PRODUCTION_SEEDS[0], Settings(**json.loads(settings)), PRODUCTION_STEPS)
    elif args.frozen_settings:
        main(args.output.resolve(), args.frozen_settings.resolve())
    else:
        parser.error("--frozen-settings is required unless running an individual profile")
