"""Run the shared Ising model with VMC using RMSProp."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "shared"))
from ising_backend import run_algorithm

if __name__ == "__main__":
    run_algorithm("rmsprop", Path(__file__).resolve().parents[1] / "results")
