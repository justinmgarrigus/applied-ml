"""Import the shared backend independently of the caller's working directory."""
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ising-matplotlib"))
SHARED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHARED / "vendor" / "jaqmc_spring" / "src"))
sys.path.insert(0, str(SHARED))
