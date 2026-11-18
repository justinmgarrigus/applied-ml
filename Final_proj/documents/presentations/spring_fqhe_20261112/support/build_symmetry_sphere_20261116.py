"""Build the boundary-free Haldane-sphere schematic for the revised slides.

This is a geometry illustration, not an electron density or numerical result.
It reads no simulation or ED arrays and leaves archived physics assets intact.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spring_symmetry_sphere_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
BLUE = "#35577F"
TEAL = "#158371"
INK = "#18334D"
LIGHT = "#EFF4F7"


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    ax.set_aspect("equal")
    ax.set_xlim(-1.60, 1.66)
    ax.set_ylim(-1.91, 1.45)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1, facecolor=LIGHT, edgecolor=BLUE, lw=1.6))
    ax.add_patch(Ellipse((0, 0), .78, 2, fill=False, edgecolor=BLUE,
                         lw=.8, alpha=.32))
    ax.add_patch(Ellipse((0, 0), 2, .56, fill=False, edgecolor=BLUE,
                         lw=.8, alpha=.32, linestyle=(0, (3, 3))))
    for angle in (32, 72, 117, 158, 205, 250, 294, 335):
        v = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        ax.annotate("", xy=1.27*v, xytext=1.005*v,
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2,
                                    mutation_scale=9))
    ax.plot(0, 0, "o", color=INK, ms=4)
    ax.annotate("monopole", xy=(-.025, .025), xytext=(-.35, .60),
                color=INK, fontsize=12, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=.75))
    ax.text(0, 1.34, r"North: $m=+Q$", fontsize=13, color=INK, ha="center")
    ax.text(0, -1.32, r"South: $m=-Q$", fontsize=13, color=INK, ha="center")
    ax.text(-1.55, 1.03, r"$\Phi=N_\phi\Phi_0$", fontsize=14, color=BLUE,
            ha="left")
    ax.text(1.00, .84, r"radial $B$", fontsize=12, color=TEAL, ha="left")
    ax.text(.17, -.44, r"$R_s=\sqrt{Q}\,\ell_B$", fontsize=14,
            color=INK, ha="center")
    ax.text(0, -1.60, r"$N_\phi=18,\quad Q=9,\quad 19$ LLL orbitals",
            fontsize=12.3, color=INK, ha="center")
    ax.text(0, -1.84, r"Rotational symmetry: $[H,L^2]=0$",
            fontsize=12.3, color=TEAL, ha="center")
    outputs = []
    for ext in ("pdf", "svg", "png"):
        path = OUT / f"haldane_sphere_symmetry.{ext}"
        fig.savefig(path, transparent=True, dpi=240, bbox_inches="tight", pad_inches=.04)
        outputs.append(path)
    plt.close(fig)
    metadata = dict(
        scope="Equatorial orthographic Haldane-sphere geometry schematic; no density or simulation data",
        sphere_has_no_boundary=True,
        projected_edge_potential_present=False,
        conventions=dict(flux="N_phi=2Q", radius="sqrt(Q)*ell_B",
                         orbital_order="m=-Q,...,+Q; north localized m=+Q"),
        illustrated_system=dict(N_phi=18, Q=9, LLL_orbitals=19,
                                interaction_rotationally_invariant=True),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        no_training=True,
        no_ED=True,
    )
    (OUT / "symmetry_sphere_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Wrote haldane_sphere_symmetry.pdf/.svg/.png; no numerical calculation.")


if __name__ == "__main__":
    main()
