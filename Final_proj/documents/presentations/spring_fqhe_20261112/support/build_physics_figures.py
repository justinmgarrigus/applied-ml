"""Rebuild the physics illustrations from the exact current wall convention.

Only pre-existing code and ED reference arrays are read. No training is run.
Outputs are transparent vector PDF/SVG and PNG assets for the slide deck.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, Polygon
import numpy as np


SUPPORT = Path(__file__).resolve().parent
FINAL_PROJ = next(p for p in SUPPORT.parents if (p / "SPRING_Haldane_FQHE").is_dir())
PROJECT = FINAL_PROJ / "SPRING_Haldane_FQHE"
sys.path.insert(0, str(PROJECT / "src"))

from fqh_spring.north_droplet import profile_a_epsilon, profile_sha256


BLUE = "#31547d"
TEAL = "#16856f"
ORANGE = "#de940e"
INK = "#243446"
LIGHT = "#e9f0f6"
F = 18
U = 10.0
XC = 0.88


def save_asset(fig, path):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix("." + ext), transparent=True, dpi=240,
                    bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def sphere_schematic(output):
    """Front-view sphere: the cap boundary is the exact projected latitude.

    Latitude circles are seen edge-on in this equatorial orthographic view.
    The shaded projected segment is not a density map. Twelve percent refers
    to spherical surface area, not to its projected two-dimensional area.
    """
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.69)
    ax.set_ylim(-1.59, 1.48)
    ax.axis("off")

    zc = 1.0 - 2.0 * XC
    rc = np.sqrt(1.0 - zc**2)
    theta = np.arccos(zc)
    ax.add_patch(Circle((0, 0), 1, facecolor=LIGHT, edgecolor=BLUE, lw=1.65, zorder=1))
    ax.add_patch(Ellipse((0, 0), .80, 2, fill=False, edgecolor=BLUE,
                         lw=.8, alpha=.32, zorder=2))
    ax.plot([-1, 1], [0, 0], color=BLUE, ls=(0, (3, 3)), alpha=.24, lw=.8, zorder=2)

    polar = np.linspace(theta, 2 * np.pi - theta, 240)
    cap = np.column_stack((np.sin(polar), np.cos(polar)))
    ax.add_patch(Polygon(cap, closed=True, facecolor=ORANGE,
                         edgecolor="none", alpha=.53, zorder=3))
    ax.plot([-rc, rc], [zc, zc], color="#b87505", lw=1.55, zorder=4)

    # Radial field lines projected in the meridian plane. Their arrowheads
    # point outwards and are schematic; they are not orbital or density data.
    for degrees in (35, 70, 122, 160, 205):
        a = np.deg2rad(degrees)
        vector = np.array([np.cos(a), np.sin(a)])
        ax.annotate("", xy=1.24 * vector, xytext=1.005 * vector,
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.2,
                                    mutation_scale=9), zorder=5)

    ax.plot([0, 0], [0, .83], color=INK, lw=.9, alpha=.55, zorder=5)
    ax.plot([0, rc], [0, zc], color=INK, lw=.9, alpha=.55, zorder=5)
    ax.add_patch(Arc((0, 0), .73, .73, theta1=90 - np.rad2deg(theta), theta2=90,
                     color=INK, lw=1, zorder=6))
    ax.plot(0, 0, "o", color=INK, ms=3.2, zorder=7)
    ax.text(.40, .15, r"$\theta_c$", fontsize=14, color=INK, ha="left", va="center")

    ax.text(0, 1.34, r"North: $m=+Q$", fontsize=13, color=INK, ha="center")
    ax.text(0, -1.28, r"South: $m=-Q$", fontsize=13, color=INK, ha="center")
    ax.text(-1.52, 1.03, r"$\Phi=F\Phi_0$", fontsize=14, color=BLUE, ha="left")
    ax.text(1.04, .91, r"radial $B$", fontsize=12, color=TEAL, ha="left")
    ax.annotate("monopole", xy=(-.045, .025), xytext=(-.73, .48),
                fontsize=11.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=.7, alpha=.65), zorder=8)
    ax.annotate(r"$U\,\Theta(x-x_c)$" + "\n" + r"$x_c=0.88$" + "\n12% of sphere area",
                xy=(.33, -.89), xytext=(.82, -.62), fontsize=11.5,
                color="#945d00", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color="#945d00", lw=.9),
                zorder=8)
    ax.text(0, -1.51, r"$x=\sin^2(\theta/2)$", fontsize=14, color=INK, ha="center")
    save_asset(fig, output / "haldane_sphere_wall")


def wall_profile(output, epsilon):
    a = np.arange(F + 1)
    m = a - F / 2
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.plot(m, epsilon, color=BLUE, marker="o", markersize=5.6,
            markerfacecolor=BLUE, markeredgewidth=.6, markeredgecolor="white", lw=1.7)
    ax.set_xlabel(r"Orbital angular momentum $m=a-Q$", fontsize=16, color=INK)
    ax.set_ylabel(r"Projected wall energy $\epsilon_m/V_1$", fontsize=16, color=INK)
    ax.set_xlim(-9.4, 9.4)
    ax.set_ylim(-.20, 10.0)
    ax.set_xticks([-9, -6, -3, 0, 3, 6, 9])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="both", labelsize=14, colors=INK, length=4)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(.9)
    ax.grid(axis="y", color="#dce2e9", linewidth=.7)
    fig.tight_layout(pad=.55)
    save_asset(fig, output / "projected_wall_profile")


def main():
    output = SUPPORT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    epsilon = profile_a_epsilon(F, U, XC)
    reference = PROJECT / "results/ne7_gg_edge_20261111/references/wall10"
    metadata = json.loads((reference / "metadata.json").read_text())
    digest = profile_sha256(epsilon)
    if digest != metadata["orbital_profile_sha256"]:
        raise ValueError("Wall profile hash differs from saved ED benchmark")
    if not np.array_equal(epsilon, np.load(reference / "epsilon_a.npy")):
        raise ValueError("Wall profile array differs from saved ED benchmark")
    sphere_schematic(output)
    wall_profile(output, epsilon)
    with (output / "projected_wall_profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["a", "m", "k_outward", "epsilon_over_V1"])
        for a, value in enumerate(epsilon):
            writer.writerow([a, a - F / 2, F - a, float(value)])
    source = PROJECT / "src/fqh_spring/north_droplet.py"
    record = dict(flux=F, Q=F / 2, orbitals=F+1, confinement_strength_V1=U,
                  wall_x=XC, polar_boundary_degrees=float(np.degrees(np.arccos(1-2*XC))),
                  penalized_sphere_area_fraction=1-XC,
                  orbital_order="a ascending; m=a-Q; north has m=+Q",
                  formula="epsilon_a=U*betaincc(F-a+1,a+1,wall_x)",
                  orbital_profile_sha256=digest,
                  reference_epsilon_array_exactly_equal=True,
                  profile_implementation=str(source),
                  profile_implementation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  figure_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sphere_scope="Equatorial orthographic schematic, not electron density; area fraction is spherical, not projected",
                  artifacts=["haldane_sphere_wall.pdf", "haldane_sphere_wall.png", "haldane_sphere_wall.svg",
                             "projected_wall_profile.pdf", "projected_wall_profile.png", "projected_wall_profile.svg",
                             "projected_wall_profile.csv"])
    (output / "physics_figures_metadata.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"Wrote two transparent scientific assets; wall profile SHA256={digest}")


if __name__ == "__main__":
    main()
