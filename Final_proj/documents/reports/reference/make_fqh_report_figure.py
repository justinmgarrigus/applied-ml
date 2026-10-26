"""Analytical illustration for the reading report; no neural or ED results."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 10, "mathtext.fontset": "stix"})
xi = 0.25
x = np.linspace(-0.65, 0.65, 401)
y = np.linspace(-0.55, 0.55, 341)
xx, yy = np.meshgrid(x, y)
z = xx + 1j * yy
psi = z * (z * z - xi * xi)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
im = axes[0].pcolormesh(x, y, np.angle(psi), cmap="twilight", vmin=-np.pi, vmax=np.pi, shading="auto", rasterized=True)
axes[0].scatter([-xi, 0, xi], [0, 0, 0], facecolor="white", edgecolor="black", s=35, zorder=5)
axes[0].set(xlabel=r"$\mathrm{Re}\,z$ (illustrative units)", ylabel=r"$\mathrm{Im}\,z$", title=r"Local model: $\psi=z(z^2-\xi^2)$", aspect="equal")
cb = fig.colorbar(im, ax=axes[0], shrink=0.85, ticks=[-np.pi, 0, np.pi])
cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
cb.set_label(r"$\arg\psi$")
r = np.geomspace(0.008, 3, 400)
slope = 1 + 2 * r**2 / (r**2 + xi**2)
axes[1].semilogx(r, slope, color="#215A78", lw=2, label="split zeros; probe along imaginary axis")
axes[1].axhline(3, color="#D77A2B", ls="--", label=r"cubic zero: $\psi=z^3$")
axes[1].axhline(1, color="0.6", ls=":")
axes[1].axvline(xi, color="0.6", ls=":")
axes[1].set(xlabel=r"$r$ (illustrative units)", ylabel=r"$\partial\log|\psi(ir)|/\partial\log r$", title="Linear-to-cubic crossover", ylim=(0.85, 3.16))
axes[1].legend(loc="lower right", fontsize=8)
fig.savefig(OUT / "fqh_zero_splitting_schematic.pdf")
fig.savefig(OUT / "fqh_zero_splitting_schematic.png", dpi=180)
plt.close(fig)
