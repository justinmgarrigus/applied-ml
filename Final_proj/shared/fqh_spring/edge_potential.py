"""Diagonal sphere-pole orbital potential U(n_0+n_{N_phi}).

The occupations use a=m+N_phi/2 in the canonical Slater basis. Penalizing the
two extreme orbitals preserves particle number, total Lz and pole interchange,
but breaks full SO(3) rotation symmetry and does not commute with total L^2.
This is a finite one-body potential; it does not remove orbitals or project
the many-body state. U is measured in V1 units.
"""
from __future__ import annotations

import math
from numbers import Real


def validate_edge_strength(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError('edge_strength must be finite and nonnegative')
    return float(value)


def make_edge_local_energy(interaction_local, edge_strength):
    """Add a diagonal potential to a matrix-free interaction-local evaluator.

    ``energy`` is the total physical energy. Preserve the original channel and
    L2 local values, while exposing the interaction and one-body contributions
    separately. U=0 returns the original callable, preserving its arithmetic
    and dictionary exactly.
    """
    strength = validate_edge_strength(edge_strength)
    if strength == 0:
        return interaction_local

    import jax

    @jax.jit
    def evaluate(params, occupations):
        values = interaction_local(params, occupations)
        edge_occupation = occupations[:, 0] + occupations[:, -1]
        edge_energy = strength * edge_occupation
        return values | dict(energy=values['energy'] + edge_energy,
                             interaction_energy=values['energy'],
                             edge_energy=edge_energy, edge_occupation=edge_occupation)

    return evaluate
