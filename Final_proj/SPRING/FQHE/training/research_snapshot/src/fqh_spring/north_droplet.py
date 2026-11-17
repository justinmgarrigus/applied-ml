"""North-centered fixed-K sphere sector and projected real-space cap wall.

The existing Slater convention is a=m+F/2. North-pole orbitals therefore have
k=F-a=0, and total outward degree is K=sum(k*n_a)=N*F/2-M. The real-space
coordinate x=sin(theta/2)^2 increases from north to south. Projection of the
finite wall U*Theta(x-x_c) into orbital k gives the beta-distribution survival
probability U*betaincc(k+1,F-k+1,x_c); this is not an orbital cutoff.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real


def north_sector(ne=8, extra_flux=2, delta_k=0):
    for name, value, minimum in (('ne', ne, 2), ('extra_flux', extra_flux, 0), ('delta_k', delta_k, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    ne, extra_flux, delta_k = int(ne), int(extra_flux), int(delta_k)
    flux = 3 * (ne - 1) + extra_flux
    compact_k = 3 * ne * (ne - 1) // 2
    outward_k = compact_k + delta_k
    smallest = ne * (ne - 1) // 2
    largest = ne * flux - smallest
    if not smallest <= outward_k <= largest:
        raise ValueError('Requested outward K is outside the fixed-particle orbital range')
    return dict(ne=ne, extra_flux=extra_flux, delta_k=delta_k, flux=flux,
                two_lz=ne * flux - 2 * outward_k, orbitals=flux + 1,
                compact_K=compact_k, outward_K=outward_k, target_sum_a=ne * flux - outward_k)


def validate_cap_wall(confinement_strength, wall_x, profile='cap_step'):
    if (isinstance(confinement_strength, bool) or not isinstance(confinement_strength, Real)
            or not math.isfinite(confinement_strength) or confinement_strength < 0):
        raise ValueError('confinement_strength must be finite and nonnegative, in V1 units')
    if isinstance(wall_x, bool) or not isinstance(wall_x, Real) or not math.isfinite(wall_x) or not 0 < wall_x < 1:
        raise ValueError('wall_x must be finite and strictly between 0 and 1')
    if profile != 'cap_step':
        raise ValueError('profile must be cap_step: projected U*Theta(sin(theta/2)^2-wall_x)')


def profile_a_epsilon(flux, confinement_strength=10., wall_x=.88, profile='cap_step'):
    """Return epsilon in existing ascending-a order, with beta survival tails."""
    validate_cap_wall(confinement_strength, wall_x, profile)
    if isinstance(flux, bool) or not isinstance(flux, Integral) or flux < 1:
        raise ValueError('flux must be a positive integer')
    import numpy as np
    from scipy.special import betaincc

    k = int(flux) - np.arange(int(flux) + 1)
    return np.asarray(float(confinement_strength) * betaincc(k + 1, int(flux) - k + 1, float(wall_x)), dtype=np.float64)


def linear_a_epsilon(flux, intercept=0., slope=1.):
    """Diagnostic affine-in-k profile; its fixed-N,K energy is constant."""
    import numpy as np

    return np.asarray(intercept + slope * (flux - np.arange(flux + 1)), dtype=np.float64)


def profile_sha256(epsilon_a):
    """Hash raw canonical little-endian float64 orbital energies, not JSON."""
    import numpy as np

    values = np.asarray(epsilon_a, dtype='<f8')
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('epsilon_a must be a finite real orbital vector')
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_confinement_local_energy(interaction_local, epsilon_a):
    """Add the projected finite one-body wall without an ED matrix or labels."""
    import numpy as np

    epsilon_a = np.asarray(epsilon_a, dtype=np.float64)
    if epsilon_a.ndim != 1 or len(epsilon_a) < 2 or not np.isfinite(epsilon_a).all():
        raise ValueError('epsilon_a must be a finite one-dimensional orbital vector')
    if not np.any(epsilon_a):
        return interaction_local
    import jax
    import jax.numpy as jnp

    epsilon = jnp.asarray(epsilon_a)

    @jax.jit
    def evaluate(params, occupations):
        values = interaction_local(params, occupations)
        confinement = occupations @ epsilon
        return values | dict(energy=values['energy'] + confinement,
                             interaction_energy=values['energy'], confinement_energy=confinement)

    return evaluate
