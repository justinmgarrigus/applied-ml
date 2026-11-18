"""Isolated normalized spherical V5 extension; preserves the archived operator.

P_R projects an orbital pair onto J=N_phi-R. The base implementation already
provides normalized P1 and P3. This subclass adds P5 from the same pair L^2
blocks and reuses the base canonical-Fock connection signs unchanged.
"""
from __future__ import annotations

import numpy as np
from fqh_spring.physics import SphereOperator


class SphereOperatorV5(SphereOperator):
    """H/V1 = sum_(i<j) [P1(ij) + v3 P3(ij) + v5 P5(ij)]."""

    def __init__(self, ne, flux=None, v3=0., v5=0., two_lz=0):
        self.v5 = float(v5)
        if not np.isfinite(self.v5):
            raise ValueError('v5 must be finite')
        super().__init__(ne, flux=flux, v3=v3, two_lz=two_lz)

    def _build_pair_blocks(self):
        super()._build_pair_blocks()
        if self.v5 != 0. and self.flux < 5:
            raise ValueError('Nonzero v5 requires flux >= 5')
        self.channel_blocks[5] = {}
        angular = self.flux-5
        for pair_sum, l2 in self.pair_l2_blocks.items():
            m = pair_sum-self.flux
            projector = np.zeros_like(l2)
            if angular >= 0 and abs(m) <= angular:
                eigenvalues, eigenvectors = np.linalg.eigh(l2)
                selected = np.flatnonzero(np.isclose(eigenvalues, angular*(angular+1),
                    rtol=2e-12, atol=2e-10))
                if len(selected) != 1:
                    raise ArithmeticError('Could not resolve normalized R=5 pair channel')
                vector = eigenvectors[:, selected[0]]
                projector = np.outer(vector, vector)
            self.channel_blocks[5][pair_sum] = projector
        # At v5=0 leave every base compact interaction array bitwise untouched.
        if self.v5 == 0.:
            return
        for pair_sum, projector in self.channel_blocks[5].items():
            self.pair_blocks[pair_sum] = self.pair_blocks[pair_sum]+self.v5*projector
        self.pair_coefficients.fill(0.)
        for global_row, (a, b) in enumerate(self.pairs):
            pairs = self.pairs_by_sum[int(a+b)]
            local_row = int(np.flatnonzero(np.all(pairs==(a,b), axis=1))[0])
            self.pair_coefficients[global_row, :len(pairs)] = self.pair_blocks[int(a+b)][:,local_row]
