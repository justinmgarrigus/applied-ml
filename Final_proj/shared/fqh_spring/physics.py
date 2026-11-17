"""Matrix-free V1+v3 V3 interactions in one Haldane-sphere Lz sector.

Only two-particle blocks and dynamic-programming completion counts are stored.
The many-body basis and Hamiltonian are never enumerated.  Orbital a has
2m=2a-flux; Fock creation operators are ordered by ascending a.

Pair blocks resolve normalized projectors onto L_pair=flux-R for R=1,3.
The returned many-body matrix elements include canonical fermionic signs and
have the same normalization as independent small-system diagonalization.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from numbers import Integral

import numpy as np


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _randbelow(rng: np.random.Generator, limit: int) -> int:
    """Uniform integer for arbitrary-size Python bounds, without float weights."""
    if limit <= np.iinfo(np.int64).max:
        return int(rng.integers(limit))
    width = limit.bit_length()
    mask = (1 << width) - 1
    nbytes = (width + 7) // 8
    while True:
        value = int.from_bytes(rng.bytes(nbytes), "little") & mask
        if value < limit:
            return value


class SphereOperator:
    """Compact interaction and exact fixed-sector initializer.

    ``connections(n)`` returns the configurations n' and real H[n,n'] for
    H=sum_(i<j)[P_R=1(ij)+v3 P_R=3(ij)].  The diagonal is returned first,
    including when its value is zero.  Off-diagonal duplicates are combined.

    Public compact arrays for accelerator implementations:

    * ``pairs``: [number_of_orbital_pairs, 2] in ascending orbital order.
    * ``pair_index[a,b]``: row of ``pairs`` for a<b, or -1 otherwise.
    * ``pair_destinations``: [number_of_pairs, max_pair_block, 2].
    * ``pair_coefficients`` and ``pair_valid``: corresponding scattering
      coefficients and padding mask.  Padding indices are zero and masked.
    * ``pairs_by_sum`` and ``pair_blocks``: dictionaries of small fixed-pair-M
      bases and their interaction matrices. ``channel_blocks[R]`` resolves the
      individual normalized R=1 or R=3 channel.

    These arrays contain orbital-pair data only, never many-body basis rows.
    """

    def __init__(self, ne: int, flux: int | None = None, v3: float = 0.0,
                 two_lz: int = 0):
        self.ne = _integer(ne, "ne")
        self.flux = (3 * (self.ne - 1) if flux is None
                     else _integer(flux, "flux"))
        self.two_lz = _integer(two_lz, "two_lz")
        self.norb = self.flux + 1
        if self.ne < 2 or self.flux < 1 or self.ne > self.norb:
            raise ValueError("Require ne >= 2, flux >= 1, and ne <= flux+1.")
        self.v3 = float(v3)
        if not np.isfinite(self.v3):
            raise ValueError("v3 must be finite.")
        if self.v3 != 0.0 and self.flux < 3:
            raise ValueError("Nonzero v3 requires flux >= 3.")
        numerator = self.ne * self.flux + self.two_lz
        if numerator % 2:
            raise ValueError("The requested fixed-Lz sector is empty (wrong parity).")
        self.target_sum = numerator // 2

        @lru_cache(maxsize=None)
        def completions(start: int, remaining: int, target: int) -> int:
            if remaining == 0:
                return int(target == 0)
            if remaining < 0 or self.norb - start < remaining:
                return 0
            smallest = remaining * (2 * start + remaining - 1) // 2
            largest = remaining * (2 * self.flux - remaining + 1) // 2
            if target < smallest or target > largest:
                return 0
            return (completions(start + 1, remaining, target)
                    + completions(start + 1, remaining - 1, target - start))

        self._completions = completions
        if self.dimension() == 0:
            raise ValueError("The requested fixed-Lz sector is empty.")
        self._build_pair_blocks()

    def dimension(self) -> int:
        """Exact fixed-Ne,Lz dimension, using integer dynamic programming."""
        return self._completions(0, self.ne, self.target_sum)

    def _build_pair_blocks(self):
        self.pairs = np.asarray(list(combinations(range(self.norb), 2)),
                                dtype=np.int32)
        self.pair_index = np.full((self.norb, self.norb), -1, dtype=np.int32)
        self.pairs_by_sum = {}
        for pair_sum in range(1, 2 * self.flux):
            rows = self.pairs[self.pairs.sum(axis=1) == pair_sum]
            if len(rows):
                self.pairs_by_sum[pair_sum] = rows
        self.channel_blocks = {1: {}, 3: {}}
        self.pair_blocks = {}
        self.pair_l2_blocks = {}
        for pair_sum, pairs in self.pairs_by_sum.items():
            # The ladder maps this pair-M block into pair-M+1.  On ascending
            # two-fermion states an adjacent orbital raise has positive sign.
            target = self.pairs_by_sum.get(pair_sum + 1, np.empty((0, 2), int))
            target_index = {tuple(pair): row for row, pair in enumerate(target)}
            lplus = np.zeros((len(target), len(pairs)), dtype=np.float64)
            for column, (a, b) in enumerate(pairs):
                if a + 1 < b:
                    row = target_index[(a + 1, b)]
                    lplus[row, column] += np.sqrt((self.flux - a) * (a + 1))
                if b < self.flux:
                    row = target_index[(a, b + 1)]
                    lplus[row, column] += np.sqrt((self.flux - b) * (b + 1))
            m = pair_sum - self.flux
            l2 = lplus.T @ lplus + m * (m + 1.0) * np.eye(len(pairs))
            self.pair_l2_blocks[pair_sum] = l2
            eigenvalues, eigenvectors = np.linalg.eigh(l2)
            for relative in (1, 3):
                angular = self.flux - relative
                projector = np.zeros_like(l2)
                if angular >= 0 and abs(m) <= angular:
                    desired = angular * (angular + 1)
                    selected = np.flatnonzero(np.isclose(eigenvalues, desired,
                                                        rtol=2e-12, atol=2e-10))
                    if len(selected) != 1:
                        raise ArithmeticError("Could not resolve a pair angular-momentum channel.")
                    vector = eigenvectors[:, selected[0]]
                    projector = np.outer(vector, vector)
                self.channel_blocks[relative][pair_sum] = projector
            self.pair_blocks[pair_sum] = (
                self.channel_blocks[1][pair_sum]
                + self.v3 * self.channel_blocks[3][pair_sum]
            )

        self.max_pair_block = max(map(len, self.pairs_by_sum.values()))
        npairs = len(self.pairs)
        self.pair_destinations = np.zeros((npairs, self.max_pair_block, 2),
                                          dtype=np.int32)
        self.pair_coefficients = np.zeros((npairs, self.max_pair_block),
                                          dtype=np.float64)
        self.pair_valid = np.zeros((npairs, self.max_pair_block), dtype=bool)
        for global_row, (a, b) in enumerate(self.pairs):
            self.pair_index[a, b] = global_row
            pairs = self.pairs_by_sum[int(a + b)]
            local_row = int(np.flatnonzero(np.all(pairs == (a, b), axis=1))[0])
            count = len(pairs)
            self.pair_destinations[global_row, :count] = pairs
            self.pair_coefficients[global_row, :count] = self.pair_blocks[int(a + b)][:, local_row]
            self.pair_valid[global_row, :count] = True

    def sample_uniform(self, rng: np.random.Generator, count: int) -> np.ndarray:
        """Exactly uniform samples in the sector, with replacement, using DP.

        This initializes a Markov chain; it is not sampling the variational
        density |psi|².  Integer completion counts avoid approximate floating
        probabilities, even when the sector dimension exceeds machine integers.
        """
        count = _integer(count, "count")
        if count < 0:
            raise ValueError("count must be nonnegative.")
        result = np.zeros((count, self.norb), dtype=np.int8)
        for sample in result:
            remaining, target = self.ne, self.target_sum
            for orbital in range(self.norb):
                if remaining == 0:
                    break
                occupied_count = self._completions(orbital + 1, remaining - 1,
                                                   target - orbital)
                empty_count = self._completions(orbital + 1, remaining, target)
                if _randbelow(rng, occupied_count + empty_count) < occupied_count:
                    sample[orbital] = 1
                    remaining -= 1
                    target -= orbital
        return result

    def connections(self, occupation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All nonzero two-body connections and the combined diagonal.

        For a real Hermitian Hamiltonian the generated H[n',n] equals the
        requested row element H[n,n'].  No wavefunction values are evaluated.
        """
        occupation = np.asarray(occupation)
        if occupation.shape != (self.norb,) or not np.all((occupation == 0) | (occupation == 1)):
            raise ValueError("occupation must be a binary vector of length norb.")
        occupied = np.flatnonzero(occupation)
        if len(occupied) != self.ne or int(occupied.sum()) != self.target_sum:
            raise ValueError("occupation is outside the requested Ne,Lz sector.")
        initial_bits = sum(1 << int(a) for a in occupied)
        entries = {initial_bits: 0.0}
        for c, d in combinations(map(int, occupied), 2):
            # B_cd=c_d c_c, acting right to left on the input state.
            after_c = initial_bits ^ (1 << c)
            after_cd = after_c ^ (1 << d)
            annihilation_parity = ((initial_bits & ((1 << c) - 1)).bit_count()
                                   + (after_c & ((1 << d) - 1)).bit_count())
            pair_row = self.pair_index[c, d]
            for (a_np, b_np), coefficient, valid in zip(
                    self.pair_destinations[pair_row],
                    self.pair_coefficients[pair_row], self.pair_valid[pair_row]):
                if not valid or coefficient == 0.0:
                    continue
                a, b = int(a_np), int(b_np)
                if after_cd & ((1 << a) | (1 << b)):
                    continue
                # B†_ab=c†_a c†_b: create b, then a.
                after_b = after_cd | (1 << b)
                final_bits = after_b | (1 << a)
                creation_parity = ((after_cd & ((1 << b) - 1)).bit_count()
                                   + (after_b & ((1 << a) - 1)).bit_count())
                sign = -1.0 if (annihilation_parity + creation_parity) % 2 else 1.0
                entries[final_bits] = entries.get(final_bits, 0.0) + sign * coefficient
        ordered = [initial_bits] + sorted(bits for bits, value in entries.items()
                                          if bits != initial_bits and value != 0.0)
        neighbors = np.array([[(bits >> a) & 1 for a in range(self.norb)]
                              for bits in ordered], dtype=np.int8)
        coefficients = np.array([entries[bits] for bits in ordered], dtype=np.float64)
        return neighbors, coefficients
