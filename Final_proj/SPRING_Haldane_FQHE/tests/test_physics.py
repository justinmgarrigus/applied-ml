"""Independent checks of canonical signs, LLL physics and neural local energy."""
from itertools import combinations
from math import comb

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from fqh_spring import model
from fqh_spring.local_energy import make_local_energy
from fqh_spring.physics import SphereOperator
from fqh_spring.references import build_reference, load_reference, sparse_operators


def analytic_v1_matrix(bits, flux):
    """Build R=1 from its analytic pair vector, independent of pair eigensolvers.

    At a+b fixed, the normalized pair vector is proportional to
    (b-a)*sqrt(binomial(flux,a)*binomial(flux,b)). Generic creation/annihilation
    parity is evaluated at each operation, independently of the local kernel.
    """
    pairs = list(combinations(range(flux + 1), 2))
    blocks = {}
    for total in sorted({a + b for a, b in pairs}):
        block_pairs = [(a, b) for a, b in pairs if a + b == total]
        vector = np.asarray([(b - a) * np.sqrt(float(comb(flux, a) * comb(flux, b)))
                             for a, b in block_pairs])
        vector /= np.linalg.norm(vector)
        blocks[total] = (block_pairs, np.outer(vector, vector))
    index = {int(value): row for row, value in enumerate(bits)}
    matrix = np.zeros((len(bits), len(bits)))
    negative_hops = 0
    for column, raw in enumerate(bits):
        initial = int(raw)
        occupied = [a for a in range(flux + 1) if (initial >> a) & 1]
        for c, d in combinations(occupied, 2):
            block_pairs, block = blocks[c + d]
            source = block_pairs.index((c, d))
            for destination, (a, b) in enumerate(block_pairs):
                value, sign, valid = initial, 1, True
                for orbital, create in ((c, False), (d, False), (b, True), (a, True)):
                    present = bool((value >> orbital) & 1)
                    if present == create:
                        valid = False
                        break
                    sign *= (-1)**((value & ((1 << orbital) - 1)).bit_count())
                    value ^= 1 << orbital
                if valid:
                    matrix[index[value], column] += sign * block[destination, source]
                    negative_hops += int(sign < 0)
    assert negative_hops > 0, "Test must exercise fermionic exchange signs"
    return matrix


def test_canonical_many_body_signs_and_sphere_ground_state():
    operator = SphereOperator(4, v3=0.0)
    bits, H, raising, _ = sparse_operators(operator)
    independent = analytic_v1_matrix(bits, operator.flux)
    np.testing.assert_allclose(H.toarray(), independent, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(independent, independent.T, atol=1e-14)
    values, vectors = np.linalg.eigh(independent)
    assert abs(values[0]) < 1e-12
    assert values[1] > 0.1
    assert np.linalg.norm(raising @ vectors[:, 0]) < 1e-11
    L2 = (raising.T @ raising).toarray()
    np.testing.assert_allclose(independent @ L2, L2 @ independent, atol=2e-12)


@pytest.mark.parametrize("kind", ["mlp", "dense", "attention"])
def test_generic_local_energy_includes_canonical_signs_and_all_channels(kind):
    operator = SphereOperator(4, v3=0.2)
    bits, H, raising, _ = sparse_operators(operator)
    _, H1, _, _ = sparse_operators(SphereOperator(4, v3=0.0))
    H3 = (H - H1) / 0.2
    occupations = ((bits[:, None] >> np.arange(operator.norb)) & 1).astype(float)
    params = model.initialize(operator.norb, width=8, layers=2, kind=kind, seed=71)
    # Activate every readout, including newly introduced correction branches.
    params = {name: (value + 0.01 if name.endswith("out") or name.startswith("out") else value)
              for name, value in params.items()}
    for name in params:
        if "out" in name or "_head_" in name:
            params[name] = params[name] + 0.003
    logs = np.asarray(model.batch_log_psi(params, jnp.asarray(occupations)))
    psi = np.exp(logs - logs.real.max())
    actual = make_local_energy(operator, model.log_psi, pair_chunk_size=4)(params, jnp.asarray(occupations))
    expected = dict(energy=(H @ psi) / psi, C1=(H1 @ psi) / psi,
                    C3=(H3 @ psi) / psi, L2=(raising.T @ (raising @ psi)) / psi)
    for channel in expected:
        np.testing.assert_allclose(np.asarray(actual[channel]), expected[channel], atol=3e-11, rtol=3e-11)


def test_portable_reference_integrity_and_density(tmp_path):
    from fqh_spring.evaluate import state_diagnostics
    reference_dir = tmp_path / "reference"
    reference = build_reference(4, reference_dir)
    ground = reference["eigenvectors"][:, np.argmin(reference["eigenvalues"])]
    result = state_diagnostics(ground, reference, angular=True)
    assert abs(result["exact_variational_energy_V1"]) < 1e-12
    assert result["density_RMS_contrast"] < 1e-12
    assert result["exact_L0_probability"] > 1 - 1e-12
    assert abs(result["density_particle_number_integral"] - 4) < 1e-8
    # A copied directory retains relative manifest validity.
    import shutil
    copied = tmp_path / "copied"
    shutil.copytree(reference_dir, copied)
    assert len(load_reference(copied)["bits"]) == len(reference["bits"])
    values = np.load(copied / "eigenvalues.npy")
    np.save(copied / "eigenvalues.npy", values + 0.01)
    with pytest.raises(ValueError, match="integrity"):
        load_reference(copied)
