"""Portable sparse sphere references for independent, small-system evaluation.

Reference Hamiltonians and eigenvectors never enter SPRING training.  A
reference directory uses relative filenames in its integrity manifest, so it
can be copied to another machine without rewriting any paths.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

REFERENCE_FILES = (
    "metadata.json", "basis_bits.npy", "hamiltonian.npz", "raising_operator.npz",
    "eigenvalues.npy", "eigenvectors.npy",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sector_dimension(ne, flux, two_lz=0):
    """Count the complete fixed-N, fixed-Lz sector using integer recursion."""
    if (ne * flux + two_lz) % 2:
        return 0
    target = (ne * flux + two_lz) // 2

    @lru_cache(None)
    def count(start, remaining, total):
        if remaining == 0:
            return int(total == 0)
        if remaining < 0 or flux + 1 - start < remaining:
            return 0
        low = remaining * (2 * start + remaining - 1) // 2
        high = remaining * (2 * flux - remaining + 1) // 2
        if total < low or total > high:
            return 0
        return count(start + 1, remaining, total) + count(start + 1, remaining - 1, total - start)

    return count(0, ne, target)


def enumerate_basis(operator):
    """Enumerate only configurations admitted by the compact DP initializer."""
    bits = []

    def visit(start, remaining, total, value):
        if remaining == 0:
            if total == 0:
                bits.append(value)
            return
        for orbital in range(start, operator.norb):
            if operator._completions(orbital + 1, remaining - 1, total - orbital):
                visit(orbital + 1, remaining - 1, total - orbital, value | (1 << orbital))

    if operator.norb > 63:
        raise ValueError("Sparse references require at most 63 orbitals")
    visit(0, operator.ne, operator.target_sum, 0)
    result = np.asarray(sorted(bits), dtype=np.int64)
    if len(result) != operator.dimension():
        raise ArithmeticError("Basis enumeration disagrees with DP count")
    return result


def sparse_operators(operator):
    """Assemble H and L+ for validation; never called by the training runner."""
    from .physics import SphereOperator

    bits = enumerate_basis(operator)
    occupations = ((bits[:, None] >> np.arange(operator.norb)) & 1).astype(np.int8)
    index = {int(value): row for row, value in enumerate(bits)}
    rows, columns, data = [], [], []
    for row, occupation in enumerate(occupations):
        neighbors, coefficients = operator.connections(occupation)
        for neighbor, coefficient in zip(neighbors, coefficients):
            value = sum(1 << int(a) for a in np.flatnonzero(neighbor))
            rows.append(row)
            columns.append(index[value])
            data.append(coefficient)
    matrix = sparse.csr_matrix((data, (rows, columns)), shape=(len(bits), len(bits)))
    try:
        raised_operator = SphereOperator(operator.ne, operator.flux, v3=operator.v3,
                                         two_lz=operator.two_lz + 2)
        raised_bits = enumerate_basis(raised_operator)
    except ValueError:
        # The highest-weight sector has an empty raising target.
        raised_bits = np.empty(0, dtype=np.int64)
    raised_index = {int(value): row for row, value in enumerate(raised_bits)}
    rows, columns, data = [], [], []
    for column, raw in enumerate(bits):
        value = int(raw)
        for a in range(operator.flux):
            if ((value >> a) & 1) and not ((value >> (a + 1)) & 1):
                final = value ^ (1 << a) ^ (1 << (a + 1))
                rows.append(raised_index[final])
                columns.append(column)
                # Adjacent empty-orbital raising has positive canonical sign.
                data.append(np.sqrt((operator.flux - a) * (a + 1)))
    raising = sparse.csr_matrix((data, (rows, columns)), shape=(len(raised_bits), len(bits)))
    return bits, matrix, raising, raised_bits


def load_reference(directory):
    """Check integrity, sector completeness, Hermiticity, and saved eigenpairs."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not set(REFERENCE_FILES).issubset(manifest):
        raise ValueError("Reference manifest omits required files")
    for name, expected in manifest.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Reference manifest must contain relative filenames only")
        if sha256(directory / name) != expected:
            raise ValueError(f"Reference integrity check failed: {name}")
    metadata = json.loads((directory / "metadata.json").read_text())
    ne, flux, two_lz = (int(metadata[name]) for name in ("ne", "flux", "two_lz"))
    bits = np.load(directory / "basis_bits.npy", allow_pickle=False)
    H = sparse.load_npz(directory / "hamiltonian.npz").tocsr()
    raising = sparse.load_npz(directory / "raising_operator.npz").tocsr()
    values = np.load(directory / "eigenvalues.npy", allow_pickle=False)
    vectors = np.load(directory / "eigenvectors.npy", allow_pickle=False)
    dimension = sector_dimension(ne, flux, two_lz)
    raised_dimension = sector_dimension(ne, flux, two_lz + 2)
    if (bits.shape != (dimension,) or bits.dtype.kind not in "iu"
            or H.shape != (dimension, dimension) or raising.shape != (raised_dimension, dimension)
            or vectors.shape != (dimension, len(values)) or values.ndim != 1 or not len(values)):
        raise ValueError("Reference dimensions disagree with complete fixed sector")
    if (not np.all(bits[1:] > bits[:-1]) or np.any(bits < 0)
            or np.any(bits >> (flux + 1))):
        raise ValueError("Reference basis must be sorted distinct valid bitstrings")
    occupations = ((bits[:, None] >> np.arange(flux + 1)) & 1).astype(float)
    if (not np.all(occupations.sum(1) == ne)
            or not np.all(2 * occupations @ np.arange(flux + 1) - ne * flux == two_lz)):
        raise ValueError("Reference basis lies outside the declared sector")
    if (not np.isfinite(H.data).all() or not np.isfinite(raising.data).all()
            or not np.isfinite(values).all() or not np.isfinite(vectors).all()):
        raise ValueError("Reference contains nonfinite values")
    asymmetry = H - H.T.conj()
    hermiticity_error = float(np.max(np.abs(asymmetry.data), initial=0.0))
    gram_error = float(np.max(abs(vectors.T.conj() @ vectors - np.eye(len(values)))))
    residuals = np.linalg.norm(H @ vectors - vectors * values, axis=0)
    if hermiticity_error > 1e-10 or gram_error > 1e-8 or residuals.max() > 1e-8:
        raise ValueError("Reference Hermiticity or eigenpair checks failed")
    return dict(metadata=metadata, bits=bits, occupations=occupations, H=H, raising=raising,
                eigenvalues=values, eigenvectors=vectors, manifest=manifest,
                validation=dict(hermiticity_error=hermiticity_error, gram_error=gram_error,
                                maximum_eigenpair_residual=float(residuals.max())))


def build_reference(ne, output, *, flux=None, v3=0.0, eigenpairs=4,
                    max_dimension=20000, seed=1729):
    """Generate a bounded small-system reference; refuse accidental large ED."""
    from scipy.sparse.linalg import eigsh
    from .checkpoint import atomic_json
    from .physics import SphereOperator

    if not 2 <= ne <= 10:
        raise ValueError("Reference generation supports 2 <= N <= 10")
    operator = SphereOperator(ne, flux=flux, v3=v3, two_lz=0)
    if operator.dimension() > max_dimension:
        raise ValueError(f"Sector has {operator.dimension()} states, above --max-dimension={max_dimension}")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Reference directory already exists: {output}")
    bits, H, raising, raised_bits = sparse_operators(operator)
    count = min(int(eigenpairs), len(bits))
    if count < 1:
        raise ValueError("eigenpairs must be positive")
    if len(bits) < 100 or count >= len(bits):
        values, vectors = np.linalg.eigh(H.toarray())
        values, vectors = values[:count], vectors[:, :count]
    else:
        values, vectors = eigsh(H, k=count, which="SA", tol=1e-11, maxiter=10000,
                                v0=np.random.default_rng(seed).normal(size=len(bits)))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
    output.mkdir(parents=True)
    metadata = dict(format_version=1, ne=ne, flux=operator.flux, two_lz=0, v3=float(v3),
                    dimension=len(bits), raising_target_dimension=len(raised_bits),
                    basis_convention="Ascending orbital Slater basis; a=m+flux/2",
                    Hamiltonian="sum P_R=1 + v3 sum P_R=3; normalized sphere LLL pair projectors",
                    purpose="Independent post-training diagnostics only; no ED labels enter SPRING")
    atomic_json(output / "metadata.json", metadata)
    np.save(output / "basis_bits.npy", bits)
    np.save(output / "raising_target_basis_bits.npy", raised_bits)
    sparse.save_npz(output / "hamiltonian.npz", H)
    sparse.save_npz(output / "raising_operator.npz", raising)
    np.save(output / "eigenvalues.npy", values)
    np.save(output / "eigenvectors.npy", vectors)
    names = REFERENCE_FILES + ("raising_target_basis_bits.npy",)
    atomic_json(output / "manifest.json", {name: sha256(output / name) for name in names})
    return load_reference(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ne", type=int, required=True)
    parser.add_argument("--flux", type=int)
    parser.add_argument("--v3", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eigenpairs", type=int, default=4)
    parser.add_argument("--max-dimension", type=int, default=20000)
    args = parser.parse_args()
    result = build_reference(args.ne, args.output, flux=args.flux, v3=args.v3,
                             eigenpairs=args.eigenpairs, max_dimension=args.max_dimension)
    print(json.dumps(dict(dimension=len(result["bits"]), eigenvalues=result["eigenvalues"].tolist())))


if __name__ == "__main__":
    main()
