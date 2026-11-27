"""Independent full-sector diagnostics for a frozen neural wavefunction.

This command reads small sparse references only after training.  It supports
plain parameter NPZ files and complete resumable SPRING checkpoint NPZ files.
No reference coefficients are used to construct or optimize the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .references import load_reference, sha256


def read_parameters(path):
    """Read finite flat real parameters without loading optimizer state/pickles."""
    params, _ = _read_parameter_payload(path)
    return params


def _read_parameter_payload(path):
    metadata = None
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" in archive.files:
            raw = archive["metadata"]
            if raw.shape != () or raw.dtype.kind not in "US":
                raise ValueError("Checkpoint metadata must be a JSON string scalar")
            metadata = json.loads(raw.item())
            names = metadata.get("parameter_names", [])
            if not names or len(set(names)) != len(names):
                raise ValueError("Checkpoint has invalid parameter names")
            params = {name: np.array(archive[f"param_{i:04d}"], copy=True)
                      for i, name in enumerate(names)}
        else:
            params = {name: np.array(archive[name], copy=True) for name in archive.files}
    if not params or any(value.dtype.kind != "f" or not np.isfinite(value).all()
                         for value in params.values()):
        raise ValueError("Neural parameters must be finite real floating arrays")
    return params, metadata


def _find_run_config(checkpoint, run_dir):
    if run_dir is not None:
        candidates = (Path(run_dir),)
    elif checkpoint.parent.name == "checkpoints":
        candidates = (checkpoint.parent, checkpoint.parent.parent)
    else:
        candidates = (checkpoint.parent,)
    for directory in candidates:
        for name in ("run_config.json", "config.json"):
            path = directory / name
            if path.is_file():
                return path
    if run_dir is not None:
        raise FileNotFoundError(f"No run_config.json or config.json in {run_dir}")
    return None


def angular_measure(psi, raising, max_l):
    """Resolve L probabilities at M=0 using fully reorthogonalized Lanczos.

    Only the spectral measure of this saved vector is resolved; no projection
    or change of state is performed. Integer-L and two moment checks guard
    against numerical ghost Ritz values.
    """
    from scipy.linalg import eigh_tridiagonal

    vectors, diagonal, offdiagonal = [], [], []
    vector = np.asarray(psi, dtype=complex).copy()
    vector /= np.linalg.norm(vector)
    previous = np.zeros_like(vector)
    beta = 0.0
    for k in range(max_l + 9):
        vectors.append(vector)
        product = raising.T.conj() @ (raising @ vector)
        alpha = float(np.vdot(vector, product).real)
        diagonal.append(alpha)
        residual = product - alpha * vector - beta * previous
        basis = np.asarray(vectors).T
        for _ in range(2):
            residual -= basis @ (basis.T.conj() @ residual)
        next_beta = float(np.linalg.norm(residual))
        if next_beta < 1e-9 or k == max_l + 8:
            break
        offdiagonal.append(next_beta)
        previous, vector, beta = vector, residual / next_beta, next_beta
    if len(diagonal) == 1:
        values, eigenvectors = np.asarray(diagonal), np.ones((1, 1))
    else:
        values, eigenvectors = eigh_tridiagonal(np.asarray(diagonal), np.asarray(offdiagonal))
    weights = abs(eigenvectors[0]) ** 2
    levels = np.rint((np.sqrt(np.maximum(0.0, 1 + 4 * values)) - 1) / 2).astype(int)
    errors = abs(values - levels * (levels + 1))
    substantial = weights > 1e-10
    if (np.max(errors[substantial], initial=0.0) > 1e-5
            or np.any(levels[substantial] > max_l) or np.any(levels[substantial] < 0)):
        raise ArithmeticError("Angular-momentum Ritz spectrum failed the integer-L check")
    mass = np.zeros(max_l + 1)
    for weight, level in zip(weights, levels):
        if 0 <= level <= max_l:
            mass[level] += weight
        elif weight > 1e-10:
            raise ArithmeticError("Angular-momentum weight is outside the allowed range")
    applied = raising.T.conj() @ (raising @ psi)
    mean, second = float(np.vdot(psi, applied).real), float(np.vdot(applied, applied).real)
    grid = np.arange(max_l + 1) * (np.arange(max_l + 1) + 1)
    first_error, second_error = abs(mass @ grid - mean), abs(mass @ grid**2 - second)
    if first_error > 1e-6 or second_error > 1e-3 or abs(mass.sum() - 1) > 1e-8:
        raise ArithmeticError("Angular-momentum normalization/moment checks failed")
    return dict(L=list(range(max_l + 1)), probability=mass.tolist(),
                L0_probability=float(mass[0]), mean_L2=mean,
                first_moment_error=float(first_error), second_moment_error=float(second_error),
                integer_spectrum_error=float(np.max(errors[substantial], initial=0.0)),
                lanczos_dimension=len(diagonal), no_projection_applied=True)


def state_diagnostics(psi, reference, *, angular=False):
    """Rayleigh energy, residual, L2, and exact fixed-M one-body density."""
    from scipy.integrate import simpson
    from scipy.special import comb

    metadata = reference["metadata"]
    ne, flux = int(metadata["ne"]), int(metadata["flux"])
    norb, two_lz = flux + 1, int(metadata["two_lz"])
    psi = np.asarray(psi, dtype=complex)
    if psi.shape != (len(reference["bits"]),) or not np.isfinite(psi).all():
        raise ValueError("Coefficient vector has the wrong shape or nonfinite entries")
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Coefficient vector must have finite nonzero norm")
    psi = psi / norm
    probability = abs(psi)**2
    H, raising = reference["H"], reference["raising"]
    applied = H @ psi
    energy = np.vdot(psi, applied)
    if abs(energy.imag) > 1e-10:
        raise ArithmeticError("Hermitian Rayleigh quotient has a nonreal part")
    variance = float(np.linalg.norm(applied - energy.real * psi)**2)
    magnetization = two_lz / 2
    l2 = float(np.vdot(raising @ psi, raising @ psi).real + magnetization * (magnetization + 1))
    values, vectors = reference["eigenvalues"], reference["eigenvectors"]
    weights = abs(vectors.T.conj() @ psi)**2
    ground_index = int(np.argmin(values))
    ground = vectors[:, ground_index]
    n = probability @ reference["occupations"]
    theta = np.linspace(0, np.pi, 361)
    p, a = np.cos(theta / 2)**2, np.arange(norb)
    kernels = comb(flux, a)[None, :] * p[:, None]**a * (1 - p[:, None])**(flux - a)
    ratio = (norb / ne) * (kernels @ n)
    density_integral = float(ne * 0.5 * simpson(ratio * np.sin(theta), x=theta))
    density_rms = float(np.sqrt(0.5 * simpson((ratio - 1)**2 * np.sin(theta), x=theta)))
    if abs(density_integral - ne) > 1e-6 or abs(n.sum() - ne) > 1e-10:
        raise ArithmeticError("One-body density does not integrate to the particle number")
    result = dict(
        exact_variational_energy_V1=float(energy.real),
        exact_residual_variance_V1_squared=variance, exact_L2=l2,
        ground_state_fidelity=float(weights[ground_index]),
        saved_reference_eigenvalues_V1=values.tolist(),
        saved_reference_eigenstate_weights=weights.tolist(),
        weight_outside_saved_reference_eigenvectors=float(max(0.0, 1 - weights.sum())),
        normalization=float(probability.sum()),
        effective_occupation_support=float(1 / np.sum(probability**2)),
        probability_total_variation_distance=float(0.5 * np.sum(abs(probability - abs(ground)**2))),
        maximum_fidelity_at_fixed_magnitudes=float(np.sum(abs(ground) * abs(psi))**2),
        orbital_occupations=n.tolist(), theta_degrees=np.rad2deg(theta).tolist(),
        density_relative_to_uniform=ratio.tolist(), density_RMS_contrast=density_rms,
        density_min_relative_to_uniform=float(ratio.min()),
        density_max_relative_to_uniform=float(ratio.max()),
        density_particle_number_integral=density_integral,
        density_convention="a=m+flux/2; north pole selects a=flux; density divided by N/(4*pi)",
        density_interpretation="Exact fixed-M axisymmetric density; finite-size anisotropy alone does not establish a thermodynamic CDW",
    )
    if angular:
        if two_lz != 0:
            raise ValueError("Angular spectral diagnostics currently require M=0")
        result["angular_momentum"] = angular_measure(psi, raising, ne * (norb - ne) // 2)
        result["exact_L0_probability"] = result["angular_momentum"]["L0_probability"]
    return result


def evaluate(checkpoint, reference, output, *, batch=256, save_psi=False, angular=False, run_dir=None):
    """Evaluate a frozen checkpoint using the complete independent reference."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from . import model
    from .checkpoint import atomic_json

    checkpoint, reference, output = map(Path, (checkpoint, reference, output))
    if batch < 1:
        raise ValueError("Evaluation batch must be positive")
    checkpoint_hash = sha256(checkpoint)
    params, checkpoint_metadata = _read_parameter_payload(checkpoint)
    params = {name: jnp.asarray(value) for name, value in params.items()}
    info = model.architecture(params)
    data = load_reference(reference)
    metadata = data["metadata"]
    if params["W"].shape[0] != int(metadata["flux"]) + 1:
        raise ValueError("Neural input size disagrees with reference orbitals")
    config_file = _find_run_config(checkpoint, run_dir)
    run_metadata, config = {}, {}
    if config_file is not None:
        config_hash = sha256(config_file)
        run_metadata = json.loads(config_file.read_text())
        config = run_metadata.get("configuration", run_metadata)
        for key in ("ne", "flux", "two_lz", "v3"):
            if key in config and config[key] is not None and config[key] != metadata[key]:
                raise ValueError(f"Checkpoint and reference disagree on {key}")
    amplitude_map = config.get("amplitude_map", "exponential")
    signature = run_metadata.get("config_sha256")
    invariant = run_metadata.get("invariant_configuration")
    if invariant is not None and invariant.get("amplitude_map", "exponential") != amplitude_map:
        raise ValueError("Reported amplitude map disagrees with the signed run configuration")
    if checkpoint_metadata is not None and signature is not None:
        if checkpoint_metadata.get("config_sha256") != signature:
            raise ValueError("Checkpoint and run configuration signatures disagree")
    anchor_path = None
    if amplitude_map == "cartesian":
        from .cartesian import make_amplitude

        if checkpoint_metadata is not None and not signature:
            raise ValueError("Cartesian checkpoint requires a signed run configuration")
        anchor_path = config_file.with_name("initial_parameters.npz")
        anchor_hash = run_metadata.get("frozen_anchor_sha256")
        if not anchor_hash:
            raise ValueError("Cartesian evaluation requires a frozen anchor SHA-256 in the run metadata")
        if invariant is not None:
            signed_hash = hashlib.sha256(json.dumps(invariant, sort_keys=True).encode()).hexdigest()
            if signed_hash != signature or invariant.get("frozen_anchor_sha256") != anchor_hash:
                raise ValueError("Cartesian run signature does not bind the reported anchor SHA-256")
        if sha256(anchor_path) != anchor_hash:
            raise ValueError("Cartesian frozen anchor SHA-256 disagrees with the run metadata")
        anchor = read_parameters(anchor_path)
        if set(anchor) != set(params) or any(
            (anchor[name].shape, anchor[name].dtype) != (value.shape, value.dtype)
            for name, value in params.items()
        ):
            raise ValueError("Cartesian frozen anchor and checkpoint parameter schemas differ")
        anchor = {name: jnp.asarray(value) for name, value in anchor.items()}
        forward = jax.jit(jax.vmap(make_amplitude(anchor), in_axes=(None, 0)))
    elif amplitude_map == "exponential":
        forward = jax.jit(model.batch_log_psi)
    else:
        raise ValueError(f"Unknown checkpoint amplitude map '{amplitude_map}'")
    occupations = data["occupations"]
    values = np.concatenate([np.asarray(forward(params, jnp.asarray(occupations[start:start + batch])))
                             for start in range(0, len(occupations), batch)])
    if not np.isfinite(values).all():
        raise ArithmeticError("Nonfinite wavefunction coefficients")
    psi = np.exp(values - values.real.max()) if amplitude_map == "exponential" else values
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError("Wavefunction must have finite nonzero norm")
    psi /= norm
    result = state_diagnostics(psi, data, angular=angular)
    result.update(ne=int(metadata["ne"]), flux=int(metadata["flux"]),
                  two_lz=int(metadata["two_lz"]), v3=float(metadata["v3"]),
                  orbitals=int(metadata["flux"]) + 1, basis_dimension=len(occupations),
                  architecture=info, amplitude_map=amplitude_map,
                  parameter_count=int(sum(value.size for value in params.values())),
                  checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                  reference=str(reference), reference_manifest_sha256=sha256(reference / "manifest.json"),
                  reference_validation=data["validation"],
                  ed_used_for_training=False,
                  scope="Frozen complete-sector evaluation; independent sparse matrices and ED eigenvectors are never read by the training runner")
    if anchor_path is not None:
        if sha256(anchor_path) != anchor_hash:
            raise RuntimeError("Frozen Cartesian anchor changed during evaluation")
        result.update(frozen_anchor=str(anchor_path), frozen_anchor_sha256=anchor_hash)
    if config_file is not None and sha256(config_file) != config_hash:
        raise RuntimeError("Run configuration changed during frozen evaluation")
    if sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during frozen evaluation")
    if any(sha256(reference / name) != expected for name, expected in data["manifest"].items()):
        raise RuntimeError("Reference changed during frozen evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    if save_psi:
        coefficient_file = output.with_name(output.stem + "_coefficients.npy")
        np.save(coefficient_file, psi)
        result["coefficient_file"] = coefficient_file.name
        result["coefficient_file_sha256"] = sha256(coefficient_file)
    result["inputs_unchanged"] = True
    atomic_json(output, result)
    print(f"E/V1={result['exact_variational_energy_V1']:.12g}; "
          f"F={result['ground_state_fidelity']:.8g}; L2={result['exact_L2']:.8g}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--save-psi", action="store_true")
    parser.add_argument("--angular", action="store_true", help="Resolve total-L probabilities at M=0")
    parser.add_argument("--run-dir", type=Path, help="Read run configuration and frozen anchor from this directory")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.reference, args.output, batch=args.batch,
             save_psi=args.save_psi, angular=args.angular, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
