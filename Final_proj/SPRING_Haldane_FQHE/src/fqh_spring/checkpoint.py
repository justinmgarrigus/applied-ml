"""Atomic, pickle-free full-state checkpoints for long SPRING continuations.

The optimizer is reconstructed only with a caller-provided state template.
Parameter values use the model's flat string-keyed dictionary.  Physics and
configuration compatibility are the caller's responsibility; the saved config
SHA256 can be required on load.  A single NPZ contains arrays and JSON metadata,
so a process interruption cannot mix metadata from one generation with another.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

FORMAT_VERSION = 1


def _json_default(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Object arrays cannot be serialized")
        return {"__checkpoint_numpy_array__": value.tolist(), "dtype": value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON metadata type: {type(value).__name__}")


def _json_hook(value):
    if set(value) == {"__checkpoint_numpy_array__", "dtype"}:
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays cannot be deserialized")
        return np.asarray(value["__checkpoint_numpy_array__"], dtype=dtype)
    return value


def _json_text(data):
    return json.dumps(data, default=_json_default, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        # Persist the rename as well as the file contents on local filesystems.
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    """Atomically replace one NPZ; forbid pickle-dependent object arrays."""
    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("NPZ object arrays are forbidden; use JSON metadata")
    _atomic_write(path, lambda stream: np.savez(stream, **arrays))


def atomic_json(path, data):
    """Atomically replace UTF-8 JSON, including NumPy RNG metadata if needed."""
    encoded = (_json_text(data) + "\n").encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(encoded))


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise ValueError(f"{name} must be a numeric nonobject array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _spec(array):
    return {"shape": list(array.shape), "dtype": array.dtype.str}


def _check_signature(value):
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value):
        raise ValueError("config_sha256 must be a lowercase 64-character SHA256 digest")


def save_checkpoint(path, *, params, state, walkers, key, rng_state, step,
                    training_seconds, config_sha256):
    """Save a complete resumable state through a single atomic replacement.

    Pass ``rng.bit_generator.state`` as rng_state.  ``training_seconds`` is
    cumulative active training time; burn-in, pauses and evaluations should
    be accounted for separately by the runner.  Keys are ordinary numeric
    legacy JAX PRNGKey arrays; typed key arrays must be converted by the caller.
    """
    if not isinstance(params, dict) or not params or any(not isinstance(name, str) for name in params):
        raise ValueError("params must be a nonempty flat dictionary with string keys")
    if isinstance(step, (bool, np.bool_)) or int(step) != step or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not math.isfinite(training_seconds) or training_seconds < 0:
        raise ValueError("training_seconds must be finite and nonnegative")
    if not isinstance(rng_state, dict):
        raise ValueError("rng_state must be a NumPy bit-generator state dictionary")
    _check_signature(config_sha256)
    arrays = {}
    names = sorted(params)
    parameter_specs = []
    for index, name in enumerate(names):
        value = _numeric(params[name], f"parameter {name}")
        arrays[f"param_{index:04d}"] = value
        parameter_specs.append(_spec(value))
    leaves, definition = jax.tree.flatten(state)
    state_specs = []
    for index, leaf in enumerate(leaves):
        value = _numeric(leaf, f"optimizer leaf {index}")
        arrays[f"state_{index:04d}"] = value
        state_specs.append(_spec(value))
    for name, value in (("walkers", walkers), ("key", key)):
        arrays[name] = _numeric(value, name)
    metadata = {
        "format_version": FORMAT_VERSION,
        "parameter_names": names, "parameter_specs": parameter_specs,
        "state_treedef": str(definition), "state_specs": state_specs,
        "walkers_spec": _spec(arrays["walkers"]), "key_spec": _spec(arrays["key"]),
        "rng_state": rng_state, "step": int(step),
        "training_seconds": float(training_seconds), "config_sha256": config_sha256,
    }
    arrays["metadata"] = np.asarray(_json_text(metadata))
    atomic_npz(path, **arrays)


def _read_array(archive, name, spec):
    value = _numeric(archive[name], name)
    if list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
        raise ValueError(f"Checkpoint shape/dtype mismatch for {name}")
    return value


def load_checkpoint(path, state_template, *, expected_config_sha256=None):
    """Load a numeric-only checkpoint and validate its optimizer template.

    This function never loads pickles or imports classes named in a checkpoint.
    The state template supplies the trusted optimizer pytree definition.
    """
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("Missing checkpoint metadata")
        text = archive["metadata"]
        if text.shape != () or text.dtype.kind not in "US":
            raise ValueError("Checkpoint metadata must be a nonobject JSON scalar")
        scalar = text.item()
        if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8")
        metadata = json.loads(scalar, object_hook=_json_hook)
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format version")
        _check_signature(metadata["config_sha256"])
        if expected_config_sha256 is not None and metadata["config_sha256"] != expected_config_sha256:
            raise ValueError("Checkpoint configuration SHA256 does not match")
        leaves, definition = jax.tree.flatten(state_template)
        if str(definition) != metadata["state_treedef"] or len(leaves) != len(metadata["state_specs"]):
            raise ValueError("Checkpoint optimizer pytree does not match state_template")
        names = metadata["parameter_names"]
        if (not names or len(names) != len(set(names)) or any(not isinstance(name, str) for name in names)
                or len(names) != len(metadata["parameter_specs"])):
            raise ValueError("Invalid checkpoint parameter metadata")
        required = {"metadata", "walkers", "key"}
        required.update(f"param_{index:04d}" for index in range(len(names)))
        required.update(f"state_{index:04d}" for index in range(len(leaves)))
        if set(archive.files) != required:
            raise ValueError("Checkpoint contains missing or unexpected arrays")
        params = {name: jnp.asarray(_read_array(archive, f"param_{index:04d}", spec))
                  for index, (name, spec) in enumerate(zip(names, metadata["parameter_specs"]))}
        restored_leaves = []
        for index, (template, spec) in enumerate(zip(leaves, metadata["state_specs"])):
            if _spec(np.asarray(template)) != spec:
                raise ValueError(f"Checkpoint optimizer leaf {index} shape/dtype does not match state_template")
            restored_leaves.append(jnp.asarray(_read_array(archive, f"state_{index:04d}", spec)))
        walkers = jnp.asarray(_read_array(archive, "walkers", metadata["walkers_spec"]))
        key = jnp.asarray(_read_array(archive, "key", metadata["key_spec"]))
    step = metadata["step"]
    seconds = metadata["training_seconds"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Invalid checkpoint step")
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid checkpoint training_seconds")
    if not isinstance(metadata["rng_state"], dict):
        raise ValueError("Invalid checkpoint NumPy RNG state")
    return {
        "params": params, "state": jax.tree.unflatten(definition, restored_leaves),
        "walkers": walkers, "key": key, "rng_state": metadata["rng_state"],
        "step": step, "training_seconds": float(seconds),
        "config_sha256": metadata["config_sha256"],
    }
