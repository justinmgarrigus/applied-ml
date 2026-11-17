"""Occupation MLP with a readout from every hidden tanh layer.

For x=2n-1, h1=tanh(x W+b), hl=tanh(h(l-1) Wl+bl), and
output=h1 out+sum(l>=2) hl outl+bias. The two real heads are log
amplitude and phase. All parameters remain real. Extra readouts start at
zero, embedding the supplied shallow or deep checkpoint exactly. No
Laughlin coefficients, analytic zeros, or ED data enter this model.

Depth is inferred from static parameter keys when JAX traces a function.
Enable jax_enable_x64 before constructing the parameters, as in the runner.
"""
from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np


def hidden_layers(params):
    """Infer and validate consecutive layer keys without reading array values."""
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a flat dictionary")
    required = {"W", "b", "out", "bias"}
    layers = 1
    while f"W{layers + 1}" in params:
        layers += 1
        required.update((f"W{layers}", f"b{layers}", f"out{layers}"))
    if set(params) != required:
        raise ValueError("Require complete consecutive Wl/bl/outl layer keys")
    return layers


def upgrade_params(params, layers=6, seed=1729):
    """Copy a checkpoint and add zero-readout layers without changing its state.

    Layer 2 uses exactly ``deep_model.upgrade_params``'s random key convention.
    Thus upgrading a shallow model to either two or six layers starts with
    identical W2/b2, and identical wavefunction coefficients. Later layers use
    layer-index-folded keys, independently of the requested final depth.
    """
    if isinstance(layers, bool) or not isinstance(layers, Integral) or layers < 1:
        raise ValueError("layers must be a positive integer")
    existing = hidden_layers(params)
    if layers < existing:
        raise ValueError("Cannot discard existing hidden layers")
    arrays = {name: jnp.array(value, copy=True) for name, value in params.items()}
    if any(value.dtype.kind != "f" for value in arrays.values()):
        raise ValueError("The occupation model requires real floating parameters")
    if arrays["W"].ndim != 2:
        raise ValueError("W must have shape (n_orbitals, hidden)")
    width = arrays["W"].shape[1]
    if (width < 1 or arrays["b"].shape != (width,)
            or arrays["out"].shape != (width, 2) or arrays["bias"].shape != (2,)):
        raise ValueError("Inconsistent first-layer parameter shapes")
    for layer in range(2, existing + 1):
        if (arrays[f"W{layer}"].shape != (width, width)
                or arrays[f"b{layer}"].shape != (width,)
                or arrays[f"out{layer}"].shape != (width, 2)):
            raise ValueError(f"Inconsistent layer {layer} parameter shapes")
    dtype = arrays["W"].dtype
    key = jax.random.PRNGKey(seed)
    for layer in range(existing + 1, int(layers) + 1):
        layer_key = key if layer == 2 else jax.random.fold_in(key, layer)
        key_weight, key_bias = jax.random.split(layer_key)
        arrays[f"W{layer}"] = (0.8 / np.sqrt(width)) * jax.random.normal(
            key_weight, (width, width), dtype=dtype)
        arrays[f"b{layer}"] = 0.1 * jax.random.normal(key_bias, (width,), dtype=dtype)
        arrays[f"out{layer}"] = jnp.zeros((width, 2), dtype=dtype)
    return arrays


def output_from_first_hidden(params, first_hidden):
    """Return the two real heads from arbitrary leading batch dimensions."""
    hidden = first_hidden
    output = hidden @ params["out"]
    for layer in range(2, hidden_layers(params) + 1):
        hidden = jnp.tanh(hidden @ params[f"W{layer}"] + params[f"b{layer}"])
        output = output + hidden @ params[f"out{layer}"]
    return output + params["bias"]


def log_psi(params, occupation):
    """Complex log coefficient of one canonical LLL Slater determinant."""
    first_hidden = jnp.tanh((2.0 * occupation - 1.0) @ params["W"] + params["b"])
    output = output_from_first_hidden(params, first_hidden)
    return output[0] + 1j * output[1]


batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0))


def score_tree(params, occupations):
    """Complex logarithmic derivatives with respect to all real parameters."""
    real = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).real), in_axes=(None, 0))(params, occupations)
    imag = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).imag), in_axes=(None, 0))(params, occupations)
    return jax.tree.map(lambda a, b: a + 1j * b, real, imag)
