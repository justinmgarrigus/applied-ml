"""Occupation MLP with one readout from the final hidden layer.

For x=2n-1, h1=tanh(x W+b), hl=tanh(h(l-1) Wl+bl), and
output=hL final_out+bias. The two real channels are log magnitude and phase.
All hidden weights, biases and the final readout are jointly trainable.
Historical additive-readout checkpoints are handled by legacy_model instead.

Depth is inferred from static parameter keys when JAX traces a function.
Enable jax_enable_x64 before constructing the parameters, as in the runner.
"""
from __future__ import annotations

from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np


def hidden_layers(params):
    """Validate the final-readout schema and return its consecutive depth."""
    if not isinstance(params, dict):
        raise ValueError('Parameters must be a flat dictionary')
    required = {'W', 'b', 'final_out', 'bias'}
    layers = 1
    while f'W{layers + 1}' in params:
        layers += 1
        required.update((f'W{layers}', f'b{layers}'))
    if set(params) != required:
        raise ValueError('Final-readout MLP requires Wl/bl, final_out and bias only; '
                         'additive out/outl heads belong to the legacy model')
    if any(value.dtype.kind != 'f' for value in params.values()):
        raise ValueError('The occupation model requires real floating parameters')
    if params['W'].ndim != 2:
        raise ValueError('W must have shape (n_orbitals, hidden)')
    width = params['W'].shape[1]
    if (width < 1 or params['b'].shape != (width,)
            or params['final_out'].shape != (width, 2) or params['bias'].shape != (2,)):
        raise ValueError('Inconsistent input, final readout or bias shapes')
    for layer in range(2, layers + 1):
        if (params[f'W{layer}'].shape != (width, width)
                or params[f'b{layer}'].shape != (width,)):
            raise ValueError(f'Inconsistent hidden layer {layer} shapes')
    return layers


def initialize(norb, width=64, layers=6, seed=1729):
    """Initialize all hidden layers and one nonzero magnitude/phase readout."""
    for name, value in (('norb', norb), ('width', width), ('layers', layers)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    key = jax.random.PRNGKey(seed)
    kw, kb, ko = jax.random.split(key, 3)
    params = dict(W=.8 / np.sqrt(norb) * jax.random.normal(kw, (norb, width), dtype=jnp.float64),
                  b=.1 * jax.random.normal(kb, (width,), dtype=jnp.float64),
                  final_out=.02 / np.sqrt(width) * jax.random.normal(ko, (width, 2), dtype=jnp.float64),
                  bias=jnp.zeros((2,), dtype=jnp.float64))
    for layer in range(2, layers + 1):
        layer_key = key if layer == 2 else jax.random.fold_in(key, layer)
        weight_key, bias_key = jax.random.split(layer_key)
        params[f'W{layer}'] = .8 / np.sqrt(width) * jax.random.normal(
            weight_key, (width, width), dtype=jnp.float64)
        params[f'b{layer}'] = .1 * jax.random.normal(bias_key, (width,), dtype=jnp.float64)
    hidden_layers(params)
    return params


def output_from_first_hidden(params, first_hidden):
    """Propagate through every hidden layer, then apply the sole output head."""
    hidden = first_hidden
    for layer in range(2, hidden_layers(params) + 1):
        hidden = jnp.tanh(hidden @ params[f"W{layer}"] + params[f"b{layer}"])
    return hidden @ params['final_out'] + params['bias']


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
