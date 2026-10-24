"""One occupation-network interface for MLP, dense residual and attention arms.

All variants return coefficients of canonical antisymmetric LLL Slater states.
Adding hidden features supplies neither an analytic Laughlin factor nor L=0
projection. Both log magnitude and phase are trainable real neural outputs.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import base_model, structured_model


def variant(params):
    if any(name.startswith('attn_') for name in params):
        return 'attention'
    if any(name.startswith('dense_') for name in params):
        return 'dense'
    return 'mlp'


def log_psi(params, occupation):
    if variant(params) == 'mlp':
        return base_model.log_psi(params, occupation)
    return structured_model.log_psi(params, occupation)


batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0))


def score_tree(params, occupations):
    real = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).real), in_axes=(None, 0))(params, occupations)
    imag = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).imag), in_axes=(None, 0))(params, occupations)
    return jax.tree.map(lambda a, b: a + 1j * b, real, imag)


def architecture(params):
    if variant(params) != 'mlp':
        return structured_model.architecture(params)
    base_model.upgrade_params(params, layers=base_model.hidden_layers(params))
    count = sum(value.size for value in params.values())
    return dict(variant='mlp', base_layers=base_model.hidden_layers(params),
                base_width=int(params['W'].shape[1]), orbitals=int(params['W'].shape[0]),
                total_parameter_count=count, parameter_count=count,
                correction_parameter_count=0, residual_blocks=0)


def initialize(norb, width=64, layers=6, kind='dense', seed=1729):
    """Fresh generic state; continuation NPZ files bypass this initialization."""
    if width < 1 or layers < 1:
        raise ValueError('width and layers must be positive')
    key = jax.random.PRNGKey(seed)
    kw, kb, ko = jax.random.split(key, 3)
    params = dict(W=.8 / jnp.sqrt(float(norb)) * jax.random.normal(kw, (norb, width), dtype=jnp.float64),
                  b=.1 * jax.random.normal(kb, (width,), dtype=jnp.float64),
                  out=.02 / jnp.sqrt(float(width)) * jax.random.normal(ko, (width, 2), dtype=jnp.float64),
                  bias=jnp.zeros((2,), dtype=jnp.float64))
    return prepare(params, norb=norb, width=width, layers=layers, kind=kind, seed=seed)


def prepare(params, *, norb, width, layers, kind, seed=1729):
    """Validate a trained source, adding only missing zero-readout branches.

    Existing structured parameters are retained exactly. Changing variant or
    discarding existing layers is rejected instead of silently resetting them.
    """
    if kind not in ('mlp', 'dense', 'attention'):
        raise ValueError('Unknown network variant')
    if any(jnp.asarray(value).dtype.kind != 'f' for value in params.values()):
        raise ValueError('Source parameters must be real floating arrays')
    params = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in params.items()}
    existing = variant(params)
    if params['W'].shape != (norb, width):
        raise ValueError('Source orbital count or hidden width does not match configuration')
    if existing != 'mlp':
        if existing != kind or base_model.hidden_layers(structured_model.split_base_params(params)) != layers:
            raise ValueError('An existing structured source requires the same variant and depth')
        architecture(params)
        return params
    params = base_model.upgrade_params(params, layers=layers, seed=seed)
    if kind != 'mlp':
        params = structured_model.init_params(params, variant=kind, seed=seed)
    architecture(params)
    return params
