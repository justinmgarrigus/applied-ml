"""Final-readout MLP interface with historical-model compatibility.

All variants return coefficients of canonical antisymmetric LLL Slater states.
Adding hidden features supplies neither an analytic Laughlin factor nor L=0
projection. Both log magnitude and phase are trainable real neural outputs.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import base_model, legacy_model, structured_model


def variant(params):
    if any(name.startswith('attn_') for name in params):
        return 'attention'
    if any(name.startswith('dense_') for name in params):
        return 'dense'
    return 'mlp'


def log_psi(params, occupation):
    if variant(params) == 'mlp':
        implementation = base_model if readout_kind(params) == 'final' else legacy_model
        return implementation.log_psi(params, occupation)
    return structured_model.log_psi(params, occupation)


batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0))


def score_tree(params, occupations):
    real = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).real), in_axes=(None, 0))(params, occupations)
    imag = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).imag), in_axes=(None, 0))(params, occupations)
    return jax.tree.map(lambda a, b: a + 1j * b, real, imag)


def readout_kind(params):
    """Parameter names identify the schema without reinterpreting old states."""
    return 'final' if 'final_out' in params else 'legacy_additive'


def architecture(params):
    if variant(params) != 'mlp':
        if readout_kind(params) == 'final':
            raise ValueError('Final-readout MLP cannot contain legacy correction branches')
        return dict(structured_model.architecture(params), readout='legacy_additive')
    readout = readout_kind(params)
    implementation = base_model if readout == 'final' else legacy_model
    layers = implementation.hidden_layers(params)
    if readout == 'legacy_additive':
        legacy_model.upgrade_params(params, layers=layers)
    count = sum(value.size for value in params.values())
    return dict(variant='mlp', readout=readout, base_layers=layers,
                base_width=int(params['W'].shape[1]), orbitals=int(params['W'].shape[0]),
                total_parameter_count=count, parameter_count=count,
                correction_parameter_count=0, residual_blocks=0)


def initialize(norb, width=64, layers=6, kind='mlp', seed=1729, readout='final'):
    """Fresh final-readout state by default; legacy models require opt-in."""
    if readout == 'final':
        if kind != 'mlp':
            raise ValueError('Final readout requires variant="mlp"; correction branches are legacy models')
        return base_model.initialize(norb, width=width, layers=layers, seed=seed)
    if readout != 'legacy_additive':
        raise ValueError('Unknown readout architecture')
    if width < 1 or layers < 1:
        raise ValueError('width and layers must be positive')
    key = jax.random.PRNGKey(seed)
    kw, kb, ko = jax.random.split(key, 3)
    params = dict(W=.8 / jnp.sqrt(float(norb)) * jax.random.normal(kw, (norb, width), dtype=jnp.float64),
                  b=.1 * jax.random.normal(kb, (width,), dtype=jnp.float64),
                  out=.02 / jnp.sqrt(float(width)) * jax.random.normal(ko, (width, 2), dtype=jnp.float64),
                  bias=jnp.zeros((2,), dtype=jnp.float64))
    return prepare(params, norb=norb, width=width, layers=layers, kind=kind, seed=seed,
                   readout=readout)


def prepare(params, *, norb, width, layers, kind, seed=1729, readout='final'):
    """Validate a source without silently changing its readout architecture.

    Existing structured parameters are retained exactly. Changing variant or
    discarding existing layers is rejected instead of silently resetting them.
    """
    if kind not in ('mlp', 'dense', 'attention'):
        raise ValueError('Unknown network variant')
    if any(jnp.asarray(value).dtype.kind != 'f' for value in params.values()):
        raise ValueError('Source parameters must be real floating arrays')
    params = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in params.items()}
    if readout not in ('final', 'legacy_additive'):
        raise ValueError('Unknown readout architecture')
    if readout_kind(params) != readout:
        raise ValueError('Source readout architecture differs: legacy additive checkpoints cannot '
                         'initialize a final-readout MLP; use source=null for a fresh model')
    existing = variant(params)
    if params['W'].shape != (norb, width):
        raise ValueError('Source orbital count or hidden width does not match configuration')
    if readout == 'final':
        if kind != 'mlp' or existing != 'mlp':
            raise ValueError('Final readout requires variant="mlp"')
        if base_model.hidden_layers(params) != layers:
            raise ValueError('Cannot change final-readout depth while preserving a source state; '
                             'use source=null for a fresh model')
        architecture(params)
        return params
    if existing != 'mlp':
        if existing != kind or legacy_model.hidden_layers(structured_model.split_base_params(params)) != layers:
            raise ValueError('An existing structured source requires the same variant and depth')
        architecture(params)
        return params
    params = legacy_model.upgrade_params(params, layers=layers, seed=seed)
    if kind != 'mlp':
        params = structured_model.init_params(params, variant=kind, seed=seed)
    architecture(params)
    return params
