"""Complex angular heads with two private tanh layers on shared MLP features.

This model contains no angular projector or ED input. Independent random
private hidden matrices supply different sector features. Zero final
readouts preserve the previous model's initial coherent physical state;
private hidden matrices begin receiving gradients after the readouts move.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from fqh_spring import model


def initialize(norb, nsectors, *, seed=7, width=64, layers=6, head_width=16):
    base = model.initialize(norb, width=width, layers=layers, kind='mlp',
                            seed=seed, readout='legacy_additive')
    params = {'base__' + name: value for name, value in base.items()}
    first, second = jax.random.split(jax.random.PRNGKey(seed + 100003))
    params['sector_W1'] = jax.random.normal(
        first, (nsectors, layers * width, head_width), dtype=jnp.float64
    ) / jnp.sqrt(float(layers * width))
    params['sector_b1'] = jnp.zeros((nsectors, head_width), dtype=jnp.float64)
    params['sector_W2'] = jax.random.normal(
        second, (nsectors, head_width, head_width), dtype=jnp.float64
    ) / jnp.sqrt(float(head_width))
    params['sector_b2'] = jnp.zeros((nsectors, head_width), dtype=jnp.float64)
    params['sector_out'] = jnp.zeros((nsectors, head_width, 2), dtype=jnp.float64)
    params['sector_bias'] = jnp.zeros((nsectors, 2), dtype=jnp.float64)
    params['mix_logits'] = jnp.zeros(nsectors, dtype=jnp.float64)
    return params


def components(params, occupations):
    """Return base complex logs, relative complex heads, and coherent mixing."""
    base = {name[6:]: value for name, value in params.items()
            if name.startswith('base__')}
    hidden = jnp.tanh((2. * occupations - 1.) @ base['W'] + base['b'])
    features = [hidden]
    result = hidden @ base['out']
    layers = 1 + sum(name.startswith('W') and name != 'W' for name in base)
    for layer in range(2, layers + 1):
        hidden = jnp.tanh(hidden @ base[f'W{layer}'] + base[f'b{layer}'])
        features.append(hidden)
        result = result + hidden @ base[f'out{layer}']
    result = result + base['bias']
    all_features = jnp.concatenate(features, axis=1)
    private1 = jnp.tanh(jnp.einsum('bf,sfw->bsw', all_features,
                                  params['sector_W1']) + params['sector_b1'])
    private2 = jnp.tanh(jnp.einsum('bsw,swv->bsv', private1,
                                  params['sector_W2']) + params['sector_b2'])
    output = jnp.einsum('bsw,swo->bso', private2,
                        params['sector_out']) + params['sector_bias']
    relative = 1. + output[:, :, 0] + 1j * output[:, :, 1]
    return (result[:, 0] + 1j * result[:, 1], relative,
            jax.nn.softmax(params['mix_logits']))


def amplitudes(params, occupations):
    logs, relative, mixing = components(params, occupations)
    shift = jax.lax.stop_gradient(jnp.max(logs.real))
    heads = jnp.exp(logs - shift)[:, None] * relative
    return heads, heads @ mixing, mixing, shift
