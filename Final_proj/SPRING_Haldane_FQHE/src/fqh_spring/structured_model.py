"""Matched dense and pair-aware attention corrections to an occupation MLP.

The supplied occupation-coefficient network is retained, with every parameter
trainable.  A new zero-initialized two-channel readout embeds its wavefunction
exactly.  Corrections act on canonical LLL Slater coefficients; this is not a
real-space FermiNet and neither supplies Laughlin zeros nor projects L=0.

The attention correction uses 22 orbital tokens [n_a, m_a/Q, (m_a/Q)^2],
two width-eight, two-head residual attention blocks, and pair biases depending
on orbital separation.  Dense and attention controls both have two actual
residual hidden blocks and differ in added parameter count by less than 1%.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import base_model as multilayer_model

ATTENTION_WIDTH = 8
ATTENTION_HEADS = 2
ATTENTION_FF_WIDTH = 24
DENSE_WIDTH = 22
BLOCKS = 2
PREFIXES = ("attn_", "dense_")


def split_base_params(params):
    """Return the unmodified flat dictionary for the embedded multilayer MLP."""
    return {key: value for key, value in params.items()
            if not key.startswith(PREFIXES)}


def _variant(params):
    attention = any(key.startswith("attn_") for key in params)
    dense = any(key.startswith("dense_") for key in params)
    if attention == dense:
        raise ValueError("Require exactly one structured correction prefix")
    return "attention" if attention else "dense"


def init_params(base, variant, seed=1729):
    """Add a correction whose initially zero readout preserves every coefficient."""
    if variant not in ("attention", "dense"):
        raise ValueError("variant must be 'attention' or 'dense'")
    multilayer_model.hidden_layers(base)
    params = {key: jnp.array(value, copy=True) for key, value in base.items()}
    dtype = params["W"].dtype
    norb = params["W"].shape[0]
    if dtype.kind != "f" or norb < 2:
        raise ValueError("Require real floating parameters and at least two orbitals")
    key = jax.random.PRNGKey(seed)

    def normal(shape, scale):
        nonlocal key
        key, draw = jax.random.split(key)
        return scale * jax.random.normal(draw, shape, dtype=dtype)

    if variant == "attention":
        width, heads, ff = ATTENTION_WIDTH, ATTENTION_HEADS, ATTENTION_FF_WIDTH
        params["attn_embed_W"] = normal((3, width), .8 / np.sqrt(3))
        params["attn_embed_b"] = normal((width,), .1)
        for block in range(BLOCKS):
            prefix = f"attn_{block}_"
            for norm in ("norm1", "norm2"):
                params[prefix + norm + "_scale"] = jnp.ones((width,), dtype=dtype)
                params[prefix + norm + "_bias"] = jnp.zeros((width,), dtype=dtype)
            params[prefix + "qkv_W"] = normal((width, 3 * width), .8 / np.sqrt(width))
            params[prefix + "qkv_b"] = normal((3 * width,), .05)
            params[prefix + "mix_W"] = normal((width, width), .8 / np.sqrt(width))
            params[prefix + "mix_b"] = jnp.zeros((width,), dtype=dtype)
            params[prefix + "pair_W"] = normal((4, heads), .1)
            params[prefix + "ff1_W"] = normal((width, ff), .8 / np.sqrt(width))
            params[prefix + "ff1_b"] = normal((ff,), .05)
            params[prefix + "ff2_W"] = normal((ff, width), .8 / np.sqrt(ff))
            params[prefix + "ff2_b"] = jnp.zeros((width,), dtype=dtype)
        params["attn_head_W"] = jnp.zeros((2 * width, 2), dtype=dtype)
        params["attn_head_b"] = jnp.zeros((2,), dtype=dtype)
    else:
        width = DENSE_WIDTH
        params["dense_embed_W"] = normal((norb, width), .8 / np.sqrt(norb))
        params["dense_embed_b"] = normal((width,), .1)
        for block in range(BLOCKS):
            params[f"dense_{block}_W"] = normal((width, width), .8 / np.sqrt(width))
            params[f"dense_{block}_b"] = normal((width,), .05)
        params["dense_head_W"] = jnp.zeros((width, 2), dtype=dtype)
        params["dense_head_b"] = jnp.zeros((2,), dtype=dtype)
    architecture(params)  # Validate names and shapes before returning a model.
    return params


def _rms_norm(hidden, scale, bias):
    return hidden * jax.lax.rsqrt(jnp.mean(hidden**2, axis=-1, keepdims=True) + 1e-6) * scale + bias


def correction_output(params, occupation):
    """Two real correction outputs: log magnitude and phase in radians."""
    occupation = jnp.asarray(occupation, dtype=params["W"].dtype)
    if _variant(params) == "dense":
        hidden = jnp.tanh((2.0 * occupation - 1.0) @ params["dense_embed_W"]
                          + params["dense_embed_b"])
        for block in range(BLOCKS):
            hidden = (hidden + jnp.tanh(hidden @ params[f"dense_{block}_W"]
                                       + params[f"dense_{block}_b"])) / jnp.sqrt(2.0)
        return hidden @ params["dense_head_W"] + params["dense_head_b"]

    norb = occupation.shape[0]
    coordinate = jnp.linspace(-1.0, 1.0, norb, dtype=params["W"].dtype)
    tokens = jnp.stack((occupation, coordinate, coordinate**2), axis=-1)
    hidden = jnp.tanh(tokens @ params["attn_embed_W"] + params["attn_embed_b"])
    difference = coordinate[:, None] - coordinate[None, :]
    pairs = jnp.stack((difference, jnp.abs(difference), difference**2,
                       coordinate[:, None] * coordinate[None, :]), axis=-1)
    width, heads = ATTENTION_WIDTH, ATTENTION_HEADS
    head_width = width // heads
    for block in range(BLOCKS):
        prefix = f"attn_{block}_"
        normalized = _rms_norm(hidden, params[prefix + "norm1_scale"], params[prefix + "norm1_bias"])
        qkv = (normalized @ params[prefix + "qkv_W"] + params[prefix + "qkv_b"]).reshape(
            (norb, 3, heads, head_width))
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        logits = jnp.einsum("ahd,bhd->hab", query, key) / jnp.sqrt(float(head_width))
        logits = logits + jnp.einsum("abf,fh->hab", pairs, params[prefix + "pair_W"])
        weights = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("hab,bhd->ahd", weights, value).reshape((norb, width))
        hidden = (hidden + attended @ params[prefix + "mix_W"]
                  + params[prefix + "mix_b"]) / jnp.sqrt(2.0)
        normalized = _rms_norm(hidden, params[prefix + "norm2_scale"], params[prefix + "norm2_bias"])
        feed_forward = jnp.tanh(normalized @ params[prefix + "ff1_W"] + params[prefix + "ff1_b"])
        hidden = (hidden + feed_forward @ params[prefix + "ff2_W"]
                  + params[prefix + "ff2_b"]) / jnp.sqrt(2.0)
    occupied = jnp.sum(occupation[:, None] * hidden, axis=0) / jnp.maximum(jnp.sum(occupation), 1.0)
    empty = jnp.sum((1.0 - occupation[:, None]) * hidden, axis=0) / jnp.maximum(jnp.sum(1.0 - occupation), 1.0)
    pooled = jnp.concatenate((occupied, empty))
    return pooled @ params["attn_head_W"] + params["attn_head_b"]


def log_psi(params, occupation):
    """Complex log coefficient in the same canonical LLL Slater basis."""
    base = multilayer_model.log_psi(split_base_params(params), occupation)
    output = correction_output(params, occupation)
    return base + output[0] + 1j * output[1]


batch_log_psi = jax.vmap(log_psi, in_axes=(None, 0))


def score_tree(params, occupations):
    """Complex logarithmic derivatives for every trainable real parameter."""
    real = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).real), in_axes=(None, 0))(params, occupations)
    imag = jax.vmap(jax.grad(lambda p, n: log_psi(p, n).imag), in_axes=(None, 0))(params, occupations)
    return jax.tree.map(lambda a, b: a + 1j * b, real, imag)


def correction_parameter_groups(params):
    """Static correction key groups for checking head and hidden activation."""
    prefix = "attn_" if _variant(params) == "attention" else "dense_"
    head = tuple(sorted(key for key in params if key.startswith(prefix + "head_")))
    internal = tuple(sorted(key for key in params if key.startswith(prefix) and key not in head))
    return {"head": head, "internal": internal, "readout": head, "hidden": internal,
            "base": tuple(sorted(split_base_params(params)))}


def architecture(params):
    """Describe and validate the static architecture without inspecting values."""
    variant = _variant(params)
    base = split_base_params(params)
    layers = multilayer_model.hidden_layers(base)
    if base["W"].ndim != 2:
        raise ValueError("Base input weights must be a matrix")
    norb = base["W"].shape[0]
    base_width = base["W"].shape[1]
    base_shapes = {"W": (norb, base_width), "b": (base_width,),
                   "out": (base_width, 2), "bias": (2,)}
    for layer in range(2, layers + 1):
        base_shapes.update({f"W{layer}": (base_width, base_width),
                            f"b{layer}": (base_width,), f"out{layer}": (base_width, 2)})
    if any(base[key].shape != shape for key, shape in base_shapes.items()):
        raise ValueError("Base hidden-layer parameter shapes disagree")
    if variant == "attention":
        width, heads, ff = ATTENTION_WIDTH, ATTENTION_HEADS, ATTENTION_FF_WIDTH
        shapes = {"attn_embed_W": (3, width), "attn_embed_b": (width,),
                  "attn_head_W": (2 * width, 2), "attn_head_b": (2,)}
        for block in range(BLOCKS):
            prefix = f"attn_{block}_"
            for norm in ("norm1", "norm2"):
                shapes[prefix + norm + "_scale"] = (width,)
                shapes[prefix + norm + "_bias"] = (width,)
            shapes.update({prefix + "qkv_W": (width, 3 * width), prefix + "qkv_b": (3 * width,),
                           prefix + "mix_W": (width, width), prefix + "mix_b": (width,),
                           prefix + "pair_W": (4, heads), prefix + "ff1_W": (width, ff),
                           prefix + "ff1_b": (ff,), prefix + "ff2_W": (ff, width), prefix + "ff2_b": (width,)})
    else:
        width = DENSE_WIDTH
        shapes = {"dense_embed_W": (norb, width), "dense_embed_b": (width,),
                  "dense_head_W": (width, 2), "dense_head_b": (2,)}
        for block in range(BLOCKS):
            shapes[f"dense_{block}_W"] = (width, width)
            shapes[f"dense_{block}_b"] = (width,)
    correction_keys = set(params) - set(base)
    if correction_keys != set(shapes) or any(params[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("Structured correction names/shapes disagree with the architecture")
    added = int(sum(params[key].size for key in correction_keys))
    base_count = int(sum(value.size for value in base.values()))
    total_count = int(sum(value.size for value in params.values()))
    return {"variant": variant, "base_layers": layers, "base_width": int(base["W"].shape[1]),
            "base_hidden": int(base["W"].shape[1]),
            "orbitals": int(norb), "residual_blocks": BLOCKS, "correction_width": width,
            "attention_heads": ATTENTION_HEADS if variant == "attention" else 0,
            "feed_forward_width": ATTENTION_FF_WIDTH if variant == "attention" else None,
            "base_parameter_count": base_count, "correction_parameter_count": added,
            "total_parameter_count": total_count,
            "added_parameter_count": added, "parameter_count": total_count,
            "zero_initial_readout": True,
            "pair_features": ["m_difference_over_Q", "abs_m_difference_over_Q", "squared_m_difference_over_Q", "m_product_over_Q_squared"] if variant == "attention" else [],
            "scope": "Trainable occupation-coefficient correction in the unchanged LLL Slater basis; no analytic Laughlin factor or L=0 projection."}
