"""Experimental geometry-guided candidate selection for SPRING.

Geometry is measured in centered log-wavefunction derivatives, not occupation
distance or overlaps between orthogonal Slater basis states.  This pilot uses
a *diagonal* reference quantum metric and a ridge-smoothed projector; it is not
the full quantum-geometric tensor.  All candidate and anchor derivatives must
be evaluated at the same parameters.  No ED labels enter these functions.

Sampling is with replacement and has known, strictly positive probabilities.
Inverse selection weights recover linear moments of the original candidate
pool in expectation.  Normalizing these weights, and estimating covariance
means, retains finite-sample ratio/covariance bias.  Selecting the farthest
rows deterministically would not have this correction guarantee.
"""
from __future__ import annotations

import numbers

import jax
import jax.numpy as jnp


def _score_arrays(scores, anchor_scores):
    scores, anchors = jnp.asarray(scores), jnp.asarray(anchor_scores)
    if (scores.ndim != 2 or anchors.ndim != 2 or scores.shape[0] < 1
            or anchors.shape[0] < 1 or scores.shape[1] < 1
            or scores.shape[1] != anchors.shape[1]):
        raise ValueError('Scores and anchors must be nonempty [samples,parameters] arrays')
    return scores, anchors


def geometry_novelty(scores, anchor_scores, *, diagonal_floor=1e-6, ridge=1e-6):
    """Return [pool] fractions outside a regularized anchor score subspace.

    Center both sets using the pool mean, whiten with the pool's diagonal
    quantum metric, and form the realified anchor rows A.  The smoothed
    projector is A.T (A A.T + tau I)^-1 A, with tau scaled to its mean Gram
    diagonal.  Its eigenvalues lie between zero and one.  The returned score
    is the squared residual fraction, adding the real and imaginary channels
    for each configuration.  It is bounded in [0,1].

    Scalar hyperparameters are Python configuration values, hence static when
    this function is called inside jit.  The tiny reference floor matters
    when a pool has identical rows; it keeps unseen anchor directions finite.
    """
    if diagonal_floor <= 0 or ridge <= 0:
        raise ValueError('diagonal_floor and ridge must be positive')
    scores, anchors = _score_arrays(scores, anchor_scores)
    center = jnp.mean(scores, axis=0)
    centered = scores - center
    diagonal = jnp.mean(jnp.abs(centered)**2, axis=0)
    reference_scale = jnp.maximum(jnp.mean(diagonal), 1e-12)
    diagonal = jnp.maximum(diagonal, diagonal_floor * reference_scale)
    whitened = centered / jnp.sqrt(diagonal)
    anchor = (anchors - center) / jnp.sqrt(diagonal)
    anchor_real = jnp.concatenate((anchor.real, anchor.imag), axis=0)
    candidate_real = jnp.concatenate((whitened.real, whitened.imag), axis=0)
    gram = anchor_real @ anchor_real.T
    gram_scale = jnp.maximum(jnp.trace(gram) / gram.shape[0], 1e-12)
    regularized = gram + ridge * gram_scale * jnp.eye(gram.shape[0], dtype=gram.dtype)
    cross = candidate_real @ anchor_real.T
    coefficients = jnp.linalg.solve(regularized, cross.T).T
    residual = candidate_real - coefficients @ anchor_real
    count = scores.shape[0]
    numerator = jnp.sum(residual[:count]**2 + residual[count:]**2, axis=1)
    denominator = jnp.sum(jnp.abs(whitened)**2, axis=1)
    fraction = jnp.where(denominator > 0, numerator / jnp.maximum(denominator, 1e-30), 0.)
    return jnp.clip(fraction, 0., 1.)


def selection_probabilities(novelty, *, mix=.8, novelty_floor=.01):
    """Mix uniform pool selection with geometry scores; all entries stay >0."""
    if not 0 <= mix <= 1 or novelty_floor <= 0:
        raise ValueError('Require 0 <= mix <= 1 and novelty_floor > 0')
    novelty = jnp.asarray(novelty)
    if novelty.ndim != 1 or novelty.shape[0] < 1:
        raise ValueError('novelty must be a nonempty vector')
    quality = jnp.clip(novelty, 0., 1.) + novelty_floor
    return (1. - mix) / novelty.size + mix * quality / jnp.sum(quality)


def select_candidates(key, probabilities, batch_size):
    """Sample indices with replacement and return their exact pool corrections.

    If a pool has M rows and pi gives index probabilities, raw_importance is
    1/(M*pi_i).  A batch mean of raw_importance*f has conditional expectation
    equal to the pool's original uniform average.  weights are instead
    normalized for the covariance/metric adapter; these are ratio estimates.
    """
    if (isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral)
            or batch_size < 1):
        raise ValueError('batch_size must be a positive integer')
    probabilities = jnp.asarray(probabilities)
    if probabilities.ndim != 1 or probabilities.shape[0] < 1:
        raise ValueError('probabilities must be a nonempty vector')
    indices = jax.random.choice(key, probabilities.size, shape=(int(batch_size),),
                                replace=True, p=probabilities)
    raw = 1. / (probabilities.size * probabilities[indices])
    return dict(indices=indices, probabilities=probabilities, raw_importance=raw,
                weights=raw / jnp.sum(raw))


def weighted_score_data(scores, weights):
    """Frozen signed complex rows for the existing JaQMC SPRING adapter.

    Its unweighted centering and division by sqrt(number_of_rows) then give
    exactly Re sum_i w_i (O_i-mean_w O)^* (O_i-mean_w O).  Signed duplicates
    add no information; they preserve the historical weighted-metric interface.
    """
    scores, weights = jnp.asarray(scores), jnp.asarray(weights)
    if (scores.ndim != 2 or weights.ndim != 1 or scores.shape[0] < 1
            or weights.shape[0] != scores.shape[0]):
        raise ValueError('scores [batch,parameters] and weights [batch] must agree')
    weights = weights / jnp.sum(weights)
    centered = scores - weights @ scores
    scaled = jnp.sqrt(scores.shape[0] * weights)[:, None] * centered
    return jax.lax.stop_gradient(jnp.concatenate((scaled, -scaled), axis=0))
