"""Tempered candidate weights and sequential score-space diversity.

Candidates come from q_beta(n) proportional to |psi(n)|**(2*beta).
Selection remains stochastic, WITH replacement and full support. At each
draw the known conditional index probability corrects the linear empirical
pool moments; the additional p/q_beta factor restores the Born objective.
Self-normalization and estimated covariance means retain finite-batch bias.

Geometry uses an importance-weighted diagonal Born metric. First remove a
ridge-smoothed previous-anchor component, then project sequentially selected
real/imaginary directions from that residual feature space. The anchor step
is regularized rather than an exact subspace projection; this is a diversity
heuristic, not a full-QGT or invariant principal-angle construction.
"""
from __future__ import annotations

import numbers

import jax
import jax.numpy as jnp

from .geometry_sampling import selection_probabilities


def _check_beta(beta):
    if not 0 < beta <= 1:
        raise ValueError('beta must be in (0,1]')


def _check_batch(batch_size):
    if (isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral)
            or batch_size < 1):
        raise ValueError('batch_size must be a positive integer')


def tempered_selection_weights(log_amplitudes, conditional_probabilities, *, beta, pool_size):
    """Return both-stage corrections for already selected pool indices.

    The unnormalized factor is |psi|**(2*(1-beta))/(M*pi_t), where pi_t
    is the conditional probability at its draw. Unknown proposal and Born
    normalizers cancel in the normalized ratio. Use ``weights`` or
    ``log_correction`` for numerical work: raw_correction can overflow under
    an extreme arbitrary global wavefunction-amplitude gauge.
    """
    _check_beta(beta)
    if isinstance(pool_size, bool) or not isinstance(pool_size, numbers.Integral) or pool_size < 1:
        raise ValueError('pool_size must be a positive integer')
    amplitudes, probability = map(jnp.asarray, (log_amplitudes, conditional_probabilities))
    if amplitudes.ndim != 1 or amplitudes.shape != probability.shape or amplitudes.size < 1:
        raise ValueError('Selected log amplitudes and probabilities must be equal nonempty vectors')
    log_correction = 2. * (1. - beta) * amplitudes - jnp.log(pool_size * probability)
    return dict(weights=jax.nn.softmax(log_correction), log_correction=log_correction,
                raw_correction=jnp.exp(log_correction))


def _residual_geometry(scores, anchor_scores, log_amplitudes, *, beta, diagfloor, ridge):
    _check_beta(beta)
    if diagfloor <= 0 or ridge <= 0:
        raise ValueError('diagfloor and ridge must be positive')
    scores, anchor_scores, log_amplitudes = map(jnp.asarray, (scores, anchor_scores, log_amplitudes))
    if (scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1
            or anchor_scores.ndim != 2 or anchor_scores.shape[0] < 1
            or anchor_scores.shape[1] != scores.shape[1]
            or log_amplitudes.shape != (scores.shape[0],)):
        raise ValueError('Expected scores [M,P], anchors [A,P], and pool log amplitudes [M]')
    pool_weights = jax.nn.softmax(2. * (1. - beta) * log_amplitudes)
    center = pool_weights @ scores
    centered = scores - center
    diagonal = pool_weights @ jnp.abs(centered)**2
    reference_scale = jnp.maximum(jnp.mean(diagonal), 1e-12)
    diagonal = jnp.maximum(diagonal, diagfloor * reference_scale)
    whitened = centered / jnp.sqrt(diagonal)
    anchors = (anchor_scores - center) / jnp.sqrt(diagonal)
    rows = jnp.concatenate((whitened.real, whitened.imag), axis=0)
    anchor_rows = jnp.concatenate((anchors.real, anchors.imag), axis=0)
    anchor_gram = anchor_rows @ anchor_rows.T
    scale = jnp.maximum(jnp.trace(anchor_gram) / anchor_gram.shape[0], 1e-12)
    regularized = anchor_gram + ridge * scale * jnp.eye(anchor_gram.shape[0], dtype=anchor_gram.dtype)
    cross = rows @ anchor_rows.T
    coefficients = jnp.linalg.solve(regularized, cross.T).T
    residual = rows - coefficients @ anchor_rows
    denominator = jnp.sum(jnp.abs(whitened)**2, axis=1)
    return residual, denominator


def _novelty_from_diagonal(diagonal, denominator):
    count = denominator.shape[0]
    numerator = jnp.maximum(diagonal[:count], 0.) + jnp.maximum(diagonal[count:], 0.)
    return jnp.clip(jnp.where(denominator > 0, numerator / jnp.maximum(denominator, 1e-30), 0.), 0., 1.)


def project_gram_direction(gram, index, *, reference_norm):
    """Gram matrix after removing one selected residual-score direction.

    This is the Gram-space equivalent of a Gram--Schmidt rank-one update.
    The relative cutoff avoids interpreting roundoff in an already removed
    direction as an additional unit vector.
    """
    column = gram[:, index]
    norm = jnp.maximum(gram[index, index], 0.)
    threshold = 1e-12 * jnp.maximum(reference_norm, 1e-30)
    factor = jnp.where(norm > threshold, 1. / jnp.maximum(norm, 1e-300), 0.)
    result = gram - factor * column[:, None] * column[None, :]
    return .5 * (result + result.T)


def select_sequential(key, scores, anchor_scores, batch_size, *, beta=.5,
                      log_amplitudes, mix=.8, floor=.01, diagfloor=1e-6, ridge=1e-6):
    """Select a full-support, conditionally reweighted diverse update batch.

    Scores and anchors must be evaluated at the same frozen theta. Centering
    uses one pool estimate of Born moments. After every stochastic selection,
    both real and imaginary residual directions enter the selected span.
    Repeated indices remain possible because removing them would invalidate
    the simple per-draw inverse-probability correction for the full pool.
    """
    _check_batch(batch_size)
    # Validate static mixture arguments even when called through jax.jit.
    if not 0 <= mix <= 1 or floor <= 0:
        raise ValueError('Require 0 <= mix <= 1 and floor > 0')
    residual, denominator = _residual_geometry(scores, anchor_scores, log_amplitudes,
                                               beta=beta, diagfloor=diagfloor, ridge=ridge)
    count = denominator.shape[0]
    gram = residual @ residual.T
    initial_novelty = _novelty_from_diagonal(jnp.diag(gram), denominator)

    def draw(carry, _):
        current, current_key = carry
        novelty = _novelty_from_diagonal(jnp.diag(current), denominator)
        probabilities = selection_probabilities(novelty, mix=mix, novelty_floor=floor)
        current_key, draw_key = jax.random.split(current_key)
        index = jax.random.choice(draw_key, count, p=probabilities)
        next_gram = project_gram_direction(current, index, reference_norm=denominator[index])
        next_gram = project_gram_direction(next_gram, count + index,
                                           reference_norm=denominator[index])
        diagnostics = (index, probabilities[index], novelty[index], jnp.mean(novelty),
                       jnp.min(probabilities), jnp.max(probabilities))
        return (next_gram, current_key), diagnostics

    (final_gram, final_key), values = jax.lax.scan(draw, (gram, key), None, length=int(batch_size))
    indices, conditional, selected_novelty, pool_novelty, minimum, maximum = values
    corrections = tempered_selection_weights(jnp.asarray(log_amplitudes)[indices], conditional,
                                              beta=beta, pool_size=count)
    return corrections | dict(
        indices=indices, conditional_probabilities=conditional,
        novelty_mean=jnp.mean(initial_novelty), selected_novelty_mean=jnp.mean(selected_novelty),
        conditional_pool_novelty_mean=jnp.mean(pool_novelty),
        final_novelty_mean=jnp.mean(_novelty_from_diagonal(jnp.diag(final_gram), denominator)),
        probability_min=jnp.min(minimum), probability_max=jnp.max(maximum), key=final_key)


def select_static_tempered(key, scores, anchor_scores, batch_size, *, beta=.5,
                           log_amplitudes, mix=.8, floor=.01, diagfloor=1e-6, ridge=1e-6):
    """Static previous-anchor novelty with the same tempered Born correction."""
    _check_batch(batch_size)
    residual, denominator = _residual_geometry(scores, anchor_scores, log_amplitudes,
                                               beta=beta, diagfloor=diagfloor, ridge=ridge)
    novelty = _novelty_from_diagonal(jnp.sum(residual**2, axis=1), denominator)
    probability = selection_probabilities(novelty, mix=mix, novelty_floor=floor)
    indices = jax.random.choice(key, probability.size, shape=(int(batch_size),), replace=True, p=probability)
    conditional = probability[indices]
    corrections = tempered_selection_weights(jnp.asarray(log_amplitudes)[indices], conditional,
                                              beta=beta, pool_size=probability.size)
    return corrections | dict(
        indices=indices, conditional_probabilities=conditional, probabilities=probability,
        novelty_mean=jnp.mean(novelty), selected_novelty_mean=jnp.mean(novelty[indices]),
        conditional_pool_novelty_mean=jnp.mean(novelty), final_novelty_mean=jnp.mean(novelty),
        probability_min=jnp.min(probability), probability_max=jnp.max(probability))
