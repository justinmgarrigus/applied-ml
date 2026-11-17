"""Born-distribution Metropolis sampling without artificial tempering.

Both local pair moves and uniform-sector independence refreshes target
p_theta(n) proportional to |psi_theta(n)|^2. There is no numerical or physical
temperature parameter. Independent chains still require equilibration and
mixing; their outputs need not be independent samples.

Sector-preserving Metropolis sampling without basis enumeration.

Orbitals are indexed 0,...,flux.  Replacing two occupied orbitals by two with
the same index sum preserves particle count and Lz.  The uniform source pair
and uniform all-orbital target pair give identical forward/reverse proposal
probabilities, including rejection when another particle occupies the target.
"""

from functools import partial

import jax
import jax.numpy as jnp


def make_sampler(log_fn, ne, flux):
    """Build a jitted independent-walker pair-hopping sampler.

    The returned function accepts (params, walkers, key, sweeps) and returns
    (walkers, new_key, accept_rate). One sweep means ne attempted moves per
    walker. The target is exactly exp(2 Re log_fn), the Born distribution.

    Input walkers must be binary rows of length flux+1, with ne particles in
    the desired common index-sum sector.  Different rows use independent
    random draws.  accept_rate counts accepted *changed* states divided by
    all attempts, excluding self proposals from the numerator.
    """
    if ne < 2 or ne > flux + 1:
        raise ValueError("require 2 <= ne <= flux+1")
    batch_log = jax.vmap(log_fn, in_axes=(None, 0))

    @jax.jit
    def sample(params, walkers, key, sweeps):
        rows = jnp.arange(walkers.shape[0])
        old_log = 2.0 * batch_log(params, walkers).real

        def attempt(_, carry):
            current, current_log, current_key, moved = carry
            current_key, source_a_key, source_b_key, target_key, accept_key = jax.random.split(
                current_key, 5
            )
            occupied = jax.vmap(lambda row: jnp.nonzero(row, size=ne)[0])(current)
            rank_a = jax.random.randint(source_a_key, rows.shape, 0, ne)
            other_rank = jax.random.randint(source_b_key, rows.shape, 0, ne - 1)
            rank_b = other_rank + (other_rank >= rank_a)
            source_a = occupied[rows, rank_a]
            source_b = occupied[rows, rank_b]
            pair_sum = source_a + source_b

            # All distinct unordered orbital pairs (a,b) with a+b=pair_sum.
            lower = jnp.maximum(0, pair_sum - flux)
            upper = (pair_sum - 1) // 2
            count = upper - lower + 1
            target_a = lower + jax.random.randint(target_key, rows.shape, 0, count)
            target_b = pair_sum - target_a

            cleared = current.at[rows, source_a].set(0).at[rows, source_b].set(0)
            valid = (cleared[rows, target_a] == 0) & (cleared[rows, target_b] == 0)
            proposed = cleared.at[rows, target_a].set(1).at[rows, target_b].set(1)
            proposed = jnp.where(valid[:, None], proposed, current)
            proposed_log = 2.0 * batch_log(params, proposed).real
            uniform = jax.random.uniform(accept_key, rows.shape, dtype=current_log.dtype)
            accepted = valid & jnp.isfinite(proposed_log) & (
                jnp.log(uniform) < jnp.minimum(0.0, proposed_log - current_log)
            )
            changed = jnp.any(proposed != current, axis=1)
            current = jnp.where(accepted[:, None], proposed, current)
            current_log = jnp.where(accepted, proposed_log, current_log)
            moved += jnp.sum(accepted & changed, dtype=jnp.int32)
            return current, current_log, current_key, moved

        walkers, _, key, moved = jax.lax.fori_loop(
            0, ne * sweeps, attempt,
            (walkers, old_log, key, jnp.asarray(0, dtype=jnp.int32)),
        )
        attempts = walkers.shape[0] * ne * sweeps
        rate = moved / jnp.maximum(attempts, 1)
        return walkers, key, rate

    return sample


@partial(jax.jit, static_argnums=(0,))
def global_refresh(log_fn, params, walkers, proposals, key):
    """One independence-MH refresh using externally drawn uniform-sector rows.

    Each proposal must be an independent uniform draw from the SAME sector as
    its walker, e.g. from the runner's exact dynamic-programming initializer.
    Uniform independence proposals have symmetric q, so no proposal-density
    ratio appears.  This assumption would fail for biased global proposals.
    Returns walkers, new key, and accepted changed-state fraction.
    """
    key, accept_key = jax.random.split(key)
    batch_log = jax.vmap(log_fn, in_axes=(None, 0))
    before = 2.0 * batch_log(params, walkers).real
    after = 2.0 * batch_log(params, proposals).real
    uniforms = jax.random.uniform(accept_key, before.shape, dtype=before.dtype)
    accepted = jnp.isfinite(after) & (
        jnp.log(uniforms) < jnp.minimum(0.0, after - before)
    )
    changed = jnp.any(proposals != walkers, axis=1)
    result = jnp.where(accepted[:, None], proposals, walkers)
    return result, key, jnp.mean(accepted & changed)
