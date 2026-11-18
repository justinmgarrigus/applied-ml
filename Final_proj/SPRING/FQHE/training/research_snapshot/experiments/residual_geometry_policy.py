"""Finite-pool importance selection using lagged residuals and score sketches.

This is a heuristic priority, not a full QGT or an exact optimal proposal.
Selection is random with replacement; inverse probabilities remain explicit.
"""
from __future__ import annotations

from collections import OrderedDict
import time

import jax
import jax.numpy as jnp
import numpy as np

from learned_sector_operators import occupation_bits
from fqh_spring.geometry_sampling import geometry_novelty


class ResidualGeometryPolicy:
    def __init__(self, problem, probes, anchors, *, exploration=.2, max_history=512,
                 seed_residuals=()):
        self.problem = problem; self.probes = jnp.asarray(probes)
        self.anchors = np.asarray(anchors, dtype=np.int8)
        self.exploration = exploration; self.max_history = max_history
        self.history = OrderedDict()
        for row in seed_residuals:
            self.history[int(row['bits'])] = (float(row['residual']), 0)
        while len(self.history) > max_history: self.history.popitem(last=False)
        self.novelty = jax.jit(lambda s, a: geometry_novelty(s, a,
            diagonal_floor=1e-6, ridge=1e-3))
        self.compiled_sketch = None

    def compile(self, t, pool):
        roots = jnp.asarray(np.concatenate((pool, self.anchors)))
        started = time.perf_counter()
        self.compiled_sketch = self.problem.sketch.lower(t, roots, self.probes).compile()
        values, _ = self.compiled_sketch(t, roots, self.probes)
        self.novelty(values[:len(pool)], values[len(pool):]).block_until_ready()
        return time.perf_counter()-started

    def select(self, t, pool, rng, update, batch_size=32):
        started = time.perf_counter()
        roots = jnp.asarray(np.concatenate((pool, self.anchors)))
        values, mass = self.compiled_sketch(t, roots, self.probes)
        novelty = np.asarray(self.novelty(values[:len(pool)], values[len(pool):]))
        mass = np.asarray(mass[:len(pool)])
        median = max(float(np.median([v[0] for v in self.history.values()])), 1e-8) if self.history else 1.
        factors = []; hits = 0
        for row in pool:
            value = self.history.get(occupation_bits(row))
            if value is None: factors.append(1.); continue
            hits += 1
            residual, age = value
            freshness = np.exp(-max(update-age, 0)/200.)
            factors.append(1.+freshness*(np.clip(residual/median, .25, 4.)-1.))
        residual_factor = np.asarray(factors)
        priority = np.clip(mass, .25, 4.)*residual_factor*(.5+novelty)
        probabilities = self.exploration/len(pool)+(1.-self.exploration)*priority/priority.sum()
        selected = rng.choice(len(pool), size=batch_size, replace=True, p=probabilities)
        weights = 1./(len(pool)*probabilities[selected])
        record = dict(candidate_pool=np.asarray(pool), selection_indices=selected.astype(np.int32),
            selection_probabilities=probabilities, priority_scores=priority,
            geometry_novelty=novelty, residual_factor=residual_factor, mass_proxy=mass)
        stats = dict(pool_size=len(pool), pool_residual_cache_hits=hits,
            pool_mean_novelty=float(novelty.mean()), pool_min_probability=float(probabilities.min()),
            pool_max_probability=float(probabilities.max()),
            pool_unique_selected_indices=len(np.unique(selected)), priority_seconds=time.perf_counter()-started)
        return pool[selected], weights, record, stats

    def observe(self, roots, residuals, update):
        combined = {}
        for row, residual in zip(roots, np.asarray(residuals)):
            key = occupation_bits(row)
            if np.isfinite(residual): combined.setdefault(key, []).append(float(residual))
        for key, values in combined.items():
            self.history[key] = (float(np.median(values)), update)
            self.history.move_to_end(key)
        while len(self.history) > self.max_history: self.history.popitem(last=False)
        self.anchors = np.asarray(roots, dtype=np.int8)
