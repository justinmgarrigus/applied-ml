"""Weighted sampled angular-head SPRING; no ED or full-sector inputs.

Root-selection probabilities are frozen data. Every ratio and every
projective metric block includes the same inverse-selection weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'experiments')]
from fqh_spring.physics import SphereOperator
from learned_sector_operators import RootConnectionCache
from residual_geometry_operators import operator_batch_from_roots
import learned_sector_nonlinear_model as network

SOURCE_RUN = PROJECT/'results/ne7_nonlinear_sector_2layer_20261116/runs/angular_1_seed71'
DATA_KEYS = ('occupations', 'root_indices', 'H_indices', 'H_coefficients',
             'K_indices', 'K_coefficients', 'root_weights')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source():
    config_path = SOURCE_RUN/'config.json'
    checkpoint = SOURCE_RUN/'checkpoints/step_001000.npz'
    config = json.loads(config_path.read_text())
    with np.load(checkpoint, allow_pickle=False) as saved:
        flat = np.asarray(saved['parameters'])
        previous = np.asarray(saved['previous_direction'])
        source_step = int(saved['step'])
    params = {}; offset = 0
    for name in config['parameter_names']:
        shape = config['parameter_shapes'][name]
        size = int(np.prod(shape))
        params[name] = jnp.asarray(flat[offset:offset+size].reshape(shape))
        offset += size
    if offset != flat.size or flat.size != 294464:
        raise ValueError('Unexpected source parameter layout')
    return params, jnp.asarray(previous), dict(
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha(checkpoint),
        source_config=str(config_path), source_config_sha256=sha(config_path),
        source_step=source_step, source_architecture=config['architecture'])


def prepare_batch(operator, roots, rng, cache, h_draws=4, root_weights=None):
    sample = operator_batch_from_roots(operator, roots, rng, cache=cache,
        h_samples_per_root=h_draws, h_pad_to=h_draws, k_pad_to=43)
    n = len(sample['occupations'])
    padded = next((size for size in (768, 1024, 1504, 1600) if size >= n), None)
    if padded is None:
        raise ValueError('Sampled NN union exceeds bounded N7 data-batch capacity')
    occupation = np.asarray(sample['occupations'])
    sample['occupations'] = np.concatenate((occupation,
        np.repeat(occupation[:1], padded-n, axis=0)))
    sample['root_weights'] = np.ones(len(roots), dtype=np.float64) if root_weights is None else np.asarray(root_weights, dtype=np.float64)
    if sample['root_weights'].shape != (len(roots),) or not np.all(sample['root_weights'] > 0):
        raise ValueError('Every drawn root needs a positive inverse-selection weight')
    data = {key: jnp.asarray(sample[key]) for key in DATA_KEYS}
    coverage = dict(sampled_roots=len(roots), unique_NN_inputs=n, padded_NN_rows=padded)
    for key in ('unique_root_configurations', 'cached_root_rows',
                'cumulative_unique_sampled_roots', 'cumulative_unique_amplitude_configurations'):
        coverage[key] = int(sample['coverage'][key])
    return data, sample, coverage


class SampledProblem:
    def __init__(self, params, *, gamma=1., eta=.001, mu=.9,
                 parameter_radius=.015, state_radius=.01, cg_maxiter=384,
                 cg_tol=1e-6):
        self.flat, self.unravel = ravel_pytree(params)
        self.op = SphereOperator(7, flux=18, v3=0., two_lz=0)
        counts = [SphereOperator(7, flux=18, v3=0., two_lz=2*L).dimension()
                  for L in range(43)] + [0]
        self.levels = np.asarray([L for L in range(43) if counts[L] > counts[L+1]], dtype=np.int32)
        self.targets = jnp.asarray(self.levels*(self.levels+1), dtype=jnp.float64)
        self.S = len(self.levels); self.P = int(self.flat.size)
        self.gamma = gamma; self.eta = eta; self.mu = mu
        self.parameter_radius = parameter_radius; self.state_radius = state_radius
        self.cg_maxiter = cg_maxiter; self.cg_tol = cg_tol
        if self.S != 42 or self.op.dimension() != 1656:
            raise ValueError('Wrong no-wall N7 sector')
        self.force_stats = jax.jit(self._force_stats)
        self.force_parts = jax.jit(self._force_parts)
        self.predict = jax.jit(self._predict)
        self.step = jax.jit(self._step)
        self.validation_change = jax.jit(self._validation_change)
        self.metric_action = jax.jit(self._metric_action)
        self.sketch = jax.jit(self._sketch)

    def fields(self, t, occupations):
        return network.amplitudes(self.unravel(t), occupations)

    def _parts(self, t, data):
        heads, _, mixing, _ = self.fields(t, data['occupations'])
        root = heads[data['root_indices']]
        hhead = jnp.sum(data['H_coefficients'][:, :, None]*heads[data['H_indices']], axis=1)
        khead = jnp.sum(data['K_coefficients'][:, :, None]*heads[data['K_indices']], axis=1)
        coherent = root@mixing; hcoherent = hhead@mixing
        w = jax.lax.stop_gradient(data['root_weights'])
        denominator = jnp.sum(w*abs(coherent)**2)
        head_denominator = jnp.sum(w[:, None]*abs(root)**2, axis=0)
        energy = jnp.sum(w*jnp.conj(coherent)*hcoherent).real/jnp.maximum(denominator, 1e-280)
        residual = (khead-root*self.targets[None, :])/630.
        per_head = jnp.sum(w[:, None]*abs(residual)**2, axis=0)/jnp.maximum(head_denominator, 1e-280)
        angular = jnp.mean(per_head)
        stats = dict(sampled_energy=energy, sampled_angular_loss=angular,
            sampled_total_loss=energy+self.gamma*angular,
            root_importance_ess=denominator**2/jnp.maximum(jnp.sum((w*abs(coherent)**2)**2), 1e-280),
            selection_weight_ess=jnp.sum(w)**2/jnp.sum(w**2),
            coherent_root_norm_squared=denominator,
            minimum_head_root_norm_squared=jnp.min(head_denominator),
            maximum_head_root_norm_squared=jnp.max(head_denominator))
        return energy, angular, stats, root, coherent, hcoherent, residual

    def loss(self, t, data):
        e, a, stats, *_ = self._parts(t, data)
        return e+self.gamma*a, stats

    def _force_stats(self, t, data):
        (_, stats), gradient = jax.value_and_grad(self.loss, has_aux=True)(t, data)
        return .5*gradient, stats

    def _force_parts(self, t, data):
        energy_force = .5*jax.grad(lambda p: self._parts(p, data)[0])(t)
        angular_force = .5*self.gamma*jax.grad(lambda p: self._parts(p, data)[1])(t)
        return energy_force, angular_force, self.loss(t, data)[1]

    def _predict(self, t, data):
        e, a, stats, root, coherent, hcoherent, residual = self._parts(t, data)
        local_e = hcoherent/jnp.where(abs(coherent) > 1e-100, coherent, 1e-100+0j)
        angular_local = jnp.mean(abs(residual)**2/jnp.maximum(abs(root)**2, 1e-200), axis=1)
        priority = jnp.sqrt(abs(local_e-e)**2+self.gamma*angular_local+1e-12)
        return dict(root_residual=priority, energy_local_real=local_e.real,
                    angular_local=angular_local, **stats)

    def sample_state(self, t, roots, weights):
        heads, psi, _, _ = self.fields(t, roots)
        w = jax.lax.stop_gradient(weights)
        coherent = jnp.sqrt(w)*psi/jnp.sqrt(jnp.maximum(jnp.sum(w*abs(psi)**2), 1e-280))
        normalized = jnp.sqrt(w[:, None])*heads/jnp.sqrt(jnp.maximum(jnp.sum(w[:, None]*abs(heads)**2, axis=0), 1e-280))[None, :]
        z = jnp.concatenate((coherent, normalized.ravel()))
        return jnp.concatenate((z.real, z.imag))

    def _metric(self, t, roots, weights):
        dimension = len(roots)*(self.S+1)
        unpack = lambda q: q[:dimension]+1j*q[dimension:]
        pack = lambda z: jnp.concatenate((z.real, z.imag))
        q, push = jax.linearize(lambda u: self.sample_state(u, roots, weights), t)
        pull = jax.linear_transpose(push, jnp.zeros_like(t))
        z = unpack(q); coherent = z[:len(roots)]; heads = z[len(roots):].reshape(len(roots), self.S)
        def project(v):
            v0 = v[:len(roots)]; vh = v[len(roots):].reshape(len(roots), self.S)
            v0 = v0-coherent*jnp.vdot(coherent, v0)
            vh = (vh-heads*jnp.sum(heads.conj()*vh, axis=0)[None, :])/self.S
            return jnp.concatenate((v0, vh.ravel()))
        def metric(v):
            return pull(pack(project(unpack(push(v)))))[0]
        return metric

    def _metric_action(self, t, data, vector):
        roots = data['occupations'][data['root_indices']]
        return self._metric(t, roots, data['root_weights'])(vector)

    def _validation_change(self, old, new, data):
        dimension = len(data['root_indices'])*(self.S+1)
        roots = data['occupations'][data['root_indices']]
        def state(t):
            a = self.sample_state(t, roots, data['root_weights'])
            return a[:dimension]+1j*a[dimension:]
        before = state(old); after = state(new)
        B = len(roots)
        physical = jnp.sqrt(jnp.maximum(1.-abs(jnp.vdot(before[:B], after[:B]))**2, 0.))
        h0 = before[B:].reshape(B, self.S); h1 = after[B:].reshape(B, self.S)
        head_d2 = jnp.maximum(1.-abs(jnp.sum(h0.conj()*h1, axis=0))**2, 0.)
        augmented = jnp.sqrt(physical**2+jnp.mean(head_d2))
        j0 = self.loss(old, data)[0]; j1 = self.loss(new, data)[0]
        return dict(heldout_loss_before=j0, heldout_loss_after=j1,
                    heldout_delta=j1-j0, heldout_coherent_distance=physical,
                    heldout_augmented_distance=augmented)

    def _step(self, t, memory, data, damping, validation_data, check_validation):
        (objective, stats), gradient = jax.value_and_grad(self.loss, has_aux=True)(t, data)
        force = .5*gradient
        roots = data['occupations'][data['root_indices']]
        metric = self._metric(t, roots, data['root_weights'])
        operator = lambda v: metric(v)+damping*v
        rhs = force+damping*self.mu*memory
        start = self.mu*memory; r = rhs-operator(start); rr = r@r
        tol = jnp.maximum(self.cg_tol*jnp.linalg.norm(rhs), 1e-12)
        def condition(c): return (c[0] < self.cg_maxiter)&(c[4] > tol**2)
        def body(c):
            iteration, estimate, residual, p, squared = c
            product = operator(p); alpha = squared/jnp.maximum(p@product, 1e-300)
            estimate = estimate+alpha*p; rn = residual-alpha*product; ns = rn@rn
            return iteration+1, estimate, rn, rn+(ns/jnp.maximum(squared, 1e-300))*p, ns
        it, direction, _, _, _ = jax.lax.while_loop(condition, body,
            (jnp.int32(0), start, r, r, rr))
        equation_residual = jnp.linalg.norm(operator(direction)-rhs)
        norm = jnp.linalg.norm(direction)
        metric_norm = jnp.sqrt(jnp.maximum(direction@metric(direction), 0.))
        alpha = jnp.minimum(self.eta, self.parameter_radius/(norm+1e-30))
        alpha = jnp.minimum(alpha, self.state_radius/(metric_norm+1e-30))
        empty = {k: jnp.asarray(0., dtype=jnp.float64) for k in
                 ('heldout_loss_before', 'heldout_loss_after', 'heldout_delta',
                  'heldout_coherent_distance', 'heldout_augmented_distance')}
        def candidate(a):
            trial = t-a*direction; value, details = self.loss(trial, data)
            guard = jax.lax.cond(check_validation,
                lambda _: self._validation_change(t, trial, validation_data), lambda _: empty, None)
            bad = ((value > objective+1e-10)|(~jnp.isfinite(value))|
                (check_validation & ((guard['heldout_delta'] > 1e-10)|
                 (guard['heldout_augmented_distance'] > self.state_radius)|
                 (~jnp.isfinite(guard['heldout_loss_after']))|
                 (~jnp.isfinite(guard['heldout_coherent_distance']))|
                 (~jnp.isfinite(guard['heldout_augmented_distance'])))))
            return trial, value, details, guard, bad
        trial, value, details, guard, bad = candidate(alpha)
        def bc(c): return (c[0] < 8)&c[5]
        def bb(c):
            a = .5*c[6]; p, v, d, g, b = candidate(a)
            return c[0]+1, p, v, d, g, b, a
        backs, trial, value, details, guard, reject, alpha = jax.lax.while_loop(
            bc, bb, (jnp.int32(0), trial, value, details, guard, bad, alpha))
        updated = jnp.where(reject, t, trial)
        memory_new = jnp.where(reject, memory, direction)
        details = jax.tree.map(lambda old, new: jnp.where(reject, old, new), stats, details)
        actual_guard = jax.tree.map(lambda g: jnp.where(reject, 0., g), guard)
        # A rejected step is unchanged; preserve before/after held-out loss.
        for key in ('heldout_loss_before', 'heldout_loss_after'):
            actual_guard[key] = jnp.where(reject, guard['heldout_loss_before'], guard[key])
        named = self.unravel(gradient)
        metrics = details|actual_guard|dict(
            preupdate_sampled_energy=stats['sampled_energy'],
            preupdate_sampled_angular_loss=stats['sampled_angular_loss'],
            preupdate_sampled_total_loss=objective, cg_iterations=it,
            cg_absolute_residual=equation_residual, cg_absolute_tolerance=tol,
            cg_converged=equation_residual <= 1.01*tol,
            raw_direction_norm=norm, force_norm=jnp.linalg.norm(force),
            effective_eta=jnp.where(reject, 0., alpha), update_norm=jnp.linalg.norm(updated-t),
            backtracks=backs, rejected=reject, predicted_augmented_distance=alpha*metric_norm,
            heldout_checked=check_validation, damping=damping,
            private_hidden_gradient_norm=jnp.sqrt(sum(jnp.sum(named[k]**2) for k in
                ('sector_W1', 'sector_b1', 'sector_W2', 'sector_b2'))),
            private_output_gradient_norm=jnp.sqrt(sum(jnp.sum(named[k]**2) for k in
                ('sector_out', 'sector_bias'))))
        return updated, memory_new, metrics

    def _sketch(self, t, occupations, probes):
        def log_fields(u):
            logs, relative, mixing = network.components(self.unravel(u), occupations)
            r = jnp.where(abs(relative) > 1e-100, relative, 1e-100+0j)
            c = relative@mixing
            c = jnp.where(abs(c) > 1e-100, c, 1e-100+0j)
            return jnp.concatenate(((logs+jnp.log(c))[:, None], logs[:, None]+jnp.log(r)), axis=1)
        _, push = jax.linearize(log_fields, t)
        values = jax.vmap(push)(probes)
        values = jnp.transpose(values, (1, 2, 0)).reshape(len(occupations), -1)
        heads, psi, _, _ = self.fields(t, occupations)
        mass0 = abs(psi)**2/jnp.maximum(jnp.mean(abs(psi)**2), 1e-200)
        mass_h = jnp.mean(abs(heads)**2/jnp.maximum(jnp.mean(abs(heads)**2, axis=0), 1e-200)[None, :], axis=1)
        return values, .5*(mass0+mass_h)
