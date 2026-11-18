"""Isolated numerical fallback for the unchanged pure-V1 Ne12 SPRING system.

Ordinary CG4096 is attempted first using the frozen metric/recurrence. Only
when its independently recomputed original residual fails does same-batch
fixed-diagonal PCG run, starting from that CG estimate. The preconditioner uses
32 separate deterministic output-probe pullbacks. The physical metric,
damping, RHS, tolerance, caps, guards and sampler are unchanged.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def projective_metric_factors(state_map,parameters,*,root_rows,head_count):
    """Return F,F^T,G where G equals the frozen normalized projective metric.

    state_map returns [Re(z),Im(z)], z=[coherent_B,heads_(B*S)], each
    coherent/head column normalized independently. All batch norms must be
    positive. F uses head weight 1/sqrt(S); G=F^T F uses head weight1/S.
    """
    if root_rows<1 or head_count<1:raise ValueError('Positive B and S required')
    dimension=root_rows*(head_count+1)
    q,push=jax.linearize(state_map,parameters)
    if q.shape!=(2*dimension,):raise ValueError('Unexpected packed normalized-state shape')
    pull=jax.linear_transpose(push,jnp.zeros_like(parameters))
    def unpack(v):return v[:dimension]+1j*v[dimension:]
    def pack(v):return jnp.concatenate((v.real,v.imag))
    state=unpack(q);coherent=state[:root_rows]
    heads=state[root_rows:].reshape((root_rows,head_count))
    def project_and_sqrtweight(v):
        coherent_v=v[:root_rows]
        heads_v=v[root_rows:].reshape((root_rows,head_count))
        coherent_v=coherent_v-coherent*jnp.vdot(coherent,coherent_v)
        heads_v=heads_v-heads*jnp.sum(heads.conj()*heads_v,axis=0)[None,:]
        heads_v=heads_v/jnp.sqrt(jnp.asarray(head_count,dtype=parameters.dtype))
        return jnp.concatenate((coherent_v,heads_v.ravel()))
    def factor_push(v):return pack(project_and_sqrtweight(unpack(push(v))))
    def factor_pull(v):return pull(pack(project_and_sqrtweight(unpack(v))))[0]
    def metric(v):return factor_pull(factor_push(v))
    return factor_push,factor_pull,metric,q


def nonnegative_metric_diagonal(factor_pull,parameters,*,output_dimension,probe_count=16,probe_seed=731904):
    """Estimate diag(F^T F) from squared output-Rademacher pullbacks.

    Separate deterministic probe keys do not consume experiment RNG streams.
    Probes are generated sequentially: O(P+D) working storage, no dense O/J.
    Freeze this result throughout one standard-PCG solve. No speed guarantee.
    """
    if probe_count<1 or output_dimension<1:raise ValueError('Positive probe dimensions required')
    key=jax.random.PRNGKey(probe_seed)
    def body(index,diagonal):
        signs=jax.random.rademacher(jax.random.fold_in(key,index),
            (output_dimension,),dtype=parameters.dtype)
        row=factor_pull(signs)
        return diagonal+row*row
    total=jax.lax.fori_loop(0,probe_count,body,jnp.zeros_like(parameters))
    return total/probe_count


def solve_fixed_diagonal_pcg(metric,rhs,start,*,damping,metric_diagonal,
                             maxiter=4096,relative_tolerance=1e-6,absolute_floor=1e-12):
    """Solve original A=G+damping*I, b=rhs, x0=start with fixed SPD M.

    The tolerance is computed in original units exactly as the frozen solver.
    The final acceptance test recomputes ||A*x-b||, with the same1.01 allowance.
    This pure solver does not apply caps or commit an optimization update.
    """
    if maxiter<1:raise ValueError('Positive solver ceiling required')
    if metric_diagonal.shape!=rhs.shape:raise ValueError('Diagonal shape mismatch')
    operator=lambda v:metric(v)+damping*v
    diagonal=damping+jnp.maximum(metric_diagonal,0.)
    tolerance=jnp.maximum(relative_tolerance*jnp.linalg.norm(rhs),absolute_floor)
    residual=rhs-operator(start)
    z=residual/diagonal;rz=residual@z;rr=residual@residual
    finite=(damping>0)&jnp.all(jnp.isfinite(diagonal))&jnp.all(diagonal>0)&jnp.isfinite(rr)&jnp.isfinite(rz)
    def condition(state):return (state[0]<maxiter)&(state[5]>tolerance**2)&state[6]
    def body(state):
        iteration,estimate,r,p,old_rz,_,healthy=state
        product=operator(p);curvature=p@product
        alpha=old_rz/jnp.maximum(curvature,1e-300)
        estimate=estimate+alpha*p
        after=r-alpha*product;after_z=after/diagonal
        after_rz=after@after_z;after_rr=after@after
        direction=after_z+(after_rz/jnp.maximum(old_rz,1e-300))*p
        healthy=healthy&(curvature>0)&jnp.isfinite(curvature)&jnp.isfinite(after_rr)&jnp.isfinite(after_rz)
        return iteration+1,estimate,after,direction,after_rz,after_rr,healthy
    result=jax.lax.while_loop(condition,body,(jnp.int32(0),start,residual,z,rz,rr,finite))
    iterations,estimate,_,_,_,recursive_squared,healthy=result
    original_residual=jnp.linalg.norm(operator(estimate)-rhs)
    converged=healthy&jnp.isfinite(original_residual)&(original_residual<=1.01*tolerance)
    return estimate,dict(iterations=iterations,original_residual=original_residual,
        original_tolerance=tolerance,recursive_residual=jnp.sqrt(recursive_squared),
        converged=converged,healthy=healthy,preconditioner_min=jnp.min(diagonal),
        preconditioner_max=jnp.max(diagonal))

from ne12_pure_v1_batch256_cg4096_runtime import (Ne12CG4096Problem,FixedAnchorPolicy,
    prepare_batch,angular_levels,DATA_KEYS,CONSERVATIVE_UNION_BOUND,NN_BUCKETS)


def ordinary_cg(operator,rhs,start,*,maxiter,tolerance):
    """Literal frozen ordinary-CG recurrence, with original residual receipt."""
    r=rhs-operator(start);rr=r@r
    def condition(c):return (c[0]<maxiter)&(c[4]>tolerance**2)
    def body(c):
        iteration,estimate,residual,p,squared=c
        product=operator(p);alpha=squared/jnp.maximum(p@product,1e-300)
        estimate=estimate+alpha*p;rn=residual-alpha*product;ns=rn@rn
        return iteration+1,estimate,rn,rn+(ns/jnp.maximum(squared,1e-300))*p,ns
    result=jax.lax.while_loop(condition,body,(jnp.int32(0),start,r,r,rr))
    return result[1],result[0],jnp.linalg.norm(operator(result[1])-rhs)


class Ne12PCGProblem(Ne12CG4096Problem):
    solver_identifier='ordinary_CG4096_then_fixed_output_probe_PCG8192_v1'

    def __init__(self,params,*,pcg_maxiter=8192,pcg_probe_count=32,pcg_probe_seed=731904,**kwargs):
        if (pcg_maxiter,pcg_probe_count,pcg_probe_seed)!=(8192,32,731904):
            raise ValueError('This isolated fallback pins8192 iterations/32 probes/seed731904')
        self.pcg_maxiter=int(pcg_maxiter);self.pcg_probe_count=int(pcg_probe_count)
        self.pcg_probe_seed=int(pcg_probe_seed)
        super().__init__(params,**kwargs)
        self.solve_linear=jax.jit(self._linear_system)

    def _linear_system(self,t,memory,data,damping,force,metric=None):
        roots=data['occupations'][data['root_indices']]
        if metric is None:metric=self._metric(t,roots,data['root_weights'])
        operator=lambda v:metric(v)+damping*v
        rhs=force+damping*self.mu*memory
        start=self.mu*memory
        tol=jnp.maximum(self.cg_tol*jnp.linalg.norm(rhs),1e-12)
        ordinary,it,ordinary_residual=ordinary_cg(operator,rhs,start,
            maxiter=self.cg_maxiter,tolerance=tol)
        use_pcg=(~jnp.isfinite(ordinary_residual))|(ordinary_residual>1.01*tol)
        def fallback(_):
            _,pull,_,q=projective_metric_factors(
                lambda u:self.sample_state(u,roots,data['root_weights']),t,
                root_rows=len(roots),head_count=self.S)
            diagonal=nonnegative_metric_diagonal(pull,t,output_dimension=q.shape[0],
                probe_count=self.pcg_probe_count,probe_seed=self.pcg_probe_seed)
            return solve_fixed_diagonal_pcg(metric,rhs,ordinary,damping=damping,
                metric_diagonal=diagonal,maxiter=self.pcg_maxiter,
                relative_tolerance=self.cg_tol,absolute_floor=1e-12)
        def unchanged(_):
            return ordinary,dict(iterations=jnp.int32(0),original_residual=ordinary_residual,
                original_tolerance=tol,recursive_residual=ordinary_residual,
                converged=ordinary_residual<=1.01*tol,healthy=jnp.isfinite(ordinary_residual),
                preconditioner_min=damping,preconditioner_max=damping)
        direction,pcg=jax.lax.cond(use_pcg,fallback,unchanged,operand=None)
        true_residual=jnp.linalg.norm(operator(direction)-rhs)
        stats=dict(cg_iterations=it+pcg['iterations'],ordinary_cg_iterations=it,
            ordinary_cg_absolute_residual=ordinary_residual,pcg_iterations=pcg['iterations'],
            pcg_fallback_used=use_pcg,pcg_preconditioner_min=pcg['preconditioner_min'],
            pcg_preconditioner_max=pcg['preconditioner_max'],pcg_healthy=pcg['healthy'],
            cg_absolute_residual=true_residual,cg_absolute_tolerance=tol,
            cg_converged=pcg['healthy']&jnp.isfinite(true_residual)&(true_residual<=1.01*tol))
        return direction,stats

    def _step(self, t, memory, data, damping, validation_data, check_validation):
        (objective, stats), gradient = jax.value_and_grad(self.loss, has_aux=True)(t, data)
        force = .5*gradient
        roots = data['occupations'][data['root_indices']]
        metric = self._metric(t, roots, data['root_weights'])
        direction, linear_stats = self._linear_system(t,memory,data,damping,force,metric=metric)
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
            preupdate_sampled_total_loss=objective,
            raw_direction_norm=norm, force_norm=jnp.linalg.norm(force),
            effective_eta=jnp.where(reject, 0., alpha), update_norm=jnp.linalg.norm(updated-t),
            backtracks=backs, rejected=reject, predicted_augmented_distance=alpha*metric_norm,
            heldout_checked=check_validation, damping=damping,
            private_hidden_gradient_norm=jnp.sqrt(sum(jnp.sum(named[k]**2) for k in
                ('sector_W1', 'sector_b1', 'sector_W2', 'sector_b2'))),
            private_output_gradient_norm=jnp.sqrt(sum(jnp.sum(named[k]**2) for k in
                ('sector_out', 'sector_bias'))))
        metrics = metrics|linear_stats
        return updated, memory_new, metrics
