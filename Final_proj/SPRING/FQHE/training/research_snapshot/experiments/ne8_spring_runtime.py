"""Size-aware sampled angular-head SPRING for the Ne8 LS6 experiment.

No ED state, full many-body basis, dense projector, or dense parameter metric
is constructed. The optimizer/metric/guards are inherited unchanged from the
audited sampled runtime. Only geometry-dependent dimensions and data shapes
are generalized here.
"""
from __future__ import annotations

from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from pseudopotential_v5_operator import SphereOperatorV5
from learned_sector_operators import occupation_bits
from residual_geometry_operators import operator_batch_from_roots
from residual_geometry_runtime import SampledProblem as BaseProblem
from residual_geometry_policy import ResidualGeometryPolicy
from l0_positive_penalty_runtime import incoming_lplus_parents,positive_l2_from_union,DATA_KEYS


@lru_cache(maxsize=None)
def _completions(flux,start,remaining,target):
    if remaining==0: return int(target==0)
    if remaining<0 or flux+1-start<remaining: return 0
    smallest=remaining*(2*start+remaining-1)//2
    largest=remaining*(2*flux-remaining+1)//2
    if target<smallest or target>largest: return 0
    return (_completions(flux,start+1,remaining,target)+
        _completions(flux,start+1,remaining-1,target-start))


def sector_dimension(ne,flux,two_lz=0):
    value=ne*flux+two_lz
    return 0 if value%2 else _completions(int(flux),0,int(ne),value//2)


def angular_levels(ne,flux):
    maximum=ne*(flux-ne+1)//2
    counts=[sector_dimension(ne,flux,2*L) for L in range(maximum+2)]
    return np.asarray([L for L in range(maximum+1) if counts[L]>counts[L+1]],dtype=np.int32)


class FixedAnchorPolicy(ResidualGeometryPolicy):
    def __init__(self,*args,anchor_rows=32,**kwargs):
        self.anchor_rows=int(anchor_rows)
        super().__init__(*args,**kwargs)
        self._check_anchor_shape()

    def _check_anchor_shape(self):
        if self.anchors.shape!=(self.anchor_rows,self.problem.op.norb):
            raise ValueError('Fixed geometry anchor shape changed')

    def observe(self,roots,residuals,update):
        super().observe(roots,residuals,update)
        self.anchors=np.asarray(roots[:self.anchor_rows],dtype=np.int8).copy()
        self._check_anchor_shape()


def prepare_batch(operator,roots,rng,cache,h_draws=8,root_weights=None,*,
                  output_operator,lplus_rng,output_count=128,
                  nn_buckets=(4096,6144,8704)):
    # Conservative path bound: <=Ne^2 ladder paths plus the diagonal.
    k_width=1+operator.ne**2
    sample=operator_batch_from_roots(operator,roots,rng,cache=cache,
        h_samples_per_root=h_draws,h_pad_to=h_draws,k_pad_to=k_width)
    outputs=output_operator.sample_uniform(lplus_rng,output_count)
    union=[n.copy() for n in sample['occupations']]
    lookup={occupation_bits(n):i for i,n in enumerate(union)}
    indices=np.zeros((output_count,operator.ne),dtype=np.int32)
    coefficients=np.zeros((output_count,operator.ne),dtype=np.float64)
    for row,output in enumerate(outputs):
        parents,values=incoming_lplus_parents(output,ne=operator.ne,flux=operator.flux)
        for column,(parent,value) in enumerate(zip(parents,values)):
            bits=occupation_bits(parent)
            if bits not in lookup:
                lookup[bits]=len(union);union.append(parent)
            indices[row,column]=lookup[bits];coefficients[row,column]=value
    count=len(union)
    capacity=next((n for n in nn_buckets if n>=count),None)
    if capacity is None: raise ValueError(f'NN union {count} exceeds configured buckets {nn_buckets}; no truncation')
    sample['occupations']=np.asarray(union+[union[0]]*(capacity-count),dtype=np.int8)
    sample['occupation_mask']=np.arange(capacity)<count
    weights=np.ones(len(roots)) if root_weights is None else np.asarray(root_weights,dtype=np.float64)
    if weights.shape!=(len(roots),) or not np.isfinite(weights).all() or not np.all(weights>0):
        raise ValueError('Positive finite root weights required')
    sample.update(root_weights=weights,Lplus_indices=indices,Lplus_coefficients=coefficients,
        Lplus_outputs=outputs,Lplus_scale=np.asarray((output_operator.dimension()/operator.dimension())*
            (len(roots)/output_count),dtype=np.float64))
    cache.visited_amplitude_configurations.update(lookup)
    coverage=dict(sampled_roots=len(roots),unique_root_configurations=len({occupation_bits(n) for n in roots}),
        unique_NN_inputs=count,padded_NN_rows=capacity,K_row_width=k_width,Lplus_output_rows=output_count,
        cached_root_rows=len(cache._rows),cumulative_unique_sampled_roots=len(cache.visited_roots),
        cumulative_unique_amplitude_configurations=len(cache.visited_amplitude_configurations))
    return {key:jnp.asarray(sample[key]) for key in DATA_KEYS},sample,coverage


class Ne8Problem(BaseProblem):
    """Original sampled SPRING, with a positive raw coherent L2 penalty."""
    method_identifier='weighted_sampled_spring_positive_L2_no_GN_size_aware_v1'

    def __init__(self,params,*,ne=8,flux=21,v3=.8,v5=.6,kappa=.1,gamma=1.,eta=.001,
                 mu=.9,parameter_radius=.015,state_radius=.01,cg_maxiter=384,cg_tol=1e-6):
        self.flat,self.unravel=ravel_pytree(params)
        self.op=SphereOperatorV5(ne,flux=flux,v3=v3,v5=v5,two_lz=0)
        self.levels=angular_levels(ne,flux);self.S=len(self.levels);self.P=int(self.flat.size)
        self.targets=jnp.asarray(self.levels*(self.levels+1),dtype=jnp.float64)
        self.head_angular_scale=float(ne*(flux/2)*(flux/2+1))
        self.kappa=float(kappa);self.gamma=float(gamma);self.eta=float(eta);self.mu=float(mu)
        self.parameter_radius=float(parameter_radius);self.state_radius=float(state_radius)
        self.cg_maxiter=int(cg_maxiter);self.cg_tol=float(cg_tol)
        if params['base__W'].shape[0]!=self.op.norb or params['sector_W1'].shape[0]!=self.S:
            raise ValueError('NN input/head shapes do not match this physical system')
        if self.op.dimension()!=sector_dimension(ne,flux,0): raise ValueError('Independent DP dimensions disagree')
        self.force_stats=jax.jit(self._force_stats);self.force_parts=jax.jit(self._force_parts)
        self.predict=jax.jit(self._predict);self.step=jax.jit(self._step)
        self.validation_change=jax.jit(self._validation_change)
        self.metric_action=jax.jit(self._metric_action);self.sketch=jax.jit(self._sketch)

    def _parts(self,t,data):
        heads,_,mixing,_=self.fields(t,data['occupations'])
        root=heads[data['root_indices']]
        hhead=jnp.sum(data['H_coefficients'][:,:,None]*heads[data['H_indices']],axis=1)
        khead=jnp.sum(data['K_coefficients'][:,:,None]*heads[data['K_indices']],axis=1)
        coherent=root@mixing;hcoherent=hhead@mixing
        w=jax.lax.stop_gradient(data['root_weights'])
        denominator=jnp.sum(w*abs(coherent)**2)
        head_denominator=jnp.sum(w[:,None]*abs(root)**2,axis=0)
        energy=jnp.sum(w*jnp.conj(coherent)*hcoherent).real/jnp.maximum(denominator,1e-280)
        residual=(khead-root*self.targets[None,:])/self.head_angular_scale
        angular=jnp.mean(jnp.sum(w[:,None]*abs(residual)**2,axis=0)/jnp.maximum(head_denominator,1e-280))
        l2,lplus_norm=positive_l2_from_union(heads@mixing,data)
        stats=dict(sampled_energy=energy,sampled_angular_loss=angular,sampled_L2=l2,
            sampled_L2_penalty=self.kappa*l2,sampled_Lplus_norm_squared=lplus_norm,
            sampled_total_loss=energy+self.gamma*angular+self.kappa*l2,
            root_importance_ess=denominator**2/jnp.maximum(jnp.sum((w*abs(coherent)**2)**2),1e-280),
            selection_weight_ess=jnp.sum(w)**2/jnp.sum(w**2),coherent_root_norm_squared=denominator,
            minimum_head_root_norm_squared=jnp.min(head_denominator),maximum_head_root_norm_squared=jnp.max(head_denominator))
        return energy,angular,stats,root,coherent,hcoherent,residual

    def loss(self,t,data):
        energy,angular,stats,*_=self._parts(t,data)
        return energy+self.gamma*angular+stats['sampled_L2_penalty'],stats

    def _force_parts(self,t,data):
        energy=.5*jax.grad(lambda p:self._parts(p,data)[0])(t)
        angular=.5*self.gamma*jax.grad(lambda p:self._parts(p,data)[1])(t)
        penalty=.5*jax.grad(lambda p:self._parts(p,data)[2]['sampled_L2_penalty'])(t)
        return energy,angular,penalty,self.loss(t,data)[1]
