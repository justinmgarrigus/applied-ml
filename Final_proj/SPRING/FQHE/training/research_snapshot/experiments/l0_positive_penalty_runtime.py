"""Positive sampled <L^2> penalty at M=0 from exact L_+ amplitudes.

The numerator samples output configurations in M=1 uniformly and evaluates
every incoming M=0 parent before squaring the coherent amplitude. This is
||L_+ psi||^2 / ||psi||^2 = <L^2> at M=0, not <(L^2)^2>.
The finite sampled numerator/denominator ratio remains statistically biased.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from learned_sector_operators import occupation_bits
from residual_geometry_operators import operator_batch_from_roots
from residual_geometry_runtime import SampledProblem as BaseProblem, DATA_KEYS as BASE_DATA_KEYS

DATA_KEYS = BASE_DATA_KEYS+('Lplus_indices','Lplus_coefficients','Lplus_scale','Lplus_outputs')


def incoming_lplus_parents(output, *, ne=7, flux=18):
    """Return M0 parents n and exact <r|L_+|n> for an M1 output r.

    For adjacent ladder hops the canonical Fock sign is positive. A parent
    lowers an occupied b in r to vacant b-1; its raising coefficient is
    sqrt(b*(flux-b+1)).
    """
    output = np.asarray(output,dtype=np.int8)
    if output.shape!=(flux+1,) or not np.all((output==0)|(output==1)):
        raise ValueError('Invalid Lplus output occupation')
    if output.sum()!=ne or 2*(output@np.arange(flux+1))-ne*flux!=2:
        raise ValueError('Lplus outputs must have Ne fixed and M=1')
    parents=[]; coefficients=[]
    for b in np.flatnonzero(output):
        b=int(b)
        if b==0 or output[b-1]: continue
        parent=output.copy();parent[b]=0;parent[b-1]=1
        parents.append(parent);coefficients.append(np.sqrt(float(b*(flux-b+1))))
    return np.asarray(parents,dtype=np.int8).reshape(-1,flux+1),np.asarray(coefficients,dtype=np.float64)


def prepare_batch(operator, roots, rng, cache, h_draws=8, root_weights=None,
                  *, output_operator, lplus_rng, output_count=32):
    sample=operator_batch_from_roots(operator,roots,rng,cache=cache,
        h_samples_per_root=h_draws,h_pad_to=h_draws,k_pad_to=43)
    outputs=output_operator.sample_uniform(lplus_rng,output_count)
    union=[row.copy() for row in sample['occupations']]
    lookup={occupation_bits(row):i for i,row in enumerate(union)}
    indices=np.zeros((output_count,operator.ne),dtype=np.int32)
    coefficients=np.zeros((output_count,operator.ne),dtype=np.float64)
    for row,output in enumerate(outputs):
        parents,values=incoming_lplus_parents(output,ne=operator.ne,flux=operator.flux)
        if len(parents)>operator.ne: raise ValueError('Lplus parent width exceeded')
        for column,(parent,value) in enumerate(zip(parents,values)):
            bits=occupation_bits(parent)
            if bits not in lookup:
                lookup[bits]=len(union);union.append(parent)
            indices[row,column]=lookup[bits];coefficients[row,column]=value
    count=len(union)
    capacity=next((n for n in (768,1024,1504,1600,1792) if n>=count),None)
    if capacity is None: raise ValueError('Lplus-augmented NN union capacity exceeded')
    sample['occupations']=np.asarray(union+[union[0]]*(capacity-count),dtype=np.int8)
    sample['occupation_mask']=np.arange(capacity)<count
    sample['root_weights']=np.ones(len(roots)) if root_weights is None else np.asarray(root_weights,dtype=np.float64)
    if sample['root_weights'].shape!=(len(roots),) or not np.all(sample['root_weights']>0):
        raise ValueError('Positive root importance weights required')
    sample.update(Lplus_indices=indices,Lplus_coefficients=coefficients,Lplus_outputs=outputs,
        Lplus_scale=np.asarray((output_operator.dimension()/operator.dimension())*(len(roots)/output_count),dtype=np.float64))
    cache.visited_amplitude_configurations.update(lookup)
    coverage=dict(sampled_roots=len(roots),unique_NN_inputs=count,padded_NN_rows=capacity,
        Lplus_output_rows=output_count,Lplus_unique_outputs=len({occupation_bits(n) for n in outputs}),
        Lplus_unique_parent_inputs=len({int(i) for row,vals in zip(indices,coefficients) for i,v in zip(row,vals) if v!=0}),
        Lplus_output_dimension=output_operator.dimension())
    for key in ('unique_root_configurations','cached_root_rows','cumulative_unique_sampled_roots'):
        coverage[key]=int(sample['coverage'][key])
    coverage['cumulative_unique_amplitude_configurations']=len(cache.visited_amplitude_configurations)
    return {key:jnp.asarray(sample[key]) for key in DATA_KEYS},sample,coverage


def positive_l2_from_union(coherent,data):
    """Pure differentiable positive estimator from a coherent NN amplitude union."""
    root=coherent[data['root_indices']]
    denominator=jnp.sum(jax.lax.stop_gradient(data['root_weights'])*abs(root)**2)
    action=jnp.sum(data['Lplus_coefficients']*coherent[data['Lplus_indices']],axis=1)
    numerator=jnp.sum(abs(action)**2)
    return data['Lplus_scale']*numerator/jnp.maximum(denominator,1e-280),numerator


class L0PenaltyProblem(BaseProblem):
    """Original weighted SPRING metric with physical coherent-state L2 loss."""

    def __init__(self,params,*,kappa=0.,**kwargs):
        self.kappa=float(kappa)
        if self.kappa<0 or not np.isfinite(self.kappa): raise ValueError('Nonnegative finite kappa required')
        super().__init__(params,**kwargs)

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
        residual=(khead-root*self.targets[None,:])/630.
        angular=jnp.mean(jnp.sum(w[:,None]*abs(residual)**2,axis=0)/jnp.maximum(head_denominator,1e-280))
        union_coherent=heads@mixing
        l2,lplus_norm=positive_l2_from_union(union_coherent,data)
        penalty=self.kappa*l2
        stats=dict(sampled_energy=energy,sampled_angular_loss=angular,
            sampled_L2=l2,sampled_L2_penalty=penalty,sampled_Lplus_norm_squared=lplus_norm,
            sampled_total_loss=energy+self.gamma*angular+penalty,
            root_importance_ess=denominator**2/jnp.maximum(jnp.sum((w*abs(coherent)**2)**2),1e-280),
            selection_weight_ess=jnp.sum(w)**2/jnp.sum(w**2),
            coherent_root_norm_squared=denominator,minimum_head_root_norm_squared=jnp.min(head_denominator),
            maximum_head_root_norm_squared=jnp.max(head_denominator))
        return energy,angular,stats,root,coherent,hcoherent,residual

    def loss(self,t,data):
        energy,angular,stats,*_=self._parts(t,data)
        return energy+self.gamma*angular+stats['sampled_L2_penalty'],stats

    def _force_parts(self,t,data):
        # This isolated class exposes all three actual loss derivatives.
        energy=.5*jax.grad(lambda p:self._parts(p,data)[0])(t)
        angular=.5*self.gamma*jax.grad(lambda p:self._parts(p,data)[1])(t)
        penalty=.5*jax.grad(lambda p:self._parts(p,data)[2]['sampled_L2_penalty'])(t)
        return energy,angular,penalty,self.loss(t,data)[1]

    # Inherited _predict uses physical energy and original head residuals.
    # Lplus is deliberately excluded from sampling priorities in both arms.
