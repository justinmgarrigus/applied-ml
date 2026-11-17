"""Isolated B256 pure-V1 Ne12 runtime with rematerialized NN components.

Chunking changes only evaluation storage. Unshifted log amplitudes and relative
heads are concatenated before the ONE global stop-gradient amplitude shift.
The objective, its derivatives, projective metric and SPRING step are inherited.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import learned_sector_nonlinear_model as network
from ne12_l2_spring_runtime import Ne12Problem as BaseNe12Problem
from ne8_spring_runtime import (FixedAnchorPolicy,angular_levels,DATA_KEYS,
    prepare_batch as _prepare_batch)

BATCH=256
POOL=1024
ANCHORS=32
NN_BUCKETS=(16384,24576,32768,43008)
CONSERVATIVE_UNION_BOUND=BATCH*(1+8+(1+12**2))+BATCH*12
assert CONSERVATIVE_UNION_BOUND==42496 and NN_BUCKETS[-1]>=CONSERVATIVE_UNION_BOUND


def prepare_batch(operator,roots,rng,cache,h_draws=8,root_weights=None,*,
                  output_operator,lplus_rng,output_count=BATCH,nn_buckets=NN_BUCKETS):
    if (operator.ne,operator.flux,operator.two_lz)!=(12,33,0):
        raise ValueError('This isolated runtime requires Ne12/F33/M0')
    if h_draws!=8 or len(roots)!=BATCH or output_count!=BATCH or tuple(nn_buckets)!=NN_BUCKETS:
        raise ValueError('This continuation requires B256/Lplus256 and its audited buckets')
    data,sample,coverage=_prepare_batch(operator,roots,rng,cache,h_draws,root_weights,
        output_operator=output_operator,lplus_rng=lplus_rng,output_count=output_count,nn_buckets=nn_buckets)
    if coverage['unique_NN_inputs']>CONSERVATIVE_UNION_BOUND:
        raise ValueError('Exact NN union exceeded conservative bound; no truncation')
    return data,sample,coverage


class Ne12Batch256Problem(BaseNe12Problem):
    """Same pure-V1 SPRING; rematerialized component chunks limit activations."""
    method_identifier='weighted_sampled_spring_positive_L2_no_GN_Ne12_B256_chunked_v1'

    def __init__(self,params,*,chunk_size=512,**kwargs):
        self.chunk_size=int(chunk_size)
        if self.chunk_size<1:raise ValueError('Positive NN chunk size required')
        kwargs.setdefault('v3',0.);kwargs.setdefault('v5',0.);kwargs.setdefault('kappa',.001)
        super().__init__(params,**kwargs)

    def fields(self,t,occupations):
        params=self.unravel(t)
        rows=occupations.shape[0]
        if rows<=self.chunk_size:
            return network.amplitudes(params,occupations)
        padding=(-rows)%self.chunk_size
        padded=jnp.concatenate((occupations,jnp.repeat(occupations[:1],padding,axis=0)),axis=0)
        chunks=padded.reshape((-1,self.chunk_size,occupations.shape[1]))

        @jax.checkpoint
        def components(chunk):
            logs,relative,_=network.components(params,chunk)
            return logs,relative

        logs,relative=jax.lax.map(components,chunks)
        logs=logs.reshape((-1,))[:rows]
        relative=relative.reshape((-1,self.S))[:rows]
        mixing=jax.nn.softmax(params['mix_logits'])
        shift=jax.lax.stop_gradient(jnp.max(logs.real))
        heads=jnp.exp(logs-shift)[:,None]*relative
        return heads,heads@mixing,mixing,shift
