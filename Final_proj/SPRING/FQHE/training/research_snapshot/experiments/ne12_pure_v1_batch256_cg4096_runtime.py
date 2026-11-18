"""Numerical-only recovery: increase the unchanged CG iteration ceiling.

The equation, stopping tolerance, objective, metric, caps and NN chunking are
inherited verbatim. No optimizer memory or sampler state is reset here.
"""
from ne12_pure_v1_batch256_runtime import (
    Ne12Batch256Problem,FixedAnchorPolicy,prepare_batch,angular_levels,DATA_KEYS,
    CONSERVATIVE_UNION_BOUND,NN_BUCKETS,
)


class Ne12CG4096Problem(Ne12Batch256Problem):
    def __init__(self,params,**kwargs):
        if kwargs.get('cg_maxiter',4096)!=4096:
            raise ValueError('This isolated numerical recovery pins CG maxiter4096')
        kwargs['cg_maxiter']=4096
        super().__init__(params,**kwargs)
