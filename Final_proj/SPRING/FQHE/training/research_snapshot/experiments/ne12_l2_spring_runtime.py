"""Isolated Ne12/F33 no-wall port of the audited size-aware SPRING runtime.

The raw coherent L2 penalty is kappa*||Lplus psi||^2/Z. The angular-head
auxiliary uses Ne*Q*(Q+1)=3465; the coherent penalty is not divided by it.
No full Ne12 many-body basis or ED target is constructed.
"""
from __future__ import annotations

from ne8_spring_runtime import (
    Ne8Problem,FixedAnchorPolicy,angular_levels,sector_dimension,DATA_KEYS,
    prepare_batch as _prepare_batch,
)

BATCH=32
POOL=128
ANCHORS=32
NN_BUCKET=5376
# K is padded to 1+Ne^2=145. Counting every padded path independently and
# retaining an additional root gives a conservative (deliberately loose) bound.
CONSERVATIVE_UNION_BOUND=BATCH*(1+8+(1+12**2))+BATCH*12
assert CONSERVATIVE_UNION_BOUND==5312 and NN_BUCKET>=CONSERVATIVE_UNION_BOUND


def prepare_batch(operator,roots,rng,cache,h_draws=8,root_weights=None,*,
                  output_operator,lplus_rng,output_count=BATCH,nn_buckets=(NN_BUCKET,)):
    if operator.ne!=12 or operator.flux!=33 or operator.two_lz!=0:
        raise ValueError('This isolated runtime requires Ne12/F33/M0')
    if h_draws!=8 or len(roots)!=BATCH or output_count!=BATCH or tuple(nn_buckets)!=(NN_BUCKET,):
        raise ValueError('This pilot requires B32/Lplus32 and one static5376 bucket')
    data,sample,coverage=_prepare_batch(operator,roots,rng,cache,h_draws,root_weights,
        output_operator=output_operator,lplus_rng=lplus_rng,output_count=output_count,nn_buckets=nn_buckets)
    if coverage['unique_NN_inputs']>CONSERVATIVE_UNION_BOUND:
        raise ValueError('Exact NN union exceeded its conservative path bound; no truncation')
    return data,sample,coverage


class Ne12Problem(Ne8Problem):
    """Fresh Ne12 angular-head model; same objective/metric, CG cap1024."""
    method_identifier='weighted_sampled_spring_positive_L2_no_GN_Ne12_F33_v1'

    def __init__(self,params,**kwargs):
        if 'ne' in kwargs or 'flux' in kwargs:
            raise ValueError('Ne12/F33 geometry is fixed in this isolated pilot')
        kwargs.setdefault('cg_maxiter',1024)
        super().__init__(params,ne=12,flux=33,**kwargs)
