"""Plain original-CG numerical recovery, with only its iteration ceiling raised.

The frozen Ne12Batch256Problem supplies the entire step/metric/CG recurrence.
No copied step, preconditioner, PCG or GMRES is used by this runtime.
"""
from ne12_pure_v1_batch256_runtime import (
    Ne12Batch256Problem,FixedAnchorPolicy,prepare_batch,angular_levels,DATA_KEYS,
    CONSERVATIVE_UNION_BOUND,NN_BUCKETS,
)


class Ne12CG65536Problem(Ne12Batch256Problem):
    def __init__(self,params,**kwargs):
        if kwargs.get('cg_maxiter',65536)!=65536:
            raise ValueError('This isolated plain-CG recovery pins maxiter65536')
        kwargs['cg_maxiter']=65536
        super().__init__(params,**kwargs)
