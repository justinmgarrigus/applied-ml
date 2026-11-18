"""Optimizer conventions and small exact-gradient/dual-solve parity checks."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest
import optax
from jaqmc.optimizer.sr.jasr import robust_sr
from fqh_spring.optimizer import make_spring,make_minsr,make_optimizer,score_data
from fqh_spring import model
from fqh_spring.physics import SphereOperator
from fqh_spring.references import sparse_operators


def fixture():
    rng=np.random.default_rng(417)
    raw=rng.normal(size=(7,5))+1j*rng.normal(size=(7,5))
    rows=raw-raw.mean(axis=0)
    residual=rng.normal(size=7)+1j*rng.normal(size=7);residual-=residual.mean()
    A=np.concatenate((rows.real,rows.imag),axis=0)/np.sqrt(7)
    eps=np.concatenate((residual.real,residual.imag))/np.sqrt(7)
    force=A.T@eps
    params={'w':jnp.asarray(rng.normal(size=5))}
    return params,jnp.asarray(rows),jnp.asarray(force),A,eps


def test_spring_adapter_matches_direct_pinned_upstream_with_memory():
    params,rows,force,_,_=fixture()
    log_fn=lambda p,row:jnp.dot(jax.lax.stop_gradient(row),p['w'])
    direct=robust_sr(log_fn,learning_rate=.02,damping=.1,spring_mu=.9,max_norm=.3,
        max_cond_num=None,robust_gamma=None,march_beta=None,mixed_precision=True,
        score_in_axes=0,score_chunk_size=None,score_norm_clip=None,gram_num_chunks=None,
        gram_dot_prec='F64',axis_name=None,prune_inactive=False)
    adapter=make_spring(learning_rate=.02,damping=.1,mu=.9,max_norm=.3)
    a,b=adapter.init(params),direct.init(params)
    for scale in (1.,.8):
        ua,a=adapter.update({'w':scale*force},a,params,data=rows)
        ub,b=direct.update({'w':scale*force},b,params,data=rows)
        for x,y in zip(jax.tree.leaves((ua,a)),jax.tree.leaves((ub,b))):
            np.testing.assert_allclose(x,y,atol=1e-13,rtol=1e-13)


def test_minsr_is_memory_free_damped_dual_and_primal_solve():
    params,rows,force,A,eps=fixture();lam=.1
    optimizer=make_minsr(learning_rate=.02,damping=lam,max_norm=1.)
    state=optimizer.init(params)
    updates,state=optimizer.update({'w':force},state,params,data=rows)
    primal=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),np.asarray(force))
    dual=A.T@np.linalg.solve(A@A.T+lam*np.eye(A.shape[0]),eps)
    np.testing.assert_allclose(primal,dual,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(state[0].prev_delta,dual,atol=2e-11,rtol=2e-11)
    np.testing.assert_allclose(updates['w'],-.02*dual,atol=2e-12,rtol=2e-12)
    repeated,_=optimizer.update({'w':force},state,params,data=rows)
    fresh,_=optimizer.update({'w':force},optimizer.init(params),params,data=rows)
    np.testing.assert_allclose(repeated['w'],fresh['w'],atol=2e-12,rtol=2e-12)


def test_vmc_uses_full_physical_gradient_and_common_applied_radius():
    operator=SphereOperator(4);_,H,_,_=sparse_operators(operator)
    # This exact sector is a test oracle, never training data.
    bits=np.asarray(sorted(int(v)for v in __import__('fqh_spring.references',fromlist=['enumerate_basis']).enumerate_basis(operator)))
    occupations=jnp.asarray(((bits[:,None]>>np.arange(10))&1),dtype=jnp.float64)
    params=model.initialize(10,width=4,layers=2,seed=71,readout='legacy_additive')
    matrix=jnp.asarray(H.toarray())
    def energy(p):
        logs=model.batch_log_psi(p,occupations);psi=jnp.exp(logs-jnp.max(logs.real))
        return (jnp.vdot(psi,matrix@psi)/jnp.vdot(psi,psi)).real
    gradient=jax.grad(energy)(params)
    logs=model.batch_log_psi(params,occupations);psi=jnp.exp(logs-logs.real.max())
    probability=abs(psi)**2/jnp.sum(abs(psi)**2)
    local=(matrix@psi)/psi;residual=local-jnp.sum(probability*local)
    scores=model.score_tree(params,occupations)
    force=jax.tree.map(lambda o:jnp.tensordot(probability*residual,o.conj(),axes=(0,0)).real,scores)
    for name in params:np.testing.assert_allclose(gradient[name],2*force[name],atol=1e-11,rtol=1e-10)
    optimizer=make_optimizer('vmc_sgd',learning_rate=.02,max_norm=.001)
    updates,_=optimizer.update(jax.tree.map(lambda value:2*value,force),optimizer.init(params),params)
    g=ravel_pytree(gradient)[0];u=ravel_pytree(updates)[0]
    expected=-g*jnp.minimum(.02,.001/jnp.linalg.norm(g))
    np.testing.assert_allclose(u,expected,atol=2e-12,rtol=2e-12)
    assert float(jnp.linalg.norm(u))<=.001+1e-12


def test_unimplemented_optimizers_cannot_be_selected():
    for method in ('adam','kfac','unknown'):
        with pytest.raises(ValueError,match='Unknown method'):make_optimizer(method)

@pytest.mark.parametrize('algorithm,method',[('SPRING','spring'),('VMC','vmc_sgd'),('MinSR','minsr')])
def test_method_wrapper_rejects_wrong_config_and_ambiguous_resume(tmp_path,algorithm,method):
    from pathlib import Path
    import json,os,subprocess,sys
    root=Path(__file__).resolve().parents[2]
    script=root/algorithm/'FQHE/training/run.py'
    env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    missing=subprocess.run([sys.executable,str(script),'--resume'],env=env,capture_output=True,text=True)
    assert missing.returncode!=0 and 'explicit --output' in missing.stderr
    config=json.loads((script.parent/'configs/ne4.json').read_text())
    config['method']='vmc_sgd' if method=='spring' else 'spring'
    path=tmp_path/'wrong.json';path.write_text(json.dumps(config))
    wrong=subprocess.run([sys.executable,str(script),'--config',str(path)],env=env,capture_output=True,text=True)
    assert wrong.returncode!=0 and 'requires method=' in wrong.stderr
