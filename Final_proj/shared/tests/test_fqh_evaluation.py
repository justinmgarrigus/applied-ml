"""Small portable frozen-state evaluation, with no archived-data dependency."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from fqh_spring import model
from fqh_spring.references import build_reference
from fqh_spring.evaluate import evaluate

@pytest.mark.parametrize('readout',['final','legacy_additive'])
def test_ne4_frozen_evaluator_matches_direct_rayleigh_and_overlap(tmp_path,readout):
    reference=build_reference(4,tmp_path/'reference',eigenpairs=4)
    params=model.initialize(10,width=8,layers=2,seed=71,readout=readout)
    checkpoint=tmp_path/'parameters.npz';np.savez(checkpoint,**{k:np.asarray(v)for k,v in params.items()})
    result=evaluate(checkpoint,tmp_path/'reference',tmp_path/'evaluation.json',angular=True)
    logs=np.asarray(model.batch_log_psi(params,jnp.asarray(reference['occupations'])))
    psi=np.exp(logs-logs.real.max());psi/=np.linalg.norm(psi)
    energy=float(np.vdot(psi,reference['H']@psi).real)
    variance=float(np.linalg.norm(reference['H']@psi-energy*psi)**2)
    fidelity=float(abs(np.vdot(reference['eigenvectors'][:,0],psi))**2)
    assert abs(result['exact_variational_energy_V1']-energy)<1e-11
    assert abs(result['exact_residual_variance_V1_squared']-variance)<1e-11
    assert abs(result['ground_state_fidelity']-fidelity)<1e-11
    assert result['inputs_unchanged']and not result['ed_used_for_training']
