"""Final-head architecture, energy gradients and historical-checkpoint safety."""
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from fqh_spring import base_model, legacy_model, model
from fqh_spring.physics import SphereOperator
from fqh_spring.references import sparse_operators
from fqh_spring.train import load_config


@pytest.mark.parametrize('layers,expected_count', [(1,1602),(2,5762),(6,22402)])
def test_only_final_readout_is_stored_and_used(layers, expected_count):
    params = model.initialize(22, width=64, layers=layers, seed=71)
    assert 'final_out' in params
    assert not any(name.startswith('out') for name in params)
    assert model.architecture(params)['parameter_count'] == expected_count
    assert model.architecture(params)['readout'] == 'final'
    occupation = np.zeros(22)
    occupation[[0,3,6,9,12,15,18,21]] = 1
    hidden = 2 * occupation - 1
    for layer in range(1, layers+1):
        suffix = '' if layer == 1 else str(layer)
        hidden = np.tanh(hidden @ np.asarray(params['W'+suffix]) + np.asarray(params['b'+suffix]))
    expected = hidden @ np.asarray(params['final_out']) + np.asarray(params['bias'])
    np.testing.assert_allclose(model.log_psi(params,jnp.asarray(occupation)),
                               expected[0]+1j*expected[1],atol=1e-13)


@pytest.mark.parametrize('layers', [1,2,6])
def test_energy_gradient_reaches_every_hidden_layer(layers):
    operator = SphereOperator(4)
    bits,H,_,_ = sparse_operators(operator)
    occupations = jnp.asarray(((bits[:,None]>>np.arange(operator.norb))&1),dtype=jnp.float64)
    matrix = jnp.asarray(H.toarray())
    params = model.initialize(operator.norb,width=8,layers=layers,seed=71)

    def energy(p):
        logs = model.batch_log_psi(p,occupations)
        psi = jnp.exp(logs-jnp.max(logs.real))
        return (jnp.vdot(psi,matrix@psi)/jnp.vdot(psi,psi)).real

    gradient = jax.grad(energy)(params)
    logs = model.batch_log_psi(params,occupations)
    psi = jnp.exp(logs-jnp.max(logs.real))
    probability = abs(psi)**2/jnp.sum(abs(psi)**2)
    local_energy = (matrix@psi)/psi
    residual = local_energy-jnp.sum(probability*local_energy)
    scores = model.score_tree(params,occupations)
    covariance = jax.tree.map(lambda s:2*jnp.tensordot(probability*residual,s.conj(),axes=(0,0)).real,scores)
    for name in params:
        assert np.isfinite(np.asarray(gradient[name])).all()
        np.testing.assert_allclose(gradient[name],covariance[name],atol=1e-11,rtol=1e-9)
    for layer in range(1,layers+1):
        suffix = '' if layer == 1 else str(layer)
        assert float(jnp.linalg.norm(gradient['W'+suffix])) > 1e-12
        assert float(jnp.linalg.norm(gradient['b'+suffix])) > 1e-12
    # A nonconstant initialized phase avoids trapping a real positive state.
    assert float(jnp.std(logs.imag)) > 1e-8
    # Independently check an early hidden weight and a final phase weight.
    for name,index in (('W',(0,0)),('final_out',(0,1))):
        epsilon=1e-5
        plus=dict(params); minus=dict(params)
        plus[name]=params[name].at[index].add(epsilon)
        minus[name]=params[name].at[index].add(-epsilon)
        difference=(float(energy(plus))-float(energy(minus)))/(2*epsilon)
        assert abs(difference-float(gradient[name][index]))<2e-8


def test_legacy_states_are_not_silently_reinterpreted():
    params = model.initialize(10,width=8,layers=3,seed=71,readout='legacy_additive')
    params['out2'] = params['out2']+.07
    params['out3'] = params['out3']-.03
    occupation = jnp.asarray([1,0,0,1,0,0,1,0,0,1],dtype=jnp.float64)
    np.testing.assert_allclose(model.log_psi(params,occupation),legacy_model.log_psi(params,occupation),atol=1e-14)
    assert model.architecture(params)['readout']=='legacy_additive'
    with pytest.raises(ValueError,match='additive'):
        base_model.hidden_layers(params)
    with pytest.raises(ValueError,match='readout architecture differs'):
        model.prepare(params,norb=10,width=8,layers=3,kind='mlp')
    final = model.initialize(10,width=8,layers=3,seed=71)
    with pytest.raises(ValueError,match='Cannot change final-readout depth'):
        model.prepare(final,norb=10,width=8,layers=4,kind='mlp')
    with pytest.raises(ValueError,match='readout architecture differs'):
        model.prepare(final,norb=10,width=8,layers=3,kind='mlp',readout='legacy_additive')


def test_primary_configuration_is_fresh_six_layer_all_readout_spring():
    root=Path(__file__).resolve().parents[2]
    config=load_config(root/'SPRING/FQHE/training/configs/ne4.json')
    assert config['method']=='spring' and config['ne']==4
    assert config['variant']=='mlp' and config['readout']=='legacy_additive'
    assert config['layers']==6 and config['width']==64 and config['source'] is None
    assert Path(config['output']).parent==root/'SPRING/FQHE/results'
