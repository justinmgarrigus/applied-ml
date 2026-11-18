"""Fresh/resumable Ne12 pure-V1 no-wall SPRING benchmark with coherent L2.

Defaults: F33/34LLL orbitals, pure normalized P1 (V3=V5=0), kappa=.1, eta=.001, six
shared64 all-layer readouts plus private2x16 angular heads; B32/pool128/Lplus32.
Fresh seed7 parameters only; no Laughlin factor, coefficients, ED labels, or
fixed target state are supplied to optimization.
SIGUSR1/SIGTERM request a safe checkpoint at the next complete update.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

PROCESS_STARTED=time.monotonic()
STOP_REASON=''


def request_stop(signum,_frame):
    global STOP_REASON
    STOP_REASON='signal_'+signal.Signals(signum).name


for _name in ('SIGUSR1','SIGTERM','SIGINT'):
    signal.signal(getattr(signal,_name),request_stop)
os.environ.setdefault('JAX_PLATFORMS','cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'4')

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'experiments')]
import learned_sector_nonlinear_model as network
from pseudopotential_v5_operator import SphereOperatorV5
from learned_sector_operators import RootConnectionCache,occupations_from_bits
from ne12_l2_spring_runtime import Ne12Problem,FixedAnchorPolicy,prepare_batch,angular_levels,DATA_KEYS,CONSERVATIVE_UNION_BOUND

BATCH=32;POOL=128;ANCHORS=32;H_DRAWS=8
BUCKETS=(5376,)
STUDY=PROJECT/'results/ne12_pure_v1_l2_20261116'
STREAM_NAMES=('pool','selection','H','validation','Lplus','validation_Lplus')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_npz(path,**arrays):
    path=Path(path);temporary=path.with_name('.'+path.name+'.tmp.npz')
    np.savez(temporary,**arrays);temporary.replace(path)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def source_inventory():
    paths=['experiments/train_ne12_v1_l2_spring.py','experiments/ne12_l2_spring_runtime.py','experiments/ne8_spring_runtime.py',
        'experiments/learned_sector_nonlinear_model.py','experiments/residual_geometry_runtime.py',
        'experiments/residual_geometry_policy.py','experiments/residual_geometry_operators.py',
        'experiments/learned_sector_operators.py','experiments/l0_positive_penalty_runtime.py',
        'experiments/pseudopotential_v5_operator.py','src/fqh_spring/physics.py',
        'src/fqh_spring/model.py','src/fqh_spring/legacy_model.py','src/fqh_spring/geometry_sampling.py']
    return {p:sha(PROJECT/p) for p in paths}


def cache_state(cache,prefix):
    return {prefix+'_cache_LRU_bits':np.asarray(list(cache._rows),dtype=np.int64),
        prefix+'_visited_roots':np.asarray(sorted(cache.visited_roots),dtype=np.int64),
        prefix+'_visited_amplitudes':np.asarray(sorted(cache.visited_amplitude_configurations),dtype=np.int64),
        prefix+'_cache_hits':np.asarray(cache.hits),prefix+'_cache_misses':np.asarray(cache.misses)}


def restore_cache(cache,saved,prefix):
    for bits in saved[prefix+'_cache_LRU_bits']:
        cache.connections(occupations_from_bits([int(bits)],cache.operator.norb)[0])
    cache.visited_roots={int(v) for v in saved[prefix+'_visited_roots']}
    cache.visited_amplitude_configurations={int(v) for v in saved[prefix+'_visited_amplitudes']}
    cache.hits=int(saved[prefix+'_cache_hits']);cache.misses=int(saved[prefix+'_cache_misses'])


def params_from_checkpoint(saved,metadata):
    result={};offset=0
    for name in metadata['parameter_names']:
        shape=metadata['parameter_shapes'][name];size=int(np.prod(shape))
        result[name]=jnp.asarray(saved['parameters'][offset:offset+size].reshape(shape));offset+=size
    if offset!=saved['parameters'].size: raise ValueError('Checkpoint parameter layout mismatch')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=None,help='Fresh directory; default study/train_seed<seed>; required for resume')
    parser.add_argument('--kappa',type=float,default=.1,help='Nonnegative coherent raw-L2 coefficient; same value required on resume')
    parser.add_argument('--active-seconds',type=float,default=1800.)
    parser.add_argument('--max-wall-seconds',type=float,default=2100.)
    parser.add_argument('--deadline-unix',type=float,default=None,help='Optional hard soft-stop deadline in Unix seconds')
    parser.add_argument('--steps',type=int,default=1000,help='Maximum additional complete updates')
    parser.add_argument('--save-every',type=int,default=1000)
    parser.add_argument('--checkpoint-seconds',type=float,default=60.)
    parser.add_argument('--log-every',type=int,default=25)
    parser.add_argument('--seed',type=int,default=71)
    parser.add_argument('--init-seed',type=int,default=7)
    parser.add_argument('--resume',type=Path,default=None,help='Compatible Ne12 checkpoint; output remains a new directory')
    parser.add_argument('--require-gpu',action='store_true')
    args=parser.parse_args()
    if min(args.steps,args.save_every,args.log_every)<1 or min(args.active_seconds,args.max_wall_seconds,args.checkpoint_seconds)<=0:
        raise ValueError('Positive budgets and intervals required')
    if not np.isfinite(args.kappa) or args.kappa<0:raise ValueError('Nonnegative finite kappa required')
    if args.resume is not None and args.output is None:raise ValueError('Resume requires a new explicit --output directory')
    if args.require_gpu and not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('GPU required but JAX sees '+str(jax.devices()))
    output=(args.output if args.output is not None else STUDY/f'train_seed{args.seed}').expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'checkpoints').mkdir()
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(output.parent/'jax_cache'))
    hashes=source_inventory();saved=None;resume_metadata=None
    if args.resume:
        args.resume=args.resume.expanduser().resolve()
        with np.load(args.resume,allow_pickle=False) as z:
            saved={k:np.array(z[k],copy=True) for k in z.files}
        resume_metadata=json.loads(str(saved['metadata_json']))
        for key,value in dict(ne=12,flux=33,two_lz=0,v3=0.,v5=0.,kappa=args.kappa,batch=BATCH,pool_size=POOL,Lplus_output_rows=BATCH).items():
            if resume_metadata[key]!=value: raise ValueError('Incompatible resume system/settings: '+key)
        if resume_metadata['seed']!=args.seed or resume_metadata['init_seed']!=args.init_seed:
            raise ValueError('Resume seed labels must match the checkpoint')
        if resume_metadata['input_sha256']!=hashes: raise ValueError('Resume source code hashes differ')
        params=params_from_checkpoint(saved,resume_metadata)
    else:
        levels=angular_levels(12,33)
        params=network.initialize(34,len(levels),seed=args.init_seed,width=64,layers=6,head_width=16)
    problem=Ne12Problem(params,v3=0.,v5=0.,kappa=args.kappa)
    output_operator=SphereOperatorV5(12,flux=33,v3=0.,v5=0.,two_lz=2)
    x=problem.flat if saved is None else jnp.asarray(saved['parameters'])
    previous=jnp.zeros_like(x) if saved is None else jnp.asarray(saved['previous_direction'])
    if not np.array_equal(np.asarray(problem.flat),np.asarray(x)): raise ValueError('NN reconstruction changed the state')
    rngs={name:np.random.default_rng(stream) for name,stream in
        zip(STREAM_NAMES,np.random.SeedSequence(args.seed).spawn(len(STREAM_NAMES)))}
    if saved is not None:
        states=json.loads(str(saved['rng_json']))
        if set(states)!=set(STREAM_NAMES): raise ValueError('All six sampler streams are required')
        for name in STREAM_NAMES:rngs[name].bit_generator.state=states[name]
    def rng_state(): return {name:rng.bit_generator.state for name,rng in rngs.items()}
    initial_rng_before_guard=json.loads(json.dumps(rng_state()))
    if saved is None:
        geometry=np.random.SeedSequence([args.seed,8371]).spawn(2)
        probes=np.random.default_rng(geometry[0]).normal(size=(8,problem.P))/np.sqrt(problem.P)
        anchors=problem.op.sample_uniform(np.random.default_rng(geometry[1]),ANCHORS)
    else:
        probes=saved['probes'];anchors=saved['policy_anchors']
        # Preserve exactly the compact floating Hamiltonian used before resume.
        if saved['operator_pair_coefficients'].shape!=problem.op.pair_coefficients.shape:
            raise ValueError('Compact operator shape changed')
        if not np.allclose(saved['operator_pair_coefficients'],problem.op.pair_coefficients,rtol=1e-12,atol=1e-13):
            raise ValueError('Compact Hamiltonian does not match this model')
        problem.op.pair_coefficients[:]=saved['operator_pair_coefficients']
    policy=FixedAnchorPolicy(problem,probes,anchors,anchor_rows=ANCHORS,exploration=.2,max_history=512)
    cache=RootConnectionCache(problem.op,max_rows=128)
    validation_cache=RootConnectionCache(problem.op,max_rows=128)
    if saved is None:
        vdata,_,_=prepare_batch(problem.op,problem.op.sample_uniform(rngs['validation'],BATCH),
            rngs['validation'],validation_cache,H_DRAWS,output_operator=output_operator,
            lplus_rng=rngs['validation_Lplus'],output_count=BATCH,nn_buckets=BUCKETS)
        source_step=0;source_active=0.
    else:
        for bits,residual,age in zip(saved['policy_history_bits'],saved['policy_history_residuals'],saved['policy_history_ages']):
            policy.history[int(bits)]=(float(residual),int(age))
        restore_cache(cache,saved,'operator');restore_cache(validation_cache,saved,'validation')
        vdata={key:jnp.asarray(saved['validation_'+key]) for key in DATA_KEYS}
        source_step=int(saved['absolute_step']);source_active=float(saved['cumulative_active_seconds'])
    initial_rng_after_guard=json.loads(json.dumps(rng_state()))
    metadata=dict(format_version=1,ne=12,flux=33,norb=34,two_lz=0,v1=1.,v3=0.,v5=0.,kappa=args.kappa,
        edge_potential=False,strict_LLL=True,sector_dimension=problem.op.dimension(),
        output_sector_dimension=output_operator.dimension(),L=problem.levels.tolist(),head_count=problem.S,
        head_angular_scale=problem.head_angular_scale,head_angular_scale_rule='Ne*(flux/2)*(flux/2+1)',
        raw_coherent_L2_unscaled=True,gamma=1.,gamma_aux=1.,eta=.001,learning_rate=.001,damping=.1,mu=.9,
        parameter_radius=.015,state_radius=.01,cg_maxiter=1024,cg_tol=1e-6,
        nominal_tau_H=.001,nominal_tau_L=.001*args.kappa,batch=BATCH,pool_size=POOL,Lplus_output_rows=BATCH,
        validation_root_rows=BATCH,validation_Lplus_output_rows=BATCH,geometry_anchor_rows=ANCHORS,
        geometry_anchor_update='history observes all32; anchors retain all32 selected roots',
        architecture='6x64 shared tanh all-layer readouts; private 2x16 tanh angular heads',
        parameter_count=problem.P,parameter_names=sorted(params),
        parameter_shapes={name:list(value.shape) for name,value in params.items()},
        seed=args.seed,init_seed=args.init_seed,nn_buckets=list(BUCKETS),K_row_width=1+12**2,conservative_NN_union_bound=CONSERVATIVE_UNION_BOUND,
        single_static_NN_bucket=5376,pilot_scope='pure normalized V1 Ne12 no-wall benchmark; no ED/full basis or Laughlin target',
        H_draws=H_DRAWS,cache_rows=128,history_rows=512,independent_validation_every=50,
        no_GN_curvature=True,method=problem.method_identifier,no_ED_or_full_basis_in_training=True,
        energy_estimator='finite-batch weighted sampled-row ratio with H8; no variational bound per batch',
        L2_estimator='positive exact incoming Lplus amplitudes; ratio remains finite-batch biased',
        active_budget_seconds=args.active_seconds,max_wall_seconds=args.max_wall_seconds,
        deadline_unix=args.deadline_unix,save_every=args.save_every,checkpoint_seconds=args.checkpoint_seconds,
        source_step=source_step,resume_checkpoint=str(args.resume) if args.resume else None,
        resume_checkpoint_sha256=sha(args.resume) if args.resume else None,
        optimizer_memory_reset=False if saved is not None else 'fresh zero initialization',
        initial_validation_guard='restored exactly' if saved is not None else 'fresh continued validation streams',
        input_sha256=hashes,devices=[str(d) for d in jax.devices()],output=str(output),
        stream_hash_scope='current process segment; immediate source prefixes stored in checkpoints')
    atomic_json(output/'config.json',metadata)
    for relative in hashes:
        path=output/'sources'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PROJECT/relative,path)
    print(json.dumps(dict(phase='initialized',ne=12,dimension=problem.op.dimension(),heads=problem.S,
        parameters=problem.P,source_step=source_step,output=str(output),devices=metadata['devices'])),flush=True)
    warm_rng=np.random.default_rng(np.random.SeedSequence([args.seed,9002]))
    compile_seconds=policy.compile(x,problem.op.sample_uniform(warm_rng,POOL))
    warm_cache=RootConnectionCache(problem.op,max_rows=128)
    warm_data,_,_=prepare_batch(problem.op,problem.op.sample_uniform(warm_rng,BATCH),warm_rng,warm_cache,H_DRAWS,
        output_operator=output_operator,lplus_rng=warm_rng,output_count=BATCH,nn_buckets=BUCKETS)
    tick=time.monotonic();problem.predict(x,warm_data)['root_residual'].block_until_ready()
    compile_seconds+=time.monotonic()-tick
    compiled={}
    warm_shape=(warm_data['occupations'].shape[0],vdata['occupations'].shape[0])
    print(json.dumps(dict(phase='compiling_step',shape=warm_shape)),flush=True)
    tick=time.monotonic()
    compiled[warm_shape]=problem.step.lower(x,previous,warm_data,jnp.asarray(.1),vdata,jnp.asarray(False)).compile()
    compile_seconds+=time.monotonic()-tick
    print(json.dumps(dict(phase='compiled',shape=warm_shape,total_compilation_seconds=compile_seconds)),flush=True)
    batch_hash=hashlib.sha256();pool_hash=hashlib.sha256();lplus_hash=hashlib.sha256()
    step=source_step;segment_step=0;loop_seconds=0.;active_start=time.monotonic()
    last_checkpoint=active_start;last_checkpoint_step=source_step;last_log=active_start;last_metrics=None

    def sampler_state():
        history=list(policy.history.items())
        return dict(policy_state_version=np.asarray(1),
            policy_history_bits=np.asarray([key for key,_ in history],dtype=np.int64),
            policy_history_residuals=np.asarray([value[0] for _,value in history],dtype=np.float64),
            policy_history_ages=np.asarray([value[1] for _,value in history],dtype=np.int64),
            policy_anchors=np.asarray(policy.anchors),probes=np.asarray(probes),
            **cache_state(cache,'operator'),**cache_state(validation_cache,'validation'),
            **{'validation_'+key:np.asarray(vdata[key]) for key in DATA_KEYS})

    restored=dict(parameters=np.asarray(x),previous_direction=np.asarray(previous),
        rng_json=np.asarray(json.dumps(rng_state())),**sampler_state())
    if saved is not None:
        exact={key:bool(np.array_equal(value,saved[key])) for key,value in restored.items()}
        if not all(exact.values()):raise RuntimeError('Incomplete resume restoration: '+str(exact))
        boundary=dict(status='passed',complete_restoration_exact=exact,no_replay=True,no_reset=True,
            source_checkpoint=str(args.resume),source_checkpoint_sha256=sha(args.resume))
    else:
        boundary=dict(status='passed',fresh_Ne12_initialization=True,no_other_system_or_wall_parameter_transfer=True,
            initial_rng_before_guard=initial_rng_before_guard,initial_rng_after_guard=initial_rng_after_guard)
    atomic_json(output/'boundary_verification.json',boundary)

    def save(archive=False):
        nonlocal last_checkpoint,last_checkpoint_step
        elapsed=time.monotonic()-active_start
        payload=dict(parameters=np.asarray(x),previous_direction=np.asarray(previous),step=step,
            absolute_step=step,policy_clock=step,segment_step=segment_step,training_seconds=loop_seconds,
            accounted_elapsed_seconds=elapsed,cumulative_active_seconds=source_active+elapsed,
            rng_json=np.asarray(json.dumps(rng_state())),metadata_json=np.asarray(json.dumps(metadata)),
            batch_stream_sha256=np.asarray(batch_hash.hexdigest()),pool_stream_sha256=np.asarray(pool_hash.hexdigest()),
            Lplus_stream_sha256=np.asarray(lplus_hash.hexdigest()),operator_pair_coefficients=problem.op.pair_coefficients,
            source_prefix_batch_stream_sha256=np.asarray(str(saved['batch_stream_sha256']) if saved is not None else ''),
            source_prefix_pool_stream_sha256=np.asarray(str(saved['pool_stream_sha256']) if saved is not None else ''),
            source_prefix_Lplus_stream_sha256=np.asarray(str(saved['Lplus_stream_sha256']) if saved is not None else ''),
            **sampler_state())
        latest=output/'checkpoint_latest.npz';atomic_npz(latest,**payload)
        if archive:
            path=output/f'checkpoints/step_{step:09d}.npz';temporary=path.with_name('.'+path.name+'.tmp')
            try:os.link(latest,temporary)
            except OSError:shutil.copyfile(latest,temporary)
            temporary.replace(path)
        last_checkpoint=time.monotonic();last_checkpoint_step=step
        print(json.dumps(dict(phase='checkpoint',step=step,segment_step=segment_step,
            cumulative_active_seconds=source_active+elapsed,path=str(latest),archive=archive)),flush=True)

    save(archive=True)

    def stop_reason():
        if STOP_REASON:return STOP_REASON
        if time.monotonic()-PROCESS_STARTED>=args.max_wall_seconds:return 'max_wall_seconds'
        if args.deadline_unix is not None and time.time()>=args.deadline_unix:return 'deadline_unix'
        if time.monotonic()-active_start>=args.active_seconds:return 'active_budget'
        if segment_step>=args.steps:return 'maximum_updates'
        return ''

    try:
        with (output/'metrics.csv').open('x',newline='',buffering=1) as handle:
            writer=None
            while not stop_reason():
                started=time.monotonic();next_step=step+1
                pool=problem.op.sample_uniform(rngs['pool'],POOL);pool_hash.update(pool.tobytes())
                roots,weights,selection,pool_stats=policy.select(x,pool,rngs['selection'],next_step,batch_size=BATCH)
                data,sample,coverage=prepare_batch(problem.op,roots,rngs['H'],cache,H_DRAWS,weights,
                    output_operator=output_operator,lplus_rng=rngs['Lplus'],output_count=BATCH,nn_buckets=BUCKETS)
                lplus_hash.update(sample['Lplus_outputs'].tobytes())
                for key in DATA_KEYS:batch_hash.update(np.asarray(sample[key]).tobytes())
                checked=next_step%50==0
                if checked:
                    vdata,_,_=prepare_batch(problem.op,problem.op.sample_uniform(rngs['validation'],BATCH),
                        rngs['validation'],validation_cache,H_DRAWS,output_operator=output_operator,
                        lplus_rng=rngs['validation_Lplus'],output_count=BATCH,nn_buckets=BUCKETS)
                shape=(data['occupations'].shape[0],vdata['occupations'].shape[0])
                if shape not in compiled:
                    print(json.dumps(dict(phase='compiling_step',shape=shape,step=next_step)),flush=True)
                    tick=time.monotonic()
                    compiled[shape]=problem.step.lower(x,previous,data,jnp.asarray(.1),vdata,jnp.asarray(checked)).compile()
                    compile_seconds+=time.monotonic()-tick
                tick=time.monotonic()
                new,raw,metrics=compiled[shape](x,previous,data,jnp.asarray(.1),vdata,jnp.asarray(checked))
                jax.block_until_ready(new);loop_seconds+=time.monotonic()-tick
                numeric={key:float(value) for key,value in metrics.items()}
                if not np.isfinite(np.asarray(new)).all() or not np.isfinite(np.asarray(raw)).all():
                    raise RuntimeError('Nonfinite proposed parameters/momentum; candidate not committed')
                if not np.isfinite(list(numeric.values())).all() or not numeric['cg_converged']:
                    raise RuntimeError('Nonfinite metrics or unconverged sampled CG: '+str(numeric))
                if min(numeric['minimum_head_root_norm_squared'],numeric['coherent_root_norm_squared'])<1e-200:
                    raise RuntimeError('Sampled normalization underflow')
                x,previous=new,raw;step=next_step;segment_step+=1
                observed=problem.predict(x,data)
                policy.observe(roots,np.asarray(observed['root_residual']),step)
                row=dict(update=segment_step,absolute_update=step,priority_clock=step,
                    training_seconds=loop_seconds,accounted_elapsed_seconds=time.monotonic()-active_start,
                    process_wall_seconds=time.monotonic()-PROCESS_STARTED,iteration_seconds=time.monotonic()-started,
                    compilation_seconds=compile_seconds,**coverage,**pool_stats,**numeric)
                row.update(tau_H_actual=numeric['effective_eta'],tau_L_actual=args.kappa*numeric['effective_eta'],
                    tau_aux_actual=numeric['effective_eta'])
                if writer is None:writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);last_metrics=row
                if segment_step%args.log_every==0 or time.monotonic()-last_log>=30:
                    print(json.dumps(dict(phase='training',**row)),flush=True);last_log=time.monotonic()
                if segment_step%args.save_every==0 or time.monotonic()-last_checkpoint>=args.checkpoint_seconds:
                    save(archive=segment_step%args.save_every==0)
        reason=stop_reason() or 'stopped'
        save(archive=True)
        for relative,digest in hashes.items():
            if sha(PROJECT/relative)!=digest:raise RuntimeError('Source changed during training: '+relative)
        receipt=dict(status='completed',stop_reason=reason,scientific_convergence_claimed=False,
            additional_updates=segment_step,source_step=source_step,final_step=step,
            active_elapsed_seconds=time.monotonic()-active_start,update_call_seconds=loop_seconds,
            total_process_seconds=time.monotonic()-PROCESS_STARTED,compilation_seconds=compile_seconds,
            checkpoint=str(output/'checkpoint_latest.npz'),checkpoint_sha256=sha(output/'checkpoint_latest.npz'),
            source_hashes_unchanged=True,last_metrics=last_metrics,
            batch_stream_sha256=batch_hash.hexdigest(),pool_stream_sha256=pool_hash.hexdigest(),
            Lplus_stream_sha256=lplus_hash.hexdigest())
        atomic_json(output/'completed.json',receipt)
        print(json.dumps(dict(phase='completed',**receipt)),flush=True)
    except Exception as error:
        candidate_pending='next_step' in locals() and next_step>step
        atomic_json(output/'failure.json',dict(status='failed',error=repr(error),last_in_memory_parameter_step=step,last_durable_checkpoint_step=last_checkpoint_step,
            failed_candidate_step=next_step if candidate_pending else None,failed_candidate_not_committed=candidate_pending,
            latest_checkpoint=str(output/'checkpoint_latest.npz'),last_metrics=last_metrics))
        raise


if __name__=='__main__':main()
