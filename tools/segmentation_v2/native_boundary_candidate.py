#!/usr/bin/env python3
"""One prospective boundary-loss campaign. Never executes without authorisation.
The parent publishes durable RUNNING before spawn, and actual supervision after.
This does not reconstruct the absent historical training process.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import numpy as np
from diagnostic_shadow_capture import durable_json,supervise
from shadow_mask_paste import digest,source_inventory

OUTPUT=Path('/home/user/epd-v2-artifacts/native-boundary-candidate-001')
IDENTITY='epd_native_boundary_candidate_001'


def atomic_json(path,value):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(name):os.unlink(name)


def supervise_training(command,output,wall,rss_kib):
    initial={'schema':'epd_training_supervisor/v1','status':'RUNNING','supervisor_pid':os.getpid(),
        'started_unix_ns':time.time_ns(),'exit_code':None,'resource_limit_reason':None,
        'wall_seconds':None,'peak_worker_rss_kib':None,
        'exit_code_semantics':'supervised worker OS exit, not supervisor self-observed OS exit'}
    durable_json(output/'process_started.json',initial);atomic_json(output/'process.json',initial)
    result=supervise(command,output,wall,rss_kib)
    record={**initial,**result,'finished_unix_ns':time.time_ns(),'supervision_sha256':digest(output/'supervision.json')}
    atomic_json(output/'process.json',record)
    return record


def verify_live_supervisor(args):
    """Reject direct hidden worker calls before opening training inputs."""
    import ctypes
    import signal
    try:
        parent=os.getppid()
        started=json.loads((OUTPUT/'process_started.json').read_text())
        admission=json.loads((OUTPUT/'admission.json').read_text())
        launch=json.loads((OUTPUT/'launch.json').read_text())
        # Popen can schedule child before the parent fsyncs worker_identity.
        deadline=time.monotonic()+2.
        while not (OUTPUT/'worker_identity.json').exists() and time.monotonic()<deadline:
            if os.getppid()!=parent:raise ValueError('parent died')
            time.sleep(.02)
        identity=json.loads((OUTPUT/'worker_identity.json').read_text())
        expected=[sys.executable,str(Path(__file__).resolve()),'--worker','--authorise-one-campaign',
            '--config',str(args.config.resolve()),'--review',str(args.review.resolve()),'--review-sha256',args.review_sha256]
        sig=ctypes.c_int()
        if ctypes.CDLL(None,use_errno=True).prctl(2,ctypes.byref(sig),0,0,0)!=0:raise ValueError('PDEATHSIG unavailable')
        if (started['status']!='RUNNING' or started['supervisor_pid']!=parent or launch['launcher_pid']!=parent or
            launch['command']!=expected or launch['wall_limit_seconds']!=10800 or launch['rss_limit_kib']!=6144*1024 or
            identity!={'pid':os.getpid(),'pgid':os.getpid(),'launcher_pid':parent} or os.getpgrp()!=os.getpid() or
            sig.value!=signal.SIGKILL or admission['configuration_sha256']!=digest(args.config) or
            admission['review_sha256']!=args.review_sha256):raise ValueError('launch binding mismatch')
    except (OSError,KeyError,ValueError,TypeError) as exc:
        raise ValueError('LIVE_SUPERVISOR_REQUIRED: '+str(exc)) from exc


def verify_inputs(p):
    for path,sha in {**p['sources_sha256'],**p['evidence_sha256']}.items():
        if digest(path)!=sha:raise ValueError('INPUT_SOURCE_DRIFT '+path)
    if digest(p['dataset']['path'])!=p['dataset']['sha256'] or digest(p['initialization']['path'])!=p['initialization']['sha256']:
        raise ValueError('DATASET_INITIALIZER_DRIFT')
    if source_inventory()!=p['runtime_inventory']:raise ValueError('RUNTIME_DRIFT')
    original=json.loads(Path(p['initialization']['process']).read_text())
    if original['exit_code']!=0 or original['resource_limit_reason'] is not None:raise ValueError('INITIALIZER_PROCESS_NOT_COMPLETE')
    data=json.loads(Path(p['dataset']['path']).read_text())
    train=sorted(r['scene_id'] for r in data['records'] if r['split']=='train')
    val=sorted(r['scene_id'] for r in data['records'] if r['split']=='validation')
    if train!=p['dataset']['train_ids'] or val!=p['dataset']['validation_ids'] or set(train)&set(val):raise ValueError('FROZEN_SPLIT_DRIFT')
    if len(train)!=300 or len(val)!=12 or len(set(train))!=300 or len(set(val))!=12:raise ValueError('SPLIT_COUNTS')
    if p['identity']!=IDENTITY or p['output_path']!=str(OUTPUT):raise ValueError('CANDIDATE_SCOPE_DRIFT')
    expected={'epochs':1,'updates':300,'seed':260110,'batch_size':1,'workers':0,'cpu_threads':2,
        'learning_rate':.0001,'momentum':.9,'weight_decay':.0001,'roi_mask_loss_weight':2.,
        'auxiliary_weight':.25,'boundary_radius':2.,'wall_seconds':10800,'rss_mib':6144,'disk_reserve_gib':5}
    if p['recipe']!=expected:raise ValueError('CANDIDATE_RECIPE_DRIFT')
    for item in p['baseline']['predictions']:
        if digest(item['path'])!=item['sha256']:raise ValueError('BASELINE_ARCHIVE_DRIFT')
    if sorted(r['scene_id'] for r in p['baseline']['predictions'])!=val:raise ValueError('BASELINE_SPLIT_DRIFT')
    return p


def state_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def worker(config,output):
    import torch
    import cv2
    from campaign import campaign_rows,epoch_order,validate_prediction
    from continuation import atomic_checkpoint
    from mask_resolution import configure56
    from recover_original import safe_load,reconstruct
    from original_finetune import deny_held_out
    from train import tensors
    from native_boundary_aux import training_losses,freeze_non_mask,non_mask_snapshot,assert_frozen,MASK_PREFIXES
    from native_boundary_validation import compare_scene,accept_scenes
    p=json.loads(config.read_text());verify_inputs(p)
    if torch.cuda.is_available():raise ValueError('CPU_ONLY')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(2)
    torch.use_deterministic_algorithms(True);random.seed(p['recipe']['seed']);np.random.seed(p['recipe']['seed']);torch.manual_seed(p['recipe']['seed'])
    manifest=json.loads(Path(p['dataset']['path']).read_text());deny_held_out(manifest['records'])
    train,val=campaign_rows(Path(p['dataset']['path']),{'training_samples':300})
    initial=safe_load(Path(p['initialization']['path']),p['initialization']['sha256'])
    if initial['epoch']!=2 or initial['step']!=600 or initial['history_sha256']!=p['initialization']['history_sha256'] or initial['validation_sha256']!=p['initialization']['validation_sha256']:raise ValueError('INITIALIZER_HISTORY_IDENTITY')
    model=configure56(reconstruct(initial['model']));freeze_non_mask(model)
    names=[n for n,q in model.named_parameters() if q.requires_grad]
    if len(names)!=12 or not all(n.startswith(MASK_PREFIXES) for n in names):raise ValueError('MASK_FREEZE_CONTRACT')
    frozen=non_mask_snapshot(model);frozen_hash=state_hash(frozen);del initial
    optimizer=torch.optim.SGD([q for q in model.parameters() if q.requires_grad],lr=.0001,momentum=.9,weight_decay=.0001)
    durable_json(output/'configuration.json',{'config':p,'configuration_sha256':digest(config),
        'trainable_names':names,'non_mask_state_sha256':frozen_hash,'held_out_files_opened':0,'stage_a2_used':False})
    history=output/'history.jsonl';history.touch(exist_ok=False);order=epoch_order(300,p['recipe']['seed'],1)
    for step,index in enumerate(order,1):
        if shutil.disk_usage(output).free<5*1024**3:raise ValueError('DISK_RESERVE')
        row,folder,report,ids=train[index];image,target=tensors(folder,report,ids,torch)
        model.train();optimizer.zero_grad(set_to_none=True);info={};start=time.monotonic()
        parts=training_losses(model,[image],[target],weight=p['recipe']['auxiliary_weight'],radius=p['recipe']['boundary_radius'],diagnostics=info)
        loss=sum(v*(2. if k=='loss_mask' else 1.) for k,v in parts.items())
        if not torch.isfinite(loss):raise ValueError('NONFINITE_LOSS')
        loss.backward()
        if any(q.grad is not None and not torch.isfinite(q.grad).all() for q in model.parameters()):raise ValueError('NONFINITE_GRADIENT')
        optimizer.step();assert_frozen(model,frozen)
        entry={'step':step,'epoch':1,'scene_id':row['scene_id'],'seconds':time.monotonic()-start,'loss':float(loss.detach()),
            'auxiliary':info,**{k:float(v.detach()) for k,v in parts.items()}}
        with history.open('a') as stream:
            stream.write(json.dumps(entry,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        del parts,loss,image,target
        if step%50==0:print(json.dumps({'updates':step,'expected':300}),flush=True)
    recorded=[json.loads(line) for line in history.read_text().splitlines()]
    if len(recorded)!=300 or [r['step'] for r in recorded]!=list(range(1,301)) or [r['scene_id'] for r in recorded]!=[train[i][0]['scene_id'] for i in order]:raise ValueError('HISTORY_INCOMPLETE')
    baseline={r['scene_id']:r for r in p['baseline']['predictions']};validation=[];archives=[];model.eval()
    folder=output/'validation';folder.mkdir()
    with torch.no_grad():
        for row,scene,report,ids in val:
            image,_=tensors(scene,report,ids,torch);pred={k:v.cpu().numpy() for k,v in model([image])[0].items()};validate_prediction(pred)
            path=folder/(row['scene_id']+'.npz')
            from diagnostic_shadow_capture import durable_npz
            durable_npz(path,pred);archives.append({'scene_id':row['scene_id'],'path':str(path),'sha256':digest(path)})
            with np.load(baseline[row['scene_id']]['path'],allow_pickle=False) as b:base={k:b[k] for k in b.files}
            raw256=np.fromfile(scene/'capture.json.ids.u32',dtype='<u4').reshape(256,256)
            cubes={int(r['renderer_id']) for r in report['mapping'] if r['class_id']==1}
            validation.append({'scene_id':row['scene_id'],**compare_scene(base,pred,ids,raw256,cubes)})
    if sorted(r['scene_id'] for r in validation)!=p['dataset']['validation_ids']:raise ValueError('VALIDATION_INCOMPLETE')
    gate=accept_scenes(validation,p['gate']['sentinels']);durable_json(output/'validation.json',{'scenes':validation,'gate':gate,'archives':archives})
    assert_frozen(model,frozen);verify_inputs(p)
    checkpoint=atomic_checkpoint(output/'UNQUALIFIED_epoch_01.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'epoch':1,'step':300,'policy':p,'history_sha256':digest(history),'validation_sha256':digest(output/'validation.json'),
        'initialization_sha256':p['initialization']['sha256'],'non_mask_state_sha256':frozen_hash,
        'torch_rng_state':torch.get_rng_state(),'python_rng_state':random.getstate()},torch)
    durable_json(output/'result.json',{'decision':gate['decision'],'updates':300,'validation_scenes':12,'gate':gate,
        'checkpoint':checkpoint,'history_sha256':digest(history),'validation_sha256':digest(output/'validation.json'),
        'frozen_before_sha256':frozen_hash,'frozen_after_sha256':state_hash(non_mask_snapshot(model)),
        'training_supervision':'AWAITING_PARENT_RECEIPT','original_56_training_supervision':'UNKNOWN',
        'stage_a2':'BLOCKED_0/2','production_qualified':False,'configuration_sha256':digest(config)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True)
    p.add_argument('--authorise-one-campaign',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.authorise_one_campaign:raise ValueError('SEPARATE_USER_TRAINING_AUTHORISATION_REQUIRED')
    if args.worker:verify_live_supervisor(args)
    policy=verify_inputs(json.loads(args.config.read_text()))
    review=json.loads(args.review.read_text())
    if digest(args.review)!=args.review_sha256 or review.get('configuration_sha256')!=digest(args.config) or review.get('decision')!='APPROVE_IMPLEMENTATION_ONLY' or not review.get('reviewer','').startswith('/root/'):
        raise ValueError('INDEPENDENT_REVIEW_REQUIRED')
    if args.worker:
        try:worker(args.config,OUTPUT);return 0
        except BaseException as exc:
            durable_json(OUTPUT/'failure.json',{'reason':type(exc).__name__+': '+str(exc),'partial_preserved':True});raise
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('OUTPUT_NOT_EXCLUSIVE')
    if shutil.disk_usage(OUTPUT.parent).free<6*1024**3:raise ValueError('DISK_RESERVE')
    OUTPUT.mkdir(exist_ok=False);durable_json(OUTPUT/'admission.json',{'configuration_sha256':digest(args.config),'review_sha256':args.review_sha256,'one_campaign_only':True})
    snapshots=OUTPUT/'executed_sources';snapshots.mkdir()
    for i,(path,sha) in enumerate(policy['sources_sha256'].items()):
        if Path(path).suffix in ('.py','.json','.cpp','.h','.hpp'):
            shutil.copyfile(path,snapshots/(str(i)+'_'+Path(path).name))
    durable_json(snapshots/'binding.json',policy['sources_sha256'])
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--authorise-one-campaign','--config',str(args.config.resolve()),'--review',str(args.review.resolve()),'--review-sha256',args.review_sha256]
    process=supervise_training(cmd,OUTPUT,10800,6144*1024)
    complete=process['status']=='WORKER_COMPLETED' and (OUTPUT/'result.json').exists()
    result=json.loads((OUTPUT/'result.json').read_text()) if complete else None
    if complete:
        verify_inputs(policy)
        complete=(result['updates']==300 and result['validation_scenes']==12 and result['frozen_before_sha256']==result['frozen_after_sha256'] and
            digest(OUTPUT/'history.jsonl')==result['history_sha256'] and digest(OUTPUT/'validation.json')==result['validation_sha256'] and
            digest(result['checkpoint']['path'])==result['checkpoint']['sha256'] and result['configuration_sha256']==digest(args.config))
    durable_json(OUTPUT/'completion.json',{'decision':'COMPLETE_TRAINING_ONLY' if complete else 'BLOCKED_FAILURE',
        'process_sha256':digest(OUTPUT/'process.json'),'result_sha256':digest(OUTPUT/'result.json') if result else None,
        'production_qualified':False,'stage_a2':'BLOCKED_0/2','original_training_supervision':'UNKNOWN','no_retry':True})
    print(json.dumps(process));return 0 if complete and result['decision']=='DEVELOPMENT_PASS_NOT_PRODUCTION' else 2

if __name__=='__main__':raise SystemExit(main())
