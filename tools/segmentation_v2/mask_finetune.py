#!/usr/bin/env python3
"""One bounded original-weight experiment. No automatic retries or export."""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import numpy as np
from pilot import HERE, sha, write_json, bounded
from recover_original import safe_load, unpack, reconstruct
from original_finetune import projection, deny_held_out
from train import tensors
from campaign import campaign_rows, epoch_order, validate_epoch, checkpoint_key
from continuation import atomic_checkpoint


def policy(path=None):
    return json.loads((path or HERE/'mask_finetune.json').read_text())


def weighted_loss(parts, weight):
    if 'loss_mask' not in parts:raise ValueError('instance segmentation loss required')
    return sum(value*(weight if name=='loss_mask' else 1.) for name,value in parts.items())


def admission(p):
    folder=Path(p['admission_evidence'])
    if sha(folder/'admission.json')!=p['admission_sha256']:raise ValueError('modified resource evidence')
    measured=json.loads((folder/'admission.json').read_text())
    process=json.loads((folder/'process.json').read_text())
    identity=json.loads((folder/'identity.json').read_text())
    if process['resource_limit_reason'] is not None or process['exit_code']!=2 or measured['benchmark_optimizer_updates']!=4:
        raise ValueError('incomplete retained measurement')
    if identity['checkpoint_sha256']!=p['checkpoint_sha256'] or identity['dataset_sha256']!=p['dataset_sha256']:
        raise ValueError('resource model/dataset mismatch')
    if not identity['frozen_backbone'] or identity['model_transform']['min_size']!=[800]:
        raise ValueError('incompatible compute architecture')
    timings=measured['measurements']
    train=max(r['seconds'] for r in timings if r['kind']=='disposable_full_SGD_step')
    val=max(r['seconds'] for r in timings if r['kind']=='disposable_validation_forward')
    result=projection(p,train,val,max(measured['peak_rss_kib'],process['sampled_process_peak_rss_kib']),shutil.disk_usage(folder).free)
    return dict(result,resource_evidence_sha256=p['admission_sha256'],
        basis=('same original architecture and forward pass; mask-only backward subset; 28-grid pooling/weighted BCE covered by unchanged1.5 factor'
            if p.get('schema')=='epd_boundary_mask/1' else
            'same original model, frozen backbone, batch1 FP32 two CPU threads; only scalar loss weight/LR differ; unchanged1.5 factor'),
        prior_eight_epoch_admission_remains_failed=True)


def worker(args):
    import torch,importlib.metadata
    p=policy(args.policy);admit=admission(p)
    write_json(args.output/'admission.json',admit)
    if not admit['admitted']:raise RuntimeError('resource admission rejected')
    if torch.cuda.is_available():raise RuntimeError('CPU resource identity changed')
    for name in ('torch','torchvision'):
        if importlib.metadata.version(name).split('+')[0]!=p[name]:raise RuntimeError('dependency drift')
    if sha(args.artifacts)!=p['dataset_sha256']:raise ValueError('dataset drift')
    data=json.loads(args.artifacts.read_text());deny_held_out(data['records'])
    train,val=campaign_rows(args.artifacts,p)
    torch.set_num_threads(p['cpu_threads']);torch.use_deterministic_algorithms(True)
    random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed'])
    boundary=p.get('schema')=='epd_boundary_mask/1'
    if boundary:
        if sha(args.checkpoint)!=p['checkpoint_sha256']:raise ValueError('original checkpoint drift')
        initial=safe_load(Path(p['initialization_checkpoint']),p['initialization_sha256'])
        state=initial['model'];metadata=dict(initial_epoch=initial['epoch'],initial_step=initial['step'])
        model=reconstruct(state)
        import torchvision.models.detection.roi_heads as roi_heads
        from boundary_loss import boundary_maskrcnn_loss
        from functools import partial
        roi_heads.maskrcnn_loss=partial(boundary_maskrcnn_loss,factor=p['boundary_factor'])
        for name,parameter in model.named_parameters():
            parameter.requires_grad=name.startswith(('roi_heads.mask_head.','roi_heads.mask_predictor.'))
        del initial
    else:
        state,metadata=unpack(safe_load(args.checkpoint,p['checkpoint_sha256']))
        model=reconstruct(state)
    for parameter in model.backbone.parameters():parameter.requires_grad=False
    optimizer=torch.optim.SGD([v for v in model.parameters() if v.requires_grad],lr=p['learning_rate'],momentum=p['momentum'],weight_decay=p['weight_decay'])
    sources=('mask_finetune.py','mask_finetune.json','recover_original.py','original_finetune.py','train.py','campaign.py',
             'pilot.py','dataset.py','dataset_bindings.py','continuation.py','expansion.py')
    if boundary:sources+=('boundary_loss.py','boundary_mask.json')
    binding=dict(sources={name:sha(HERE/name) for name in sources},dataset=sha(args.artifacts),original=p['checkpoint_sha256'],
        versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy')})
    snapshots=args.output/'executed_sources';snapshots.mkdir()
    for name in sources:shutil.copy2(HERE/name,snapshots/name)
    write_json(args.output/'configuration.json',dict(policy=p,binding=binding,original_metadata_self_reported=metadata,
        fresh_optimizer=True,held_out_files_opened=0,trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
    # Genuine original baseline, same model/tensors/scorer as candidates; no Stage-A2 input in selection.
    if boundary:
        baseline_path=Path(p['baseline_metrics'])
        if sha(baseline_path)!=p['baseline_metrics_sha256']:raise ValueError('retained baseline changed')
        baseline=json.loads(baseline_path.read_text())
    else:
        baseline=validate_epoch(model,val,'cpu',torch,args.output/'baseline',0,p)
    write_json(args.output/'baseline_summary.json',baseline)
    del state
    start=time.monotonic();history=[];checkpoints=[];scores=[]
    for epoch in range(1,p['epochs']+1):
        model.train()
        for index in epoch_order(len(train),p['seed'],epoch):
            if shutil.disk_usage(args.output).free<p['disk_reserve_gib']*1024**3:raise RuntimeError('disk reserve')
            row,folder,report,ids=train[index];t=time.monotonic();image,target=tensors(folder,report,ids,torch)
            optimizer.zero_grad(set_to_none=True);parts=model([image],[target]);loss=weighted_loss(parts,p['mask_loss_weight'])
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();optimizer.step()
            record=dict(epoch=epoch,step=len(history)+1,scene_id=row['scene_id'],total_loss=float(loss.detach()),
                seconds=time.monotonic()-t,**{k:float(v.detach()) for k,v in parts.items()})
            history.append(record)
            with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
            if len(history)%100==0:print(json.dumps(dict(step=len(history),epoch=epoch,seconds=time.monotonic()-start)),flush=True)
            del parts,loss
        score=validate_epoch(model,val,'cpu',torch,args.output/f'validation_{epoch:02d}',epoch,p);scores.append(score)
        n=np.random.get_state()
        checkpoint=atomic_checkpoint(args.output/f'epoch_{epoch:02d}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            epoch=epoch,step=len(history),seed=p['seed'],binding=binding,policy=p,torch_rng_state=torch.get_rng_state(),
            python_rng_state=random.getstate(),numpy_rng_state=dict(algorithm=n[0],keys=n[1].tolist(),pos=n[2],has_gauss=n[3],cached_gaussian=n[4]),
            history_sha256=sha(args.output/'history.jsonl'),validation_sha256=sha(args.output/f'validation_{epoch:02d}/metrics.json')),torch)
        checkpoints.append(checkpoint)
        write_json(args.output/f'epoch_{epoch:02d}_summary.json',dict(checkpoint=checkpoint,metrics=score,
            mean_losses={key:float(np.mean([r[key] for r in history if r['epoch']==epoch])) for key in
                ('total_loss','loss_mask','loss_classifier','loss_box_reg','loss_objectness','loss_rpn_box_reg')}))
        print(json.dumps(dict(epoch=epoch,recovered=score['recovered'],iou=score['mean_iou'],spill=score['background_spill'],eligible=score['eligible'])),flush=True)
    valid=[s for s in scores if s['eligible']];best=max(valid,key=checkpoint_key) if valid else None
    result=dict(status='VALIDATION_PASS' if best else 'TRAINED_REJECTED',steps=len(history),epochs=len(scores),
        training_validation_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        baseline=baseline,checkpoints=checkpoints,epoch_metrics=scores,
        selected=next(c for c in checkpoints if c['epoch']==best['epoch']) if best else None,
        held_out_files_opened=0,stage_a2_used_for_selection=False,exported=False)
    write_json(args.output/'result.json',result)
    if sha(args.checkpoint)!=p['checkpoint_sha256']:raise ValueError('original checkpoint changed')
    print(json.dumps({key:result[key] for key in ('status','steps','epochs','training_validation_seconds','peak_rss_kib','selected')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','worker'])
    parser.add_argument('--policy',type=Path,default=HERE/'mask_finetune.json')
    for name in ('checkpoint','artifacts','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.action=='worker':worker(args)
    else:
        p=policy(args.policy);admit=admission(p)
        if not admit['admitted']:raise RuntimeError('resource admission rejected before campaign')
        args.output.mkdir()
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--policy',str(args.policy.resolve())]
        for name in ('checkpoint','artifacts','output'):command+=['--'+name,str(getattr(args,name).resolve())]
        process=bounded(command,args.output,p['wall_limit_seconds'],dict(os.environ,OMP_NUM_THREADS='2'),memory_mib=p['memory_mib'])
        write_json(args.output/'process.json',process);print(json.dumps(process))
        sys.exit(0 if process['exit_code']==0 else 2)
