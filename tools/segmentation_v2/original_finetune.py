#!/usr/bin/env python3
"""Resource admission for the original cube checkpoint; no downloads or export.

The disposable benchmark performs four actual SGD steps, never publishes weights
and cannot seed a campaign. Failed admission prohibits campaign execution.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import numpy as np
from pilot import HERE, sha, write_json, bounded, HISTORICAL_RGB
from recover_original import safe_load, unpack, reconstruct, CHECKPOINT_SHA
from train import tensors, verified_rows
from dataset import enumerate_records
from integrate_dataset import validate_tensors
from campaign import audit_identities


def projection(policy, train_max, validation_max, peak_kib, free_bytes):
    values=(train_max,validation_max,peak_kib,free_bytes)
    if any(not math.isfinite(v) or v<=0 for v in values):
        return dict(admitted=False,reason='invalid resource measurement')
    steps=policy['epochs']*policy['training_samples']
    forwards=(policy['epochs']+1)*policy['validation_samples']
    factor=policy['feasibility_safety_factor']
    seconds=(steps*train_max+forwards*validation_max)*factor+policy['feasibility_overhead_seconds']
    rss=peak_kib*factor
    disk=policy['disk_reserve_gib']*1024**3+policy['epochs']*400*1024**2
    failures=[]
    if seconds>policy['wall_limit_seconds']:failures.append('wall_time')
    if rss>policy['memory_mib']*1024:failures.append('rss')
    if free_bytes<=disk:failures.append('disk')
    return dict(admitted=not failures,failures=failures,projected_seconds=seconds,
        projected_rss_kib=rss,required_disk_bytes=disk,optimizer_steps=steps,
        validation_forwards=forwards,method='maximum observed full SGD step and validation forward; 1.5 safety factor +240s startup/checkpoint/scoring overhead')


def save_measurement(folder, measurements):
    write_json(folder/f"measurement_{len(measurements):02d}.json",measurements[-1])


def deny_held_out(records):
    forbidden=[Path(r['folder']).resolve() for r in records if r['split']=='test']
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            path=Path(os.fsdecode(args[0])).resolve()
            if any(path==folder or folder in path.parents for folder in forbidden):
                raise RuntimeError('held-out artifact access prohibited')
    sys.addaudithook(guard)


def worker(args):
    import torch
    start=time.monotonic();policy=json.loads(args.policy.read_text())
    if sha(args.artifacts)!=policy['dataset_sha256'] or policy['checkpoint_sha256']!=CHECKPOINT_SHA:
        raise ValueError('dataset/initialization identity mismatch')
    for name in ('torch','torchvision'):
        if importlib.metadata.version(name).split('+')[0]!=policy[name]:raise ValueError('dependency mismatch')
    data=json.loads(args.artifacts.read_text());deny_held_out(data['records'])
    for split,count in [('train',300),('validation',12),('test',12)]:
        if len(enumerate_records(args.artifacts,split))!=count:raise ValueError('split membership')
    for name in ('capture.json.rgb8','capture.json.native512.rgb8'):
        hashes=[r['sha256'][name] for r in data['records']]
        if len(set(hashes))!=324 or set(hashes)&HISTORICAL_RGB:raise ValueError('duplicate/regression RGB')
    audit=audit_identities(args.artifacts);write_json(args.output/'identity_audit.json',audit)
    rows={split:list(verified_rows(args.artifacts,split)) for split in ('train','validation')}
    torch.set_num_threads(policy['cpu_threads']);torch.manual_seed(policy['seed'])
    random.seed(policy['seed']);np.random.seed(policy['seed']);torch.use_deterministic_algorithms(True)
    counts=Counter();sizes={};records=[]
    for split,items in rows.items():
        for row,folder,report,ids in items:
            image,target=tensors(folder,report,ids,torch);validate_tensors(image,target,torch)
            n=len(target['labels']);area=int(target['masks'].sum());sizes[row['scene_id']]=(n,area)
            counts[split+'_instances']+=n;counts[split+'_negatives']+=n==0
            records.append(dict(scene_id=row['scene_id'],seed=row['seed'],split=split,
                visible_instances=n,visible_pixels=area))
    if (counts['train_instances'],counts['train_negatives'],counts['validation_instances'])!=(484,50,22):
        raise ValueError('visible instance/negative count mismatch')
    write_json(args.output/'dataset_verification.json',dict(dataset_sha256=sha(args.artifacts),
        scenes={'train':300,'validation':12,'test_manifest_only':12},counts=dict(counts),
        duplicate_rgb=0,duplicate_scene_ids=0,duplicate_seeds=0,held_out_files_opened=0,
        class_mapping=data['class_mapping'],preprocessing=data['preprocessing'],records=records))
    checkpoint=safe_load(args.checkpoint,policy['checkpoint_sha256']);state,metadata=unpack(checkpoint)
    model=reconstruct(state)
    if not policy['train_backbone']:
        for parameter in model.backbone.parameters():parameter.requires_grad=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.cuda.manual_seed_all(policy['seed']);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.cuda.reset_peak_memory_stats()
    model.to(device)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
        lr=policy['learning_rate'],momentum=policy['momentum'],weight_decay=policy['weight_decay'])
    info=dict(device=device,cuda_available=torch.cuda.is_available(),torch_cuda_build=torch.version.cuda,
        cuda_device_count=torch.cuda.device_count(),cpu_count=os.cpu_count(),
        cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        available_ram_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))),
        free_disk_bytes=shutil.disk_usage(args.output).free,
        versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','onnx','onnxruntime','opencv-python')},
        checkpoint_sha256=sha(args.checkpoint),checkpoint_metadata_self_reported=metadata,
        strict_state_tensors=len(state),frozen_backbone=not policy['train_backbone'],
        model_transform=dict(min_size=list(model.transform.min_size),max_size=model.transform.max_size,
            mean=model.transform.image_mean,std=model.transform.image_std),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        policy_sha256=sha(args.policy),dataset_sha256=sha(args.artifacts),
        source_sha256={name:sha(HERE/name) for name in ('original_finetune.py','recover_original.py','train.py','pilot.py',
            'dataset.py','dataset_bindings.py','integrate_dataset.py','expansion.py','campaign.py')})
    if device=='cuda':info['gpu']=dict(name=torch.cuda.get_device_name(),total_vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    write_json(args.output/'identity.json',info)
    snapshots=args.output/'executed_sources';snapshots.mkdir()
    for name in info['source_sha256']:shutil.copy2(HERE/name,snapshots/name)
    shutil.copy2(args.policy,snapshots/'policy.json')
    # Geometry-based selection only; never predictions, held-out or historical input.
    train=rows['train'];key=lambda r:sizes[r[0]['scene_id']]
    selected=[next(r for r in train if key(r)[0]==0)]+sorted(train,key=key,reverse=True)[:2]
    selected.append(min((r for r in train if key(r)[0]),key=lambda r:key(r)[1]))
    measured=[]
    for row,folder,report,ids in selected:
        image,target=tensors(folder,report,ids,torch);model.train();optimizer.zero_grad(set_to_none=True)
        if device=='cuda':torch.cuda.synchronize()
        t=time.monotonic();parts=model([image.to(device)],[{k:v.to(device) for k,v in target.items()}])
        loss=sum(parts.values())
        if not torch.isfinite(loss) or 'loss_mask' not in parts:raise ValueError('invalid segmentation loss')
        loss.backward();optimizer.step()
        if device=='cuda':torch.cuda.synchronize()
        record=dict(kind='disposable_full_SGD_step',scene_id=row['scene_id'],seconds=time.monotonic()-t,
            visible_instances=len(target['labels']),losses={k:float(v.detach()) for k,v in parts.items()},
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        measured.append(record);save_measurement(args.output,measured);print(json.dumps(record),flush=True)
        del parts,loss
    optimizer.zero_grad(set_to_none=True);model.eval()
    val=rows['validation']
    with torch.inference_mode():
        for row,folder,report,ids in [val[0],max(val,key=key)]:
            image,_=tensors(folder,report,ids,torch)
            if device=='cuda':torch.cuda.synchronize()
            t=time.monotonic();pred=model([image.to(device)])[0]
            if device=='cuda':torch.cuda.synchronize()
            measured.append(dict(kind='disposable_validation_forward',scene_id=row['scene_id'],seconds=time.monotonic()-t))
            del pred
    write_json(args.output/'measurements.json',measured)
    train_max=max(r['seconds'] for r in measured if r['kind']=='disposable_full_SGD_step')
    val_max=max(r['seconds'] for r in measured if r['kind']=='disposable_validation_forward')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result=dict(**projection(policy,train_max,val_max,peak,shutil.disk_usage(args.output).free),
        device=device,measured_wall_seconds=time.monotonic()-start,peak_rss_kib=peak,
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        benchmark_optimizer_updates=len(selected),campaign_optimizer_updates=0,completed_epochs=0,
        checkpoint_saved=False,disposable_state_discarded=True,held_out_files_opened=0,
        validation_quality_evaluated=False,measurements=measured,identity_sha256=sha(args.output/'identity.json'))
    pinned=sha(args.checkpoint)
    if pinned!=CHECKPOINT_SHA:raise ValueError('original checkpoint changed')
    write_json(args.output/'admission.json',result);print(json.dumps(result),flush=True)
    return 0 if result['admitted'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['measure','worker'])
    parser.add_argument('--policy',type=Path,default=HERE/'original_finetune.json')
    for name in ('checkpoint','artifacts','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.action=='worker':sys.exit(worker(args))
    policy=json.loads(args.policy.read_text())
    if shutil.disk_usage(args.output.parent).free<=policy['disk_reserve_gib']*1024**3:raise ValueError('disk reserve')
    args.output.mkdir()
    command=[sys.executable,str(Path(__file__).resolve()),'worker']
    for name in ('policy','checkpoint','artifacts','output'):command+=['--'+name,str(getattr(args,name).resolve())]
    process=bounded(command,args.output,policy['benchmark_timeout_seconds'],
        dict(os.environ,OMP_NUM_THREADS=str(policy['cpu_threads']),CUBLAS_WORKSPACE_CONFIG=':4096:8'),memory_mib=policy['memory_mib'])
    write_json(args.output/'process.json',process);print(json.dumps(process))
    sys.exit(0 if process['exit_code']==0 else 2)
