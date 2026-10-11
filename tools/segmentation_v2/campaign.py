#!/usr/bin/env python3
"""One full supervised campaign. Frozen validation gate; no held-out access.

Reuses the pilot's pinned architecture/initialization, tensor preparation and native
label verification. CPU requires an independently recorded full-run feasibility
pass, rather than falling back to a smoke model. Large artifacts remain external.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import cv2
import numpy as np
from pilot import HERE, sha, write_json, mask_metrics, bounded
from train import build_model, tensors, verified_rows


def validate_identity_mapping(spec, report):
    objects=spec['objects'];expected={o['name']:o for o in objects}
    mapping=report['mapping'];inventory=report['inventory']
    if len(expected)!=len(objects) or len(mapping)!=len(objects) or not inventory['complete'] or inventory['collisions']:
        raise ValueError('ambiguous authored/native object inventory')
    for field in ('model','renderer_id','item_id','visual_entity','link_entity','model_entity'):
        if len({r[field] for r in mapping})!=len(mapping):raise ValueError('duplicate identity: '+field)
    if {r['model'] for r in mapping}!=set(expected):raise ValueError('authored object identity mismatch')
    tables={kind:{r['id']:r for r in inventory[kind]} for kind in ('models','links','visuals')}
    for row in mapping:
        obj=expected[row['model']]
        if row['class_id']!=obj['class_id'] or row['class_id'] not in (0,1):raise ValueError('authored class mismatch')
        visual=tables['visuals'].get(row['visual_entity']);link=tables['links'].get(row['link_entity'])
        if visual is None or link is None or row['model_entity'] not in tables['models'] or \
           visual['parent']!=row['link_entity'] or link['parent']!=row['model_entity']:
            raise ValueError('native visual/link/model ownership mismatch')
        if visual['geometry']['type']!='BOX' or not np.allclose(visual['geometry']['size'],obj['size'],rtol=1e-6,atol=1e-9):
            raise ValueError('authored/native geometry identity mismatch')


def audit_identities(artifacts):
    plan=json.loads((artifacts/'scenes.json').read_text())
    specs={s['scene_id']:s for s in plan['scenes']};checked=[]
    for split in ('train','validation'):
        for row,folder,report,_ in verified_rows(artifacts,split):
            spec=specs[row['scene_id']]
            if json.loads((folder/'scene.json').read_text())!=spec:raise ValueError('scene specification drift')
            validate_identity_mapping(spec,report)
            checked.append(dict(scene_id=folder.name,split=split,mapped_objects=len(report['mapping'])))
    acquisition=json.loads((artifacts/'acquisition.json').read_text())
    if any(r['runtime']['exit_code']!=0 for r in acquisition['scenes']):raise ValueError('failed retained acquisition')
    return dict(scope='authored object/class, unique renderer/Item/visual/link/model identities and native ownership; '
                 'one-to-one mask evaluation assignments; NOT EMD runtime association or zero pixel contamination',
                contract_violations=0,checked_scenes=checked,held_out_files_opened=0,
                acquisition_outcomes_verified_from_manifest=48)


def eligible(metrics, policy):
    gate=policy['gate']
    return (metrics['positives']==gate['positives'] and
            metrics['recovered']>=gate['minimum_recovered'] and
            metrics['mean_iou']>gate['mean_iou_strict_gt'] and
            metrics['false_positives']<=gate['maximum_false_positives'] and
            metrics['background_spill']<gate['background_spill_strict_lt'] and
            metrics['contract_violations']==gate['maximum_contract_violations'])


def checkpoint_key(metrics):
    return (metrics['recovered'],metrics['mean_iou'],-metrics['background_spill'],-metrics['epoch'])


def validate_prediction(pred, size=512):
    if set(pred)!={'boxes','labels','scores','masks'}:raise ValueError('output keys')
    n=len(pred['scores'])
    for name,shape,dtype in [('boxes',(n,4),np.float32),('labels',(n,),np.int64),
                              ('scores',(n,),np.float32),('masks',(n,1,size,size),np.float32)]:
        value=pred[name]
        if value.shape!=shape or value.dtype!=dtype:raise ValueError(name+' shape/dtype')
        if not np.isfinite(value).all():raise ValueError(name+' nonfinite')
    if (pred['labels']!=1).any():raise ValueError('cube class index')
    for name in ('scores','masks'):
        if ((pred[name]<0)|(pred[name]>1)).any():raise ValueError(name+' probability range')
    boxes=pred['boxes']
    if (boxes<0).any() or (boxes>size).any():raise ValueError('out-of-raster boxes')
    if (boxes[:,2:]<=boxes[:,:2]).any():raise ValueError('malformed boxes')


def score_predictions(scene, pred, ids, targets):
    """Evaluation-only one-to-one IoU matching, preserving every prediction.

    Native IDs are used solely as scoring targets, never fed to the model or used
    for cropping/correcting its output. Match algorithm mirrors historical metrics.
    """
    size=ids.shape[0];validate_prediction(pred,size)
    keep=np.flatnonzero(pred['scores']>=.8)
    masks=pred['masks'][keep,0]
    edges=[];detections=[]
    for j,mask in enumerate(masks):
        candidates=[]
        for identity in sorted(targets):
            metric=mask_metrics(mask,ids,identity,targets)
            edges.append((metric['iou'],j,identity,metric));candidates.append((identity,metric))
        identity,metric=max(candidates,key=lambda pair:pair[1]['iou']) if candidates else (
            -1,mask_metrics(mask,ids,-1,targets))
        detections.append(dict(scene_id=scene,prediction_index=int(keep[j]),score=float(pred['scores'][keep[j]]),
            label=int(pred['labels'][keep[j]]),box=pred['boxes'][keep[j]].tolist(),
            reference_identity=identity,**metric))
    used_p=set();used_t=set();instances=[];recovered=0
    for iou,j,identity,metric in sorted(edges,key=lambda e:e[0],reverse=True):
        if j in used_p or identity in used_t or iou<=0:continue
        used_p.add(j);used_t.add(identity);recovered+=int(iou>=.5)
        instances.append(dict(scene_id=scene,renderer_identity=identity,prediction_index=int(keep[j]),**metric))
    for identity in sorted(targets-used_t):
        instances.append(dict(scene_id=scene,renderer_identity=identity,prediction_index=None,
            **mask_metrics(np.zeros_like(ids,dtype=np.float32),ids,identity,targets)))
    return dict(scene_id=scene,negative=not bool(targets),positives=len(targets),
        confident_detections=len(keep),recovered=recovered,false_positives=len(keep)-recovered,
        mean_iou=float(np.mean([m['iou'] for m in instances])) if instances else 0.,
        background_spill=sum(d['background_spill'] for d in detections),
        cross_object=sum(d['cross_object'] for d in detections),
        missing_visible=sum(m['missing_visible'] for m in instances),
        instances=instances,detections=detections,contract_violations=0)


def validate_epoch(model, rows, device, torch, folder, epoch, policy):
    folder.mkdir();model.eval();scenes=[];raw=[]
    with torch.no_grad():
        for row,scene,report,ids in rows:
            start=time.monotonic();image,_=tensors(scene,report,ids,torch)
            result=model([image.to(device)])[0]
            pred={name:value.detach().cpu().numpy() for name,value in result.items()}
            validate_prediction(pred)
            path=folder/(scene.name+'.npz');np.savez_compressed(path,**pred)
            raw.append(dict(scene_id=scene.name,path=str(path),sha256=sha(path),seconds=time.monotonic()-start,
                            predictions=len(pred['scores'])))
            targets={r['renderer_id'] for r in report['mapping'] if r['class_id']==1 and (ids==r['renderer_id']).any()}
            scenes.append(score_predictions(scene.name,pred,ids,targets))
    instances=[m for s in scenes for m in s['instances']]
    summary={key:sum(s[key] for s in scenes) for key in ('positives','recovered','false_positives',
             'background_spill','cross_object','missing_visible','contract_violations')}
    summary.update(epoch=epoch,mean_iou=float(np.mean([m['iou'] for m in instances])),
        mean_boundary_px=float(np.mean([m['boundary_mean_px'] if m['boundary_mean_px'] is not None else 512
                                      for m in instances])),
        max_boundary_px=max((m['boundary_max_px'] if m['boundary_max_px'] is not None else 512) for m in instances),
        confidence_threshold=.8,mask_threshold_strict_gt=.5,scenes=scenes,raw_predictions=raw,
        held_out_evaluated=False,authoritative_emd_association='NOT_EXECUTED')
    summary['recall']=summary['recovered']/summary['positives'];summary['eligible']=eligible(summary,policy)
    write_json(folder/'metrics.json',summary)
    return summary


def worker(args):
    import torch
    cfg=json.loads((HERE/'train.json').read_text());policy=json.loads((HERE/'campaign.json').read_text())
    # Do not admit policy drift or a different attempt after the preflight.
    identity=json.loads((args.preflight/'identity.json').read_text())
    if sha(HERE/'campaign.json')!=identity['campaign_sha256'] or sha(HERE/'train.json')!=identity['config_sha256']:
        raise ValueError('configuration changed after feasibility declaration')
    for name,digest in identity['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('source changed after feasibility: '+name)
    if sha(args.weights)!=cfg['weights_sha256']:raise ValueError('initialization SHA mismatch')
    for name in ('torch','torchvision'):
        if importlib.metadata.version(name).split('+')[0]!=cfg[name]:raise ValueError('dependency version mismatch')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cpu':
        feasibility=json.loads((args.preflight/'feasibility.json').read_text())
        if not feasibility['feasible'] or feasibility['optimizer_updates']!=0:raise ValueError('CPU full-run feasibility not approved')
    if cfg['seed']!=policy['seed'] or cfg['batch_size']!=1 or cfg['epochs']!=8:raise ValueError('recipe drift')
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True)
    if device=='cuda':
        torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True;torch.cuda.reset_peak_memory_stats()
    train=list(verified_rows(args.artifacts,'train'));val=list(verified_rows(args.artifacts,'validation'))
    if sha(args.artifacts/'acquisition.json')!=identity['acquisition_sha256']:raise ValueError('acquisition identity drift')
    write_json(args.output/'identity_audit.json',audit_identities(args.artifacts))
    model=build_model(cfg,args.weights,torch).to(device)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],
        momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    snapshots=args.output/'executed_sources';snapshots.mkdir()
    for name in ('campaign.py','campaign.json','train.py','train.json','pilot.py'):
        shutil.copy2(HERE/name,snapshots/name)
    write_json(args.output/'run_config.json',dict(device=device,recipe=cfg,policy=policy,
        sources={p.name:sha(p) for p in snapshots.iterdir()},weights_sha256=sha(args.weights),
        acquisition_sha256=sha(args.artifacts/'acquisition.json'),seed=seed,batch_size=1,
        initialized_from_rejected_smoke=False,smoke_only=False))
    start=time.monotonic();step=0;history=[];epochs=[];best=None;checkpoints=[]
    for epoch in range(1,policy['epochs']+1):
        if shutil.disk_usage(args.output).free<policy['disk_reserve_gib']*1024**3:raise RuntimeError('disk reserve')
        model.train();order=list(range(len(train)));random.Random(seed+epoch-1).shuffle(order)
        for index in order:
            row,scene,report,ids=train[index];image,target=tensors(scene,report,ids,torch)
            iteration=time.monotonic();optimizer.zero_grad(set_to_none=True)
            parts=model([image.to(device)],[{k:v.to(device) for k,v in target.items()}])
            loss=sum(parts.values())
            if not torch.isfinite(loss) or 'loss_mask' not in parts:raise ValueError('invalid segmentation loss')
            loss.backward();optimizer.step();step+=1
            record=dict(epoch=epoch,step=step,scene_id=scene.name,total_loss=float(loss.detach()),
                seconds=time.monotonic()-iteration,elapsed_seconds=time.monotonic()-start,
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                **{k:float(v.detach()) for k,v in parts.items()})
            history.append(record)
            with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
        checkpoint=args.output/f'epoch_{epoch:02d}.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,epoch=epoch,step=step,
            smoke_only=False,config=cfg,campaign=policy,torch_rng_state=torch.get_rng_state(),
            numpy_rng_state=np.random.get_state(),python_rng_state=random.getstate(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if device=='cuda' else None),checkpoint)
        checkpoints.append(dict(epoch=epoch,path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size))
        score=validate_epoch(model,val,device,torch,args.output/f'validation_{epoch:02d}',epoch,policy)
        epochs.append(score)
        if score['eligible'] and (best is None or checkpoint_key(score)>checkpoint_key(best)):best=score
        write_json(args.output/f'epoch_{epoch:02d}_summary.json',dict(checkpoint=checkpoints[-1],metrics=score))
        print(json.dumps(dict(epoch=epoch,steps=step,recovered=score['recovered'],mean_iou=score['mean_iou'],
            false_positives=score['false_positives'],background_spill=score['background_spill'],eligible=score['eligible'],
            wall_seconds=time.monotonic()-start)),flush=True)
    with (args.output/'loss.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
    vals=[r['total_loss'] for r in history];scale=max(vals) or 1
    points=' '.join(f'{20+560*i/max(1,len(vals)-1):.2f},{180-160*v/scale:.2f}' for i,v in enumerate(vals))
    (args.output/'loss.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="220">'
        '<rect width="600" height="220" fill="white"/><polyline points="'+points+'" fill="none" stroke="blue"/>'
        '<text x="20" y="210">Optimizer steps vs loss: full eight-epoch campaign</text></svg>')
    selected=next(c for c in checkpoints if c['epoch']==best['epoch']) if best else None
    write_json(args.output/'result.json',dict(status='VALIDATION_ELIGIBLE' if best else 'V2_TRAINED_BUT_REJECTED',
        device=device,steps=step,epochs=len(epochs),wall_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        checkpoints=checkpoints,selected=selected,selected_validation=best,
        epoch_metrics=[{k:s[k] for k in ('epoch','recovered','positives','recall','mean_iou','false_positives',
                'background_spill','cross_object','missing_visible','mean_boundary_px','contract_violations','eligible')} for s in epochs],
        held_out_evaluated=False,historical_regression_evaluated=False,exported=False,production_qualified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','worker']);parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.action=='worker':worker(args)
    else:
        policy=json.loads((HERE/'campaign.json').read_text());args.output.mkdir()
        identity=json.loads((args.preflight/'identity.json').read_text())
        timeout=policy['cuda_timeout_seconds'] if identity['cuda_available'] else policy['cpu_timeout_seconds']
        result=bounded([sys.executable,str(Path(__file__).resolve()),'worker','--artifacts',str(args.artifacts),
            '--weights',str(args.weights),'--preflight',str(args.preflight),'--output',str(args.output)],
            args.output,timeout,dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2'),
            memory_mib=policy['memory_mib'])
        write_json(args.output/'process.json',result);print(json.dumps(result),flush=True)
        raise SystemExit(result['exit_code'])
