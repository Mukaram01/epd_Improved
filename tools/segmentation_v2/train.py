#!/usr/bin/env python3
"""Supervised Mask R-CNN recipe. Never loads held-out scenes during training.

No ONNX fine-tuning and no automatic weight downloads. Use explicit fetch-weights.
CPU smoke checkpoints are deliberately ineligible for export/model selection.
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
import time
import urllib.request

import cv2
import numpy as np
from pilot import (HERE, HISTORICAL_RGB, sha, validate_native, write_json, mask_metrics,
                   bounded, validate_split_membership)


def verified_rows(artifacts, split):
    artifacts=Path(artifacts)
    if artifacts.is_file():
        from dataset import verified_rows as dataset_rows
        yield from dataset_rows(artifacts,split)
        return
    acquisition = json.loads((artifacts/'acquisition.json').read_text())
    if acquisition['status']!='ACQUIRED' or len(acquisition['scenes'])!=48:
        raise ValueError('training requires all 48 genuinely acquired scenes')
    if sha(artifacts/'scenes.json')!=acquisition['frozen_plan_sha256']:
        raise ValueError('changed frozen plan')
    validate_split_membership(json.loads((artifacts/'scenes.json').read_text()), acquisition)
    # Enumerate only train/validation records. Never inspect test pixels here.
    rows = [r for r in acquisition['scenes'] if r['split']==split]
    if len(rows)!={'train':24,'validation':12}[split]: raise ValueError('wrong split count')
    for row in rows:
        folder = artifacts/'dataset'/row['scene_id']
        for name, digest in row['sha256'].items():
            if sha(folder/name)!=digest: raise ValueError('changed acquisition artifact')
        if sha(folder/'capture.json.rgb8') in HISTORICAL_RGB: raise ValueError('historical input excluded')
        report = json.loads((folder/'capture.json').read_text())
        ids = np.fromfile(folder/'capture.json.native512.ids.u32',dtype='<u4').reshape(512,512)
        validate_native(report, ids)
        yield row, folder, report, ids


def tensors(folder, report, ids, torch):
    rgb = np.fromfile(folder/'capture.json.rgb8',dtype='uint8').reshape(256,256,3)
    rgb = cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR)
    image = torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255
    masks = []; boxes = []
    for r in report['mapping']:
        if r['class_id']!=1: continue
        mask = ids==r['renderer_id']
        if not mask.any(): continue  # no invisible-object target
        y,x = np.nonzero(mask)
        boxes.append([int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1])
        masks.append(mask)
    target = dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                  labels=torch.ones(len(boxes),dtype=torch.int64),
                  masks=torch.from_numpy(np.asarray(masks,dtype='uint8').reshape(-1,512,512)))
    return image, target


def fetch_weights(path, cfg):
    if shutil.disk_usage(path.parent).free<3*1024**3: raise ValueError('disk reserve <3 GiB')
    if path.exists(): raise FileExistsError(path)
    # This command is explicit; the immutable version and published hash prefix
    # are checked before weights are deserialized. Record the full SHA256.
    try:
        with urllib.request.urlopen(cfg['weights_url'],timeout=30) as response, path.open('xb') as out:
            total=0; start=time.monotonic()
            while True:
                block=response.read(1024*1024)
                if not block: break
                total+=len(block)
                if total>200*1024**2 or time.monotonic()-start>120: raise RuntimeError('weight download budget')
                out.write(block)
        digest=sha(path)
        if digest!=cfg['weights_sha256']: raise ValueError('pretrained hash mismatch')
        write_json(path.with_suffix('.manifest.json'),dict(url=cfg['weights_url'],sha256=digest,
                   architecture=cfg['architecture'],weights=cfg['initial_weights'],
                   code_license='BSD-3-Clause (torchvision)',training_data='COCO; see COCO image licences'))
    except Exception:
        # Preserve partial downloads, clearly named and never train on them.
        if path.exists(): path.rename(path.with_suffix('.failed-download'))
        raise


def build_model(cfg, weights, torch):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    model=maskrcnn_resnet50_fpn(weights=None,weights_backbone=None,min_size=512,max_size=512)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
    model.roi_heads.mask_predictor=MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels,256,2)
    if not cfg['train_backbone']:
        for p in model.backbone.parameters(): p.requires_grad=False
    return model


def selection_score(mean_iou, boundary, false_positives, positives, scenes):
    return mean_iou-boundary/512-false_positives/(positives+scenes)


def prediction_metrics(masks, ids, targets):
    """Score every confident prediction, including unmatched/negative detections.

    Best-IoU assignment here is evaluation only, never a modification to masks.
    Native pixel purity is not qualification of EMD's conservative association.
    """
    rows=[]
    for mask in masks:
        candidates=[(identity,mask_metrics(mask,ids,identity,targets)) for identity in targets]
        identity,m=max(candidates,key=lambda pair:pair[1]['iou']) if candidates else (
            -1,mask_metrics(mask,ids,-1,targets))
        rows.append(dict(reference_identity=identity,**m))
    return rows


def validation(model, rows, device, torch):
    """One-to-one greedy IoU matching; all metrics on unmodified >.5 masks."""
    model.eval(); metrics=[]; detections=[]; positives=0; recovered=0; false_positives=0
    with torch.no_grad():
        for _,folder,report,ids in rows:
            image,_=tensors(folder,report,ids,torch)
            pred=model([image.to(device)])[0]
            keep=pred['scores'].cpu().numpy()>=.8
            masks=pred['masks'].cpu().numpy()[keep,0]
            targets={r['renderer_id'] for r in report['mapping'] if r['class_id']==1 and (ids==r['renderer_id']).any()}
            detections.extend(dict(scene_id=folder.name,**m) for m in prediction_metrics(masks,ids,targets))
            positives+=len(targets)
            edges=[]
            for j,mask in enumerate(masks):
                for identity in targets:
                    m=mask_metrics(mask,ids,identity,targets);edges.append((m['iou'],j,identity,m))
            used_p=set();used_t=set();scene_recovered=0
            for iou,j,identity,m in sorted(edges,key=lambda e:e[0],reverse=True):
                if j in used_p or identity in used_t or iou<=0: continue
                used_p.add(j);used_t.add(identity);metrics.append(m)
                scene_recovered+=int(iou>=.5)
            # A recovered detection must meet IoU >=.5; mere overlap does not
            # excuse a false positive, including detections on negative images.
            recovered+=scene_recovered
            false_positives+=len(masks)-scene_recovered
            for identity in targets-used_t:
                metrics.append(mask_metrics(np.zeros_like(ids,dtype='f4'),ids,identity,targets))
    mean_iou=float(np.mean([m['iou'] for m in metrics])) if metrics else 0
    boundary=float(np.mean([m['boundary_mean_px'] if m['boundary_mean_px'] is not None else 512 for m in metrics])) if metrics else 512
    return dict(recall_iou50_at_confidence80=recovered/positives if positives else None,
                positives=positives,recovered=recovered,false_positives=false_positives,
                mean_iou=mean_iou,mean_boundary_px=boundary,
                selection_score=selection_score(mean_iou,boundary,false_positives,positives,len(rows)),
                instances=metrics,detections=detections,
                association_contract='NOT_VERIFIED_WITH_EMD_CONSERVATIVE_ASSOCIATION')


def worker(artifacts, weights, output, cfg):
    # Verify data BEFORE importing torch or loading any checkpoint.
    train=list(verified_rows(artifacts,'train')); val=list(verified_rows(artifacts,'validation'))
    manifest=json.loads(weights.with_suffix('.manifest.json').read_text())
    if sha(weights)!=manifest['sha256'] or manifest['sha256']!=cfg['weights_sha256']:
        raise ValueError('unpinned weights')
    for name in ('torch','torchvision'):
        if importlib.metadata.version(name).split('+')[0]!=cfg[name]: raise ValueError('training version mismatch: '+name)
    import torch
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(cfg['cpu_threads'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    smoke=device=='cpu';model=build_model(cfg,weights,torch).to(device)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                             lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    history=[];best=-float('inf');step=0;start=time.monotonic()
    write_json(output/'run_config.json',dict(config=cfg,device=device,smoke_only=smoke,
               training_source_sha256=sha(__file__),config_sha256=sha(HERE/'train.json'),
               dataset_manifest_sha256=sha(artifacts/'acquisition.json'),weights=manifest))
    for epoch in range(1 if smoke else cfg['epochs']):
        model.train();order=list(range(len(train)));random.Random(seed+epoch).shuffle(order)
        for index in order:
            _,folder,report,ids=train[index];image,target=tensors(folder,report,ids,torch)
            loss_parts=model([image.to(device)],[{k:v.to(device) for k,v in target.items()}])
            loss=sum(loss_parts.values())
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            if 'loss_mask' not in loss_parts: raise ValueError('missing instance segmentation loss')
            optimizer.zero_grad();loss.backward();optimizer.step();step+=1
            row=dict(epoch=epoch+1,step=step,total_loss=float(loss.detach()),
                     **{k:float(v.detach()) for k,v in loss_parts.items()})
            history.append(row)
            with (output/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n');f.flush()
            if smoke and step>=cfg['cpu_max_steps']: break
        score=None
        if not smoke:
            score=validation(model,val,device,torch)
            write_json(output/f'validation_epoch_{epoch+1}.json',score)
        if smoke or score['selection_score']>best:
            if score is not None: best=score['selection_score']
            checkpoint=output/('smoke.pt' if smoke else 'selected.pt')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,
                            epoch=epoch+1,step=step,smoke_only=smoke,config=cfg),checkpoint)
        if smoke: break
    with (output/'loss.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(history)
    # Dependency-free loss curve artifact; CSV/JSONL retain exact values.
    vals=[r['total_loss'] for r in history];scale=max(vals) or 1
    points=' '.join(f'{20+560*i/max(1,len(vals)-1):.2f},{180-160*v/scale:.2f}' for i,v in enumerate(vals))
    (output/'loss.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="220"><rect width="600" height="220" fill="white"/><polyline points="{points}" fill="none" stroke="blue"/><text x="20" y="210">Optimizer step vs total loss ({device})</text></svg>')
    write_json(output/'result.json',dict(status='CPU_SMOKE_ONLY_FULL_TRAINING_BLOCKED' if smoke else 'TRAINED',
               steps=step,epochs=epoch+1,wall_seconds=time.monotonic()-start,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               cuda_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
               checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
               held_out_evaluated=False,exported=False,production_qualified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['fetch-weights','train','worker'])
    p.add_argument('--artifacts',type=Path);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args()
    cfg=json.loads((HERE/'train.json').read_text())
    if not 1<=cfg['epochs']<=8 or cfg['cpu_max_steps']>2: raise ValueError('training budget exceeded')
    if a.action=='fetch-weights': fetch_weights(a.weights,cfg)
    elif a.action=='worker': worker(a.artifacts,a.weights,a.output,cfg)
    else:
        if not a.artifacts or not a.output: p.error('--artifacts and --output required')
        # An exclusive output directory permits exactly one training attempt.
        a.output.mkdir()
        import sys
        result=bounded([sys.executable,str(Path(__file__).resolve()),'worker','--artifacts',str(a.artifacts),
                        '--weights',str(a.weights),'--output',str(a.output)],a.output,
                       cfg['cuda_timeout_seconds'] if os.path.exists('/dev/nvidia0') else cfg['cpu_timeout_seconds'],
                       dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2'))
        write_json(a.output/'process.json',result)
        raise SystemExit(result['exit_code'])
