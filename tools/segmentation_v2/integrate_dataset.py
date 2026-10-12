#!/usr/bin/env python3
"""Verify and integrate genuine training data. Never trains or runs a model."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time
import cv2
import numpy as np
import dataset
from expansion import (CAMERA, validate_plan, verify_completed, require_complete,
                       label_integrity, validate_sample, register_rgb, definition_sha)
from pilot import sha, write_json, scene_plan
from train import verified_rows as pilot_rows, tensors


def check_membership(original,new):
    all_scenes=original+new
    if len(original)!=48 or len(new)!=276:raise ValueError('membership count')
    for field in ('scene_id','seed'):
        if len({s[field] for s in all_scenes})!=324:raise ValueError('split contamination: '+field)
    if original!=scene_plan()['scenes'] or any(s['split']!='train' for s in new):raise ValueError('frozen split change')
    if len({definition_sha(s) for s in all_scenes})!=324:raise ValueError('repeated scene definitions')


def projected_silhouette(obj,camera):
    """Geometric raster estimate for occlusion statistics ONLY; never a label/input."""
    x,y,z,r,p,yaw=obj['pose'];sx,sy,sz=np.array(obj['size'])/2
    cr,sr,cp,sp,cy,syaw=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(yaw),math.sin(yaw)
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    rz=np.array([[cy,-syaw,0],[syaw,cy,0],[0,0,1]])
    vertices=np.array(list(itertools.product((-sx,sx),(-sy,sy),(-sz,sz))))@(rz@ry@rx).T+[x,y,z]
    pos=np.array(camera['position']);forward=np.array(camera['look_at'])-pos;forward/=np.linalg.norm(forward)
    right=np.cross(forward,camera['up']);right/=np.linalg.norm(right);up=np.cross(right,forward)
    delta=vertices-pos;depth=delta@forward
    if (depth<=camera['near']).any():raise ValueError('occlusion projection near plane')
    focal=256/math.tan(camera['fovy_radians']/2)
    points=np.column_stack((256+focal*(delta@right)/depth,256-focal*(delta@up)/depth))
    hull=cv2.convexHull(np.rint(points).astype('int32'));mask=np.zeros((512,512),dtype='u1')
    cv2.fillConvexPoly(mask,hull,1);return mask.astype(bool)


def occlusion_stat(obj,ids,identity,camera):
    silhouette=projected_silhouette(obj,camera);visible=ids==identity
    area=int(silhouette.sum());count=int(visible.sum())
    return dict(projected_silhouette_pixels_estimate=area,visible_pixels=count,
                estimated_occluded_fraction=max(0.,min(1.,1-count/max(1,area))),
                visible_pixels_outside_estimated_silhouette=int((visible&~silhouette).sum()))


def validate_tensors(image,target,torch):
    if image.shape!=(3,512,512) or image.dtype!=torch.float32 or not torch.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError('RGB01 CHW tensor contract')
    n=len(target['labels'])
    for name,shape,dtype in [('boxes',(n,4),torch.float32),('labels',(n,),torch.int64),('masks',(n,512,512),torch.uint8)]:
        if target[name].shape!=shape or target[name].dtype!=dtype:raise ValueError('target '+name+' shape/dtype')
    if (target['labels']!=1).any() or not torch.isfinite(target['boxes']).all():raise ValueError('class/box integrity')
    if n and ((target['boxes'][:,2:]<=target['boxes'][:,:2]).any() or
              (target['boxes']<0).any() or (target['boxes']>512).any() or
              (target['masks'].flatten(1).sum(1)==0).any()):raise ValueError('invalid visible object target')
    if ((target['masks']!=0)&(target['masks']!=1)).any():raise ValueError('nonbinary supervisory masks')


def verify_batch_seals(root,plan):
    paths=sorted((root/'attempts').glob('*.json'))
    if [p.name for p in paths]!=[f'{i:03d}.json' for i in range(276)]:raise ValueError('attempt coverage')
    attempts=[json.loads(p.read_text()) for p in paths];require_complete(plan,attempts)
    hashes={p.name:sha(p) for p in paths};acquisition=json.loads((root/'acquisition.json').read_text())
    if acquisition['status']!='ACQUIRED' or acquisition['attempt_manifest_sha256']!=hashes:raise ValueError('acquisition seal')
    seals=sorted((root/'batches').glob('*.json'))
    if [p.name for p in seals]!=[f'batch_{i:02d}.json' for i in range(6)]:raise ValueError('batch coverage')
    for batch,path in enumerate(seals):
        seal=json.loads(path.read_text());expected={f'{i:03d}.json':hashes[f'{i:03d}.json'] for i in range(batch*46,(batch+1)*46)}
        if seal['batch']!=batch or seal['status']!='ACQUIRED' or seal['attempt_sha256']!=expected:raise ValueError('batch seal coverage')
    return attempts


def integrate(pilot,expansion,output):
    output.mkdir(exist_ok=False)
    preflight=json.loads((expansion/'preflight.json').read_text());plan=json.loads((expansion/'scenes.json').read_text());validate_plan(plan)
    if sha(expansion/'scenes.json')!=preflight['plan_sha256']:raise ValueError('plan identity drift')
    for name,digest in preflight['pilot_manifests'].items():
        if sha(pilot/name)!=digest:raise ValueError('pilot provenance drift')
    acquisition=json.loads((pilot/'acquisition.json').read_text());original=json.loads((pilot/'scenes.json').read_text())['scenes']
    check_membership(original,plan['scenes'])
    verify_batch_seals(expansion,plan)
    attempts,_=verify_completed(expansion,preflight,plan);require_complete(plan,attempts)
    specs={s['scene_id']:s for s in original+plan['scenes']}
    seen={r['sha256']['capture.json.rgb8'] for r in acquisition['scenes']}
    if len(seen)!=48:raise ValueError('duplicate original pilot RGB')
    records=[];measurements=[]
    for row in acquisition['scenes']:
        records.append(dict(scene_id=row['scene_id'],seed=row['seed'],split=row['split'],origin='pilot_001',
                            folder=str(pilot/'dataset'/row['scene_id']),sha256=row['sha256']))
    for row in attempts:
        spec=specs[row['scene_id']];folder=expansion/'dataset'/row['scene_id']
        if json.loads((folder/'scene.json').read_text())!=spec:raise ValueError('changed authored scene')
        if row['binary_sha256']!=preflight['binary_sha256'] or row['plan_sha256']!=preflight['plan_sha256']:raise ValueError('per-scene provenance')
        stats=validate_sample(folder,spec)
        if stats!=row['labels']:raise ValueError('retained label statistics mismatch')
        register_rgb(row['sha256']['capture.json.rgb8'],seen)
        records.append(dict(scene_id=row['scene_id'],seed=row['seed'],split='train',origin='expansion_002',folder=str(folder),sha256=row['sha256']))
        ids=np.fromfile(folder/'capture.json.native512.ids.u32',dtype='<u4').reshape(512,512)
        for instance in stats['instances']:
            obj=next(o for o in spec['objects'] if o['name']==instance['model'])
            instance.update(occlusion_stat(obj,ids,instance['renderer_id'],CAMERA))
        measurements.append(dict(scene_id=row['scene_id'],arrangement=spec['arrangement'],contrast=spec['contrast'],
            ambient=spec['ambient'],light_direction=spec['light_direction'],
            authored_colours={o['name']:o['colour'] for o in spec['objects']},labels=stats))
    for name in ('capture.json.rgb8','capture.json.native512.rgb8'):
        if len({r['sha256'][name] for r in records})!=324:raise ValueError('duplicate RGB raster: '+name)
    # Original train labels are verified independently without opening evaluation pixels.
    for _,folder,report,ids in pilot_rows(pilot,'train'):
        instances=[]
        for row in report['mapping']:
            if row['class_id']==1:instances.append(dict(model=row['model'],visible_pixels=int((ids==row['renderer_id']).sum())))
        if specs[folder.name]['negative']!=(not bool(instances)):raise ValueError('original train negative mismatch')
    manifest=output/'dataset.json'
    write_json(manifest,dict(schema='epd_v2_dataset/2',records=records,counts={'train':300,'validation':12,'test':12},
        pilot_root=str(pilot),expansion_root=str(expansion),pilot_manifest_sha256=preflight['pilot_manifests'],
        expansion_plan_sha256=sha(expansion/'scenes.json'),expansion_acquisition_sha256=sha(expansion/'acquisition.json'),
        preprocessing=plan['preprocessing'],camera=CAMERA,class_mapping=plan['class_mapping'],
        validation_heldout_files_opened=0,production_qualified=False))
    if len(dataset.enumerate_records(manifest,'train'))!=300:raise ValueError('training enumeration')
    areas=[m['visible_pixels'] for s in measurements for m in s['labels']['instances'] if m['visible_pixels']]
    fractions=[m['estimated_occluded_fraction'] for s in measurements for m in s['labels']['instances']]
    cubes=[o for spec in plan['scenes'] for o in spec['objects'] if o['class_id']==1]
    poses=np.array([o['pose'] for o in cubes]);colours=np.array([o['colour'] for o in cubes]);lights=np.array([s['light_direction'] for s in plan['scenes']])
    stat=dict(schema='epd_v2_expansion_quality/1',successful=276,negative=46,
        authored_object_count_distribution=dict(Counter(s['labels']['authored_objects'] for s in measurements)),
        visible_object_count_distribution=dict(Counter(s['labels']['visible_objects'] for s in measurements)),
        arrangements=dict(Counter(s['arrangement'] for s in measurements)),contrast_regimes=dict(Counter(s['contrast'] for s in measurements)),
        visible_area_pixels=dict(zip(('min','p25','median','p75','max'),np.percentile(areas,[0,25,50,75,100]).tolist())),
        cube_edge_m_range=[min(o['size'][0] for o in cubes),max(o['size'][0] for o in cubes)],
        cube_pose_component_ranges=np.stack((poses.min(0),poses.max(0))).tolist(),
        cube_colour_component_ranges=np.stack((colours.min(0),colours.max(0))).tolist(),
        light_direction_component_ranges=np.stack((lights.min(0),lights.max(0))).tolist(),
        background_clutter_count_distribution=dict(Counter(sum(o['name'].startswith('clutter_') for o in spec['objects']) for spec in plan['scenes'])),
        ambient_range=[min(s['ambient'] for s in measurements),max(s['ambient'] for s in measurements)],
        occlusion_estimate_counts=dict(clear_under_5_percent=sum(f<.05 for f in fractions),
            moderate_5_to_30_percent=sum(.05<=f<.30 for f in fractions),heavy_at_least_30_percent=sum(f>=.30 for f in fractions)),
        occlusion_method='native visible area relative to camera-projected authored BOX convex hull (raster estimate, not amodal ground truth); no labels modified',
        validation_test_pixels_opened=0,duplicate_rgb=0,duplicate_scene_definitions=0,unmapped_ids=0,
        per_scene=measurements)
    write_json(output/'quality.json',stat)
    write_json(output/'manifest_identity.json',dict(path=str(manifest),sha256=sha(manifest),bytes=manifest.stat().st_size))
    return manifest


def loader_smoke(manifest,output):
    import torch
    torch.set_num_threads(2);start=time.monotonic();consumed=[]
    expected=dataset.enumerate_records(manifest,'train')
    forbidden=[Path(r['folder']).resolve() for split in ('validation','test') for r in dataset.enumerate_records(manifest,split)]
    opened=set();training_folders={Path(r['folder']).resolve() for r in expected}
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if any(path==folder or folder in path.parents for folder in forbidden):raise RuntimeError('evaluation split file access prohibited')
        if path.parent in training_folders:opened.add(str(path.parent))
    import os
    sys.addaudithook(guard)
    class SupervisedRecords(torch.utils.data.IterableDataset):
        def __iter__(self):
            for row,folder,report,ids in dataset.verified_rows(manifest,'train'):
                image,target=tensors(folder,report,ids,torch)
                visible=[r['renderer_id'] for r in report['mapping'] if r['class_id']==1 and (ids==r['renderer_id']).any()]
                if len(visible)!=len(target['labels']) or any(not np.array_equal(target['masks'][i].numpy(),ids==identity) for i,identity in enumerate(visible)):
                    raise ValueError('supervisory mask/native ID mismatch')
                yield row['scene_id'],image,target
    loader=torch.utils.data.DataLoader(SupervisedRecords(),batch_size=None,num_workers=0)
    for name,image,target in loader:
        validate_tensors(image,target,torch)
        consumed.append(dict(scene_id=name,visible_objects=len(target['labels']),
            image_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
            target_sha256={key:hashlib.sha256(value.numpy().tobytes()).hexdigest() for key,value in target.items()}))
    if len(opened)!=300:raise ValueError('training sample access trace incomplete')
    if [r['scene_id'] for r in consumed]!=[r['scene_id'] for r in expected]:raise ValueError('loader membership drift')
    write_json(output,dict(status='PASS',samples=len(consumed),visible_instances=sum(r['visible_objects'] for r in consumed),
        negative_samples=sum(r['visible_objects']==0 for r in consumed),wall_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,manifest_sha256=sha(manifest),
        model_created=False,inference_calls=0,optimizer_steps=0,validation_test_pixels_opened=0,
        scene_directories_opened=len(opened),evaluation_access_guard='Python open audit hook rejects every validation/test scene directory',records=consumed))
    print(json.dumps(dict(samples=len(consumed),optimizer_steps=0,status='PASS')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['integrate','smoke'])
    parser.add_argument('--pilot',type=Path);parser.add_argument('--expansion',type=Path)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.action=='integrate':print(integrate(args.pilot.resolve(),args.expansion.resolve(),args.output.resolve()))
    else:loader_smoke(args.manifest.resolve(),args.output.resolve())
