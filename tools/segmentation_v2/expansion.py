#!/usr/bin/env python3
"""Training-only native acquisition. No model, inference or evaluation entry point."""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pilot import (HERE, HEADERS, HISTORICAL_RGB, scene_plan, sha, write_json,
                   world_sdf, validate_native, bounded)
from train import verified_rows as pilot_rows
from campaign import validate_identity_mapping

CAMERA=dict(position=[0,-.40,.28],look_at=[0,0,.025],up=[0,0,1],
            near=.01,far=2,aspect=1,fovy_radians=1.0471975511965976)
BUDGET=dict(scene_seconds=30,batch_seconds=600,campaign_seconds=4000,
            memory_mib=2048,disk_reserve_gib=5,batches=6,scenes_per_batch=46)


def expansion_plan():
    scenes=[];positive=0
    for i in range(276):
        seed=260111000+i;rng=random.Random(seed);negative=i%6==0
        colour=lambda:[round(rng.uniform(.08,.98),6) for _ in range(3)]
        background=colour();contrast=('low','high','mixed')[i%3]
        objects=[dict(name='background_table',class_id=0,size=[1.2,1.2,.02],
                      pose=[0,0,-.015,0,0,0],colour=background),
                 dict(name='background_wall',class_id=0,size=[1.2,.02,.8],
                      pose=[0,.35,.25,0,0,0],colour=colour())]
        n=0 if negative else 1+positive%3
        arrangement='negative' if negative else ('separated','touching','partial_occlusion')[(positive//3)%3]
        sizes=[rng.uniform(.020,.085) for _ in range(n)]
        cx=rng.uniform(-.075,.075);cy=rng.uniform(-.04,.09)
        for j,size in enumerate(sizes):
            roll,pitch,yaw=rng.uniform(-.5,.5),rng.uniform(-.5,.5),rng.uniform(-math.pi,math.pi)
            if arrangement=='separated':
                x=cx+(j-(n-1)/2)*.11;y=cy+rng.uniform(-.025,.025)
            elif arrangement=='touching':
                x=cx+sum(sizes[:j])+(size-sum(sizes))/2;y=cy
                roll=pitch=yaw=0.
            else:
                x=cx+(j-(n-1)/2)*rng.uniform(.010,.025);y=cy+j*.045
            cube_colour=[round(min(.99,max(.02,c+rng.uniform(-.035,.035))),6) for c in background] if contrast=='low' else colour()
            if contrast=='high':cube_colour=[round(1-c,6) for c in background]
            objects.append(dict(name=f'cube_{j}',class_id=1,size=[size]*3,
                                pose=[x,y,size*(.5 if arrangement=='touching' else .8),roll,pitch,yaw],colour=cube_colour))
        # Class-zero rigid clutter is rendered and labelled as background, never a cube target.
        clutter_count=1+i%3 if negative else int(i%4==1)
        for j in range(clutter_count):
            objects.append(dict(name=f'clutter_{j}',class_id=0,
                size=[rng.uniform(.007,.018),rng.uniform(.015,.04),rng.uniform(.035,.11)],
                pose=[rng.uniform(-.18,.18),rng.uniform(-.075,.12),.04,0,0,rng.uniform(-math.pi,math.pi)],colour=colour()))
        scenes.append(dict(scene_id=f'expansion_{i:03d}',seed=seed,split='train',batch=i//46,
            negative=negative,arrangement=arrangement,contrast=contrast,objects=objects,
            ambient=round(rng.uniform(.12,.82),6),light_direction=[rng.uniform(-1,1),rng.uniform(-1,.3),rng.uniform(.2,1)]))
        positive+=int(not negative)
    return dict(schema='epd_v2_expansion/1',scenes=scenes,camera=CAMERA,budgets=BUDGET,
                preprocessing=scene_plan()['preprocessing'],source_resolution=256,native_label_resolution=512,
                class_mapping={'0':'__background__','1':'cube'},rgb_profile='EPD_directional_diffuse_MRT_v1_no_shadows_no_MSAA',
                collisions='static matching BOX collision per visual; inventory verified; physics not executed',
                baseline_commit='34b78f6fe1c48e6e26185db7f1455cebf045b23d',historical_excluded=True)


def definition_sha(spec):
    return __import__('hashlib').sha256(json.dumps({k:spec[k] for k in ('objects','ambient','light_direction')},
                                               sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_plan(plan):
    scenes=plan['scenes'];pilot=scene_plan()['scenes']
    for field in ('scene_id','seed'):
        vals=[s[field] for s in scenes]
        if len(set(vals))!=276 or set(vals)&{s[field] for s in pilot}:raise ValueError('duplicate scene ID/seed')
    definitions=[definition_sha(s) for s in scenes]
    if len(set(definitions))!=276 or set(definitions)&{definition_sha(s) for s in pilot}:raise ValueError('duplicate definition')
    if plan!=expansion_plan():raise ValueError('changed frozen expansion')


def register_rgb(digest,seen):
    if digest in HISTORICAL_RGB or digest in seen:raise ValueError('duplicate or historical RGB')
    seen.add(digest)


def label_integrity(spec,report,ids):
    if ids.shape!=(512,512) or ids.dtype!=np.dtype('<u4'):raise ValueError('native label shape/dtype')
    inventory=report['inventory'];without=copy.deepcopy(report);without['inventory']['collisions']=[]
    validate_identity_mapping(spec,without)
    collisions=inventory['collisions'];mapping=report['mapping']
    if len(collisions)!=len(mapping) or len({c['id'] for c in collisions})!=len(collisions):raise ValueError('collision count/identity')
    owners=[]
    for row in mapping:
        matches=[c for c in collisions if c['parent']==row['link_entity']]
        obj=next(o for o in spec['objects'] if o['name']==row['model'])
        if len(matches)!=1 or matches[0]['geometry']['type']!='BOX' or not np.allclose(
                matches[0]['geometry']['size'],obj['size'],rtol=1e-6,atol=1e-9):raise ValueError('collision ownership/geometry')
        owners.append(dict(model=row['model'],renderer_id=row['renderer_id'],visual_entity=row['visual_entity'],
                          collision_entity=matches[0]['id'],link_entity=row['link_entity'],model_entity=row['model_entity']))
    if not set(np.unique(ids)).issubset({0}|{r['renderer_id'] for r in mapping}):raise ValueError('unmapped labels')
    targets=[r for r in mapping if r['class_id']==1];instances=[]
    for row in targets:
        y,x=np.nonzero(ids==row['renderer_id'])
        instances.append(dict(model=row['model'],renderer_id=row['renderer_id'],visible_pixels=len(x),
            box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None))
    visible=sum(r['visible_pixels']>0 for r in instances)
    if spec['negative']!=not_targets(targets) or (not spec['negative'] and not visible):raise ValueError('negative/visible target labels')
    return dict(authored_objects=len(targets),visible_objects=visible,instances=instances,ownership=owners,
                background_pixels=int(np.isin(ids,[0]+[r['renderer_id'] for r in mapping if r['class_id']==0]).sum()))


def not_targets(targets):
    return not bool(targets)


def progress_state(plan,attempts):
    expected=[s['scene_id'] for s in plan['scenes']];actual=[r['scene_id'] for r in attempts]
    if actual!=expected[:len(actual)] or len(set(actual))!=len(actual):raise ValueError('noncontiguous/duplicate attempts')
    if any(r['status'] not in ('ACQUIRED','FAILED') for r in attempts):raise ValueError('unknown attempt state')
    failed=[r['scene_id'] for r in attempts if r['status']=='FAILED']
    return dict(successful=sum(r['status']=='ACQUIRED' for r in attempts),failed=failed,
                complete=len(attempts)==276 and not failed,unattempted=expected[len(actual):])


def require_complete(plan,attempts):
    state=progress_state(plan,attempts)
    if not state['complete']:raise ValueError('incomplete expansion')
    return state


def expansion_world(spec):
    root=ET.fromstring(world_sdf(spec))
    for model in root.find('world').findall('model'):
        link=model.find('link');collision=ET.SubElement(link,'collision',name='solid')
        collision.append(copy.deepcopy(link.find('visual/geometry')))
    return ET.tostring(root,encoding='unicode')


def freeze(pilot,output,emd_repo):
    output.mkdir(parents=True,exist_ok=False)
    evidence=HERE.parent.parent/'evidence/segmentation_v2/pilot_001'
    for name in ('scenes.json','acquisition.json','build_manifest.json'):
        if sha(pilot/name)!=sha(evidence/name):raise ValueError('pilot manifest drift: '+name)
    # Verify original train only; validation/test identity and RGB hashes come from immutable manifests.
    original_train=list(pilot_rows(pilot,'train'))
    build=json.loads((pilot/'build_manifest.json').read_text());binary=pilot/'build/epd_v2_capture'
    if sha(binary)!=build['binary_sha256']:raise ValueError('capture binary drift')
    for name in ('capture.cpp','CMakeLists.txt'):
        if sha(HERE/name)!=build['sources'][name]:raise ValueError('native source drift')
    header_manifest=json.loads((pilot/'source_manifest.json').read_text())
    import hashlib
    for name in HEADERS:
        committed=subprocess.run(['git','-C',str(emd_repo),'show',
            f"{header_manifest['emd_commit']}:scripts/stage_a_rgbd/{name}"],
            check=True,capture_output=True,timeout=10).stdout
        if sha(pilot/'headers'/name)!=header_manifest['headers'][name] or hashlib.sha256(committed).hexdigest()!=header_manifest['headers'][name]:
            raise ValueError('pinned EMD header drift')
    plan=expansion_plan();validate_plan(plan);write_json(output/'scenes.json',plan)
    snapshot=output/'executed_sources';snapshot.mkdir()
    for name in ('expansion.py','dataset.py','dataset_bindings.py','integrate_dataset.py','pilot.py','train.py','campaign.py','capture.cpp','CMakeLists.txt'):
        shutil.copy2(HERE/name,snapshot/name)
    versions={n:importlib.metadata.version(n) for n in ('numpy','opencv-python','torch','torchvision')}
    pkg=subprocess.run(['pkg-config','--modversion','ignition-gazebo6','ignition-rendering6','OGRE-Next','jsoncpp','gl'],
                       check=True,capture_output=True,text=True,timeout=10)
    write_json(output/'preflight.json',dict(pilot_root=str(pilot),binary=str(binary),binary_sha256=sha(binary),
        plan_sha256=sha(output/'scenes.json'),pilot_manifests={n:sha(pilot/n) for n in ('scenes.json','acquisition.json','build_manifest.json','source_manifest.json')},
        headers=header_manifest,source_sha256={p.name:sha(p) for p in snapshot.iterdir()},
        train_samples_verified=len(original_train),memberships_manifest_only={'train':24,'validation':12,'test':12},
        validation_test_pixels_opened=0,versions=versions,native_pkg_versions=pkg.stdout.splitlines(),
        memory_available_kib=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))),
        free_disk_bytes=shutil.disk_usage(output).free,camera=CAMERA,budgets=BUDGET,
        camera_provenance='fixed camera in hash-verified capture.cpp; fovy/position/lookAt match CAMERA',
        rgb_profile=plan['rgb_profile']))
    (output/'attempts').mkdir();(output/'dataset').mkdir();(output/'batches').mkdir()
    print(json.dumps(dict(frozen=276,negative=46,manifest_sha256=sha(output/'scenes.json'))),flush=True)


def validate_sample(folder,spec):
    report=json.loads((folder/'capture.json').read_text())
    for name,count in (('capture.json.rgb8',256*256*3),('capture.json.native512.rgb8',512*512*3),
                       ('capture.json.ids.u32',256*256*4),('capture.json.native512.ids.u32',512*512*4)):
        if (folder/name).stat().st_size!=count:raise ValueError('raster byte length: '+name)
    ids256=np.fromfile(folder/'capture.json.ids.u32',dtype='<u4').reshape(256,256)
    ids=np.fromfile(folder/'capture.json.native512.ids.u32',dtype='<u4').reshape(512,512)
    validate_native(report,ids256);validate_native(report,ids)
    native=report['native512']
    if any(native.get(k)!=v for k,v in dict(width=512,height=512,shared_scene_pass=True,
        id_format='R32_UINT',render_pass_count=1,depth_test=True,depth_write=True,samples=1).items()):raise ValueError('native512 contract')
    for key,size in (('capture.json.rgb8',256),('capture.json.native512.rgb8',512)):
        rgb=np.fromfile(folder/key,dtype='u1').reshape(size,size,3)
        if not np.ptp(rgb):raise ValueError('blank RGB')
    return label_integrity(spec,report,ids)


def load_attempts(root):
    return [json.loads(p.read_text()) for p in sorted((root/'attempts').glob('*.json'))]


def verify_completed(root,preflight,plan):
    attempts=load_attempts(root);state=progress_state(plan,attempts)
    for attempt in attempts:
        folder=root/'dataset'/attempt['scene_id']
        for name,digest in attempt['sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('completed artifact drift')
    seals=sorted((root/'batches').glob('*.json'))
    if [p.name for p in seals]!=[f'batch_{i:02d}.json' for i in range(len(attempts)//46)]:raise ValueError('completed batch coverage')
    for batch,p in enumerate(seals):
        seal=json.loads(p.read_text());rows=attempts[batch*46:(batch+1)*46]
        expected={f'{i:03d}.json':sha(root/'attempts'/f'{i:03d}.json') for i in range(batch*46,(batch+1)*46)}
        if seal['batch']!=batch or seal['status']!='ACQUIRED' or seal['attempt_sha256']!=expected or any(r['status']!='ACQUIRED' for r in rows):
            raise ValueError('invalid completed batch seal')
    if (root/'acquisition.json').exists():
        acquisition=json.loads((root/'acquisition.json').read_text())
        expected={f'{i:03d}.json':sha(root/'attempts'/f'{i:03d}.json') for i in range(len(attempts))}
        if acquisition['attempt_manifest_sha256']!=expected or (acquisition['status']=='ACQUIRED')!=state['complete']:
            raise ValueError('acquisition/attempt seal mismatch')
    if state['failed']:raise ValueError('failed sample retained; automatic resume prohibited')
    return attempts,state


def acquire(root):
    preflight=json.loads((root/'preflight.json').read_text());plan=json.loads((root/'scenes.json').read_text())
    validate_plan(plan)
    if sha(root/'scenes.json')!=preflight['plan_sha256'] or sha(preflight['binary'])!=preflight['binary_sha256']:raise ValueError('frozen source/plan drift')
    for name,digest in preflight['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('capture source changed after freeze: '+name)
    attempts,state=verify_completed(root,preflight,plan)
    if attempts:raise ValueError('existing attempts: explicit reviewed continuation required; no automatic retry')
    pilot=Path(preflight['pilot_root']);prior=json.loads((pilot/'acquisition.json').read_text())
    seen={r['sha256']['capture.json.rgb8'] for r in prior['scenes']}
    start=time.monotonic();batch_start=start
    try:
        for index,spec in enumerate(plan['scenes']):
            if index%46==0:batch_start=time.monotonic()
            if time.monotonic()-start>BUDGET['campaign_seconds'] or time.monotonic()-batch_start>BUDGET['batch_seconds']:raise RuntimeError('capture time budget')
            if shutil.disk_usage(root).free<BUDGET['disk_reserve_gib']*1024**3:raise RuntimeError('free disk reserve')
            folder=root/'dataset'/spec['scene_id'];folder.mkdir()
            row=dict(scene_id=spec['scene_id'],seed=spec['seed'],split='train',batch=spec['batch'],status='FAILED')
            try:
                write_json(folder/'scene.json',spec);(folder/'world.sdf').write_text(expansion_world(spec))
                env=dict(os.environ,LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='2',
                         IGN_PARTITION=spec['scene_id']+'_v2',GZ_PARTITION=spec['scene_id']+'_v2')
                runtime=bounded([preflight['binary'],str(folder/'world.sdf'),str(folder/'scene.json'),str(folder/'capture.json')],
                                folder,BUDGET['scene_seconds'],env,memory_mib=BUDGET['memory_mib'])
                write_json(folder/'process.json',runtime);row['runtime']=runtime
                if runtime['exit_code']:raise RuntimeError('native exit '+str(runtime['exit_code']))
                row['labels']=validate_sample(folder,spec)
                register_rgb(sha(folder/'capture.json.rgb8'),seen)
                row['status']='ACQUIRED'
            except Exception as error:
                row['failure']=str(error);raise
            finally:
                row['sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
                row['binary_sha256']=preflight['binary_sha256'];row['plan_sha256']=preflight['plan_sha256']
                write_json(root/'attempts'/f'{index:03d}.json',row);attempts.append(row)
                write_json(root/f'progress_{index+1:03d}.json',progress_state(plan,attempts))
            if (index+1)%46==0:
                batch=index//46;write_json(root/'batches'/f'batch_{batch:02d}.json',dict(batch=batch,status='ACQUIRED',
                    wall_seconds=time.monotonic()-batch_start,attempt_sha256={f'{j:03d}.json':sha(root/'attempts'/f'{j:03d}.json') for j in range(batch*46,(batch+1)*46)}))
                print(json.dumps(dict(batch=batch,successful=len(attempts),wall_seconds=time.monotonic()-start)),flush=True)
        require_complete(plan,attempts)
    finally:
        state=progress_state(plan,attempts)
        write_json(root/'acquisition.json',dict(schema='epd_v2_expansion_acquisition/1',status='ACQUIRED' if state['complete'] else 'BLOCKED',
            **state,wall_seconds=time.monotonic()-start,plan_sha256=preflight['plan_sha256'],
            attempt_manifest_sha256={p.name:sha(p) for p in sorted((root/'attempts').glob('*.json'))}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','acquire','verify'])
    parser.add_argument('--emd-repo',type=Path);parser.add_argument('--pilot',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.action=='freeze':freeze(args.pilot.resolve(),args.output.resolve(),args.emd_repo.resolve())
    elif args.action=='acquire':acquire(args.output.resolve())
    else:
        root=args.output.resolve();plan=json.loads((root/'scenes.json').read_text());validate_plan(plan)
        preflight=json.loads((root/'preflight.json').read_text())
        if sha(root/'scenes.json')!=preflight['plan_sha256']:raise ValueError('changed frozen plan')
        _,state=verify_completed(root,preflight,plan)
        print(json.dumps(state,sort_keys=True))
