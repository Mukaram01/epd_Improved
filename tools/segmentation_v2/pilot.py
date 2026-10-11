#!/usr/bin/env python3
"""Finite EPD acquisition, frozen scene splits and native-label verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EMD_COMMIT = 'e25d03831292098543c2ed3467d50c65ce000762'
HISTORICAL_RGB = {'92f1ba8ad58b2437fa5ccf55e73d53cc9dcba469b04efefbe624ce1505585793',
                  '920560881e06ae7285a33fd79d93e906a4001aac33f49583cf4af07016d1f526'}
HEADERS = ['fragment_mrt.hh', 'paired_native/native_pass.hh',
           'physics_owner/identity_inventory.hh']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, sort_keys=True); f.write('\n')


def bounded(command, folder, timeout, env=None, memory_mib=8192):
    """Kill the whole disposable process group; retain logs even on timeout."""
    start = time.monotonic()
    with (folder / 'stdout.log').open('xb') as out, (folder / 'stderr.log').open('xb') as err:
        p = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
        peak = 0; reason = None
        while True:
            try: code = p.wait(timeout=.25); break
            except subprocess.TimeoutExpired:
                try:
                    status = Path(f'/proc/{p.pid}/status').read_text()
                    rss = int(next(line.split()[1] for line in status.splitlines() if line.startswith('VmRSS:')))
                    peak = max(peak, rss)
                except (OSError, StopIteration): pass
                if time.monotonic()-start>timeout: reason = 'wall_timeout'
                elif peak>memory_mib*1024: reason = 'memory_budget'
                if reason:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(); code = 124 if reason=='wall_timeout' else 137
                    break
    return dict(command=command, exit_code=code, wall_seconds=time.monotonic()-start,
                sampled_process_peak_rss_kib=peak, resource_limit_reason=reason,
                child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def scene_plan():
    scenes = []
    for i in range(48):
        seed = 26011000 + i
        rng = random.Random(seed)
        split = 'train' if i < 24 else 'validation' if i < 36 else 'test'
        negative = i % 6 == 0
        colour = lambda: [round(rng.uniform(.12, .95), 6) for _ in range(3)]
        objects = [dict(name='background_table', class_id=0, size=[1.2, 1.2, .02],
                        pose=[0, 0, -.015, 0, 0, 0], colour=colour()),
                   dict(name='background_wall', class_id=0, size=[1.2, .02, .8],
                        pose=[0, .35, .25, 0, 0, 0], colour=colour())]
        for j in range(0 if negative else 1 + i % 3):
            size = rng.uniform(.025, .075)
            x = rng.uniform(-.11, .11); y = rng.uniform(-.06, .10)
            if i % 4 == 1: x, y = .012*j, .045*j  # depth-tested partial overlap
            objects.append(dict(name=f'cube_{j}', class_id=1, size=[size]*3,
                                pose=[x, y, size*.7, rng.uniform(-.4,.4), rng.uniform(-.4,.4),
                                      rng.uniform(-3.14,3.14)], colour=colour()))
        scenes.append(dict(scene_id=f'scene_{i:03d}', seed=seed, split=split,
                           negative=negative, objects=objects,
                           ambient=round(rng.uniform(.18, .65), 6),
                           light_direction=[rng.uniform(-1,1), rng.uniform(-1,0), rng.uniform(.3,1)]))
    return dict(schema='epd_v2_scenes/1', emd_baseline=EMD_COMMIT, scenes=scenes,
                historical_excluded=True, source_resolution=256, native_label_resolution=512,
                class_mapping={'0':'__background__', '1':'cube'},
                preprocessing='RGB256 INTER_LINEAR to512 RGB float32 CHW /255; no padding',
                confidence_threshold=.8, mask_threshold_strict_gt=.5)


def world_sdf(spec):
    root = ET.Element('sdf', version='1.8'); world = ET.SubElement(root, 'world', name='epd_pilot')
    physics = ET.SubElement(world, 'physics', name='fixed', type='ignored')
    ET.SubElement(physics, 'max_step_size').text = '.001'
    scene = ET.SubElement(world, 'scene'); ET.SubElement(scene, 'grid').text = 'false'
    for obj in spec['objects']:
        model = ET.SubElement(world, 'model', name=obj['name'])
        ET.SubElement(model, 'static').text = 'true'
        ET.SubElement(model, 'pose').text = ' '.join(map(str, obj['pose']))
        link = ET.SubElement(model, 'link', name='body')
        visual = ET.SubElement(link, 'visual', name='surface')
        box = ET.SubElement(ET.SubElement(visual, 'geometry'), 'box')
        ET.SubElement(box, 'size').text = ' '.join(map(str, obj['size']))
        mat = ET.SubElement(visual, 'material')
        for kind in ('ambient', 'diffuse'):
            ET.SubElement(mat, kind).text = ' '.join(map(str, obj['colour'] + [1]))
    return ET.tostring(root, encoding='unicode')


def validate_native(report, ids):
    for key, expected in dict(success=True, width=256, height=256, shared_scene_pass=True,
                              id_format='R32_UINT', depth_test=True, depth_write=True,
                              render_pass_count=1, frozen_equal=True).items():
        if report.get(key) != expected: raise ValueError(f'native contract: {key}')
    if report['acquisition']['step'] != 2 or report['acquisition']['stamp_ns'] != 2000000:
        raise ValueError('stale acquisition')
    mapping = report['mapping']; known = {r['renderer_id'] for r in mapping}
    if len(known) != len(mapping) or 0 in known or 4294967295 in known:
        raise ValueError('ambiguous renderer mapping')
    if not set(np.unique(ids)).issubset(known | {0}): raise ValueError('unmapped renderer ID')


def validate_split_membership(plan, acquisition):
    if plan != scene_plan(): raise ValueError('changed frozen scene plan')
    expected = {(s['scene_id'], s['seed'], s['split']) for s in plan['scenes']}
    actual = [(s['scene_id'], s['seed'], s['split']) for s in acquisition['scenes']]
    if len(actual)!=48 or len({s[0] for s in actual})!=48 or set(actual)!=expected:
        raise ValueError('duplicate scenes or changed frozen split membership')


def mask_metrics(prob, ids, identity, object_ids):
    pred = prob > .5; target = ids == identity
    background = ~np.isin(ids, list(object_ids))
    spill = int((pred & background).sum())
    cross = int((pred & ~target & ~background).sum())
    missing = int((target & ~pred).sum()); union = int((pred | target).sum())
    def boundary(a):
        return a & ~cv2.erode(a.astype('uint8'), np.ones((3,3), 'uint8'),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    p, t = boundary(pred), boundary(target)
    if p.any() and t.any():
        dp = cv2.distanceTransform((~p).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dt = cv2.distanceTransform((~t).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        mean = float((dt[p].mean() + dp[t].mean()) / 2)
        maximum = float(max(dt[p].max(), dp[t].max()))
    else: mean = maximum = None
    return dict(iou=float((pred & target).sum()/union) if union else 0,
                background_spill=spill, cross_object=cross, missing_visible=missing,
                boundary_mean_px=mean, boundary_max_px=maximum,
                strict_accept=bool(pred.any() and (pred & target).any() and spill==0 and cross==0))


def prepare(emd_repo, artifacts):
    artifacts.mkdir(parents=True, exist_ok=False)
    headers = artifacts/'headers'; hashes = {}
    for name in HEADERS:
        data = subprocess.run(['git','-C',str(emd_repo),'show',f'{EMD_COMMIT}:scripts/stage_a_rgbd/{name}'],
                              check=True, capture_output=True, timeout=10).stdout
        path = headers/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        hashes[name] = sha(path)
    write_json(artifacts/'source_manifest.json', dict(emd_commit=EMD_COMMIT, headers=hashes))
    snapshot = artifacts/'executed_sources'; snapshot.mkdir()
    for p in HERE.iterdir():
        if p.suffix in ('.py','.cpp','.txt','.json'): shutil.copyfile(p,snapshot/p.name)
    write_json(artifacts/'scenes.json', scene_plan())
    build = artifacts/'build'; build.mkdir()
    configure = artifacts/'configure'; configure.mkdir()
    r = bounded(['cmake','-S',str(HERE),'-B',str(build),f'-DEMD_HEADERS={headers}'], configure, 60)
    write_json(configure/'result.json', r)
    if r['exit_code']: raise RuntimeError('native configure failed; inspect logs')
    logs = artifacts/'compile'; logs.mkdir()
    r = bounded(['cmake','--build',str(build),'-j2'], logs, 180)
    write_json(logs/'result.json', r)
    if r['exit_code']: raise RuntimeError('native build failed; inspect logs')
    write_json(artifacts/'build_manifest.json', dict(binary_sha256=sha(build/'epd_v2_capture'),
               scenes_sha256=sha(artifacts/'scenes.json'),
               sources={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.cpp','.txt')}))


def acquire(artifacts):
    build = json.loads((artifacts/'build_manifest.json').read_text())
    binary = artifacts/'build/epd_v2_capture'
    if sha(binary)!=build['binary_sha256'] or sha(artifacts/'scenes.json')!=build['scenes_sha256']:
        raise ValueError('changed frozen build or scenes')
    plan = json.loads((artifacts/'scenes.json').read_text())
    if plan != scene_plan(): raise ValueError('unexpected scene plan')
    root = artifacts/'dataset'; root.mkdir()  # exclusive; failed attempts cannot be overwritten
    summary = dict(schema='epd_v2_acquisition/1', status='RUNNING', scenes=[],
                   frozen_plan_sha256=build['scenes_sha256'], binary_sha256=build['binary_sha256'])
    start = time.monotonic()
    try:
        for spec in plan['scenes']:
            if time.monotonic()-start>900: raise RuntimeError('total acquisition budget exceeded')
            if shutil.disk_usage(artifacts).free < 3*1024**3: raise RuntimeError('disk reserve <3 GiB')
            folder = root/spec['scene_id']; folder.mkdir()
            write_json(folder/'scene.json', spec)
            (folder/'world.sdf').write_text(world_sdf(spec))
            env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE='1', OMP_NUM_THREADS='2',
                       IGN_PARTITION=spec['scene_id']+'_v2', GZ_PARTITION=spec['scene_id']+'_v2')
            result = bounded([str(binary),str(folder/'world.sdf'),str(folder/'scene.json'),
                              str(folder/'capture.json')], folder, 30, env)
            write_json(folder/'process.json', result)
            if result['exit_code']: raise RuntimeError(f"{spec['scene_id']}: native exit {result['exit_code']}")
            report = json.loads((folder/'capture.json').read_text())
            ids = np.fromfile(folder/'capture.json.ids.u32', dtype='<u4').reshape(256,256)
            ids512 = np.fromfile(folder/'capture.json.native512.ids.u32', dtype='<u4').reshape(512,512)
            validate_native(report, ids); validate_native(report, ids512)
            p512 = report['native512']
            if any(p512.get(k)!=v for k,v in dict(width=512,height=512,shared_scene_pass=True,
                       id_format='R32_UINT',render_pass_count=1,depth_test=True,depth_write=True).items()):
                raise ValueError('invalid native512 pass')
            rgb = np.fromfile(folder/'capture.json.rgb8', dtype='uint8').reshape(256,256,3)
            if sha(folder/'capture.json.rgb8') in HISTORICAL_RGB: raise ValueError('historical RGB excluded')
            if not np.ptp(rgb): raise ValueError('blank rendered RGB')
            mapping = report['mapping']
            if {r['model'] for r in mapping}!={o['name'] for o in spec['objects']}:
                raise ValueError('incomplete authored/native mapping')
            visible = [r['renderer_id'] for r in mapping if r['class_id']==1 and (ids512==r['renderer_id']).any()]
            if not spec['negative'] and not visible: raise ValueError('all target objects invisible')
            files = {p.name:sha(p) for p in folder.iterdir() if p.is_file()}
            summary['scenes'].append(dict(scene_id=spec['scene_id'], split=spec['split'], seed=spec['seed'],
                                          visible_ids=visible, sha256=files, runtime=result))
        summary['status']='ACQUIRED'
    except Exception as e:
        summary['status']='BLOCKED'; summary['failure']=str(e)
        raise
    finally:
        summary['wall_seconds']=time.monotonic()-start
        write_json(artifacts/'acquisition.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','acquire']); p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--emd-repo',type=Path)
    a = p.parse_args()
    if a.action=='prepare':
        if not a.emd_repo: p.error('--emd-repo required')
        prepare(a.emd_repo, a.artifacts.resolve())
    else: print(json.dumps(acquire(a.artifacts.resolve()), indent=2))
