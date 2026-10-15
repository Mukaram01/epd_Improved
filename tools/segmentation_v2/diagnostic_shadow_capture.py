#!/usr/bin/env python3
"""One separately reviewed diagnostic-only frozen-logit experiment. No training.
Original strict admission and original missing supervision are never changed.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
import shadow_mask_paste as s

HERE=Path(__file__).resolve().parent
SCHEMA='epd_diagnostic_only_qualification/v1'
PURPOSE='ONE_FOUR_SCENE_FROZEN_LOGIT_SHADOW_PASTE_EXPERIMENT_ONLY'
LIMITS={'cpu_threads':2,'max_scenes':4,'wall_seconds':300,'rss_mib':6144}
POLICY_SHA='dada1f4af5c18e3fa7ce7f214c6e0adb4e2c2a24403dd9b49980d63554890d75'
RUN_PATH=Path('/home/user/epd-v2-artifacts/mask-paste-shadow-experiment-001')


def durable_json(path,value):
    path=Path(path)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_npz(path,values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**values);stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def validate_scope(p):
    expected={'schema':SCHEMA,'purpose':PURPOSE,'implementer':'/root',
      'checkpoint_sha256':s.CHECKPOINT_SHA,'dataset_sha256':s.DATASET_SHA,
      'scene_ids':list(s.SCENES),'controls':list(s.CONTROLS),'limits':LIMITS,
      'original_training':{'parent_exit':'UNKNOWN','parent_wall':'UNKNOWN','resource_enforcement':'UNKNOWN'},
      'production_qualified':False,'stage_a2':'BLOCKED_0/2'}
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('DIAGNOSTIC_SCOPE_DRIFT')


def validate_review(p,proposal_sha,review):
    if (not isinstance(review,dict) or review.get('decision')!='APPROVE_DIAGNOSTIC_ONLY' or
        not isinstance(review.get('reviewer'),str) or review['reviewer']==p['implementer'] or
        not review['reviewer'].startswith('/root/') or
        review.get('qualification_sha256')!=proposal_sha or
        review.get('sources_sha256')!=p['sources_sha256'] or review.get('evidence_sha256')!=p['evidence_sha256']):
        raise ValueError('INDEPENDENT_REVIEW_REQUIRED_OR_DRIFT')


def admit(qualification,review_path,review_pin):
    p=json.loads(Path(qualification).read_text());validate_scope(p)
    if (p.get('output_path')!=str(RUN_PATH) or p.get('preregistration_sha256')!=POLICY_SHA):
        raise ValueError('FIXED_EXPERIMENT_SCOPE_DRIFT')
    if s.digest(review_path)!=review_pin:raise ValueError('REVIEW_SHA_DRIFT')
    review=json.loads(Path(review_path).read_text());validate_review(p,s.digest(qualification),review)
    for path,sha in {**p['sources_sha256'],**p['evidence_sha256']}.items():
        if s.digest(path)!=sha:raise ValueError('BOUND_INPUT_DRIFT: '+path)
    policy=json.loads((HERE/'shadow_mask_paste_preregistration.json').read_text())
    if s.digest(HERE/'shadow_mask_paste_preregistration.json')!=POLICY_SHA or s.source_inventory()!=policy['source_inventory']:
        raise ValueError('PREREGISTERED_SOURCE_DRIFT')
    # This is explicitly a separate disposition, never original gate success.
    if s.admission(policy)['status']!='BLOCKED_MISSING_TRAINING_PROVENANCE':
        raise ValueError('ORIGINAL_PROVENANCE_STATE_DRIFT_REQUIRES_REVIEW')
    if s.digest(policy['checkpoint']['path'])!=s.CHECKPOINT_SHA or s.digest(policy['dataset']['path'])!=s.DATASET_SHA:
        raise ValueError('CHECKPOINT_DATASET_DRIFT')
    rows=s.select_scenes(json.loads(Path(policy['dataset']['path']).read_text())['records'])
    if rows!=policy['scenes']:raise ValueError('SCENE_BINDING_DRIFT')
    if shutil.disk_usage(RUN_PATH.parent).free<6*1024**3:raise ValueError('DISK_RESERVE')
    return p,policy


def _parent_death_signal(parent):
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'PDEATHSIG_FAILED')
    if os.getppid()!=parent:os.kill(os.getpid(),signal.SIGKILL)


def supervise(command,output,wall_seconds=300,rss_kib=6144*1024):
    """Sample VmRSS/VmHWM at20ms; wait4 also rejects missed high-water breaches.
    PDEATHSIG kills worker on launcher death; finally kills whole worker group.
    No retry. Original campaign process.json is not read or written here.
    """
    start=time.monotonic();peak=0;reason=None;p=None;code=None;failure=None;launcher_pid=os.getpid()
    durable_json(output/'launch.json',{'command':command,'launcher_pid':os.getpid(),
                 'wall_limit_seconds':wall_seconds,'rss_limit_kib':rss_kib,'started_unix_ns':time.time_ns()})
    with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
        try:
            p=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True,
                preexec_fn=lambda:_parent_death_signal(launcher_pid),env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            durable_json(output/'worker_identity.json',{'pid':p.pid,'pgid':p.pid,'launcher_pid':os.getpid()})
            while True:
                try:
                    status=Path(f'/proc/{p.pid}/status').read_text()
                    for line in status.splitlines():
                        if line.startswith(('VmRSS:','VmHWM:')):peak=max(peak,int(line.split()[1]))
                except FileNotFoundError:pass
                pid,exit_status,usage=os.wait4(p.pid,os.WNOHANG)
                if pid:
                    code=os.waitstatus_to_exitcode(exit_status);p.returncode=code
                    peak=max(peak,int(usage.ru_maxrss));break
                if time.monotonic()-start>wall_seconds:reason='wall'
                elif peak>rss_kib:reason='memory'
                if reason:
                    os.killpg(p.pid,signal.SIGKILL)
                    _,exit_status,usage=os.wait4(p.pid,0)
                    code=os.waitstatus_to_exitcode(exit_status);p.returncode=code
                    peak=max(peak,int(usage.ru_maxrss));break
                time.sleep(.02)
            if peak>rss_kib:reason='memory'
            if time.monotonic()-start>wall_seconds and reason is None:reason='wall'
        except BaseException as exc:
            failure=type(exc).__name__+': '+str(exc)
        finally:
            if p is not None:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                if p.returncode is None:
                    _,exit_status,usage=os.wait4(p.pid,0);code=os.waitstatus_to_exitcode(exit_status);p.returncode=code
                    peak=max(peak,int(usage.ru_maxrss))
            stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
    result={'status':'BLOCKED_RESOURCE_LIMIT' if reason else ('BLOCKED_WORKER_FAILURE' if code!=0 or failure else 'WORKER_COMPLETED'),
        'exit_code':code,'resource_limit_reason':reason,'failure':failure,'wall_seconds':time.monotonic()-start,
        'peak_worker_rss_kib':peak,'command':command,'sampling_interval_seconds':.02,
        'rss_enforcement':'VmRSS/VmHWM polling plus final wait4 high-water check; not kernel cgroup memory.max',
        'pdeathsig':'SIGKILL','process_group_cleanup':True}
    durable_json(output/'supervision.json',result)
    return result


def validate_capture(b):
    n=len(b['scores'])
    spec={'logits':((n,2,56,56),np.float32),'boxes':((n,4),np.float32),
          'transformed_boxes':((n,4),np.float32),'labels':((n,),np.int64),
          'scores':((n,),np.float32),'masks':((n,1,512,512),np.float32)}
    for key,(shape,dtype) in spec.items():
        v=b[key]
        if v.shape!=shape or v.dtype!=dtype or not np.isfinite(v).all():raise ValueError('CAPTURE_CONTRACT: '+key)
    if (b['labels']!=1).any() or ((b['scores']<0)|(b['scores']>1)).any() or ((b['masks']<0)|(b['masks']>1)).any():
        raise ValueError('CAPTURE_IDENTITY_OR_PROBABILITY_DRIFT')
    if (b['boxes']<0).any() or (b['boxes']>512).any() or (b['boxes'][:,2:]<=b['boxes'][:,:2]).any():
        raise ValueError('CAPTURE_BOX_DRIFT')


def scoring_ids(raw,cubes):
    # Ground-truth class interpretation only. No prediction is altered.
    return np.where(np.isin(raw,list(cubes)),raw,0).astype(np.uint32)


def metrics(mask,ids,identity):
    if identity is not None and (type(identity) is not int or identity<=0):raise ValueError('IDENTITY_DRIFT')
    if identity is not None and (ids==identity).any():return s.pixel_metrics(mask,ids,identity)
    target=np.zeros_like(mask) if identity is None else ids==identity
    return {'iou':0. if identity is not None else None,'background_spill':int((mask&(ids==0)).sum()),
            'missing_foreground':int((target&~mask).sum()),'cross_object':int((mask&(ids!=0)&~target).sum()),
            'spill_distance_bins_le1_1to2_2to3_gt3':None,'spill_distance_max':None}


def fixed_matching(masks,keep,ids,targets):
    refs=[];edges=[];assignment={}
    for index in keep:
        candidates=[]
        for identity in sorted(targets):
            m=metrics(masks[index,0]>.5,ids,int(identity));candidates.append((m['iou'],int(identity)))
            edges.append((m['iou'],int(index),int(identity)))
        refs.append(max(candidates)[1] if candidates else None)
    used=set()
    for iou,index,identity in sorted(edges,reverse=True):
        if iou>0 and index not in used and identity not in assignment:
            assignment[identity]=index;used.add(index)
    return refs,assignment


def scene_inputs(row):
    from pilot import validate_native
    folder=Path(row['folder'])
    for name,sha in row['sha256'].items():
        if Path(name).name!=name or s.digest(folder/name)!=sha:raise ValueError('SCENE_INPUT_DRIFT: '+name)
    spec=json.loads((folder/'scene.json').read_text());process=json.loads((folder/'process.json').read_text())
    if any(spec[k]!=row[k] for k in ('scene_id','seed','split')) or spec['split']!='validation' or process['exit_code']!=0:
        raise ValueError('INVALID_SCENE_PROVENANCE')
    report=json.loads((folder/'capture.json').read_text())
    raw512=np.fromfile(folder/'capture.json.native512.ids.u32',dtype='<u4').reshape(512,512)
    raw256=np.fromfile(folder/'capture.json.ids.u32',dtype='<u4').reshape(256,256)
    validate_native(report,raw512);validate_native(report,raw256)
    cubes={int(r['renderer_id']) for r in report['mapping'] if r['class_id']==1}
    if bool(cubes)==spec['negative']:raise ValueError('NEGATIVE_SCENE_IDENTITY_DRIFT')
    return report,raw512,raw256,cubes


def state_sha(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def summarize_scene(b,raw512,raw256,cubes):
    import torch
    ids512=scoring_ids(raw512,cubes);ids256=scoring_ids(raw256,cubes)
    keep=np.flatnonzero(b['scores']>=.8)
    targets={int(t) for t in cubes if (ids512==t).any()}
    refs,assignment=fixed_matching(b['masks'],keep,ids512,targets)
    logits=torch.from_numpy(b['logits'])[torch.arange(len(b['labels'])),torch.from_numpy(b['labels'])][:,None]
    boxes=torch.from_numpy(b['boxes']);original=torch.from_numpy(b['masks'])
    parity=s.parity(original,s.reconstruct(logits,boxes,(512,512)))
    arms={}
    for control in ('baseline',*s.CONTROLS):
        masks=b['masks'] if control=='baseline' else s.reconstruct(logits,boxes,(512,512),control).numpy()
        detections=[]
        for j,index in enumerate(keep):
            prob=masks[index,0];old=b['masks'][index,0];box=b['boxes'][index];identity=refs[j]
            roi,mapped=s.project(prob,box);old_roi,old_map=s.project(old,box)
            k=roi.reshape(256,2,256,2).sum(axis=(1,3))
            yx=np.argwhere(mapped&(ids256==0));y=2*yx[:,0,None]+np.array([0,0,1,1]);x=2*yx[:,1,None]+np.array([0,1,0,1])
            support=ids512[y,x];active=roi[y,x];all_bg=(support==0).all(axis=1)
            has_target=(support==identity).any(axis=1) if identity else np.zeros(len(yx),bool)
            active_bg=((support==0)&active).any(axis=1)
            detections.append({'prediction_index':int(index),'reference_identity':identity,
                'native_full':metrics(prob>.5,ids512,identity),'native_roi':metrics(roi,ids512,identity),
                'mapped_union':metrics(mapped,ids256,identity),
                'native_changed_pixels':int(((prob>.5)!=(old>.5)).sum()),
                'roi_added_pixels':int((roi&~old_roi).sum()),'roi_removed_pixels':int((old_roi&~roi).sum()),
                'mapped_added_cells':int((mapped&~old_map).sum()),'mapped_removed_cells':int((old_map&~mapped).sum()),
                'max_abs_probability_change':float(np.abs(prob-old).max(initial=0)),
                'union_occupancy_1_to_4':[int((k==n).sum()) for n in range(1,5)],
                'union_inactive_slots':int((4-k[mapped]).sum()),
                'mapped_bg_partial_support':int((mapped&(ids256==0)&(k<4)).sum()),
                'mapped_bg_all_native_bg':int(all_bg.sum()),
                'mapped_bg_all_native_bg_partial':int((all_bg&(active.sum(axis=1)<4)).sum()),
                'mapped_bg_target_support_no_active_bg':int((has_target&~active_bg).sum()),
                'native_label_disagreement_under_roi':int(((ids512!=ids256.repeat(2,0).repeat(2,1))&roi).sum())})
        instances=[]
        for identity in sorted(targets):
            index=assignment.get(identity)
            full=masks[index,0]>.5 if index is not None else np.zeros((512,512),bool)
            roi,mapped=s.project(masks[index,0],b['boxes'][index]) if index is not None else (np.zeros((512,512),bool),np.zeros((256,256),bool))
            instances.append({'renderer_identity':identity,'prediction_index':index,
                'native_full':metrics(full,ids512,identity),'native_roi':metrics(roi,ids512,identity),
                'mapped_union':metrics(mapped,ids256,identity)})
        totals={}
        for raster in ('native_full','native_roi','mapped_union'):
            totals[raster]={
                'background_spill':sum(x[raster]['background_spill'] for x in detections),
                'cross_object':sum(x[raster]['cross_object'] for x in detections),
                'missing_foreground':sum(x[raster]['missing_foreground'] for x in instances),
                'mean_instance_iou':float(np.mean([x[raster]['iou'] for x in instances])) if instances else None}
        recovered=sum(x['native_full']['iou']>=.5 for x in instances)
        arms[control]={'detections':detections,'instances':instances,'totals':totals,
                       'recovered':recovered,'false_positives':len(keep)-recovered}
    return {'parity':parity,'candidate_detections':len(b['scores']),'confident_detections':len(keep),
            'negative':not targets,'targets':sorted(targets),'baseline_fixed_assignment':assignment,'arms':arms}


def worker(args):
    if os.getppid()!=args.launcher_pid or args.launcher_pid<=1:raise ValueError('LAUNCHER_IDENTITY_DRIFT')
    output=RUN_PATH
    if json.loads((output/'launch.json').read_text())['launcher_pid']!=os.getppid():raise ValueError('LAUNCH_LEDGER_DRIFT')
    p,policy=admit(args.qualification,args.review,args.review_sha256)
    durable_json(output/'diagnostic_qualification.json',{'decision':'DIAGNOSTIC_ONLY_ADMITTED',
        'qualification_sha256':s.digest(args.qualification),'review_sha256':args.review_sha256,
        'original_training':p['original_training'],'stage_a2':'BLOCKED_0/2','production_qualified':False})
    import training_provenance_audit as forensic
    integrity=forensic.audit(forensic.CAMPAIGN,output/'integrity')
    if integrity['frozen_saved_state']['frozen_tensors_equal']!=295 or integrity['history']['updates']!=300:
        raise ValueError('INDEPENDENT_EVIDENCE_DRIFT')
    import torch,cv2
    from recover_original import make_model
    from mask_resolution import configure56
    from campaign import validate_prediction
    from torchvision.models.detection.transform import resize_boxes
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(260110);np.random.seed(260110);cv2.setNumThreads(2)
    if torch.cuda.is_available():raise ValueError('CPU_ONLY_EXECUTION_REQUIRED')
    state=torch.load(policy['checkpoint']['path'],map_location='cpu',weights_only=True)
    model=configure56(make_model());model.load_state_dict(state['model'],strict=True);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if any(t.device.type!='cpu' for t in model.state_dict().values()):raise ValueError('MODEL_DEVICE_DRIFT')
    del state;gc.collect();weight_sha=state_sha(model)
    observed={}
    def logits_hook(module,inputs,result):observed.setdefault('logits',[]).append(result.detach().clone())
    def roi_hook(module,inputs,result):
        r=result[0][0]
        observed.setdefault('roi',[]).append({k:r[k].detach().clone() for k in ('boxes','labels','scores','masks')})
    def transform_hook(module,inputs,result):observed.setdefault('sizes',[]).append(result[0].image_sizes)
    hooks=[model.roi_heads.mask_predictor.register_forward_hook(logits_hook),
           model.roi_heads.register_forward_hook(roi_hook),model.transform.register_forward_hook(transform_hook)]
    receipts=[]
    try:
        for row in policy['scenes']:
            scene=row['scene_id'];start=time.monotonic();observed.clear()
            report,raw512,raw256,cubes=scene_inputs(row)
            rgb=np.fromfile(Path(row['folder'])/'capture.json.rgb8',np.uint8).reshape(256,256,3)
            image=torch.from_numpy(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR).transpose(2,0,1).copy()).float()/255
            with torch.inference_mode():prediction=model([image])[0]
            if any(len(observed.get(k,[]))!=1 for k in ('logits','roi','sizes')):raise ValueError('INCOMPLETE_HOOK_CAPTURE')
            logits=observed['logits'][0];roi=observed['roi'][0];size=observed['sizes'][0][0]
            values={k:v.detach().numpy().copy() for k,v in prediction.items()};validate_prediction(values)
            values.update(logits=logits.numpy().copy(),transformed_boxes=roi['boxes'].numpy().copy(),
                raw_ids512=raw512,raw_ids256=raw256,image_rgb256=rgb)
            validate_capture(values)
            capture=output/(scene+'_capture.npz');durable_npz(capture,values)  # preserve even on parity failure
            indices=torch.arange(len(prediction['labels']))
            selected=logits[indices,prediction['labels']][:,None]
            if (not torch.equal(logits.sigmoid()[indices,prediction['labels']][:,None],roi['masks']) or
                not torch.equal(roi['scores'],prediction['scores']) or not torch.equal(roi['labels'],prediction['labels']) or
                not torch.equal(resize_boxes(roi['boxes'],size,(512,512)),prediction['boxes'])):
                raise ValueError('HOOK_CLASS_BOX_TRANSFORM_IDENTITY_DRIFT')
            paste_parity=s.parity(prediction['masks'],s.reconstruct(selected,prediction['boxes'],(512,512)))
            archived_path=Path(policy['checkpoint']['path']).parent/'validation_01'/(scene+'.npz')
            with np.load(archived_path,allow_pickle=False) as archived:
                historic={k:np.array_equal(values[k],archived[k]) for k in ('boxes','labels','scores','masks')}
            receipt={'historic_validation_arrays_bitwise_equal':historic,'scene_id':scene,'capture_sha256':s.digest(capture),'parity':paste_parity,
                'original_size':[512,512],'transformed_size':list(size),'seconds':time.monotonic()-start,
                'candidate_detections':len(prediction['scores']),'confident_detections':int((prediction['scores']>=.8).sum()),
                'input_sha256':row['sha256'],'array_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in values.items()}}
            durable_json(output/(scene+'_capture.json'),receipt);receipts.append(receipt)
            print(json.dumps({'captured':scene,'seconds':receipt['seconds'],'parity':paste_parity}),flush=True)
            del prediction,values,logits,roi,selected,raw512,raw256,image;observed.clear();gc.collect()
    finally:
        for hook in hooks:hook.remove()
    if state_sha(model)!=weight_sha:raise ValueError('IMMUTABLE_WEIGHT_STATE_DRIFT')
    del model;gc.collect()
    if [r['scene_id'] for r in receipts]!=list(s.SCENES):raise ValueError('INCOMPLETE_CAPTURE')
    scenes=[]
    for row,receipt in zip(policy['scenes'],receipts):
        path=output/(row['scene_id']+'_capture.npz')
        if s.digest(path)!=receipt['capture_sha256']:raise ValueError('CAPTURE_ARCHIVE_DRIFT')
        with np.load(path,allow_pickle=False) as bundle:b={k:bundle[k] for k in bundle.files}
        validate_capture(b);report,raw512,raw256,cubes=scene_inputs(row)
        if not np.array_equal(raw512,b['raw_ids512']) or not np.array_equal(raw256,b['raw_ids256']):raise ValueError('RENDERER_IDENTITY_DRIFT')
        scene=summarize_scene(b,raw512,raw256,cubes);scene['scene_id']=row['scene_id'];scenes.append(scene)
        durable_json(output/(row['scene_id']+'_comparisons.json'),scene)
    # Recheck every bound input/source after the fixed controls; no changed masks/weights published.
    admit(args.qualification,args.review,args.review_sha256)
    aggregate={}
    for arm in ('baseline',*s.CONTROLS):
        aggregate[arm]={}
        for raster in ('native_full','native_roi','mapped_union'):
            instances=[i[raster] for scene in scenes for i in scene['arms'][arm]['instances']]
            aggregate[arm][raster]={k:sum(scene['arms'][arm]['totals'][raster][k] for scene in scenes)
                for k in ('background_spill','cross_object','missing_foreground')}
            aggregate[arm][raster]['mean_instance_iou']=float(np.mean([i['iou'] for i in instances])) if instances else None
    materiality={}
    for arm in s.CONTROLS:
        reductions={r:1-aggregate[arm][r]['background_spill']/aggregate['baseline'][r]['background_spill']
                    if aggregate['baseline'][r]['background_spill'] else 0. for r in ('native_roi','mapped_union')}
        improved=sum(all(scene['arms'][arm]['totals'][r]['background_spill']<scene['arms']['baseline']['totals'][r]['background_spill']
                         for r in ('native_roi','mapped_union')) for scene in scenes if not scene['negative'])
        harms=[]
        for scene in scenes:
            for raster in ('native_full','native_roi','mapped_union'):
                old=scene['arms']['baseline']['totals'][raster];new=scene['arms'][arm]['totals'][raster]
                if ((old['mean_instance_iou'] is not None and new['mean_instance_iou']<old['mean_instance_iou']) or
                     new['missing_foreground']>old['missing_foreground'] or new['cross_object']>old['cross_object']):harms.append([scene['scene_id'],raster])
        materiality[arm]={'relative_spill_reduction':reductions,'positive_scenes_improved_both':improved,
            'harm_scene_rasters':harms,'passes_preregistered_rule':all(v>=.1 for v in reductions.values()) and improved>=2 and not harms}
    result={'schema':'epd_diagnostic_shadow_results/v1','decision':'DIAGNOSTIC_ONLY_COMPLETED_NO_PROMOTION',
        'qualification_sha256':s.digest(args.qualification),'review_sha256':args.review_sha256,
        'checkpoint_sha256':s.CHECKPOINT_SHA,'dataset_sha256':s.DATASET_SHA,'preregistration_sha256':POLICY_SHA,
        'sources_sha256':p['sources_sha256'],'weight_state_sha256_before_and_after':weight_sha,
        'original_training':p['original_training'],'stage_a2':'BLOCKED_0/2','production_qualified':False,
        'inference_calls':4,'held_out_access':False,'stage_a2_scene_access':False,
        'scenes':scenes,'capture_receipts':receipts,'aggregate':aggregate,'materiality':materiality,
        'worker_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'limitations':['Only these four frozen validation scenes; no generalization/promotion claim.',
          'Controls share logits and isolate paste-rule effects conditional on fixed boxes; effects are not additive.',
          'ROIAlign, learned logit quality and continuous renderer truth not causally isolated.',
          'Class0 renderer IDs mapped to diagnostic background for scoring only; all raw labels retained.',
          'RSS cap is polled plus final HWM rejection, not delegated kernel cgroup memory.max.']}
    durable_json(output/'results.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification',type=Path,required=True);parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--launcher-pid',type=int,default=0,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        try:worker(args);return 0
        except BaseException as exc:
            durable_json(RUN_PATH/'failure.json',{'status':'BLOCKED_MANDATORY_STOP','reason':type(exc).__name__+': '+str(exc),
                         'partial_evidence_preserved':True,'retry_authorized':False})
            raise
    start=time.monotonic()
    if RUN_PATH.exists() or RUN_PATH.is_symlink():raise ValueError('OUTPUT_NOT_EXCLUSIVE_NO_RETRY')
    admit(args.qualification,args.review,args.review_sha256)  # before launching/model loading
    RUN_PATH.mkdir(exist_ok=False)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--launcher-pid',str(os.getpid()),
        '--qualification',str(args.qualification.resolve()),'--review',str(args.review.resolve()),'--review-sha256',args.review_sha256]
    process=supervise(command,RUN_PATH,max(0,300-(time.monotonic()-start)),6144*1024)
    success=process['status']=='WORKER_COMPLETED' and (RUN_PATH/'results.json').is_file()
    durable_json(RUN_PATH/'completion.json',{'decision':'DIAGNOSTIC_ONLY_COMPLETED_NO_PROMOTION' if success else 'BLOCKED_MANDATORY_STOP',
        'results_sha256':s.digest(RUN_PATH/'results.json') if (RUN_PATH/'results.json').exists() else None,
        'supervision_sha256':s.digest(RUN_PATH/'supervision.json'),'total_wall_seconds':time.monotonic()-start,
        'original_training_supervision':'UNKNOWN','stage_a2':'BLOCKED_0/2','production_qualified':False})
    print(json.dumps(process,sort_keys=True))
    return 0 if success else 2


if __name__=='__main__':raise SystemExit(main())
