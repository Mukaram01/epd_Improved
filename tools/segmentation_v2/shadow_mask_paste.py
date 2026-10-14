#!/usr/bin/env python3
"""Shadow-only paste reconstruction, measurements and fail-closed source/admission audit.

Does not load a checkpoint or run model inference. A missing/unpinned training
supervisor blocks scene execution, regardless of checkpoint/validation success.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
CHECKPOINT_SHA='bf3ac58116a3a692806ed20b25fb5414277e76da605cf0f60e2e5a23656bfbfe'
DATASET_SHA='04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11'
ACCEPTANCE_RGB_SHA='92f1ba8ad58b2437fa5ccf55e73d53cc9dcba469b04efefbe624ce1505585793'
SCENES=tuple(f'scene_{i:03}' for i in range(24,28))
CONTROLS=('round_expanded_box','exclusive_extent','align_corners_true')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def reconstruct(logits,boxes,shape,control='baseline'):
    """Independent orchestration of eager TV paste; low-level Torch interpolation.

    Inputs are selected-class PRE-sigmoid logits. Original boxes never change;
    controls alter exactly one derived paste rule, never upstream ROI features.
    """
    import torch
    import torch.nn.functional as F
    if (control not in ('baseline',*CONTROLS) or logits.dtype!=torch.float32 or
            boxes.dtype!=torch.float32 or logits.device.type!='cpu' or boxes.device.type!='cpu' or
            logits.ndim!=4 or tuple(logits.shape[1:])!=(1,56,56) or
            tuple(boxes.shape)!=(len(logits),4) or not torch.isfinite(logits).all() or
            not torch.isfinite(boxes).all() or (boxes[:,2:]<=boxes[:,:2]).any() or
            len(shape)!=2 or any(type(v) is not int or v<=0 or v>512 for v in shape)):
        raise ValueError('PASTE_CONTRACT: corrupt logits/boxes/shape/control')
    if ((boxes < 0).any() or (boxes[:,[0,2]] > shape[1]).any() or
            (boxes[:,[1,3]] > shape[0]).any()):
        raise ValueError('PASTE_CONTRACT: original model boxes must be image-clipped')
    # Same arithmetic order as installed expand_boxes; do not round originals.
    prob=F.pad(logits.sigmoid(),(1,1,1,1))
    half=(boxes[:,2:]-boxes[:,:2])*.5
    centre=(boxes[:,2:]+boxes[:,:2])*.5
    half=half*(58./56.)
    expanded=torch.empty_like(boxes)
    expanded[:,:2]=centre-half
    expanded[:,2:]=centre+half
    integer=(expanded.round() if control=='round_expanded_box' else expanded).to(torch.int64)
    h_image,w_image=shape
    out=logits.new_zeros((len(logits),1,h_image,w_image))
    extent=0 if control=='exclusive_extent' else 1
    for i,box in enumerate(integer):
        x0,y0,x2,y2=map(int,box)
        width=max(x2-x0+extent,1); height=max(y2-y0+extent,1)
        resized=F.interpolate(prob[i:i+1],size=(height,width),mode='bilinear',
                              align_corners=control=='align_corners_true')[0,0]
        left,top=max(x0,0),max(y0,0)
        right,bottom=min(x2+extent,w_image),min(y2+extent,h_image)
        if right>left and bottom>top:
            out[i,0,top:bottom,left:right]=resized[top-y0:bottom-y0,left-x0:right-x0]
    return out


def parity(original,reconstructed):
    import torch
    if (original.shape!=reconstructed.shape or original.dtype!=reconstructed.dtype or
            not torch.isfinite(original).all() or not torch.isfinite(reconstructed).all() or
            not torch.equal(original,reconstructed)):
        raise ValueError('BASELINE_PARITY: require bitwise probability and threshold parity')
    return {'max_abs':0.,'threshold_differences':0,'bitwise_equal':True}


def project(prob,box):
    """Preserve consumer ROI truncation, strict >.5 and conservative ANY2x2."""
    if (prob.ndim!=2 or any(d%2 for d in prob.shape) or prob.dtype!=np.float32 or
            not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any() or
            np.asarray(box).shape!=(4,) or not np.isfinite(box).all()):
        raise ValueError('PROJECTION_CONTRACT')
    h,w=prob.shape
    x0,y0,x1,y1=np.asarray(box).clip([0,0,0,0],[w,h,w,h]).astype(int)
    if not (x0<x1 and y0<y1): raise ValueError('PROJECTION_CONTRACT: empty ROI')
    native=np.zeros_like(prob,dtype=bool)
    native[y0:y1,x0:x1]=prob[y0:y1,x0:x1]>.5
    return native,native.reshape(h//2,2,w//2,2).any(axis=(1,3))


def pixel_metrics(mask,ids,identity):
    import cv2
    if (isinstance(identity,(bool,np.bool_)) or not isinstance(identity,(int,np.integer)) or identity<=0):
        raise ValueError('METRIC_IDENTITY: require a positive integer renderer ID')
    target=ids==identity
    if not target.any(): raise ValueError('METRIC_IDENTITY: target absent')
    spill=mask&(ids==0)
    distances=cv2.distanceTransform((~target).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[spill]
    union=int((mask|target).sum())
    return {'iou':int((mask&target).sum())/union,
            'background_spill':int(spill.sum()),
            'missing_foreground':int((target&~mask).sum()),
            'cross_object':int((mask&(ids!=0)&~target).sum()),
            'spill_distance_bins_le1_1to2_2to3_gt3':[
                int((distances<=1).sum()),int(((distances>1)&(distances<=2)).sum()),
                int(((distances>2)&(distances<=3)).sum()),int((distances>3).sum())],
            'spill_distance_max':float(distances.max()) if len(distances) else None}


def measure(prob,box,ids512,ids256,identity,original):
    if (ids512.dtype!=np.uint32 or ids256.dtype!=np.uint32 or
            ids512.shape!=prob.shape or ids256.shape!=tuple(d//2 for d in prob.shape) or
            original.shape!=prob.shape or original.dtype!=np.float32 or
            not np.isfinite(original).all()):
        raise ValueError('METRIC_CONTRACT')
    roi,mapped=project(prob,box)
    old_roi,old_map=project(original,box)
    occupancy=roi.reshape(prob.shape[0]//2,2,prob.shape[1]//2,2).sum(axis=(1,3))
    disagree=ids512!=ids256.repeat(2,0).repeat(2,1)
    return {'native_full':pixel_metrics(prob>.5,ids512,identity),
            'native_roi':pixel_metrics(roi,ids512,identity),
            'mapped_union':pixel_metrics(mapped,ids256,identity),
            'union_occupancy_1_to_4':[int((occupancy==n).sum()) for n in range(1,5)],
            'mapped_bg_partial_support':int((mapped&(ids256==0)&(occupancy<4)).sum()),
            'native_label_disagreement_under_roi':int((disagree&roi).sum()),
            'native_changed_pixels':int(((prob>.5)!=(original>.5)).sum()),
            'roi_added_pixels':int((roi&~old_roi).sum()),'roi_removed_pixels':int((old_roi&~roi).sum()),
            'mapped_changed_cells':int((mapped!=old_map).sum()),
            'mapped_added_cells':int((mapped&~old_map).sum()),'mapped_removed_cells':int((old_map&~mapped).sum()),
            'max_abs_probability_change':float(np.abs(prob-original).max(initial=0))}


def compare_shadow(logits,boxes,original,ids512,ids256,identities):
    """Require baseline parity BEFORE any counterfactual; fixed identities supplied.

    The caller must bind identities once against baseline, never reassign per
    control. Scores/selection and weight inference are outside this pure replay.
    """
    baseline=reconstruct(logits,boxes,tuple(ids512.shape))
    result={'parity':parity(original,baseline),'comparisons':{}}
    if len(identities)!=len(boxes): raise ValueError('METRIC_IDENTITY: wrong count')
    for identity in identities:
        if (isinstance(identity,(bool,np.bool_)) or not isinstance(identity,(int,np.integer)) or
                identity<=0 or not (ids512==identity).any() or not (ids256==identity).any()):
            raise ValueError('METRIC_IDENTITY: invalid or absent fixed renderer ID')
    for control in ('baseline',*CONTROLS):
        masks=baseline if control=='baseline' else reconstruct(logits,boxes,tuple(ids512.shape),control)
        result['comparisons'][control]=[
            measure(p[0].detach().numpy(),b.detach().numpy(),ids512,ids256,identity,o[0].detach().numpy())
            for p,b,identity,o in zip(masks,boxes,identities,original)]
    return result


def select_scenes(rows):
    selected=sorted((r for r in rows if r['scene_id'] in SCENES),key=lambda r:r['scene_id'])
    if (tuple(r['scene_id'] for r in selected)!=SCENES or
            len({r['seed'] for r in selected})!=4 or any(
                r['split']!='validation' or r['sha256'].get('capture.json.rgb8')==ACCEPTANCE_RGB_SHA
                for r in selected)):
        raise ValueError('SCENE_POLICY: only four frozen independent validation scenes')
    return selected


def admission(policy):
    for key in ('checkpoint','dataset'):
        row=policy[key]
        if digest(row['path'])!=row['sha256']: raise ValueError('INPUT_SHA: '+key)
    for row in policy.get('campaign_files',[]):
        if digest(row['path'])!=row['sha256']: raise ValueError('CAMPAIGN_SHA')
    row=policy['supervisor']; path=Path(row['path'])
    if not path.is_file(): return {'status':'BLOCKED_MISSING_TRAINING_PROVENANCE','missing':str(path)}
    if not row['sha256']: return {'status':'BLOCKED_UNPINNED_TRAINING_PROVENANCE','path':str(path)}
    if digest(path)!=row['sha256']: raise ValueError('SUPERVISOR_SHA')
    record=json.loads(path.read_text())
    if record.get('exit_code')!=0 or record.get('resource_limit_reason') is not None:
        raise ValueError('SUPERVISOR_NOT_QUALIFIED')
    return {'status':'PROVENANCE_HASH_VERIFIED'}


def source_inventory():
    tv=Path(metadata.distribution('torchvision').locate_file('torchvision'))
    torch=Path(metadata.distribution('torch').locate_file('torch'))
    paths=[tv/p for p in ('models/detection/roi_heads.py','models/detection/transform.py',
           'models/detection/mask_rcnn.py','models/detection/generalized_rcnn.py',
           'ops/poolers.py','ops/roi_align.py','_C.so')]
    paths += [torch/'nn/functional.py',torch/'lib/libtorch_cpu.so']
    paths += [HERE/p for p in ('recover_original.py','mask_resolution.py','train.py',
                              'stage_a2_mask_check.py','shadow_mask_paste.py','pilot.py')]
    inventory={str(p):digest(p) for p in paths}
    functions={}
    for p in paths:
        if p.suffix!='.py': continue
        text=p.read_text(); lines=text.splitlines(keepends=True)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {
                    'paste_mask_in_image','paste_masks_in_image','expand_boxes','expand_masks',
                    'maskrcnn_inference','resize_boxes','_roi_align','roi_align','interpolate'}:
                body=''.join(lines[node.lineno-1:node.end_lineno]).encode()
                functions[str(p)+':'+str(node.lineno)+':'+node.name]=hashlib.sha256(body).hexdigest()
    return {'versions':{n:metadata.version(n) for n in ('torch','torchvision','numpy','opencv-python')},
            'files_sha256':inventory,'functions_sha256':functions}


def audit(policy_path,output):
    start=time.monotonic()
    if output.exists() or output.is_symlink(): raise ValueError('OUTPUT_NOT_EXCLUSIVE')
    policy=json.loads(Path(policy_path).read_text())
    if (policy['checkpoint']['sha256']!=CHECKPOINT_SHA or policy['dataset']['sha256']!=DATASET_SHA or
            policy['controls']!=list(CONTROLS) or
            policy['freeze']['confidence_threshold']!=.8 or policy['freeze']['mask_threshold_strict_gt']!=.5 or
            policy['parity_gate']['probability_bitwise_equal'] is not True or
            policy['parity_gate']['threshold_differences']!=0 or policy['limits']!={'cpu_threads':2,'max_scenes':4,'wall_seconds':300,'rss_mib':6144}):
        raise ValueError('PREREGISTRATION_DRIFT')
    source=source_inventory()
    if source!=policy['source_inventory']: raise ValueError('SOURCE_VERSION_DRIFT')
    if shutil.disk_usage(output.parent).free<6*1024**3: raise ValueError('DISK_RESERVE')
    # Metadata only; no per-scene file is touched before provenance admission.
    status=admission(policy)
    dataset=json.loads(Path(policy['dataset']['path']).read_text())
    rows=select_scenes(dataset['records'])
    if rows!=policy['scenes']: raise ValueError('SCENE_BINDING_DRIFT')
    result={'schema':'epd_shadow_paste_admission/v1',**status,
            'preregistration_sha256':digest(policy_path),'diagnostic_source_sha256':digest(__file__),
            'source_inventory':source,'campaign_files':policy.get('campaign_files',[]),
            'checkpoint_sha256':CHECKPOINT_SHA,'dataset_sha256':DATASET_SHA,
            'selected_scene_ids':list(SCENES),'scene_pixels_opened':0,'held_out_files_opened':0,
            'stage_a2_files_opened':0,'inference_calls':0,'model_loaded':False,
            'real_scene_parity':None,'real_scene_geometry_comparisons':None,
            'strict_stage_a2':'0/2_UNCHANGED','promotion':False,
            'wall_seconds':time.monotonic()-start,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    output.mkdir(exist_ok=False)
    (output/'admission_report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preregistration',type=Path,default=HERE/'shadow_mask_paste_preregistration.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.worker:
        from pilot import bounded
        if args.output.exists() or args.output.is_symlink():
            print('OUTPUT_NOT_EXCLUSIVE',file=sys.stderr); return 2
        args.output.mkdir(parents=False,exist_ok=False)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',
                 '--preregistration',str(args.preregistration.resolve()),
                 '--output',str(args.output.resolve()/'admission')]
        process=bounded(command,args.output,300,dict(os.environ,OMP_NUM_THREADS='2',
                        OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),memory_mib=6144)
        (args.output/'audit_process.json').write_text(json.dumps(process,indent=2,sort_keys=True)+'\n')
        print(json.dumps(process,sort_keys=True))
        return 0 if process['exit_code']==0 else 2
    try: result=audit(args.preregistration,args.output)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc)}),file=sys.stderr); return 2
    print(json.dumps(result,sort_keys=True))
    return 2 if result['status'].startswith('BLOCKED') else 0


if __name__=='__main__': raise SystemExit(main())
