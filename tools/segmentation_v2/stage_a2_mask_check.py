#!/usr/bin/env python3
"""One post-selection Stage-A2 check using retained strict association unchanged."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from pilot import sha,write_json,bounded
from recover_original import make_model
from campaign import validate_prediction


def worker(args):
    import torch
    run=json.loads((args.campaign/'result.json').read_text());chosen=run['selected']
    if run['status']!='VALIDATION_PASS' or chosen is None:raise ValueError('validation-qualified checkpoint required')
    if sha(chosen['path'])!=chosen['sha256']:raise ValueError('candidate modified')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    state=torch.load(chosen['path'],map_location='cpu',weights_only=True)
    model=make_model();model.load_state_dict(state['model'],strict=True);model.eval()
    # Lossless extraction of genuine retained artifacts. Never launch Gazebo/EMD.
    capture=args.output/'capture';capture.mkdir()
    hashes=json.loads((args.retained/'sha256.json').read_text())
    for name,digest in hashes.items():
        source=args.retained/name
        raw=source.read_bytes() if source.exists() else gzip.decompress((args.retained/(name+'.gz')).read_bytes())
        target=capture/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        if sha(target)!=digest:raise ValueError('retained capture hash mismatch')
    (capture/'raw_sha256.json').write_bytes((args.retained/'sha256.json').read_bytes())
    sys.path.insert(0,str(args.emd_scripts))
    from stage_a_epd_retained import load_capture, map_mask, associate
    owner,report,rgb,ids,binding=load_capture(capture)
    image=torch.from_numpy(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR).transpose(2,0,1).copy()).float()/255
    t=time.monotonic()
    with torch.inference_mode():result=model([image])[0]
    pred={k:v.numpy() for k,v in result.items()};validate_prediction(pred)
    np.savez_compressed(args.output/'predictions.npz',**pred)
    detections=[];mapped=[];measurements=[]
    native_ids=np.fromfile(capture/'capture.json.native512.ids.u32',dtype='<u4').reshape(512,512)
    for i,(box,label,score,mask) in enumerate(zip(pred['boxes'],pred['labels'],pred['scores'],pred['masks'][:,0])):
        box=[int(v) for v in np.clip(box,0,512)]
        x0,y0,x1,y1=box;roi=mask[y0:y1,x0:x1]
        mapped_mask=map_mask(roi,box);mapped.append(mapped_mask)
        detections.append(dict(prediction_index=i,class_index=int(label),label='cube',confidence=float(score),bbox=box))
        values,counts=np.unique(ids[mapped_mask],return_counts=True)
        identity=max((int(v) for v in np.unique(ids) if v),key=lambda v:int(((ids==v)&mapped_mask).sum()))
        measurements.append(dict(prediction_index=i,score=float(score),reference_identity=identity,
            mapped_missing_pixels=int(((ids==identity)&~mapped_mask).sum()),
            native512_raw_missing=int(((native_ids==identity)&~(mask>.5)).sum()),
            mapped_id_counts={str(int(v)):int(c) for v,c in zip(values,counts)},
            native512_raw_background=int(((mask>.5)&(native_ids==0)).sum()),
            native512_roi_background=int(((roi>.5)&(native_ids[y0:y1,x0:x1]==0)).sum())))
    association=associate(owner,report,rgb,ids,detections,mapped)
    write_json(args.output/'result.json',dict(status='STAGE_A2_ASSOCIATION_PASS' if association['accepted']==2 else 'BLOCKED_STRICT_ASSOCIATION',
        checkpoint=chosen,source_sha256=sha(__file__),input_rgb_sha256=report['rgb_sha256'],capture_manifest_sha256=sha(args.retained/'sha256.json'),
        association=association,measurements=measurements,inference_seconds=time.monotonic()-t,
        mapping='unchanged EPD integer clipped bbox ROI; exact values; existing union2x2 map_mask; >.5 and >=.8',
        emd_sources={name:sha(args.emd_scripts/name) for name in ('stage_a_epd_retained.py','stage_a_fragment_identity.py')},
        ids_used_as_model_inputs=False,held_out_files_opened=0,used_for_selection=False,exported=False))
    print(json.dumps(dict(accepted=association['accepted'],measurements=measurements)))
    return 0 if association['accepted']==2 else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker'])
    for name in ('campaign','retained','emd-scripts','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.action=='worker':sys.exit(worker(args))
    args.output.mkdir();command=[sys.executable,str(Path(__file__).resolve()),'worker']
    for name in ('campaign','retained','emd_scripts','output'):command+=['--'+name.replace('_','-'),str(getattr(args,name).resolve())]
    process=bounded(command,args.output,180,memory_mib=6144)
    write_json(args.output/'process.json',process);print(json.dumps(process));sys.exit(0 if process['exit_code']==0 else 2)
