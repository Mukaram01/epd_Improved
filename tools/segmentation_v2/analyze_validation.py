#!/usr/bin/env python3
"""Offline reanalysis of retained genuine validation predictions; no inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from pilot import sha,write_json
from train import verified_rows,prediction_metrics


def analyze(artifacts,output):
    folder=artifacts/'validation-diagnostic-corrected'
    report=json.loads((folder/'result.json').read_text())
    rows=list(verified_rows(artifacts,'validation'))
    for name,model in report['models'].items():
        detections=[]
        for row,scene,cap,ids in rows:
            path=folder/f"{name}_{row['scene_id']}.npz"
            recorded=next(i for i in model['inferences'] if i['scene_id']==row['scene_id'])
            if sha(path)!=recorded['sha256']: raise ValueError('changed retained prediction')
            raw=np.load(path)
            targets={v['renderer_id'] for v in cap['mapping'] if v['class_id']==1 and (ids==v['renderer_id']).any()}
            detections.extend(dict(scene_id=row['scene_id'],**m)
                              for m in prediction_metrics(raw['masks_confident'][:,0],ids,targets))
        model['all_confident_detection_metrics']=detections
        model['all_confident_pixel_totals']={k:sum(m[k] for m in detections)
                                           for k in ('background_spill','cross_object','missing_visible')}
        model['native_pixel_purity']=dict(accepted=sum(m['strict_accept'] for m in detections),
            rejected=sum(not m['strict_accept'] for m in detections),
            scope='512 native raster only; NOT EMD conservative association')
        model['strict_emd_association_counts']=None
    report['offline_analysis']='All confident detections rescored from retained genuine NPZ outputs without inference, including unmatched and negative-scene predictions; per-object metrics remain separately in instances.'
    report['offline_analysis_sources']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    write_json(output,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.artifacts,a.output)
