#!/usr/bin/env python3
"""Recompute retained Stage1 validation metrics; no model/training/inference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--repository',type=Path,required=True)
parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--stage1',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
sys.path.insert(0,str(args.repository/'tools/segmentation_v2'))
from train import verified_rows
from campaign import score_predictions,eligible
from pilot import sha,write_json
rows=list(verified_rows(args.artifacts,'validation'));reports=[]
policy=json.loads((args.stage1/'run_config.json').read_text())['policy']
for epoch in range(1,5):
    metrics=json.loads((args.stage1/f'validation_{epoch:02d}/metrics.json').read_text())
    raw={p['scene_id']:p for p in metrics['raw_predictions']}
    assert set(raw)=={r['scene_id'] for r,_,_,_ in rows}
    recomputed=[]
    for row,folder,report,ids in rows:
        prediction=raw[row['scene_id']];assert sha(prediction['path'])==prediction['sha256']
        with np.load(prediction['path'],allow_pickle=False) as saved:
            targets={r['renderer_id'] for r in report['mapping'] if r['class_id']==1 and (ids==r['renderer_id']).any()}
            recomputed.append(score_predictions(row['scene_id'],dict(saved),ids,targets))
    assert recomputed==metrics['scenes'],'per-instance/detection/negative replay mismatch'
    instances=[m for scene in recomputed for m in scene['instances']]
    score={k:sum(s[k] for s in recomputed) for k in ('positives','recovered','false_positives','background_spill',
                'cross_object','missing_visible','contract_violations')}
    score.update(mean_iou=float(np.mean([m['iou'] for m in instances])),
        mean_boundary_px=float(np.mean([m['boundary_mean_px'] if m['boundary_mean_px'] is not None else 512 for m in instances])))
    for key,value in score.items():assert value==metrics[key],key
    assert eligible(score,policy)==metrics['eligible']
    reports.append(dict(epoch=epoch,exact_metric_replay=True,**score,eligible=metrics['eligible']))
write_json(args.output,dict(epochs=reports,validation_scenes=12,held_out_pixels_opened=0,
          historical_regression_predictions=0,model_inference_executed=False,source_sha256=sha(__file__)))
print('All4 epoch metrics and per-instance/negative/detection results replay exactly')
