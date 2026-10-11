#!/usr/bin/env python3
"""Recompute full-campaign decisions from hash-pinned raw validation outputs.

No model inference, checkpoint loading, training, held-out scene access or mask
correction. Also verifies exactly one visit to every training scene per epoch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from campaign import audit_identities, score_predictions, eligible, checkpoint_key
from train import verified_rows
from pilot import sha, write_json


def audit(artifacts, training):
    result=json.loads((training/'result.json').read_text())
    run=json.loads((training/'run_config.json').read_text());policy=run['policy']
    identity=audit_identities(artifacts);count=policy.get('training_samples',24)
    if result['epochs']!=policy['epochs'] or result['steps']!=count*policy['epochs']:
        raise ValueError('full campaign incomplete')
    if [c['epoch'] for c in result['checkpoints']]!=list(range(1,policy['epochs']+1)):
        raise ValueError('checkpoint epochs')
    history=[json.loads(line) for line in (training/'history.jsonl').read_text().splitlines()]
    steps=count*policy['epochs']
    if len(history)!=steps or [r.get('step') for r in history]!=list(range(1,steps+1)) or any(
            r['epoch']!=(r['step']-1)//count+1 for r in history):
        raise ValueError('training history')
    expected={r['scene_id'] for r,_,_,_ in verified_rows(artifacts,'train')}
    for epoch in range(1,policy['epochs']+1):
        rows=[r for r in history if r['epoch']==epoch]
        if len(rows)!=count or {r['scene_id'] for r in rows}!=expected:raise ValueError('training split visits')
    validation=list(verified_rows(artifacts,'validation'));epochs=[];best=None
    for checkpoint in result['checkpoints']:
        if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('checkpoint hash')
        epoch=checkpoint['epoch'];folder=training/f'validation_{epoch:02d}'
        original=json.loads((folder/'metrics.json').read_text());scenes=[]
        raw={r['scene_id']:r for r in original['raw_predictions']}
        if set(raw)!={r['scene_id'] for r,_,_,_ in validation}:raise ValueError('validation membership')
        for row,scene,report,ids in validation:
            item=raw[row['scene_id']]
            if sha(item['path'])!=item['sha256']:raise ValueError('prediction hash')
            with np.load(item['path'],allow_pickle=False) as pred:
                targets={r['renderer_id'] for r in report['mapping'] if r['class_id']==1 and (ids==r['renderer_id']).any()}
                scenes.append(score_predictions(scene.name,dict(pred),ids,targets))
        instances=[m for scene in scenes for m in scene['instances']]
        score={k:sum(s[k] for s in scenes) for k in ('positives','recovered','false_positives','background_spill',
               'cross_object','missing_visible','contract_violations')}
        score.update(epoch=epoch,mean_iou=float(np.mean([m['iou'] for m in instances])),
            mean_boundary_px=float(np.mean([m['boundary_mean_px'] if m['boundary_mean_px'] is not None else 512 for m in instances])),
            max_boundary_px=max(m['boundary_max_px'] if m['boundary_max_px'] is not None else 512 for m in instances),
            scenes=scenes,checkpoint=checkpoint)
        for key in ('positives','recovered','false_positives','background_spill','cross_object','missing_visible',
                    'contract_violations','mean_iou','mean_boundary_px'):
            if score[key]!=original[key]:raise ValueError('offline metric mismatch: '+key)
        score['eligible']=eligible(score,policy)
        if score['eligible']!=original['eligible']:raise ValueError('gate drift')
        score['gate_failures']=[key for key,passed in {
            'recall_at_least_18_of_22':score['recovered']>=18 and score['positives']==22,
            'mean_iou_strictly_above_0.685890':score['mean_iou']>.685890,
            'false_positives_at_most_1':score['false_positives']<=1,
            'background_spill_strictly_below_1915':score['background_spill']<1915,
            'zero_scoped_contract_violations':score['contract_violations']==0}.items() if not passed]
        epochs.append(score)
        if score['eligible'] and (best is None or checkpoint_key(score)>checkpoint_key(best)):best=score
    if (best is None)!=(result['selected'] is None):raise ValueError('selection discrepancy')
    if best and result['selected']['sha256']!=best['checkpoint']['sha256']:raise ValueError('selected checkpoint discrepancy')
    return dict(status='VALIDATION_ELIGIBLE' if best else 'V2_TRAINED_BUT_REJECTED',
        identity_audit=identity,epochs=epochs,selected=best['checkpoint'] if best else None,
        original_metrics_reproduced_exactly=True,zero_boundary_reporting_fixed=True,
        checkpoint_count=len(epochs),training_steps_verified=len(history),
        held_out_files_opened=0,historical_regression_evaluated=False,
        policy_sha256=run['sources'].get('campaign_300.json',run['sources'].get('campaign.json')),audit_source_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=audit(args.artifacts,args.training);write_json(args.output,report)
    print(report['status'])
