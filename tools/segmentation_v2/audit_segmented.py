#!/usr/bin/env python3
"""Audit sealed 4+4 campaign and replay validation. Never trains or infers."""
import argparse
import json
from pathlib import Path
import numpy as np
from campaign import epoch_order, score_predictions, eligible, checkpoint_key, campaign_rows
from continuation import binding, load_checkpoint, validate_progress
from pilot import HERE, sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def verify_seal(folder, stage, expected_binding):
    seal=read(folder/'stage_complete.json')
    if (seal.get('schema')!='epd_v2_completed_stage/1' or seal.get('stage')!=stage
            or seal.get('binding')!=expected_binding):
        raise ValueError('seal identity mismatch')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}-{'stage_complete.json'}
    if actual!=set(seal['files']) or any(n.endswith('.partial') for n in actual):
        raise ValueError('seal inventory mismatch')
    for name,digest in seal['files'].items():
        if sha(folder/name)!=digest:raise ValueError('sealed hash mismatch: '+name)
    return seal


def verify_history(first, second, ids, seed):
    count=len(ids)
    expected=[(e,(e-1)*count+i+1,ids[j]) for e in range(1,9)
              for i,j in enumerate(epoch_order(count,seed,e))]
    actual=[(r.get('epoch'),r.get('step'),r.get('scene_id')) for r in first+second]
    if len(first)!=4*count or len(second)!=4*count or actual!=expected:
        raise ValueError('missing/repeated/out-of-order segmented steps')


def choose(epochs):
    accepted=[m for m in epochs if m['eligible']]
    return max(accepted,key=checkpoint_key)['checkpoint'] if accepted else None


def audit(artifacts, stage1, stage2, expected_resume):
    import torch
    policy_path=HERE/'campaign_segmented_300.json';recipe_path=HERE/'train.json'
    policy=read(policy_path);cfg=read(recipe_path)
    expected_binding=binding(artifacts,recipe_path,policy_path)
    seals=[verify_seal(folder,i,expected_binding) for i,folder in enumerate((stage1,stage2),1)]
    results=[read(folder/'result.json') for folder in (stage1,stage2)]
    processes=[read(folder/'process.json') for folder in (stage1,stage2)]
    first,last=results
    if (first['stage']!=1 or first['steps']!=1200 or first['epochs']!=4 or
            last['stage']!=2 or last['steps']!=2400 or last['stage_steps']!=1200 or last['epochs']!=8 or
            last['first_epoch']!=5 or last['last_epoch']!=8 or last['resumed_from']!=expected_resume or
            Path(last['resume_stage'])!=stage1 or first['checkpoints'][-1]['sha256']!=expected_resume):
        raise ValueError('continuation identity/count mismatch')
    for i,(result,process,seal) in enumerate(zip(results,processes,seals)):
        if result['status']!='STAGE_COMPLETE' or process['exit_code']!=0 or process['resource_limit_reason'] is not None:
            raise ValueError('stage process incomplete')
        prior=0 if i==0 else seals[0]['aggregate_wall_seconds']
        if (result['binding']!=expected_binding or result['prior_wall_seconds']!=prior or
                seal['aggregate_wall_seconds']!=prior+process['wall_seconds'] or
                process['wall_seconds']>policy['stage_timeout_seconds'] or
                max(process['child_peak_rss_kib'],process['sampled_process_peak_rss_kib'])>policy['memory_mib']*1024):
            raise ValueError('binding or resource accounting mismatch')
    if seals[-1]['aggregate_wall_seconds']>policy['aggregate_timeout_seconds']:
        raise ValueError('aggregate wall budget exceeded')
    train,validation=campaign_rows(artifacts,policy)
    visible_counts=[sum(entry['class_id']==1 and bool((labels==entry['renderer_id']).any())
                        for entry in report['mapping']) for _,_,report,labels in train]
    if len(train)!=300 or sum(visible_counts)!=484 or visible_counts.count(0)!=50:
        raise ValueError('frozen training population mismatch')
    histories=[[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
               for folder in (stage1,stage2)]
    ids=[r['scene_id'] for r,_,_,_ in train]
    verify_history(*histories,ids,cfg['seed'])
    checkpoints=last['checkpoints']
    if [c['epoch'] for c in checkpoints]!=list(range(1,9)) or checkpoints[:4]!=first['checkpoints']:
        raise ValueError('checkpoint sequence mismatch')
    epochs=[]
    for checkpoint in checkpoints:
        epoch=checkpoint['epoch'];folder=stage1 if epoch<=4 else stage2
        if Path(checkpoint['path'])!=folder/f'epoch_{epoch:02d}.pt':raise ValueError('checkpoint path')
        state=load_checkpoint(checkpoint['path'],checkpoint['sha256'],torch)
        required={'model','optimizer','python_rng_state','numpy_rng_state','torch_rng_state','cuda_rng_state'}
        if not required<=state.keys() or state['cuda_rng_state'] is not None:
            raise ValueError('incomplete CPU continuation state')
        history=(histories[0]+histories[1])[:epoch*300]
        validate_progress(state,history,expected_binding,ids,cfg['seed'],epoch)
        if state['config']!=cfg or state['campaign']!=policy:raise ValueError('checkpoint configuration')
        for group in state['optimizer']['param_groups']:
            if (group['lr']!=cfg['learning_rate'] or group['momentum']!=cfg['momentum'] or
                    group['weight_decay']!=cfg['weight_decay']):raise ValueError('optimizer drift')
        if not state['optimizer']['state']:raise ValueError('missing optimizer momentum')
        del state
        original=read(folder/f'validation_{epoch:02d}/metrics.json')
        summary=read(folder/f'epoch_{epoch:02d}_summary.json')
        if summary['checkpoint']!=checkpoint or summary['metrics']!=original:raise ValueError('epoch summary drift')
        raw={p['scene_id']:p for p in original['raw_predictions']}
        if len(original['raw_predictions'])!=12 or set(raw)!={r['scene_id'] for r,_,_,_ in validation}:
            raise ValueError('validation prediction membership')
        scenes=[]
        for row,scene,report,labels in validation:
            prediction=raw[row['scene_id']]
            if Path(prediction['path'])!=folder/f'validation_{epoch:02d}'/(scene.name+'.npz'):
                raise ValueError('raw prediction path')
            if sha(prediction['path'])!=prediction['sha256']:raise ValueError('prediction hash')
            with np.load(prediction['path'],allow_pickle=False) as saved:
                targets={r['renderer_id'] for r in report['mapping'] if r['class_id']==1 and (labels==r['renderer_id']).any()}
                scenes.append(score_predictions(row['scene_id'],dict(saved),labels,targets))
        if scenes!=original['scenes']:raise ValueError('per-instance/detection/negative replay mismatch')
        instances=[m for scene in scenes for m in scene['instances']]
        score={k:sum(s[k] for s in scenes) for k in ('positives','recovered','false_positives','background_spill',
            'cross_object','missing_visible','contract_violations')}
        score.update(epoch=epoch,mean_iou=float(np.mean([m['iou'] for m in instances])),
            mean_boundary_px=float(np.mean([m['boundary_mean_px'] if m['boundary_mean_px'] is not None else 512 for m in instances])),
            max_boundary_px=max(m['boundary_max_px'] if m['boundary_max_px'] is not None else 512 for m in instances))
        for key,value in score.items():
            if original[key]!=value:raise ValueError('offline metric mismatch: '+key)
        if original['confidence_threshold']!=.8 or original['mask_threshold_strict_gt']!=.5:
            raise ValueError('threshold drift')
        score['eligible']=eligible(score,policy)
        if score['eligible']!=original['eligible']:raise ValueError('gate drift')
        score['gate_failures']=[name for name,passed in dict(recall=score['recovered']>=18,
            iou=score['mean_iou']>.685890,false_positives=score['false_positives']<=1,
            background_spill=score['background_spill']<1915,contract=score['contract_violations']==0).items() if not passed]
        score.update(checkpoint=checkpoint,mean_losses=summary['mean_losses'],
            negative_scenes=[s for s in scenes if s['negative']],exact_per_instance_replay=True)
        epochs.append(score)
    selected=choose(epochs)
    if selected!=last['selected']:raise ValueError('selection mismatch')
    for score,record in zip(epochs,last['epoch_metrics']):
        if any(score[k]!=v for k,v in record.items() if k!='recall'):raise ValueError('combined epoch metric mismatch')
        if record['recall']!=score['recovered']/score['positives']:raise ValueError('recall mismatch')
    return dict(classification='V2_300_VALIDATION_PASS_EXPORT_PENDING' if selected else 'V2_300_EIGHT_EPOCHS_COMPLETED_REJECTED',
        epochs=epochs,selected=selected,diagnostic_checkpoint=max(epochs,key=checkpoint_key)['checkpoint'],
        training_steps=2400,epochs_completed=8,exact_offline_replay=True,validation_prediction_files=96,
        training_samples=300,training_visible_instances=484,training_negative_scenes=50,
        stage_wall_seconds=[p['wall_seconds'] for p in processes],aggregate_wall_seconds=seals[-1]['aggregate_wall_seconds'],
        peak_rss_kib=max(p['child_peak_rss_kib'] for p in processes),binding=expected_binding,
        stage_seal_sha256=[sha(f/'stage_complete.json') for f in (stage1,stage2)],
        held_out_pixels_opened=0,historical_regression_predictions=0,model_inference_executed=False,
        audit_source_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts','stage1','stage2','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume-sha256',required=True);args=parser.parse_args()
    report=audit(args.artifacts,args.stage1,args.stage2,args.resume_sha256)
    write_json(args.output,report);print(report['classification'])
