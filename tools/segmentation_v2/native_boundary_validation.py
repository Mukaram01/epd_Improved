"""Prospective full-validation paired gate; synthetic tests only in this task."""
import numpy as np
from campaign import validate_prediction
from diagnostic_shadow_capture import fixed_matching,metrics,scoring_ids
from shadow_mask_paste import project


def compare_scene(baseline,candidate,raw512,raw256,cube_ids):
    validate_prediction(baseline);validate_prediction(candidate)
    for key in ('boxes','labels','scores'):
        if not np.array_equal(baseline[key],candidate[key]):raise ValueError('FROZEN_DETECTION_DRIFT '+key)
    ids512=scoring_ids(raw512,cube_ids);ids256=scoring_ids(raw256,cube_ids)
    targets={int(i) for i in cube_ids if (ids512==i).any()}
    keep=np.where((baseline['scores']>=.8)&(baseline['labels']==1))[0]
    refs,assignment=fixed_matching(baseline['masks'],keep,ids512,targets)
    outcomes={}
    for name,pred in [('baseline',baseline),('candidate',candidate)]:
        projected={int(i):project(pred['masks'][i,0],pred['boxes'][i]) for i in keep}
        out={}
        for dim,ids,shape in [(0,ids512,(512,512)),(1,ids256,(256,256))]:
            detections=[metrics(projected[int(i)][dim],ids,ref) for i,ref in zip(keep,refs)]
            instances={str(identity):metrics(projected[assignment[identity]][dim] if identity in assignment else np.zeros(shape,bool),ids,identity) for identity in sorted(targets)}
            out['native' if dim==0 else 'mapped']={'spill':sum(m['background_spill'] for m in detections),
                'cross_object':sum(m['cross_object'] for m in detections),'instances':instances,
                'detection_metrics':detections,'recovered':sum(m['iou']>=.5 for m in instances.values())}
        outcomes[name]=out
    return {'negative':not targets,'confident_detections':len(keep),'frozen_assignment':{str(k):v for k,v in assignment.items()},**outcomes}


def accept_scenes(scenes,sentinel_gate=None):
    failures=[];totals={arm:{r:0 for r in ('native','mapped')} for arm in ('baseline','candidate')}
    for scene in scenes:
        if scene['negative'] and scene['confident_detections']:failures.append('NEGATIVE_FALSE_POSITIVE')
        for raster in ('native','mapped'):
            base=scene['baseline'][raster];new=scene['candidate'][raster]
            for arm in totals:totals[arm][raster]+=scene[arm][raster]['spill']
            if new['spill']>base['spill'] or new['cross_object']>base['cross_object'] or new['recovered']<base['recovered']:failures.append('SCENE_REGRESSION')
            if set(base['instances'])!=set(new['instances']):raise ValueError('VALIDATION_OWNER_DRIFT')
            for key,b in base['instances'].items():
                c=new['instances'][key]
                if c['missing_foreground']>b['missing_foreground'] or c['iou']<b['iou'] or c['cross_object']>b['cross_object']:failures.append('INSTANCE_REGRESSION')
            if len(base['detection_metrics'])!=len(new['detection_metrics']):raise ValueError('DETECTION_COUNT_DRIFT')
            for b,c in zip(base['detection_metrics'],new['detection_metrics']):
                if c['cross_object']>b['cross_object'] or c['missing_foreground']>b['missing_foreground'] or (b['iou'] is not None and c['iou']<b['iou']):failures.append('DETECTION_REGRESSION')
    for raster in ('native','mapped'):
        if totals['baseline'][raster]<=0 or 5*totals['candidate'][raster]>4*totals['baseline'][raster]:failures.append('MATERIALITY_'+raster)
    if sentinel_gate is not None:
        selected=[s for s in scenes if s.get('scene_id') in sentinel_gate['scene_ids']]
        if sorted(s['scene_id'] for s in selected)!=sorted(sentinel_gate['scene_ids']):raise ValueError('SENTINEL_MEMBERSHIP_DRIFT')
        for raster in ('native','mapped'):
            b=sum(s['baseline'][raster]['spill'] for s in selected);c=sum(s['candidate'][raster]['spill'] for s in selected)
            if b!=sentinel_gate['baseline_spill'][raster]:raise ValueError('SENTINEL_BASELINE_DRIFT')
            if c>sentinel_gate['maximum_candidate_spill'][raster]:failures.append('SENTINEL_MATERIALITY_'+raster)
    return {'decision':'DEVELOPMENT_PASS_NOT_PRODUCTION' if not failures else 'REJECTED',
        'failures':sorted(set(failures)),'spill_totals':totals,'stage_a2':'BLOCKED_0/2','production_qualified':False}
