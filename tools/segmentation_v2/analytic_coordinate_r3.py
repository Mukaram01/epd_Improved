#!/usr/bin/env python3
"""R3 source-bound synthetic-only admission and complete-result gate.
Core geometry/numerical benchmark is unchanged; no real scene or weights loaded.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import analytic_coordinate_contract as a
from diagnostic_shadow_capture import durable_json,supervise
from shadow_mask_paste import digest

HERE=Path(__file__).resolve().parent
OUTPUT=Path('/home/user/epd-v2-artifacts/analytic-coordinate-contract-002')
IDENTITY='epd_analytic_coordinate_R3_002'


def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def verify_bindings(policy):
    for name,sha in policy['sources_sha256'].items():
        if digest(name)!=sha:raise ValueError('SOURCE_DRIFT '+name)


def verify_review(review,policy_sha):
    if (review.get('decision')!='APPROVE_R3_SYNTHETIC_ONLY' or
        review.get('preregistration_sha256')!=policy_sha or
        not isinstance(review.get('reviewer'),str) or
        not review['reviewer'].startswith('/root/')):
        raise ValueError('INDEPENDENT_REVIEW_REQUIRED_OR_DRIFT')


def admit(policy_path,review_path,review_sha):
    policy=json.loads(policy_path.read_text())
    expected={'experiment_identity':IDENTITY,'output_path':str(OUTPUT),
        'expected_roundtrip_cases':1944,'expected_arm_measurements':7776,'expected_traces':72,
        'expected_parity':{'roi_numpy_bitwise_checks':1944,'paste_bitwise_checks':3888,
            'replicate_unit_deconv_bitwise_checks':1944,'max_abs':0.,'all_originals_before_shadows':True},
        'limits':{'cpu_threads':2,'wall_seconds':300,'rss_mib':2048}}
    if any(policy.get(k)!=v for k,v in expected.items()):raise ValueError('R3_SCOPE_DRIFT')
    for k,v in json.loads(json.dumps(a.policy_record())).items():
        if policy.get(k)!=v:raise ValueError('GEOMETRY_CONTROL_DRIFT '+k)
    verify_bindings(policy)
    if a.s.source_inventory()!=policy['source_inventory']:raise ValueError('VERSION_SOURCE_DRIFT')
    if digest(review_path)!=review_sha:raise ValueError('REVIEW_HASH_DRIFT')
    verify_review(json.loads(review_path.read_text()),digest(policy_path))
    return policy


def exclusive_output(output):
    if output.exists() or output.is_symlink():raise ValueError('OUTPUT_NOT_EXCLUSIVE')
    output.mkdir(exist_ok=False)


def case_key(row):
    return json.dumps([row[k] for k in ('shape','size','offset','placement','scale','origin','analytic_box','roi_box','bridge')],sort_keys=True)


def validate_rows(rows):
    expected={case_key({**case,'bridge':bridge}) for case in a.cases() for bridge in a.BRIDGES}
    if len(rows)!=1944 or len({case_key(r) for r in rows})!=1944 or {case_key(r) for r in rows}!=expected:
        raise ValueError('CASE_MATRIX_INCOMPLETE_OR_DRIFT')
    for row in rows:
        if not finite(row):raise ValueError('INVALID_NUMERIC_RESULT')
        if set(row['arms'])!=set(a.ARMS):raise ValueError('CASE_ARM_DRIFT')
        for m in row['arms'].values():
            if (m['foreground_delta']!=m['spill']-m['missing'] or
                m['foreground_delta']!=m['predicted_foreground']-m['reference_foreground']):
                raise ValueError('FOREGROUND_CONSERVATION_FAILURE')
            if not 0<=m['iou']<=1 or any(m[k]<0 for k in ('spill','missing','predicted_foreground','reference_foreground','boundary_crossings')):
                raise ValueError('INVALID_NUMERIC_RANGE')


def expected_trace_names():
    return {str(index)+'_'+bridge+'.npz' for index,case in enumerate(a.cases())
        if case['size']==[32,28] and case['offset']==[.25,.75] and case['placement']=='interior' for bridge in a.BRIDGES}


def trace_shapes(case):
    n=int(128*case['scale'])
    shapes={'expanded_box_float':(4,),'expanded_box_truncated':(4,),'inclusive_paste_size_hw':(2,),
        'canonical56_world_x':(56,),'canonical56_world_y':(56,),'feature_field':(3,n,n),
        'reference_centre_mask':(128,128),'reference_coverage8':(128,128),'roi_box':(4,),'analytic_box':(4,)}
    for flag in ('unaligned','aligned'):
        shapes.update({f'synthetic_logits56_{flag}':(1,1,56,56),f'bridge_world_coordinates_{flag}':(1,2,56,56),
            f'pooled_{flag}':(3,28,28),f'grid_{flag}':(28,28,2,2,2)})
    shapes.update({arm:(128,128) for arm in a.ARMS})
    return shapes


def validate_trace_arrays(values,case):
    shapes=trace_shapes(case)
    if set(values)!=set(shapes):raise ValueError('TRACE_FIELD_INCOMPLETE')
    for key,shape in shapes.items():
        if values[key].shape!=shape:raise ValueError('TRACE_SHAPE_DRIFT '+key)
        if not np.isfinite(values[key]).all():raise ValueError('INVALID_TRACE_NUMERIC')


def verify_complete(output,policy):
    # Absence of any receipt, category or trace rejects the entire result.
    if not all((output/name).is_file() for name in ('results.json','operator_parity.json','measurements.jsonl')):
        raise ValueError('INCOMPLETE_RESULT_RECEIPTS')
    result=json.loads((output/'results.json').read_text())
    parity=json.loads((output/'operator_parity.json').read_text())
    if parity!=policy['expected_parity'] or result['operator_parity']!=parity:raise ValueError('ORIGINAL_PARITY_FAILURE')
    if (result['feature_settings']!=972 or result['roundtrip_cases']!=1944 or result['arm_measurements']!=7776 or
        result['policy_sha256']!=policy['_sha256'] or result['sources_sha256']!=policy['sources_sha256']):
        raise ValueError('RESULT_IDENTITY_DRIFT')
    if not finite(result):raise ValueError('INVALID_NUMERIC_RESULT')
    if digest(output/'measurements.jsonl')!=result['measurements_sha256']:raise ValueError('MEASUREMENT_HASH_DRIFT')
    rows=[json.loads(line) for line in (output/'measurements.jsonl').read_text().splitlines()]
    validate_rows(rows)
    if a.summary(rows)!=result['summary']:raise ValueError('SUMMARY_DRIFT')
    names=expected_trace_names();actual={p.name for p in (output/'traces').iterdir()}
    listed={Path(item['path']).name for item in result['snapshots']}
    if len(result['snapshots'])!=72 or actual!=names or listed!=names:raise ValueError('TRACE_MATRIX_INCOMPLETE')
    matrix=list(a.cases())
    for item in result['snapshots']:
        path=output/'traces'/Path(item['path']).name
        if digest(path)!=item['sha256']:raise ValueError('TRACE_HASH_DRIFT')
        with np.load(path,allow_pickle=False) as values:
            case=matrix[int(path.name.split('_')[0])]
            validate_trace_arrays(values,case)
            if not np.array_equal(values['roi_box'],case['roi_box']) or not np.array_equal(values['analytic_box'],case['analytic_box']):raise ValueError('TRACE_BOX_DRIFT')
            x,y=a.canonical56_coordinates(values['roi_box'])
            if not np.array_equal(values['canonical56_world_x'],x) or not np.array_equal(values['canonical56_world_y'],y):raise ValueError('TRACE_COORDINATE_DRIFT')
            for arm in a.ARMS:
                if values[arm].shape!=(128,128) or np.any((values[arm]<0)|(values[arm]>1)):raise ValueError('TRACE_PROBABILITY_INVALID')
    verify_bindings(policy)
    return {'decision':'COMPLETE_SYNTHETIC_ONLY','roundtrip_cases':len(rows),'arm_measurements':len(rows)*4,
        'complete_traces':72,'conservation_checks':7776,'results_sha256':digest(output/'results.json'),
        'measurements_sha256':digest(output/'measurements.jsonl'),'policy_sha256':policy['_sha256'],
        'stage_a2':'BLOCKED_0/2','original_training_supervision':'UNKNOWN'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--review-sha256',required=True);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    policy=admit(args.policy,args.review,args.review_sha256);policy['_sha256']=digest(args.policy)
    if args.worker:
        try:
            a.run(args.policy,OUTPUT)
            receipt=verify_complete(OUTPUT,policy)
            durable_json(OUTPUT/'validated.json',receipt)
            return 0
        except BaseException as exc:
            durable_json(OUTPUT/'failure.json',{'reason':type(exc).__name__+': '+str(exc),'partial_evidence_preserved':True})
            raise
    exclusive_output(OUTPUT)
    durable_json(OUTPUT/'preregistration.json',json.loads(args.policy.read_text()))
    durable_json(OUTPUT/'independent_prerun_review.json',json.loads(args.review.read_text()))
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--policy',str(args.policy.resolve()),
        '--review',str(args.review.resolve()),'--review-sha256',args.review_sha256]
    process=supervise(cmd,OUTPUT,300,2048*1024)
    success=process['status']=='WORKER_COMPLETED' and (OUTPUT/'validated.json').exists()
    durable_json(OUTPUT/'completion.json',{'decision':'COMPLETE_SYNTHETIC_ONLY' if success else 'BLOCKED_FAILURE',
        'validated_sha256':digest(OUTPUT/'validated.json') if success else None,
        'supervision_sha256':digest(OUTPUT/'supervision.json'),'no_automatic_retry':True})
    print(json.dumps(process,sort_keys=True))
    return 0 if success else 2

if __name__=='__main__':raise SystemExit(main())
