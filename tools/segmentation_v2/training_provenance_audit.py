#!/usr/bin/env python3
"""Read-only training provenance forensics. Never trains, constructs a model,
infers, repairs a supervisor record or changes the shadow experiment gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAMPAIGN=Path('/home/user/epd-v2-artifacts/mask-resolution-campaign-001')
INITIAL=Path('/home/user/epd-v2-artifacts/mask-finetune-001/epoch_02.pt')
DATASET=Path('/home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json')
PINS={'result.json':'ecc693d2eb5222a63dbeea4c3270d14b9b0e8492b3c80e5266fe2c6feacab133',
      'binding.json':'bedd3d0c7589b1ac9a85c8b1aadd200d38e958dd89a261ee38f4626388811530',
      'history.jsonl':'2c10eeb6ef97d0bd34bf9863e759af7d3aed4d9168ed74266b2e4b1afe4deb86',
      'epoch_01.pt':'bf3ac58116a3a692806ed20b25fb5414277e76da605cf0f60e2e5a23656bfbfe'}
# result/binding/checkpoint are independently pinned by prior #225/#226;
# history pin is corroborated below by embedded checkpoint metadata.


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def require_hash(path,expected):
    if digest(path)!=expected:raise ValueError('INPUT_SHA: '+str(path))


def read(path):return json.loads(Path(path).read_text())


def check_history(history,expected_order):
    if (len(history)!=len(expected_order) or len(set(expected_order))!=len(expected_order) or
            [r['step'] for r in history]!=list(range(1,len(expected_order)+1)) or
            [r['scene_id'] for r in history]!=expected_order or any(r['epoch']!=1 for r in history)):
        raise ValueError('HISTORY_SEQUENCE: count/order/epoch drift')
    for row in history:
        values=[v for k,v in row.items() if k in ('seconds','loss') or k.startswith('loss_')]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('HISTORY_VALUES: nonfinite/negative')
        weighted=sum(v*(2 if k=='loss_mask' else 1) for k,v in row.items() if k.startswith('loss_'))
        if not math.isclose(row['loss'],weighted,rel_tol=1e-5,abs_tol=1e-6):
            raise ValueError('HISTORY_VALUES: weighted loss mismatch')
    return {'updates':len(history),'unique_scenes':len(set(expected_order)),
            'summed_update_seconds':sum(r['seconds'] for r in history),
            'min_update_seconds':min(r['seconds'] for r in history),
            'max_update_seconds':max(r['seconds'] for r in history),'weighted_losses_consistent':True}


def compare_states(initial,final):
    import torch
    if set(initial)!=set(final):raise ValueError('FROZEN_STATE: tensor inventory drift')
    frozen=changed=mask=0
    for key,left in initial.items():
        right=final[key]
        if (not torch.is_tensor(left) or not torch.is_tensor(right) or left.dtype!=right.dtype or
                left.shape!=right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all()):
            raise ValueError('FROZEN_STATE: shape/dtype/finite drift '+key)
        equal=torch.equal(left,right)
        if key.startswith(('roi_heads.mask_head.','roi_heads.mask_predictor.')):
            mask+=1;changed+=not equal
        else:
            if not equal:raise ValueError('FROZEN_STATE: changed parameter or buffer '+key)
            frozen+=1
    return {'model_tensors':len(initial),'frozen_tensors_equal':frozen,'mask_tensors':mask,
            'mask_tensors_changed':changed,'comparison':'torch.equal including every non-mask buffer'}


def check_checkpoint_metadata(state,binding,policy,history_sha,validation_sha):
    expected={'binding':binding,'policy':policy,'epoch':1,'step':300,'seed':260110,
              'architecture':'mask_roi28_logits56','mask_roi_output_size':28,'mask_logits_size':56,
              'history_sha256':history_sha,'validation_sha256':validation_sha}
    for key,value in expected.items():
        if state.get(key)!=value:raise ValueError('CHECKPOINT_BINDING: '+key)


def matches_parent(record,campaign):
    cmd=record.get('command',[])
    if not isinstance(cmd,list):return False
    try:
        return (len(cmd)>3 and Path(cmd[1]).name=='mask_resolution_campaign.py' and cmd[2]=='worker' and
                Path(cmd[cmd.index('--output')+1]).resolve()==campaign.resolve())
    except (ValueError,IndexError,TypeError):return False


def provenance_matrix(parent_present=False):
    return {
      'worker_completion':{'status':'VERIFIED','scope':'hash-bound published 300-update checkpoint/validation/result; not process exit'},
      'frozen_non_mask_state':{'status':'VERIFIED','scope':'saved tensor equality only; not continuous runtime attestation'},
      'parent_completion':{'status':'PARTIALLY VERIFIED' if parent_present else 'MISSING',
                           'scope':'unreviewed candidate record only' if parent_present else 'no independently bound wait/exit record'},
      'parent_observed_exit_code':{'status':'MISSING','value':None},
      'parent_observed_wall_seconds':{'status':'MISSING','value':None},
      'parent_observed_peak_rss_kib':{'status':'MISSING','value':None},
      'parent_resource_limit_reason':{'status':'MISSING','value':None},
      'worker_runtime_resource_claims':{'status':'PARTIALLY VERIFIED','scope':'worker getrusage/monotonic and filesystem corroboration; no independent supervisor'},
      'continuous_budget_enforcement':{'status':'MISSING','scope':'configured limits and runner source do not prove parent stayed alive'},
      'historic_proc_wait_and_vm_samples':{'status':'UNRECOVERABLE','scope':'from retained campaign artifacts/current proc after reboot; unknown external backups remain possible'},
    }


def inventory(campaign):
    result={}
    for path in sorted(campaign.rglob('*')):
        if not path.is_file():continue
        st=path.stat()
        result[str(path.relative_to(campaign))]={'sha256':digest(path),'bytes':st.st_size,
            'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,'inode':st.st_ino,
            'device':st.st_dev,'mode':oct(st.st_mode&0o777),'uid':st.st_uid,'gid':st.st_gid}
    return result


def process_candidates(artifacts,campaign):
    candidates=[];checked=[]
    # Do not open acquisition/held-out scene records, pixels or other checkouts.
    for folder in sorted(artifacts.iterdir()):
        if not folder.is_dir() or folder.name.startswith(('pilot','training-expansion','adapter')):continue
        for directory,dirs,files in os.walk(folder):
            dirs[:]=[d for d in dirs if d not in ('dataset','capture','executed_sources','validation_01')]
            for name in sorted(files):
                if 'process' not in name or not name.endswith('.json'):continue
                path=Path(directory)/name
                try:record=read(path)
                except (ValueError,OSError):continue
                checked.append({'path':str(path),'sha256':digest(path)})
                if matches_parent(record,campaign):candidates.append({'path':str(path),'sha256':digest(path),'record':record})
    return {'scope':'non-dataset non-adapter artifact process JSONs; no external backup guarantee',
            'checked':checked,'matching_training_parent_candidates':candidates}


def audit(campaign,output):
    start=time.monotonic()
    if output.exists() or output.is_symlink():raise ValueError('OUTPUT_NOT_EXCLUSIVE')
    before=inventory(campaign)
    for name,pin in PINS.items():require_hash(campaign/name,pin)
    binding=read(campaign/'binding.json');cfg=read(campaign/'configuration.json')
    admitted=read(campaign/'admission.json');result=read(campaign/'result.json')
    require_hash(INITIAL,binding['checkpoint']);require_hash(DATASET,binding['dataset'])
    if cfg['source_binding']!=binding or cfg['policy']!=binding['policy']:
        raise ValueError('CONFIG_BINDING_DRIFT')
    if (admitted['training_updates_authorized']!=300 or admitted['training_epochs_authorized']!=1 or
            admitted['checkpoint_sha256']!=binding['checkpoint'] or admitted['dataset_sha256']!=binding['dataset'] or
            admitted['memory_limit_mib']!=6144 or admitted['wall_limit_seconds']!=10800):
        raise ValueError('ADMISSION_BINDING_DRIFT')
    feasibility={}
    for name,pin in admitted['source_evidence_sha256'].items():
        path=ROOT/'evidence/segmentation_v2/mask_resolution_001'/name
        require_hash(path,pin);feasibility[name]={'sha256':pin,'record':read(path)}
    source_checks={}
    for name,pin in binding['sources'].items():
        require_hash(campaign/'executed_sources'/name,pin)
        source_checks[name]={'snapshot_sha256':pin,'current_sha256':digest(HERE/name)}
    records=read(DATASET)['records'];train=[r for r in records if r['split']=='train']
    order=list(range(len(train)));random.Random(260110).shuffle(order)
    history=[json.loads(line) for line in (campaign/'history.jsonl').read_text().splitlines()]
    history_stats=check_history(history,[train[i]['scene_id'] for i in order])
    if history_stats['updates']!=300:raise ValueError('HISTORY_SEQUENCE: expected300')
    metrics=read(campaign/'validation_01/metrics.json')
    if result['epoch_metrics']!=[metrics] or result['steps']!=300 or result['epochs']!=1:
        raise ValueError('RESULT_METRICS_DRIFT')
    if (not metrics['eligible'] or result['status']!='VALIDATION_PASS' or metrics['positives']!=22 or
            metrics['recovered']<18 or metrics['mean_iou']<=.685890 or metrics['false_positives']>1 or
            metrics['background_spill']>=1915 or metrics['contract_violations']!=0):
        raise ValueError('VALIDATION_GATE_DRIFT')
    raw=metrics['raw_predictions']
    if [r['scene_id'] for r in raw]!=[f'scene_{i:03}' for i in range(24,36)]:
        raise ValueError('VALIDATION_SCENE_DRIFT')
    for row in raw:
        path=campaign/'validation_01'/(row['scene_id']+'.npz')
        if Path(row['path']).resolve()!=path.resolve():raise ValueError('VALIDATION_PATH_DRIFT')
        require_hash(path,row['sha256'])
    import torch
    torch.set_num_threads(2)
    # Deserialize tensors only, never reconstruct a network or call forward.
    state=torch.load(campaign/'epoch_01.pt',map_location='cpu',weights_only=True)
    initial=torch.load(INITIAL,map_location='cpu',weights_only=True)
    check_checkpoint_metadata(state,binding,binding['policy'],digest(campaign/'history.jsonl'),
                              digest(campaign/'validation_01/metrics.json'))
    frozen=compare_states(initial['model'],state['model'])
    if frozen['model_tensors']!=307 or frozen['mask_tensors']!=12:
        raise ValueError('MODEL_TENSOR_INVENTORY')
    if set(cfg['trainable_names'])!={k for k in state['model'] if k.startswith(('roi_heads.mask_head.','roi_heads.mask_predictor.'))}:
        raise ValueError('TRAINABLE_NAMES_DRIFT')
    selected=result['selected']
    if (selected!=result['checkpoints'][0] or selected['sha256']!=PINS['epoch_01.pt'] or
            selected['bytes']!=before['epoch_01.pt']['bytes'] or Path(selected['path']).resolve()!=(campaign/'epoch_01.pt').resolve()):
        raise ValueError('SELECTED_CHECKPOINT_DRIFT')
    optimizer=state['optimizer'];groups=optimizer['param_groups']
    if (len(groups)!=1 or len(groups[0]['params'])!=12 or len(optimizer['state'])!=12 or
            any(groups[0][k]!=binding['policy'][v] for k,v in [('lr','learning_rate'),('momentum','momentum'),('weight_decay','weight_decay')])):
        raise ValueError('OPTIMIZER_BINDING_DRIFT')
    stdout=[json.loads(line) for line in (campaign/'stdout.log').read_text().splitlines()]
    if [r.get('completed_updates') for r in stdout[:-1]]!=list(range(50,301,50)):
        raise ValueError('WORKER_STDOUT_DRIFT')
    last=stdout[-1]
    if any(last[k]!=result[k] for k in ('status','steps','epochs','selected','training_validation_seconds')):
        raise ValueError('WORKER_STDOUT_DRIFT')
    search=process_candidates(campaign.parent,campaign)
    after=inventory(campaign)
    if before!=after:raise ValueError('SOURCE_MODIFIED_DURING_FORENSICS')
    elapsed_fs=(before['result.json']['mtime_ns']-before['admission.json']['mtime_ns'])/1e9
    report={'schema':'epd_training_provenance_forensics/v1','status':'BLOCKED_MISSING_TRAINING_PROVENANCE',
        'gate_changed':False,'matrix':provenance_matrix(bool(search['matching_training_parent_candidates'])),
        'campaign_inventory':before,'original_source_checks':source_checks,'feasibility_records':feasibility,
        'checkpoint_bindings_verified':True,'frozen_saved_state':frozen,'history':history_stats,
        'validation':{k:metrics[k] for k in ('recovered','positives','mean_iou','background_spill','false_positives','missing_visible','cross_object','eligible')},
        'validation_archives_hash_verified':len(raw),'optimizer_parameter_states':len(optimizer['state']),
        'worker_evidence':{'reported_training_validation_seconds':result['training_validation_seconds'],
            'reported_peak_rss_kib':result['peak_rss_kib'],'stdout_completed_updates':300,
            'stderr_bytes':before['stderr.log']['bytes'],'validation_forward_seconds_sum':sum(r['seconds'] for r in raw),
            'filesystem_admission_to_result_seconds':elapsed_fs,
            'filesystem_timing_is_not_monotonic_supervisor_timing':True},
        'process_record_search':search,'artifact_data_and_metadata_unchanged':True,
        'tensor_deserialization_only':True,'model_constructed':False,'training_or_inference_calls':0,
        'held_out_scene_files_opened':0,'worker_success_does_not_imply_parent_exit':True,
        'forensic_source_sha256':digest(__file__),'audit_seconds':time.monotonic()-start,
        'forensic_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    output.mkdir(exist_ok=False)
    (output/'provenance_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=CAMPAIGN)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:report=audit(args.campaign,args.output)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'BLOCKED_INPUT_DRIFT','reason':str(exc)}),file=sys.stderr);return 2
    print(json.dumps({k:report[k] for k in ('status','history','frozen_saved_state','worker_evidence','audit_seconds','forensic_peak_rss_kib')},sort_keys=True))
    return 0  # forensic audit completed; experiment admission remains blocked


if __name__=='__main__':raise SystemExit(main())
