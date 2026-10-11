#!/usr/bin/env python3
"""Verify sealed Stage1 and prepare CPU Stage2 admission; never trains/infers."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import random
import numpy as np
import shutil
import torch
from pilot import HERE,sha,write_json
from campaign import campaign_rows
from continuation import binding,verify_stage,restore_state,SOURCE_NAMES
from train import build_model


def prepare(stage1,digest,artifacts,weights,output):
    policy_path=HERE/'campaign_segmented_300.json';recipe_path=HERE/'train.json'
    policy=json.loads(policy_path.read_text());cfg=json.loads(recipe_path.read_text())
    torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True)
    train,_=campaign_rows(artifacts,policy)
    bindings=binding(artifacts,recipe_path,policy_path)
    state,result,seal=verify_stage(stage1,digest,bindings,[r['scene_id'] for r,_,_,_ in train],cfg['seed'],torch)
    if sha(weights)!=cfg['weights_sha256']:raise ValueError('initialization hash')
    model=build_model(cfg,weights,torch)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    restore_state(state,model,optimizer,torch)
    if any(not torch.equal(value,model.state_dict()[name]) for name,value in state['model'].items()):
        raise ValueError('model restore mismatch')
    # state_dict parameter keys/order must be identical to the saved optimizer.
    if optimizer.state_dict()['param_groups']!=state['optimizer']['param_groups']:
        raise ValueError('optimizer restore mismatch')
    for key,values in state['optimizer']['state'].items():
        for name,value in values.items():
            restored=optimizer.state_dict()['state'][key][name]
            if torch.is_tensor(value):
                if not torch.equal(value,restored):raise ValueError('optimizer tensor restore mismatch')
            elif value!=restored:raise ValueError('optimizer value restore mismatch')
    if not torch.equal(torch.get_rng_state(),state['torch_rng_state']) or random.getstate()!=state['python_rng_state']:
        raise ValueError('RNG restoration mismatch')
    actual=np.random.get_state();expected=state['numpy_rng_state']
    if actual[0]!=expected[0] or not np.array_equal(actual[1],expected[1]) or actual[2:]!=expected[2:]:
        raise ValueError('NumPy RNG restoration mismatch')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    admission=result['stage2_admission'];free=shutil.disk_usage(output.parent).free
    report=dict(device='cpu',stage=2,optimizer_updates=0,checkpoint_saved=False,
        estimated_stage_seconds=admission['estimated_remaining_seconds'],
        estimated_aggregate_seconds=seal['aggregate_wall_seconds']+admission['estimated_remaining_seconds'],
        peak_rss_kib=peak,projected_rss_kib=max(peak,json.loads((stage1/'process.json').read_text())['sampled_process_peak_rss_kib'])*1.5,
        resumed_checkpoint_sha256=digest,source_stage_seal_sha256=sha(stage1/'stage_complete.json'),
        model_restore_verified=True,optimizer_full_state_restore_verified=True,rng_restore_verified=True,
        held_out_files_opened=0,historical_regression_evaluated=False,
        remaining_aggregate_seconds=policy['aggregate_timeout_seconds']-seal['aggregate_wall_seconds'])
    report['feasible']=(admission['admitted'] and report['estimated_aggregate_seconds']<=policy['aggregate_timeout_seconds']
        and report['projected_rss_kib']<=policy['memory_mib']*1024
        and free>policy['disk_reserve_gib']*1024**3+4*400*1024**2)
    output.mkdir()
    write_json(output/'identity.json',dict(cuda_available=False,campaign_sha256=sha(policy_path),config_sha256=sha(recipe_path),
        acquisition_sha256=sha(artifacts),source_sha256={name:sha(HERE/name) for name in SOURCE_NAMES},
        versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy')},weights_sha256=sha(weights)))
    write_json(output/'feasibility.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1',type=Path,required=True);parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.stage1,args.checkpoint_sha256,args.artifacts,args.weights,args.output)),flush=True)
