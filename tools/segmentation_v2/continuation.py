"""Hash-bound epoch-boundary continuation. No implicit pickle trust or repair."""
import json
import os
from pathlib import Path
import random
import numpy as np
from pilot import HERE,sha

SOURCE_NAMES=('campaign.py','continuation.py','train.py','pilot.py','dataset.py',
              'dataset_bindings.py','expansion.py','measure_segmented.py','segmented_policy.py','prepare_stage2.py')


def binding(dataset, recipe, policy):
    import importlib.metadata
    return dict(dataset=sha(dataset),recipe=sha(recipe),policy=sha(policy),
                versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy')},
                sources={name:sha(HERE/name) for name in SOURCE_NAMES})


def atomic_checkpoint(path, state, torch):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    if path.exists():raise FileExistsError(path)
    with temporary.open('xb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    # link publishes without overwriting another successful checkpoint.
    os.link(temporary,path);temporary.unlink()
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return dict(epoch=state.get('epoch'),path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def capture_state(model, optimizer, torch, seed, epoch, step, binding, **metadata):
    return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,
                epoch=epoch,step=step,binding=binding,smoke_only=False,
                torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),cuda_rng_state=None,**metadata)


def load_checkpoint(path, digest, torch):
    if sha(path)!=digest:raise ValueError('checkpoint hash mismatch')
    # Trusted local pickle only after external SHA and immutable evidence checks.
    return torch.load(path,map_location='cpu',weights_only=False)


def restore_state(state, model, optimizer, torch):
    required={'model','optimizer','seed','epoch','step','torch_rng_state','numpy_rng_state','python_rng_state','binding','smoke_only'}
    if not required<=set(state):raise ValueError('incomplete checkpoint state')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng_state']);np.random.set_state(state['numpy_rng_state'])
    torch.set_rng_state(state['torch_rng_state'])


def validate_progress(state, history, expected_binding, scene_ids, seed, epoch):
    from campaign import epoch_order
    if state.get('binding')!=expected_binding:raise ValueError('incompatible dataset/configuration/source binding')
    count=len(scene_ids)
    if state.get('seed')!=seed or state.get('epoch')!=epoch or state.get('step')!=epoch*count or state.get('smoke_only') is not False:
        raise ValueError('wrong checkpoint seed/epoch/step/type')
    expected=[(e,(e-1)*count+i+1,scene_ids[j]) for e in range(1,epoch+1)
              for i,j in enumerate(epoch_order(count,seed,e))]
    actual=[(r.get('epoch'),r.get('step'),r.get('scene_id')) for r in history]
    if actual!=expected:raise ValueError('missing/repeated/out-of-order training evidence')


def publish_stage(folder, policy):
    from pilot import write_json
    result=json.loads((folder/'result.json').read_text());process=json.loads((folder/'process.json').read_text())
    if process['exit_code']!=0 or result['status']!='STAGE_COMPLETE':raise ValueError('incomplete stage')
    stage=result['stage'];expected_end=policy['stages'][stage-1][1]
    if result['steps']!=expected_end*policy['training_samples']:raise ValueError('incomplete stage steps')
    files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    if any(name.endswith('.partial') for name in files):raise ValueError('partial checkpoint')
    seal=dict(schema='epd_v2_completed_stage/1',stage=stage,files=files,
              checkpoint=result['checkpoints'][-1],binding=result['binding'],
              aggregate_wall_seconds=result.get('prior_wall_seconds',0)+process['wall_seconds'])
    temporary=folder/'stage_complete.json.partial';write_json(temporary,seal)
    os.link(temporary,folder/'stage_complete.json');temporary.unlink()
    return seal


def verify_stage(folder, checkpoint_sha, expected_binding, scene_ids, seed, torch):
    folder=Path(folder);seal=json.loads((folder/'stage_complete.json').read_text())
    if seal['schema']!='epd_v2_completed_stage/1' or seal['stage']!=1 or seal['binding']!=expected_binding:
        raise ValueError('wrong stage or stale source/configuration binding')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}-{'stage_complete.json'}
    if actual!=set(seal['files']):raise ValueError('missing or unexpected stage evidence')
    for name,digest in seal['files'].items():
        if sha(folder/name)!=digest:raise ValueError('modified stage evidence: '+name)
    result=json.loads((folder/'result.json').read_text());process=json.loads((folder/'process.json').read_text())
    if process['exit_code']!=0 or result['status']!='STAGE_COMPLETE' or result['stage']!=1:
        raise ValueError('incomplete stage process')
    if result['epochs']!=4 or result['steps']!=4*len(scene_ids) or [c['epoch'] for c in result['checkpoints']]!=[1,2,3,4]:
        raise ValueError('missing completed epochs')
    checkpoint=seal['checkpoint']
    if checkpoint!=result['checkpoints'][-1] or checkpoint['sha256']!=checkpoint_sha:
        raise ValueError('wrong epoch-4 checkpoint hash')
    for c in result['checkpoints']:
        if Path(c['path'])!=folder/f"epoch_{c['epoch']:02d}.pt" or sha(c['path'])!=c['sha256']:
            raise ValueError('checkpoint location/hash mismatch')
        epoch=c['epoch'];summary=json.loads((folder/f'epoch_{epoch:02d}_summary.json').read_text())
        metrics=json.loads((folder/f'validation_{epoch:02d}/metrics.json').read_text())
        if summary['checkpoint']!=c or summary['metrics']!=metrics:raise ValueError('incomplete epoch summary')
        if metrics['epoch']!=epoch or len(metrics['raw_predictions'])!=12:raise ValueError('incomplete validation')
        for raw in metrics['raw_predictions']:
            if sha(raw['path'])!=raw['sha256']:raise ValueError('modified validation prediction')
    state=load_checkpoint(checkpoint['path'],checkpoint_sha,torch)
    if not {'model','optimizer','torch_rng_state','numpy_rng_state','python_rng_state','config','campaign'}<=set(state):
        raise ValueError('incomplete resumable checkpoint')
    history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    validate_progress(state,history,expected_binding,scene_ids,seed,4)
    if result['binding']!=expected_binding:raise ValueError('result binding mismatch')
    if seal['aggregate_wall_seconds']!=result.get('prior_wall_seconds',0)+process['wall_seconds']:
        raise ValueError('aggregate wall accounting mismatch')
    return state,result,seal
