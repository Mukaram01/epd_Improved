"""Bind integrated records to frozen membership and acquisition metadata only."""
import json
from pathlib import Path
from pilot import scene_plan,sha


def validate_metadata(data):
    from expansion import expansion_plan
    old_specs=scene_plan()['scenes'];new_specs=expansion_plan()['scenes']
    expected={(s['scene_id'],s['seed'],s['split']) for s in old_specs+new_specs}
    records=data['records']
    if {(r['scene_id'],r['seed'],r['split']) for r in records}!=expected:raise ValueError('frozen dataset membership')
    pilot=Path(data['pilot_root']);extra=Path(data['expansion_root'])
    path=pilot/'acquisition.json'
    if sha(path)!=data['pilot_manifest_sha256']['acquisition.json']:raise ValueError('pilot acquisition identity')
    original=json.loads(path.read_text())
    if original['status']!='ACQUIRED' or len(original['scenes'])!=48:raise ValueError('pilot acquisition incomplete')
    if {(r['scene_id'],r['seed'],r['split']) for r in original['scenes']}!={(s['scene_id'],s['seed'],s['split']) for s in old_specs}:
        raise ValueError('pilot source membership')
    old={r['scene_id']:r for r in original['scenes']}
    path=extra/'acquisition.json'
    if sha(path)!=data['expansion_acquisition_sha256']:raise ValueError('expansion acquisition identity')
    acquisition=json.loads(path.read_text())
    expected_attempts={f'{i:03d}.json' for i in range(276)}
    if acquisition['status']!='ACQUIRED' or set(acquisition['attempt_manifest_sha256'])!=expected_attempts:raise ValueError('expansion incomplete')
    new={}
    for i,spec in enumerate(new_specs):
        name=f'{i:03d}.json';path=extra/'attempts'/name
        if sha(path)!=acquisition['attempt_manifest_sha256'][name]:raise ValueError('attempt identity')
        row=json.loads(path.read_text())
        if row['status']!='ACQUIRED' or (row['scene_id'],row['seed'],row['split'])!=(spec['scene_id'],spec['seed'],spec['split']):raise ValueError('attempt membership')
        new[row['scene_id']]=row
    for row in records:
        identity=row['scene_id'];source=old[identity] if identity in old else new[identity]
        root=pilot if identity in old else extra;origin='pilot_001' if identity in old else 'expansion_002'
        if Path(row['folder'])!=root/'dataset'/identity or row['origin']!=origin or row['sha256']!=source['sha256']:
            raise ValueError('record folder/hash/origin substitution')
