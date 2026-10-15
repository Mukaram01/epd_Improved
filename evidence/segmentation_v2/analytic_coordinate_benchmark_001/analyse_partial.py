"""Offline-only failed-attempt analysis; no operators or inference are executed."""
import collections
import hashlib
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
assert len(rows)==145 and {r['shape'] for r in rows}=={'circle'}
assert not (root/'results.json').exists() and not list((root/'traces').iterdir())
assert all(m['foreground_delta']==m['spill']-m['missing'] for r in rows for m in r['arms'].values())
groups=[]
for origin in ['cell_centred','fixed_input_centre']:
    for scale in [1.,.5,.25]:
        selected=[r for r in rows if r['placement']=='interior' and r['size']==[12,10] and r['origin']==origin and r['scale']==scale]
        assert len(selected)==6 # 3 offsets x 2 bridges; not independent replicates
        mean=lambda seq:sum(seq)/len(seq)
        groups.append({'origin':origin,'scale':scale,'rows':6,
            'ramp_errors':{flag:{axis:mean([r['ramp_roi_to_box_centre_error'][flag][axis] for r in selected]) for axis in ['dx_mean','dy_mean']} for flag in ['False','True']},
            'arms':{arm:{'iou_mean':mean([r['arms'][arm]['iou'] for r in selected]),
                'boundary_abs_mean_of_case_means':mean([r['arms'][arm]['boundary_abs_mean'] for r in selected]),
                **{key:sum(r['arms'][arm][key] for r in selected) for key in ['spill','missing','foreground_delta','error_outside_sampling_band']}}
                for arm in selected[0]['arms']}})
print(json.dumps({'status':'INCOMPLETE_SINGLE_ATTEMPT_NO_RETRY','output':str(root),'rows':len(rows),'intended_rows':1944,
    'shapes':dict(collections.Counter(r['shape'] for r in rows)),'conservation_checks':len(rows)*4,
    'groups':groups,'group_caveat':'Exploratory description of a failed prefix: circle12x10 interior only; offsets/bridges are not independent replicates. No full-matrix or production conclusion.',
    'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}},indent=2,sort_keys=True))
