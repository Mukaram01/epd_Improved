"""Read-only R3 summary reconstruction; no geometry operators/inference executed."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1])
rows=[json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
assert len(rows)==1944
assert json.loads((root/'completion.json').read_text())['decision']=='COMPLETE_SYNTHETIC_ONLY'
assert all(m['foreground_delta']==m['spill']-m['missing'] for r in rows for m in r['arms'].values())
arms=list(rows[0]['arms'])

def aggregate(selected,arm):
    ms=[r['arms'][arm] for r in selected]
    total=sum(m['boundary_crossings'] for m in ms)
    centroid=[m['centroid_dxdy'] for m in ms if m['centroid_dxdy'] is not None]
    return {'cases':len(ms),'mean_iou':statistics.mean(m['iou'] for m in ms),
        'mean_coverage_iou8':statistics.mean(m['coverage_iou8'] for m in ms),
        **{k:sum(m[k] for m in ms) for k in ['spill','missing','foreground_delta','predicted_foreground','reference_foreground','shape_extinguished','error_outside_sampling_band']},
        'centroid_cases':len(centroid),'centroid_dxdy_mean':[statistics.mean(c[i] for c in centroid) for i in (0,1)] if centroid else None,
        'boundary_crossings':total,
        **{k:sum(m[k]*m['boundary_crossings'] for m in ms if m['boundary_crossings'])/total if total else None for k in ['boundary_abs_mean','boundary_signed_outward_mean']},
        'boundary_abs_max':max((m['boundary_abs_max'] for m in ms if m['boundary_crossings']),default=None)}

origin_scale=[];shape_origin_scale=[];marginals=[];ramps=[];loss=[]
for origin,scale in itertools.product(sorted({r['origin'] for r in rows}),sorted({r['scale'] for r in rows},reverse=True)):
    selected=[r for r in rows if r['origin']==origin and r['scale']==scale and r['shape']!='empty']
    origin_scale.append({'origin':origin,'scale':scale,'arms':{arm:aggregate(selected,arm) for arm in arms}})
    interior=[r for r in selected if r['placement']=='interior']
    shifts={}
    for flag in ['False','True']:
        expected=(.5/scale if flag=='False' else 0.) if origin=='cell_centred' else (.5 if flag=='False' else .5-.5/scale)
        shifts[flag]={'theoretical_each_axis':expected,
            **{axis:statistics.mean(r['ramp_roi_to_box_centre_error'][flag][axis] for r in interior) for axis in ['dx_mean','dy_mean']},
            'max_abs_case_mean_residual':max(abs(r['ramp_roi_to_box_centre_error'][flag][axis]-expected) for r in interior for axis in ['dx_mean','dy_mean'])}
    ramps.append({'origin':origin,'scale':scale,'interior_cases':len(interior),'flags':shifts})
    for shape in sorted({r['shape'] for r in rows}):
        subset=[r for r in rows if r['origin']==origin and r['scale']==scale and r['shape']==shape]
        shape_origin_scale.append({'origin':origin,'scale':scale,'shape':shape,'arms':{arm:aggregate(subset,arm) for arm in arms}})
        if shape!='empty':
            loss.append({'origin':origin,'scale':scale,'shape':shape,'cases':len(subset),
                'no_positive_feature_samples':sum(r['feature_positive_samples']==0 for r in subset),
                'pooled_all_nonpositive':{flag:sum(r['roi_positive_bins'][flag]==0 for r in subset) for flag in ['False','True']},
                'positive_feature_but_pooled_nonpositive':{flag:sum(r['feature_positive_samples']>0 and r['roi_positive_bins'][flag]==0 for r in subset) for flag in ['False','True']},
                'synthetic56_all_nonpositive':{flag:sum(r['synthetic56_positive_bins'][flag]==0 for r in subset) for flag in ['False','True']}})
for dim in ['shape','size','offset','placement','bridge']:
    vals={json.dumps(r[dim],sort_keys=True):r[dim] for r in rows}
    for key,value in sorted(vals.items()):
        subset=[r for r in rows if r[dim]==value and (r['shape']!='empty' or dim=='shape')]
        marginals.append({'dimension':dim,'value':value,'arms':{arm:aggregate(subset,arm) for arm in arms}})
print(json.dumps({'schema':'epd_r3_offline_summary/v1','output':str(root),'cases':len(rows),'arm_measurements':len(rows)*4,
    'origin_scale_nonempty':origin_scale,'shape_origin_scale':shape_origin_scale,'marginals':marginals,
    'interior_ramp_offsets':ramps,'sampling_loss_counts':loss,
    'uncertainty':{'sampling_band_half_width_px':2**-.5,'coverage4_8_max_abs_area_difference':max(abs(r['reference']['coverage_area4']-r['reference']['coverage_area8']) for r in rows),
        'coverage4_8_mean_abs_area_difference':statistics.mean(abs(r['reference']['coverage_area4']-r['reference']['coverage_area8']) for r in rows),
        'boundary_frame_edges_omitted':True,'quadrature_difference_is_not_error_bound':True},
    'artifact_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}},indent=2,sort_keys=True))
