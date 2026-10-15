import json,pathlib,hashlib,math,sys
sys.path.insert(0,'tools/segmentation_v2')
import analytic_coordinate_r3 as gate
root=pathlib.Path('/home/user/epd-v2-artifacts/analytic-coordinate-contract-002')
p=pathlib.Path('tools/segmentation_v2/analytic_coordinate_preregistration_r3.json');policy=json.loads(p.read_text());policy['_sha256']=gate.digest(p)
receipt=gate.verify_complete(root,policy)
rows=[json.loads(x) for x in (root/'measurements.jsonl').read_text().splitlines()]
off=json.load(open('evidence/segmentation_v2/analytic_coordinate_benchmark_002/offline_summary.json'))
checks=0
for groupname in ['origin_scale_nonempty','shape_origin_scale','marginals']:
 for group in off[groupname]:
  subset=rows
  if groupname=='marginals':subset=[r for r in rows if r[group['dimension']]==group['value'] and (r['shape']!='empty' or group['dimension']=='shape')]
  else:
   subset=[r for r in rows if r['origin']==group['origin'] and r['scale']==group['scale'] and (r['shape']==group['shape'] if 'shape' in group else r['shape']!='empty')]
  for arm,target in group['arms'].items():
   ms=[r['arms'][arm] for r in subset];n=len(ms);cross=sum(m['boundary_crossings'] for m in ms);cs=[m['centroid_dxdy'] for m in ms if m['centroid_dxdy'] is not None]
   actual={'cases':n,'mean_iou':sum(m['iou'] for m in ms)/n,'mean_coverage_iou8':sum(m['coverage_iou8'] for m in ms)/n,'centroid_cases':len(cs),'centroid_dxdy_mean':[sum(c[i] for c in cs)/len(cs) for i in (0,1)] if cs else None,'boundary_crossings':cross,'boundary_abs_max':max([m['boundary_abs_max'] for m in ms if m['boundary_crossings']],default=None)}
   for k in ['spill','missing','foreground_delta','predicted_foreground','reference_foreground','shape_extinguished','error_outside_sampling_band']:actual[k]=sum(m[k] for m in ms)
   for k in ['boundary_abs_mean','boundary_signed_outward_mean']:actual[k]=sum(m[k]*m['boundary_crossings'] for m in ms if m['boundary_crossings'])/cross if cross else None
   for k,v in target.items():
    av=actual[k]
    if isinstance(v,list):assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in zip(v,av))
    elif isinstance(v,(int,float)):assert math.isclose(v,av,rel_tol=1e-12,abs_tol=1e-12),(groupname,k,v,av)
    else:assert v==av
    checks+=1
for g in off['sampling_loss_counts']:
 subset=[r for r in rows if all(r[k]==g[k] for k in ['origin','scale','shape'])]
 assert g['cases']==len(subset)
 assert g['no_positive_feature_samples']==sum(r['feature_positive_samples']==0 for r in subset)
 for flag in ['False','True']:
  assert g['pooled_all_nonpositive'][flag]==sum(r['roi_positive_bins'][flag]==0 for r in subset)
  assert g['positive_feature_but_pooled_nonpositive'][flag]==sum(r['feature_positive_samples']>0 and r['roi_positive_bins'][flag]==0 for r in subset)
  assert g['synthetic56_all_nonpositive'][flag]==sum(r['synthetic56_positive_bins'][flag]==0 for r in subset)
for g in off['interior_ramp_offsets']:
 subset=[r for r in rows if r['origin']==g['origin'] and r['scale']==g['scale'] and r['placement']=='interior' and r['shape']!='empty']
 for flag,stats in g['flags'].items():
  expect=(.5/g['scale'] if flag=='False' else 0) if g['origin']=='cell_centred' else (.5 if flag=='False' else .5-.5/g['scale'])
  assert stats['theoretical_each_axis']==expect
  residual=max(abs(r['ramp_roi_to_box_centre_error'][flag][axis]-expect) for r in subset for axis in ['dx_mean','dy_mean'])
  assert residual==stats['max_abs_case_mean_residual']
old=json.load(open('evidence/segmentation_v2/analytic_coordinate_benchmark_001/postrun_review.json'))
assert all(gate.digest(p)==h for p,h in old['artifact_sha256'].items())
assert all(gate.digest(root/p)==h for p,h in off['artifact_sha256'].items())
print(json.dumps({'receipt':receipt,'independent_aggregate_scalar_checks':checks,'old_artifacts_unchanged':True,'all_offline_manifest_hashes_valid':True}))
