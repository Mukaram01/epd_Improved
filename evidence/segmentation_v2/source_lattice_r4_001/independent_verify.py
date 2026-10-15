import json,hashlib,itertools, pathlib, math
import numpy as np
P=pathlib.Path; root=P('/home/user/epd-v2-artifacts/source-lattice-contract-001'); repo=P('/home/user/epd-v2-supervised-pilot')
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
r=json.loads((root/'results.json').read_text()); policy=json.loads((repo/'tools/segmentation_v2/source_lattice_preregistration.json').read_text()); assert r['policy']==policy
expected=list(itertools.product(((64,64),(65,67),(95,96)),('impulse','ramp_x','ramp_y','checkerboard','border','empty'),(0,.25,.5,.75),('centre','symmetric','right')))+[((800,800),'ramp_x',.25,k) for k in ('centre','symmetric','right')]
assert [(tuple(v['size']),v['pattern'],v['phase'],v['kernel']) for v in r['rows']]==expected
assert len(r['rows'])==r['cases']==219
checks=0; chains=[]; maxback=maxfpn=maxhead=0; full=[]
def finite(x):
 if isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
 elif isinstance(x,float):assert math.isfinite(x)
finite(r)
for i,row in enumerate(r['rows']):
 assert row['index']==i
 assert [t['movement'] for t in row['translations']]==[0,1,32]
 h,w=row['size']; h+=-h%32;w+=-w%32
 for t in row['translations']:
  maxback=max(maxback,t['backbone_max_abs']);maxfpn=max(maxfpn,t['fpn_max_abs'])
  for l,stride in zip(t['levels'],(4,8,16,32)):
   assert l['level']==int(math.log2(stride));assert l['stride']==stride; assert l['shape']==[h//stride,w//stride]; assert l['nominal_anchor']==.5
   assert l['comparison_integer_lag']==t['movement']//stride and l['phase_remainder']==t['movement']%stride
   if t['movement']==0:
    c=l['mask_chain'];chains.append(c);assert all(c[k] is True for k in ('roi_bitwise','deconv_bitwise','paste_bitwise','union_bitwise'));assert 0<=c['union_inactive_slots']==4*c['mapped_foreground']-c['native_foreground'];maxhead=max(maxhead,c['head_max_abs']);checks+=5
    assert np.array_equal(np.asarray(c['native512_box'],np.float32),np.asarray(c['transformed_box'],np.float32)*np.asarray([512/w,512/h]*2,np.float32));checks+=1
   else:assert 'mask_chain' not in l
   if row['size']==[800,800]:full.append(dict(kernel=row['kernel'],movement=t['movement'],level=l['level'],n=l['boundary_free_points'],maximum=l['boundary_free_max_abs'],mean=l['full_translation_mean_abs']))
 if row['size']==[800,800]:assert row['multiscale_selection']['bitwise_equal'] is True and row['multiscale_selection']['assigned_levels']==[2,3,4,5]
assert len(chains)==876 and max(maxback,maxfpn,maxhead)<=3e-6
assert r['level_assignment']==[2,3,4,5] and r['opencv_uint8_resize_max_lsb_difference']<=1
for p in root.joinpath('traces').iterdir():
 assert sha(p)==r['trace_sha256'][p.name]
 with np.load(p,allow_pickle=False) as a:
  n=64//2**int(p.stem[-1]);assert set(a.files)=={'input','C','P','paste'}
  assert {k:a[k].shape for k in a.files}=={'input':(64,64),'C':(n,n),'P':(n,n),'paste':(512,512)}
  assert all(np.isfinite(a[k]).all() for k in a.files)
  assert np.count_nonzero(a['paste']>.5)==chains[int(p.stem[-1])-2]['native_foreground']
for p,h in policy['sources_sha256'].items():assert sha(p)==h
s=json.loads((root/'supervision.json').read_text());assert s['exit_code']==0 and s['wall_seconds']<300 and s['peak_worker_rss_kib']<2097152 and s['resource_limit_reason'] is None and s['process_group_cleanup'] is True
assert (root/'stdout.log').stat().st_size==(root/'stderr.log').stat().st_size==0
assert json.loads((root/'completion.json').read_text())['supervision_sha256']==sha(root/'supervision.json')
summary={'geometry_cases':219,'graph_evaluations':657,'mask_chains':len(chains),'scalar_invariants_checked':checks,'max_backbone_abs':maxback,'max_fpn_abs':maxfpn,'max_head_abs':maxhead,'preprocessing_max_abs':max(v['max_abs'] for v in r['preprocessing']),'opencv_max_lsb':r['opencv_uint8_resize_max_lsb_difference'],'native_foreground_sum':sum(c['native_foreground'] for c in chains),'mapped_foreground_sum':sum(c['mapped_foreground'] for c in chains),'inactive_slots_sum':sum(c['union_inactive_slots'] for c in chains),'full800':full}
P('/tmp/epd-r4-independent-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='full800'},indent=2))
