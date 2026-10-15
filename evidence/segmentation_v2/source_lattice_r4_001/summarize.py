"""Offline-only summary of the existing single R4 execution; no operators run."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
root=Path(sys.argv[1]);raw=json.loads((root/'results.json').read_text());rows=raw['rows']
assert len(rows)==219
assert json.loads((root/'completion.json').read_text())['decision']=='COMPLETE_SYNTHETIC_ONLY'
def finite(x):
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return not isinstance(x,float) or math.isfinite(x)
assert finite(raw)
assert len({(tuple(r['size']),r['pattern'],r['phase'],r['kernel']) for r in rows})==219
translations=[t for r in rows for t in r['translations']]
chains=[l['mask_chain'] for r in rows for l in r['translations'][0]['levels']]
assert len(translations)==657 and len(chains)==876
assert all(all(c[k] for k in ['roi_bitwise','deconv_bitwise','paste_bitwise','union_bitwise']) for c in chains)
full=[]
for r in rows:
 if r['size']==[800,800]:
  for t in r['translations']:
   for l in t['levels']:
    full.append({'kernel':r['kernel'],'movement':t['movement'],'level':l['level'],
      'boundary_free_points':l['boundary_free_points'],'boundary_free_max_abs':l['boundary_free_max_abs'],
      'full_frame_mean_abs':l['full_translation_mean_abs']})
aliasing=[]
for kernel in ['centre','symmetric','right']:
 for pattern in ['impulse','checkerboard','empty']:
  z=[r for r in rows if r['kernel']==kernel and r['pattern']==pattern]
  aliasing.append({'kernel':kernel,'pattern':pattern,'cases':len(z),
    'zero_energy_P2_to_P5':[sum(r['translations'][0]['levels'][l]['energy']==0 for r in z) for l in range(4)]})
print(json.dumps({'schema':'epd_r4_offline_summary/v1','output':str(root),'cases':len(rows),'graphs':len(translations),'chains':len(chains),
 'max_absolute_differences':{'backbone':max(t['backbone_max_abs'] for t in translations),'fpn':max(t['fpn_max_abs'] for t in translations),
 'head':max(c['head_max_abs'] for c in chains),'preprocessing':max(p['max_abs'] for p in raw['preprocessing']),'opencv_lsb':raw['opencv_uint8_resize_max_lsb_difference']},
 'bitwise_chain_receipts':{k:sum(c[k] for c in chains) for k in ['roi_bitwise','deconv_bitwise','paste_bitwise','union_bitwise']},
 'foreground_counts':{'native':sum(c['native_foreground'] for c in chains),'mapped':sum(c['mapped_foreground'] for c in chains),'inactive_slots':sum(c['union_inactive_slots'] for c in chains)},
 'full800_translation':full,'aliasing':aliasing,'preprocessing':raw['preprocessing'],
 'supervision':json.loads((root/'supervision.json').read_text()),'worker_seconds':raw['worker_seconds'],
 'artifact_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}},indent=2,sort_keys=True))
