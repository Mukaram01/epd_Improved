"""Read retained masks/annotations only. No model, checkpoint load or evaluation run."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/segmentation_v2'))
from dataset import verified_rows
from shadow_mask_paste import project,digest
from diagnostic_shadow_capture import scoring_ids
ROOT=Path('/home/user/epd-v2-artifacts/native-boundary-candidate-001')
policy=json.loads((ROOT/'configuration.json').read_text())['config']
validation=json.loads((ROOT/'validation.json').read_text())
assert digest(ROOT/'validation.json')==json.loads((ROOT/'result.json').read_text())['validation_sha256']
rows={r['scene_id']:(folder,report,ids) for r,folder,report,ids in verified_rows(Path(policy['dataset']['path']),'validation')}
base={p['scene_id']:p for p in policy['baseline']['predictions']}
new={p['scene_id']:p for p in validation['archives']}
result=[];bindings={str(ROOT/'validation.json'):digest(ROOT/'validation.json')}
for s in validation['scenes']:
 sid=s['scene_id'];folder,report,raw=rows[sid];ids=scoring_ids(raw,{int(r['renderer_id']) for r in report['mapping'] if r['class_id']==1})
 for rec in [base[sid],new[sid]]:
  assert digest(rec['path'])==rec['sha256'];bindings[rec['path']]=rec['sha256']
 bindings[str(folder/'capture.json.native512.ids.u32')]=digest(folder/'capture.json.native512.ids.u32')
 with np.load(base[sid]['path'],allow_pickle=False) as b,np.load(new[sid]['path'],allow_pickle=False) as c:
  for owner,index in s['frozen_assignment'].items():
   gt=ids==int(owner);bm=project(b['masks'][index,0],b['boxes'][index])[0];cm=project(c['masks'][index,0],c['boxes'][index])[0]
   lost=gt&bm&~cm;gained=gt&~bm&cm
   d=cv2.distanceTransform(np.pad(gt.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
   bins=[d<=2,(d>2)&(d<=4),d>4]
   result.append({'scene_id':sid,'owner':owner,'visible_foreground':int(gt.sum()),'lost_foreground':int(lost.sum()),'gained_foreground':int(gained.sum()),'lost_distance_bins_le2_2to4_gt4':[int((lost&z).sum()) for z in bins], 'gained_distance_bins_le2_2to4_gt4':[int((gained&z).sum()) for z in bins]})
out={'method':'retained masks only; baseline frozen assignment/consumer ROI/strict threshold unchanged; distance to visible annotation background, not amodal edge or thickness', 'instances':result,'totals':{k:sum(r[k] for r in result) for k in ['lost_foreground','gained_foreground']},'lost_distance_bins_le2_2to4_gt4':np.sum([r['lost_distance_bins_le2_2to4_gt4'] for r in result],axis=0).tolist(),'thin_feature_attribution':'UNKNOWN: local distance to background does not identify thickness','occlusion_attribution':'UNKNOWN: visible renderer IDs do not isolate occlusion edges from outer silhouette','bindings_sha256':bindings,'source_sha256':digest(__file__),'model_inference_runs':0,'training_updates':0,'stage_a2_used':False}
Path(sys.argv[1]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
