import sys,json,time,resource,importlib.metadata,shutil,random,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import verified_rows,build_model,tensors
from pilot import sha,write_json
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--artifacts',type=Path,required=True)
parser.add_argument('--weights',type=Path,required=True);args=parser.parse_args()
root=Path(__file__).resolve().parents[2];out=args.output;out.mkdir()
art=args.artifacts;weights=args.weights
cfg=json.loads((root/'tools/segmentation_v2/train.json').read_text());policy=json.loads((root/'tools/segmentation_v2/campaign_300.json').read_text())
train=list(verified_rows(art,'train'));val=list(verified_rows(art,'validation'))
assert sha(weights)==cfg['weights_sha256']
assert sha(art)==policy['dataset_sha256']
from campaign import audit_identities
write_json(out/'identity_audit.json',audit_identities(art))
assert len(train)==300 and len(val)==12
assert sum(sum(r['class_id']==1 and bool((ids==r['renderer_id']).any()) for r in report['mapping']) for _,_,report,ids in train)==484
assert sum(not any(r['class_id']==1 and (ids==r['renderer_id']).any() for r in report['mapping']) for _,_,report,ids in train)==50
import torch,numpy as np
for n in ('torch','torchvision'): assert importlib.metadata.version(n).split('+')[0]==cfg[n]
torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);torch.use_deterministic_algorithms(True)
info=dict(cuda_available=torch.cuda.is_available(),cuda_device_count=torch.cuda.device_count(),torch_cuda_build=torch.version.cuda,
 versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','onnx','onnxruntime','opencv-python')},
 available_ram_kib=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))),
 free_disk_bytes=shutil.disk_usage(out).free,cpu_count=os.cpu_count(),
 train_scenes=len(train),validation_scenes=len(val),frozen_test_scenes_manifest_only=12,
 config_sha256=sha(root/'tools/segmentation_v2/train.json'),campaign_sha256=sha(root/'tools/segmentation_v2/campaign_300.json'),
 acquisition_sha256=sha(art),plan_sha256=sha(art),weights_sha256=sha(weights),
 prior_smoke_loaded=False,
 source_sha256={n:sha(root/'tools/segmentation_v2'/n) for n in ('campaign.py','pilot.py','train.py','dataset.py','dataset_bindings.py','expansion.py','measure_300.py')},
 initialization='fresh pinned COCO_V1',
 train_validation_files_verified=sum(len(r[0]['sha256']) for r in train+val),held_out_files_opened=0,
 optimizer_updates=0,checkpoint_saved=False)
write_json(out/'identity.json',info)
device='cuda' if info['cuda_available'] else 'cpu'
if info['cuda_available']:
 info['gpus']=[dict(name=torch.cuda.get_device_name(i),total_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
 write_json(out/'cuda.json',info)
 torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
model=build_model(cfg,weights,torch).to(device)
# One negative and two most populated/largest visible target scenes, before any updates.
def complexity(row):
 _,folder,report,ids=row
 _,target=tensors(folder,report,ids,torch)
 return (len(target['labels']),int(target['masks'].sum()))
selected=[next(r for r in train if complexity(r)[0]==0)]+sorted(train,key=complexity,reverse=True)[:2]
selected.append(min((r for r in train if complexity(r)[0]),key=lambda r:complexity(r)[1]))
timings=[]
for row in selected:
 _,folder,report,ids=row;image,target=tensors(folder,report,ids,torch)
 image=image.to(device);target={k:v.to(device) for k,v in target.items()}
 model.train();model.zero_grad(set_to_none=True)
 if device=='cuda':torch.cuda.synchronize()
 start=time.monotonic()
 losses=model([image],[target]);loss=sum(losses.values());assert torch.isfinite(loss);loss.backward()
 if device=='cuda':torch.cuda.synchronize()
 timings.append(dict(scene_id=folder.name,kind='forward_backward_without_update',seconds=time.monotonic()-start,
  objects=len(target['labels']),losses={k:float(v.detach()) for k,v in losses.items()}))
 del losses,loss
model.zero_grad(set_to_none=True);model.eval()
with torch.no_grad():
 for row in [val[0],max(val,key=complexity)]:
  _,folder,report,ids=row;image,_=tensors(folder,report,ids,torch);start=time.monotonic();pred=model([image.to(device)])[0]
  if device=='cuda':torch.cuda.synchronize()
  timings.append(dict(scene_id=folder.name,kind='validation_forward_only',seconds=time.monotonic()-start,detections=len(pred['scores'])))
  del pred
train_max=max(r['seconds'] for r in timings if r['kind'].startswith('forward'))
val_max=max(r['seconds'] for r in timings if r['kind'].startswith('validation'))
estimate=(2400*train_max+96*val_max)*policy['feasibility_safety_factor']+policy['feasibility_overhead_seconds']
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report=dict(device=device,measurements=timings,estimated_full_campaign_seconds=estimate,declared_budget_seconds=policy[device+'_timeout_seconds'],
 peak_rss_kib=peak,projected_rss_kib=peak*1.5,declared_memory_mib=policy['memory_mib'],
 feasible=estimate<=policy[device+'_timeout_seconds'] and peak*1.5<=policy['memory_mib']*1024 and info['free_disk_bytes']>policy['disk_reserve_gib']*1024**3+8*400*1024**2,
 optimizer_updates=0,checkpoint_saved=False,held_out_files_opened=0)
write_json(out/'feasibility.json',report);print(json.dumps(report),flush=True)
