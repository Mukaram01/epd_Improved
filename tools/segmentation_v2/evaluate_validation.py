#!/usr/bin/env python3
"""Bounded validation diagnostics only, never held-out evaluation or selection.

Compare genuine baseline ORT and CPU-smoke PyTorch outputs under the historical
RGB256 -> RGB512/255 contract. Does not claim EPD-main runtime compatibility.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import sys
import os
import importlib.metadata
import numpy as np
from pilot import sha, write_json, bounded
from train import build_model, validation, verified_rows

BASELINE_SHA='21362d62d816bd684f2b5c7769d0bcd1cd568af86419788fca9d095c46f2bad2'


def run(artifacts, baseline, weights, output):
    rows=list(verified_rows(artifacts,'validation'))
    if sha(baseline)!=BASELINE_SHA: raise ValueError('changed baseline ONNX')
    result=json.loads((artifacts/'training/result.json').read_text())
    checkpoint=Path(result['checkpoint'])
    if sha(checkpoint)!=result['checkpoint_sha256']: raise ValueError('changed checkpoint')
    import torch
    import onnxruntime as ort
    torch.set_num_threads(2);torch.manual_seed(260110);torch.use_deterministic_algorithms(True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['smoke_only'] is not True: raise ValueError('this command is smoke diagnostics only')
    model=build_model(state['config'],weights,torch);model.load_state_dict(state['model'],strict=True);model.eval()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(baseline),sess_options=options,providers=['CPUExecutionProvider'])
    if [o.name for o in session.get_outputs()]!=['boxes','labels','scores','masks']:
        raise ValueError('baseline output order mismatch')
    if session.get_inputs()[0].shape!=[1,3,512,512]: raise ValueError('baseline input shape mismatch')
    report=dict(schema='epd_v2_validation_diagnostic/1',split='validation',model_selection=False,
                held_out_evaluated=False,epd_main_compatible=False,
                confidence_threshold=.8,mask_threshold_strict_gt=.5,
                baseline_sha256=BASELINE_SHA,checkpoint_sha256=sha(checkpoint),
                preprocessing='RGB256 INTER_LINEAR512 CHW float32 /255; historical contract',
                source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','onnxruntime','numpy','opencv-python')},
                strict_emd_association='NOT_EXECUTED: current EPD model-profile adapter is missing',
                models={})
    class Recorder(torch.nn.Module):
        def __init__(self,name): super().__init__();self.name=name;self.calls=[]
        def forward(self,images):
            start=time.monotonic()
            if self.name=='baseline':
                values=session.run(None,{session.get_inputs()[0].name:images[0].numpy()[None]})
                pred={k:torch.from_numpy(v) for k,v in zip(('boxes','labels','scores','masks'),values)}
            else: pred=model(images)[0]
            n=len(pred['scores'])
            if pred['boxes'].shape!=(n,4) or pred['labels'].shape!=(n,) or pred['masks'].shape!=(n,1,512,512):
                raise ValueError('actual model output contract mismatch')
            if any(not torch.isfinite(v).all() for v in pred.values()): raise ValueError('nonfinite prediction')
            if pred['labels'].dtype!=torch.int64 or (pred['labels']!=1).any():
                raise ValueError('expected cube-only foreground class_id=1')
            keep=pred['scores']>=.8
            identity=rows[len(self.calls)][0]['scene_id']
            path=output/f'{self.name}_{identity}.npz'
            # Retain all raw box/class/confidence outputs, and every mask used
            # in metrics. No masks or detections are substituted or adjusted.
            np.savez_compressed(path,boxes=pred['boxes'].numpy(),labels=pred['labels'].numpy(),
                                scores=pred['scores'].numpy(),masks_confident=pred['masks'][keep].numpy())
            self.calls.append(dict(scene_id=identity,predictions=n,confident=int(keep.sum()),
                                   wall_seconds=time.monotonic()-start,artifact=path.name,sha256=sha(path)))
            return [pred]
    for name in ('baseline','v2_cpu_smoke'):
        recorder=Recorder(name)
        report['models'][name]=validation(recorder,rows,'cpu',torch)
        report['models'][name]['inferences']=recorder.calls
        report['models'][name]['runtime_seconds']=sum(c['wall_seconds'] for c in recorder.calls)
    report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['improvement_claim']=False
    write_json(output/'result.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker: run(a.artifacts,a.baseline,a.weights,a.output)
    else:
        a.output.mkdir()
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--artifacts',str(a.artifacts),
                 '--baseline',str(a.baseline),'--weights',str(a.weights),'--output',str(a.output)]
        result=bounded(command,a.output,180,dict(os.environ,OMP_NUM_THREADS='2'))
        write_json(a.output/'process.json',result)
        raise SystemExit(result['exit_code'])
