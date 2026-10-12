#!/usr/bin/env python3
"""Isolated 56-grid feasibility only. Disposable SGD/export; never a candidate."""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
import numpy as np
from pilot import sha, write_json, bounded

INITIAL = '/home/user/epd-v2-artifacts/mask-finetune-001/epoch_02.pt'
INITIAL_SHA = 'f3e7531c25cb20920c04283d205d7241028d7321b1ea34e2f1848415744a3b56'
DATASET = '/home/user/epd-v2-artifacts/training-expansion-002/integration/dataset.json'
DATASET_SHA = '04c1306545833921724a6248c0a8f6a85a8dffec1a31e2b3504795eaffbe9f11'
PROBE = Path('/home/user/epd-v2-artifacts/adapter-build/epd_p3_contract_probe')
PROBE_SHA = '8b942ecc6d96d13a15e538e41b5014b2cd9dc808358df17a64f144f473f1b1e6'


def configure56(model):
    """No new tensors: 28-bin ROIAlign + learned convolutions/deconv ->56 logits."""
    pool = model.roi_heads.mask_roi_pool
    if pool.output_size != (14, 14) or pool.sampling_ratio != 2:
        raise ValueError('unexpected original mask ROIAlign')
    pool.output_size = (28, 28)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(('roi_heads.mask_head.', 'roi_heads.mask_predictor.'))
    return model


def worker(output, prior=None):
    import torch, cv2, onnx, onnxruntime as ort
    from recover_original import safe_load, reconstruct
    from dataset import enumerate_records
    from original_finetune import deny_held_out
    from train import tensors
    from verify_adapter import probe, compare
    start = time.monotonic()
    if torch.cuda.is_available() or sha(DATASET) != DATASET_SHA:
        raise ValueError('resource/dataset identity drift')
    if shutil.disk_usage(output).free < 6 * 1024**3:
        raise ValueError('5GiB reserve plus1GiB disposable export budget required')
    deny_held_out(json.loads(Path(DATASET).read_text())['records'])
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    random.seed(260110); np.random.seed(260110); torch.manual_seed(260110)
    model = configure56(reconstruct(safe_load(Path(INITIAL), INITIAL_SHA)['model']))
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                               lr=.0001, momentum=.9, weight_decay=.0001)
    seen = []
    hook = model.roi_heads.mask_predictor.register_forward_hook(
        lambda module, inputs, result: seen.append(list(result.shape)))
    records = {r['scene_id']: r for r in enumerate_records(DATASET, 'train')}
    measured = []
    prior_process = None
    cases = ('scene_000', 'expansion_122', 'expansion_093', 'expansion_123')
    if prior:
        audit = json.loads((prior/'failure_audit.json').read_text())
        if (sha(prior/'process.json') != audit['process_sha256'] or
                sha(prior/'stderr.log') != audit['stderr_sha256'] or
                audit['completed_disposable_updates'] != 2 or (prior/'result.json').exists()):
            raise ValueError('failed probe identity mismatch')
        prior_process = json.loads((prior/'process.json').read_text())
        if prior_process['exit_code'] != 1 or prior_process['resource_limit_reason'] is not None:
            raise ValueError('continuation prohibited for resource/other failure')
        cases = cases[2:]  # Explicit unattempted work only; never retry a batch.
    # Same retained geometry-selected benchmark cases as #220; no prediction selection.
    for identity in cases:
        row = records[identity]; folder = Path(row['folder'])
        for name, digest in row['sha256'].items():
            if sha(folder/name) != digest: raise ValueError('training artifact drift')
        report = json.loads((folder/'capture.json').read_text())
        ids = np.fromfile(folder/'capture.json.native512.ids.u32', dtype='<u4').reshape(512,512)
        image, target = tensors(folder, report, ids, torch)
        model.train(); optimizer.zero_grad(set_to_none=True); t = time.monotonic()
        parts = model([image], [target]); loss = sum(2*v if k == 'loss_mask' else v for k,v in parts.items())
        if not torch.isfinite(loss): raise ValueError('nonfinite disposable loss')
        loss.backward(); optimizer.step()
        measured.append(dict(scene_id=identity, seconds=time.monotonic()-t,
                             instances=len(target['labels']), logits_shape=seen[-1],
                             losses={k:float(v.detach()) for k,v in parts.items()},
                             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write_json(output/('measurement_'+identity+'.json'), measured[-1])
        del loss, parts, ids, target
    write_json(output/'measurements.json', measured)
    optimizer.zero_grad(set_to_none=True); del optimizer
    model.eval(); hook.remove()

    class Interface(torch.nn.Module):
        def __init__(self, network):
            super().__init__(); self.network = network
        def forward(self, images):
            prediction = self.network([images[0]])[0]
            values = tuple(prediction[k] for k in ('boxes','labels','scores','masks'))
            self.retained = tuple(v.detach().numpy() for v in values)
            return values

    wrapper = Interface(model); export = output/'DISPOSABLE_UNQUALIFIED_56.onnx'
    t = time.monotonic()
    with torch.inference_mode():
        torch.onnx.export(wrapper, image[None], str(export), opset_version=17, dynamo=False,
            input_names=['images'], output_names=['boxes','labels','scores','masks'],
            dynamic_axes={name:{0:'N'} for name in ('boxes','labels','scores','masks')})
    export_seconds = time.monotonic()-t
    names = ('boxes','labels','scores','masks'); reference = dict(zip(names, wrapper.retained))
    np.savez_compressed(output/'DISPOSABLE_trace_outputs.npz', **reference)
    del wrapper, model
    import gc; gc.collect()
    graph = onnx.load(str(export)); onnx.checker.check_model(graph)
    # Fixed input raster guarantees fixed full-image output dimensions. Explicit metadata
    # is checked against real ORT outputs below; detection count stays symbolic.
    for value, shape in zip(graph.graph.output, [('N',4),('N',),('N',),('N',1,512,512)]):
        for dim, size in zip(value.type.tensor_type.shape.dim, shape):
            dim.ClearField('dim_param'); dim.ClearField('dim_value')
            if isinstance(size,str): dim.dim_param=size
            else: dim.dim_value=size
    onnx.checker.check_model(graph); onnx.save(graph,str(export)); del graph
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
    t=time.monotonic(); session=ort.InferenceSession(str(export),sess_options=options,providers=['CPUExecutionProvider'])
    result=dict(zip(names,session.run(list(names),{'images':image[None].numpy()})))
    ort_seconds=time.monotonic()-t
    if result['masks'].shape != (len(result['scores']),1,512,512): raise ValueError('full-image output mismatch')
    differences={k:float(np.max(np.abs(result[k]-reference[k]),initial=0)) for k in names if k!='labels'}
    tolerances=dict(boxes=1e-3,scores=1e-5,masks=1e-4)
    if not np.array_equal(result['labels'],reference['labels']) or any(differences[k]>tolerances[k] for k in differences):
        write_json(output/'parity_failure.json',differences); raise ValueError('disposable export parity failed')
    np.savez_compressed(output/'DISPOSABLE_ort_outputs.npz',**result)
    del session
    if sha(PROBE)!=PROBE_SHA: raise ValueError('EPD probe identity drift')
    png=output/'training_rgb512.png'; cv2.imwrite(str(png), cv2.cvtColor((image.numpy().transpose(1,2,0)*255).round().astype('uint8'),cv2.COLOR_RGB2BGR))
    t=time.monotonic(); cpp=probe(PROBE,export,png,output/'cpp.json')
    cpp_seconds=time.monotonic()-t
    cpp_parity=compare(cpp,result)
    max_step=max(r['seconds'] for r in measured)
    if prior_process:
        # Lost second-step timing cannot be invented. Its entire failed-process
        # duration is a conservative upper bound on that completed positive step.
        max_step=max(max_step,prior_process['wall_seconds'])
    # Conservative reused validation bound: exporting/ORT/C++ execution is not a
    # validation timing sample. Budget12 samples at max measured training-step time.
    projection=(300*max_step+24*max_step)*1.5+240
    summary=dict(status='FEASIBILITY_PASS',disposable=True,qualified_model=False,
        architecture='mask_roi_pool28 -> existing four256-channel convs -> existing stride2 deconv ->56 logits',
        initialization_sha256=INITIAL_SHA,dataset_sha256=DATASET_SHA,source_sha256=sha(__file__),
        benchmark_updates=len(measured)+(2 if prior else 0),campaign_updates=0,training_epochs=0,measurements=measured,
        prior_failed_probe=str(prior) if prior else None,step_time_upper_bound=max_step,
        export_seconds=export_seconds,ort_seconds=ort_seconds,cpp_seconds=cpp_seconds,
        export_sha256=sha(export),probe_sha256=sha(PROBE),parity_max_abs=differences,tolerances=tolerances,
        cpp_parity=cpp_parity,dynamic_N_metadata=True,full_model_zero_detection_runtime='NOT_TESTED',
        projected_one_epoch_seconds=projection,projected_two_epoch_seconds=(600+36)*max_step*1.5+240,
        projected_rss_kib=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            prior_process['child_peak_rss_kib'] if prior_process else 0)*1.5,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_seconds=time.monotonic()-start,
        held_out_pixels_opened=0,stage_a2_inference=False,accuracy_evaluated=False)
    write_json(output/'result.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','wall_seconds','peak_rss_kib','projected_one_epoch_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','worker']); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-failed-probe',type=Path)
    args=parser.parse_args()
    if args.action=='worker': worker(args.output,args.prior_failed_probe)
    else:
        args.output.mkdir()
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--output',str(args.output.resolve())]
        remaining=590
        if args.prior_failed_probe:
            remaining-=json.loads((args.prior_failed_probe/'process.json').read_text())['wall_seconds']
            command+=['--prior-failed-probe',str(args.prior_failed_probe.resolve())]
        if remaining<=0: raise ValueError('aggregate benchmark budget exhausted')
        process=bounded(command,args.output,remaining,dict(os.environ,OMP_NUM_THREADS='2'),memory_mib=6144)
        write_json(args.output/'process.json',process); print(json.dumps(process))
        sys.exit(0 if process['exit_code']==0 else 2)
