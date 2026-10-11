#!/usr/bin/env python3
"""Offline contract checks. No training, acquisition, held-out access or model download.

The probe links EPD's real OrtBase/P3OrtBase. Generated ONNX graphs are contract
TEST FIXTURES, never trained/exported candidates. Historical replay uses one
validation scene; its RGB256->512 step is explicit, outside the exact-raster adapter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import resource
import time
import numpy as np
import cv2

BASELINE_SHA = '21362d62d816bd684f2b5c7769d0bcd1cd568af86419788fca9d095c46f2bad2'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pinned(path, digest):
    if sha(path) != digest:
        raise ValueError(f'SHA256 mismatch: {path}')


def probe(binary, model, image, output, mode='rgb01', classes=2, consumer=None):
    start = time.monotonic()
    memory_file=output.with_suffix('.memory.txt')
    command = ['/usr/bin/time','-v','-o',str(memory_file),str(binary),str(model),str(image),
               mode,str(classes),str(output)]+([consumer] if consumer else [])
    def limits():
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(45,50))
    process = subprocess.Popen(command,env=dict(os.environ,EPD_EXECUTION_BACKEND='cpu',
                               OMP_NUM_THREADS='2'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,start_new_session=True,preexec_fn=limits)
    timed_out=False
    try:
        stdout,stderr=process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(process.pid,signal.SIGKILL)
        stdout,stderr=process.communicate()
    output.with_suffix('.process.json').write_text(json.dumps(dict(
        command=command,exit_code=process.returncode,seconds=time.monotonic()-start,
        timed_out=timed_out,limits=dict(wall_seconds=60,cpu_seconds=45,address_space_bytes=4*1024**3),
        stdout=stdout,stderr=stderr),indent=2)+'\n')
    if timed_out: raise TimeoutError('bounded C++ probe timed out; process group killed, evidence retained')
    if process.returncode: raise subprocess.CalledProcessError(process.returncode,command,stdout,stderr)
    return json.loads(output.read_text())


def compare(actual, expected, full=True, tolerance=1e-5):
    keep = expected['scores'] >= .8
    reference_boxes = expected['boxes'][keep]
    # C++ detection infer intentionally leaves the user threshold to its caller.
    detections = [d for d in actual['detections'] if d['score'] >= .8]
    if len(detections) != len(reference_boxes):
        raise AssertionError('confident detection count mismatch')
    masks = expected['masks_confident'] if 'masks_confident' in expected else expected['masks'][keep]
    differences = dict(box_max_abs=0., score_max_abs=0., mask_max_abs=0., labels_exact=True,
                       count=len(detections), roi_masks_bitwise_exact=True)
    for i, detection in enumerate(detections):
        mask_full = masks[i, 0]
        height, width = mask_full.shape
        # Independent NumPy reference: clipped xyxy truncated, half-open slice.
        box = np.clip(reference_boxes[i], [0,0,0,0], [width,height,width,height]).astype(np.int64)
        differences['box_max_abs'] = max(differences['box_max_abs'], float(np.max(np.abs(box-detection['box']))))
        differences['score_max_abs'] = max(differences['score_max_abs'], abs(float(expected['scores'][keep][i])-detection['score']))
        if int(expected['labels'][keep][i]) != detection['label']:
            raise AssertionError('label mismatch')
        reference = mask_full[box[1]:box[3],box[0]:box[2]] if full else mask_full
        actual_mask = np.fromfile(detection['mask_path'], np.float32).reshape(
            detection['mask_height'], detection['mask_width'])
        if actual_mask.shape != reference.shape:
            raise AssertionError('ROI dimensions mismatch')
        differences['mask_max_abs'] = max(differences['mask_max_abs'], float(np.max(np.abs(actual_mask-reference),initial=0)))
        differences['roi_masks_bitwise_exact'] &= actual_mask.tobytes()==reference.tobytes()
    if any(differences[k] > tolerance for k in ('box_max_abs','score_max_abs','mask_max_abs')):
        raise AssertionError(differences)
    return differences


def constant_model(path, boxes, labels, scores, masks):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as t
    height, width = masks.shape[-2:]
    outputs = [('boxes',boxes),('labels',labels),('scores',scores),('masks',masks)]
    graph = h.make_graph([h.make_node('Constant', [], [name], value=nh.from_array(value))
                          for name,value in outputs], 'retained_prediction_contract_fixture',
                         [h.make_tensor_value_info('images',t.FLOAT,[1,3,height,width])],
                         [h.make_tensor_value_info(name,t.INT64 if name=='labels' else t.FLOAT,list(value.shape))
                          for name,value in outputs])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',11)],ir_version=7)
    onnx.checker.check_model(model); onnx.save(model,path)


def dynamic_model(path):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as t
    values = {'half':np.array(.5,np.float32),'channel':np.array([0],np.int64),
              'base_boxes':np.array([[0,0,3,2]],np.float32),
              'base_labels':np.array([1],np.int64),'base_scores':np.array([.8],np.float32)}
    nodes=[h.make_node('ReduceMean',['images'],['mean'],keepdims=0),
           h.make_node('Greater',['mean','half'],['positive']),
           h.make_node('Unsqueeze',['positive'],['condition'],axes=[0]),
           h.make_node('Gather',['images','channel'],['base_masks'],axis=1)]
    for name in ('boxes','labels','scores','masks'):
        nodes.append(h.make_node('Compress',['base_'+name,'condition'],[name],axis=0))
    graph=h.make_graph(nodes,'dynamic_zero_contract_fixture',
      [h.make_tensor_value_info('images',t.FLOAT,[1,3,2,3])],
      [h.make_tensor_value_info('boxes',t.FLOAT,['N',4]),
       h.make_tensor_value_info('labels',t.INT64,['N']),
       h.make_tensor_value_info('scores',t.FLOAT,['N']),
       h.make_tensor_value_info('masks',t.FLOAT,['N',1,2,3])],
      [nh.from_array(value,name) for name,value in values.items()])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',11)],ir_version=7)
    onnx.checker.check_model(model); onnx.save(model,path)


def fixtures(args):
    # Pin references against the previously recorded validation run manifest.
    recorded=json.loads((args.validation/'result.json').read_text())
    cases=[]
    for name in ('baseline','v2_cpu_smoke'):
        for row in recorded['models'][name]['inferences']:
            identity=row['scene_id']
            if identity not in {f'scene_{i:03d}' for i in range(24,36)}:
                raise ValueError('only frozen validation fixtures allowed')
            reference=args.validation/row['artifact']; pinned(reference,row['sha256'])
            with np.load(reference,allow_pickle=False) as data:
                keep=data['scores']>=.8
                model=args.output/(name+'_'+identity+'.fixture.onnx')
                constant_model(model,data['boxes'][keep],data['labels'][keep],data['scores'][keep],data['masks_confident'])
                image=args.output/'blank512.png'
                if not image.exists(): cv2.imwrite(str(image),np.zeros((512,512,3),np.uint8))
                actual=probe(args.probe,model,image,args.output/(model.stem+'.json'))
                differences=compare(actual,data,tolerance=0)
                cases.append(dict(reference=str(reference),reference_sha256=sha(reference),
                    fixture=str(model),fixture_sha256=sha(model),differences=differences))
    model=args.output/'dynamic.fixture.onnx'; dynamic_model(model)
    dynamic=[]
    for value,n in ((10,0),(240,1)):
        image=args.output/f'dynamic_{n}.png'; cv2.imwrite(str(image),np.full((2,3,3),value,np.uint8))
        actual=probe(args.probe,model,image,args.output/f'dynamic_{n}.json')
        if len(actual['detections'])!=n: raise AssertionError('dynamic count')
        if n:
            d=actual['detections'][0]
            mask=np.fromfile(d['mask_path'],np.float32)
            np.testing.assert_array_equal(mask,np.full(6,np.float32(value)/np.float32(255),np.float32))
        dynamic.append(dict(input=value,count=n))
    # Actual ORT metadata/runtime rejection, beyond the header-only shape tests.
    import onnx
    from onnx import helper as h, numpy_helper as nh
    rejections=[]
    base=args.output/'invalid_base.fixture.onnx'
    constant_model(base,np.array([[0,0,3,2]],np.float32),np.array([1],np.int64),
                   np.array([.8],np.float32),np.zeros((1,1,2,3),np.float32))
    for case in ('input_name','input_type','input_layout','output_order','label_type',
                 'mask_channel','nan_mask','reversed_box'):
        graph=onnx.load(base)
        if case=='input_name': graph.graph.input[0].name='ambiguous'
        elif case=='input_type': graph.graph.input[0].type.tensor_type.elem_type=2
        elif case=='input_layout':
            shape=graph.graph.input[0].type.tensor_type.shape
            for d,v in zip(shape.dim,[1,2,3,3]): d.dim_value=v
        elif case=='output_order':
            graph.graph.output.reverse()
        elif case=='label_type':
            graph.graph.output[1].type.tensor_type.elem_type=1
            graph.graph.node[1].attribute[0].t.CopyFrom(nh.from_array(np.array([1],np.float32)))
        elif case=='mask_channel':
            graph.graph.output[3].type.tensor_type.shape.dim[1].dim_value=2
            graph.graph.node[3].attribute[0].t.CopyFrom(nh.from_array(np.zeros((1,2,2,3),np.float32)))
        elif case=='nan_mask':
            graph.graph.node[3].attribute[0].t.CopyFrom(nh.from_array(np.full((1,1,2,3),np.nan,np.float32)))
        elif case=='reversed_box':
            graph.graph.node[0].attribute[0].t.CopyFrom(nh.from_array(np.array([[3,0,0,2]],np.float32)))
        path=args.output/(case+'.invalid.fixture.onnx'); onnx.save(graph,path)
        try: probe(args.probe,path,args.output/'dynamic_1.png',args.output/(case+'.json'))
        except subprocess.CalledProcessError as error:
            if error.returncode!=2: raise
            rejections.append(case)
        else: raise AssertionError('C++ accepted invalid contract: '+case)
    # Legacy fixture: ROI probabilities must remain complete, byte-exact.
    with np.load(args.validation/'baseline_scene_025.npz',allow_pickle=False) as data:
        keep=data['scores']>=.8; model=args.output/'legacy.fixture.onnx'
        constant_model(model,data['boxes'][keep],data['labels'][keep],data['scores'][keep],data['masks_confident'])
        actual=probe(args.probe,model,args.output/'blank512.png',args.output/'legacy.json',mode='legacy')
        legacy=compare(actual,data,full=False,tolerance=0)
    consumers=[]
    for consumer in ('localization','tracking'):
        actual=probe(args.probe,args.output/'baseline_scene_025.fixture.onnx',
                     args.output/'blank512.png',args.output/(consumer+'.json'),consumer=consumer)
        consumers.append(dict(consumer=consumer,exact_binary_roi_masks=actual['consumer_exact_masks']))
    return dict(kind='contract_test_fixtures_not_models',retained_cases=cases,dynamic=dynamic,legacy=legacy,rejected_contracts=rejections,consumers=consumers)


def historical(args):
    pinned(args.model,BASELINE_SHA)
    recorded=json.loads((args.validation/'result.json').read_text())
    row=next(r for r in recorded['models']['baseline']['inferences'] if r['scene_id']=='scene_025')
    reference=args.validation/row['artifact']; pinned(reference,row['sha256'])
    raw=args.dataset/'scene_025/capture.json.rgb8'
    acquisition=json.loads((args.dataset.parent/'acquisition.json').read_text())
    scene=next(r for r in acquisition['scenes'] if r['scene_id']=='scene_025')
    if scene['split']!='validation': raise ValueError('historical check requires validation input')
    pinned(raw,scene['sha256']['capture.json.rgb8'])
    rgb=np.fromfile(raw,np.uint8).reshape(256,256,3)
    image=args.output/'historical_bgr512.png'
    cv2.imwrite(str(image),cv2.cvtColor(cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR),cv2.COLOR_RGB2BGR))
    actual=probe(args.probe,args.model,image,args.output/'historical.json')
    with np.load(reference,allow_pickle=False) as expected: differences=compare(actual,expected)
    return dict(kind='one_historical_validation_inference_not_v2_qualification',model_sha256=sha(args.model),
      reference_sha256=sha(reference),rgb_sha256=sha(raw),prepared_bgr_png_sha256=sha(image),differences=differences)


def candidate(args):
    """Reusable future comparison: NPZ must be raw PyTorch outputs on this BGR image."""
    pinned(args.model,args.model_sha256); pinned(args.reference,args.reference_sha256)
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.model),sess_options=options,providers=['CPUExecutionProvider'])
    bgr=cv2.imread(str(args.image),cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError('unreadable BGR image')
    tensor=np.ascontiguousarray(bgr[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/np.float32(255)
    raw=session.run(None,{'images':tensor})
    raw_differences={}
    with np.load(args.reference,allow_pickle=False) as expected:
        for name,actual_raw in zip(('boxes','labels','scores','masks'),raw):
            reference=expected[name]
            if reference.dtype!=actual_raw.dtype or reference.shape!=actual_raw.shape:
                raise AssertionError(f'{name} raw dtype/shape mismatch')
            if name=='labels': np.testing.assert_array_equal(actual_raw,reference)
            else: np.testing.assert_allclose(actual_raw,reference,rtol=0,atol=args.atol)
            raw_differences[name]=float(np.max(np.abs(actual_raw-reference),initial=0))
        actual=probe(args.probe,args.model,args.image,args.output/'candidate.json',classes=args.classes)
        differences=compare(actual,expected,tolerance=args.atol)
    return dict(kind='future_candidate_cpp_vs_pytorch',model_sha256=sha(args.model),
      pytorch_reference_sha256=sha(args.reference),raw_output_max_abs=raw_differences,differences=differences)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['fixtures','historical','candidate'])
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--model-sha256')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--image',type=Path)
    parser.add_argument('--classes',type=int,default=2)
    parser.add_argument('--atol',type=float,default=1e-5)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=globals()[args.mode](args)
    report.update(probe_sha256=sha(args.probe),held_out_accessed=False,training_executed=False)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
