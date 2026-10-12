#!/usr/bin/env python3
"""Safe original cube model recovery. No training, downloads or model export."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import numpy as np
from pilot import sha, write_json, bounded

CHECKPOINT_SHA='b2668971dd1230cdbc93bbc7d0182b46c91e7469717dc210052fa4eb20fa8bdb'
ONNX_SHA='21362d62d816bd684f2b5c7769d0bcd1cd568af86419788fca9d095c46f2bad2'
PNG_SHA='32ec128ba594ec93ac32757538c05780ffd94c5d8974e85a5cf3503864138995'
REFERENCE_SHA='31294906f437b944cc013ee6ccdb5757ca4c6dff947fd3d72af965a3dc5e694b'
TOLERANCES=dict(boxes=1e-3,scores=1e-5,masks=1e-4)


def pinned(path,digest):
    if sha(path)!=digest:raise ValueError('artifact hash mismatch: '+str(path))


def safe_load(path,digest):
    import torch
    pinned(path,digest)
    return torch.load(path,map_location='cpu',weights_only=True)


def unpack(checkpoint):
    import torch
    if not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('model_state'),dict):
        raise ValueError('unsupported checkpoint schema; no fallback')
    if (checkpoint.get('architecture')!='maskrcnn_resnet50_fpn' or checkpoint.get('num_classes')!=2
            or checkpoint.get('class_names')!=['background','cube']):
        raise ValueError('architecture/class metadata mismatch')
    state=checkpoint['model_state']
    expected={'roi_heads.box_predictor.cls_score.weight':(2,1024),
              'roi_heads.box_predictor.bbox_pred.weight':(8,1024),
              'roi_heads.mask_predictor.mask_fcn_logits.weight':(2,256,1,1)}
    for k,shape in expected.items():
        if k not in state or tuple(state[k].shape)!=shape:raise ValueError('class/head tensor mismatch')
    if any(not isinstance(k,str) or not torch.is_tensor(v) or not torch.isfinite(v).all()
           for k,v in state.items()):raise ValueError('invalid/nonfinite state tensor')
    metadata={k:v for k,v in checkpoint.items() if k!='model_state'}
    # Unsupported objects are not rendered via repr or silently serialized.
    json.dumps(metadata,allow_nan=False)
    return state,metadata


def make_model():
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.misc import FrozenBatchNorm2d
    backbone=resnet_fpn_backbone(backbone_name='resnet50',weights=None,
        norm_layer=FrozenBatchNorm2d,trainable_layers=3)
    model=MaskRCNN(backbone,num_classes=2,min_size=800,max_size=1333,
        image_mean=[.485,.456,.406],image_std=[.229,.224,.225],
        box_score_thresh=.05,box_nms_thresh=.5,box_detections_per_img=100,
        rpn_nms_thresh=.7)
    # COCO_V1 construction overwrites FrozenBN epsilon to0. The original ONNX
    # folded scales independently confirm this; default1e-5 is incompatible.
    for module in model.modules():
        if isinstance(module,FrozenBatchNorm2d):module.eps=0.0
    return model


def reconstruct(state):
    model=make_model()
    model.load_state_dict(state,strict=True)
    return model.eval()


def graph_identity(graph,state,model):
    """Verify all95 weights and106 folded FrozenBN scale/bias initializers."""
    import torch
    from onnx import numpy_helper
    from torchvision.ops.misc import FrozenBatchNorm2d
    arrays={i.name:numpy_helper.to_array(i) for i in graph.graph.initializer}
    matched=[];folded=[];covered=set()
    for name,actual in arrays.items():
        key=name.removeprefix('network.')
        if key in state:
            if not np.array_equal(actual,state[key].numpy()):raise ValueError('ONNX weight mismatch: '+key)
            matched.append(key);covered.add(name)
    modules=dict(model.named_modules())
    for node in graph.graph.node:
        if node.op_type not in ('Mul','Add'):continue
        initializers=[name for name in node.input if name in arrays and name.startswith('onnx::')]
        if not initializers:continue
        tokens=node.name.split('/')[2:-1]
        tokens=[t for i,t in enumerate(tokens) if i+1==len(tokens) or not tokens[i+1].startswith(t+'.')]
        key='.'.join(tokens);module=modules.get(key)
        if not isinstance(module,FrozenBatchNorm2d):raise ValueError('unmapped folded initializer: '+node.name)
        scale=module.weight*(module.running_var+module.eps).rsqrt()
        expected=scale if node.op_type=='Mul' else module.bias-module.running_mean*scale
        expected=expected.reshape(1,-1,1,1).numpy()
        for name in initializers:
            actual=arrays[name]
            if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=1e-6,atol=1e-6):
                raise ValueError('FrozenBN mismatch: '+name)
            folded.append(dict(name=name,module=key,operation=node.op_type,
                max_abs=float(np.max(np.abs(actual-expected)))))
            covered.add(name)
    if len(state)!=307 or len(matched)!=95 or len(folded)!=106 or covered!=set(arrays):
        raise ValueError('incomplete graph/checkpoint weight identity')
    constants={n.name:numpy_helper.to_array(next(a.t for a in n.attribute if a.name=='value'))
               for n in graph.graph.node if n.op_type=='Constant'}
    expected_constants={'/network/transform/Constant':np.array([.485,.456,.406],np.float32).reshape(3,1,1),
        '/network/transform/Constant_1':np.array([.229,.224,.225],np.float32).reshape(3,1,1),
        '/network/transform/Constant_2':np.array([512,512]),
        '/network/transform/Constant_3':np.array(800,np.float32),
        '/network/transform/Constant_4':np.array(1333,np.float32),
        '/network/roi_heads/Constant_80':np.array(.05,np.float32),
        '/network/roi_heads/Constant_89':np.array([100])}
    for name,value in expected_constants.items():
        if name not in constants or not np.array_equal(constants[name],value):
            raise ValueError('unsupported graph configuration: '+name)
    return dict(named_weights_bitwise_equal=matched,folded_normalization=folded,
        initializer_count=len(arrays),all_initializers_accounted_for=True,
        transform=dict(mean=[.485,.456,.406],std=[.229,.224,.225],min_size=800,max_size=1333),
        model_score_prefilter=.05,consumer_confidence_threshold=.8,mask_threshold_strict_gt=.5,
        producer=graph.producer_name,producer_version=graph.producer_version,
        opsets=[dict(domain=o.domain,version=o.version) for o in graph.opset_import])


def compare(left,right):
    result=dict(passed=False,tolerances=TOLERANCES,correspondence='same descending-score output order',
        count_left=len(left['scores']),count_right=len(right['scores']))
    if any(left[k].shape!=right[k].shape or left[k].dtype!=right[k].dtype for k in ('boxes','labels','scores','masks')):
        return dict(result,reason='shape/dtype/count mismatch')
    if any(not np.isfinite(v).all() for p in (left,right) for v in p.values()):
        return dict(result,reason='nonfinite output')
    differences={k:float(np.max(np.abs(left[k]-right[k]),initial=0)) for k in TOLERANCES}
    foreground=int(np.count_nonzero((left['masks']>.5)!=(right['masks']>.5)))
    confidence=int(np.count_nonzero((left['scores']>=.8)!=(right['scores']>=.8)))
    labels=np.array_equal(left['labels'],right['labels'])
    passed=labels and not foreground and not confidence and all(differences[k]<=TOLERANCES[k] for k in TOLERANCES)
    return dict(result,passed=bool(passed),labels_exact=labels,max_abs=differences,
        foreground_decision_differences=foreground,confidence_decision_differences=confidence)


def confident(pred):
    """Retained diagnostic schema stores only >=.8 masks, but all boxes/scores."""
    keep=np.flatnonzero(pred['scores']>=.8)
    masks=pred['masks'][keep] if 'masks' in pred else pred['masks_confident']
    if masks.shape[0]!=len(keep):raise ValueError('retained confident mask count mismatch')
    return dict(boxes=pred['boxes'][keep],labels=pred['labels'][keep],scores=pred['scores'][keep],masks=masks)


def verification_exit_code(report):
    return 0 if report.get("classification")=="ORIGINAL_MODEL_RECOVERED" else 2


def finalize(args):
    """Offline report recovery from hash-pinned outputs; never invokes a model."""
    pinned(args.checkpoint,CHECKPOINT_SHA);pinned(args.onnx,ONNX_SHA)
    pinned(args.image,PNG_SHA);pinned(args.reference,REFERENCE_SHA)
    folder=args.predictions
    for name,digest in (('pytorch.npz',args.pytorch_sha256),('onnx.npz',args.onnx_output_sha256),
                        ('inspection.json',args.inspection_sha256),('process.json',args.process_sha256)):
        pinned(folder/name,digest)
    report=json.loads((folder/'inspection.json').read_text())
    pinned(args.executed_source,report['source_sha256'])
    process=json.loads((folder/'process.json').read_text())
    if process['resource_limit_reason'] is not None or process['wall_seconds']>180:
        raise ValueError('inference process resource failure')
    with np.load(folder/'pytorch.npz',allow_pickle=False) as f:pt=dict(f)
    with np.load(folder/'onnx.npz',allow_pickle=False) as f:actual=dict(f)
    with np.load(args.reference,allow_pickle=False) as f:reference=dict(f)
    report.update(parity=compare(pt,actual),onnx_retained_reference=compare(confident(actual),confident(reference)),
        original_inference_process=process,offline_finalization=True,model_inference_executed=False,
        reporting_failure='Original process exited1 after both inferences: retained reference uses masks_confident. No inference repeated.',
        inference_peak_rss_kib=process['child_peak_rss_kib'],inference_process_wall_seconds=process['wall_seconds'],
        finalizer_source_sha256=sha(__file__))
    report['classification']=('ORIGINAL_MODEL_RECOVERED' if report['parity']['passed'] and report['onnx_retained_reference']['passed']
        else 'BLOCKED — numerical parity mismatch; original weight identity recovered')
    args.output.mkdir()
    write_json(args.output/'result.json',report)
    print(json.dumps({k:report[k] for k in ('classification','parity','onnx_retained_reference')}))
    return verification_exit_code(report)


def worker(args):
    import torch,onnx,onnxruntime as ort,cv2
    start=time.monotonic();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    checkpoint=safe_load(args.checkpoint,CHECKPOINT_SHA);state,metadata=unpack(checkpoint)
    pinned(args.onnx,ONNX_SHA);pinned(args.image,PNG_SHA);pinned(args.reference,REFERENCE_SHA)
    graph=onnx.load(str(args.onnx));onnx.checker.check_model(graph)
    model=reconstruct(state);identity=graph_identity(graph,state,model)
    report=dict(checkpoint=dict(path=str(args.checkpoint),bytes=args.checkpoint.stat().st_size,sha256=CHECKPOINT_SHA),
        metadata_self_reported=metadata,optimizer_state_present=any('optim' in k.lower() for k in checkpoint),
        tensor_inventory={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()},
        graph_identity=identity,strict_state_load=True,
        versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','onnx','onnxruntime','numpy')},
        source_sha256=sha(__file__),safe_deserialization='torch.load CPU weights_only=True; no allowlist/fallback',
        provenance_unknown=['training/export source','original data identities','seed','optimizer','scheduler','learning rate','freeze policy'],
        onnx_sha256=ONNX_SHA,input_image_sha256=PNG_SHA,retained_reference_sha256=REFERENCE_SHA,
        input_scene='frozen validation scene_025, retained adapter RGB512 PNG; no ID raster opened',
        input_contract='RGB256 INTER_LINEAR512 then RGB01 CHW float32; model owns normalization/800 resize',
        held_out_accessed=False,historical_two_cube_regression_accessed=False,training_executed=False,export_executed=False)
    write_json(args.output/'inspection.json',report)
    bgr=cv2.imread(str(args.image),cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape!=(512,512,3):raise ValueError('invalid retained input')
    image=np.ascontiguousarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).transpose(2,0,1),dtype=np.float32)/255.
    np.save(args.output/'input.npy',image[None],allow_pickle=False)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.log_severity_level=3
    session=ort.InferenceSession(str(args.onnx),sess_options=options,providers=['CPUExecutionProvider'])
    if [(i.name,i.type,i.shape) for i in session.get_inputs()]!=[('images','tensor(float)',[1,3,512,512])]:
        raise ValueError('ONNX input contract')
    if [o.name for o in session.get_outputs()]!=['boxes','labels','scores','masks']:raise ValueError('ONNX output ordering')
    report['onnx_metadata']=dict(inputs=[dict(name=i.name,type=i.type,shape=i.shape) for i in session.get_inputs()],
        outputs=[dict(name=i.name,type=i.type,shape=i.shape) for i in session.get_outputs()])
    t=time.monotonic()
    with torch.inference_mode():pred=model([torch.from_numpy(image)])[0]
    pt={k:v.cpu().numpy() for k,v in pred.items()};pt_seconds=time.monotonic()-t
    t=time.monotonic();values=session.run(None,{'images':image[None]});ort_seconds=time.monotonic()-t
    actual=dict(zip(('boxes','labels','scores','masks'),values))
    np.savez_compressed(args.output/'pytorch.npz',**pt);np.savez_compressed(args.output/'onnx.npz',**actual)
    with np.load(args.reference,allow_pickle=False) as old:reference={k:old[k] for k in old.files}
    report.update(parity=compare(pt,actual),onnx_retained_reference=compare(confident(actual),confident(reference)),
        pytorch_seconds=pt_seconds,onnx_seconds=ort_seconds,wall_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report['classification']=('ORIGINAL_MODEL_RECOVERED' if report['parity']['passed'] and report['onnx_retained_reference']['passed']
        else 'BLOCKED — numerical parity mismatch; original weight identity recovered')
    report['artifacts']=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(args.output.iterdir())
                         if p.is_file() and p.name not in ('stdout.log','stderr.log')]
    write_json(args.output/'result.json',report)
    pinned(args.checkpoint,CHECKPOINT_SHA);pinned(args.onnx,ONNX_SHA)
    print(json.dumps({k:report[k] for k in ('classification','parity','onnx_retained_reference','pytorch_seconds','onnx_seconds','peak_rss_kib')}),flush=True)
    return verification_exit_code(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['run','worker','audit'])
    for name in ('checkpoint','onnx','image','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--predictions',type=Path);parser.add_argument('--executed-source',type=Path)
    for name in ('pytorch-sha256','onnx-output-sha256','inspection-sha256','process-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if args.action=='audit':sys.exit(finalize(args))
    elif args.action=='worker':sys.exit(worker(args))
    else:
        if shutil.disk_usage(args.output.parent).free<5*1024**3:raise ValueError('disk reserve')
        args.output.mkdir()
        command=[sys.executable,str(Path(__file__).resolve()),'worker']
        for name in ('checkpoint','onnx','image','reference','output'):command+=['--'+name,str(getattr(args,name))]
        process=bounded(command,args.output,180,memory_mib=6144)
        write_json(args.output/'process.json',process);print(json.dumps(process))
        if process['exit_code']!=0:sys.exit(1)
