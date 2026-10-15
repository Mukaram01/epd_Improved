#!/usr/bin/env python3
"""Proposed source-lattice synthetic diagnostic. Full run requires new approval.
No weights/scenes are loaded. Meta construction inspects structure only.
"""
import argparse
from collections import OrderedDict
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align,MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.poolers import LevelMapper
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.detection.transform import GeneralizedRCNNTransform,resize_boxes
import analytic_coordinate_contract as old
from shadow_mask_paste import digest,project
from diagnostic_shadow_capture import durable_json,durable_npz,supervise

KERNELS=('centre','symmetric','right')
PATTERNS=('impulse','ramp_x','ramp_y','checkerboard','border','empty')
SIZES=((64,64),(65,67),(95,96))
PHASES=(0.,.25,.5,.75)
OUTPUT=Path('/home/user/epd-v2-artifacts/source-lattice-contract-001')
IDENTITY='epd_source_lattice_proposed_001'
PARITY_ATOL=3e-6


def architecture():
    from recover_original import make_model
    from mask_resolution import configure56
    with torch.device('meta'):model=configure56(make_model())
    body=model.backbone.body
    assert (body.conv1.kernel_size,body.conv1.stride,body.conv1.padding)==((7,7),(2,2),(3,3))
    assert (body.maxpool.kernel_size,body.maxpool.stride,body.maxpool.padding)==(3,2,1)
    blocks=[len(getattr(body,'layer'+str(k))) for k in range(1,5)]
    assert blocks==[3,4,6,3]
    for k in range(1,5):
        for j,block in enumerate(getattr(body,'layer'+str(k))):
            assert block.conv1.stride==(1,1) and block.conv2.kernel_size==(3,3) and block.conv2.padding==(1,1)
            assert block.conv2.stride==((2,2) if k>1 and j==0 else (1,1))
            assert block.conv3.kernel_size==(1,1)
    fpn=model.backbone.fpn
    assert len(fpn.inner_blocks)==4 and len(fpn.layer_blocks)==4
    for inner,layer in zip(fpn.inner_blocks,fpn.layer_blocks):
        assert inner[0].kernel_size==(1,1) and inner[0].out_channels==256
        assert layer[0].kernel_size==(3,3) and layer[0].padding==(1,1) and layer[0].stride==(1,1)
    assert fpn.extra_blocks.__class__.__name__=='LastLevelMaxPool'
    assert model.transform.image_mean==[.485,.456,.406] and model.transform.image_std==[.229,.224,.225]
    from torchvision.ops.misc import FrozenBatchNorm2d
    assert all(m.eps==0. for m in model.modules() if isinstance(m,FrozenBatchNorm2d))
    pool=model.roi_heads.mask_roi_pool
    assert pool.featmap_names==['0','1','2','3'] and pool.sampling_ratio==2
    for layer in model.roi_heads.mask_head:
        assert layer[0].kernel_size==(3,3) and layer[0].stride==(1,1) and layer[0].padding==(1,1)
    assert len(model.roi_heads.mask_head)==4
    predictor=model.roi_heads.mask_predictor
    assert predictor.conv5_mask.kernel_size==(2,2) and predictor.conv5_mask.stride==(2,2)
    assert model.transform.min_size==(800,) and model.transform.max_size==1333 and model.transform.size_divisible==32
    return {'construction':'actual EPD make_model/configure56 on meta device; no forward/state load',
        'blocks':blocks,'strides':[4,8,16,32],'mask_pool':list(pool.output_size),'sampling_ratio':pool.sampling_ratio,
        'feature_names':pool.featmap_names,'min_size':800,'max_size':1333,'padding_divisor':32,
        'trained_response_origin':'UNKNOWN'}


def backbone_ledger():
    jump,rf=1,1
    rf+=6*jump;jump*=2;rf+=2*jump;jump*=2
    out=[]
    for stage,count in enumerate((3,4,6,3)):
        for block in range(count):
            rf+=2*jump
            if stage and block==0:jump*=2
        out.append((jump,rf,.5))
    return out


def fpn_support(level,index):
    """Interior infinite-image deepest RF envelope, exact2x FPN sizes only."""
    if level not in (2,3,4,5) or not isinstance(index,int):raise ValueError('INVALID_SUPPORT_ASSUMPTION')
    q=2**(5-level);left=(index-1)//q;right=(index+1)//q
    return .5+32*left-427/2,.5+32*right+427/2


def nearest_indices(source,target):
    if source<=0 or target<=0:raise ValueError('INVALID_NEAREST_SIZE')
    return np.minimum(np.floor(np.arange(target)*source/target).astype(int),source-1)


def kernel(n,kind):
    if kind not in KERNELS or n not in (3,7):raise ValueError('INVALID_SYNTHETIC_KERNEL')
    k=np.zeros((n,n),np.float32)
    if kind=='symmetric':
        v=np.array([1,2,1] if n==3 else [1,6,15,20,15,6,1],np.float32);v/=v.sum();k=np.outer(v,v)
    else:k[n//2,n//2+(kind=='right')]=1
    return k


def numpy_conv(x,k,stride):
    if x.ndim!=2 or x.dtype!=np.float32 or not np.isfinite(x).all() or stride not in (1,2):raise ValueError('INVALID_FIELD')
    p=len(k)//2;y=np.pad(x,p);h=(x.shape[0]-1)//stride+1;w=(x.shape[1]-1)//stride+1
    out=np.zeros((h,w),np.float64)
    for iy,ix in np.argwhere(k!=0):out+=float(k[iy,ix])*y[iy:iy+h*stride:stride,ix:ix+w*stride:stride]
    return out.astype(np.float32)


def torch_conv(x,k,stride):
    return F.conv2d(torch.from_numpy(x)[None,None],torch.from_numpy(k)[None,None],stride=stride,padding=len(k)//2)[0,0].numpy()


def compare(expected,actual,label):
    if expected.shape!=actual.shape or not np.isfinite(actual).all():raise ValueError('PARITY_SHAPE_OR_NUMERIC '+label)
    delta=float(np.max(np.abs(expected-actual))) if expected.size else 0.
    if delta>PARITY_ATOL:raise ValueError('PARITY_DRIFT '+label+' '+str(delta))
    return delta


def numpy_resize(x,size):
    h,w=size;y=(np.arange(h)+.5)*x.shape[0]/h-.5;z=(np.arange(w)+.5)*x.shape[1]/w-.5
    y=np.clip(y,0,x.shape[0]-1);z=np.clip(z,0,x.shape[1]-1);yl=y.astype(int);xl=z.astype(int)
    yh=np.minimum(yl+1,x.shape[0]-1);xh=np.minimum(xl+1,x.shape[1]-1);dy=y-yl;dx=z-xl
    return ((1-dy[:,None])*((1-dx)*x[yl[:,None],xl]+dx*x[yl[:,None],xh])+dy[:,None]*((1-dx)*x[yh[:,None],xl]+dx*x[yh[:,None],xh])).astype(np.float32)


def torch_resize(x,size):return F.interpolate(torch.from_numpy(x)[None,None],size=size,mode='bilinear',align_corners=False)[0,0].numpy()


def pad32(x):
    h,w=x.shape;return np.pad(x,((0,(-h)%32),(0,(-w)%32)))


def maxpool(x,independent):
    if not independent:return F.max_pool2d(torch.from_numpy(x)[None,None],3,2,1)[0,0].numpy()
    y=np.pad(x,1,constant_values=-np.inf);h=(len(x)-1)//2+1;w=(x.shape[1]-1)//2+1
    return np.maximum.reduce([y[i:i+2*h:2,j:j+2*w:2] for i,j in itertools.product(range(3),repeat=2)])


def backbone(x,kind,independent):
    conv=numpy_conv if independent else torch_conv
    x=maxpool(conv(x,kernel(7,kind),2),independent);features=[]
    for stage,count in enumerate((3,4,6,3)):
        for b in range(count):
            stride=2 if stage and b==0 else 1
            # Scalar synthetic analogue: fixed .5 branch weights, identity BN,
            # nonnegative fields make actual ReLU algebra an identity.
            x=.5*conv(x,kernel(3,kind),stride)+.5*x[::stride,::stride]
        features.append(x)
    return features


def check_fpn(xs,kind):
    inner=xs[-1].copy();ys=[numpy_conv(inner,kernel(3,kind),1)]
    for lateral in reversed(xs[:-1]):
        yi=nearest_indices(inner.shape[0],lateral.shape[0]);xi=nearest_indices(inner.shape[1],lateral.shape[1])
        top=inner[yi[:,None],xi]
        observed=F.interpolate(torch.from_numpy(inner)[None,None],size=lateral.shape,mode='nearest')[0,0].numpy()
        old.require_exact(top,observed,'FPN nearest')
        inner=lateral+top;ys.insert(0,numpy_conv(inner,kernel(3,kind),1))
    # Actual installed FPN class, one controlled channel, no random state used.
    fpn=FeaturePyramidNetwork([1,1,1,1],1)
    with torch.no_grad():
        for module in fpn.inner_blocks:
            module[0].weight.fill_(1);module[0].bias.zero_()
        for module in fpn.layer_blocks:
            module[0].weight.copy_(torch.from_numpy(kernel(3,kind))[None,None]);module[0].bias.zero_()
        actual=fpn(OrderedDict((str(i),torch.from_numpy(x)[None,None]) for i,x in enumerate(xs)))
    deltas=[compare(y,actual[str(i)][0,0].numpy(),'FPN'+str(i)) for i,y in enumerate(ys)]
    return ys,deltas


def levels(sides):
    boxes=torch.tensor([[0,0,x,x] for x in sides],dtype=torch.float32)
    actual=LevelMapper(2,5)([boxes])+2
    reference=np.clip(np.floor(4+np.log2(np.asarray(sides)/224)+1e-6),2,5).astype(int)
    if actual.tolist()!=reference.tolist():raise ValueError('LEVEL_ASSIGNMENT_PARITY')
    return actual.tolist()


def mask_chain(field,stride,kind,transformed_size=(128,128)):
    # Full800 probes are interior; small synthetic grids are boundary stress.
    box=np.array([256.25,272.75,480.25,496.75] if transformed_size==(800,800) else [8.25,10.75,48.25,50.75],np.float32);scale=1/stride
    image_size=512
    expected,_=old.numpy_roi(field[None],box*scale,1.,False)
    pooled=roi_align(torch.from_numpy(field)[None,None],torch.tensor([[0,*box]]),28,scale,2,False)[0].numpy()
    old.require_exact(expected,pooled,'actual-stride ROIAlign')
    t=pooled[0];n=t.copy();maxdelta=0.
    for _ in range(4):
        t=torch_conv(t,kernel(3,kind),1);n=numpy_conv(n,kernel(3,kind),1)
        maxdelta=max(maxdelta,compare(n,t,'synthetic head'))
    logits=F.conv_transpose2d(torch.from_numpy(t)[None,None],torch.ones(1,1,2,2),stride=2)-.5
    old.require_exact(logits.numpy(),(np.repeat(np.repeat(t,2,0),2,1)[None,None]-.5),'synthetic deconv')
    transformed_boxes=torch.from_numpy(box)[None]
    boxes=resize_boxes(transformed_boxes,transformed_size,(512,512))
    reference_box=box*np.array([512/transformed_size[1],512/transformed_size[0]]*2,np.float32)
    old.require_exact(boxes[0].numpy(),reference_box,'postprocess resize_boxes')
    golden=paste_masks_in_image(logits.sigmoid(),boxes,(image_size,image_size))
    actual=old.s.reconstruct(logits,boxes,(image_size,image_size));old.require_exact(golden.numpy(),actual.numpy(),'paste')
    prob=golden[0,0].numpy();native,mapped=project(prob,boxes[0].numpy())
    explicit=np.zeros_like(mapped)
    for dy,dx in itertools.product((0,1),repeat=2):explicit|=native[dy::2,dx::2]
    old.require_exact(mapped,explicit,'ANY2x2')
    return {'roi_bitwise':True,'deconv_bitwise':True,'paste_bitwise':True,'union_bitwise':True,
        'transformed_box':box.tolist(),'native512_box':boxes[0].tolist(),'transformed_size':list(transformed_size),
        'head_max_abs':maxdelta,'native_foreground':int(native.sum()),'mapped_foreground':int(mapped.sum()),
        'union_inactive_slots':int(4*mapped.sum()-native.sum())},prob


def pattern(size,name,phase):
    h,w=size;y,x=np.mgrid[:h,:w];cx=w//2+phase;cy=h//2+phase
    if name=='impulse':return (np.maximum(1-np.abs(x-cx),0)*np.maximum(1-np.abs(y-cy),0)).astype(np.float32)
    if name=='ramp_x':return ((x+phase)/(w+1)).astype(np.float32)
    if name=='ramp_y':return ((y+phase)/(h+1)).astype(np.float32)
    if name=='checkerboard':
        base=((x+y)%2).astype(np.float32)
        # Bilinear fractional translation of an infinite checkerboard.
        return (((1-phase)**2+phase**2)*base+2*phase*(1-phase)*(1-base)).astype(np.float32)
    if name=='border':return ((x==0)|(y==0)|(x==w-1)|(y==h-1)).astype(np.float32)
    if name=='empty':return np.zeros(size,np.float32)
    raise ValueError('INVALID_PATTERN')


def shift(x,pixels):
    out=np.zeros_like(x)
    if pixels==0:return x.copy()
    out[pixels:,pixels:]=x[:-pixels,:-pixels];return out


def cases():
    # Three full800 anchors provide nonempty structural boundary-free regions.
    return list(itertools.product(SIZES,PATTERNS,PHASES,KERNELS))+[((800,800),'ramp_x',.25,k) for k in KERNELS]


def bindings(policy):
    for path,sha in policy['sources_sha256'].items():
        if digest(path)!=sha:raise ValueError('SOURCE_DRIFT '+path)


def exclusive(output):
    if output.exists() or output.is_symlink():raise ValueError('OUTPUT_NOT_EXCLUSIVE')
    output.mkdir(exist_ok=False)


def validate_complete(rows,traces):
    actual=[(tuple(r['size']),r['pattern'],r['phase'],r['kernel']) for r in rows]
    if len(rows)!=219 or set(actual)!=set(cases()):raise ValueError('CASE_INCOMPLETE')
    chains=0
    for row in rows:
        if len(row['translations'])!=3 or {t['movement'] for t in row['translations']}!={0,1,32}:raise ValueError('TRANSLATIONS_INCOMPLETE')
        for t in row['translations']:
            if len(t['levels'])!=4 or {l['level'] for l in t['levels']}!={2,3,4,5}:raise ValueError('LEVELS_INCOMPLETE')
            for level in t['levels']:
                if t['movement']==0:
                    chains+=1
                    if not all(level['mask_chain'][k] for k in ('roi_bitwise','deconv_bitwise','paste_bitwise','union_bitwise')):raise ValueError('MASK_PARITY_FAILURE')
        if tuple(row['size'])==(800,800) and row['multiscale_selection']['bitwise_equal'] is not True:raise ValueError('MULTISCALE_PARITY_FAILURE')
    if chains!=876:raise ValueError('MASK_CHAINS_INCOMPLETE')
    names={p.name for p in traces.iterdir()}
    if names!={'level'+str(l)+'.npz' for l in (2,3,4,5)}:raise ValueError('TRACES_INCOMPLETE')
    for path in traces.iterdir():
        with np.load(path,allow_pickle=False) as values:
            if set(values.files)!={'input','C','P','paste'}:raise ValueError('TRACE_FIELDS_INCOMPLETE')
            for key in values.files:
                if not np.isfinite(values[key]).all():raise ValueError('TRACE_NUMERIC_INVALID')
            level=int(path.stem[-1]);n=64//(2**level)
            if values['input'].shape!=(64,64) or values['C'].shape!=(n,n) or values['P'].shape!=(n,n) or values['paste'].shape!=(512,512):raise ValueError('TRACE_SHAPE_DRIFT')


def run(policy,output):
    start=time.monotonic();bindings(policy);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    if torch.cuda.is_available():raise ValueError('CPU_ONLY')
    manifest=architecture();durable_json(output/'architecture.json',manifest)
    # Actual normalization/resize/batch padding, no model forward.
    transform=GeneralizedRCNNTransform(800,1333,[.485,.456,.406],[.229,.224,.225]).eval()
    preprocessing=[]
    import cv2
    raw=(pattern((256,256),'ramp_x',.25)*255).astype(np.uint8)
    up=cv2.resize(raw,(512,512),interpolation=cv2.INTER_LINEAR)
    cv_ref=numpy_resize(raw.astype(np.float32),(512,512))
    cv_delta=float(np.abs(up.astype(np.float32)-cv_ref).max())
    if cv_delta>1:raise ValueError('CV_UINT8_PARITY_DRIFT')
    for size in [(256,256),(255,257),(257,255),(511,513),(512,512)]:
        image=up.astype(np.float32)/255 if size==(512,512) else pattern(size,'ramp_x',.25);rgb=torch.from_numpy(np.stack([image]*3))
        resized,_=transform.resize(transform.normalize(rgb),None)
        expected_size=tuple(int(v*min(800/min(size),1333/max(size))) for v in size)
        if tuple(resized.shape[-2:])!=expected_size:raise ValueError('RESIZE_DIMENSION_DRIFT')
        expected=numpy_resize((image-.485)/.229,expected_size)
        delta=compare(expected,resized[0].numpy(),'transform')
        padded=transform.batch_images([resized]);pad_ref=pad32(resized[0].numpy())
        old.require_exact(padded[0,0].numpy(),pad_ref,'padding')
        preprocessing.append({'input':size,'resized':expected_size,'padded':list(padded.shape[-2:]),'max_abs':delta})
    rows=[];traces=output/'traces';traces.mkdir()
    for index,(size,name,phase,kind) in enumerate(cases()):
        image=pad32(pattern(size,name,phase));base=None;translations=[]
        for movement in (0,1,32):
            field=shift(image,movement);ind=backbone(field,kind,True);observed=backbone(field,kind,False)
            deltas=[compare(n,t,'backbone') for n,t in zip(ind,observed)]
            features,fpn_deltas=check_fpn(observed,kind)
            if movement==0:base=features
            perlevel=[]
            for level,(feature,(stride,rf,anchor)) in enumerate(zip(features,backbone_ledger())):
                lag=movement//stride;pred=shift(base[level],lag)
                # Structural RF frame excludes boundary-affected deepest FPN support.
                yy,xx=np.mgrid[:feature.shape[0],:feature.shape[1]]
                margin=(427+4*stride)/2
                valid=(yy*stride>margin+movement)&(xx*stride>margin+movement)&(yy*stride<image.shape[0]-margin)&(xx*stride<image.shape[1]-margin)
                stats={'level':level+2,'stride':stride,'nominal_anchor':anchor,'shape':list(feature.shape),
                    'translation_input_pixels':movement,'comparison_integer_lag':lag,'phase_remainder':movement%stride,
                    'full_translation_mean_abs':float(np.abs(feature-pred).mean()),'boundary_free_points':int(valid.sum()),
                    'boundary_free_max_abs':float(np.abs(feature-pred)[valid].max()) if valid.any() else None,
                    'energy':float(feature.sum()),'nonzero':int(np.count_nonzero(feature))}
                if feature.sum()>0:stats['energy_centroid_yx']=[float((feature*axis).sum()/feature.sum()) for axis in (yy,xx)]
                else:stats['energy_centroid_yx']=None
                if movement==0:
                    chain,prob=mask_chain(feature,stride,kind,transformed_size=image.shape);stats['mask_chain']=chain
                    if index==0:durable_npz(traces/('level'+str(level+2)+'.npz'),{'input':field,'C':observed[level],'P':feature,'paste':prob})
                perlevel.append(stats)
            translations.append({'movement':movement,'backbone_max_abs':max(deltas),'fpn_max_abs':max(fpn_deltas),'levels':perlevel})
        multiscale=None
        if size==(800,800):
            pool=MultiScaleRoIAlign(['0','1','2','3'],28,2)
            boxes=torch.tensor([[160.25,160.75,160.25+w,160.75+w] for w in [32,112,224,448]],dtype=torch.float32)
            maps=OrderedDict((str(i),torch.from_numpy(x)[None,None]) for i,x in enumerate(base))
            actual=pool(maps,[boxes],[(800,800)])
            for i,(box,lev) in enumerate(zip(boxes,levels([32,112,224,448]))):
                expected=roi_align(maps[str(lev-2)],torch.cat([torch.zeros(1),box])[None],28,1/(2**lev),2,False)
                old.require_exact(actual[i:i+1].numpy(),expected.numpy(),'MultiScaleROIAlign selection')
            multiscale={'boxes':boxes.tolist(),'assigned_levels':levels([32,112,224,448]),'bitwise_equal':True}
        rows.append({'index':index,'size':size,'pattern':name,'phase':phase,'kernel':kind,'multiscale_selection':multiscale,'translations':translations})
    validate_complete(rows,traces)
    bindings(policy)
    durable_json(output/'results.json',{'decision':'COMPLETE_SYNTHETIC_ONLY','cases':len(rows),'rows':rows,
        'opencv_uint8_resize_max_lsb_difference':cv_delta,'trace_sha256':{p.name:digest(p) for p in sorted(traces.iterdir())},'preprocessing':preprocessing,'level_assignment':levels([32,112,224,448]),'worker_seconds':time.monotonic()-start,
        'policy':policy,'trained_response_origin':'UNKNOWN','stage_a2':'BLOCKED_0/2','original_training_supervision':'UNKNOWN'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True)
    p.add_argument('--authorise-full-run',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.authorise_full_run:raise ValueError('FULL_RUN_REQUIRES_SEPARATE_USER_AUTHORISATION')
    policy=json.loads(args.policy.read_text());bindings(policy)
    if old.s.source_inventory()!=policy['runtime_inventory']:raise ValueError('RUNTIME_VERSION_DRIFT')
    review=json.loads(args.review.read_text())
    if (digest(args.review)!=args.review_sha256 or review.get('decision')!='APPROVE_PROPOSED_METHODOLOGY_ONLY' or
        review.get('preregistration_sha256')!=digest(args.policy) or not review.get('reviewer','').startswith('/root/')):
        raise ValueError('INDEPENDENT_REVIEW_REQUIRED')
    if policy['experiment_identity']!=IDENTITY or policy['output_path']!=str(OUTPUT) or policy['geometry_cases']!=219 or policy['limits']!={'wall_seconds':300,'rss_mib':2048,'cpu_threads':2}:raise ValueError('SCOPE_DRIFT')
    if args.worker:
        try:run(policy,OUTPUT);return 0
        except BaseException as exc:
            durable_json(OUTPUT/'failure.json',{'reason':type(exc).__name__+': '+str(exc),'partial_preserved':True});raise
    exclusive(OUTPUT)
    durable_json(OUTPUT/'admission.json',{'policy_sha256':digest(args.policy),'review_sha256':args.review_sha256,'separate_user_authorisation_required':True})
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--authorise-full-run','--policy',str(args.policy.resolve()),'--review',str(args.review.resolve()),'--review-sha256',args.review_sha256]
    process=supervise(command,OUTPUT,300,2048*1024)
    good=process['status']=='WORKER_COMPLETED' and (OUTPUT/'results.json').exists()
    durable_json(OUTPUT/'completion.json',{'decision':'COMPLETE_SYNTHETIC_ONLY' if good else 'BLOCKED_FAILURE','supervision_sha256':digest(OUTPUT/'supervision.json'),'no_retry':True})
    print(json.dumps(process));return 0 if good else 2

if __name__=='__main__':raise SystemExit(main())
