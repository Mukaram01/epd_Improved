#!/usr/bin/env python3
"""CPU analytic ROIAlign/paste contract, no learned weights or real scene data."""
from __future__ import annotations
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models.detection.roi_heads import paste_masks_in_image
import shadow_mask_paste as s
from diagnostic_shadow_capture import durable_json,durable_npz,supervise

HERE=Path(__file__).resolve().parent
SHAPES=('circle','rectangle','rotated_rectangle','sharp_edge','thin_feature','empty')
ARMS=('legacy','continuous_paste_only','aligned_roi_only','aligned_and_continuous')
BRIDGES=('bilinear','replicate')
SIZES=((12,10),(32,28),(64,52))
OFFSETS=((0.,0.),(.25,.75),(.5,.5))
PLACEMENTS=('interior','border','far_border')
SCALES=(1.,.5,.25)
ORIGINS=('cell_centred','fixed_input_centre')
IMAGE=128


def require_exact(left,right,label):
    if left.dtype!=right.dtype or left.shape!=right.shape or not np.array_equal(left,right):
        delta=float(np.max(np.abs(left-right))) if left.shape==right.shape and left.size else None
        raise ValueError('OPERATOR_PARITY_DRIFT '+label+' maxabs='+str(delta))


def canonical56_coordinates(box):
    """Cross the Torch/NumPy trace boundary explicitly; never mutate ROI boxes."""
    values=box.detach().cpu().numpy() if isinstance(box,torch.Tensor) else np.asarray(box)
    x0,y0,x1,y1=map(float,values)
    centres=(np.arange(56)+.5)/56
    return x0+centres*(x1-x0),y0+centres*(y1-y0)


def numpy_roi(field,box,scale,aligned):
    """Independent NumPy float32 sampling/accumulation, matched to pinned CPU C++.
    Source formula attribution: torchvision v0.22.0 roi_align_common.h/kernel.cpp
    (BSD3; complete source and license retained in analytic_coordinate_sources).
    """
    if (field.dtype!=np.float32 or field.ndim!=3 or not np.isfinite(field).all() or
        np.asarray(box).shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1] or
        scale not in SCALES):raise ValueError('ROI_CONTRACT: invalid/degenerate input')
    f=np.float32;box=np.asarray(box,dtype=np.float32);offset=f(.5 if aligned else 0.)
    start=box[:2]*f(scale)-offset;end=box[2:]*f(scale)-offset;extent=end-start
    if not aligned:extent=np.maximum(extent,f(1.))
    bins=extent/f(28)
    axes=[(start[k]+np.arange(28,dtype=np.float32)[:,None]*bins[k])+
          ((np.arange(2,dtype=np.float32)[None,:]+f(.5))*bins[k])/f(2.) for k in (0,1)]
    x=np.broadcast_to(axes[0][None,:,None,:],(28,28,2,2)).copy()
    y=np.broadcast_to(axes[1][:,None,:,None],(28,28,2,2)).copy()
    grid=np.stack((x,y),axis=-1)
    _,h,w=field.shape;valid=(x>=-1)&(x<=w)&(y>=-1)&(y<=h)
    x=np.clip(x,f(0),f(w-1));y=np.clip(y,f(0),f(h-1))
    xl=x.astype(int);yl=y.astype(int);xh=np.minimum(xl+1,w-1);yh=np.minimum(yl+1,h-1)
    lx=x-xl.astype(np.float32);ly=y-yl.astype(np.float32);hx=f(1)-lx;hy=f(1)-ly
    values=(((hy*hx)*field[:,yl,xl]+(hy*lx)*field[:,yl,xh])+(ly*hx)*field[:,yh,xl])+(ly*lx)*field[:,yh,xh]
    values=np.where(valid,values,f(0))
    result=np.zeros((len(field),28,28),np.float32)
    for iy in range(2):
        for ix in range(2):result=result+values[:,:,:,iy,ix]
    return result/f(4),grid


def shape_local(x,y,box):
    x0,y0,x1,y1=box;return x-(x0+x1)/2,y-(y0+y1)/2,x1-x0,y1-y0


def contains(shape,x,y,box):
    """Boolean analytic membership, independent of interpolated masks and SDF sign."""
    x,y,w,h=shape_local(np.asarray(x),np.asarray(y),box)
    if shape=='empty':return np.zeros(np.broadcast_shapes(x.shape,y.shape),bool)
    if shape=='circle':return x*x+y*y<(.30*min(w,h))**2
    if shape=='rectangle':return (np.abs(x)<.32*w)&(np.abs(y)<.26*h)
    if shape=='rotated_rectangle':
        u=math.cos(math.pi/6)*x+math.sin(math.pi/6)*y;v=-math.sin(math.pi/6)*x+math.cos(math.pi/6)*y
        return (np.abs(u)<.32*w)&(np.abs(v)<.20*h)
    if shape=='thin_feature':return (np.abs(x)<.75)&(np.abs(y)<.30*h)
    if shape=='sharp_edge':return (x<.4*w)&(y<.35*h)&(y>-(.35*h/(.4*w))*x)
    raise ValueError('UNKNOWN_SHAPE')


def sdf(shape,x,y,box):
    """Exact Euclidean signed distance, positive inside, for the six known shapes."""
    x,y,w,h=shape_local(np.asarray(x),np.asarray(y),box)
    if shape=='empty':return np.full(np.broadcast_shapes(x.shape,y.shape),-20.)
    if shape=='circle':return .30*min(w,h)-np.hypot(x,y)
    if shape=='sharp_edge':
        vertices=[(-.4*w,.35*h),(.4*w,.35*h),(.4*w,-.35*h)]
        distance=np.full(np.broadcast_shapes(x.shape,y.shape),np.inf)
        crosses=[]
        for (ax,ay),(bx,by) in zip(vertices,vertices[1:]+vertices[:1]):
            dx,dy=bx-ax,by-ay;t=np.clip(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy),0,1)
            distance=np.minimum(distance,np.hypot(x-(ax+t*dx),y-(ay+t*dy)))
            crosses.append(dx*(y-ay)-dy*(x-ax))
        inside=np.all(np.stack(crosses)<0,axis=0)
        return np.where(inside,distance,-distance)
    if shape=='rotated_rectangle':
        x,y=math.cos(math.pi/6)*x+math.sin(math.pi/6)*y,-math.sin(math.pi/6)*x+math.cos(math.pi/6)*y
        hx,hy=.32*w,.20*h
    elif shape=='rectangle':hx,hy=.32*w,.26*h
    elif shape=='thin_feature':hx,hy=.75,.30*h
    else:raise ValueError('UNKNOWN_SHAPE')
    qx,qy=np.abs(x)-hx,np.abs(y)-hy
    return -(np.hypot(np.maximum(qx,0),np.maximum(qy,0))+np.minimum(np.maximum(qx,qy),0))


def reference(shape,box):
    y,x=np.mgrid[:IMAGE,:IMAGE];x=x+.5;y=y+.5
    mask=contains(shape,x,y,box);distance=sdf(shape,x,y,box)
    band=np.abs(distance)<=math.sqrt(.5) if shape!='empty' else np.zeros_like(mask)
    coverage={}
    for n in (4,8):
        result=mask.astype(float);ys,xs=np.nonzero(band)
        shifts=(np.arange(n)+.5)/n-.5
        xx=xs[:,None,None]+.5+shifts[None,None,:];yy=ys[:,None,None]+.5+shifts[None,:,None]
        result[ys,xs]=contains(shape,xx,yy,box).mean(axis=(1,2))
        coverage[n]=result
    return {'mask':mask,'band':band,'coverage':coverage[8],'centre_area':int(mask.sum()),
            'uncertain_pixels':int(band.sum()),'coverage_area4':float(coverage[4].sum()),'coverage_area8':float(coverage[8].sum())}


def bridge(pooled,kind):
    if kind=='bilinear':return F.interpolate(pooled,size=(56,56),mode='bilinear',align_corners=False)
    if kind=='replicate':return pooled.repeat_interleave(2,-2).repeat_interleave(2,-1)
    raise ValueError('UNKNOWN_SYNTHETIC_BRIDGE')


def continuous_paste(logits,box):
    """Map image pixel centres to unpadded ROI bin centres, no rounding/+1 extent.
    This is a shadow mathematical definition, never a production replacement.
    """
    axis=torch.arange(IMAGE,dtype=torch.float32)+.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    grid=torch.stack(((xx-box[0])/(box[2]-box[0])*2-1,(yy-box[1])/(box[3]-box[1])*2-1),-1)[None]
    return F.grid_sample(logits.sigmoid(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,0].numpy()


def numpy_continuous_paste(prob,box):
    y,x=np.mgrid[:IMAGE,:IMAGE];x=(x+.5-box[0])/(box[2]-box[0])*56-.5;y=(y+.5-box[1])/(box[3]-box[1])*56-.5
    xl=np.floor(x).astype(int);yl=np.floor(y).astype(int);lx=x-xl;ly=y-yl;out=np.zeros((IMAGE,IMAGE))
    for dy,wy in [(0,1-ly),(1,ly)]:
        for dx,wx in [(0,1-lx),(1,lx)]:
            valid=(xl+dx>=0)&(xl+dx<56)&(yl+dy>=0)&(yl+dy<56)
            out+=np.where(valid,wy*wx*prob[np.clip(yl+dy,0,55),np.clip(xl+dx,0,55)],0)
    return out.astype(np.float32)


def measure(prob,shape,box,ref):
    if prob.shape!=(IMAGE,IMAGE) or prob.dtype!=np.float32 or not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any():
        raise ValueError('MEASUREMENT_CONTRACT')
    pred=prob>.5;target=ref['mask'];spill=pred&~target;missing=target&~pred;union=int((pred|target).sum())
    centroid=None
    if pred.any() and target.any():
        yp,xp=np.nonzero(pred);yt,xt=np.nonzero(target);centroid=[float(xp.mean()-xt.mean()),float(yp.mean()-yt.mean())]
    crossings=[]
    for axis in (0,1):
        p0,p1=(prob[:-1],prob[1:]) if axis==0 else (prob[:,:-1],prob[:,1:])
        yy,xx=np.nonzero((p0>.5)!=(p1>.5))
        t=(.5-p0[yy,xx])/(p1[yy,xx]-p0[yy,xx])
        crossings.append(-sdf(shape,xx+.5+(t if axis==1 else 0),yy+.5+(t if axis==0 else 0),box))
    outward=np.concatenate(crossings)
    area=float(ref['coverage'].sum());intersection=float(ref['coverage'][pred].sum());coverage_union=float(pred.sum())+area-intersection
    return {'iou':int((pred&target).sum())/union if union else 1.,'spill':int(spill.sum()),'missing':int(missing.sum()),
        'predicted_foreground':int(pred.sum()),'reference_foreground':int(target.sum()),
        'shape_extinguished':bool(ref['coverage_area8']>0 and not pred.any()),
        'foreground_delta':int(pred.sum())-int(target.sum()),
        'error_outside_sampling_band':int(((spill|missing)&~ref['band']).sum()),
        'centroid_dxdy':centroid,'boundary_crossings':len(outward),
        'boundary_signed_outward_mean':float(outward.mean()) if len(outward) else None,
        'boundary_abs_mean':float(np.abs(outward).mean()) if len(outward) else None,
        'boundary_abs_p95':float(np.percentile(np.abs(outward),95)) if len(outward) else None,
        'boundary_abs_max':float(np.abs(outward).max()) if len(outward) else None,
        'coverage_iou8':intersection/coverage_union if coverage_union else 1.,
        'coverage_area_error8':float(pred.sum())-area}


def cases():
    for shape,(w,h),offset,placement in itertools.product(SHAPES,SIZES,OFFSETS,PLACEMENTS):
        x,y=(32.,36.) if placement=='interior' else ((-1.25,-1.5) if placement=='border' else (IMAGE-w+1.25,IMAGE-h+1.5))
        x+=offset[0];y+=offset[1]
        box=np.array([x,y,x+w,y+h],np.float32)
        for scale,origin in itertools.product(SCALES,ORIGINS):
            yield {'shape':shape,'size':[w,h],'offset':list(offset),'placement':placement,'scale':scale,'origin':origin,
                   'analytic_box':box.tolist(),'roi_box':np.clip(box,0,IMAGE).tolist()}


def summary(rows):
    out={}
    for origin,kind,arm in itertools.product(ORIGINS,BRIDGES,ARMS):
        selected=[r['arms'][arm] for r in rows if r['origin']==origin and r['bridge']==kind and r['shape']!='empty']
        boundary=[m for m in selected if m['boundary_crossings']]
        centroids=[m['centroid_dxdy'] for m in selected if m['centroid_dxdy'] is not None]
        out[origin+'/'+kind+'/'+arm]={
            'nonempty_shape_cases':len(selected),'mean_iou':float(np.mean([m['iou'] for m in selected])),
            'mean_coverage_iou8':float(np.mean([m['coverage_iou8'] for m in selected])),
            'spill':sum(m['spill'] for m in selected),'missing':sum(m['missing'] for m in selected),
            'foreground_delta':sum(m['foreground_delta'] for m in selected),
            'extinguished_cases':sum(m['shape_extinguished'] for m in selected),
            'error_outside_sampling_band':sum(m['error_outside_sampling_band'] for m in selected),
            'centroid_dxdy_mean':np.mean(centroids,axis=0).tolist() if centroids else None,
            'boundary_crossings':sum(m['boundary_crossings'] for m in boundary),
            'boundary_signed_outward_weighted_mean':sum(m['boundary_signed_outward_mean']*m['boundary_crossings'] for m in boundary)/sum(m['boundary_crossings'] for m in boundary) if boundary else None,
            'boundary_abs_weighted_mean':sum(m['boundary_abs_mean']*m['boundary_crossings'] for m in boundary)/sum(m['boundary_crossings'] for m in boundary) if boundary else None,
            'boundary_abs_max':max((m['boundary_abs_max'] for m in boundary),default=None)}
    return out


def policy_record():
    return {'schema':'epd_analytic_coordinate_contract/v1','image_size':IMAGE,'shapes':list(SHAPES),'sizes':SIZES,
        'offsets':OFFSETS,'placements':PLACEMENTS,'scales':SCALES,'origins':ORIGINS,
        'bridges':BRIDGES,'arms':ARMS,'cases':len(list(cases())),'roi_output':28,'sampling_ratio':2,
        'synthetic_logits_size':56,'logit_field':'positive-inside exact signed Euclidean distance /1px; empty=-20',
        'reference':'Independent strict analytic membership at image pixel centres (j+.5); conservative abs(SDF)<=sqrt(.5) sampling band;4x4/8x8 subpixel area estimates inside that band.',
        'bridge_definition':'Bilinear28to56 align_corners=False; replicate is exactly unit2x2 stride2 transposed-convolution, not trained weights or the learned mask head.',
        'origin_definition':{'cell_centred':'feature world coordinate (j+.5)/scale','fixed_input_centre':'feature world coordinate .5+j/scale'},
        'shadow_definition':'unaligned/aligned ROIAlign crossed with original/continuous unrounded bin-centre paste; paired arm changes two conventions, interaction is not additive.',
        'parity_gate':'NumPy CPU ROI reference bitwise equal for both flags; independent original paste bitwise equal; all originals checked before shadow comparisons.',
        'limits':{'cpu_threads':2,'wall_seconds':300,'rss_mib':2048},'threshold_strict_gt':.5,
        'snapshots':'size32x28, offset(.25,.75), interior: all shapes/scales/origins/bridges (72 traces)',
        'interpretation':'No winner chosen from spill alone; report IoU, missing, area conservation, coverage uncertainty and signed/absolute displacement, including empties/thin losses and origin sensitivity.',
        'limitations':['Contour statistics sample level0.5 crossings between image pixel centres and omit image-frame boundary edges; they are not full Hausdorff distance.',
          '4x4/8x8 coverage estimates and their differences are quadrature/convergence diagnostics, not guaranteed error bars. The analytic SDF sampling band is conservative.',
          'Both feature origins are synthetic assumptions, not established origins of the trained FPN.',
          'Synthetic bilinear/unit-deconvolution bridges do not reproduce learned mask-head convolutions, nonlinearities or trained weights.',
          'Boundary means exclude absent contours; extinguished-feature counts, IoU and missing/area measurements must be inspected together.'],
        'real_data_access':False,'trained_weights':False,'stage_a2':'BLOCKED_0/2','original_training_supervision':'UNKNOWN'}


def run(policy_path,output):
    start=time.monotonic();policy=json.loads(Path(policy_path).read_text())
    actual=policy_record()
    # JSON roundtrip canonicalizes tuples. Do not silently adapt preregistered cases.
    if {k:policy[k] for k in actual}!=json.loads(json.dumps(actual)):raise ValueError('PREREGISTRATION_DRIFT')
    for path,sha in policy['sources_sha256'].items():
        if s.digest(path)!=sha:raise ValueError('SOURCE_DRIFT '+path)
    if s.source_inventory()!=policy['source_inventory']:raise ValueError('VERSION_SOURCE_DRIFT')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():raise ValueError('CPU_ONLY')
    prepared=[];refs={};roi_checks=0;paste_checks=0;replicate_checks=0
    for index,case in enumerate(cases()):
        box=np.asarray(case['analytic_box']);roi=np.asarray(case['roi_box'],np.float32)
        refkey=json.dumps([case[k] for k in ('shape','size','offset','placement')])
        if refkey not in refs:refs[refkey]=reference(case['shape'],box)
        scale=case['scale'];n=int(IMAGE*scale);origin=.5/scale if case['origin']=='cell_centred' else .5
        yy,xx=np.mgrid[:n,:n];x=origin+xx/scale;y=origin+yy/scale
        field=np.stack((sdf(case['shape'],x,y,box),x,y)).astype(np.float32)
        pooled={};grids={};baseline={}
        for aligned in (False,True):
            expected,grid=numpy_roi(field,roi,scale,aligned)
            observed=roi_align(torch.from_numpy(field)[None],torch.tensor([[0,*roi]],dtype=torch.float32),28,scale,2,aligned)[0]
            require_exact(expected,observed.numpy(),'ROI case'+str(index)+' aligned'+str(aligned));roi_checks+=1
            pooled[aligned]=observed;grids[aligned]=grid
            for kind in BRIDGES:
                logits=bridge(observed[None,:1],kind)
                if kind=='replicate':
                    exact=F.conv_transpose2d(observed[None,:1],torch.ones(1,1,2,2),stride=2)
                    require_exact(logits.numpy(),exact.numpy(),'unit_deconv');replicate_checks+=1
                golden=paste_masks_in_image(logits.sigmoid(),torch.from_numpy(roi)[None],(IMAGE,IMAGE))
                independent=s.reconstruct(logits,torch.from_numpy(roi)[None],(IMAGE,IMAGE))
                require_exact(golden.numpy(),independent.numpy(),'paste case'+str(index));paste_checks+=1
                baseline[aligned,kind]=golden[0,0].numpy()
        canonical_x=roi[0]+(np.arange(28)+.5)*(roi[2]-roi[0])/28
        canonical_y=roi[1]+(np.arange(28)+.5)*(roi[3]-roi[1])/28
        ramp={str(flag):{'dx_mean':float(np.mean(pooled[flag][1].numpy()-canonical_x[None,:])),
                        'dy_mean':float(np.mean(pooled[flag][2].numpy()-canonical_y[:,None]))} for flag in (False,True)}
        prepared.append((case,refkey,field,pooled,grids,baseline,ramp))
    durable_json(output/'operator_parity.json',{'roi_numpy_bitwise_checks':roi_checks,'paste_bitwise_checks':paste_checks,
        'replicate_unit_deconv_bitwise_checks':replicate_checks,'max_abs':0.,'all_originals_before_shadows':True})
    rows=[];snapshots=[];traces=output/'traces';traces.mkdir()
    with (output/'measurements.jsonl').open('x') as stream:
        for index,(case,refkey,field,pooled,grids,baseline,ramp) in enumerate(prepared):
            roi=torch.tensor(case['roi_box'],dtype=torch.float32);box=np.asarray(case['analytic_box']);ref=refs[refkey]
            for kind in BRIDGES:
                outputs={'legacy':baseline[False,kind],'aligned_roi_only':baseline[True,kind]}
                for aligned,arm in [(False,'continuous_paste_only'),(True,'aligned_and_continuous')]:
                    outputs[arm]=continuous_paste(bridge(pooled[aligned][None,:1],kind),roi)
                metrics={arm:measure(outputs[arm],case['shape'],box,ref) for arm in ARMS}
                for m in metrics.values():
                    if m['foreground_delta']!=m['spill']-m['missing']:raise ValueError('CONSERVATION_FAILURE')
                row={**case,'bridge':kind,'ramp_roi_to_box_centre_error':ramp,
                    'reference':{k:ref[k] for k in ('centre_area','uncertain_pixels','coverage_area4','coverage_area8')},
                    'feature_positive_samples':int((field[0]>0).sum()),
                    'roi_positive_bins':{str(flag):int((pooled[flag][0].numpy()>0).sum()) for flag in (False,True)},
                    'synthetic56_positive_bins':{str(flag):int((bridge(pooled[flag][None,:1],kind).numpy()>0).sum()) for flag in (False,True)},
                    'arms':metrics}
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');rows.append(row)
                if case['size']==[32,28] and case['offset']==[.25,.75] and case['placement']=='interior':
                    from torchvision.models.detection.roi_heads import expand_boxes
                    expanded=expand_boxes(roi[None],58./56.)[0];integer=expanded.to(torch.int64)
                    canonical_x56,canonical_y56=canonical56_coordinates(roi)
                    values={'expanded_box_float':expanded.numpy(),'expanded_box_truncated':integer.numpy(),
                        'inclusive_paste_size_hw':np.array([max(int(integer[3]-integer[1]+1),1),max(int(integer[2]-integer[0]+1),1)]),
                        'synthetic_logits56_unaligned':bridge(pooled[False][None,:1],kind).numpy(),
                        'synthetic_logits56_aligned':bridge(pooled[True][None,:1],kind).numpy(),
                        'bridge_world_coordinates_unaligned':bridge(pooled[False][None,1:],kind).numpy(),
                        'bridge_world_coordinates_aligned':bridge(pooled[True][None,1:],kind).numpy(),
                        'canonical56_world_x':canonical_x56,
                        'canonical56_world_y':canonical_y56,
                        'feature_field':field,'reference_centre_mask':ref['mask'],'reference_coverage8':ref['coverage'],
                        'roi_box':roi.numpy(),'analytic_box':box,'pooled_unaligned':pooled[False].numpy(),
                        'pooled_aligned':pooled[True].numpy(),'grid_unaligned':grids[False],'grid_aligned':grids[True],
                        **{arm:values for arm,values in outputs.items()}}
                    path=traces/(str(index)+'_'+kind+'.npz');durable_npz(path,values)
                    snapshots.append({'path':str(path.relative_to(output)),'sha256':s.digest(path)})
        stream.flush();os.fsync(stream.fileno())
    for path,sha in policy['sources_sha256'].items():
        if s.digest(path)!=sha:raise ValueError('SOURCE_DRIFT_AFTER_RUN')
    result={'schema':'epd_analytic_coordinate_results/v1','decision':'SYNTHETIC_DIAGNOSTIC_ONLY_NO_PRODUCTION_CHANGE',
        'limitations':policy['limitations'],'policy_sha256':s.digest(policy_path),'sources_sha256':policy['sources_sha256'],'source_inventory':policy['source_inventory'],
        'feature_settings':len(prepared),'roundtrip_cases':len(rows),'arm_measurements':len(rows)*len(ARMS),
        'operator_parity':json.loads((output/'operator_parity.json').read_text()),'summary':summary(rows),
        'measurements_sha256':s.digest(output/'measurements.jsonl'),'snapshots':snapshots,
        'worker_seconds':time.monotonic()-start,'real_scene_inference':False,'checkpoints_loaded':0,
        'stage_a2':'BLOCKED_0/2','original_training_supervision':'UNKNOWN'}
    durable_json(output/'results.json',result)


def launch(policy_path,output):
    if output.exists() or output.is_symlink():raise ValueError('OUTPUT_NOT_EXCLUSIVE')
    output.mkdir(exist_ok=False)
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--policy',str(policy_path.resolve()),'--output',str(output.resolve())]
    process=supervise(cmd,output,300,2048*1024)
    success=process['status']=='WORKER_COMPLETED' and (output/'results.json').exists()
    durable_json(output/'completion.json',{'decision':'COMPLETED_SYNTHETIC_ONLY' if success else 'BLOCKED_FAILURE',
        'results_sha256':s.digest(output/'results.json') if (output/'results.json').exists() else None,
        'supervision_sha256':s.digest(output/'supervision.json'),'no_automatic_retry':True})
    return process


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    if args.worker:
        try:run(args.policy,args.output);return 0
        except BaseException as exc:
            durable_json(args.output/'failure.json',{'reason':type(exc).__name__+': '+str(exc),'partial_evidence_preserved':True});raise
    result=launch(args.policy,args.output);print(json.dumps(result,sort_keys=True));return 0 if result['status']=='WORKER_COMPLETED' else 2


if __name__=='__main__':raise SystemExit(main())
