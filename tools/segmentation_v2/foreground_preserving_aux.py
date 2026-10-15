"""Training-only foreground-anchored native512 boundary loss. Historical module is unchanged. No inference replacements.
Ground truth owns discrete bands; predictions remain continuous through the
installed Torchvision sigmoid/paste. No binary predicted mask is differentiated.
"""
from collections import defaultdict
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.detection.transform import resize_boxes

MASK_PREFIXES=('roi_heads.mask_head.','roi_heads.mask_predictor.')


def validate_targets(targets,classes):
    for target in targets:
        masks,boxes,labels=(target[k] for k in ('masks','boxes','labels'))
        n=len(labels)
        if (masks.device.type!='cpu' or masks.shape!=(n,512,512) or masks.dtype not in (torch.uint8,torch.bool) or
            boxes.device.type!='cpu' or boxes.shape!=(n,4) or boxes.dtype!=torch.float32 or not torch.isfinite(boxes).all() or
            labels.device.type!='cpu' or labels.dtype!=torch.int64 or labels.shape!=(n,) or
            ((labels<1)|(labels>=classes)).any() or ((masks!=0)&(masks!=1)).any()):raise ValueError('NATIVE_TARGET_CONTRACT')
        if (masks.sum(0)>1).any():raise ValueError('OWNER_OVERLAP: visible native annotations must be exclusive')
        if n and ((boxes<0).any() or (boxes>512).any() or (boxes[:,2:]<=boxes[:,:2]).any()):raise ValueError('TARGET_BOX_CONTRACT')
        for mask,box in zip(masks,boxes):
            yy,xx=torch.where(mask!=0)
            if not len(xx):raise ValueError('EMPTY_OWNER: omit invisible instances')
            if not (box[0]<=xx.min() and box[1]<=yy.min() and box[2]>=xx.max()+1 and box[3]>=yy.max()+1):raise ValueError('OWNER_BOX_MISMATCH')


def boundary_bands(mask,radius):
    if (mask.shape!=(512,512) or mask.dtype not in (torch.uint8,torch.bool) or mask.device.type!='cpu' or
        not math.isfinite(radius) or radius<=0 or ((mask!=0)&(mask!=1)).any()):raise ValueError('BOUNDARY_CONTRACT')
    data=mask.detach().numpy().astype(np.uint8)
    # Annotation-only Euclidean distances, padded to include image-frame edges.
    fg=cv2.distanceTransform(np.pad(data,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    bg=cv2.distanceTransform(1-data,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return torch.from_numpy((data!=0)&(fg<=radius)),torch.from_numpy((data==0)&(bg<=radius))


def foreground_probability_loss(prob,mask,bands,roi):
    """Previous boundary objective plus a fixed, owner-visible foreground anchor.
    No baseline masks, thresholding or amodal completion in the training graph.
    """
    from native_boundary_aux import probability_boundary_loss
    base=probability_boundary_loss(prob,bands,roi)
    if (mask.shape!=(512,512) or mask.device.type!='cpu' or mask.dtype not in (torch.uint8,torch.bool) or
        ((mask!=0)&(mask!=1)).any() or roi.shape!=(512,512) or roi.dtype!=torch.bool):
        raise ValueError('FOREGROUND_ANCHOR_CONTRACT')
    active=mask.bool()&roi
    anchor=F.binary_cross_entropy(prob[active],torch.ones_like(prob[active])) if active.any() else prob.sum()*0
    return base+.5*anchor


def native_boundary_loss(logits,proposals,matched_idxs,native_targets,image_shapes,radius=2.):
    if (logits.device.type!='cpu' or logits.dtype!=torch.float32 or logits.ndim!=4 or logits.shape[2:]!=(56,56) or
        logits.shape[1]<2 or not torch.isfinite(logits).all() or
        not len(proposals)==len(matched_idxs)==len(native_targets)==len(image_shapes)):
        raise ValueError('MASK_LOGIT_OR_BATCH_CONTRACT')
    if not math.isfinite(radius) or radius<=0:raise ValueError('BOUNDARY_RADIUS_CONTRACT')
    validate_targets(native_targets,logits.shape[1])
    if sum(len(p) for p in proposals)!=len(logits):raise ValueError('PROPOSAL_LOGIT_ORDER')
    owner_losses=defaultdict(list);offset=0;skipped=0;outside=0
    for batch,(boxes,owners,target,size) in enumerate(zip(proposals,matched_idxs,native_targets,image_shapes)):
        h,w=size
        if h<=0 or w<=0 or boxes.device.type!='cpu' or owners.device.type!='cpu' or boxes.shape!=(len(owners),4) or boxes.dtype!=torch.float32 or owners.dtype!=torch.int64:
            raise ValueError('PROPOSAL_CONTRACT')
        if not torch.isfinite(boxes).all() or (boxes<0).any() or (boxes[:,[0,2]]>w).any() or (boxes[:,[1,3]]>h).any() or (boxes[:,2:]<=boxes[:,:2]).any():
            raise ValueError('PROPOSAL_ROI_CONTRACT')
        if ((owners<0)|(owners>=len(target['labels']))).any():raise ValueError('OWNER_INDEX_CONTRACT')
        native_boxes=resize_boxes(boxes.detach(),size,(512,512))
        bands={int(i):boundary_bands(target['masks'][int(i)],radius) for i in owners.unique()}
        for j,(box,owner) in enumerate(zip(native_boxes,owners.tolist())):
            cls=int(target['labels'][owner]);selected=logits[offset+j,cls][None,None]
            # This installed original path is differentiable wrt probabilities.
            prob=paste_masks_in_image(selected.sigmoid(),box[None],(512,512))[0,0]
            x0,y0,x1,y1=box.clamp(0,512).to(torch.int64).tolist()
            if x0>=x1 or y0>=y1:skipped+=1;continue # unrepresentable consumer ROI, never threshold predictions
            roi=torch.zeros((512,512),dtype=torch.bool);roi[y0:y1,x0:x1]=True
            outside+=int((target['masks'][owner].bool()&~roi).sum())
            if not any((band&roi).any() for band in bands[owner]) and not (target['masks'][owner].bool()&roi).any():skipped+=1;continue
            owner_losses[batch,owner].append(foreground_probability_loss(prob,target['masks'][owner],bands[owner],roi))
        offset+=len(boxes)
    terms=[torch.stack(values).mean() for values in owner_losses.values()]
    loss=torch.stack(terms).mean() if terms else logits.sum()*0
    if not torch.isfinite(loss):raise ValueError('NONFINITE_NATIVE_LOSS')
    return loss,{'owners':len(terms),'proposals':len(logits),'skipped_proposals':skipped,
        'outside_roi_foreground_samples':outside,'boundary_radius_native_pixels':radius,
        'loss_revision':'foreground_anchor_v1','foreground_anchor_coefficient':.5}


def freeze_non_mask(model):
    for name,parameter in model.named_parameters():parameter.requires_grad_(name.startswith(MASK_PREFIXES))


def non_mask_snapshot(model):
    return {n:t.detach().clone() for n,t in model.state_dict().items() if not n.startswith(MASK_PREFIXES)}


def assert_frozen(model,snapshot):
    now={n:t for n,t in model.state_dict().items() if not n.startswith(MASK_PREFIXES)}
    if set(now)!=set(snapshot) or any(not torch.equal(now[n],t) for n,t in snapshot.items()):raise ValueError('FROZEN_NON_MASK_STATE_DRIFT')


def training_losses(model,images,targets,weight=0.,radius=2.,diagnostics=None):
    if not math.isfinite(weight) or weight<0:raise ValueError('INVALID_AUXILIARY_WEIGHT')
    if weight==0:return model(images,targets) # exact legacy dispatch; no hooks/target validation/additional graph
    if not model.training:raise ValueError('TRAINING_ONLY')
    if any(p.requires_grad for n,p in model.named_parameters() if not n.startswith(MASK_PREFIXES)):
        raise ValueError('NON_MASK_MUST_BE_FROZEN')
    if len(images)!=len(targets) or any(tuple(image.shape)!=(3,512,512) or image.device.type!='cpu' or image.dtype!=torch.float32 or not torch.isfinite(image).all() or ((image<0)|(image>1)).any() for image in images):raise ValueError('NATIVE_INPUT_CONTRACT')
    validate_targets(targets,model.roi_heads.mask_predictor.mask_fcn_logits.out_channels)
    # Clone labels/annotations/boxes so transformed target ownership is checkable.
    native=[{k:t[k].detach().clone() for k in ('masks','boxes','labels')} for t in targets]
    # Keep parameter-free state updates out of frozen modules, too. Restore flags.
    flags={m:m.training for m in model.modules()}
    model.backbone.eval()
    for name,module in model.named_modules():
        if not name.startswith(MASK_PREFIXES) and isinstance(module,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)):
            module.eval()
    captured={};handles=[]
    def roi_pre(module,args):
        _,_,sizes,transformed=args
        if transformed is None or len(transformed)!=len(native):raise ValueError('TRANSFORM_TARGET_MISSING')
        for original,target,size in zip(native,transformed,sizes):
            projected=resize_boxes(original['boxes'],(512,512),size)
            masks=F.interpolate(original['masks'][:,None].float(),size=size,mode='nearest')[:,0].to(original['masks'].dtype)
            if not torch.equal(original['labels'],target['labels']) or not torch.equal(projected,target['boxes']) or not torch.equal(masks,target['masks']):
                raise ValueError('TRANSFORM_OWNER_ORDER_DRIFT')
        captured['targets']=transformed;captured['sizes']=sizes
    def pool_pre(module,args):
        _,boxes,sizes=args
        captured['proposals']=[b.detach().clone() for b in boxes]
        if sizes!=captured.get('sizes'):raise ValueError('ROI_IMAGE_SIZE_DRIFT')
    def predictor_post(module,args,result):captured['logits']=result
    try:
        handles.append(model.roi_heads.register_forward_pre_hook(roi_pre))
        handles.append(model.roi_heads.mask_roi_pool.register_forward_pre_hook(pool_pre))
        handles.append(model.roi_heads.mask_predictor.register_forward_hook(predictor_post))
        losses=model(images,targets) # original ROI56 loss and all other stock loss tensors unchanged
        if 'loss_mask' not in losses or set(captured)!={'targets','sizes','proposals','logits'}:raise ValueError('MASK_CAPTURE_INCOMPLETE')
        transformed=captured['targets'];boxes=captured['proposals']
        # Reuse the actual per-model matcher on exactly the positive mask proposals.
        if model.roi_heads.proposal_matcher.allow_low_quality_matches:raise ValueError('MATCHER_SCOPE_DRIFT')
        owners=[];labels=[]
        for b,t in zip(boxes,transformed):
            if not len(b):
                owners.append(torch.empty(0,dtype=torch.int64));labels.append(torch.empty(0,dtype=torch.int64))
            else:
                matched,label=model.roi_heads.assign_targets_to_proposals([b],[t['boxes']],[t['labels']])
                owners.extend(matched);labels.extend(label)
        if any((label<=0).any() for label in labels):raise ValueError('POSITIVE_MASK_OWNER_MISMATCH')
        aux,info=native_boundary_loss(captured['logits'],boxes,owners,native,captured['sizes'],radius)
        if diagnostics is not None:diagnostics.update(info,unweighted_loss=float(aux.detach()))
        return {**losses,'loss_mask_native_boundary':aux*weight}
    finally:
        for handle in handles:handle.remove()
        for module,flag in flags.items():module.training=flag
