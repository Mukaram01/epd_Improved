"""Training-only boundary-weighted BCE; original RoIAlign targets unchanged."""
import torch
from torch.nn import functional as F
from torchvision.models.detection.roi_heads import project_masks_on_boxes


def boundary_weights(targets, factor):
    if factor<1:raise ValueError('boundary factor must be >=1')
    padded=F.pad(targets[:,None],(1,1,1,1),value=0.)
    high=F.max_pool2d(padded,3,stride=1)
    low=-F.max_pool2d(-padded,3,stride=1)
    band=(high-low>0).squeeze(1)
    return 1.+(factor-1.)*band.to(targets.dtype)


def boundary_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, factor=4.):
    # Identical torchvision class selection and target projection; only BCE weights differ.
    size=mask_logits.shape[-1]
    labels=torch.cat([label[index] for label,index in zip(gt_labels,mask_matched_idxs)])
    targets=torch.cat([project_masks_on_boxes(mask,box,index,size)
                      for mask,box,index in zip(gt_masks,proposals,mask_matched_idxs)])
    if targets.numel()==0:return mask_logits.sum()*0
    logits=mask_logits[torch.arange(labels.shape[0],device=labels.device),labels]
    weights=boundary_weights(targets,factor)
    losses=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    return (losses*weights).sum()/weights.sum()
