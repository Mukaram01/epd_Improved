"""Synthetic units only: no checkpoints, real scenes or optimizer updates."""
import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
import native_boundary_aux as a


def target():
    mask=torch.zeros((1,512,512),dtype=torch.uint8);mask[0,200:216,200:216]=1
    return {'masks':mask,'labels':torch.tensor([1]),'boxes':torch.tensor([[200.,200.,216.,216.]])}

class AuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_balanced_fp_fn_and_no_collapse(self):
        t=target();bands=a.boundary_bands(t['masks'][0],2)
        ideal=t['masks'][0].float()*.98+.01
        good=a.probability_boundary_loss(ideal,bands,torch.ones((512,512),dtype=torch.bool))
        collapse=a.probability_boundary_loss(torch.full_like(ideal,.01),bands,torch.ones_like(ideal,dtype=torch.bool))
        expanded=a.probability_boundary_loss(torch.full_like(ideal,.99),bands,torch.ones_like(ideal,dtype=torch.bool))
        self.assertLess(float(good),float(collapse));self.assertLess(float(good),float(expanded))
        self.assertAlmostEqual(float(collapse),float(expanded),places=5)
    def test_boundary_gradient_signs_and_interior_untouched(self):
        t=target();bands=a.boundary_bands(t['masks'][0],2)
        p=torch.full((512,512),.5,requires_grad=True)
        loss=a.probability_boundary_loss(p,bands,torch.ones((512,512),dtype=torch.bool));loss.backward()
        self.assertTrue((p.grad[bands[0]]<0).all());self.assertTrue((p.grad[bands[1]]>0).all())
        self.assertEqual(float(p.grad[~(bands[0]|bands[1])].abs().sum()),0)

    def test_real_paste_gradient_and_class_ownership(self):
        t=target();logits=torch.zeros((1,2,56,56),requires_grad=True)
        loss,info=a.native_boundary_loss(logits,[t['boxes']],[torch.tensor([0])],[t],[(512,512)])
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad[:,1].abs().sum()),0)
        self.assertEqual(float(logits.grad[:,0].abs().sum()),0);self.assertEqual(info['proposals'],1)
    def test_instance_owner_not_union_and_balanced_duplicate_proposals(self):
        t=target();second=t['masks'].roll(32,2);t['masks']=torch.cat([t['masks'],second]);t['labels']=torch.tensor([1,1]);t['boxes']=torch.cat([t['boxes'],t['boxes']+torch.tensor([32.,0.,32.,0.])])
        logits=torch.zeros((2,2,56,56),requires_grad=True)
        one,_=a.native_boundary_loss(logits,[t['boxes']],[torch.tensor([0,1])],[t],[(512,512)])
        # Duplicate only first object's proposals: owner-balanced mean stays equal.
        many,_=a.native_boundary_loss(logits[[0,0,1]],[t['boxes'][[0,0,1]]],[torch.tensor([0,0,1])],[t],[(512,512)])
        torch.testing.assert_close(one,many,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'OWNER'):a.native_boundary_loss(logits,[t['boxes']],[torch.tensor([0,2])],[t],[(512,512)])
    def test_occluded_thin_and_empty(self):
        t=target();t['masks'][0,204:212,200:208]=0 # visible fragmented occlusion
        loss,info=a.native_boundary_loss(torch.zeros(1,2,56,56,requires_grad=True),[t['boxes']],[torch.tensor([0])],[t],[(512,512)])
        self.assertTrue(torch.isfinite(loss));self.assertEqual(info['owners'],1)
        thin=torch.zeros((512,512),dtype=torch.uint8);thin[100:120,100]=1
        fg,bg=a.boundary_bands(thin,2);self.assertEqual(int(fg.sum()),20);self.assertGreater(int(bg.sum()),0)
        empty={'masks':torch.empty((0,512,512),dtype=torch.uint8),'labels':torch.empty(0,dtype=torch.int64),'boxes':torch.empty((0,4))}
        z=torch.empty((0,2,56,56),requires_grad=True)
        loss,info=a.native_boundary_loss(z,[torch.empty((0,4))],[torch.empty(0,dtype=torch.int64)],[empty],[(512,512)])
        loss.backward();self.assertEqual(float(loss),0);self.assertEqual(info['owners'],0)
    def test_invalid_masks_boxes_values_and_extreme_logits(self):
        t=target()
        for z in [torch.full((1,2,56,56),float('nan')),torch.zeros(1,2,28,28)]:
            with self.assertRaises(ValueError):a.native_boundary_loss(z,[t['boxes']],[torch.tensor([0])],[t],[(512,512)])
        for b in [torch.tensor([[1.,1.,1.,2.]]),torch.tensor([[-1.,1.,2.,2.]])]:
            with self.assertRaises(ValueError):a.native_boundary_loss(torch.zeros(1,2,56,56),[b],[torch.tensor([0])],[t],[(512,512)])
        for value in [-100.,100.]:
            z=torch.full((1,2,56,56),value,requires_grad=True)
            loss,_=a.native_boundary_loss(z,[t['boxes']],[torch.tensor([0])],[t],[(512,512)])
            loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(z.grad).all())
    def test_zero_weight_exact_legacy_no_hooks_or_targets(self):
        class Legacy(nn.Module):
            def forward(self,images,targets):return {'loss_mask':torch.tensor(3.)}
        model=Legacy();out=model(None,None)
        with patch.object(a,'native_boundary_loss',side_effect=AssertionError('must not run')):
            result=a.training_losses(model,None,None,weight=0)
        self.assertEqual(set(result),set(out));torch.testing.assert_close(result['loss_mask'],out['loss_mask'],rtol=0,atol=0)


# A small randomly initialised synthetic network, not an EPD checkpoint/model.
# Only forward/backward checks; NO optimizer step or training campaign.
def synthetic_model():
    from collections import OrderedDict
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.faster_rcnn import TwoMLPHead,FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNHeads,MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign
    class Backbone(nn.Module):
        out_channels=4
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Conv2d(3,4,3,4,1),nn.BatchNorm2d(4),nn.ReLU())
        def forward(self,x):return OrderedDict([('0',self.net(x))])
    class RPN(nn.Module):
        def forward(self,images,features,targets):return [t['boxes'].clone() for t in targets],{'loss_objectness':features['0'].sum()*0}
    torch.manual_seed(17)
    model=MaskRCNN(Backbone(),num_classes=None,min_size=64,max_size=64,
        box_roi_pool=MultiScaleRoIAlign(['0'],7,2),box_head=TwoMLPHead(4*7*7,16),box_predictor=FastRCNNPredictor(16,2),
        mask_roi_pool=MultiScaleRoIAlign(['0'],28,2),mask_head=MaskRCNNHeads(4,[4,4,4,4],1),mask_predictor=MaskRCNNPredictor(4,4,2))
    model.rpn=RPN();a.freeze_non_mask(model);model.train();model.backbone.eval()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n.startswith(a.MASK_PREFIXES):p.fill_(.03)
    return model

class GradientIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_actual_roihead_gradient_legacy_terms_and_frozen_buffers(self):
        m=synthetic_model();image=torch.ones(3,512,512)*.4;t=target();before=a.non_mask_snapshot(m)
        torch.manual_seed(1);legacy=m([image],[t]);torch.manual_seed(1);info={}
        parts=a.training_losses(m,[image],[t],weight=.25,diagnostics=info)
        for name in legacy:torch.testing.assert_close(parts[name],legacy[name],rtol=0,atol=0)
        parts['loss_mask_native_boundary'].backward()
        gradients={n:float(p.grad.abs().sum()) for n,p in m.named_parameters() if p.requires_grad and p.grad is not None}
        self.assertGreater(sum(gradients.values()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertGreater(sum(v for n,v in gradients.items() if n.startswith('roi_heads.mask_head.')),0)
        self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith(a.MASK_PREFIXES)))
        a.assert_frozen(m,before)
        self.assertEqual(info['owners'],1)
        self.assertFalse(m.backbone.training)
    def test_deterministic_multi_image_empty_and_hook_failure_cleanup(self):
        m=synthetic_model();t=target();empty={'masks':torch.empty((0,512,512),dtype=torch.uint8),'labels':torch.empty(0,dtype=torch.int64),'boxes':torch.empty((0,4))}
        images=[torch.ones(3,512,512)*.4,torch.zeros(3,512,512)]
        torch.manual_seed(7);first=a.training_losses(m,images,[t,empty],weight=.25)
        torch.manual_seed(7);second=a.training_losses(m,images,[t,empty],weight=.25)
        for k in first:torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
        counts=(len(m.roi_heads._forward_pre_hooks),len(m.roi_heads.mask_roi_pool._forward_pre_hooks),len(m.roi_heads.mask_predictor._forward_hooks))
        flags={module:module.training for module in m.modules()}
        with patch.object(a,'native_boundary_loss',side_effect=ValueError('synthetic failure')):
            with self.assertRaisesRegex(ValueError,'synthetic failure'):a.training_losses(m,images,[t,empty],weight=.25)
        self.assertEqual(counts,(len(m.roi_heads._forward_pre_hooks),len(m.roi_heads.mask_roi_pool._forward_pre_hooks),len(m.roi_heads.mask_predictor._forward_hooks)))
        self.assertEqual(flags,{module:module.training for module in m.modules()})
    def test_frozen_bn_and_adversarial_transform_reordering(self):
        m=synthetic_model();m.train();before=a.non_mask_snapshot(m)
        t=target();t['masks']=torch.cat([t['masks'],t['masks'].roll(32,2)]);t['boxes']=torch.cat([t['boxes'],t['boxes']+torch.tensor([32.,0.,32.,0.])]);t['labels']=torch.tensor([1,1])
        a.training_losses(m,[torch.ones(3,512,512)*.4],[t],.25)
        a.assert_frozen(m,before);self.assertTrue(m.backbone.training) # flags restored, BN state unchanged
        def corrupt(module,args,result):
            images,targets=result;targets[0]['masks']=targets[0]['masks'].flip(0);return images,targets
        handle=m.transform.register_forward_hook(corrupt)
        try:
            with self.assertRaisesRegex(ValueError,'TRANSFORM_OWNER_ORDER_DRIFT'):a.training_losses(m,[torch.ones(3,512,512)*.4],[t],.25)
        finally:handle.remove()
        self.assertEqual(len(m.roi_heads._forward_pre_hooks),0)

    def test_matcher_ties_and_owner_order_validation(self):
        m=synthetic_model();t=target();t['masks']=torch.cat([t['masks'],t['masks'].roll(32,2)]);t['boxes']=torch.cat([t['boxes'],t['boxes']+torch.tensor([32.,0.,32.,0.])]);t['labels']=torch.tensor([1,1])
        # Two disjoint visible silhouettes share one bounding box. Matcher ties choose first.
        t['masks'].zero_();t['masks'][0,200,200]=1;t['masks'][0,215,215]=1;t['masks'][1,200,215]=1;t['masks'][1,215,200]=1;t['boxes'][1]=t['boxes'][0]
        info={};a.training_losses(m,[torch.ones(3,512,512)*.4],[t],.25,diagnostics=info)
        self.assertEqual(info['owners'],1)
        m.roi_heads.proposal_matcher.allow_low_quality_matches=True
        with self.assertRaisesRegex(ValueError,'MATCHER_SCOPE_DRIFT'):a.training_losses(m,[torch.ones(3,512,512)*.4],[t],.25)

if __name__=='__main__':unittest.main()
