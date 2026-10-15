"""Synthetic forward/backward only; no trained weights or optimizer updates."""
import unittest
import torch
import native_boundary_aux as old
import foreground_preserving_aux as a
from test_native_boundary_aux import synthetic_model,target

class ForegroundAnchorTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def evaluate(self,mask,prob=None,roi=None):
  p=(torch.full((512,512),.4) if prob is None else prob).requires_grad_()
  region=torch.ones((512,512),dtype=torch.bool) if roi is None else roi
  bands=old.boundary_bands(mask,2)
  oldp=p.detach().clone().requires_grad_();base=old.probability_boundary_loss(oldp,bands,region);base.backward()
  loss=a.foreground_probability_loss(p,mask,bands,region)
  loss.backward();return p,loss,oldp,base,bands
 def test_interior_gets_restorative_gradient_and_fp_unchanged(self):
  mask=target()['masks'][0];p,loss,q,base,bands=self.evaluate(mask)
  interior=mask.bool()&~bands[0]
  self.assertTrue((p.grad[interior]<0).all())
  torch.testing.assert_close(p.grad[bands[1]],q.grad[bands[1]],rtol=0,atol=0)
  self.assertTrue((p.grad[bands[0]]<q.grad[bands[0]]).all())
  expected=base+.5*torch.nn.functional.binary_cross_entropy(p[mask.bool()],torch.ones_like(p[mask.bool()]))
  torch.testing.assert_close(loss,expected,rtol=0,atol=0)
 def test_thin_foreground_restoration_doubles_and_no_collapse(self):
  mask=torch.zeros(512,512,dtype=torch.uint8);mask[100:125,100]=1
  p,loss,q,base,bands=self.evaluate(mask)
  torch.testing.assert_close(p.grad[mask.bool()],q.grad[mask.bool()]*2,rtol=0,atol=0)
  good=mask.float()*.98+.01
  _,goodloss,_,_,_=self.evaluate(mask,good)
  _,collapse,_,_,_=self.evaluate(mask,torch.full((512,512),.01))
  self.assertLess(float(goodloss),float(collapse))
 def test_partial_occlusion_owned_visible_pixels_only_and_roi(self):
  mask=target()['masks'][0];mask[204:212,200:208]=0
  roi=torch.ones((512,512),dtype=torch.bool);roi[:205]=False
  p,loss,q,base,bands=self.evaluate(mask,roi=roi)
  self.assertTrue((p.grad[mask.bool()&roi]<0).all())
  torch.testing.assert_close(p.grad[~mask.bool()],q.grad[~mask.bool()],rtol=0,atol=0)
  self.assertEqual(float(p.grad[~roi].abs().sum()),0)
 def test_nonincreasing_foreground_cannot_improve_anchor(self):
  mask=target()['masks'][0]
  _,first,_,base,_=self.evaluate(mask)
  values=torch.full((512,512),.4);values[mask.bool()]=.2
  _,second,_,base2,_=self.evaluate(mask,values)
  self.assertGreater(float(second-base2),float(first-base))
 def test_empty_finite_saturation_and_deterministic(self):
  empty=torch.zeros(512,512,dtype=torch.uint8)
  for value in [0.,1.,1e-7,.5,1-1e-7]:
   p,loss,_,_,_=self.evaluate(empty,torch.full((512,512),value))
   self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(p.grad).all())
  mask=target()['masks'][0]
  for v in [0.,1.]:
   p,loss,_,_,_=self.evaluate(mask,torch.full((512,512),v))
   self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(p.grad).all())
  first=self.evaluate(mask);second=self.evaluate(mask)
  torch.testing.assert_close(first[1],second[1],rtol=0,atol=0);torch.testing.assert_close(first[0].grad,second[0].grad,rtol=0,atol=0)
 def test_all_twelve_gradients_stock_terms_frozen_all_state_and_zero_weight(self):
  model=synthetic_model();allstate={n:t.clone() for n,t in model.state_dict().items()};before=old.non_mask_snapshot(model)
  image=torch.ones(3,512,512)*.4;t=target()
  torch.manual_seed(1);legacy=model([image],[t]);torch.manual_seed(1);zero=a.training_losses(model,[image],[t],0)
  for k in legacy:torch.testing.assert_close(legacy[k],zero[k],rtol=0,atol=0)
  torch.manual_seed(1);parts=a.training_losses(model,[image],[t],.25)
  for k in legacy:torch.testing.assert_close(legacy[k],parts[k],rtol=0,atol=0)
  parts['loss_mask_native_boundary'].backward()
  gradients=[p.grad for n,p in model.named_parameters() if n.startswith(old.MASK_PREFIXES)]
  self.assertEqual(len(gradients),12);self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients))
  self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if not n.startswith(old.MASK_PREFIXES)))
  old.assert_frozen(model,before)
  self.assertTrue(all(torch.equal(t,model.state_dict()[n]) for n,t in allstate.items()))

 def test_owner_balancing_class_selection_empty_and_interior_only_roi(self):
  t=target();logits=torch.zeros(1,2,56,56,requires_grad=True)
  # Interior-only proposal used to be skipped by the boundary-only objective.
  boxes=torch.tensor([[205.,205.,210.,210.]])
  loss,info=a.native_boundary_loss(logits,[boxes],[torch.tensor([0])],[t],[(512,512)])
  loss.backward();self.assertEqual(info['owners'],1);self.assertGreater(float(logits.grad[:,1].abs().sum()),0)
  self.assertEqual(float(logits.grad[:,0].abs().sum()),0)
  dup,_=a.native_boundary_loss(logits.detach().repeat(2,1,1,1),[boxes.repeat(2,1)],[torch.tensor([0,0])],[t],[(512,512)])
  torch.testing.assert_close(loss.detach(),dup,rtol=0,atol=0)
  empty={'masks':torch.empty(0,512,512,dtype=torch.uint8),'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.int64)}
  z=torch.empty(0,2,56,56,requires_grad=True)
  zero,info=a.native_boundary_loss(z,[torch.empty(0,4)],[torch.empty(0,dtype=torch.int64)],[empty],[(512,512)])
  zero.backward();self.assertEqual(float(zero),0);self.assertEqual(info['owners'],0)
 def test_invalid_ownership_probabilities_and_extreme_logits(self):
  t=target()
  with self.assertRaisesRegex(ValueError,'OWNER_INDEX'):
   a.native_boundary_loss(torch.zeros(1,2,56,56),[t['boxes']],[torch.tensor([1])],[t],[(512,512)])
  for value in [-100.,100.]:
   z=torch.full((1,2,56,56),value,requires_grad=True)
   loss,_=a.native_boundary_loss(z,[t['boxes']],[torch.tensor([0])],[t],[(512,512)])
   loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(z.grad).all())
  for value in [float('nan'),-.1,1.1]:
   with self.assertRaisesRegex(ValueError,'PROBABILITY_CONTRACT'):
    a.foreground_probability_loss(torch.full((512,512),value),t['masks'][0],old.boundary_bands(t['masks'][0],2),torch.ones(512,512,dtype=torch.bool))

# Execute the historical ownership/transform/cleanup integration contracts on
# the separate revised implementation, without altering historical tests.
import test_native_boundary_aux as historical
from unittest.mock import patch
class RevisedIntegrationTests(historical.GradientIntegrationTests):
 def run(self,result=None):
  with patch.object(historical,'a',a):return super().run(result)

if __name__=='__main__':unittest.main()
