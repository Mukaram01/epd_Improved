import importlib.util
import unittest


class BoundaryLossTests(unittest.TestCase):
    def test_boundary_weights_preserve_targets_and_cover_both_sides(self):
        self.assertIsNotNone(importlib.util.find_spec('boundary_loss'))
        import torch
        from boundary_loss import boundary_weights
        target=torch.zeros((1,9,9));target[:,2:7,2:7]=1;original=target.clone()
        weights=boundary_weights(target,4.)
        self.assertTrue(torch.equal(target,original))
        self.assertEqual(float(weights[0,1,4]),4.)
        self.assertEqual(float(weights[0,2,4]),4.)
        self.assertEqual(float(weights[0,4,4]),1.)
        self.assertEqual(float(weights[0,0,4]),1.)

    def test_actual_torchvision_loss_uniform_equivalence_empty_and_gradients(self):
        import torch
        from torchvision.models.detection.roi_heads import maskrcnn_loss
        from boundary_loss import boundary_maskrcnn_loss
        logits=torch.zeros((1,2,28,28),requires_grad=True)
        proposals=[torch.tensor([[0.,0.,32.,32.]])]
        masks=[torch.zeros((1,32,32),dtype=torch.uint8)];masks[0][:,6:26,6:26]=1
        labels=[torch.tensor([1])];matched=[torch.tensor([0])]
        original=maskrcnn_loss(logits,proposals,masks,labels,matched)
        uniform=boundary_maskrcnn_loss(logits,proposals,masks,labels,matched,factor=1.)
        self.assertTrue(torch.allclose(original,uniform,atol=1e-7,rtol=0))
        weighted=boundary_maskrcnn_loss(logits,proposals,masks,labels,matched,factor=4.)
        weighted.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[:,0],torch.zeros_like(logits.grad[:,0])))
        z=boundary_maskrcnn_loss(logits[:0], [proposals[0][:0]], masks, labels, [matched[0][:0]])
        self.assertEqual(float(z),0.)


if __name__=='__main__':unittest.main()
