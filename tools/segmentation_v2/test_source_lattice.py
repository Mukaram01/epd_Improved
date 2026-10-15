"""Small deterministic source-contract tests, never the full preregistered matrix."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import source_lattice as s

class LatticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_actual_construction_without_weights_or_forward(self):
        m=s.architecture()
        self.assertEqual(m['mask_pool'],[28,28]);self.assertEqual(m['sampling_ratio'],2)
        self.assertEqual(m['trained_response_origin'],'UNKNOWN')
        self.assertEqual(m['strides'],[4,8,16,32]);self.assertEqual(m['blocks'],[3,4,6,3])
    def test_coordinates_receptive_fields_and_invalid_assumptions(self):
        self.assertEqual(s.backbone_ledger(),[(4,35,.5),(8,91,.5),(16,267,.5),(32,427,.5)])
        np.testing.assert_array_equal(s.nearest_indices(3,6),[0,0,1,1,2,2])
        np.testing.assert_array_equal(s.nearest_indices(3,5),[0,0,1,1,2])
        with self.assertRaises(ValueError):s.nearest_indices(0,5)
        self.assertEqual({s.fpn_support(2,i)[1]-s.fpn_support(2,i)[0] for i in range(8)},{427.,459.})
        self.assertEqual(s.fpn_support(5,10)[1]-s.fpn_support(5,10)[0],491.)
        with self.assertRaises(ValueError):s.fpn_support(6,0)
    def test_convolution_kernels_odd_even_and_borders(self):
        for n in [8,9]:
            x=np.arange(n*(n+1),dtype=np.float32).reshape(n,n+1)/128
            for kind in s.KERNELS:
                for stride in [1,2]:
                    s.compare(s.numpy_conv(x,s.kernel(3,kind),stride),s.torch_conv(x,s.kernel(3,kind),stride),'conv')
    def test_scalar_backbone_scheduled_operators(self):
        x=np.arange(17*19,dtype=np.float32).reshape(17,19)/512
        for kind in s.KERNELS:
            independent=s.backbone(x,kind,True);actual=s.backbone(x,kind,False)
            for left,right in zip(independent,actual):s.compare(left,right,'backbone unit')

    def test_fpn_exact_nearest_parity_and_phase(self):
        xs=[np.arange(n*n,dtype=np.float32).reshape(n,n)/256 for n in [16,8,4,2]]
        for kind in s.KERNELS:s.check_fpn(xs,kind)
        shift=s.nearest_indices(4,8)*8-np.arange(8)*4
        np.testing.assert_array_equal(shift,[0,-4,0,-4,0,-4,0,-4])
    def test_padding_resize_halfpixel_and_level_assignment(self):
        for h,w in [(8,8),(9,11),(11,9)]:
            x=np.arange(h*w,dtype=np.float32).reshape(h,w)/128
            s.compare(s.numpy_resize(x,(h+3,w+4)),s.torch_resize(x,(h+3,w+4)),'resize')
            padded=s.pad32(x);self.assertEqual(padded.shape,(32,32))
            np.testing.assert_array_equal(padded[:h,:w],x);self.assertEqual(float(padded[h:].sum()),0)
        self.assertEqual(s.levels([32,112,224,448]),[2,3,4,5])
    def test_mask_chain_and_union_strict_threshold(self):
        field=np.ones((32,32),np.float32)*.25
        receipt,prob=s.mask_chain(field,4,'centre')
        self.assertEqual(prob.shape,(512,512))
        self.assertEqual(receipt['native512_box'],[33.,43.,193.,203.])
        p=np.full((512,512),.5,np.float32);p[2,3]=np.nextafter(np.float32(.5),np.float32(1))
        native,mapped=s.project(p,np.array([0,0,512,512]))
        self.assertEqual(native.sum(),1);self.assertEqual(mapped.sum(),1);self.assertTrue(mapped[1,1])
    def test_drift_exclusivity_and_evidence_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'bound';p.write_text('fixed');policy={'sources_sha256':{str(p):s.digest(p)}}
            s.bindings(policy);p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'DRIFT'):s.bindings(policy)
            (root/'keep').write_text('original')
            with self.assertRaisesRegex(ValueError,'EXCLUSIVE'):s.exclusive(root)
            self.assertEqual((root/'keep').read_text(),'original')
    def test_fractional_checkerboard_phase_and_fixed_zero_border_semantics(self):
        a=s.pattern((8,8),'checkerboard',0.)
        b=s.pattern((8,8),'checkerboard',.5)
        self.assertFalse(np.array_equal(a,b));np.testing.assert_array_equal(b,np.full((8,8),.5,np.float32))
        np.testing.assert_array_equal(s.pattern((8,8),'empty',0.),s.pattern((8,8),'empty',.75))
    def test_proposed_full_run_requires_separate_authorisation(self):
        from unittest.mock import patch
        with patch('sys.argv',['source_lattice','--policy','/absent','--review','/absent','--review-sha256','0'*64]):
            with self.assertRaisesRegex(ValueError,'SEPARATE_USER_AUTHORISATION'):s.main()
    def test_partial_result_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):s.validate_complete([],Path(tmp))

    def test_invalid_numeric_inputs(self):
        for a in [np.full((3,3),np.nan,np.float32),np.zeros((3,3),np.float64)]:
            with self.assertRaises(ValueError):s.numpy_conv(a,s.kernel(3,'centre'),1)
        with self.assertRaises(ValueError):s.kernel(3,'learned')

if __name__=='__main__':unittest.main()
